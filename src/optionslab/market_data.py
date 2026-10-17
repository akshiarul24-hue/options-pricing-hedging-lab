"""From raw option quotes to an implied-volatility smile.

Pipeline
--------
1. Get quotes (bid/ask for calls and puts across strikes and expiries).
2. Clean them: drop zero bids, crossed or very wide markets.
3. For each expiry, read the FORWARD price F and DISCOUNT FACTOR DF straight
   from put-call parity:  C - P = DF * (F - K).  Regressing (C - P) on K gives
   slope -DF and intercept DF*F. This avoids having to guess the interest rate
   or dividend yield - the market tells us.
4. Use only out-of-the-money options (puts below F, calls above F): they are
   the most liquid and, for American-style options like SPY, carry almost no
   early-exercise premium.
5. Invert the Black (1976) formula for each option to get its implied vol.

`make_synthetic_chain` builds a fake chain from a KNOWN smile so the whole
pipeline can be tested: if it is correct, it must recover the smile we put in.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .black_scholes import black76_price, bs_vega
from .implied_vol import implied_vol_black76

SECONDS_PER_YEAR = 365.0 * 24 * 3600


# ----------------------------------------------------------------------------- synthetic data
def default_smile(k, T):
    """A typical equity-index smile: downward skew that is steeper for short expiries."""
    atm = 0.14 + 0.04 * np.sqrt(T)
    skew = -0.20 / np.sqrt(T)
    curvature = 1.0 / np.sqrt(T)
    return atm + skew * k + curvature * k**2


def make_synthetic_chain(spot=100.0, r=0.045, q=0.013, expiries=(30 / 365, 91 / 365, 182 / 365),
                         moneyness=np.arange(0.80, 1.151, 0.01), smile=default_smile,
                         half_spread_vol=0.0025, seed=0) -> pd.DataFrame:
    """Option quotes generated from a known smile, with realistic bid-ask spreads and noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for T in expiries:
        F, DF = spot * np.exp((r - q) * T), np.exp(-r * T)
        for m in moneyness:
            K = round(spot * m, 2)
            true_iv = float(smile(np.log(K / F), T))
            for opt in ("call", "put"):
                fair = float(black76_price(F, K, T, DF, true_iv, opt))
                vega = float(bs_vega(spot, K, T, r, true_iv, q))
                half_spread = max(0.01, vega * half_spread_vol)
                mid = fair + rng.normal(0.0, 0.25 * half_spread)
                rows.append({"T": T, "strike": K, "type": opt,
                             "bid": max(round(mid - half_spread, 2), 0.0),
                             "ask": round(mid + half_spread, 2), "true_iv": true_iv})
    df = pd.DataFrame(rows)
    df["expiry"] = (df["T"] * 365).round().astype(int).astype(str) + "d"
    return df


# ----------------------------------------------------------------------------- live data
def fetch_yahoo_chain(ticker="SPY", target_days=(30, 60, 91, 182), min_days=7):
    """Download option chains from Yahoo Finance (needs internet and the `yfinance` package).

    Picks the listed expiry closest to each target maturity. Returns (spot, quotes DataFrame).
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="5d")["Close"].iloc[-1])
    now = pd.Timestamp.now(tz="America/New_York")
    listed = pd.to_datetime(list(tk.options)).tz_localize("America/New_York") + pd.Timedelta(hours=16)
    days = (listed - now).total_seconds() / 86400.0
    candidates = [(d, e) for d, e in zip(days, listed) if d >= min_days]
    chosen = sorted({min(candidates, key=lambda c: abs(c[0] - t))[1] for t in target_days})

    frames = []
    for exp in chosen:
        chain = tk.option_chain(exp.strftime("%Y-%m-%d"))
        for opt, table in (("call", chain.calls), ("put", chain.puts)):
            part = pd.DataFrame({
                "expiry": exp.strftime("%Y-%m-%d"),
                "T": (exp - now).total_seconds() / SECONDS_PER_YEAR,
                "strike": table["strike"].astype(float),
                "type": opt,
                "bid": table["bid"].astype(float),
                "ask": table["ask"].astype(float),
                "last": table["lastPrice"].astype(float),
                "volume": table["volume"].fillna(0),
                "open_interest": table["openInterest"].fillna(0),
                "yahoo_iv": table["impliedVolatility"].astype(float),
            })
            frames.append(part)
    return spot, pd.concat(frames, ignore_index=True)


def load_chain_csv(path) -> pd.DataFrame:
    """Load quotes from your own CSV. Required columns: expiry, T (years), strike, type, bid, ask."""
    df = pd.read_csv(path)
    missing = {"expiry", "T", "strike", "type", "bid", "ask"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    df["type"] = df["type"].str.lower().str.strip().replace({"c": "call", "p": "put"})
    return df


# ----------------------------------------------------------------------------- processing
def clean_quotes(df, max_rel_spread=0.30, min_bid=0.05, use_last_if_no_quotes=True) -> pd.DataFrame:
    """Keep two-sided, reasonably tight quotes and compute the mid price.

    Outside market hours some data sources report zero bids/asks; in that case
    we fall back to the last traded price for that expiry and flag it.
    """
    df = df.copy()
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["rel_spread"] = (df["ask"] - df["bid"]) / df["mid"]
    good = (df["bid"] >= min_bid) & (df["ask"] > df["bid"]) & (df["rel_spread"] <= max_rel_spread)
    df["price_source"] = "mid"
    out = []
    for exp, grp in df.groupby("expiry", sort=False):
        ok = grp[good.loc[grp.index]]
        if len(ok) >= 10 or not use_last_if_no_quotes or "last" not in grp:
            out.append(ok)
        else:
            warnings.warn(f"{exp}: too few live bid/ask quotes - using last traded prices (noisier).")
            fallback = grp[grp["last"] > min_bid].copy()
            fallback["mid"], fallback["price_source"] = fallback["last"], "last"
            out.append(fallback)
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]


def estimate_forward(quotes_one_expiry, spot, window=0.06, min_pairs=4,
                     rate_bounds=(-0.01, 0.12), fallback_rate=None):
    """Forward F and discount factor DF from put-call parity, C - P = DF (F - K).

    Uses strikes within `window` (e.g. 6%) of spot where both a call and a put
    are quoted, and fits the line robustly: pairs whose residual is more than
    3 robust standard deviations (MAD-based) from the fit are dropped and the
    line is refit. Stale or mismatched quotes are common in free data, and a
    single bad pair can tilt an ordinary least-squares slope a lot.

    Sanity check: the slope implies an interest rate, r = -ln(DF) / T. If it is
    outside `rate_bounds` the slope is not trustworthy (with a short expiry, a
    tiny slope error means a huge rate error). Then, if `fallback_rate` is
    given, DF is set from that rate and F is taken as the median of
    K + (C - P) / DF over the clean pairs; otherwise a ValueError is raised.

    Returns (F, DF, n_pairs_used).
    """
    q = quotes_one_expiry
    T = float(q["T"].iloc[0])
    calls = q[q["type"] == "call"].groupby("strike")["mid"].mean()
    puts = q[q["type"] == "put"].groupby("strike")["mid"].mean()
    common = calls.index.intersection(puts.index)
    common = common[np.abs(common / spot - 1.0) <= window]
    if len(common) < min_pairs:
        raise ValueError(f"only {len(common)} call/put pairs near the money; widen `window`")
    K = np.asarray(common, float)
    diff = calls.loc[common].to_numpy() - puts.loc[common].to_numpy()

    keep = np.ones(len(K), dtype=bool)
    for _ in range(5):  # fit, drop outliers, refit until stable
        slope, intercept = np.polyfit(K[keep], diff[keep], 1)
        resid = diff - (slope * K + intercept)
        centre = np.median(resid[keep])
        spread = 1.4826 * np.median(np.abs(resid[keep] - centre))   # robust std
        new_keep = np.abs(resid - centre) <= max(3.0 * spread, 0.01)
        if new_keep.sum() < min_pairs or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    slope, intercept = np.polyfit(K[keep], diff[keep], 1)

    DF = -slope
    rate = -np.log(DF) / T if DF > 0 else np.inf
    if rate_bounds[0] <= rate <= rate_bounds[1]:
        return intercept / DF, DF, int(keep.sum())
    if fallback_rate is None:
        raise ValueError(f"parity regression implies an impossible {rate:.1%} interest rate")
    DF = float(np.exp(-fallback_rate * T))
    F = float(np.median(K[keep] + diff[keep] / DF))
    return F, DF, int(keep.sum())


def compute_smile(quotes, spot, window=0.06) -> pd.DataFrame:
    """Implied vol for every out-of-the-money option, per expiry.

    Pass 1 reads F and DF from put-call parity for each expiry. Any expiry
    whose parity slope implies an impossible interest rate is retried in pass 2
    using the median rate implied by the expiries that passed (interest rates
    barely differ across expiries a few months apart), so the expiry is kept
    rather than dropped. Column `forward_method` records which pass was used.
    """
    groups = list(quotes.groupby("expiry", sort=False))
    fits, retry = {}, []
    for exp, grp in groups:
        try:
            fits[exp] = estimate_forward(grp, spot, window) + ("parity regression",)
        except ValueError as err:
            retry.append((exp, grp, err))

    good_rates = [-np.log(DF) / float(g["T"].iloc[0]) for (e, g) in groups if e in fits
                  for (_, DF, _, _) in [fits[e]]]
    fallback = float(np.median(good_rates)) if good_rates else 0.04
    for exp, grp, err in retry:
        try:
            fits[exp] = estimate_forward(grp, spot, window, fallback_rate=fallback) + ("parity at fallback rate",)
            warnings.warn(f"{exp}: {err} (noisy quotes); used the {fallback:.2%} rate from the other "
                          "expiries and read the forward from parity instead.")
        except ValueError as err2:
            warnings.warn(f"{exp}: skipped ({err2})")

    results = []
    for exp, grp in groups:
        if exp not in fits:
            continue
        F, DF, n_pairs, method = fits[exp]
        T = float(grp["T"].iloc[0])
        otm = grp[((grp["type"] == "put") & (grp["strike"] < F)) |
                  ((grp["type"] == "call") & (grp["strike"] >= F))].copy()
        otm["F"], otm["DF"], otm["parity_pairs"], otm["forward_method"] = F, DF, n_pairs, method
        otm["k"] = np.log(otm["strike"] / F)
        otm["iv"] = [implied_vol_black76(p, F, K, T, DF, o)
                     for p, K, o in zip(otm["mid"], otm["strike"], otm["type"])]
        otm["implied_rate"] = -np.log(DF) / T
        results.append(otm.dropna(subset=["iv"]))
    if not results:
        raise ValueError("no expiry produced a usable smile")
    return pd.concat(results, ignore_index=True).sort_values(["T", "strike"]).reset_index(drop=True)


def atm_term_structure(smile_df) -> pd.DataFrame:
    """ATM implied vol per expiry, by linear interpolation of the smile at k = 0."""
    rows = []
    for exp, grp in smile_df.groupby("expiry", sort=False):
        g = grp.sort_values("k")
        if g["k"].min() < 0 < g["k"].max():
            rows.append({"expiry": exp, "T": g["T"].iloc[0], "days": g["T"].iloc[0] * 365,
                         "atm_iv": float(np.interp(0.0, g["k"], g["iv"]))})
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)

"""Tests for implied volatility, the market-data pipeline and the hedging simulator."""
import numpy as np
import pytest

from optionslab import bs_price, bs_vega, derman_kamal_std, hedge_pnl_attribution, implied_vol, simulate_delta_hedge, simulate_gbm_paths
from optionslab.market_data import clean_quotes, compute_smile, estimate_forward, make_synthetic_chain

S0, K, T, R, SIGMA = 100.0, 100.0, 0.25, 0.05, 0.20


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("strike", [60.0, 80.0, 100.0, 125.0, 160.0])
@pytest.mark.parametrize("vol", [0.05, 0.2, 0.6, 1.5])
def test_implied_vol_round_trip(option, strike, vol):
    price = float(bs_price(S0, strike, 1.0, R, vol, option))
    lower_bound = max(S0 - strike * np.exp(-R), 0.0) if option == "call" else max(strike * np.exp(-R) - S0, 0.0)
    if price < 1e-6 or price - lower_bound < 1e-9:
        pytest.skip("no measurable time value, so implied vol is undefined")
    iv = implied_vol(price, S0, strike, 1.0, R, option)
    # The recovered vol must always reproduce the price...
    assert float(bs_price(S0, strike, 1.0, R, iv, option)) == pytest.approx(price, abs=1e-8)
    # ...and must equal the input vol whenever the price is actually sensitive to vol.
    # (Deep in-the-money options with almost no time value have vega ~ 0, so their
    # implied vol is ill-conditioned: many vols give the same price to 8 decimals.)
    if float(bs_vega(S0, strike, 1.0, R, vol)) > 1e-3:
        assert iv == pytest.approx(vol, abs=1e-6)


def test_implied_vol_rejects_arbitrage_prices():
    assert np.isnan(implied_vol(0.5, 120.0, 100.0, 1.0, R, "call"))   # below intrinsic value
    assert np.isnan(implied_vol(101.0, 100.0, 100.0, 1.0, R, "call"))  # above the stock price
    assert np.isnan(implied_vol(-1.0, 100.0, 100.0, 1.0, R, "call"))


def test_pipeline_recovers_known_smile():
    r, q, spot = 0.045, 0.013, 100.0
    chain = make_synthetic_chain(spot=spot, r=r, q=q, seed=0)
    quotes = clean_quotes(chain)
    for _, grp in quotes.groupby("expiry"):
        F, DF, _ = estimate_forward(grp, spot)
        T_ = grp["T"].iloc[0]
        assert F == pytest.approx(spot * np.exp((r - q) * T_), abs=0.02)
        assert DF == pytest.approx(np.exp(-r * T_), abs=0.002)
    smile = compute_smile(quotes, spot)
    assert len(smile) > 50
    assert np.max(np.abs(smile["iv"] - smile["true_iv"])) < 0.005   # within half a vol point


def test_clean_quotes_drops_bad_rows():
    chain = make_synthetic_chain(seed=1)
    chain.loc[0, "bid"] = 0.0               # no bid
    chain.loc[1, "ask"] = chain.loc[1, "bid"] - 0.01  # crossed market
    cleaned = clean_quotes(chain, use_last_if_no_quotes=False)
    assert 0 not in cleaned.index.tolist() or cleaned.loc[0, "bid"] > 0
    assert (cleaned["ask"] > cleaned["bid"]).all()


def test_hedged_pnl_is_unbiased_and_shrinks_with_frequency():
    daily = simulate_delta_hedge(S0, K, T, R, SIGMA, mu=0.15, n_rebalance=63, n_paths=20_000, seed=1)
    fine = simulate_delta_hedge(S0, K, T, R, SIGMA, mu=0.15, n_rebalance=252, n_paths=20_000, seed=2)
    assert abs(daily.mean) < 4 * daily.std / np.sqrt(20_000)
    assert fine.std == pytest.approx(daily.std / 2, rel=0.08)          # 4x more trades -> half the error
    assert daily.std == pytest.approx(derman_kamal_std(S0, K, T, R, SIGMA, 63), rel=0.06)


def test_unhedged_pnl_depends_on_drift_but_hedged_does_not():
    up = simulate_delta_hedge(S0, K, T, R, SIGMA, mu=0.30, n_rebalance=0, n_paths=20_000, seed=3)
    down = simulate_delta_hedge(S0, K, T, R, SIGMA, mu=-0.20, n_rebalance=0, n_paths=20_000, seed=3)
    assert up.mean < -3 and down.mean > 2
    for mu in (-0.2, 0.3):
        h = simulate_delta_hedge(S0, K, T, R, SIGMA, mu=mu, n_rebalance=126, n_paths=20_000, seed=4)
        assert abs(h.mean) < 0.03


@pytest.mark.parametrize("realized", [0.12, 0.28])
def test_hedging_at_realized_vol_locks_in_price_difference(realized):
    res = simulate_delta_hedge(S0, K, T, R, SIGMA, sigma_realized=realized, hedge_vol=realized,
                               n_rebalance=504, n_paths=20_000, seed=5)
    theory = float(bs_price(S0, K, T, R, SIGMA) - bs_price(S0, K, T, R, realized))
    assert res.mean == pytest.approx(theory, abs=0.02)


def test_gamma_theta_attribution_explains_pnl():
    paths = simulate_gbm_paths(S0, 0.1, SIGMA, T, n_steps=252, n_paths=50, seed=6)
    actual, predicted = [], []
    for p in paths:
        a = hedge_pnl_attribution(p, K, T, R, SIGMA)
        actual.append(a["actual"]); predicted.append(a["gamma"] + a["theta"])
    actual, predicted = np.concatenate(actual), np.concatenate(predicted)
    r2 = 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2)
    assert r2 > 0.95


def test_transaction_costs_are_positive_and_grow_with_frequency():
    few = simulate_delta_hedge(S0, K, T, R, SIGMA, n_rebalance=13, n_paths=5_000, cost_rate=5e-4, seed=7)
    many = simulate_delta_hedge(S0, K, T, R, SIGMA, n_rebalance=504, n_paths=5_000, cost_rate=5e-4, seed=7)
    assert few.costs.min() > 0 and many.costs.mean() > 2 * few.costs.mean()

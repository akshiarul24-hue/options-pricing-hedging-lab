"""Delta-hedging simulator.

A market maker who sells an option does not want to bet on where the stock
goes. So they buy `delta` shares of the stock against the short option and
re-adjust that holding as the price moves. If the market behaved exactly like
the Black-Scholes model and you could rebalance continuously, this would
remove all risk and the premium you charged would exactly pay for the hedge.

This module simulates that process on thousands of random price paths, so we
can measure what actually happens when:
  * you can only rebalance a finite number of times,
  * the real-world drift (mu) is not the risk-free rate,
  * realized volatility differs from the implied volatility you sold at,
  * every trade costs money.

All P&L numbers are reported in today's money (discounted at r).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import _check_option, bs_delta, bs_gamma, bs_price, bs_vega, payoff


@dataclass
class HedgeResult:
    pnl: np.ndarray          # final P&L per path, discounted to t = 0
    costs: np.ndarray        # transaction costs paid per path, discounted to t = 0
    S_T: np.ndarray          # terminal stock prices
    premium: float           # option price charged at inception (at implied vol)
    n_rebalance: int

    @property
    def mean(self) -> float:
        return float(self.pnl.mean())

    @property
    def std(self) -> float:
        return float(self.pnl.std(ddof=1))

    def summary(self) -> dict:
        return {
            "rebalances": self.n_rebalance,
            "mean_pnl": self.mean,
            "std_pnl": self.std,
            "std_pct_of_premium": 100 * self.std / self.premium,
            "p5": float(np.percentile(self.pnl, 5)),
            "p95": float(np.percentile(self.pnl, 95)),
            "mean_cost": float(self.costs.mean()),
        }


def simulate_delta_hedge(
    S0=100.0, K=100.0, T=0.25, r=0.05, sigma_implied=0.20, sigma_realized=None, mu=None,
    n_rebalance=63, n_paths=10_000, option="call", position="short", hedge_vol=None,
    cost_rate=0.0, seed=None,
) -> HedgeResult:
    """Simulate a delta-hedged option position from inception to expiry.

    Parameters
    ----------
    sigma_implied   volatility used to PRICE the option (what you sold it at)
    sigma_realized  volatility the stock ACTUALLY has in the simulation (default: implied)
    mu              real-world drift of the stock (default: r)
    n_rebalance     number of hedge trades over the option's life; 0 = never hedge
    position        "short" (market maker sold the option) or "long"
    hedge_vol       volatility used to compute the hedge delta (default: implied)
    cost_rate       proportional transaction cost, e.g. 0.0005 = 5 bps of traded notional
    """
    option = _check_option(option)
    sigma_realized = sigma_implied if sigma_realized is None else sigma_realized
    mu = r if mu is None else mu
    hedge_vol = sigma_implied if hedge_vol is None else hedge_vol
    qty = -1.0 if position == "short" else 1.0   # number of options held
    rng = np.random.default_rng(seed)

    premium = float(bs_price(S0, K, T, r, sigma_implied, option))

    if n_rebalance == 0:  # naked position: no hedge at all
        Z = rng.standard_normal(n_paths)
        S_T = S0 * np.exp((mu - 0.5 * sigma_realized**2) * T + sigma_realized * np.sqrt(T) * Z)
        pnl = -qty * premium + qty * payoff(S_T, K, option) * np.exp(-r * T)
        return HedgeResult(pnl, np.zeros(n_paths), S_T, premium, 0)

    dt = T / n_rebalance
    S = np.full(n_paths, float(S0))
    shares = np.full(n_paths, -qty * float(bs_delta(S0, K, T, r, hedge_vol, option)))
    cost = cost_rate * np.abs(shares) * S
    cash = -qty * premium - shares * S - cost      # premium in, shares bought, costs paid
    costs_pv = cost.copy()

    for i in range(1, n_rebalance + 1):
        Z = rng.standard_normal(n_paths)
        S = S * np.exp((mu - 0.5 * sigma_realized**2) * dt + sigma_realized * np.sqrt(dt) * Z)
        cash = cash * np.exp(r * dt)               # cash earns (or pays) interest
        t = i * dt
        if i < n_rebalance:
            new_shares = -qty * bs_delta(S, K, T - t, r, hedge_vol, option)
        else:
            new_shares = np.zeros(n_paths)          # unwind the hedge at expiry
        trade = new_shares - shares
        cost = cost_rate * np.abs(trade) * S
        cash = cash - trade * S - cost
        costs_pv += cost * np.exp(-r * t)
        shares = new_shares

    cash = cash + qty * payoff(S, K, option)       # settle the option
    pnl = cash * np.exp(-r * T)
    return HedgeResult(pnl, costs_pv, S, premium, n_rebalance)


def hedge_pnl_attribution(S_path, K, T, r, sigma, option="call"):
    """Step-by-step P&L of a SHORT option, delta-hedged at volatility `sigma`, on one path.

    Returns a dict of arrays (one entry per step):
      actual  - true change in value of (short option + hedge + cash), marked at `sigma`
      gamma   - -1/2 * Gamma * (dS)^2          what the stock's moves cost you
      theta   - +1/2 * sigma^2 * S^2 * Gamma * dt  what time decay (plus financing) pays you
    The Black-Scholes PDE says actual ~= gamma + theta when the hedge is rebalanced often.
    """
    option = _check_option(option)
    S_path = np.asarray(S_path, float)
    n = len(S_path) - 1
    dt = T / n
    t = np.arange(n + 1) * dt
    tau = T - t
    V = bs_price(S_path, K, tau, r, sigma, option)
    delta = bs_delta(S_path, K, tau, r, sigma, option)
    gamma = bs_gamma(S_path, K, tau, r, sigma)

    # Portfolio: short 1 option, long delta shares, cash = V - delta*S (self-financing at each step)
    cash = V[:-1] - delta[:-1] * S_path[:-1]
    value_before = -V[:-1] + delta[:-1] * S_path[:-1] + cash          # = 0 by construction
    value_after = -V[1:] + delta[:-1] * S_path[1:] + cash * np.exp(r * dt)
    actual = value_after - value_before

    dS = np.diff(S_path)
    gamma_term = -0.5 * gamma[:-1] * dS**2
    theta_term = 0.5 * sigma**2 * S_path[:-1] ** 2 * gamma[:-1] * dt
    return {"t": t[1:], "actual": actual, "gamma": gamma_term, "theta": theta_term}


def derman_kamal_std(S0, K, T, r, sigma, n_rebalance):
    """Approximate std of discrete-hedging error (Derman & Kamal, 1999):
    sqrt(pi/4) * vega * sigma / sqrt(N)  -- valid for options near the money."""
    vega = float(bs_vega(S0, K, T, r, sigma))
    return np.sqrt(np.pi / 4.0) * vega * sigma / np.sqrt(n_rebalance)

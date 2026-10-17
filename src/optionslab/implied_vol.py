"""Implied volatility: the sigma that makes the model price equal a market price.

Black-Scholes has five inputs. Four are observable (S, K, T, r); volatility is
not. So traders run the formula backwards: given the market price, solve for
the volatility. Because option prices increase monotonically with sigma
(vega > 0), there is at most one solution, and a bracketing root-finder
(Brent's method) finds it reliably.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .black_scholes import _check_option, black76_price

VOL_LO, VOL_HI = 1e-4, 5.0


def implied_vol_black76(price, F, K, T, DF, option="call", lo=VOL_LO, hi=VOL_HI) -> float:
    """Implied vol from a price, using the forward F and discount factor DF.

    Returns NaN when the price violates no-arbitrage bounds (below intrinsic
    value or above the maximum possible value), which happens with stale or
    bad quotes in real data.
    """
    option = _check_option(option)
    if not (np.isfinite(price) and price > 0 and T > 0):
        return np.nan
    intrinsic = DF * max(F - K, 0.0) if option == "call" else DF * max(K - F, 0.0)
    upper = DF * F if option == "call" else DF * K
    if not (intrinsic < price < upper):
        return np.nan

    def objective(sig):
        return float(black76_price(F, K, T, DF, sig, option)) - price

    f_lo, f_hi = objective(lo), objective(hi)
    if f_lo > 0 or f_hi < 0:
        return np.nan
    return brentq(objective, lo, hi, xtol=1e-10, rtol=1e-10, maxiter=200)


def implied_vol(price, S, K, T, r, option="call", q=0.0, **kwargs) -> float:
    """Implied vol from a price using spot, rate and dividend yield."""
    F = S * np.exp((r - q) * T)
    DF = np.exp(-r * T)
    return implied_vol_black76(price, F, K, T, DF, option, **kwargs)


def implied_vols_black76(prices, F, strikes, T, DF, options) -> np.ndarray:
    """Vectorised convenience wrapper: one implied vol per (price, strike, type)."""
    prices, strikes = np.asarray(prices, float), np.asarray(strikes, float)
    options = np.broadcast_to(np.asarray(options), prices.shape)
    return np.array([
        implied_vol_black76(p, F, k, T, DF, o) for p, k, o in zip(prices, strikes, options)
    ])

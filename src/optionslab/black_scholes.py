"""Closed-form Black-Scholes-Merton prices and Greeks for European options.

Conventions used everywhere in this project
-------------------------------------------
S      spot price of the underlying
K      strike price
T      time to expiry in YEARS (e.g. 3 months = 0.25)
r      continuously compounded risk-free rate (0.05 = 5% per year)
sigma  volatility per year (0.20 = 20%)
q      continuous dividend yield (default 0)

All functions accept NumPy arrays and broadcast, so you can price a whole
grid of strikes or spots in one call.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EPS_T = 1e-12  # treat T below this as "at expiry"


def _check_option(option: str) -> str:
    option = option.lower()
    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")
    return option


def d1_d2(S, K, T, r, sigma, q=0.0):
    """Return the d1 and d2 terms of the Black-Scholes formula."""
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    T_safe = np.maximum(T, _EPS_T)
    vol_sqrt_t = sigma * np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T_safe) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def payoff(S_T, K, option="call"):
    """Payoff at expiry: max(S-K, 0) for a call, max(K-S, 0) for a put."""
    option = _check_option(option)
    S_T = np.asarray(S_T, dtype=float)
    return np.maximum(S_T - K, 0.0) if option == "call" else np.maximum(K - S_T, 0.0)


def bs_price(S, K, T, r, sigma, option="call", q=0.0):
    """Black-Scholes price of a European call or put.

    Call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    Put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
    """
    option = _check_option(option)
    T = np.asarray(T, dtype=float)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    if option == "call":
        price = disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    else:
        price = disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)
    return np.where(T <= _EPS_T, payoff(S, K, option), price)


def bs_delta(S, K, T, r, sigma, option="call", q=0.0):
    """dV/dS: how many shares hedge one option. Call in (0, 1), put in (-1, 0)."""
    option = _check_option(option)
    T = np.asarray(T, dtype=float)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    growth = np.exp(-q * T)
    delta = growth * norm.cdf(d1) if option == "call" else growth * (norm.cdf(d1) - 1.0)
    at_expiry = np.where(
        np.asarray(S) > K, 1.0 if option == "call" else 0.0, 0.0 if option == "call" else -1.0
    )
    return np.where(T <= _EPS_T, at_expiry, delta)


def bs_gamma(S, K, T, r, sigma, q=0.0):
    """d2V/dS2: how fast delta changes. Same for calls and puts."""
    S, T = np.asarray(S, dtype=float), np.asarray(T, dtype=float)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    T_safe = np.maximum(T, _EPS_T)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T_safe))
    return np.where(T <= _EPS_T, 0.0, gamma)


def bs_vega(S, K, T, r, sigma, q=0.0):
    """dV/dsigma per 1.00 change in vol (divide by 100 for 'per 1 vol point')."""
    S, T = np.asarray(S, dtype=float), np.asarray(T, dtype=float)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(np.maximum(T, 0.0))
    return np.where(T <= _EPS_T, 0.0, vega)


def bs_theta(S, K, T, r, sigma, option="call", q=0.0):
    """dV/dt per YEAR of calendar time passing (divide by 365 for per-day)."""
    option = _check_option(option)
    S, T = np.asarray(S, dtype=float), np.asarray(T, dtype=float)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    T_safe = np.maximum(T, _EPS_T)
    decay = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T_safe))
    if option == "call":
        theta = decay - r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1)
    else:
        theta = decay + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)
    return np.where(T <= _EPS_T, 0.0, theta)


def bs_rho(S, K, T, r, sigma, option="call", q=0.0):
    """dV/dr per 1.00 change in the interest rate."""
    option = _check_option(option)
    T = np.asarray(T, dtype=float)
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)


def bs_greeks(S, K, T, r, sigma, option="call", q=0.0) -> dict:
    """All first/second-order Greeks in one dictionary."""
    return {
        "price": bs_price(S, K, T, r, sigma, option, q),
        "delta": bs_delta(S, K, T, r, sigma, option, q),
        "gamma": bs_gamma(S, K, T, r, sigma, q),
        "vega": bs_vega(S, K, T, r, sigma, q),
        "theta": bs_theta(S, K, T, r, sigma, option, q),
        "rho": bs_rho(S, K, T, r, sigma, option, q),
    }


def black76_price(F, K, T, DF, sigma, option="call"):
    """Black (1976) price written in terms of the forward F and discount factor DF.

    Call = DF [F N(d1) - K N(d2)],  d1 = (ln(F/K) + sigma^2 T / 2) / (sigma sqrt T)

    This is the same as Black-Scholes with F = S e^{(r-q)T} and DF = e^{-rT},
    but it does not need r or q separately - useful with market data, where
    the forward and discount factor can be read off put-call parity.
    """
    option = _check_option(option)
    F, K, T, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, sigma))
    vol_sqrt_t = sigma * np.sqrt(np.maximum(T, _EPS_T))
    d1 = (np.log(F / K) + 0.5 * vol_sqrt_t**2) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    if option == "call":
        return DF * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return DF * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def put_call_parity_gap(call, put, S, K, T, r, q=0.0):
    """C - P - (S e^{-qT} - K e^{-rT}). Should be ~0 for European options."""
    return call - put - (S * np.exp(-q * T) - K * np.exp(-r * T))

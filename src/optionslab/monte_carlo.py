"""Stock-price simulation (geometric Brownian motion) and Monte Carlo pricing.

Monte Carlo idea: simulate many possible stock prices at expiry under the
risk-neutral model (stock grows at r, not its real drift), average the payoffs,
and discount back to today. The law of large numbers makes the average converge
to the fair price; the central limit theorem says the error shrinks like 1/sqrt(N).

Two variance-reduction tricks are implemented:
  * antithetic variates - for every random draw Z also use -Z, so up/down
    shocks are perfectly balanced in each pair;
  * control variate     - use the discounted stock price, whose exact mean
    (S0 e^{-qT}) we know, to correct the noisy payoff average.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import _check_option, payoff


def simulate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths, seed=None, antithetic=False):
    """Simulate GBM price paths: S_{t+dt} = S_t exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z).

    Returns an array of shape (n_paths, n_steps + 1); column 0 is S0.
    Uses the exact solution of the SDE, so there is no discretisation error
    at the simulated dates.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    if antithetic:
        half = rng.standard_normal((int(np.ceil(n_paths / 2)), n_steps))
        Z = np.vstack([half, -half])[:n_paths]
    else:
        Z = rng.standard_normal((n_paths, n_steps))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
    return paths


def terminal_prices(S0, mu, sigma, T, Z):
    """One-step exact GBM: price at T for standard normal draws Z."""
    return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


@dataclass
class MCResult:
    price: float      # Monte Carlo estimate of the option value
    std_err: float    # standard error of that estimate (1 sigma)
    n_paths: int      # number of payoff evaluations used
    method: str

    @property
    def ci95(self):
        return (self.price - 1.96 * self.std_err, self.price + 1.96 * self.std_err)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"MCResult({self.method}: {self.price:.4f} +/- {self.std_err:.4f}, n={self.n_paths:,})"


def mc_european(S0, K, T, r, sigma, option="call", q=0.0, n_paths=100_000, method="plain", seed=None):
    """Monte Carlo price of a European option.

    method: "plain", "antithetic" or "control". All three use exactly
    `n_paths` payoff evaluations, so their standard errors are directly comparable.
    """
    option = _check_option(option)
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)
    drift = r - q  # risk-neutral drift

    if method == "plain":
        Z = rng.standard_normal(n_paths)
        Y = disc * payoff(terminal_prices(S0, drift, sigma, T, Z), K, option)
        return MCResult(Y.mean(), Y.std(ddof=1) / np.sqrt(n_paths), n_paths, method)

    if method == "antithetic":
        n_pairs = n_paths // 2
        Z = rng.standard_normal(n_pairs)
        Y_plus = disc * payoff(terminal_prices(S0, drift, sigma, T, Z), K, option)
        Y_minus = disc * payoff(terminal_prices(S0, drift, sigma, T, -Z), K, option)
        pair_mean = 0.5 * (Y_plus + Y_minus)  # the pairs are independent of each other
        return MCResult(pair_mean.mean(), pair_mean.std(ddof=1) / np.sqrt(n_pairs), 2 * n_pairs, method)

    if method == "control":
        Z = rng.standard_normal(n_paths)
        S_T = terminal_prices(S0, drift, sigma, T, Z)
        Y = disc * payoff(S_T, K, option)
        X = disc * S_T                      # control variable
        X_mean = S0 * np.exp(-q * T)        # its exact expectation under risk-neutral measure
        cov = np.cov(Y, X, ddof=1)
        b = cov[0, 1] / cov[1, 1]           # optimal coefficient
        adjusted = Y - b * (X - X_mean)
        return MCResult(adjusted.mean(), adjusted.std(ddof=1) / np.sqrt(n_paths), n_paths, method)

    raise ValueError("method must be 'plain', 'antithetic' or 'control'")

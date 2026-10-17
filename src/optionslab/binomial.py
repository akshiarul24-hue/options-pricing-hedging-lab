"""Cox-Ross-Rubinstein (CRR) binomial tree for European and American options.

At each of N steps the stock goes up by u = e^{sigma sqrt(dt)} or down by d = 1/u.
The risk-neutral probability of an up-move is p = (e^{(r-q)dt} - d) / (u - d):
the unique probability under which the stock earns the risk-free rate.

We fill in payoffs at the last step and walk backwards:
    V = e^{-r dt} [p V_up + (1 - p) V_down]
For an American option we also check, at every node, whether exercising
immediately is worth more than continuing.
"""
from __future__ import annotations

import numpy as np

from .black_scholes import _check_option, payoff


def crr_price(S0, K, T, r, sigma, option="call", q=0.0, n_steps=500, american=False) -> float:
    option = _check_option(option)
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("Time step too large: risk-neutral probability outside (0, 1)")
    disc = np.exp(-r * dt)

    # Stock prices at expiry: S0 u^j d^(N-j), j = 0..N (j = number of up-moves)
    j = np.arange(n_steps + 1)
    S = S0 * u**j * d ** (n_steps - j)
    V = payoff(S, K, option)

    for step in range(n_steps - 1, -1, -1):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        if american:
            j = np.arange(step + 1)
            S = S0 * u**j * d ** (step - j)
            V = np.maximum(V, payoff(S, K, option))
    return float(V[0])

"""Crank-Nicolson finite-difference solver for the Black-Scholes PDE.

The Black-Scholes PDE for an option value V(S, t) is

    dV/dt + (1/2) sigma^2 S^2 d2V/dS2 + (r - q) S dV/dS - r V = 0.

We change variables to x = ln(S) (makes the coefficients constant) and
tau = T - t (time remaining), which turns it into a forward-in-tau
diffusion problem, essentially the heat equation with drift and decay:

    dV/dtau = (1/2) sigma^2 V_xx + (r - q - sigma^2/2) V_x - r V,

starting from the payoff at tau = 0. Crank-Nicolson averages the explicit
and implicit schemes, which is second-order accurate in both dx and dtau
and unconditionally stable. The payoff has a kink at the strike that makes
plain Crank-Nicolson produce small oscillations, so the first few steps use
fully implicit Euler ("Rannacher smoothing").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from .black_scholes import _check_option, payoff


@dataclass
class PDEResult:
    price: float
    delta: float
    gamma: float
    S_grid: np.ndarray
    V_grid: np.ndarray


def crank_nicolson(S0, K, T, r, sigma, option="call", q=0.0, n_space=400, n_time=200,
                   width=5.0, rannacher_steps=4) -> PDEResult:
    """Price a European option on a log-price grid with Crank-Nicolson.

    n_space          number of grid intervals in x = ln S
    n_time           number of time steps
    width            grid half-width in standard deviations (sigma sqrt T)
    rannacher_steps  number of initial fully-implicit half-steps
    """
    option = _check_option(option)
    spread = width * sigma * np.sqrt(T)
    x_lo = min(np.log(S0), np.log(K)) - spread
    x_hi = max(np.log(S0), np.log(K)) + spread
    x = np.linspace(x_lo, x_hi, n_space + 1)
    dx = x[1] - x[0]
    S = np.exp(x)

    # Spatial operator L V_j = a V_{j-1} + b V_j + c V_{j+1} on interior nodes
    alpha = 0.5 * sigma**2 / dx**2
    beta = (r - q - 0.5 * sigma**2) / (2.0 * dx)
    a, b, c = alpha - beta, -2.0 * alpha - r, alpha + beta

    def boundary(tau):
        if option == "call":
            return 0.0, S[-1] * np.exp(-q * tau) - K * np.exp(-r * tau)
        return K * np.exp(-r * tau) - S[0] * np.exp(-q * tau), 0.0

    # Time stepping: Rannacher (implicit half-steps) then Crank-Nicolson
    dtau_full = T / n_time
    steps = []
    n_ran = min(rannacher_steps, 2 * n_time)
    steps += [(dtau_full / 2.0, 1.0)] * n_ran              # (dtau, theta) with theta = 1 -> implicit
    remaining = T - n_ran * dtau_full / 2.0
    n_cn = int(round(remaining / dtau_full))
    steps += [(remaining / n_cn, 0.5)] * n_cn if n_cn > 0 else []

    V = payoff(S, K, option)
    tau = 0.0
    m = n_space - 1  # number of interior unknowns
    for dtau, theta in steps:
        lo_old, hi_old = boundary(tau)
        tau_new = tau + dtau
        lo_new, hi_new = boundary(tau_new)

        # Right-hand side: (I + (1 - theta) dtau L) V_old + boundary terms
        Vi = V[1:-1]
        rhs = Vi + (1.0 - theta) * dtau * (a * V[:-2] + b * Vi + c * V[2:])
        rhs[0] += theta * dtau * a * lo_new
        rhs[-1] += theta * dtau * c * hi_new

        # Left-hand side tridiagonal matrix (I - theta dtau L) in banded form
        ab = np.zeros((3, m))
        ab[0, 1:] = -theta * dtau * c
        ab[1, :] = 1.0 - theta * dtau * b
        ab[2, :-1] = -theta * dtau * a

        V_new = np.empty_like(V)
        V_new[1:-1] = solve_banded((1, 1), ab, rhs)
        V_new[0], V_new[-1] = lo_new, hi_new
        V, tau = V_new, tau_new

    # Read off price and Greeks at S0 (Greeks from the grid via x = ln S)
    V_x = np.gradient(V, dx)
    V_xx = np.gradient(V_x, dx)
    delta_grid = V_x / S
    gamma_grid = (V_xx - V_x) / S**2
    x0 = np.log(S0)
    price = float(np.interp(x0, x, V))
    delta = float(np.interp(x0, x, delta_grid))
    gamma = float(np.interp(x0, x, gamma_grid))
    return PDEResult(price, delta, gamma, S, V)

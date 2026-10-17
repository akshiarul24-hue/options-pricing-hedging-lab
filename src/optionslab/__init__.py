"""optionslab - option pricing (closed form, Monte Carlo, PDE, binomial), Greeks,
implied volatility and delta-hedging simulation."""
from .black_scholes import (
    black76_price, bs_delta, bs_gamma, bs_greeks, bs_price, bs_rho, bs_theta, bs_vega, payoff,
)
from .binomial import crr_price
from .hedging import derman_kamal_std, hedge_pnl_attribution, simulate_delta_hedge
from .implied_vol import implied_vol, implied_vol_black76
from .monte_carlo import mc_european, simulate_gbm_paths
from .pde import crank_nicolson

__all__ = [
    "bs_price", "bs_delta", "bs_gamma", "bs_vega", "bs_theta", "bs_rho", "bs_greeks",
    "black76_price", "payoff", "crr_price", "crank_nicolson", "mc_european",
    "simulate_gbm_paths", "implied_vol", "implied_vol_black76", "simulate_delta_hedge",
    "hedge_pnl_attribution", "derman_kamal_std",
]
__version__ = "1.0.0"

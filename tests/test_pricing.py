"""Correctness tests for the closed form and the three numerical pricers."""
import numpy as np
import pytest

from optionslab import (
    bs_delta, bs_gamma, bs_price, bs_rho, bs_theta, bs_vega, crank_nicolson, crr_price, mc_european,
)
from optionslab.black_scholes import black76_price, put_call_parity_gap

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


def test_textbook_values():
    # Hull, Options, Futures and Other Derivatives: S=K=100, T=1, r=5%, sigma=20%
    assert bs_price(S0, K, T, R, SIGMA, "call") == pytest.approx(10.4506, abs=1e-4)
    assert bs_price(S0, K, T, R, SIGMA, "put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("strike", [70.0, 90.0, 100.0, 115.0, 140.0])
@pytest.mark.parametrize("q", [0.0, 0.02])
def test_put_call_parity(strike, q):
    c = bs_price(S0, strike, T, R, SIGMA, "call", q)
    p = bs_price(S0, strike, T, R, SIGMA, "put", q)
    assert put_call_parity_gap(c, p, S0, strike, T, R, q) == pytest.approx(0.0, abs=1e-10)


def test_price_bounds_and_monotonicity():
    strikes = np.linspace(50, 150, 101)
    calls = bs_price(S0, strikes, T, R, SIGMA, "call")
    assert np.all(calls >= np.maximum(S0 - strikes * np.exp(-R * T), 0) - 1e-12)
    assert np.all(calls <= S0)
    assert np.all(np.diff(calls) < 0)                       # calls cheaper at higher strikes
    vols = np.linspace(0.05, 1.0, 50)
    assert np.all(np.diff(bs_price(S0, K, T, R, vols)) > 0)  # more vol -> more value


def test_expiry_returns_payoff():
    assert bs_price(110.0, K, 0.0, R, SIGMA, "call") == pytest.approx(10.0)
    assert bs_price(90.0, K, 0.0, R, SIGMA, "put") == pytest.approx(10.0)
    assert bs_delta(110.0, K, 0.0, R, SIGMA, "call") == 1.0
    assert bs_delta(90.0, K, 0.0, R, SIGMA, "put") == -1.0


def test_black76_matches_black_scholes():
    q = 0.015
    F, DF = S0 * np.exp((R - q) * T), np.exp(-R * T)
    for opt in ("call", "put"):
        assert black76_price(F, 95.0, T, DF, SIGMA, opt) == pytest.approx(bs_price(S0, 95.0, T, R, SIGMA, opt, q))


@pytest.mark.parametrize("option", ["call", "put"])
def test_greeks_match_finite_differences(option):
    h = 1e-4
    price = lambda S=S0, t=T, r=R, v=SIGMA: float(bs_price(S, K, t, r, v, option))
    assert bs_delta(S0, K, T, R, SIGMA, option) == pytest.approx((price(S=S0 + h) - price(S=S0 - h)) / (2 * h), abs=1e-6)
    assert bs_gamma(S0, K, T, R, SIGMA) == pytest.approx((price(S=S0 + 1e-2) - 2 * price() + price(S=S0 - 1e-2)) / 1e-4, abs=1e-5)
    assert bs_vega(S0, K, T, R, SIGMA) == pytest.approx((price(v=SIGMA + h) - price(v=SIGMA - h)) / (2 * h), abs=1e-5)
    assert bs_theta(S0, K, T, R, SIGMA, option) == pytest.approx(-(price(t=T + h) - price(t=T - h)) / (2 * h), abs=1e-5)
    assert bs_rho(S0, K, T, R, SIGMA, option) == pytest.approx((price(r=R + h) - price(r=R - h)) / (2 * h), abs=1e-5)


@pytest.mark.parametrize("method", ["plain", "antithetic", "control"])
@pytest.mark.parametrize("option", ["call", "put"])
def test_monte_carlo_within_four_standard_errors(method, option):
    res = mc_european(S0, K, T, R, SIGMA, option, n_paths=200_000, method=method, seed=123)
    exact = float(bs_price(S0, K, T, R, SIGMA, option))
    assert abs(res.price - exact) < 4 * res.std_err


def test_variance_reduction_actually_reduces_variance():
    plain = mc_european(S0, K, T, R, SIGMA, n_paths=200_000, method="plain", seed=1)
    anti = mc_european(S0, K, T, R, SIGMA, n_paths=200_000, method="antithetic", seed=1)
    ctrl = mc_european(S0, K, T, R, SIGMA, n_paths=200_000, method="control", seed=1)
    assert anti.std_err < plain.std_err and ctrl.std_err < anti.std_err


@pytest.mark.parametrize("option", ["call", "put"])
def test_crank_nicolson_price_and_greeks(option):
    res = crank_nicolson(S0, K, T, R, SIGMA, option, n_space=800, n_time=400)
    assert res.price == pytest.approx(float(bs_price(S0, K, T, R, SIGMA, option)), abs=5e-4)
    assert res.delta == pytest.approx(float(bs_delta(S0, K, T, R, SIGMA, option)), abs=1e-4)
    assert res.gamma == pytest.approx(float(bs_gamma(S0, K, T, R, SIGMA)), abs=1e-4)


def test_crank_nicolson_is_second_order():
    errs = [abs(crank_nicolson(S0, K, T, R, SIGMA, n_space=n, n_time=n // 2).price - float(bs_price(S0, K, T, R, SIGMA)))
            for n in (80, 160, 320)]
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.25)
    assert errs[1] / errs[2] == pytest.approx(4.0, rel=0.25)


@pytest.mark.parametrize("option", ["call", "put"])
def test_binomial_converges_to_black_scholes(option):
    assert crr_price(S0, K, T, R, SIGMA, option, n_steps=2000) == pytest.approx(float(bs_price(S0, K, T, R, SIGMA, option)), abs=2e-3)


def test_american_options():
    euro_put = crr_price(S0, K, T, R, SIGMA, "put", n_steps=1000)
    amer_put = crr_price(S0, K, T, R, SIGMA, "put", n_steps=1000, american=True)
    assert amer_put > euro_put + 0.4                          # early exercise has value for puts
    assert amer_put == pytest.approx(6.09, abs=0.01)          # textbook value
    euro_call = crr_price(S0, K, T, R, SIGMA, "call", n_steps=1000)
    amer_call = crr_price(S0, K, T, R, SIGMA, "call", n_steps=1000, american=True)
    assert amer_call == pytest.approx(euro_call, abs=1e-10)   # never exercise a call early (no dividends)
    deep = crr_price(60.0, K, T, R, SIGMA, "put", n_steps=500, american=True)
    assert deep == pytest.approx(K - 60.0, abs=1e-8)          # deep ITM: exercise immediately

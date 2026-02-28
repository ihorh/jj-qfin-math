r"""The Black-Scholes-Merton model for pricing European options.

This module provides tools for pricing European call and put options using the
**Black-Scholes-Merton (BSM)** formula, as well as computing implied volatilities.

It includes:

- ``bsm_option_eu_prices``: calculates fair prices for European call and put options.
- ``bsm_option_eu_iv``: computes implied volatility from market option prices.
- ``get_option_price_bound``: computes no-arbitrage price bounds for European options.

"""

from math import exp, log, sqrt
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

type _OptionSide = Literal["call", "put"]


def bsm_option_eu_iv(  # noqa: PLR0913
    option_type: _OptionSide,
    option_price: float,
    *,
    s0: float,
    k: float,
    t: float,
    r: float,
    q: float = 0,
) -> float:
    r"""Compute the implied volatility of a European option from its market price.

    Parameters
    ----------
    option_type : {"call", "put"}
        Whether the option is a call or a put.
    option_price : float
        Market price of the option.
    s0 : float
        Current spot price of the underlying asset.
    k : float
        Strike price (exercise price) of the option.
    t : float
        Time to maturity in years (fractional).
    r : float
        Continuously compounded risk-free interest rate.
    q : float, default 0.0
        Continuously compounded dividend yield.

    Returns
    -------
    float
        The implied volatility ($\sigma$) that reconciles the option price with the BSM formula.
        Returns ``np.nan`` if the option price is outside the no-arbitrage bounds
        (i.e., outside the Black-Scholes model manifold).

    """
    _bsm_opt_eu_iv_preconditions(s0, k, t, r, q, option_price)

    lower, upper = get_option_price_bound(option_type, s0=s0, k=k, t=t, r=r)
    if not (lower <= option_price <= upper):
        return np.nan

    if np.isclose(option_price, lower, atol=1e-8):
        return 0.0

    def price_error_fn(sigma: float) -> float:
        c, p = bsm_option_eu_prices(s0=s0, k=k, t=t, r=r, sigma=sigma, q=q)
        return c - option_price if option_type == "call" else p - option_price

    try:
        f = brentq(price_error_fn, a=0.001, b=5.0, maxiter=100)
        if isinstance(f, float):
            return f
        msg = f"Failed to find implied volatility for {option_type} option"
        raise RuntimeError(msg)
    except ValueError as e:
        msg = f"Failed to find implied volatility for {option_type} option: {e}"
        raise RuntimeError(msg) from e


def _bsm_opt_eu_iv_preconditions(s0: float, k: float, t: float, r: float, q: float, option_price: float) -> None:  # noqa: PLR0913
    _require(s0 > 0, "s0 must be positive")
    _require(k > 0, "k must be positive")
    _require(t > 0, "t must be positive")
    _require(r > -1, "r must be >= -1")
    _require(q >= 0, "q must be non-negative")
    _require(option_price > 0, "option_price must be positive")


def bsm_option_eu_prices(*, s0: float, k: float, t: float, r: float, sigma: float, q: float = 0) -> tuple[float, float]:  # noqa: PLR0913
    r"""Calculate Black-Scholes-Merton (BSM) prices for European call and put options.

    Parameters
    ----------
    s0 : float
        Current spot price of the underlying asset.
    k : float
        Strike price (exercise price) of the option.
    t : float
        Time to maturity in years (fractional).
    r : float
        Continuously compounded risk-free interest rate (e.g., 0.05 for 5%).
    sigma : float
        Annualized volatility of the underlying asset returns.
    q : float, default 0.0
        Continuously compounded dividend yield.

    Returns
    -------
    tuple[float, float]
        A tuple containing (call_price, put_price).

        * call_price : The theoretical fair value of the European call option.
        * put_price : The theoretical fair value of the European put option.

    """
    _bsm_opt_eu_prices_preconditions(s0, k, t, r, sigma, q)

    sigma_sqrt_t = sigma * sqrt(t)

    d1 = (log(s0 / k) + (r - q + 0.5 * sigma**2) * t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t

    s0_pv = s0 * exp(-q * t)
    k_pv = k * exp(-r * t)

    sn = norm()

    call_price = s0_pv * sn.cdf(d1) - k_pv * sn.cdf(d2)
    put_price = k_pv * sn.cdf(-d2) - s0_pv * sn.cdf(-d1)

    return call_price, put_price


def _bsm_opt_eu_prices_preconditions(s0: float, k: float, t: float, r: float, sigma: float, q: float) -> None:  # noqa: PLR0913
    _require(s0 > 0, "s0 must be positive")
    _require(k > 0, "k must be positive")
    _require(t > 0, "t must be positive")
    _require(r > -1, "r must be >= -1")
    _require(sigma > 0, "sigma must be positive")
    _require(q >= 0, "q must be non-negative")


def get_option_price_bound(
    side: _OptionSide,
    *,
    s0: float,
    k: float,
    t: float,
    r: float,
) -> tuple[float, float]:
    r"""Calculate the no-arbitrage bounds for a European option price.

    Parameters
    ----------
    side : {"call", "put"}
        Whether the option is a call or a put.
    s0 : float
        Current spot price of the underlying asset.
    k : float
        Strike price (exercise price) of the option.
    t : float
        Time to maturity in years (fractional).
    r : float
        Continuously compounded risk-free interest rate.

    Returns
    -------
    tuple[float, float]
        A tuple (lower_bound, upper_bound) representing the no-arbitrage price bounds.

    """
    _get_option_price_bound_preconditions(s0, k, t, r)
    df = exp(-r * t)
    if side == "call":
        return max(0.0, s0 - k * df), s0
    return max(0.0, k * df - s0), k * df


def _get_option_price_bound_preconditions(s0: float, k: float, t: float, r: float) -> None:
    _require(s0 > 0, "s0 must be positive")
    _require(k > 0, "k must be positive")
    _require(t > 0, "t must be positive")
    _require(r > -1, "r must be >= -1")


def _require(condition: bool, descr: str | None = None, *, msg: str = "Invalid argument") -> None:  # noqa: FBT001
    if condition:
        return
    if descr:
        msg = f"{msg}: {descr}"
    raise ValueError(msg)

r"""The Black-Scholes-Merton model for pricing European options.

This module provides tools for pricing European call and put options using the
**Black-Scholes-Merton (BSM)** formula, as well as computing implied volatilities.

It includes:

- ``bsm_option_eu_prices``: calculates fair prices for European call and put options.
- ``bsm_option_eu_iv``: computes implied volatility from market option prices.
- ``get_option_price_bound``: computes no-arbitrage price bounds for European options.

"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Literal, TypedDict

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

type _OptionSide = Literal["call", "put"]


@dataclass(frozen=True, slots=True, kw_only=True)
class OptionsIVSolverSettings:
    r"""Settings for the implied volatility solver.

    Parameters
    ----------
    sigma_min : float, default 1e-6
        Lower bound for volatility search.
    sigma_max : float, default 10.0
        Upper bound for volatility search.
    sigma_atol : float, default 1e-8
        Absolute tolerance for the implied volatility solver.
    max_iter : int, default 128
        Maximum number of iterations for the optimizer.
    price_atol : float, default 1e-8
        Tolerance for checking if option price equals lower bound.

    """

    sigma_min: float = 1e-6
    sigma_max: float = 10.0
    sigma_atol: float = 1e-8
    max_iter: int = 128
    price_atol: float = 1e-8


def _to_brentq_kwargs(s: OptionsIVSolverSettings) -> _BrentqKwargs:
    return {"a": s.sigma_min, "b": s.sigma_max, "xtol": s.sigma_atol, "maxiter": s.max_iter}


OPTIONS_IV_SOLVER_SETTINGS_DEFAULT = OptionsIVSolverSettings()


class _BrentqKwargs(TypedDict):
    a: float
    b: float
    xtol: float
    maxiter: int


def bsm_option_eu_iv(  # noqa: PLR0913
    *,
    # 1) Underlying data (price, vol)
    s0: float,
    sigma: float,
    q: float = 0,
    # 2) Option data: ttm, strike, side
    t: float,
    k: float,
    option_type: _OptionSide,
    option_price: float,
    # 3) Market data
    r: float,
    # Optimizer settings
    solver_settings: OptionsIVSolverSettings = OPTIONS_IV_SOLVER_SETTINGS_DEFAULT,
) -> float:
    r"""Compute the implied volatility of a European option from its market price.

    Parameters
    ----------
    s0 : float
        Current spot price of the underlying asset.
    sigma : float
        Initial guess for the implied volatility (used by the solver).
    q : float
        Continuously compounded dividend yield.
    t : float
        Time to maturity in years (fractional).
    k : float
        Strike price (exercise price) of the option.
    option_type : {"call", "put"}
        Whether the option is a call or a put.
    option_price : float
        Market price of the option.
    r : float
        Continuously compounded risk-free interest rate.
    solver_settings : OptionsIVSolverSettings, optional
        Settings for the implied volatility solver. Defaults to
        :attr:`OptionsIVSolverSettings_DEFAULT`.

    Returns
    -------
    float
        The implied volatility ($\sigma$) that reconciles the option price with the BSM formula.
        Returns ``np.nan`` if the option price is outside the no-arbitrage bounds
        (i.e., outside the Black-Scholes model manifold).

    """
    _bsm_opt_eu_iv_preconditions(s0, sigma, q, t, k, r, option_price)

    lower, upper = get_option_price_bound(s0=s0, t=t, k=k, side=option_type, r=r)
    if not (lower <= option_price <= upper):
        return np.nan

    if np.isclose(option_price, lower, atol=solver_settings.price_atol):
        return 0.0

    def price_error_fn(sigma: float) -> float:
        c, p = bsm_option_eu_prices(s0=s0, sigma=sigma, q=q, t=t, k=k, r=r)
        return c - option_price if option_type == "call" else p - option_price

    try:
        f = brentq(price_error_fn, **_to_brentq_kwargs(solver_settings))
        if isinstance(f, float):
            return f
        msg = f"Failed to find implied volatility for {option_type} option"
        raise RuntimeError(msg)
    except ValueError as e:
        msg = f"Failed to find implied volatility for {option_type} option: {e}"
        raise RuntimeError(msg) from e


def _bsm_opt_eu_iv_preconditions(  # noqa: PLR0913
    s0: float,
    sigma: float,
    q: float,
    t: float,
    k: float,
    r: float,
    option_price: float,
) -> None:
    _require(s0 > 0, "s0 must be positive")
    _require(sigma > 0, "sigma must be positive")
    _require(t > 0, "t must be positive")
    _require(k > 0, "k must be positive")
    _require(r > -1, "r must be >= -1")
    _require(q >= 0, "q must be non-negative")
    _require(option_price > 0, "option_price must be positive")


def bsm_option_eu_prices(  # noqa: PLR0913
    *,
    # 1) Underlying data (price, vol)
    s0: float,
    sigma: float,
    q: float,
    # 2) Option data: ttm, strike
    t: float,
    k: float,
    # 3) Market data
    r: float,
) -> tuple[float, float]:
    r"""Calculate Black-Scholes-Merton (BSM) prices for European call and put options.

    Parameters
    ----------
    s0 : float
        Current spot price of the underlying asset.
    sigma : float
        Annualized volatility of the underlying asset returns.
    q : float
        Continuously compounded dividend yield.
    t : float
        Time to maturity in years (fractional).
    k : float
        Strike price (exercise price) of the option.
    r : float
        Continuously compounded risk-free interest rate (e.g., 0.05 for 5%).

    Returns
    -------
    tuple[float, float]
        A tuple containing (call_price, put_price).

        * call_price : The theoretical fair value of the European call option.
        * put_price : The theoretical fair value of the European put option.

    """
    _bsm_opt_eu_prices_preconditions(s0, sigma, q, t, k, r)

    sigma_sqrt_t = sigma * sqrt(t)

    d1 = (log(s0 / k) + (r - q + 0.5 * sigma**2) * t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t

    s0_pv = s0 * exp(-q * t)
    k_pv = k * exp(-r * t)

    sn = norm()

    call_price = s0_pv * sn.cdf(d1) - k_pv * sn.cdf(d2)
    put_price = k_pv * sn.cdf(-d2) - s0_pv * sn.cdf(-d1)

    return call_price, put_price


def _bsm_opt_eu_prices_preconditions(  # noqa: PLR0913
    s0: float,
    sigma: float,
    q: float,
    t: float,
    k: float,
    r: float,
) -> None:
    _require(s0 > 0, "s0 must be positive")
    _require(k > 0, "k must be positive")
    _require(t > 0, "t must be positive")
    _require(r > -1, "r must be >= -1")
    _require(sigma > 0, "sigma must be positive")
    _require(q >= 0, "q must be non-negative")


def get_option_price_bound(
    *,
    # 1) Underlying data (price)
    s0: float,
    # 2) Option data: ttm, strike, side
    t: float,
    k: float,
    side: _OptionSide,
    # 3) Market data
    r: float,
) -> tuple[float, float]:
    r"""Calculate the no-arbitrage bounds for a European option price.

    Parameters
    ----------
    s0 : float
        Current spot price of the underlying asset.
    t : float
        Time to maturity in years (fractional).
    k : float
        Strike price (exercise price) of the option.
    side : {"call", "put"}
        Whether the option is a call or a put.
    r : float
        Continuously compounded risk-free interest rate.

    Returns
    -------
    tuple[float, float]
        A tuple (lower_bound, upper_bound) representing the no-arbitrage price bounds.

    """
    _get_option_price_bound_preconditions(s0, t, k, r)
    df = exp(-r * t)
    if side == "call":
        return max(0.0, s0 - k * df), s0
    return max(0.0, k * df - s0), k * df


def _get_option_price_bound_preconditions(s0: float, t: float, k: float, r: float) -> None:
    _require(s0 > 0, "s0 must be positive")
    _require(t > 0, "t must be positive")
    _require(k > 0, "k must be positive")
    _require(r > -1, "r must be >= -1")


def _require(condition: bool, descr: str | None = None, *, msg: str = "Invalid argument") -> None:  # noqa: FBT001
    if condition:
        return
    if descr:
        msg = f"{msg}: {descr}"
    raise ValueError(msg)

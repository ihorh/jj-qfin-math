r"""GBM Stock Price Forecast Module.

This module provides tools for computing short- or long-term forecasts of stock prices
under the **Geometric Brownian Motion (GBM)** assumption. It assumes that log returns
are normally distributed with a given drift and volatility, and produces the corresponding
lognormal distribution of future stock prices.

It includes:

- ``GBMStockPriceForecast`` dataclass: stores expected price, standard deviation, and the
  underlying lognormal distribution.
- ``gdm_stock_price_forecast`` function: computes expected price, standard deviation, and
  the lognormal distribution for a given time horizon.
- ``_ContinuousDistribution`` protocol: defines the interface for continuous probability
  distributions used in the forecast.

Notes
-----
**Math refresher**:

Assume GBM: ``dS/S = μ * dt + s * dW_t``

- where ``μ`` is the continuously compounded drift and ``s`` is the log-return volatility.

Applying Itô's lemma to ln S_t gives:

``d(ln S_t) = (μ - ½σ²) dt + s dW_t`` => ``ln S_t = ln S₀ + (μ - ½σ²)t + s W_t``.

By definition of Brownian motion,

``W_t ~ N(0, t)`` => ``ln S_t ~ N(ln S₀ + (μ - ½σ²)t, σ² t)`` => ``S_t ~ lognormal``

"""

import math
from dataclasses import dataclass
from math import exp, log, sqrt
from textwrap import dedent
from typing import Protocol, cast, overload

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import lognorm


class _ContinuousDistribution(Protocol):
    def mean(self) -> float: ...
    def median(self) -> float: ...
    def pdf(self, xs: ArrayLike) -> np.ndarray: ...
    def cdf(self, xs: ArrayLike) -> np.ndarray: ...
    @overload
    def interval(self, confidence: float) -> tuple[float, float]: ...
    @overload
    def interval(self, confidence: ArrayLike) -> np.ndarray | float | tuple[float, ...] | tuple[np.ndarray, ...]: ...


@dataclass(slots=True, frozen=True, kw_only=True)
class GBMStockPriceForecast:
    r"""Forecast of a stock price under the **Geometric Brownian Motion** model.

    Attributes
    ----------
    price_expected : float
        The expected stock price at the forecast horizon (``E[S_t]``).
    price_std : float
        The standard deviation of the stock price at the forecast horizon.
    price_dist : _ContinuousDistribution
        The lognormal distribution of ``S_t`` representing the full probability distribution.
        Use ``price_dist.interval(0.95)`` to get the 95% confidence interval.
    _price_std_approx : float | None, optional
        Approximate standard deviation using the short-time linear approximation
        ``S0 * sigma * sqrt(t)``. Useful for very small ``t``.

    """

    price_expected: float
    price_std: float
    price_dist: _ContinuousDistribution
    _price_std_approx: float | None = None

    def __str__(self) -> str:
        return "GBMStockPriceForecast" + dedent(f"""
        - price_expected: {self.price_expected}
        - price_std: {self.price_std}
        - _price_std_approx: {self._price_std_approx}
        """)


def gdm_stock_price_forecast(*, s0: float, t: float, mu: float, sigma: float) -> GBMStockPriceForecast:
    r"""Compute the GBM-based forecast of a stock price at a given future time.

    Parameters
    ----------
    s0 : float
        Current stock price (``S_0``).
    t : float
        Time horizon to forecast expressed in the same units as the drift and volatility, (typically years).
    mu : float
        Expected arithmetic return of the stock (drift).
    sigma : float
        Standard deviation of log returns (volatility).

    Returns
    -------
    GBMStockPriceForecast
        Dataclass containing:
        - ``price_expected``: expected stock price ``E[S_t]``
        - ``price_std``: standard deviation of ``S_t``
        - ``price_dist``: lognormal distribution object of ``S_t``
        - ``_price_std_approx``: approximate standard deviation ``S0 * sigma * sqrt(t)``

    Notes
    -----
    - Assumes log returns are normally distributed:
        ``R_t = ln(S_t / S0) ~ N((μ - 0.5 σ²)t, σ² t)``
    - The resulting stock price S_t is lognormally distributed by definition:
        ``S_t = S0 * exp(R_t)``
    - `_price_std_approx` is only accurate for very short horizons.

    """
    _gdm_stock_price_forecast_preconditions(s0, t, mu, sigma)

    # log returns are assumed to be normally distributed with
    # - mean <- logr_mean
    # - std  <- logr_std (var -> logr_std**2)
    logr_mean = mu - 0.5 * sigma**2
    # log returns at time t
    logr_mean_t = logr_mean * t
    logr_std_t = sigma * sqrt(t)
    # st - price at time t is lognorm distributed with:
    # logr = ln(st) - ln(s0) = ln(st/s0) => ln(st) = ln(s0) + logr
    lst_mean = log(s0) + logr_mean_t
    lst_std = logr_std_t
    # convert to scipy parametrization
    st_mean = exp(lst_mean)
    st_dist = cast("_ContinuousDistribution", lognorm(s=lst_std, scale=st_mean))

    price_expected = s0 * exp(mu * t)
    price_std = price_expected * sqrt(exp(sigma**2 * t) - 1.0)

    return _gdm_stock_price_forecast_postconditions(
        GBMStockPriceForecast(
            price_expected=price_expected,
            price_std=price_std,
            price_dist=st_dist,
            _price_std_approx=s0 * sigma * sqrt(t),
        ),
        st_mean=st_mean,
    )


def _gdm_stock_price_forecast_preconditions(s0: float, t: float, _r_mean_cc: float, logr_std: float) -> None:
    _require(s0 > 0, "s0 must be positive")
    _require(t > 0, "t must be positive")
    _require(logr_std > 0, "logr_std must be positive")


def _gdm_stock_price_forecast_postconditions(f: GBMStockPriceForecast, *, st_mean: float) -> GBMStockPriceForecast:
    st_dist = f.price_dist
    price_expected = f.price_expected

    def isclose(a: float, b: float) -> bool:
        return math.isclose(a, b, abs_tol=10e-14, rel_tol=0)

    _check(isclose(st_dist.median(), st_mean), "dist median must be equal to st_mean")
    _check(isclose(st_dist.mean(), price_expected), "dist mean must be equal to price_expected")

    return f


def _require(condition: bool, descr: str | None = None, *, msg: str = "Invalid argument") -> None:  # noqa: FBT001
    if condition:
        return
    if descr:
        msg = f"{msg}: {descr}"
    raise ValueError(msg)


def _check(condition: bool, descr: str | None = None, *, msg: str = "Invalid state") -> None:  # noqa: FBT001
    if condition:
        return
    if descr:
        msg = f"{msg}: {descr}"
    raise RuntimeError(msg)

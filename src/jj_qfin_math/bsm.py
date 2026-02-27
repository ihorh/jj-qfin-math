r"""The Black-Scholes-Merton model for pricing options.

TODO (ihor): add vectorized version of ``bsm_option_eu_prices``.
"""

from math import exp, log, sqrt

from scipy.stats import norm


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
    sigma_sqrt_t = sigma * sqrt(t)

    d1 = (log(s0 / k) + (r - q + 0.5 * sigma**2) * t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t

    s0_pv = s0 * exp(-q * t)
    k_pv = k * exp(-r * t)

    sn = norm()

    call_price = s0_pv * sn.cdf(d1) - k_pv * sn.cdf(d2)
    put_price = k_pv * sn.cdf(-d2) - s0_pv * sn.cdf(-d1)

    return call_price, put_price

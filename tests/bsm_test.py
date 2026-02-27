from math import exp, isclose

from jj_qfin_math.bsm import bsm_option_eu_prices


def test_bsm_option_eu_prices_prob_14_4():
    c, p = bsm_option_eu_prices(
        s0=50,
        k=50,
        t=3 / 12,
        r=10 / 100,
        sigma=30 / 100,
    )

    print(c, p)

    assert isclose(c, 3.61044, rel_tol=0, abs_tol=1e-4)
    assert isclose(p, 2.37594, rel_tol=0, abs_tol=1e-4)


def test_bsm_option_eu_prices_prob_14_5():
    d = 1.50
    dt = 2 / 12
    r = 10 / 100
    c, p = bsm_option_eu_prices(
        s0=50 - d * exp(-r * dt),
        k=50,
        t=3 / 12,
        r=r,
        sigma=30 / 100,
    )

    print(c, p)

    assert isclose(c, 2.7895, rel_tol=0, abs_tol=1e-4)
    assert isclose(p, 3.0302, rel_tol=0, abs_tol=1e-4)

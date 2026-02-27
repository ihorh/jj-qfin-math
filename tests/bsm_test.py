from math import isclose

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

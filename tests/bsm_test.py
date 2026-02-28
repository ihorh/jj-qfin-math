from collections import namedtuple
from math import exp, isclose
from typing import Any, assert_never

import pytest

from jj_qfin_math.bsm import bsm_option_eu_prices


def _test_id_fn(val: Any) -> str:  # noqa: ANN401
    match val:
        case str():
            return val
        case (s0, k, t, *_):
            return f"{s0:.2f}/{k:.2f}-{t:.2f}"
        case (c, p):
            return f"{c:.2f}/{p:.2f}"
        case _:
            assert_never(val)


S0_14_5 = 50.0 - 1.50 * exp(-0.10 * 2 / 12)

HULL_TEST_CASES = [
    ("prob_14_4", (50.0, 50.0, 3 / 12, 0.10, 0.30, 0.0), (3.61044, 2.37594)),
    ("prob_14_5", (S0_14_5, 50.0, 3 / 12, 0.10, 0.30, 0.0), (2.7895, 3.0302)),
    ("prob_14_13", (52.0, 50.0, 3 / 12, 0.12, 0.30, 0.0), (5.0574, 1.5797)),
    ("prob_14_14", (69.0, 70.0, 6 / 12, 0.05, 0.35, 0.0), (7.1297, 6.4014)),
]

ExpOptPrices = namedtuple("ExpOptPrices", ["c", "p"])  # noqa: PYI024
BSMArgs = namedtuple("BSMArgs", ["s0", "k", "t", "r", "sigma", "q"])  # noqa: PYI024


@pytest.mark.parametrize(("_label", "bsm_args", "exp_opt_prices"), HULL_TEST_CASES, ids=_test_id_fn)
def test_hull_ch_14_eu_option_prices(_label: str, bsm_args: BSMArgs, exp_opt_prices: ExpOptPrices) -> None:
    """Validate BSM implementation against John C. Hull's textbook, chapter 14.

    This ensures the library accurately reproduces the closed-form solutions
    for various moneyness, interest rate, and dividend scenarios.
    """
    c, p = bsm_option_eu_prices(**BSMArgs(*bsm_args)._asdict())
    exp_call, exp_put = exp_opt_prices
    assert isclose(c, exp_call, abs_tol=1e-4), f"Call price mismatch: Expected {exp_call}, got {c:.5f}"
    assert isclose(p, exp_put, abs_tol=1e-4), f"Put price mismatch: Expected {exp_put}, got {p:.5f}"

import math

from jj_qfin_math.gbm import gdm_stock_price_forecast


def test_gbm_stock_price_forecast():
    s0 = 50.0
    r = 16 / 100
    vol = 30 / 100
    t = 1 / 252
    p = 0.95
    f = gdm_stock_price_forecast(s0=s0, t=t, mu=r, sigma=vol)
    print(f)
    print(f.price_dist.interval(0.95))

    def isclose(a: float, b: float) -> bool:
        return math.isclose(a, b, abs_tol=10e-4, rel_tol=0)

    assert isclose(f.price_expected, 50.03176)
    assert isclose(f.price_expected, s0 * math.exp(r * t))
    assert isclose(f.price_std, 0.9456)
    assert isclose(f.price_std, f.price_expected * math.sqrt(math.exp(vol**2 * t) - 1.0))
    assert f._price_std_approx  # noqa: SLF001
    assert isclose(f._price_std_approx, 0.9449)  # noqa: SLF001

    confidence_interval = f.price_dist.interval(p)
    assert isinstance(confidence_interval, tuple)
    st_min, st_max = confidence_interval
    assert isinstance(st_min, float)
    assert isinstance(st_max, float)
    assert isclose(st_min, 48.20388)
    assert isclose(st_max, 51.91040)

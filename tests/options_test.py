import numpy as np
import pytest

from jj_qfin_math.options import OptionsGrid


def test_options_grid_scalars() -> None:
    """Test OptionsGrid with scalar inputs only."""
    grid = OptionsGrid(
        s0=100.0,
        vol=0.2,
        q=0.02,
        t=0.25,
        k=100.0,
        r=0.05,
    )

    assert grid.s0 == 100.0
    assert grid.vol == 0.2
    assert grid.q == 0.02
    assert grid.t == 0.25
    assert grid.k == 100.0
    assert grid.r == 0.05
    assert grid.valid_mask is True


def test_options_grid_scalars_default_q() -> None:
    """Test OptionsGrid with default q (zero)."""
    grid = OptionsGrid(
        s0=100.0,
        vol=0.2,
        t=0.25,
        k=100.0,
        r=0.05,
    )

    assert grid.q == 0.0


def test_options_grid_arrays() -> None:
    """Test OptionsGrid with array/list inputs."""
    s0 = np.array([100.0, 105.0, 110.0])
    vol = np.array([0.2, 0.25, 0.3])
    q = np.array([0.02, 0.02, 0.02])
    t = np.array([0.25, 0.5, 1.0])
    k = np.array([100.0, 105.0, 110.0])
    r = np.array([0.05, 0.05, 0.05])

    grid = OptionsGrid(s0=s0, vol=vol, q=q, t=t, k=k, r=r)

    assert np.array_equal(grid.s0, s0)
    assert np.array_equal(grid.vol, vol)
    assert np.array_equal(grid.q, q)
    assert np.array_equal(grid.t, t)
    assert np.array_equal(grid.k, k)
    assert np.array_equal(grid.r, r)
    assert np.array_equal(grid.valid_mask, [True, True, True])


def test_options_grid_mixed() -> None:
    """Test OptionsGrid with mixed scalar and array inputs."""
    s0 = 100.0
    vol = np.array([0.2, 0.25, 0.3])
    q = 0.02
    t = [0.25, 0.5, 1.0]
    k = np.array([100.0, 105.0, 110.0])
    r = 0.05

    grid = OptionsGrid(s0=s0, vol=vol, q=q, t=t, k=k, r=r)

    assert grid.s0 == 100.0
    assert np.array_equal(grid.vol, vol)
    assert grid.q == 0.02
    assert np.array_equal(grid.t, [0.25, 0.5, 1.0])
    assert np.array_equal(grid.k, k)
    assert grid.r == 0.05


def test_options_grid_invalid_shape_raises() -> None:
    """Test OptionsGrid raises on mismatched array shapes."""
    s0 = np.array([100.0, 105.0])
    vol = np.array([0.2, 0.25, 0.3])

    with pytest.raises(ValueError, match="same shape"):
        OptionsGrid(s0=s0, vol=vol, t=[0.25, 0.5, 1.0], k=[100.0, 105.0, 110.0], r=[0.05, 0.05, 0.05])


@pytest.mark.parametrize("s0", [0, -1, -100])
def test_options_grid_invalid_s0_mask(s0: float) -> None:
    """Test valid_mask is False for non-positive s0."""
    grid = OptionsGrid(s0=s0, vol=0.2, t=0.25, k=100.0, r=0.05)
    assert grid.valid_mask is False


@pytest.mark.parametrize("vol", [0, -0.1, -1])
def test_options_grid_invalid_vol_mask(vol: float) -> None:
    """Test valid_mask is False for non-positive vol."""
    grid = OptionsGrid(s0=100.0, vol=vol, t=0.25, k=100.0, r=0.05)
    assert grid.valid_mask is False


@pytest.mark.parametrize("q", [-0.01, -0.1, -1])
def test_options_grid_invalid_q_mask(q: float) -> None:
    """Test valid_mask is False for negative q."""
    grid = OptionsGrid(s0=100.0, vol=0.2, q=q, t=0.25, k=100.0, r=0.05)
    assert grid.valid_mask is False


@pytest.mark.parametrize("t", [0, -0.1, -1])
def test_options_grid_invalid_t_mask(t: float) -> None:
    """Test valid_mask is False for non-positive t."""
    grid = OptionsGrid(s0=100.0, vol=0.2, t=t, k=100.0, r=0.05)
    assert grid.valid_mask is False


@pytest.mark.parametrize("k", [0, -1, -100])
def test_options_grid_invalid_k_mask(k: float) -> None:
    """Test valid_mask is False for non-positive k."""
    grid = OptionsGrid(s0=100.0, vol=0.2, t=0.25, k=k, r=0.05)
    assert grid.valid_mask is False


@pytest.mark.parametrize("r", [-1, -1.5, -10])
def test_options_grid_invalid_r_mask(r: float) -> None:
    """Test valid_mask is False for r <= -1."""
    grid = OptionsGrid(s0=100.0, vol=0.2, t=0.25, k=100.0, r=r)
    assert grid.valid_mask is False


def test_options_grid_valid_mask_partial_invalid() -> None:
    """Test valid_mask correctly identifies invalid entries in arrays."""
    grid = OptionsGrid(
        s0=[100.0, -10.0, 50.0],
        vol=[0.2, 0.3, 0.1],
        q=[0.02, 0.02, -0.01],
        t=[0.25, 0.5, 1.0],
        k=[100.0, 100.0, 100.0],
        r=[0.05, 0.05, 0.05],
    )

    expected_mask = [True, False, False]
    assert np.array_equal(grid.valid_mask, expected_mask)


def test_options_grid_r_boundary() -> None:
    """Test r = 0 is valid (boundary case, r > -1)."""
    grid = OptionsGrid(s0=100.0, vol=0.2, t=0.25, k=100.0, r=0.0)
    assert grid.valid_mask is True


def test_options_grid_single_element_arrays() -> None:
    """Test OptionsGrid with single-element arrays."""
    grid = OptionsGrid(
        s0=np.array([100.0]),
        vol=np.array([0.2]),
        q=np.array([0.02]),
        t=np.array([0.25]),
        k=np.array([100.0]),
        r=np.array([0.05]),
    )

    assert grid.s0 == 100.0
    assert grid.valid_mask

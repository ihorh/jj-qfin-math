from typing import Any

import numpy as np
from numpy.typing import ArrayLike


class OptionsGrid:
    def __init__(  # noqa: PLR0913
        self,
        *,
        # 1) Underlying data (price, vol)
        s0: ArrayLike,
        vol: ArrayLike,
        q: ArrayLike = 0,
        # 2) Option data: ttm, strike
        t: ArrayLike,
        k: ArrayLike,
        # 3) Market data
        r: ArrayLike,
    ) -> None:
        b_s0, b_vol, b_q, b_t, b_k, b_r = self._require_args_scalar_or_1d_arrays_of_same_shape(s0, vol, q, t, k, r)
        self._s0, self._vol, self._q, self._t, self._k, self._r = b_s0, b_vol, b_q, b_t, b_k, b_r
        self._valid_mask = (b_s0 > 0) & (b_vol > 0) & (b_q >= 0) & (b_t > 0) & (b_k > 0) & (b_r > -1)

    def _require_args_scalar_or_1d_arrays_of_same_shape(self, *args: ArrayLike) -> tuple[np.ndarray | float, ...]:
        def _as_float(a: Any) -> float:  # noqa: ANN401
            return float(a)

        _args = [_as_float(a) if np.isscalar(a) else np.atleast_1d(a) for a in args]
        n = next((a.shape[0] for a in _args if isinstance(a, np.ndarray)), None)
        req_msg = "All inputs must be scalars or have the same shape"
        _require(all(a.ndim == 1 and a.shape[0] == n for a in _args if isinstance(a, np.ndarray)), req_msg)
        return tuple(_args)

    # fmt: off
    @property
    def s0(self) -> np.ndarray | float: return self._s0
    @property
    def vol(self) -> np.ndarray | float:return self._vol
    @property
    def q(self) -> np.ndarray | float: return self._q
    @property
    def t(self) -> np.ndarray | float: return self._t
    @property
    def k(self) -> np.ndarray | float: return self._k
    @property
    def r(self) -> np.ndarray | float: return self._r
    @property
    def valid_mask(self) -> np.ndarray | bool: return self._valid_mask
    # fmt: off


def _require(condition: bool, descr: str | None = None, *, msg: str = "Invalid argument") -> None:  # noqa: FBT001
    if condition:
        return
    if descr:
        msg = f"{msg}: {descr}"
    raise ValueError(msg)

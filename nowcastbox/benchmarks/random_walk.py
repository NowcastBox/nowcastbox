r"""Random-walk (no-change) benchmark.

The forecast of every future period is the last observed value of the target,

.. math::

    \hat y_{T+h} = y_T + h\,\hat\mu,

with :math:`\hat\mu = 0` (pure random walk) or, with ``drift=True``, the mean of the
observed first differences. For observed periods the benchmark returns the
one-step-ahead forecast :math:`y_{t-1} + \hat\mu`. The no-change forecast is the
classic naive benchmark of forecast evaluation (e.g. Diebold, 2017, ch. 10; Stock &
Watson, 2007).

References
----------
Diebold, F. X. (2017). *Forecasting in Economics, Business, Finance and Beyond*.
University of Pennsylvania.

Stock, J. H. & Watson, M. W. (2007). Why has U.S. inflation become harder to forecast?
*Journal of Money, Credit and Banking*, 39(s1), 3-33.
"""

from __future__ import annotations

import numpy as np

from nowcastbox.benchmarks._base import UnivariateBenchmark
from nowcastbox.core.exceptions import NowcastDataError

__all__ = ["RandomWalk"]


class RandomWalk(UnivariateBenchmark):
    """Random walk (no-change) forecast, optionally with drift.

    Parameters
    ----------
    drift : bool, default False
        Add the mean observed first difference per period of horizon.

    Attributes
    ----------
    drift_ : float
        Estimated drift (0 when ``drift=False``).

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.benchmarks import RandomWalk
    >>> idx = pd.period_range("2020Q1", periods=4, freq="Q")
    >>> frame = pd.DataFrame({"gdp": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    >>> RandomWalk().fit(frame, "gdp", frequency="Q").predict(["2021Q1", "2021Q2"]).tolist()
    [4.0, 4.0]
    >>> RandomWalk(drift=True).fit(frame, "gdp", frequency="Q").predict(["2021Q2"]).tolist()
    [6.0]
    """

    def __init__(self, drift: bool = False) -> None:
        self.drift = drift

    _drift: float = 0.0

    def _fit_history(self, values: np.ndarray) -> None:
        if not isinstance(self.drift, bool | np.bool_):
            raise ValueError(f"drift must be a bool, got {self.drift!r}.")
        self._drift = 0.0
        if self.drift:
            diffs = np.diff(values)
            diffs = diffs[np.isfinite(diffs)]
            if diffs.size == 0:
                raise NowcastDataError("The drift needs two consecutive observations.")
            self._drift = float(diffs.mean())

    def _predict_positions(self, positions: np.ndarray) -> np.ndarray:
        assert self._history is not None  # noqa: S101
        last = self._history.size - 1
        out = np.full(positions.size, np.nan)
        ahead = positions > last
        out[ahead] = self._history[last] + (positions[ahead] - last) * self._drift
        inside = (positions >= 1) & ~ahead
        out[inside] = self._history[positions[inside] - 1] + self._drift
        return out

    @property
    def drift_(self) -> float:
        """Estimated drift per period."""
        self._check_fitted()
        return self._drift

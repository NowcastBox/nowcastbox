"""Base class of the univariate benchmarks (AR, random walk, historical mean)."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
import pandas as pd

from nowcastbox.benchmarks._utils import last_valid, native_grid, ordinals
from nowcastbox.core.base import BaseBenchmark
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError

__all__ = ["UnivariateBenchmark"]


class UnivariateBenchmark(BaseBenchmark):
    r"""Benchmark using only the history of the target at its native frequency.

    Subclasses implement :meth:`_fit_history` (estimation on the observed history
    :math:`y_0, \dots, y_T`, NaN = missing) and :meth:`_predict_positions`
    (predictions for positions on the native grid; positions :math:`> T` are
    out-of-sample forecasts from origin :math:`T`).

    Attributes
    ----------
    history_ : pandas.Series
        Target history up to its last observation (native frequency).
    """

    _history: np.ndarray | None = None
    _start: pd.Period | None = None
    _index: pd.PeriodIndex | None = None

    @property
    def history_(self) -> pd.Series:
        """Target history used in the fit (native frequency, up to the last observation)."""
        self._check_fitted()
        assert self._history is not None  # noqa: S101
        assert self._index is not None  # noqa: S101
        return pd.Series(self._history, index=self._index, name=self.target_)

    def _fit(self, data: MixedFrequencyData, target: str) -> None:
        series = native_grid(data, target)
        values = series.to_numpy(dtype=float)
        last = last_valid(values)
        if last < 0:
            raise NowcastDataError(f"The target {target!r} has no observations.")
        index = series.index
        assert isinstance(index, pd.PeriodIndex)  # noqa: S101
        self._history = values[: last + 1]
        self._index = index[: last + 1]
        self._start = index[0]
        self._fit_history(self._history)

    def _predict(self, periods: pd.PeriodIndex) -> pd.Series:
        assert self._start is not None  # noqa: S101
        positions = np.asarray(ordinals(periods) - self._start.ordinal, dtype=np.int64)
        values = self._predict_positions(positions)
        return pd.Series(values, index=periods, dtype=float)

    @abstractmethod
    def _fit_history(self, values: np.ndarray) -> None:
        r"""Estimate on the observed history (implemented by subclasses).

        Parameters
        ----------
        values : numpy.ndarray
            History :math:`y_0, \dots, y_T` (last value observed; NaN = missing).
        """

    @abstractmethod
    def _predict_positions(self, positions: np.ndarray) -> np.ndarray:
        """Predictions for positions on the native grid (implemented by subclasses).

        Parameters
        ----------
        positions : numpy.ndarray
            Integer positions relative to the first period of the grid.

        Returns
        -------
        numpy.ndarray
            Predictions (NaN where none can be made).
        """

r"""Historical-mean benchmark.

Every period is forecast by the sample mean of the observed target,
:math:`\hat y_{T+h} = \bar y`, optionally over the last ``window`` observations only
(rolling mean). The unconditional mean is the benchmark used e.g. by Giannone,
Reichlin & Small (2008) and Campbell & Thompson (2008) for out-of-sample :math:`R^2`.

References
----------
Campbell, J. Y. & Thompson, S. B. (2008). Predicting excess stock returns out of
sample: Can anything beat the historical average? *Review of Financial Studies*,
21(4), 1509-1531.

Giannone, D., Reichlin, L. & Small, D. (2008). Nowcasting: The real-time informational
content of macroeconomic data. *Journal of Monetary Economics*, 55(4), 665-676.
"""

from __future__ import annotations

import numpy as np

from nowcastbox.benchmarks._base import UnivariateBenchmark
from nowcastbox.benchmarks._utils import check_int

__all__ = ["HistoricalMean"]


class HistoricalMean(UnivariateBenchmark):
    """Historical (optionally rolling) mean of the target.

    Parameters
    ----------
    window : int, optional
        Use only the last ``window`` observations (default: all).

    Attributes
    ----------
    mean_ : float
        Estimated mean (the forecast of every period).

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.benchmarks import HistoricalMean
    >>> idx = pd.period_range("2020Q1", periods=4, freq="Q")
    >>> frame = pd.DataFrame({"gdp": [1.0, 2.0, 3.0, 6.0]}, index=idx)
    >>> HistoricalMean().fit(frame, "gdp", frequency="Q").predict(["2021Q1"]).tolist()
    [3.0]
    >>> HistoricalMean(window=2).fit(frame, "gdp", frequency="Q").mean_
    4.5
    """

    def __init__(self, window: int | None = None) -> None:
        self.window = window

    _mean: float = float("nan")

    def _fit_history(self, values: np.ndarray) -> None:
        observed = values[np.isfinite(values)]
        if self.window is not None:
            observed = observed[-check_int("window", self.window, 1) :]
        self._mean = float(observed.mean())

    def _predict_positions(self, positions: np.ndarray) -> np.ndarray:
        return np.full(positions.size, self._mean)

    @property
    def mean_(self) -> float:
        """Estimated mean."""
        self._check_fitted()
        return self._mean

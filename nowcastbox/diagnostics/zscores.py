r"""Z-scores of the indicators: the "heatmap" of the current economic situation.

Following the conjunctural heatmap of Linzenich & Meunier (2024, ECB WP 3004, §3.4), each
(transformed) indicator is expressed as a z-score relative to its long-run mean and
standard deviation, so that the latest readings of very different series can be compared
on one colour scale. Monthly series are first smoothed with the five-month moving average
whose weights :math:`(1, 2, 3, 2, 1)/9` map monthly growth rates into quarterly growth
rates of the quarterly average (Mariano & Murasawa, 2003), which makes them comparable to
quarterly series and removes most of the month-to-month noise:

.. math::

    s_{i,t} = \tfrac19\,(x_{i,t} + 2x_{i,t-1} + 3x_{i,t-2} + 2x_{i,t-3} + x_{i,t-4}),
    \qquad
    z_{i,t} = \frac{s_{i,t} - \bar s_i}{\hat\sigma_i}.

The mean :math:`\bar s_i` and standard deviation :math:`\hat\sigma_i` are computed over the
whole sample available (``window=None``) or over a trailing rolling window; with
``as_of=`` the panel is first cut to the information set of that vintage
(:meth:`~nowcastbox.core.data.MixedFrequencyData.as_of`), so nothing released later
enters the statistics. Group z-scores are the mean of the z-scores of the group members.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from nowcastbox._logging import get_logger
from nowcastbox.core.data import (
    MixedFrequencyData,
    _series_groups,  # pyright: ignore[reportPrivateUsage]
    as_mixed_frequency_data,
)
from nowcastbox.core.exceptions import DataQualityWarning
from nowcastbox.core.frequency import AggregationType, Frequency, FrequencyLike, is_period_end
from nowcastbox.core.results import NowcastResults

__all__ = ["MM_SMOOTHING_WEIGHTS", "IndicatorZScores", "indicator_zscores", "smooth_series"]

logger = get_logger(__name__)

MM_SMOOTHING_WEIGHTS: tuple[float, ...] = tuple(
    float(w) for w in AggregationType.GROWTH_RATE.weights(3) / 9.0
)
"""Mariano-Murasawa five-month weights ``(1, 2, 3, 2, 1) / 9`` (most recent first)."""

GroupSpec = str | Mapping[str, str | Sequence[str]] | None
SmoothSpec = str | Sequence[float] | None


@dataclass(frozen=True, kw_only=True, eq=False)
class IndicatorZScores:
    """Z-scores of the indicators of a panel, by series and by group.

    Parameters
    ----------
    zscores : pandas.DataFrame
        Base grid x series z-scores (NaN where the smoothed series is not available).
    smoothed : pandas.DataFrame
        The (smoothed) series the z-scores are computed from.
    mean, std : pandas.DataFrame
        Mean and standard deviation used at each period (constant over time when
        ``window`` is None).
    groups : pandas.DataFrame or None
        Base grid x group mean z-scores (None without a grouping).
    membership : dict of str to tuple of str
        Series of each group (empty without a grouping).
    smooth_weights : tuple of float or None
        Smoothing weights of monthly series (most recent first); None if unsmoothed.
    window : int or None
        Rolling window (base periods) of the statistics; None = whole sample.
    as_of : pandas.Timestamp or None
        Information date of the panel.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> idx = pd.period_range("2020-01", periods=24, freq="M")
    >>> df = pd.DataFrame({"x": np.arange(24.0) % 5}, index=idx)
    >>> z = indicator_zscores(MixedFrequencyData(df, "M"), smooth=None)
    >>> bool(abs(z.zscores["x"].mean()) < 1e-12)
    True
    """

    zscores: pd.DataFrame
    smoothed: pd.DataFrame
    mean: pd.DataFrame
    std: pd.DataFrame
    groups: pd.DataFrame | None
    membership: dict[str, tuple[str, ...]]
    smooth_weights: tuple[float, ...] | None
    window: int | None
    as_of: pd.Timestamp | None

    @property
    def series(self) -> list[str]:
        """Names of the series."""
        return [str(c) for c in self.zscores.columns]

    def _frame(self, level: str) -> pd.DataFrame:
        if level == "series":
            return self.zscores
        if level == "group":
            if self.groups is None:
                raise ValueError("No grouping: call indicator_zscores(..., by=...).")
            return self.groups
        raise ValueError(f"level must be 'series' or 'group'; got {level!r}.")

    def table(
        self,
        level: str = "series",
        *,
        last: int | None = None,
        frequency: FrequencyLike | None = None,
    ) -> pd.DataFrame:
        """Heatmap table: rows = series (or groups), columns = periods.

        Parameters
        ----------
        level : {"series", "group"}, default "series"
            Rows of the table.
        last : int, optional
            Keep only the last ``last`` periods.
        frequency : Frequency or str, optional
            Keep only the last base period of each period of this (lower) frequency and
            label the columns with it, e.g. ``"Q"`` for a quarterly heatmap.

        Returns
        -------
        pandas.DataFrame
            Z-scores.

        Raises
        ------
        ValueError
            Unknown ``level``, ``level="group"`` without a grouping, or ``last < 1``.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2020-01", periods=12, freq="M")
        >>> df = pd.DataFrame({"x": np.arange(12.0)}, index=idx)
        >>> z = indicator_zscores(MixedFrequencyData(df, "M"), smooth=None)
        >>> z.table(frequency="Q").columns.astype(str).tolist()
        ['2020Q1', '2020Q2', '2020Q3', '2020Q4']
        """
        frame = self._frame(level)
        if frequency is not None:
            freq = Frequency.from_value(frequency)
            index = pd.PeriodIndex(frame.index)
            frame = frame.loc[is_period_end(index, freq)]
            frame.index = pd.PeriodIndex(frame.index).asfreq(freq.pandas_freq)
        if last is not None:
            if last < 1:
                raise ValueError(f"last must be >= 1 or None; got {last}.")
            frame = frame.iloc[-last:]
        return frame.T.copy()

    def latest(self, level: str = "series") -> pd.DataFrame:
        """Latest available z-score of each series (or group) and its period.

        Parameters
        ----------
        level : {"series", "group"}, default "series"
            Rows of the table.

        Returns
        -------
        pandas.DataFrame
            Columns ``period`` (base period of the last value; NaT if none) and
            ``zscore``.

        Raises
        ------
        ValueError
            Unknown ``level`` or ``level="group"`` without a grouping.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2020-01", periods=12, freq="M")
        >>> df = pd.DataFrame({"x": np.arange(12.0)}, index=idx)
        >>> z = indicator_zscores(MixedFrequencyData(df, "M"), smooth=None)
        >>> str(z.latest().loc["x", "period"])
        '2020-12'
        """
        frame = self._frame(level)
        periods: dict[str, Any] = {}
        values: dict[str, float] = {}
        for column in frame.columns:
            valid = frame[column].dropna()
            periods[column] = valid.index[-1] if len(valid) else pd.NaT
            values[column] = float(valid.iloc[-1]) if len(valid) else np.nan
        out = pd.DataFrame(
            {"period": pd.Series(periods, dtype=frame.index.dtype), "zscore": pd.Series(values)}
        )
        out.index.name = level
        return out

    def summary(self) -> str:
        """Formatted text summary of the latest z-scores.

        Returns
        -------
        str
            One line per series (and per group when there is a grouping).

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2020-01", periods=12, freq="M")
        >>> df = pd.DataFrame({"x": np.arange(12.0)}, index=idx)
        >>> print(indicator_zscores(MixedFrequencyData(df, "M"), smooth=None).summary())
        ... # doctest: +ELLIPSIS
        ==...
        Indicator z-scores (...)
        ...
        """
        width = 60
        stats = "whole sample" if self.window is None else f"rolling {self.window} periods"
        smooth = "unsmoothed" if self.smooth_weights is None else "smoothed"
        lines = ["=" * width, f"Indicator z-scores ({stats}, {smooth})", "=" * width]
        levels = ["series"] if self.groups is None else ["group", "series"]
        for level in levels:
            lines.append(f"  {level:<30}{'period':>12}{'z-score':>12}")
            for name, row in self.latest(level).iterrows():
                lines.append(f"  {name!s:<30}{row['period']!s:>12}{row['zscore']:>12.2f}")
            lines.append("-" * width)
        lines[-1] = "=" * width
        return "\n".join(lines)

    def plot(self, kind: str = "heatmap", **kwargs: Any) -> Any:
        """Plot the z-scores (``kind="heatmap"``).

        Parameters
        ----------
        kind : {"heatmap"}, default "heatmap"
            Plot kind.
        **kwargs
            Passed to :func:`nowcastbox.visualization.plot_indicator_heatmap`
            (``level``, ``last``, ``frequency``, ``backend``, ``theme``...).

        Returns
        -------
        plotly.graph_objects.Figure or matplotlib.figure.Figure
            The figure.

        Raises
        ------
        ValueError
            Unknown ``kind``.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2020-01", periods=12, freq="M")
        >>> df = pd.DataFrame({"x": np.arange(12.0)}, index=idx)
        >>> z = indicator_zscores(MixedFrequencyData(df, "M"))
        >>> z.plot(last=6).data[0].type
        'heatmap'
        """
        if kind != "heatmap":
            raise ValueError(f"Unknown plot kind {kind!r}; available: ['heatmap'].")
        from nowcastbox.visualization.heatmap import plot_indicator_heatmap

        return plot_indicator_heatmap(self, **kwargs)


# ====================================================================== helpers
def _smoothing_weights(smooth: SmoothSpec) -> tuple[float, ...] | None:
    if smooth is None:
        return None
    if isinstance(smooth, str):
        if smooth.lower() in ("mm", "mariano_murasawa"):
            return MM_SMOOTHING_WEIGHTS
        if smooth.lower() == "none":
            return None
        raise ValueError(f"smooth must be 'mm', 'none', None or a weight sequence; got {smooth!r}.")
    weights = np.asarray(list(smooth), dtype=float)
    total = float(weights.sum()) if weights.size else 0.0
    if weights.ndim != 1 or not np.isfinite(weights).all() or total <= 0:
        raise ValueError("Smoothing weights must be finite with a positive sum.")
    return tuple(float(w) for w in weights / total)


def smooth_series(values: Any, weights: Sequence[float] = MM_SMOOTHING_WEIGHTS) -> np.ndarray:
    r"""Weighted moving average :math:`s_t = \sum_k w_k x_{t-k}` (most recent weight first).

    The first ``len(weights) - 1`` values, and every value whose window contains a NaN,
    are NaN.

    Parameters
    ----------
    values : array-like
        One-dimensional series.
    weights : sequence of float, default :data:`MM_SMOOTHING_WEIGHTS`
        Weights :math:`(w_0, w_1, \dots)` applied to :math:`(x_t, x_{t-1}, \dots)`.

    Returns
    -------
    numpy.ndarray
        Smoothed series, same length as ``values``.

    Examples
    --------
    >>> smooth_series([9.0, 9.0, 9.0, 9.0, 9.0, 18.0]).tolist()
    [nan, nan, nan, nan, 9.0, 10.0]
    """
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    out = np.full(x.shape, np.nan)
    if x.size >= w.size:
        windows = sliding_window_view(x, w.size)  # (x_{t-L+1}, ..., x_t)
        out[w.size - 1 :] = windows @ w[::-1]
    return out


def _smoothed_frame(panel: MixedFrequencyData, weights: tuple[float, ...] | None) -> pd.DataFrame:
    frame = panel.to_frame()
    if weights is None:
        return frame
    slots = panel.slot_mask()
    for name in panel.columns_with_frequency(Frequency.MONTHLY):
        rows = slots[name].to_numpy()
        frame.loc[rows, name] = smooth_series(frame.loc[rows, name].to_numpy(), weights)
    return frame


def _statistics(
    smoothed: pd.DataFrame, window: int | None, ddof: int, min_periods: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if window is None:
        counts = smoothed.notna().sum()
        mean = smoothed.mean().where(counts >= min_periods)
        std = smoothed.std(ddof=ddof).where(counts >= min_periods)
        full_mean = pd.DataFrame([mean] * len(smoothed), index=smoothed.index)
        full_std = pd.DataFrame([std] * len(smoothed), index=smoothed.index)
        return full_mean, full_std
    rolling = smoothed.rolling(window, min_periods=min_periods)
    return rolling.mean(), rolling.std(ddof=ddof)


def _warn_degenerate(std: pd.DataFrame) -> pd.DataFrame:
    valid = std.where(std > 0)
    bad = [str(c) for c in std.columns if valid[c].isna().all()]
    if bad:
        warnings.warn(
            f"No z-scores for series {bad}: too few observations or zero variance.",
            DataQualityWarning,
            stacklevel=3,
        )
    return valid


def _resolve_panel(
    data: object, series: Sequence[str] | None, as_of: pd.Timestamp | str | None
) -> MixedFrequencyData:
    if isinstance(data, NowcastResults):
        if data.data is None:
            raise ValueError("Results do not carry their estimation data.")
        panel = data.data
        if series is None:
            series = [c for c in panel.columns if c != data.target] or None
    else:
        panel = as_mixed_frequency_data(data)  # type: ignore[arg-type]
    if as_of is not None:
        panel = panel.as_of(as_of)
    if series is not None:
        unknown = [s for s in series if s not in panel.columns]
        if unknown:
            raise ValueError(f"Unknown series {unknown}.")
        panel = panel.select(list(series))
    return panel


def _check_options(window: int | None, ddof: int, min_periods: int | None) -> int:
    if window is not None and window < 2:
        raise ValueError(f"window must be >= 2 or None; got {window}.")
    if ddof < 0:
        raise ValueError(f"ddof must be >= 0; got {ddof}.")
    needed = max(2, ddof + 1) if min_periods is None else min_periods
    if needed <= ddof or needed < 2:
        raise ValueError(f"min_periods must be >= 2 and > ddof; got {min_periods}.")
    if window is not None and needed > window:
        raise ValueError(f"min_periods ({needed}) cannot exceed window ({window}).")
    return needed


# ====================================================================== main
def indicator_zscores(
    data: MixedFrequencyData | pd.DataFrame | NowcastResults,
    *,
    smooth: SmoothSpec = "mm",
    window: int | None = None,
    by: GroupSpec = None,
    as_of: pd.Timestamp | str | None = None,
    series: Sequence[str] | None = None,
    ddof: int = 1,
    min_periods: int | None = None,
) -> IndicatorZScores:
    """Z-scores of the (transformed) indicators relative to their long-run moments.

    Parameters
    ----------
    data : MixedFrequencyData, pandas.DataFrame or NowcastResults
        Panel of transformed series (e.g. growth rates), or fitted results (their
        estimation panel without the target, unless ``series`` is given).
    smooth : {"mm", "none"}, sequence of float or None, default "mm"
        Smoothing of monthly series: the Mariano-Murasawa weights
        :data:`MM_SMOOTHING_WEIGHTS`, none, or custom weights (most recent first,
        normalised to sum to one). Series of other frequencies are not smoothed.
    window : int, optional
        Trailing rolling window (base periods) of the mean and standard deviation;
        None (default) uses the whole sample available.
    by : {"category", "block", "frequency"}, mapping or None, optional
        Grouping for group z-scores (mean of the members' z-scores): the series
        categories, factor blocks, frequencies or a mapping ``{series: group or
        [groups]}`` (series left out go to ``"unassigned"``).
    as_of : Timestamp or str, optional
        Information date: the panel is first cut with
        :meth:`~nowcastbox.core.data.MixedFrequencyData.as_of` (needs release delays),
        so only data released by then enter the smoothing and the statistics.
    series : sequence of str, optional
        Subset of series.
    ddof : int, default 1
        Delta degrees of freedom of the standard deviation.
    min_periods : int, optional
        Minimum number of smoothed observations behind a mean/standard deviation
        (default ``max(2, ddof + 1)``).

    Returns
    -------
    IndicatorZScores
        Z-scores on the base grid, trimmed after the last period with any value.

    Raises
    ------
    ValueError
        On invalid options, unknown series or groupings, or results without data.
    NowcastDataError
        If ``as_of`` is given and release delays are missing.

    Warns
    -----
    DataQualityWarning
        For series without any z-score (too few observations or zero variance).

    Examples
    --------
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> data = simulate_two_step_example(random_state=0)
    >>> z = indicator_zscores(data, by="frequency")
    >>> list(z.membership)
    ['monthly', 'quarterly']
    >>> bool(abs(z.zscores.mean()).max() < 1e-10)
    True
    """
    needed = _check_options(window, ddof, min_periods)
    weights = _smoothing_weights(smooth)
    panel = _resolve_panel(data, series, as_of)
    smoothed = _smoothed_frame(panel, weights)
    mean, std = _statistics(smoothed, window, ddof, needed)
    zscores = (smoothed - mean) / _warn_degenerate(std)
    keep = zscores.notna().any(axis=1).to_numpy()
    end = int(np.flatnonzero(keep)[-1]) + 1 if keep.any() else 0
    zscores, smoothed, mean, std = (f.iloc[:end] for f in (zscores, smoothed, mean, std))
    groups, membership = None, {}
    if by is not None:
        members = _series_groups(panel.metadata, by)
        membership = {g: tuple(cols) for g, cols in members.items()}
        groups = pd.DataFrame(
            {g: zscores[list(cols)].mean(axis=1) for g, cols in members.items()},
            index=zscores.index,
        )
    logger.debug("indicator z-scores: %d series, %d periods", *zscores.shape[::-1])
    return IndicatorZScores(
        zscores=zscores,
        smoothed=smoothed,
        mean=mean,
        std=std,
        groups=groups,
        membership=membership,
        smooth_weights=weights,
        window=window,
        as_of=None if as_of is None else pd.Timestamp(as_of),
    )

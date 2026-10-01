"""Private helpers shared by the benchmark forecasters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import Frequency

__all__ = [
    "check_int",
    "check_ragged_edge",
    "last_valid",
    "lstsq",
    "native_grid",
    "ordinals",
    "period_positions",
    "resolve_predictors",
]

RAGGED_EDGE_METHODS = ("realign", "ar")


def check_int(name: str, value: object, minimum: int) -> int:
    """Validate an integer hyper-parameter (``ValueError`` otherwise).

    Parameters
    ----------
    name : str
        Parameter name (for the message).
    value : object
        Value to check.
    minimum : int
        Smallest admissible value.

    Returns
    -------
    int
        The value as a Python int.

    Raises
    ------
    ValueError
        If ``value`` is not an integer ``>= minimum``.

    Examples
    --------
    >>> check_int("p", 2, 0)
    2
    """
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


def check_ragged_edge(value: object) -> str:
    """Validate the ``ragged_edge`` option of the MIDAS benchmarks.

    Parameters
    ----------
    value : object
        ``"realign"`` or ``"ar"``.

    Returns
    -------
    str
        The validated value.

    Raises
    ------
    ValueError
        If the value is not admissible.

    Examples
    --------
    >>> check_ragged_edge("ar")
    'ar'
    """
    if value not in RAGGED_EDGE_METHODS:
        raise ValueError(f"ragged_edge must be one of {RAGGED_EDGE_METHODS}, got {value!r}.")
    return str(value)


def last_valid(values: np.ndarray) -> int:
    """Position of the last finite value (``-1`` when there is none).

    Parameters
    ----------
    values : numpy.ndarray
        1-D array.

    Returns
    -------
    int
        Position.

    Examples
    --------
    >>> last_valid(np.array([1.0, np.nan, 2.0, np.nan]))
    2
    """
    idx = np.flatnonzero(np.isfinite(values))
    return int(idx[-1]) if idx.size else -1


def native_grid(data: MixedFrequencyData, column: str) -> pd.Series:
    """Series ``column`` on its contiguous native period grid.

    Parameters
    ----------
    data : MixedFrequencyData
        Panel.
    column : str
        Series name.

    Returns
    -------
    pandas.Series
        Values indexed by a contiguous :class:`pandas.PeriodIndex` of the native
        frequency (NaN = not observed).

    Examples
    --------
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> mfd = MixedFrequencyData(
    ...     pd.DataFrame({"q": [None, None, 1.0, None, None, 2.0]}, index=idx), "Q"
    ... )
    >>> native_grid(mfd, "q").index.astype(str).tolist()
    ['2020Q1', '2020Q2']
    """
    series = data.to_native(column)
    index = series.index
    assert isinstance(index, pd.PeriodIndex)  # noqa: S101
    grid = pd.period_range(index[0], index[-1], freq=index.freq)
    return series.reindex(grid).astype(float)


def period_positions(
    grid_start: pd.Period, frequency: Frequency, periods: pd.PeriodIndex
) -> np.ndarray:
    r"""Positions, on a native grid starting at ``grid_start``, of the last sub-period.

    For each target period :math:`t` returns :math:`\tau(t)`, the position of the
    last period of frequency ``frequency`` contained in :math:`t` (e.g. the third
    month of a quarter), counted from ``grid_start``.

    Parameters
    ----------
    grid_start : pandas.Period
        First period of the native grid.
    frequency : Frequency
        Frequency of the grid.
    periods : pandas.PeriodIndex
        Target periods (equal or lower frequency).

    Returns
    -------
    numpy.ndarray
        Integer positions (may be negative or beyond the grid).

    Examples
    --------
    >>> q = pd.period_range("2020Q1", periods=2, freq="Q")
    >>> period_positions(pd.Period("2020-01", "M"), Frequency.MONTHLY, q).tolist()
    [2, 5]
    """
    ends = periods.asfreq(frequency.pandas_freq, how="E")
    return ordinals(ends) - grid_start.ordinal


def ordinals(index: pd.PeriodIndex) -> np.ndarray:
    """Integer ordinals of a :class:`pandas.PeriodIndex`.

    Parameters
    ----------
    index : pandas.PeriodIndex
        Periods.

    Returns
    -------
    numpy.ndarray
        ``int64`` ordinals (``Period.ordinal`` of each element).

    Examples
    --------
    >>> ordinals(pd.period_range("1970-01", periods=2, freq="M")).tolist()
    [0, 1]
    """
    return np.asarray(index.asi8, dtype=np.int64)  # pyright: ignore[reportAttributeAccessIssue]


def resolve_predictors(
    data: MixedFrequencyData, target: str, predictors: Sequence[str] | None
) -> list[str]:
    """Validate the predictors of a regression benchmark.

    Parameters
    ----------
    data : MixedFrequencyData
        Panel.
    target : str
        Target series.
    predictors : sequence of str, optional
        Predictor names; default: every other series.

    Returns
    -------
    list of str
        Predictors (in the order given or of the panel).

    Raises
    ------
    NowcastDataError
        If a predictor is unknown, equals the target, has a lower frequency than the
        target or no observation, or if there is no predictor.

    Examples
    --------
    >>> idx = pd.period_range("2020-01", periods=3, freq="M")
    >>> df = pd.DataFrame({"x": [1.0, 2, 3], "y": [None, None, 1.0]}, index=idx)
    >>> resolve_predictors(MixedFrequencyData(df, {"x": "M", "y": "Q"}), "y", None)
    ['x']
    """
    if predictors is None:
        names = [c for c in data.columns if c != target]
    else:
        names = [predictors] if isinstance(predictors, str) else list(predictors)
    if not names:
        raise NowcastDataError("At least one predictor is required.")
    target_freq = data.metadata[target].frequency
    for name in names:
        if name not in data or name == target:
            raise NowcastDataError(f"Invalid predictor {name!r} (unknown or the target).")
        freq = data.metadata[name].frequency
        if freq.is_lower_than(target_freq):
            raise NowcastDataError(
                f"Predictor {name!r} ({freq.label}) has a lower frequency than the target "
                f"({target_freq.label})."
            )
        if not data[name].notna().any():
            raise NowcastDataError(f"Predictor {name!r} has no observations.")
    return names


def lstsq(design: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares coefficients and residual sum of squares.

    Parameters
    ----------
    design : numpy.ndarray
        Regressor matrix ``(n, k)``.
    y : numpy.ndarray
        Dependent variable ``(n,)``.

    Returns
    -------
    coef : numpy.ndarray
        Coefficients ``(k,)``.
    ssr : float
        Residual sum of squares.

    Examples
    --------
    >>> X = np.column_stack([np.ones(4), np.arange(4.0)])
    >>> coef, ssr = lstsq(X, 1.0 + 2.0 * np.arange(4.0))
    >>> np.round(coef, 10).tolist(), round(ssr, 10)
    ([1.0, 2.0], 0.0)
    """
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return coef, float(resid @ resid)

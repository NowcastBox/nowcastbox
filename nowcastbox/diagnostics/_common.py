"""Shared helpers of :mod:`nowcastbox.diagnostics` (internal).

Alignment of panels and factors, aggregation of factors for lower-frequency series,
principal-component factors for raw panels and multiple-testing adjustments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from nowcastbox.core.data import FrequencySpec, MixedFrequencyData, as_mixed_frequency_data
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import AggregationType, is_fixed_ratio, is_period_end
from nowcastbox.preprocessing.aggregation import aggregation_weights, calendar_weight_matrix

FloatArray = NDArray[np.float64]

PValueCorrection = Literal["holm", "bonferroni", "fdr_bh", "none"]
"""Multiple-testing corrections accepted by the diagnostics."""

CORRECTIONS: tuple[str, ...] = ("holm", "bonferroni", "fdr_bh", "none")


def as_panel(
    data: MixedFrequencyData | pd.DataFrame, frequency: FrequencySpec | None = None
) -> MixedFrequencyData:
    """Coerce ``data`` to :class:`MixedFrequencyData` (frequencies inferred if omitted).

    Parameters
    ----------
    data : MixedFrequencyData or pandas.DataFrame
        Panel on a base-frequency :class:`pandas.PeriodIndex`.
    frequency : frequency specification, optional
        Per-series frequencies of a DataFrame.

    Returns
    -------
    MixedFrequencyData
        The panel.
    """
    return as_mixed_frequency_data(data, frequency)


def check_alpha(alpha: float) -> float:
    """Validate a significance level in ``(0, 1)``.

    Parameters
    ----------
    alpha : float
        Significance level.

    Returns
    -------
    float
        ``alpha`` as a float.

    Raises
    ------
    ValueError
        If ``alpha`` is not in ``(0, 1)``.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, int | float | np.floating):
        raise ValueError(f"alpha must be a number in (0, 1), got {alpha!r}.")
    value = float(alpha)
    if not 0.0 < value < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}.")
    return value


def check_int(value: object, name: str, minimum: int) -> int:
    """Validate an integer hyper-parameter ``>= minimum``.

    Parameters
    ----------
    value : object
        Value to check.
    name : str
        Parameter name (for the error message).
    minimum : int
        Smallest admissible value.

    Returns
    -------
    int
        The value.

    Raises
    ------
    ValueError
        If ``value`` is not an integer ``>= minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


def check_correction(correction: str) -> str:
    """Validate a multiple-testing correction name.

    Parameters
    ----------
    correction : str
        One of :data:`CORRECTIONS`.

    Returns
    -------
    str
        The name.

    Raises
    ------
    ValueError
        If the name is unknown.
    """
    if correction not in CORRECTIONS:
        raise ValueError(f"correction must be one of {CORRECTIONS}, got {correction!r}.")
    return correction


def adjust_pvalues(pvalues: Any, method: str = "holm") -> FloatArray:
    """Adjust p-values for multiple testing.

    Parameters
    ----------
    pvalues : array_like
        Raw p-values; ``NaN`` entries are ignored (and stay ``NaN``).
    method : {"holm", "bonferroni", "fdr_bh", "none"}, default "holm"
        Holm (1979) step-down, Bonferroni, Benjamini & Hochberg (1995) false discovery
        rate, or no adjustment.

    Returns
    -------
    numpy.ndarray
        Adjusted p-values (capped at 1).

    Raises
    ------
    ValueError
        If ``method`` is unknown.

    Examples
    --------
    >>> from nowcastbox.diagnostics import adjust_pvalues
    >>> adjust_pvalues([0.01, 0.04, 0.03], "holm").round(3).tolist()
    [0.03, 0.06, 0.06]
    """
    check_correction(method)
    p = np.asarray(pvalues, dtype=np.float64).copy()
    ok = np.isfinite(p)
    m = int(ok.sum())
    if method == "none" or m == 0:
        return p
    q = p[ok]
    order = np.argsort(q, kind="stable")
    ranked = q[order]
    if method == "bonferroni":
        adj_sorted = ranked * m
    elif method == "holm":
        adj_sorted = np.maximum.accumulate(ranked * (m - np.arange(m)))
    else:  # fdr_bh
        scaled = ranked * m / np.arange(1, m + 1)
        adj_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)
    p[ok] = adj
    return p


def series_weights(
    data: MixedFrequencyData, series: Sequence[str], weights: Mapping[str, Any] | None = None
) -> dict[str, FloatArray]:
    """Aggregation weights linking each series to the base-frequency factors.

    Parameters
    ----------
    data : MixedFrequencyData
        Panel (frequencies and ``aggregation`` metadata).
    series : sequence of str
        Series names.
    weights : mapping, optional
        Explicit weights (most recent period first) overriding the defaults: a vector,
        or per-period weights (a 2-D array with one row per base period of ``data``, or
        a DataFrame indexed by base periods, as ``params["aggregation_weight_paths"]``
        of calendar-aggregated EM models).

    Returns
    -------
    dict of str to numpy.ndarray
        ``[1.0]`` for base-frequency series; Mariano-Murasawa weights (or the series'
        ``aggregation`` metadata) for lower-frequency series. For calendar pairs
        (weekly or daily base grid with monthly/quarterly series) the weights depend on
        the period: an array of shape ``(n_periods, L)`` (see
        :func:`~nowcastbox.preprocessing.aggregation.calendar_weight_matrix`).
    """
    given = dict(weights or {})
    base = data.base_frequency
    out: dict[str, FloatArray] = {}
    for name in series:
        if name in given:
            out[name] = _given_weights(given[name], name, data.index)
            continue
        meta = data.metadata[name]
        agg = meta.aggregation or AggregationType.GROWTH_RATE
        if meta.frequency == base:
            out[name] = np.ones(1)
        elif not is_fixed_ratio(base, meta.frequency):
            out[name] = np.asarray(
                calendar_weight_matrix(data.index, meta.frequency, agg), dtype=np.float64
            )
        else:
            out[name] = np.asarray(
                aggregation_weights(base, meta.frequency, agg, normalize=True), dtype=np.float64
            )
    return out


def _given_weights(value: Any, name: str, index: pd.PeriodIndex) -> FloatArray:
    """Validate explicit weights: a vector or one row per base period of ``index``."""
    if isinstance(value, pd.DataFrame):
        value = value.reindex(index).fillna(0.0).to_numpy(dtype=np.float64)
    w = np.asarray(value, dtype=np.float64)
    if w.ndim == 2 and w.shape[0] == len(index) and w.shape[1] > 0:
        if not np.isfinite(w).all():
            raise ValueError(f"weights of {name!r} must be finite.")
        return w
    w = w.ravel()
    if w.size == 0 or not np.isfinite(w).all():
        raise ValueError(f"weights of {name!r} must be a non-empty finite vector.")
    return w


def aggregate_factors(factors: FloatArray, weights: FloatArray) -> FloatArray:
    r"""Weighted sum :math:`\sum_l w_l F_{t-l}` of current and lagged factors.

    Parameters
    ----------
    factors : numpy.ndarray, shape (n_periods, r)
        Factors on the base grid.
    weights : numpy.ndarray
        Weights, most recent period first: a vector, or per-period weights of shape
        ``(n_periods, L)`` (calendar aggregation; zero weights are skipped).

    Returns
    -------
    numpy.ndarray, shape (n_periods, r)
        Aggregated factors (``NaN`` where a lag with a non-zero weight precedes the
        sample, e.g. in the first ``len(weights) - 1`` periods).
    """
    n, r = factors.shape
    w = np.asarray(weights, dtype=np.float64)
    per_period = w.ndim == 2
    out = np.zeros((n, r))
    for lag in range(w.shape[-1]):
        shifted = np.full((n, r), np.nan)
        shifted[lag:] = factors[: n - lag] if lag else factors
        if per_period:
            wl = w[:, lag : lag + 1]
            out = out + np.where(wl == 0.0, 0.0, wl * shifted)
        else:
            out = out + w[lag] * shifted
    return out


def align_factors(factors: pd.DataFrame, index: pd.PeriodIndex) -> pd.DataFrame:
    """Reindex factors to the panel grid.

    Parameters
    ----------
    factors : pandas.DataFrame
        Factors on a :class:`pandas.PeriodIndex` of the base frequency.
    index : pandas.PeriodIndex
        Panel grid.

    Returns
    -------
    pandas.DataFrame
        Factors on ``index`` (``NaN`` where unavailable).

    Raises
    ------
    NowcastDataError
        If the factors are not a non-empty frame on a compatible PeriodIndex.
    """
    if not isinstance(factors, pd.DataFrame) or factors.shape[1] == 0:
        raise NowcastDataError("factors must be a non-empty pandas DataFrame.")
    if not isinstance(factors.index, pd.PeriodIndex):
        raise NowcastDataError("factors must be indexed by a pandas PeriodIndex.")
    if factors.index.freqstr != index.freqstr:
        raise NowcastDataError(
            f"factors have frequency {factors.index.freqstr!r}, the panel {index.freqstr!r}."
        )
    aligned = factors.reindex(index).astype(np.float64)
    if not np.isfinite(aligned.to_numpy()).any():
        raise NowcastDataError("The factors do not overlap the panel.")
    return aligned


def standardized(values: FloatArray) -> FloatArray:
    """Z-score of the finite entries of a vector (``NaN`` kept).

    Parameters
    ----------
    values : numpy.ndarray
        1-D array.

    Returns
    -------
    numpy.ndarray
        Standardised copy; a constant vector is only centred.
    """
    x = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return x.copy()
    mean = float(x[ok].mean())
    std = float(x[ok].std(ddof=1))
    return (x - mean) / (std if std > 0 else 1.0)


def series_design(
    data: MixedFrequencyData,
    factors: pd.DataFrame,
    name: str,
    weights: FloatArray,
) -> tuple[pd.PeriodIndex, FloatArray, FloatArray]:
    """Observed values of one series and the matching (aggregated) factors.

    Parameters
    ----------
    data : MixedFrequencyData
        Panel.
    factors : pandas.DataFrame
        Factors aligned on ``data.index`` (see :func:`align_factors`).
    name : str
        Series.
    weights : numpy.ndarray
        Aggregation weights of the series.

    Returns
    -------
    index : pandas.PeriodIndex
        Base periods used (storage slots with a value and finite regressors), in order.
    y : numpy.ndarray
        Standardised observations.
    F : numpy.ndarray, shape (n_obs, r)
        Regressors.
    """
    agg = aggregate_factors(factors.to_numpy(dtype=np.float64), weights)
    y = data.to_frame()[name].to_numpy(dtype=np.float64)
    slots = is_period_end(data.index, data.metadata[name].frequency)
    keep = slots & np.isfinite(y) & np.isfinite(agg).all(axis=1)
    return data.index[keep], standardized(y[keep]), agg[keep]


def to_native(values: pd.Series, data: MixedFrequencyData) -> pd.Series:
    """Restrict a base-grid series to its storage slots and use the native index.

    Parameters
    ----------
    values : pandas.Series
        Series on ``data.index`` named after a panel column.
    data : MixedFrequencyData
        Panel (frequency of the series).

    Returns
    -------
    pandas.Series
        Values on the native PeriodIndex.
    """
    freq = data.metadata[str(values.name)].frequency
    slots = is_period_end(data.index, freq)
    out = values.reindex(data.index)[slots].copy()
    out.index = data.index[slots].asfreq(freq.pandas_freq)
    return out


def pca_factors(data: MixedFrequencyData, n_factors: int) -> pd.DataFrame:
    """Principal-component factors of the balanced base-frequency panel.

    Parameters
    ----------
    data : MixedFrequencyData
        Panel; only base-frequency series enter.
    n_factors : int
        Number of factors.

    Returns
    -------
    pandas.DataFrame
        Columns ``f1..fr`` on ``data.index`` (``NaN`` on periods with missing values).

    Raises
    ------
    NowcastDataError
        If there are fewer base-frequency series or balanced periods than needed.
    """
    r = check_int(n_factors, "n_factors", 1)
    cols = data.columns_with_frequency(data.base_frequency)
    frame = data.to_frame()[cols]
    z = frame.apply(lambda s: pd.Series(standardized(s.to_numpy(float)), index=s.index))
    values = z.to_numpy(dtype=np.float64)
    complete = np.isfinite(values).all(axis=1)
    if len(cols) < r or complete.sum() <= r:
        raise NowcastDataError(
            f"Principal components need at least n_factors={r} base-frequency series and "
            f"more than {r} balanced periods; got {len(cols)} series and "
            f"{int(complete.sum())} periods."
        )
    x = values[complete]
    _, vecs = np.linalg.eigh(x.T @ x / x.shape[0])
    v = vecs[:, ::-1][:, :r]
    v = v * np.where(v.sum(axis=0) < 0, -1.0, 1.0)
    out = np.full((len(frame), r), np.nan)
    out[complete] = x @ v
    return pd.DataFrame(out, index=data.index, columns=[f"f{k + 1}" for k in range(r)])


def factor_block_map(
    factor_names: Sequence[str], block_names: Sequence[str]
) -> dict[str, list[str]]:
    """Group factor columns by block from ``"<block>_f<k>"`` names.

    Parameters
    ----------
    factor_names : sequence of str
        Factor columns.
    block_names : sequence of str
        Known blocks.

    Returns
    -------
    dict of str to list of str
        Factors of each block; factors that match no block go to ``"all"`` (every
        factor goes to ``"all"`` when none matches).
    """
    out: dict[str, list[str]] = {}
    for name in factor_names:
        block = next((b for b in block_names if name.startswith(f"{b}_f")), "all")
        out.setdefault(block, []).append(name)
    return out

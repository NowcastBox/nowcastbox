"""Missing values: ragged-edge masks and interior gap filling.

In a nowcasting panel three kinds of missing values coexist (Giannone, Reichlin &
Small, 2008; Bańbura & Modugno, 2014):

* **leading** - before the first observation of a series (series starting later);
* **interior** - gaps between the first and the last observation;
* **ragged edge** - after the last observation, because of publication lags.

The ragged edge carries the information structure that a nowcasting model exploits
through the Kalman filter, so it is **never filled unless explicitly requested**.
Interior gaps can be filled by a cubic spline (as in the initialisation of the EM
algorithm of Bańbura & Modugno, 2014), by linear interpolation or by a centred moving
median of neighbouring observations.

All operations run on each series' native grid: the "neighbours" of a quarterly
value are the adjacent quarters, never the empty months between them.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from nowcastbox._logging import get_logger
from nowcastbox.core.data import FrequencySpec
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.preprocessing._utils import (
    NativeView,
    as_panel,
    map_native,
    return_like,
)

__all__ = [
    "EdgeMethod",
    "FillMethod",
    "fill_missing",
    "interior_missing_mask",
    "leading_missing_mask",
    "missing_proportion",
    "ragged_edge_mask",
]

logger = get_logger(__name__)

FillMethod = Literal["spline", "linear", "moving_median"]
"""Methods to fill interior gaps."""

EdgeMethod = Literal["median", "last", "mean"]
"""Methods to fill the edges (only when explicitly requested)."""

_KIND = Literal["leading", "interior", "ragged"]


def _kind_mask(x: np.ndarray, kind: _KIND) -> np.ndarray:
    missing = np.isnan(x)
    obs = np.flatnonzero(~missing)
    pos = np.arange(len(x))
    if obs.size == 0:
        # no observation at all: everything belongs to the ragged edge
        return missing if kind == "ragged" else np.zeros(len(x), dtype=bool)
    if kind == "leading":
        return pos < obs[0]
    if kind == "ragged":
        return pos > obs[-1]
    return missing & (pos > obs[0]) & (pos < obs[-1])


def _mask(data: Any, kind: _KIND, frequency: FrequencySpec | None) -> pd.Series | pd.DataFrame:
    if isinstance(data, pd.Series):
        view = NativeView(data, frequency)  # type: ignore[arg-type]
        m = _kind_mask(view.native.to_numpy(float), kind)
        return view.to_original(m.astype(float)).fillna(0.0).astype(bool)
    panel, _ = as_panel(data, frequency)
    frame = map_native(panel, lambda _n, s, _f: _kind_mask(s.to_numpy(float), kind))
    return frame.fillna(0.0).astype(bool)


def ragged_edge_mask(
    data: Any, *, frequency: FrequencySpec | None = None
) -> pd.Series | pd.DataFrame:
    """Flag the storage slots after the last observation of each series.

    Parameters
    ----------
    data : pandas.Series, pandas.DataFrame or MixedFrequencyData
        Data indexed by a :class:`pandas.PeriodIndex` (base or native grid).
    frequency : frequency or frequency specification, optional
        Native frequency of a Series or per-series frequencies of a DataFrame
        (inferred when omitted).

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Boolean mask on the input index. Series without observations are flagged
        on every slot. Equivalent to
        :meth:`~nowcastbox.core.data.MixedFrequencyData.ragged_edge_mask`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.preprocessing.missing import ragged_edge_mask
    >>> idx = pd.period_range("2020-01", periods=4, freq="M")
    >>> ragged_edge_mask(pd.Series([1.0, np.nan, 2.0, np.nan], index=idx), frequency="M").tolist()
    [False, False, False, True]
    """
    return _mask(data, "ragged", frequency)


def leading_missing_mask(
    data: Any, *, frequency: FrequencySpec | None = None
) -> pd.Series | pd.DataFrame:
    """Flag the storage slots before the first observation of each series.

    Parameters
    ----------
    data : pandas.Series, pandas.DataFrame or MixedFrequencyData
        Data indexed by a :class:`pandas.PeriodIndex` (base or native grid).
    frequency : frequency or frequency specification, optional
        Native frequency of a Series or per-series frequencies of a DataFrame.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Boolean mask on the input index.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.preprocessing.missing import leading_missing_mask
    >>> idx = pd.period_range("2020-01", periods=3, freq="M")
    >>> leading_missing_mask(pd.Series([np.nan, 1.0, 2.0], index=idx), frequency="M").tolist()
    [True, False, False]
    """
    return _mask(data, "leading", frequency)


def interior_missing_mask(
    data: Any, *, frequency: FrequencySpec | None = None
) -> pd.Series | pd.DataFrame:
    """Flag missing storage slots strictly between the first and last observations.

    Parameters
    ----------
    data : pandas.Series, pandas.DataFrame or MixedFrequencyData
        Data indexed by a :class:`pandas.PeriodIndex` (base or native grid).
    frequency : frequency or frequency specification, optional
        Native frequency of a Series or per-series frequencies of a DataFrame.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Boolean mask on the input index.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.preprocessing.missing import interior_missing_mask
    >>> idx = pd.period_range("2020-01", periods=9, freq="M")
    >>> q = pd.Series([np.nan] * 9, index=idx)
    >>> q.iloc[[2, 8]] = [1.0, 3.0]
    >>> interior_missing_mask(q, frequency="Q").loc[lambda m: m].index.astype(str).tolist()
    ['2020-06']
    """
    return _mask(data, "interior", frequency)


def missing_proportion(data: Any, *, frequency: FrequencySpec | None = None) -> pd.Series | float:
    """Proportion of missing storage slots of each series.

    The denominator is the number of storage slots of the series in the sample
    (e.g. the number of quarters for a quarterly series on a monthly grid), so the
    structural empty months of lower-frequency series do not count as missing.

    Parameters
    ----------
    data : pandas.Series, pandas.DataFrame or MixedFrequencyData
        Data indexed by a :class:`pandas.PeriodIndex`.
    frequency : frequency or frequency specification, optional
        Native frequency of a Series or per-series frequencies of a DataFrame.

    Returns
    -------
    float or pandas.Series
        Proportion in ``[0, 1]`` (a Series indexed by name for panels).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.preprocessing.missing import missing_proportion
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> q = pd.Series([np.nan, np.nan, 1.0, np.nan, np.nan, np.nan], index=idx)
    >>> missing_proportion(q, frequency="Q")
    0.5
    """
    if isinstance(data, pd.Series):
        native = NativeView(data, frequency).native  # type: ignore[arg-type]
        return float(native.isna().mean()) if len(native) else 1.0
    panel, _ = as_panel(data, frequency)
    slots = panel.slot_mask()
    missing = panel.missing_mask()
    return (missing.sum() / slots.sum().clip(lower=1)).rename("missing_proportion")


# ---------------------------------------------------------------------- filling
def _fill_interior(x: np.ndarray, method: FillMethod, window: int) -> np.ndarray:
    out = x.copy()
    target = _kind_mask(x, "interior")
    if not target.any():
        return out
    pos = np.arange(len(x))
    obs = ~np.isnan(x)
    if method == "spline":
        spline = CubicSpline(pos[obs], x[obs], bc_type="not-a-knot")
        out[target] = spline(pos[target])
    elif method == "linear":
        out[target] = np.interp(pos[target], pos[obs], x[obs])
    else:
        out[target] = _nearest_median(x, target, window)
    return out


def _nearest_median(x: np.ndarray, target: np.ndarray, window: int) -> np.ndarray:
    """Centred moving median of the observations, widening the window when empty."""
    filled = np.full(int(target.sum()), np.nan)
    remaining = np.ones(filled.size, dtype=bool)
    w = max(window, 1)
    series = pd.Series(x)
    while remaining.any():
        med = series.rolling(w, center=True, min_periods=1).median().to_numpy()[target]
        take = remaining & ~np.isnan(med)
        filled[take] = med[take]
        remaining &= ~take
        w += 2
    return filled


def _fill_edge(x: np.ndarray, kind: _KIND, method: EdgeMethod, window: int) -> np.ndarray:
    target = _kind_mask(x, kind)
    obs = x[~np.isnan(x)]
    if not target.any() or obs.size == 0:
        return x
    out = x.copy()
    near = obs[-window:] if kind == "ragged" else obs[:window]
    if method == "median":
        out[target] = np.median(near)
    elif method == "mean":
        out[target] = float(np.mean(obs))
    else:
        out[target] = obs[-1] if kind == "ragged" else obs[0]
    return out


def _fill_array(
    x: np.ndarray,
    method: FillMethod,
    window: int,
    fill_ragged_edge: bool,
    fill_leading: bool,
    edge_method: EdgeMethod,
) -> np.ndarray:
    # edges are computed from the original observations, before interior filling
    out = _fill_interior(x, method, window)
    if fill_ragged_edge:
        out = np.where(_kind_mask(x, "ragged"), _fill_edge(x, "ragged", edge_method, window), out)
    if fill_leading:
        out = np.where(_kind_mask(x, "leading"), _fill_edge(x, "leading", edge_method, window), out)
    return out


def _check_fill_args(method: str, edge_method: str, window: int) -> None:
    if method not in ("spline", "linear", "moving_median"):
        raise ValueError(f"method must be 'spline', 'linear' or 'moving_median', got {method!r}.")
    if edge_method not in ("median", "last", "mean"):
        raise ValueError(f"edge_method must be 'median', 'last' or 'mean', got {edge_method!r}.")
    if isinstance(window, bool) or not isinstance(window, int | np.integer) or window < 1:
        raise ValueError(f"window must be a positive integer, got {window!r}.")


def fill_missing(
    data: Any,
    method: FillMethod = "spline",
    *,
    window: int = 3,
    fill_ragged_edge: bool = False,
    fill_leading: bool = False,
    edge_method: EdgeMethod = "median",
    columns: list[str] | None = None,
    frequency: FrequencySpec | None = None,
) -> Any:
    """Fill interior gaps (and, only on request, the edges) of each series.

    Parameters
    ----------
    data : pandas.Series, pandas.DataFrame or MixedFrequencyData
        Data indexed by a :class:`pandas.PeriodIndex` (base or native grid).
    method : {"spline", "linear", "moving_median"}, default "spline"
        Interior filling: not-a-knot cubic spline through the observations, linear
        interpolation, or median of the observations in a centred window of
        ``window`` native periods (widened by two periods until it contains an
        observation).
    window : int, default 3
        Window length for ``"moving_median"`` and number of observations used by
        ``edge_method="median"``.
    fill_ragged_edge : bool, default False
        Also fill the slots after the last observation. Off by default: the ragged
        edge is information for the nowcasting model.
    fill_leading : bool, default False
        Also fill the slots before the first observation.
    edge_method : {"median", "last", "mean"}, default "median"
        Edge filling: median of the ``window`` observations nearest to the edge,
        nearest observation (last/first), or full-sample mean.
    columns : list of str, optional
        Series to process for panel input (default: all).
    frequency : frequency or frequency specification, optional
        Native frequency of a Series or per-series frequencies of a DataFrame.

    Returns
    -------
    same type as ``data``
        Data with gaps filled (values only on storage slots).

    Raises
    ------
    NowcastDataError
        On invalid data or unknown ``columns``.
    ValueError
        On invalid ``method``, ``edge_method`` or ``window``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.preprocessing.missing import fill_missing
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> x = pd.Series([1.0, np.nan, 3.0, 4.0, np.nan, np.nan], index=idx)
    >>> fill_missing(x, "linear", frequency="M").tolist()
    [1.0, 2.0, 3.0, 4.0, nan, nan]
    >>> fill_missing(x, "linear", fill_ragged_edge=True, edge_method="last", frequency="M").tolist()
    [1.0, 2.0, 3.0, 4.0, 4.0, 4.0]
    """
    _check_fill_args(method, edge_method, window)
    w = int(window)

    def _do(x: np.ndarray) -> np.ndarray:
        return _fill_array(x, method, w, fill_ragged_edge, fill_leading, edge_method)

    if isinstance(data, pd.Series):
        view = NativeView(data, frequency)  # type: ignore[arg-type]
        return view.to_original(_do(view.native.to_numpy(float)))
    panel, was_frame = as_panel(data, frequency)
    cols = panel.columns if columns is None else list(columns)
    unknown = sorted(set(cols) - set(panel.columns))
    if unknown:
        raise NowcastDataError(f"Unknown series {unknown}.")
    frame = map_native(panel, lambda _n, s, _f: _do(s.to_numpy(float)), cols)
    n_filled = int(frame.notna().to_numpy().sum() - panel.observation_mask().to_numpy().sum())
    logger.debug("fill_missing filled %d value(s) with method %r", n_filled, method)
    return return_like(panel.with_data(frame), was_frame)

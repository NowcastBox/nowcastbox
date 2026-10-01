r"""Outlier detection (IQR rule) and correction (centred moving median).

An observation :math:`x_t` of a series is flagged as an outlier when

.. math::

    |x_t - \operatorname{med}(x)| > k \cdot \operatorname{IQR}(x),

where the median and the inter-quartile range are computed over the observed values
of the series (on its native grid) and :math:`k` is ``threshold`` (default 4, the
rule used in the dynamic-factor nowcasting literature, e.g. Stock & Watson, 2002, and
Giannone, Reichlin & Small, 2008, who replace such values before estimation).

Flagged values are replaced by the median of the non-outlying observations in a
centred window of ``window`` native periods (Stock & Watson, 2005, use a moving
median as well). Missing values are never filled here (see
:mod:`nowcastbox.preprocessing.missing`).
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import FrequencySpec
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.preprocessing._utils import (
    NativeView,
    PanelLike,
    as_panel,
    map_native,
    return_like,
)

__all__ = [
    "OutlierReplacement",
    "detect_outliers",
    "iqr_outlier_mask",
    "moving_median",
    "replace_outliers",
]

logger = get_logger(__name__)

OutlierReplacement = Literal["moving_median", "median", "nan"]
"""How outliers are replaced."""


def _check_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not np.isscalar(threshold):
        raise ValueError(f"threshold must be a positive number, got {threshold!r}.")
    value = float(threshold)  # type: ignore[arg-type]
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"threshold must be a positive finite number, got {threshold!r}.")
    return value


def _check_window(window: int) -> int:
    if isinstance(window, bool) or not isinstance(window, int | np.integer) or window < 1:
        raise ValueError(f"window must be a positive integer, got {window!r}.")
    return int(window)


def iqr_outlier_mask(values: np.ndarray, threshold: float = 4.0) -> np.ndarray:
    r"""Flag values farther than ``threshold`` IQRs from the median.

    Parameters
    ----------
    values : numpy.ndarray
        1-D array; NaN values are ignored (never flagged).
    threshold : float, default 4.0
        Multiple :math:`k` of the inter-quartile range.

    Returns
    -------
    numpy.ndarray of bool
        True where :math:`|x - \mathrm{median}| > k \cdot \mathrm{IQR}`. All False when
        fewer than 4 observations are available or the IQR is zero.

    Raises
    ------
    ValueError
        If ``threshold`` is not a positive number.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.preprocessing.outliers import iqr_outlier_mask
    >>> x = np.array([0.0, 1, 2, 3, 4, 5, 6, 7, 8, 100])
    >>> iqr_outlier_mask(x, threshold=4).nonzero()[0].tolist()
    [9]
    """
    k = _check_threshold(threshold)
    x = np.asarray(values, dtype=float)
    mask = np.zeros(x.shape, dtype=bool)
    obs = ~np.isnan(x)
    if int(obs.sum()) < 4:
        return mask
    q1, med, q3 = np.percentile(x[obs], [25.0, 50.0, 75.0])
    iqr = q3 - q1
    if iqr <= 0:
        return mask
    mask[obs] = np.abs(x[obs] - med) > k * iqr
    return mask


def moving_median(values: pd.Series | np.ndarray, window: int = 3) -> np.ndarray:
    """Centred moving median ignoring missing values.

    Parameters
    ----------
    values : pandas.Series or numpy.ndarray
        1-D values on a contiguous grid.
    window : int, default 3
        Window length in periods. For an even window the extra period is taken from
        the past. Windows are truncated at the sample borders.

    Returns
    -------
    numpy.ndarray
        Median of the non-missing values in each window (NaN if none).

    Raises
    ------
    ValueError
        If ``window`` is not a positive integer.

    Examples
    --------
    >>> import numpy as np
    >>> from nowcastbox.preprocessing.outliers import moving_median
    >>> moving_median(np.array([1.0, 9.0, 3.0, np.nan, 5.0]), window=3).tolist()
    [5.0, 3.0, 6.0, 4.0, 5.0]
    """
    w = _check_window(window)
    x = pd.Series(np.asarray(values, dtype=float))
    return x.rolling(w, center=True, min_periods=1).median().to_numpy(dtype=float)


def _replace_series(
    x: np.ndarray, threshold: float, window: int, replacement: OutlierReplacement
) -> tuple[np.ndarray, np.ndarray]:
    mask = iqr_outlier_mask(x, threshold)
    out = x.copy()
    if not mask.any():
        return out, mask
    clean = np.where(mask, np.nan, x)
    if replacement == "nan":
        return clean, mask
    if replacement == "median":
        out[mask] = np.nanmedian(clean)
        return out, mask
    fill = moving_median(clean, window)
    # isolated windows made only of outliers: fall back to the overall median
    fill = np.where(np.isnan(fill), np.nanmedian(clean), fill)
    out[mask] = fill[mask]
    return out, mask


def _check_replacement(replacement: str) -> OutlierReplacement:
    if replacement not in ("moving_median", "median", "nan"):
        raise ValueError(
            f"replacement must be 'moving_median', 'median' or 'nan', got {replacement!r}."
        )
    return replacement  # type: ignore[return-value]


def detect_outliers(
    data: pd.Series | PanelLike,
    threshold: float = 4.0,
    *,
    frequency: FrequencySpec | None = None,
) -> pd.Series | pd.DataFrame:
    """Flag outliers with the IQR rule, series by series on their native grids.

    Parameters
    ----------
    data : pandas.Series, pandas.DataFrame or MixedFrequencyData
        Data indexed by a :class:`pandas.PeriodIndex` (base or native grid).
    threshold : float, default 4.0
        Multiple of the inter-quartile range (see :func:`iqr_outlier_mask`).
    frequency : frequency or frequency specification, optional
        Native frequency of a Series or per-series frequencies of a DataFrame
        (inferred when omitted).

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Boolean mask on the input index (a DataFrame for panels).

    Raises
    ------
    NowcastDataError
        On invalid data.
    ValueError
        If ``threshold`` is invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.preprocessing.outliers import detect_outliers
    >>> idx = pd.period_range("2020-01", periods=8, freq="M")
    >>> x = pd.Series([0.0, 1.0, -1.0, 0.5, 30.0, -0.5, 0.2, 0.1], index=idx)
    >>> detect_outliers(x).loc[lambda m: m].index.astype(str).tolist()
    ['2020-05']
    """
    k = _check_threshold(threshold)
    if isinstance(data, pd.Series):
        view = NativeView(data, frequency)  # type: ignore[arg-type]
        mask = iqr_outlier_mask(view.native.to_numpy(float), k)
        return view.to_original(mask.astype(float)).fillna(0.0).astype(bool)
    panel, _ = as_panel(data, frequency)
    frame = map_native(panel, lambda _n, s, _f: iqr_outlier_mask(s.to_numpy(float), k))
    return frame.fillna(0.0).astype(bool)


def replace_outliers(
    data: Any,
    threshold: float = 4.0,
    *,
    window: int = 3,
    replacement: OutlierReplacement = "moving_median",
    columns: list[str] | None = None,
    frequency: FrequencySpec | None = None,
    return_mask: bool = False,
) -> Any:
    """Detect outliers (IQR rule) and replace them.

    Parameters
    ----------
    data : pandas.Series, pandas.DataFrame or MixedFrequencyData
        Data indexed by a :class:`pandas.PeriodIndex` (base or native grid).
    threshold : float, default 4.0
        Multiple of the inter-quartile range defining an outlier.
    window : int, default 3
        Length (in native periods) of the centred moving-median window used when
        ``replacement="moving_median"``. Outliers themselves are excluded from the
        window.
    replacement : {"moving_median", "median", "nan"}, default "moving_median"
        Replace by the centred moving median, by the series median, or by NaN (so
        that a model handles them as missing).
    columns : list of str, optional
        Series to process for panel input (default: all).
    frequency : frequency or frequency specification, optional
        Native frequency of a Series or per-series frequencies of a DataFrame.
    return_mask : bool, default False
        Also return the boolean mask of replaced values.

    Returns
    -------
    corrected : same type as ``data``
        Data with outliers replaced.
    mask : pandas.Series or pandas.DataFrame
        Only when ``return_mask=True``: the outliers that were replaced.

    Raises
    ------
    NowcastDataError
        On invalid data or unknown ``columns``.
    ValueError
        On invalid ``threshold``, ``window`` or ``replacement``.

    Warns
    -----
    DataQualityWarning
        For series whose inter-quartile range is zero (no outlier can be detected).

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.preprocessing.outliers import replace_outliers
    >>> idx = pd.period_range("2020-01", periods=8, freq="M")
    >>> x = pd.Series([0.0, 1.0, -1.0, 0.5, 30.0, -0.5, 0.2, 0.1], index=idx)
    >>> replace_outliers(x).tolist()
    [0.0, 1.0, -1.0, 0.5, 0.0, -0.5, 0.2, 0.1]
    """
    k = _check_threshold(threshold)
    w = _check_window(window)
    rep = _check_replacement(replacement)
    if isinstance(data, pd.Series):
        view = NativeView(data, frequency)  # type: ignore[arg-type]
        _warn_zero_iqr({str(data.name): view.native})
        values, mask = _replace_series(view.native.to_numpy(float), k, w, rep)
        out = view.to_original(values)
        mask_s = view.to_original(mask.astype(float)).fillna(0.0).astype(bool)
        return (out, mask_s) if return_mask else out
    panel, was_frame = as_panel(data, frequency)
    cols = panel.columns if columns is None else list(columns)
    unknown = sorted(set(cols) - set(panel.columns))
    if unknown:
        raise NowcastDataError(f"Unknown series {unknown}.")
    masks: dict[str, np.ndarray] = {}

    def _one(name: str, s: pd.Series, _f: object) -> np.ndarray:
        _warn_zero_iqr({name: s})
        values, m = _replace_series(s.to_numpy(float), k, w, rep)
        masks[name] = m
        return values

    frame = map_native(panel, _one, cols)
    n_replaced = {c: int(m.sum()) for c, m in masks.items() if m.any()}
    if n_replaced:
        logger.info("Replaced outliers: %s", n_replaced)
    out_panel = return_like(panel.with_data(frame), was_frame)
    if not return_mask:
        return out_panel
    mask_frame = pd.DataFrame(False, index=panel.index, columns=panel.columns)
    for name, m in masks.items():
        view = NativeView(panel[name], panel.metadata[name].frequency, name=name)
        mask_frame[name] = view.to_original(m.astype(float)).to_numpy()
    return out_panel, mask_frame.fillna(0.0).astype(bool)


def _warn_zero_iqr(series: dict[str, pd.Series]) -> None:
    for name, s in series.items():
        obs = s.dropna().to_numpy(float)
        if len(obs) >= 4:
            q1, q3 = np.percentile(obs, [25.0, 75.0])
            if q3 - q1 <= 0:
                warnings.warn(
                    f"Series {name!r} has a zero inter-quartile range; "
                    "outlier detection was skipped.",
                    DataQualityWarning,
                    stacklevel=4,
                )

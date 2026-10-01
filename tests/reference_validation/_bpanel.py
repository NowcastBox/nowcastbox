"""Emulation of the panel treatment documented for ``Bpanel`` (R package ``nowcasting``).

Clean-room note: this module re-implements the behaviour **described in the help page**
of ``Bpanel`` (``?Bpanel``: "Outliers are defined as observations that lie more than 4
IQR from the median. All missings and outliers are replaced by the median. A centered
moving average of degree k is calculated, forming a new panel. Then the missings and
outliers are replaced by their equivalent observations on this new panel."), with the
remaining details (window of ``2k + 1`` periods, edge padding, unnormalised
Mariano-Murasawa weights applied before the correction, the ``na.prop`` rule) identified
by comparing candidate rules with the black-box outputs stored in ``fixtures/r``. It is
used only to rebuild the R-processed panels inside the tests and to quantify where
nowcastbox's own (different) outlier rule diverges; it is not part of the package.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from nowcastbox.core.exceptions import DataQualityWarning
from nowcastbox.preprocessing import apply_transforms, rolling_aggregate

MM_WEIGHTS_UNNORMALISED = np.array([1.0, 2.0, 3.0, 2.0, 1.0])


def iqr_flags(x: np.ndarray, threshold: float = 4.0) -> np.ndarray:
    """``|x - median| > threshold * IQR`` over the observed values (R quantile type 7)."""
    obs = ~np.isnan(x)
    flags = np.zeros(x.shape, dtype=bool)
    if not obs.any():
        return flags
    med = np.median(x[obs])
    q1, q3 = np.percentile(x[obs], [25.0, 75.0])
    flags[obs] = np.abs(x[obs] - med) > threshold * (q3 - q1)
    return flags


def correct_series(x: np.ndarray, *, na_replace: bool, k: int = 3) -> np.ndarray:
    """Outlier (and optionally missing-value) correction of one series.

    Outliers - and, with ``na_replace``, the missing values before the last observation -
    are set to the median; a centred moving average of ``2k + 1`` periods of that series
    (edge-padded, ignoring missing values) then replaces them.
    """
    x = np.asarray(x, dtype=float)
    obs = ~np.isnan(x)
    if not obs.any():
        return x.copy()
    med = float(np.median(x[obs]))
    out = iqr_flags(x)
    fill = np.zeros(x.shape, dtype=bool)
    if na_replace:
        last = int(np.flatnonzero(obs)[-1])
        fill = ~obs & (np.arange(x.size) <= last)
    z = np.where(out | fill, med, x)
    padded = np.pad(z, k, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * k + 1)
    with np.errstate(all="ignore"):
        counts = (~np.isnan(windows)).sum(axis=1)
        ma = np.where(counts > 0, np.nansum(windows, axis=1) / np.maximum(counts, 1), np.nan)
    y = x.copy()
    target = out | fill
    y[target] = ma[target]
    return y


def transform_monthly(raw: pd.DataFrame, codes: dict[str, int]) -> pd.DataFrame:
    """Codes 0-7 applied with nowcastbox to monthly series (validated separately)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)  # empty series of USGDP
        out = apply_transforms(raw, codes, frequency="M")
    assert isinstance(out, pd.DataFrame)
    return out


def emulate_bpanel(
    raw: pd.DataFrame,
    codes: dict[str, int],
    *,
    na_replace: bool = True,
    aggregate: bool = False,
    na_prop: float = 1 / 3,
    h: int = 12,
) -> pd.DataFrame:
    """Rebuild ``Bpanel(raw, codes, NA.replace, aggregate, k.ma = 3, na.prop, h)``.

    Monthly series only (every column is treated as monthly, as Bpanel does).
    """
    x = transform_monthly(raw, codes)
    if aggregate:
        x = x.apply(
            lambda s: pd.Series(
                rolling_aggregate(s.to_numpy(), MM_WEIGHTS_UNNORMALISED), index=s.index
            )
        )
    prop = x.isna().mean()
    x = x.loc[:, prop <= na_prop] if na_prop < 1 else x.loc[:, prop < 1]
    corrected = {c: correct_series(x[c].to_numpy(), na_replace=na_replace) for c in x.columns}
    out = pd.DataFrame(corrected, index=x.index)
    if h:
        idx = pd.period_range(out.index[0], periods=len(out) + h, freq="M")
        out = out.reindex(idx)
    return out

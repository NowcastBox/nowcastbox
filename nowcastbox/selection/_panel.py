"""Shared helpers for the selection module: panel preparation, PCA and VAR by OLS.

These helpers are private. They turn the user input (``DataFrame``, ``ndarray`` or
:class:`~nowcastbox.core.data.MixedFrequencyData`) into a balanced, optionally
standardised ``T x N`` array and provide the principal-component and VAR computations
shared by :mod:`~nowcastbox.selection.bai_ng_factors` and
:mod:`~nowcastbox.selection.bai_ng_shocks`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError

__all__ = [
    "MissingPolicy",
    "PanelInput",
    "PreparedPanel",
    "fit_var_ols",
    "prepare_panel",
    "principal_components",
]

logger = get_logger(__name__)

PanelInput = pd.DataFrame | np.ndarray | MixedFrequencyData
MissingPolicy = Literal["drop_rows", "drop_columns", "raise"]

_MISSING_POLICIES = ("drop_rows", "drop_columns", "raise")


@dataclass(frozen=True)
class PreparedPanel:
    """Balanced numeric panel ready for principal components.

    Attributes
    ----------
    values : numpy.ndarray
        ``(n_periods, n_series)`` array without missing values (standardised if
        requested).
    index : pandas.Index
        Row labels kept.
    columns : pandas.Index
        Column labels kept.
    dropped_rows : pandas.Index
        Row labels removed because of missing values.
    dropped_columns : pandas.Index
        Column labels removed (missing values or lower-frequency series).
    """

    values: np.ndarray
    index: pd.Index
    columns: pd.Index
    dropped_rows: pd.Index
    dropped_columns: pd.Index

    @property
    def n_periods(self) -> int:
        """Number of time periods ``T``."""
        return int(self.values.shape[0])

    @property
    def n_series(self) -> int:
        """Number of cross-section units ``N``."""
        return int(self.values.shape[1])


def _to_frame(x: PanelInput) -> tuple[pd.DataFrame, list[str]]:
    """Convert the input to a float DataFrame; return also excluded low-frequency columns."""
    excluded: list[str] = []
    if isinstance(x, MixedFrequencyData):
        frame = x.to_frame()
        base = x.base_frequency
        excluded = [str(c) for c, f in x.frequencies.items() if f != base]
        if excluded:
            warnings.warn(
                f"{len(excluded)} series with a frequency lower than the base grid "
                f"({base.value}) were excluded from the balanced panel: {excluded}.",
                DataQualityWarning,
                stacklevel=4,
            )
            frame = frame.drop(columns=excluded)
    elif isinstance(x, pd.DataFrame):
        frame = x.copy()
    elif isinstance(x, np.ndarray):
        if x.ndim != 2:
            raise NowcastDataError(f"Array input must be 2-D (T x N); got ndim={x.ndim}.")
        frame = pd.DataFrame(x, columns=[f"x{i + 1}" for i in range(x.shape[1])])
    else:
        raise TypeError(
            "x must be a pandas DataFrame, a 2-D numpy array or a MixedFrequencyData; "
            f"got {type(x).__name__}."
        )
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError) as exc:
        raise NowcastDataError(f"Panel contains non-numeric values: {exc}") from exc
    return frame, excluded


def prepare_panel(
    x: PanelInput,
    *,
    standardize: bool = True,
    missing: MissingPolicy = "drop_rows",
    min_periods: int = 3,
    min_series: int = 2,
) -> PreparedPanel:
    """Build a balanced (optionally standardised) panel from user input.

    Parameters
    ----------
    x : DataFrame, ndarray or MixedFrequencyData
        ``T x N`` panel. For :class:`MixedFrequencyData` only the series observed at the
        base frequency are used (a :class:`DataQualityWarning` lists the others).
    standardize : bool, default True
        Standardise each column to zero mean and unit variance (``ddof=1``) after
        removing missing values.
    missing : {"drop_rows", "drop_columns", "raise"}, default "drop_rows"
        How to obtain a balanced panel: drop every period with a missing value, drop
        every series with a missing value, or raise :class:`NowcastDataError`.
    min_periods, min_series : int
        Minimum dimensions of the balanced panel.

    Returns
    -------
    PreparedPanel
        Balanced panel and bookkeeping of what was dropped.

    Raises
    ------
    NowcastDataError
        Non-finite values, too few observations/series, constant series or missing
        values with ``missing="raise"``.

    Examples
    --------
    >>> import numpy as np
    >>> p = prepare_panel(np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 5.0]]))
    >>> p.values.shape
    (3, 2)
    """
    if missing not in _MISSING_POLICIES:
        raise ValueError(f"missing must be one of {_MISSING_POLICIES}; got {missing!r}.")
    frame, excluded = _to_frame(x)
    arr = frame.to_numpy()
    if np.isinf(arr).any():
        raise NowcastDataError("Panel contains infinite values.")
    nan_mask = np.isnan(arr)
    dropped_rows = frame.index[:0]
    dropped_cols = pd.Index(excluded, dtype=object)
    if nan_mask.any():
        if missing == "raise":
            raise NowcastDataError(
                f"Panel has {int(nan_mask.sum())} missing values; use missing='drop_rows' "
                "or 'drop_columns', or fill them beforehand."
            )
        if missing == "drop_rows":
            keep: np.ndarray = np.asarray(~nan_mask.any(axis=1), dtype=bool)
            dropped_rows = frame.index[~keep]
            frame = frame.loc[keep]
            msg = f"Dropped {len(dropped_rows)} of {len(keep)} periods with missing values."
        else:
            keep = np.asarray(~nan_mask.any(axis=0), dtype=bool)
            new_drop = frame.columns[~keep]
            dropped_cols = dropped_cols.append(pd.Index(new_drop, dtype=object))
            frame = frame.loc[:, keep]
            msg = f"Dropped {len(new_drop)} series with missing values: {list(new_drop)}."
        warnings.warn(msg, DataQualityWarning, stacklevel=3)
        logger.info(msg)
    n_periods, n_series = frame.shape
    if n_periods < min_periods:
        raise NowcastDataError(
            f"Balanced panel has {n_periods} periods; at least {min_periods} are required."
        )
    if n_series < min_series:
        raise NowcastDataError(
            f"Balanced panel has {n_series} series; at least {min_series} are required."
        )
    values = frame.to_numpy(dtype=float, copy=True)
    if standardize:
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        constant = std <= np.finfo(float).eps * np.maximum(1.0, np.abs(mean))
        if constant.any():
            names = list(frame.columns[constant])
            raise NowcastDataError(f"Cannot standardise constant series: {names}.")
        values = (values - mean) / std
    return PreparedPanel(
        values=values,
        index=frame.index,
        columns=frame.columns,
        dropped_rows=dropped_rows,
        dropped_columns=dropped_cols,
    )


def principal_components(
    values: np.ndarray, n_factors: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Principal-component factors with the Bai & Ng (2002) normalisation.

    With the singular value decomposition ``X = U S V'`` of the ``T x N`` panel, the
    factors are ``F = sqrt(T) U[:, :r]`` (so that ``F'F / T = I_r``) and the loadings
    ``Lambda = X'F / T``.

    Parameters
    ----------
    values : numpy.ndarray
        Balanced ``T x N`` panel.
    n_factors : int
        Number of factors ``r`` (``0 <= r <= min(T, N)``).

    Returns
    -------
    factors : numpy.ndarray
        ``(T, r)`` factor estimates.
    loadings : numpy.ndarray
        ``(N, r)`` loadings.
    singular_values : numpy.ndarray
        All singular values of ``X`` in decreasing order (``eig(X'X) = s**2``).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.default_rng(0).normal(size=(20, 5))
    >>> f, lam, s = principal_components(x, 2)
    >>> np.allclose(f.T @ f / 20, np.eye(2))
    True
    """
    n_periods = values.shape[0]
    u, s, _ = np.linalg.svd(values, full_matrices=False)
    factors = np.sqrt(n_periods) * u[:, :n_factors]
    # Fix the sign so that the largest-|.| element of each factor's loading is positive
    loadings = values.T @ factors / n_periods
    if n_factors:
        idx = np.argmax(np.abs(loadings), axis=0)
        signs = np.sign(loadings[idx, np.arange(n_factors)])
        signs[signs == 0] = 1.0
        factors = factors * signs
        loadings = loadings * signs
    return factors, loadings, s


def fit_var_ols(
    series: np.ndarray, n_lags: int, *, trend: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Equation-by-equation OLS estimate of a VAR(p).

    ``y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t`` (Lütkepohl, 2005, §3.2).

    Parameters
    ----------
    series : numpy.ndarray
        ``(T, k)`` multivariate series.
    n_lags : int
        Lag order ``p >= 1``.
    trend : bool, default True
        Include an intercept.

    Returns
    -------
    coefficients : numpy.ndarray
        ``(k, k*p [+1])`` matrix ``[c, A_1, ..., A_p]`` (intercept first when present).
    residuals : numpy.ndarray
        ``(T - p, k)`` residuals.

    Raises
    ------
    NowcastDataError
        If there are not more observations than regressors.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.random.default_rng(1).normal(size=(50, 2))
    >>> coef, resid = fit_var_ols(y, 1)
    >>> coef.shape, resid.shape
    ((2, 3), (49, 2))
    """
    n_periods, k = series.shape
    n_obs = n_periods - n_lags
    n_regressors = k * n_lags + int(trend)
    if n_obs <= n_regressors:
        raise NowcastDataError(
            f"VAR({n_lags}) with {k} variables needs more than {n_regressors + n_lags} "
            f"periods; got {n_periods}."
        )
    lagged = [series[n_lags - j : n_periods - j] for j in range(1, n_lags + 1)]
    design = np.hstack(([np.ones((n_obs, 1))] if trend else []) + lagged)
    target = series[n_lags:]
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coef
    return coef.T, residuals

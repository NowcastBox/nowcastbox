r"""Point-forecast accuracy metrics, overall and by nowcast horizon.

Errors follow the convention :math:`e = y - \hat y` (actual minus forecast), so a
positive bias means the forecasts are too low on average. For a set of errors
:math:`e_1, \dots, e_n` (missing values are ignored):

.. math::

    \mathrm{RMSFE} = \Big(\tfrac1n \sum_t e_t^2\Big)^{1/2}, \quad
    \mathrm{MAE} = \tfrac1n \sum_t |e_t|, \quad
    \mathrm{bias} = \tfrac1n \sum_t e_t ,

and the relative RMSFE of a model with respect to a reference model is the ratio of
their RMSFEs on the common sample (values below one favour the model). Accuracy by
*nowcast horizon* - the distance between the information date and the end (or the
release) of the target period - is the standard way of reporting nowcasting
performance (Giannone, Reichlin & Small, 2008; Bańbura, Giannone, Modugno & Reichlin,
2013).

References
----------
Giannone, D., Reichlin, L. & Small, D. (2008). Nowcasting: The real-time informational
content of macroeconomic data. *Journal of Monetary Economics*, 55(4), 665-676.

Bańbura, M., Giannone, D., Modugno, M. & Reichlin, L. (2013). Now-casting and the
real-time data flow. In *Handbook of Economic Forecasting*, vol. 2A, 195-237.

Hyndman, R. J. & Koehler, A. B. (2006). Another look at measures of forecast accuracy.
*International Journal of Forecasting*, 22(4), 679-688.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
import pandas as pd

__all__ = [
    "METRICS",
    "LossLike",
    "accuracy_by_horizon",
    "bias",
    "forecast_errors",
    "loss_values",
    "mae",
    "metric_by_horizon",
    "mse",
    "relative_rmsfe",
    "rmsfe",
]

ArrayLike: TypeAlias = np.ndarray | pd.Series | Sequence[float]
LossLike: TypeAlias = str | Callable[[np.ndarray], np.ndarray]


def _clean(errors: ArrayLike) -> np.ndarray:
    values = np.asarray(errors, dtype=float).ravel()
    return values[np.isfinite(values)]


def forecast_errors(actual: ArrayLike, forecast: ArrayLike) -> np.ndarray:
    r"""Forecast errors :math:`e = y - \hat y`.

    Parameters
    ----------
    actual, forecast : array-like
        Realisations and forecasts (same length; NaN allowed).

    Returns
    -------
    numpy.ndarray
        Errors (NaN where either input is missing).

    Raises
    ------
    ValueError
        If the lengths differ.

    Examples
    --------
    >>> from nowcastbox.evaluation import forecast_errors
    >>> forecast_errors([1.0, 2.0], [0.5, 2.5]).tolist()
    [0.5, -0.5]
    """
    a = np.asarray(actual, dtype=float).ravel()
    f = np.asarray(forecast, dtype=float).ravel()
    if a.shape != f.shape:
        raise ValueError(f"actual and forecast have different lengths ({a.size}, {f.size}).")
    return a - f


def mse(errors: ArrayLike) -> float:
    """Mean squared forecast error (NaN ignored; NaN if no error is available).

    Parameters
    ----------
    errors : array-like
        Forecast errors.

    Returns
    -------
    float
        Mean squared error.

    Examples
    --------
    >>> from nowcastbox.evaluation import mse
    >>> mse([1.0, -3.0, float("nan")])
    5.0
    """
    e = _clean(errors)
    return float(np.mean(e**2)) if e.size else float("nan")


def rmsfe(errors: ArrayLike) -> float:
    """Root mean squared forecast error (NaN ignored).

    Parameters
    ----------
    errors : array-like
        Forecast errors.

    Returns
    -------
    float
        RMSFE (NaN if no error is available).

    Examples
    --------
    >>> from nowcastbox.evaluation import rmsfe
    >>> rmsfe([3.0, -4.0, 0.0, 0.0])
    2.5
    """
    return float(np.sqrt(mse(errors)))


def mae(errors: ArrayLike) -> float:
    """Mean absolute forecast error (NaN ignored).

    Parameters
    ----------
    errors : array-like
        Forecast errors.

    Returns
    -------
    float
        MAE (NaN if no error is available).

    Examples
    --------
    >>> from nowcastbox.evaluation import mae
    >>> mae([1.0, -3.0])
    2.0
    """
    e = _clean(errors)
    return float(np.mean(np.abs(e))) if e.size else float("nan")


def bias(errors: ArrayLike) -> float:
    """Mean forecast error (actual minus forecast; NaN ignored).

    Parameters
    ----------
    errors : array-like
        Forecast errors.

    Returns
    -------
    float
        Bias (NaN if no error is available).

    Examples
    --------
    >>> from nowcastbox.evaluation import bias
    >>> bias([1.0, -3.0])
    -1.0
    """
    e = _clean(errors)
    return float(np.mean(e)) if e.size else float("nan")


def _count(errors: ArrayLike) -> float:
    return float(_clean(errors).size)


def relative_rmsfe(errors: ArrayLike, reference_errors: ArrayLike) -> float:
    """RMSFE of a model divided by the RMSFE of a reference on their common sample.

    Parameters
    ----------
    errors, reference_errors : array-like
        Aligned errors of the model and of the reference (pairs with a NaN are
        dropped).

    Returns
    -------
    float
        Ratio (< 1: the model is more accurate); NaN without common observations or
        when the reference RMSFE is zero.

    Raises
    ------
    ValueError
        If the lengths differ.

    Examples
    --------
    >>> from nowcastbox.evaluation import relative_rmsfe
    >>> relative_rmsfe([1.0, 1.0, float("nan")], [2.0, 2.0, 5.0])
    0.5
    """
    e = np.asarray(errors, dtype=float).ravel()
    r = np.asarray(reference_errors, dtype=float).ravel()
    if e.shape != r.shape:
        raise ValueError("errors and reference_errors have different lengths.")
    ok = np.isfinite(e) & np.isfinite(r)
    denominator = rmsfe(r[ok])
    if not ok.any() or denominator == 0:
        return float("nan")
    return rmsfe(e[ok]) / denominator


#: Accuracy metrics available by name (functions of the error vector).
METRICS: dict[str, Callable[[ArrayLike], float]] = {
    "rmsfe": rmsfe,
    "mse": mse,
    "mae": mae,
    "bias": bias,
    "n": _count,
}


def loss_values(errors: ArrayLike, loss: LossLike = "squared") -> np.ndarray:
    """Loss of each forecast error.

    Parameters
    ----------
    errors : array-like
        Forecast errors.
    loss : {"squared", "absolute"} or callable, default "squared"
        Loss function; a callable receives the error array and returns the losses.

    Returns
    -------
    numpy.ndarray
        Losses (NaN errors give NaN losses).

    Raises
    ------
    ValueError
        If ``loss`` is unknown or a callable returns an array of another shape.

    Examples
    --------
    >>> from nowcastbox.evaluation import loss_values
    >>> loss_values([1.0, -2.0], "absolute").tolist()
    [1.0, 2.0]
    """
    e = np.asarray(errors, dtype=float).ravel()
    if callable(loss):
        out = np.asarray(loss(e), dtype=float).ravel()
        if out.shape != e.shape:
            raise ValueError("The loss function must return one loss per error.")
        return out
    if loss == "squared":
        return e**2
    if loss == "absolute":
        return np.abs(e)
    raise ValueError(f"loss must be 'squared', 'absolute' or a callable, got {loss!r}.")


def _metric(name: str) -> Callable[[ArrayLike], float]:
    if name not in METRICS:
        raise ValueError(f"Unknown metric {name!r}; available: {sorted(METRICS)}.")
    return METRICS[name]


def metric_by_horizon(
    frame: pd.DataFrame,
    metric: str = "rmsfe",
    horizon: str | None = "months_to_end",
    *,
    model_col: str = "model",
    error_col: str = "error",
) -> pd.DataFrame:
    """One accuracy metric for each model and horizon.

    Parameters
    ----------
    frame : pandas.DataFrame
        Long table with one row per forecast (columns ``model_col``, ``error_col`` and
        ``horizon``).
    metric : {"rmsfe", "mse", "mae", "bias", "n"}, default "rmsfe"
        Metric.
    horizon : str or None, default "months_to_end"
        Column defining the horizon groups; ``None`` pools all forecasts (one row
        labelled ``"all"``).
    model_col, error_col : str
        Column names of the model label and of the errors.

    Returns
    -------
    pandas.DataFrame
        Index: horizons (sorted); columns: models (in order of appearance).

    Raises
    ------
    ValueError
        If the metric is unknown.
    KeyError
        If a column is missing.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.evaluation import metric_by_horizon
    >>> frame = pd.DataFrame(
    ...     {"model": ["A", "A", "B", "B"], "h": [0, 1, 0, 1], "error": [1.0, 2.0, -1.0, 0.5]}
    ... )
    >>> metric_by_horizon(frame, "mae", "h").to_dict()
    {'A': {0: 1.0, 1: 2.0}, 'B': {0: 1.0, 1: 0.5}}
    """
    func = _metric(metric)
    missing = [c for c in (model_col, error_col, horizon) if c is not None and c not in frame]
    if missing:
        raise KeyError(f"Columns {missing} are not in the frame.")
    models = list(dict.fromkeys(frame[model_col]))
    work = frame.assign(_h="all") if horizon is None else frame.assign(_h=frame[horizon])
    grouped = work.groupby(["_h", model_col], sort=True, dropna=False)[error_col]
    long = grouped.apply(lambda e: func(e.to_numpy(dtype=float)))
    table = long.unstack(model_col)  # noqa: PD010
    table = table.reindex(columns=models)
    table.index.name = "all" if horizon is None else horizon
    table.columns.name = None
    return table.astype(float)


def accuracy_by_horizon(
    frame: pd.DataFrame,
    horizon: str | None = "months_to_end",
    metrics: Sequence[str] = ("rmsfe", "mae", "bias", "n"),
    *,
    model_col: str = "model",
    error_col: str = "error",
) -> pd.DataFrame:
    """Several accuracy metrics for each model and horizon.

    Parameters
    ----------
    frame : pandas.DataFrame
        Long table with one row per forecast.
    horizon : str or None, default "months_to_end"
        Column defining the horizon groups (``None`` pools everything).
    metrics : sequence of str, default ("rmsfe", "mae", "bias", "n")
        Metrics (keys of :data:`METRICS`).
    model_col, error_col : str
        Column names of the model label and of the errors.

    Returns
    -------
    pandas.DataFrame
        Index: horizons; columns: :class:`pandas.MultiIndex` ``(metric, model)``.

    Raises
    ------
    ValueError
        If a metric is unknown or ``metrics`` is empty.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.evaluation import accuracy_by_horizon
    >>> frame = pd.DataFrame({"model": ["A", "A"], "h": [0, 0], "error": [1.0, -1.0]})
    >>> accuracy_by_horizon(frame, "h", ("rmsfe", "bias")).iloc[0].tolist()
    [1.0, 0.0]
    """
    if not metrics:
        raise ValueError("At least one metric is required.")
    parts = {
        m: metric_by_horizon(frame, m, horizon, model_col=model_col, error_col=error_col)
        for m in metrics
    }
    return pd.concat(parts, axis=1, names=["metric", "model"])

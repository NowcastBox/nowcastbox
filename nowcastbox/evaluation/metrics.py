r"""Point-forecast accuracy metrics, overall and by nowcast horizon.

Errors follow the convention :math:`e = y - \hat y` (actual minus forecast), so a
positive bias means the forecasts are too low on average. For a set of errors
:math:`e_1, \dots, e_n` (missing values are ignored):

.. math::

    \mathrm{RMSFE} = \Big(\tfrac1n \sum_t e_t^2\Big)^{1/2}, \quad
    \mathrm{MAE} = \tfrac1n \sum_t |e_t|, \quad
    \mathrm{bias} = \tfrac1n \sum_t e_t ,

and the relative RMSFE of a model with respect to a reference model is the ratio of
their RMSFEs on the common sample (values below one favour the model).

The *forecast directional accuracy* (FDA; Linzenich & Meunier, 2024, note 9) is the
share of forecasts that get the direction of change right relative to a previous
value :math:`y^{p}_t` (by default the last value of the target observed at the
forecast's information date),

.. math::

    \mathrm{FDA} = \tfrac1n \sum_t I_t, \qquad
    I_t = \mathbb 1\big[(y_t - y^{p}_t)(\hat y_t - y^{p}_t) > 0\big],

so a forecast scores when it predicts correctly whether the variable rises or falls
(accelerates or decelerates, for a growth rate). Directional metrics need the actual,
forecast and previous values rather than only the errors, and are kept in the separate
registry :data:`DIRECTIONAL_METRICS`; the significance of a hit rate is assessed with
:func:`~nowcastbox.evaluation.pesaran_timmermann`. Accuracy by
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

Pesaran, M. H. & Timmermann, A. (1992). A simple nonparametric test of predictive
performance. *Journal of Business & Economic Statistics*, 10(4), 461-465.

Blaskowitz, O. & Herwartz, H. (2011). On economic evaluation of directional forecasts.
*International Journal of Forecasting*, 27(4), 1058-1065.

Linzenich, J. & Meunier, B. (2024). Nowcasting made easier: a toolbox for economists.
ECB Working Paper No. 3004.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
import pandas as pd

__all__ = [
    "DIRECTIONAL_METRICS",
    "METRICS",
    "LossLike",
    "accuracy_by_horizon",
    "bias",
    "directional_accuracy",
    "directional_changes",
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


def directional_changes(
    actual: ArrayLike, forecast: ArrayLike, previous: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    r"""Actual and predicted changes with respect to a previous value.

    Parameters
    ----------
    actual, forecast, previous : array-like
        Realisations :math:`y_t`, forecasts :math:`\hat y_t` and previous values
        :math:`y^{p}_t` (same length; triples with a NaN are dropped).

    Returns
    -------
    tuple of numpy.ndarray
        ``(y - y_prev, y_hat - y_prev)`` on the complete triples.

    Raises
    ------
    ValueError
        If the lengths differ.

    Examples
    --------
    >>> from nowcastbox.evaluation import directional_changes
    >>> dy, dyhat = directional_changes([2.0, 1.0, 3.0], [1.5, 2.0, float("nan")], [1.0] * 3)
    >>> dy.tolist(), dyhat.tolist()
    ([1.0, 0.0], [0.5, 1.0])
    """
    a = np.asarray(actual, dtype=float).ravel()
    f = np.asarray(forecast, dtype=float).ravel()
    p = np.asarray(previous, dtype=float).ravel()
    if not a.shape == f.shape == p.shape:
        raise ValueError(
            f"actual, forecast and previous have different lengths ({a.size}, {f.size}, {p.size})."
        )
    ok = np.isfinite(a) & np.isfinite(f) & np.isfinite(p)
    return a[ok] - p[ok], f[ok] - p[ok]


def directional_accuracy(actual: ArrayLike, forecast: ArrayLike, previous: ArrayLike) -> float:
    r"""Forecast directional accuracy (FDA): share of correctly predicted directions.

    :math:`\mathrm{FDA} = n^{-1}\sum_t \mathbb 1[(y_t - y^{p}_t)(\hat y_t - y^{p}_t) > 0]`
    (Linzenich & Meunier, 2024); a zero actual or predicted change counts as a miss.

    Parameters
    ----------
    actual, forecast, previous : array-like
        Realisations, forecasts and previous values (triples with a NaN are dropped).

    Returns
    -------
    float
        FDA in :math:`[0, 1]` (NaN if no complete triple is available; 0.5 is the
        hit rate of a coin flip).

    Raises
    ------
    ValueError
        If the lengths differ.

    See Also
    --------
    nowcastbox.evaluation.pesaran_timmermann : Test of directional predictability.

    Examples
    --------
    >>> from nowcastbox.evaluation import directional_accuracy
    >>> directional_accuracy([1.4, 0.5, 2.0, 1.2], [1.3, 0.8, 1.5, 0.9], [1.0, 1.0, 1.0, 1.0])
    0.75
    """
    dy, dyhat = directional_changes(actual, forecast, previous)
    if not dy.size:
        return float("nan")
    return float(np.mean(dy * dyhat > 0))


#: Metrics of the actual, forecast and previous values (directional accuracy).
DIRECTIONAL_METRICS: dict[str, Callable[[ArrayLike, ArrayLike, ArrayLike], float]] = {
    "fda": directional_accuracy,
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
        available = sorted([*METRICS, *DIRECTIONAL_METRICS])
        raise ValueError(f"Unknown metric {name!r}; available: {available}.")
    return METRICS[name]


def _group_metric(
    metric: str, error_col: str, directional_cols: Sequence[str]
) -> tuple[list[str], Callable[[pd.DataFrame], float]]:
    """Columns a metric needs and the function computing it on a group of rows."""
    if metric in DIRECTIONAL_METRICS:
        directional = DIRECTIONAL_METRICS[metric]
        cols = list(directional_cols)

        def from_values(group: pd.DataFrame) -> float:
            a, f, p = (group[c].to_numpy(dtype=float) for c in cols)
            return directional(a, f, p)

        return cols, from_values
    func = _metric(metric)

    def from_errors(group: pd.DataFrame) -> float:
        return func(group[error_col].to_numpy(dtype=float))

    return [error_col], from_errors


def metric_by_horizon(
    frame: pd.DataFrame,
    metric: str = "rmsfe",
    horizon: str | None = "months_to_end",
    *,
    model_col: str = "model",
    error_col: str = "error",
    actual_col: str = "actual",
    forecast_col: str = "forecast",
    previous_col: str = "previous_actual",
) -> pd.DataFrame:
    """One accuracy metric for each model and horizon.

    Parameters
    ----------
    frame : pandas.DataFrame
        Long table with one row per forecast (columns ``model_col``, ``horizon`` and
        ``error_col`` or, for a directional metric, ``actual_col``, ``forecast_col``
        and ``previous_col``).
    metric : {"rmsfe", "mse", "mae", "bias", "n", "fda"}, default "rmsfe"
        Metric: a key of :data:`METRICS` (functions of the errors) or of
        :data:`DIRECTIONAL_METRICS` (functions of the actual, forecast and previous
        values).
    horizon : str or None, default "months_to_end"
        Column defining the horizon groups; ``None`` pools all forecasts (one row
        labelled ``"all"``).
    model_col, error_col : str
        Column names of the model label and of the errors.
    actual_col, forecast_col, previous_col : str
        Column names of the actual, forecast and previous values (directional metrics
        only).

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
    >>> frame = frame.assign(actual=[2.0, 1.0, 2.0, 1.0], forecast=[1.5, 1.5, 0.5, 0.5])
    >>> metric_by_horizon(frame.assign(previous_actual=1.0), "fda", None).to_dict()
    {'A': {'all': 0.5}, 'B': {'all': 0.0}}
    """
    cols, func = _group_metric(metric, error_col, (actual_col, forecast_col, previous_col))
    missing = [c for c in (model_col, *cols, horizon) if c is not None and c not in frame]
    if missing:
        raise KeyError(f"Columns {missing} are not in the frame.")
    models = list(dict.fromkeys(frame[model_col]))
    work = frame.assign(_h="all") if horizon is None else frame.assign(_h=frame[horizon])
    grouped = work.groupby(["_h", model_col], sort=True, dropna=False)[cols]
    long = grouped.apply(func)
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
    previous_col: str = "previous_actual",
) -> pd.DataFrame:
    """Several accuracy metrics for each model and horizon.

    Parameters
    ----------
    frame : pandas.DataFrame
        Long table with one row per forecast.
    horizon : str or None, default "months_to_end"
        Column defining the horizon groups (``None`` pools everything).
    metrics : sequence of str, default ("rmsfe", "mae", "bias", "n")
        Metrics (keys of :data:`METRICS` or :data:`DIRECTIONAL_METRICS`, e.g.
        ``"fda"``, which needs the ``actual``, ``forecast`` and ``previous_col``
        columns).
    model_col, error_col : str
        Column names of the model label and of the errors.
    previous_col : str, default "previous_actual"
        Column of the previous values (directional metrics only).

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
    options = {"model_col": model_col, "error_col": error_col, "previous_col": previous_col}
    parts = {m: metric_by_horizon(frame, m, horizon, **options) for m in metrics}
    return pd.concat(parts, axis=1, names=["metric", "model"])

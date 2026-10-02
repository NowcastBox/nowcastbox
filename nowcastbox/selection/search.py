r"""Random and grid search of model specifications evaluated in pseudo real time.

The model-building step of the ECB Nowcasting Toolbox (Linzenich & Meunier, 2024,
§2.2-2.3) draws many specifications of a nowcasting model within user-given bounds -
number of factors, lags of the factor VAR, block structure, first period of the
estimation sample and number of indicators - evaluates each one in a pseudo real-time
exercise and ranks them by a score that weights the accuracy at the different
horizons (backcast, nowcast, forecast) and by several metrics (RMSFE, directional
accuracy). :class:`SpecificationSearch` implements this design:

* **Space.** ``space`` maps every searched setting to its values (a list of choices,
  an inclusive integer range ``(low, high)`` or a fixed value, see
  :class:`ParameterSpace`). Keys are parameters of the model (``set_params``) plus
  two special settings: ``"start"`` (first base period of the estimation sample) and
  ``"n_series"`` (number of indicators kept).
* **Funnel.** With ``"n_series"`` the indicators are taken from the top of a ranking:
  the :func:`~nowcastbox.selection.preselect` ranking (item 8 of the plan) or a list of
  names. ``ranking="preselect"`` (or a mapping of :func:`preselect` options)
  recomputes the ranking on the information set of every re-estimation vintage, so
  the selection uses only data available at the vintage; a fixed
  :class:`~nowcastbox.selection.PreselectionResult` computed with data released after
  the first vintage triggers a look-ahead warning.
* **Draws.** ``n_draws`` random specifications, each setting drawn uniformly over its
  values with its own generator ``default_rng(SeedSequence(random_state).spawn(n)[i])``
  (reproducible, and independent of ``n_draws``), or the exhaustive grid
  (``n_draws=None``). Duplicate draws are evaluated once.
* **Evaluation.** Every specification is a :class:`SpecifiedModel` evaluated with
  :class:`~nowcastbox.evaluation.PseudoRealTimeBacktest` (``backtest=`` gives its
  settings: vintages, release delays, re-estimation frequency, benchmarks...). The
  criteria are the metrics of :meth:`~nowcastbox.evaluation.BacktestResults.metrics`
  by horizon (default: the ``kind`` of the forecast), optionally on a sub-period of
  target periods (``periods=``, e.g. ``"ex-covid"``).
* **Score.** For metric :math:`m` with weight :math:`w_m` (``score=``) and horizon
  :math:`h` with weight :math:`v_h` (``horizon_weights=``), the score of
  specification :math:`s` is the weighted mean of normalised losses

  .. math::

      S_s = \frac{\sum_m \sum_h w_m v_h\, \tilde L_{s,m,h}}{\sum_m w_m \sum_h v_h},

  where the loss :math:`L_{s,m,h}` is the metric itself for RMSFE, MSE and MAE, the
  absolute bias, and :math:`1 - \mathrm{FDA}` for the forecast directional accuracy,
  and :math:`\tilde L` is (``normalize=``) its rank among the specifications mapped
  to :math:`[0, 1]` (``"rank"``, default: :math:`(r - 1)/(N - 1)`, 0 = best; scale
  free, so RMSFE and FDA can be mixed), its ratio to the same loss of the reference
  benchmark (``"relative"``) or the raw loss (``"none"``). Lower is better.
  Specifications with failed fits on more than ``max_missing`` of the evaluated
  forecasts, or with a missing criterion, get :math:`S_s = \infty`. Scores are
  recomputed on demand, so :meth:`SearchResults.table` can re-rank the same
  evaluations with other weights.
* **Checkpoint.** With ``checkpoint=path`` every evaluated specification is written to
  Parquet (CSV when no Parquet engine is installed) and a new run with the same
  settings resumes where the previous one stopped.
* **Covid robustness.** :meth:`SearchResults.covid_robustness` re-evaluates the best
  specifications with each treatment of the pandemic observations (none, impulse
  dummies, masking, outlier correction; see :data:`COVID_TREATMENTS`) on the target
  periods after the pandemic (``evaluate_from``) and ranks the (specification,
  treatment) pairs with the same score.

References
----------
Linzenich, J. & Meunier, B. (2024). Nowcasting made easier: a toolbox for economists.
ECB Working Paper No. 3004.

Bergstra, J. & Bengio, Y. (2012). Random search for hyper-parameter optimization.
*Journal of Machine Learning Research*, 13, 281-305.

Bańbura, M., Giannone, D., Modugno, M. & Reichlin, L. (2013). Now-casting and the
real-time data flow. In *Handbook of Economic Forecasting*, vol. 2A, 195-237.

Pesaran, M. H. & Timmermann, A. (1992). A simple nonparametric test of predictive
performance. *Journal of Business & Economic Statistics*, 10(4), 461-465.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import FrequencySpec, MixedFrequencyData, as_mixed_frequency_data
from nowcastbox.core.exceptions import DataQualityWarning
from nowcastbox.core.formula import resolve_target
from nowcastbox.evaluation.backtest import BacktestResults, PseudoRealTimeBacktest
from nowcastbox.selection._search_checkpoint import SearchCheckpoint, fingerprint
from nowcastbox.selection._search_space import ParameterSpace, spec_key
from nowcastbox.selection._search_treatments import (
    COVID_TREATMENTS,
    OutlierCorrection,
    Treatment,
    TreatmentPlan,
    resolve_treatments,
)
from nowcastbox.selection._specified import SpecifiedModel, configured_model, ranking_names
from nowcastbox.selection.blocks import ValidationSettings
from nowcastbox.selection.preselection import PreselectionResult, preselect

__all__ = [
    "COVID_TREATMENTS",
    "NORMALIZATIONS",
    "SCORE_METRICS",
    "SPECIAL_SETTINGS",
    "CovidRobustness",
    "OutlierCorrection",
    "ParameterSpace",
    "SearchResults",
    "SpecificationSearch",
    "SpecifiedModel",
    "TreatmentPlan",
    "weighted_score",
]

logger = get_logger(__name__)

SPECIAL_SETTINGS: tuple[str, ...] = ("start", "n_series")
"""Settings of a space that are not model parameters (sample start, funnel size)."""

SCORE_METRICS: tuple[str, ...] = ("rmsfe", "mse", "mae", "bias", "fda")
"""Metrics that can enter the score."""

NORMALIZATIONS: tuple[str, ...] = ("rank", "relative", "none")
"""Normalisations of the losses before they are weighted."""

_DEFAULT_METRICS = ("rmsfe", "mae", "fda")
_MODEL_NAME = "specification"
_NO_FORECAST = "no evaluable forecast"
_REF = "ref|"
_FORBIDDEN_BACKTEST = ("model", "data", "target", "model_name")
_PRESELECT_FIXED = ("data", "target", "as_of", "top")
_MAX_GRID = 100_000
_DEFAULT_VINTAGES = 12

RankingSpec = Sequence[str] | Mapping[str, Any] | PreselectionResult | str | None


# ====================================================================== scoring
def _loss(values: pd.Series, metric: str) -> pd.Series:
    """Loss of a metric (lower is better)."""
    if metric == "fda":
        return 1.0 - values
    if metric == "bias":
        return values.abs()
    return values


def _weight(value: Any, label: str) -> float:
    weight = float(value)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError(f"{label} must be a finite non-negative number, got {value!r}.")
    return weight


def _metric_weights(score: Mapping[str, float] | str | None) -> dict[str, float]:
    """Validated metric weights (``None`` = RMSFE only)."""
    if score is None:
        return {"rmsfe": 1.0}
    if isinstance(score, str):
        score = {score: 1.0}
    if not isinstance(score, Mapping) or not score:
        raise ValueError("score must be a non-empty mapping {metric: weight}.")
    out: dict[str, float] = {}
    for metric, value in score.items():
        if metric not in SCORE_METRICS:
            raise ValueError(f"Unknown score metric {metric!r}; available: {SCORE_METRICS}.")
        out[str(metric)] = _weight(value, f"score[{metric!r}]")
    if sum(out.values()) <= 0:
        raise ValueError("The score weights must have a positive sum.")
    return out


def _horizon_weights(horizon_weights: Mapping[Any, float] | None) -> dict[str, float] | None:
    if horizon_weights is None:
        return None
    if not isinstance(horizon_weights, Mapping) or not horizon_weights:
        raise ValueError("horizon_weights must be a non-empty mapping {horizon: weight}.")
    out = {str(h): _weight(w, f"horizon_weights[{h!r}]") for h, w in horizon_weights.items()}
    if sum(out.values()) <= 0:
        raise ValueError("The horizon weights must have a positive sum.")
    return out


def _horizons(columns: Sequence[Any], metric: str) -> list[str]:
    prefix = f"{metric}|"
    return [str(c)[len(prefix) :] for c in columns if str(c).startswith(prefix)]


def _equal_horizons(columns: Sequence[Any], metrics: dict[str, float]) -> dict[str, float]:
    metric = next(m for m, w in metrics.items() if w > 0)
    found = _horizons(columns, metric)
    if not found:
        raise ValueError(f"No {metric!r} criterion was computed.")
    return dict.fromkeys(found, 1.0)


def _warn_dropped(dropped: list[str], kept: Mapping[str, float]) -> None:
    warnings.warn(
        f"Horizons {dropped} were not evaluated (e.g. no backcast in the evaluation window); "
        f"the score uses the weights of {sorted(kept)} only.",
        DataQualityWarning,
        stacklevel=5,
    )


def _present_horizons(columns: Sequence[Any], metrics: dict[str, float]) -> set[str]:
    """Horizons with a criterion for at least one weighted metric."""
    return {h for m, w in metrics.items() if w > 0 for h in _horizons(columns, m)}


def _resolve_horizons(
    weights: dict[str, float] | None, columns: Sequence[Any], metrics: dict[str, float]
) -> dict[str, float]:
    """Horizon weights restricted to the evaluated horizons (equal weights by default)."""
    if weights is None:
        return _equal_horizons(columns, metrics)
    positive = {h: w for h, w in weights.items() if w > 0}
    present = _present_horizons(columns, metrics)
    kept = {h: w for h, w in positive.items() if h in present}
    if not kept:
        raise ValueError(
            f"None of the weighted horizons {sorted(weights)} was evaluated; available: "
            f"{sorted(present)}."
        )
    dropped = sorted(set(positive) - set(kept))
    if dropped:
        _warn_dropped(dropped, kept)
    return kept


def _rank01(loss: pd.Series) -> pd.Series:
    """Ranks of the finite losses mapped to [0, 1] (0 = best); NaN elsewhere."""
    finite = loss[np.isfinite(loss.to_numpy(dtype=float))]
    if len(finite) <= 1:
        return (finite * 0.0).reindex(loss.index)
    ranks = finite.rank(method="average")
    return ((ranks - 1.0) / (len(finite) - 1.0)).reindex(loss.index)


def _normalised(criteria: pd.DataFrame, metric: str, horizon: str, normalize: str) -> pd.Series:
    column = f"{metric}|{horizon}"
    if column not in criteria:
        available = sorted(c for c in map(str, criteria.columns) if "|" in c and c[:4] != _REF)
        raise ValueError(f"Criterion {column!r} was not computed; available: {available}.")
    loss = _loss(criteria[column].astype(float), metric)
    if normalize == "none":
        return loss
    if normalize == "relative":
        reference = _REF + column
        if reference not in criteria:
            raise ValueError(
                "normalize='relative' needs a reference benchmark: add one to "
                "backtest['benchmarks']."
            )
        return loss / _loss(criteria[reference].astype(float), metric)
    return _rank01(loss)


def weighted_score(
    criteria: pd.DataFrame,
    score: Mapping[str, float] | str | None = None,
    horizon_weights: Mapping[Any, float] | None = None,
    *,
    normalize: str = "rank",
) -> pd.Series:
    r"""Weighted score of specifications from their accuracy criteria (lower is better).

    Parameters
    ----------
    criteria : pandas.DataFrame
        One row per specification with columns ``"<metric>|<horizon>"`` (and
        ``"ref|<metric>|<horizon>"`` for the reference benchmark with
        ``normalize="relative"``); other columns are ignored.
    score : mapping or str, optional
        Metric weights, e.g. ``{"rmsfe": 0.7, "fda": 0.3}`` (default RMSFE only).
        Metrics: :data:`SCORE_METRICS`.
    horizon_weights : mapping, optional
        Horizon weights, e.g. ``{"nowcast": 0.5, "backcast": 0.25, "forecast": 0.25}``
        (default: equal weights over the evaluated horizons). Horizons with no
        criterion at all (not evaluated) are dropped with a warning and the remaining
        weights are used.
    normalize : {"rank", "relative", "none"}, default "rank"
        Normalisation of each loss before weighting (see the module notes).

    Returns
    -------
    pandas.Series
        Score of every row (``inf`` when a weighted criterion is missing).

    Raises
    ------
    ValueError
        On invalid weights, an unknown normalisation, a metric that was not computed
        or no evaluated horizon.

    Warns
    -----
    DataQualityWarning
        If some weighted horizons were not evaluated.

    Examples
    --------
    >>> import pandas as pd
    >>> criteria = pd.DataFrame(
    ...     {
    ...         "rmsfe|nowcast": [1.0, 2.0, 3.0],
    ...         "rmsfe|backcast": [0.5, 0.4, 0.6],
    ...         "fda|nowcast": [0.5, 0.9, 0.7],
    ...     }
    ... )
    >>> weighted_score(criteria, {"rmsfe": 1.0}, {"nowcast": 1.0}).tolist()
    [0.0, 0.5, 1.0]
    >>> weighted_score(criteria, {"rmsfe": 0.5, "fda": 0.5}, {"nowcast": 1}).tolist()
    [0.5, 0.25, 0.75]
    >>> weighted_score(criteria, "fda", {"nowcast": 1}, normalize="none").round(2).tolist()
    [0.5, 0.1, 0.3]
    """
    if normalize not in NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {NORMALIZATIONS}, got {normalize!r}.")
    metrics = _metric_weights(score)
    horizons = _resolve_horizons(_horizon_weights(horizon_weights), list(criteria.columns), metrics)
    total = pd.Series(0.0, index=criteria.index)
    for metric, w_metric in metrics.items():
        for horizon, w_horizon in horizons.items():
            if w_metric * w_horizon > 0:
                loss = _normalised(criteria, metric, horizon, normalize)
                total = total + w_metric * w_horizon * loss
    total = total / (sum(metrics.values()) * sum(horizons.values()))
    return total.where(np.isfinite(total.to_numpy(dtype=float)), np.inf).rename("score")


def _scores(
    evaluations: pd.DataFrame,
    valid: pd.Series,
    score: Mapping[str, float] | str | None,
    horizon_weights: Mapping[Any, float] | None,
    normalize: str,
) -> pd.Series:
    """Scores of the valid rows (``inf`` for the others)."""
    out = pd.Series(np.inf, index=evaluations.index, name="score")
    if bool(valid.any()):
        # criteria missing for every valid row (horizons never evaluated) are dropped
        criteria = evaluations[valid].dropna(axis=1, how="all")
        values = weighted_score(criteria, score, horizon_weights, normalize=normalize)
        out[valid] = values
    return out


def _valid(evaluations: pd.DataFrame, max_missing: float) -> pd.Series:
    """Rows evaluated successfully with enough forecasts."""
    if evaluations.empty:
        return pd.Series(False, index=evaluations.index)
    ok = evaluations["status"].astype(str) == "ok"
    coverage = pd.to_numeric(evaluations["coverage"], errors="coerce").fillna(0.0)
    return ok & (coverage >= 1.0 - max_missing - 1e-12)


# ====================================================================== evaluation
class _Compose:
    """Two preprocessing functions applied in turn (picklable)."""

    def __init__(
        self,
        first: Callable[[MixedFrequencyData], MixedFrequencyData],
        second: Callable[[MixedFrequencyData], MixedFrequencyData],
    ) -> None:
        self.first = first
        self.second = second

    def __call__(self, panel: MixedFrequencyData) -> MixedFrequencyData:
        return self.second(self.first(panel))

    def __repr__(self) -> str:
        return f"_Compose({self.first!r}, {self.second!r})"


def _compose(first: Any, second: Any) -> Any:
    if first is None:
        return second
    if second is None:
        return first
    return _Compose(first, second)


@dataclass(frozen=True)
class _Context:
    """Everything a worker needs to evaluate specifications (picklable)."""

    model: Any
    data: MixedFrequencyData
    target: str
    backtest: Mapping[str, Any]
    ranking: tuple[str, ...] | dict[str, Any] | None
    metrics: tuple[str, ...]
    horizon: str


@dataclass(frozen=True)
class _Task:
    """One backtest: a specification, possibly with a Covid treatment."""

    draw: int
    spec: Mapping[str, Any]
    key: str
    params: Mapping[str, Any] = field(default_factory=dict)
    preprocess: Callable[[MixedFrequencyData], MixedFrequencyData] | None = None
    periods: Any = None


def _split_spec(spec: Mapping[str, Any]) -> tuple[dict[str, Any], Any, int | None]:
    """Model parameters, sample start and funnel size of a specification."""
    params = {k: v for k, v in spec.items() if k not in SPECIAL_SETTINGS}
    n_series = spec.get("n_series")
    return params, spec.get("start"), None if n_series is None else int(n_series)


def _specified(
    model: Any,
    spec: Mapping[str, Any],
    ranking: tuple[str, ...] | dict[str, Any] | None,
    params: Mapping[str, Any] | None = None,
    preprocess: Any = None,
) -> SpecifiedModel:
    own, start, n_series = _split_spec(spec)
    return SpecifiedModel(
        model,
        {**own, **dict(params or {})},
        start=start,
        n_series=n_series,
        ranking=ranking if n_series is not None else None,
        preprocess=preprocess,
    )


def _criteria(result: BacktestResults, ctx: _Context, periods: Any) -> tuple[dict[str, Any], str]:
    """Coverage and metrics by horizon of the evaluated specification (and reference)."""
    try:
        sub = result if periods is None else next(iter(result.split_periods(periods).values()))
    except ValueError as err:
        return {"coverage": 0.0, "n_forecasts": 0}, str(err)
    frame = sub.forecasts
    rows = frame[(frame["model"] == _MODEL_NAME) & frame["actual"].notna()]
    finite = np.isfinite(rows["forecast"].to_numpy(dtype=float))
    out: dict[str, Any] = {
        "coverage": float(finite.mean()) if finite.size else 0.0,
        "n_forecasts": int(finite.sum()),
    }
    if not finite.any():
        return out, _NO_FORECAST
    models = [_MODEL_NAME] + ([sub.reference] if sub.reference else [])
    table = sub.metrics(ctx.horizon, ctx.metrics, models=models)
    for metric in ctx.metrics:
        block = table[metric]
        for prefix, model in zip(("", _REF), models, strict=False):
            values = block[model].to_numpy(dtype=float)
            out.update(
                {
                    f"{prefix}{metric}|{h}": float(v)
                    for h, v in zip(block.index, values, strict=True)
                }
            )
    return out, ""


def _status(criteria: Mapping[str, Any], error: str, result: BacktestResults) -> dict[str, Any]:
    failures = list(result.info.get("failures", []))
    ok = any("|" in k and not k.startswith(_REF) and np.isfinite(v) for k, v in criteria.items())
    message = failures[0] if failures and error in ("", _NO_FORECAST) else error
    return {
        "status": "ok" if ok else "failed",
        "error": message,
        "n_failed_fits": len(failures),
    }


def _run_task(ctx: _Context, task: _Task) -> tuple[dict[str, Any], BacktestResults]:
    """Backtest one specification and summarise its accuracy."""
    started = time.perf_counter()
    kwargs = dict(ctx.backtest)
    kwargs["preprocess"] = _compose(kwargs.get("preprocess"), task.preprocess)
    model = _specified(ctx.model, task.spec, ctx.ranking, task.params)
    backtest = PseudoRealTimeBacktest(
        model=model, data=ctx.data, target=ctx.target, model_name=_MODEL_NAME, **kwargs
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = backtest.run()
        criteria, error = _criteria(result, ctx, task.periods)
    record = {
        "draw": task.draw,
        "spec_key": task.key,
        **_status(criteria, error, result),
        "n_warnings": len(caught),
        "seconds": time.perf_counter() - started,
        **criteria,
    }
    return record, result


def _parallel(
    func: Callable[[Any], Any], items: Sequence[Any], n_jobs: int | None
) -> Iterator[Any]:
    """Results of ``func`` over ``items`` in order, as they complete (joblib if parallel)."""
    if n_jobs in (None, 1) or len(items) <= 1:
        yield from map(func, items)
        return
    try:
        from joblib import Parallel, delayed
    except ImportError:  # pragma: no cover - joblib is a dependency
        yield from map(func, items)
        return
    try:
        runner = Parallel(n_jobs=n_jobs, return_as="generator")
    except TypeError:  # pragma: no cover - joblib < 1.3
        runner = Parallel(n_jobs=n_jobs)
    yield from runner(delayed(func)(item) for item in items)


# ====================================================================== setup helpers
def _template(model: Any) -> Any:
    if isinstance(model, type):
        model = model()
    if not callable(getattr(model, "fit", None)):
        raise TypeError(f"model must be a nowcaster (with fit), got {type(model).__name__}.")
    return model


def _model_param_names(model: Any) -> set[str]:
    get_params = getattr(model, "get_params", None)
    params = get_params(deep=False) if callable(get_params) else None
    return set(params) if isinstance(params, Mapping) else set()


def _check_value(model: Any, name: str, value: Any) -> None:
    try:
        estimator = configured_model(model, {name: value})
        validate = getattr(estimator, "_validate_params", None)
        if callable(validate):
            validate()
    except (ValueError, TypeError) as err:
        raise ValueError(f"Invalid value {value!r} for {name!r} in the space: {err}") from err


def _check_model_settings(space: ParameterSpace, model: Any) -> None:
    names = _model_param_names(model)
    unknown = [n for n in space.names if n not in SPECIAL_SETTINGS and n not in names]
    if unknown:
        raise ValueError(
            f"Space settings {unknown} are neither parameters of {type(model).__name__} "
            f"({sorted(names)}) nor special settings {SPECIAL_SETTINGS}."
        )
    for name in space.names:
        if name not in SPECIAL_SETTINGS:
            for value in space.endpoints(name):
                _check_value(model, name, value)


def _check_start(space: ParameterSpace, panel: MixedFrequencyData) -> None:
    freq = panel.base_frequency.pandas_freq
    for value in space.values("start"):
        if value is None:
            continue
        try:
            first = pd.Period(value, freq=freq)
        except (ValueError, TypeError) as err:
            raise ValueError(f"Invalid sample start {value!r} in the space: {err}") from err
        if first > panel.end:
            raise ValueError(f"Sample start {value!r} is after the end of the data.")


def _check_n_series(space: ParameterSpace, available: int | None) -> None:
    if available is None:
        raise ValueError("'n_series' in the space needs a ranking (funnel strategy).")
    values = space.endpoints("n_series")
    if not all(isinstance(v, int | np.integer) and not isinstance(v, bool) for v in values):
        raise ValueError("'n_series' values must be integers.")
    if min(values) < 1 or max(values) > available:
        raise ValueError(
            f"'n_series' must lie in [1, {available}] (series available in the ranking), "
            f"got {list(values)}."
        )


def _warn_look_ahead(ranking: PreselectionResult, first_vintage: pd.Timestamp) -> None:
    if ranking.as_of is not None and pd.Timestamp(ranking.as_of) <= first_vintage:
        return
    warnings.warn(
        f"The ranking was computed with data released after the first vintage "
        f"({first_vintage.date()}; as_of={ranking.as_of}): the funnel looks ahead. Use "
        "ranking='preselect' (or a mapping of preselect options) to recompute it at each "
        "vintage, or preselect(..., as_of=<first vintage>).",
        DataQualityWarning,
        stacklevel=4,
    )


def _preselect_options(options: Mapping[str, Any]) -> dict[str, Any]:
    allowed = set(inspect.signature(preselect).parameters) - set(_PRESELECT_FIXED)
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f"Invalid preselect options {unknown}; allowed: {sorted(allowed)}.")
    return dict(options)


def _resolve_ranking(
    ranking: RankingSpec, panel: MixedFrequencyData, target: str, first_vintage: pd.Timestamp
) -> tuple[str, ...] | dict[str, Any] | None:
    """Static ranking (names) or per-vintage pre-selection options."""
    if ranking is None:
        return None
    if isinstance(ranking, str):
        if ranking != "preselect":
            raise ValueError(f"ranking must be 'preselect', options or names, got {ranking!r}.")
        return {}
    if isinstance(ranking, Mapping):
        return _preselect_options(ranking)
    return _static_ranking(ranking, panel, target, first_vintage)


def _static_ranking(
    ranking: Sequence[str] | PreselectionResult,
    panel: MixedFrequencyData,
    target: str,
    first_vintage: pd.Timestamp,
) -> tuple[str, ...]:
    if isinstance(ranking, PreselectionResult):
        _warn_look_ahead(ranking, first_vintage)
    names = tuple(n for n in ranking_names(ranking) if n != target)
    unknown = [n for n in names if n not in panel.columns]
    if unknown:
        raise ValueError(f"Ranked series {unknown} are not in the data.")
    return names


def _available(
    ranking: tuple[str, ...] | dict[str, Any] | None, panel: MixedFrequencyData
) -> int | None:
    if ranking is None:
        return None
    return panel.n_series - 1 if isinstance(ranking, dict) else len(ranking)


def _backtest_kwargs(
    backtest: Mapping[str, Any] | None, panel: MixedFrequencyData, target: str
) -> dict[str, Any]:
    """Validated keyword arguments of the backtests (default vintage window filled in)."""
    kwargs = dict(backtest or {})
    forbidden = sorted(set(kwargs) & set(_FORBIDDEN_BACKTEST))
    if forbidden:
        raise ValueError(f"backtest cannot set {forbidden}: the search provides them.")
    n_vintages = int(kwargs.pop("n_vintages", _DEFAULT_VINTAGES))
    if kwargs.get("start") is None or kwargs.get("end") is None:
        settings = ValidationSettings(
            start=kwargs.get("start"), end=kwargs.get("end"), n_vintages=n_vintages
        )
        kwargs["start"], kwargs["end"] = settings.dates(panel, target)
    kwargs.setdefault("errors", "warn")
    try:
        PseudoRealTimeBacktest(data=panel, target=target, **kwargs)
    except TypeError as err:
        raise ValueError(f"Invalid backtest settings: {err}") from err
    return kwargs


def _metrics(score: dict[str, float], metrics: Sequence[str]) -> tuple[str, ...]:
    chosen = list(dict.fromkeys([*score, *metrics]))
    unknown = [m for m in chosen if m not in SCORE_METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}; available: {SCORE_METRICS}.")
    return tuple(chosen)


def _as_periods(periods: Any) -> Any:
    """A single sub-period of target periods for :meth:`BacktestResults.split_periods`."""
    if periods is None:
        return None
    if isinstance(periods, str):
        if periods.lower() != "ex-covid":
            raise ValueError("periods must be 'ex-covid', a (first, last) pair or a list.")
        return "ex-covid"
    if isinstance(periods, Mapping):
        if len(periods) != 1:
            raise ValueError("periods must define exactly one sub-period.")
        return dict(periods)
    return {"evaluation": periods}


def _check_int(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


# ====================================================================== search
@dataclass(frozen=True)
class _Setup:
    """Validated run of a search."""

    context: _Context
    space: ParameterSpace
    tasks: list[_Task]
    info: dict[str, Any]
    score: dict[str, float]
    horizon_weights: dict[str, float] | None
    checkpoint: SearchCheckpoint | None
    periods: Any


class SpecificationSearch:
    """Random or grid search of model specifications scored in pseudo real time.

    Parameters
    ----------
    model : nowcaster or nowcaster class
        Model template, e.g. :class:`~nowcastbox.models.MixedFreqDFM` (a class is
        instantiated with its defaults).
    data : MixedFrequencyData or pandas.DataFrame
        Final dataset (the backtest builds the pseudo real-time vintages).
    target : str
        Target series, or a formula restricting the candidate indicators
        (``"gdp ~ . - x3"``).
    space : mapping or ParameterSpace
        Searched settings (see :class:`ParameterSpace`): parameters of ``model`` and
        the special settings ``"start"`` (first base period of the estimation sample)
        and ``"n_series"`` (number of indicators taken from ``ranking``).
    n_draws : int or None, default 100
        Number of random draws; ``None`` evaluates the exhaustive grid.
    ranking : PreselectionResult, sequence of str, "preselect" or mapping, optional
        Ranking of the funnel strategy (required with ``"n_series"``). A
        :class:`~nowcastbox.selection.PreselectionResult` or a list of names is fixed
        (a look-ahead warning is issued when the pre-selection used data released
        after the first vintage); ``"preselect"`` or a mapping of
        :func:`~nowcastbox.selection.preselect` options recomputes it on the
        information set of every re-estimation vintage.
    backtest : mapping, optional
        Keyword arguments of :class:`~nowcastbox.evaluation.PseudoRealTimeBacktest`
        (``start``, ``end``, ``step``, ``delay``/``calendar``, ``refit_every``,
        ``target_offsets``, ``benchmarks``, ``window``, ``fit_kwargs``,
        ``preprocess``...), plus ``n_vintages`` (default 12) to place the vintages
        before the last observed target period when ``start``/``end`` are missing.
        ``errors`` defaults to ``"warn"`` (failed fits count as missing forecasts).
    score : mapping or str, optional
        Metric weights of the score (default ``{"rmsfe": 1}``); metrics in
        :data:`SCORE_METRICS`.
    horizon_weights : mapping, optional
        Weights of the horizons (values of the ``horizon`` column, e.g. ``"backcast"``,
        ``"nowcast"``, ``"forecast"``); default: equal weights.
    random_state : int or numpy.random.SeedSequence, optional
        Seed of the draws.
    n_jobs : int, optional
        Specifications evaluated in parallel (joblib); ``None``/1 = sequential, -1 = all
        cores.
    checkpoint : str or pathlib.Path, optional
        Parquet file where evaluated specifications are stored; an existing file of a
        search with the same settings is resumed.
    normalize : {"rank", "relative", "none"}, default "rank"
        Normalisation of the losses in the score (see the module notes);
        ``"relative"`` needs a benchmark in ``backtest["benchmarks"]``.
    periods : (first, last), list of pairs, mapping or "ex-covid", optional
        Target periods on which the criteria are computed (default: all).
    horizon : str, default "kind"
        Column of the forecast table defining the horizons (``"kind"``,
        ``"months_to_end"``, ``"offset"``...).
    metrics : sequence of str, default ("rmsfe", "mae", "fda")
        Metrics stored for every specification besides those of ``score`` (so the
        results can be re-scored).
    max_missing : float, default 0.1
        Largest share of missing forecasts of a valid specification.
    frequency : frequency specification, optional
        Per-series frequencies for DataFrame input.

    Raises
    ------
    ValueError
        If a setting is invalid (raised by :meth:`run`).

    See Also
    --------
    nowcastbox.selection.preselect : Ranking of the indicators (funnel).
    nowcastbox.selection.select_blocks : Greedy block selection.

    Examples
    --------
    >>> import warnings
    >>> import nowcastbox as nb
    >>> from nowcastbox.benchmarks import AR
    >>> data = nb.load_simulated_dfm().data
    >>> search = SpecificationSearch(
    ...     model=nb.TwoStepDFM,
    ...     data=data,
    ...     target="gdp",
    ...     space={"n_factors": [1, 2], "n_series": (4, 12), "start": ["2000-01", "2006-01"]},
    ...     n_draws=3,
    ...     ranking="preselect",
    ...     backtest={"start": "2018-01-15", "end": "2019-03-15", "benchmarks": [AR(p=1)]},
    ...     score={"rmsfe": 0.7, "fda": 0.3},
    ...     horizon_weights={"nowcast": 0.5, "backcast": 0.25, "forecast": 0.25},
    ...     random_state=0,
    ... )
    >>> out = search.run()
    >>> out.table()[["rank", "draw", "n_factors", "n_series", "start", "status"]].shape
    (3, 6)
    >>> bool(out.table()["score"].is_monotonic_increasing)
    True
    >>> type(out.best_model()).__name__
    'SpecifiedModel'
    """

    def __init__(
        self,
        model: Any,
        data: MixedFrequencyData | pd.DataFrame,
        target: str,
        space: Mapping[str, Any] | ParameterSpace,
        n_draws: int | None = 100,
        ranking: RankingSpec = None,
        backtest: Mapping[str, Any] | None = None,
        score: Mapping[str, float] | str | None = None,
        horizon_weights: Mapping[Any, float] | None = None,
        random_state: int | np.random.SeedSequence | None = None,
        n_jobs: int | None = None,
        checkpoint: str | Path | None = None,
        *,
        normalize: str = "rank",
        periods: Any = None,
        horizon: str = "kind",
        metrics: Sequence[str] = _DEFAULT_METRICS,
        max_missing: float = 0.1,
        frequency: FrequencySpec | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.target = target
        self.space = space
        self.n_draws = n_draws
        self.ranking = ranking
        self.backtest = backtest
        self.score = score
        self.horizon_weights = horizon_weights
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.checkpoint = checkpoint
        self.normalize = normalize
        self.periods = periods
        self.horizon = horizon
        self.metrics = metrics
        self.max_missing = max_missing
        self.frequency = frequency

    # ------------------------------------------------------------------ setup
    def _check_options(self) -> None:
        if self.normalize not in NORMALIZATIONS:
            raise ValueError(f"normalize must be one of {NORMALIZATIONS}, got {self.normalize!r}.")
        if not 0.0 <= float(self.max_missing) < 1.0:
            raise ValueError(f"max_missing must lie in [0, 1), got {self.max_missing!r}.")
        if not isinstance(self.horizon, str) or not self.horizon:
            raise ValueError(f"horizon must be a column name, got {self.horizon!r}.")

    def _panel(self) -> tuple[MixedFrequencyData, str]:
        panel = as_mixed_frequency_data(self.data, self.frequency)
        target, regressors = resolve_target(self.target, panel.columns)
        keep = {target, *regressors}
        if len(keep) != panel.n_series:
            panel = panel.select([c for c in panel.columns if c in keep])
        return panel, target

    def _space(
        self, model: Any, panel: MixedFrequencyData, available: int | None
    ) -> ParameterSpace:
        space = self.space if isinstance(self.space, ParameterSpace) else ParameterSpace(self.space)
        _check_model_settings(space, model)
        if "start" in space:
            _check_start(space, panel)
        if "n_series" in space:
            _check_n_series(space, available)
        return space

    def _tasks(self, space: ParameterSpace) -> tuple[list[_Task], dict[str, Any]]:
        if self.n_draws is None:
            if space.size > _MAX_GRID:
                raise ValueError(
                    f"The grid has {space.size} specifications (> {_MAX_GRID}); use n_draws."
                )
            specs, entropy = space.grid(), None
        else:
            n = _check_int(self.n_draws, "n_draws", 1)
            seed = (
                self.random_state
                if isinstance(self.random_state, np.random.SeedSequence)
                else np.random.SeedSequence(self.random_state)
            )
            specs, entropy = space.draw(n, seed), seed.entropy
        seen: set[str] = set()
        tasks = []
        for draw, spec in enumerate(specs):
            key = spec_key(spec)
            if key not in seen:
                seen.add(key)
                tasks.append(_Task(draw, spec, key))
        info = {
            "grid": self.n_draws is None,
            "n_draws": len(specs),
            "n_unique": len(tasks),
            "n_duplicates": len(specs) - len(tasks),
            "space_size": space.size,
            "entropy": entropy,
        }
        return tasks, info

    def _setup(self) -> _Setup:
        self._check_options()
        model = _template(self.model)
        panel, target = self._panel()
        kwargs = _backtest_kwargs(self.backtest, panel, target)
        ranking = _resolve_ranking(self.ranking, panel, target, pd.Timestamp(kwargs["start"]))
        space = self._space(model, panel, _available(ranking, panel))
        score = _metric_weights(self.score)
        horizon_weights = _horizon_weights(self.horizon_weights)
        context = _Context(
            model=model,
            data=panel,
            target=target,
            backtest=kwargs,
            ranking=ranking,
            metrics=_metrics(score, self.metrics),
            horizon=self.horizon,
        )
        periods = _as_periods(self.periods)
        tasks, info = self._tasks(space)
        digest = fingerprint(
            {
                "model": model,
                "target": target,
                "backtest": kwargs,
                "ranking": ranking,
                "metrics": context.metrics,
                "horizon": self.horizon,
                "periods": periods,
                "metadata": [repr(panel.metadata[c]) for c in panel.columns],
            },
            panel.data,
        )
        info["fingerprint"] = digest
        store = None if self.checkpoint is None else SearchCheckpoint(self.checkpoint, digest)
        return _Setup(context, space, tasks, info, score, horizon_weights, store, periods)

    # ------------------------------------------------------------------ run
    def run(self) -> SearchResults:
        """Evaluate the specifications and rank them.

        Returns
        -------
        SearchResults
            Evaluations, scores and the tools to pick a model and test its
            robustness to the Covid treatment.

        Raises
        ------
        ValueError
            If a setting is invalid, or the checkpoint belongs to a search with
            other settings.

        Warns
        -----
        DataQualityWarning
            If a fixed ranking was computed with data released after the first
            vintage (look-ahead).

        Examples
        --------
        >>> # see the class docstring
        """
        started = time.perf_counter()
        setup = self._setup()
        done = {} if setup.checkpoint is None else setup.checkpoint.load()
        pending = [t for t in setup.tasks if t.key not in done]
        logger.info(
            "Specification search: %d specifications (%d from the checkpoint).",
            len(setup.tasks),
            len(setup.tasks) - len(pending),
        )
        tasks = [dataclasses.replace(t, periods=setup.periods) for t in pending]
        records = dict(done)
        backtests: dict[int, BacktestResults] = {}
        work = functools.partial(_run_task, setup.context)
        for record, result in _parallel(work, tasks, self.n_jobs):
            records[str(record["spec_key"])] = record
            backtests[int(record["draw"])] = result
            logger.info("Draw %d: %s.", record["draw"], record["status"])
            if setup.checkpoint is not None:
                setup.checkpoint.save(list(records.values()))
        info = {
            **setup.info,
            "n_resumed": len(setup.tasks) - len(pending),
            "n_evaluated": len(pending),
            "seconds": time.perf_counter() - started,
            "checkpoint": None if setup.checkpoint is None else str(setup.checkpoint.path),
            "periods": setup.periods,
            "n_jobs": self.n_jobs,
        }
        return SearchResults(
            target=setup.context.target,
            space=setup.space,
            specs={t.draw: dict(t.spec) for t in setup.tasks},
            evaluations=_evaluations(setup.tasks, records, setup.space),
            score_weights=setup.score,
            horizon_weights=setup.horizon_weights,
            normalize=self.normalize,
            max_missing=float(self.max_missing),
            backtests=backtests,
            info=info,
            context=setup.context,
        )


_RECORD_FIRST = ("status", "coverage", "n_forecasts")


def _evaluations(
    tasks: Sequence[_Task], records: Mapping[str, Mapping[str, Any]], space: ParameterSpace
) -> pd.DataFrame:
    """One row per specification: settings, run status and criteria (index ``draw``)."""
    rows = []
    for task in tasks:
        record = {k: v for k, v in records[task.key].items() if k not in ("draw", "spec_key")}
        rows.append({"draw": task.draw, **dict(task.spec), **record, "spec_key": task.key})
    frame = pd.DataFrame.from_records(rows)
    return frame[_column_order(list(map(str, frame.columns)), space)].set_index("draw")


def _column_order(columns: list[str], space: ParameterSpace) -> list[str]:
    """Draw, settings, status, model criteria, reference criteria, the rest."""
    own = sorted(c for c in columns if "|" in c and not c.startswith(_REF))
    reference = sorted(c for c in columns if c.startswith(_REF))
    head = ["draw", *space.names, *_RECORD_FIRST]
    rest = [c for c in columns if c not in head and c not in own and c not in reference]
    return [*head, *own, *reference, *rest]


def _ordered(frame: pd.DataFrame, scores: np.ndarray, first: Sequence[str]) -> pd.DataFrame:
    """Rows sorted by score with ``rank`` and ``score`` columns in front."""
    out = frame.assign(score=scores).sort_values("score", kind="mergesort")
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    columns = ["rank", *first, "score"]
    return out[[*columns, *[c for c in out.columns if c not in columns]]]


# ====================================================================== results
@dataclass(frozen=True)
class SearchResults:
    """Outcome of :meth:`SpecificationSearch.run`.

    Attributes
    ----------
    target : str
        Target series.
    space : ParameterSpace
        Searched settings.
    specs : dict
        Draw number to specification (unique specifications only).
    evaluations : pandas.DataFrame
        One row per specification (index ``draw``): its settings, ``status``
        (``"ok"`` or ``"failed"``), ``coverage`` (share of evaluated forecasts
        available), ``n_forecasts``, the criteria ``"<metric>|<horizon>"`` (and
        ``"ref|..."`` for the reference benchmark), ``error``, ``n_failed_fits``,
        ``n_warnings``, ``seconds`` and ``spec_key``.
    score_weights : dict
        Metric weights of the default score.
    horizon_weights : dict or None
        Horizon weights of the default score (``None``: equal).
    normalize : str
        Default normalisation of the losses.
    max_missing : float
        Largest share of missing forecasts of a valid specification.
    backtests : dict
        Draw number to :class:`~nowcastbox.evaluation.BacktestResults`, for the
        specifications evaluated in this run (not those read from a checkpoint).
    info : dict
        Run information (number of draws, duplicates, resumed evaluations, seed
        entropy, fingerprint, timing...).
    context : object
        Settings needed by :meth:`covid_robustness` (``None`` for results assembled
        by hand).

    Examples
    --------
    >>> out.table().columns[:4].tolist()  # doctest: +SKIP
    ['rank', 'draw', 'score', 'n_factors']
    """

    target: str
    space: ParameterSpace
    specs: dict[int, dict[str, Any]]
    evaluations: pd.DataFrame
    score_weights: dict[str, float] = field(default_factory=lambda: {"rmsfe": 1.0})
    horizon_weights: dict[str, float] | None = None
    normalize: str = "rank"
    max_missing: float = 0.1
    backtests: dict[int, BacktestResults] = field(default_factory=dict, repr=False)
    info: dict[str, Any] = field(default_factory=dict)
    context: Any = field(default=None, repr=False, compare=False)

    def scores(
        self,
        score: Mapping[str, float] | str | None = None,
        horizon_weights: Mapping[Any, float] | None = None,
        normalize: str | None = None,
    ) -> pd.Series:
        """Score of every specification (lower is better; ``inf`` when invalid).

        Parameters
        ----------
        score, horizon_weights, normalize : optional
            Override the weights and normalisation of the search (see
            :func:`weighted_score`); the stored criteria are re-weighted, nothing is
            re-estimated.

        Returns
        -------
        pandas.Series
            Index ``draw``.

        Raises
        ------
        ValueError
            On invalid weights or a criterion that was not computed.

        Examples
        --------
        >>> out.scores({"fda": 1.0}, {"nowcast": 1.0})  # doctest: +SKIP
        """
        return _scores(
            self.evaluations,
            _valid(self.evaluations, self.max_missing),
            self.score_weights if score is None else score,
            self.horizon_weights if horizon_weights is None else horizon_weights,
            self.normalize if normalize is None else normalize,
        )

    def table(
        self,
        score: Mapping[str, float] | str | None = None,
        horizon_weights: Mapping[Any, float] | None = None,
        normalize: str | None = None,
        *,
        top: int | None = None,
    ) -> pd.DataFrame:
        """Specifications sorted by score (best first).

        Parameters
        ----------
        score, horizon_weights, normalize : optional
            Override the score settings (see :meth:`scores`).
        top : int, optional
            Keep the first ``top`` rows.

        Returns
        -------
        pandas.DataFrame
            Columns ``rank``, ``draw``, ``score``, the settings, ``status``,
            ``coverage``, ``n_forecasts``, the criteria and the run information.

        Raises
        ------
        ValueError
            On invalid score settings.

        Examples
        --------
        >>> out.table(top=5)  # doctest: +SKIP
        """
        scores = self.scores(score, horizon_weights, normalize)
        table = _ordered(self.evaluations.reset_index(), scores.to_numpy(), ["draw"])
        table = table.reset_index(drop=True)
        return table if top is None else table.head(int(top))

    def best_spec(self, rank: int = 1) -> dict[str, Any]:
        """Settings of the specification at a given rank.

        Parameters
        ----------
        rank : int, default 1
            Position in :meth:`table` (1 = best).

        Returns
        -------
        dict
            The specification.

        Raises
        ------
        ValueError
            If ``rank`` is out of range or that specification has no finite score.

        Examples
        --------
        >>> out.best_spec()  # doctest: +SKIP
        {'n_factors': 1, 'n_series': 8, 'start': '2006-01'}
        """
        table = self.table()
        position = _check_int(rank, "rank", 1)
        if position > len(table) or not np.isfinite(table["score"].iloc[position - 1]):
            raise ValueError(f"No valid specification at rank {rank}.")
        return dict(self.specs[int(table["draw"].iloc[position - 1])])

    def best_model(self, rank: int = 1) -> SpecifiedModel:
        """Unfitted model of the specification at a given rank.

        Parameters
        ----------
        rank : int, default 1
            Position in :meth:`table` (1 = best).

        Returns
        -------
        SpecifiedModel
            Model with the specification's parameters, sample start and funnel
            (``fit(data, target)`` re-ranks the indicators on ``data`` when the search
            used ``ranking="preselect"``).

        Raises
        ------
        ValueError
            If there is no valid specification at that rank or the results carry no
            search context.

        Examples
        --------
        >>> out.best_model().fit(data, "gdp")  # doctest: +SKIP
        """
        context = self._context()
        spec = self.best_spec(rank)
        return _specified(
            context.model, spec, context.ranking, preprocess=context.backtest.get("preprocess")
        )

    def _context(self) -> _Context:
        if not isinstance(self.context, _Context):
            raise ValueError("These results carry no search context (run a SpecificationSearch).")
        return self.context

    def covid_robustness(
        self,
        top: int = 5,
        treatments: Sequence[str] | Mapping[str, Treatment | str] = (
            "none",
            "dummy",
            "mask",
            "outliers",
        ),
        evaluate_from: Any = None,
        *,
        backtest: Mapping[str, Any] | None = None,
        score: Mapping[str, float] | str | None = None,
        horizon_weights: Mapping[Any, float] | None = None,
        normalize: str | None = None,
        n_jobs: int | None = None,
    ) -> CovidRobustness:
        """Re-evaluate the best specifications with each treatment of the pandemic.

        Parameters
        ----------
        top : int, default 5
            Number of best specifications (with a finite score) re-evaluated.
        treatments : sequence of str or mapping, default ("none", "dummy", "mask", "outliers")
            Treatments (:data:`COVID_TREATMENTS`) or ``{label: name or function}``.
            Treatments a model cannot apply are skipped with a note.
        evaluate_from : period-like, optional
            First target period of the evaluation (e.g. ``"2022Q1"``); default: the
            periods of the search.
        backtest : mapping, optional
            Backtest settings overriding those of the search (e.g. a later ``end``).
        score, horizon_weights, normalize : optional
            Score settings (default: those of the search). Horizons that cannot be
            evaluated in the window (e.g. backcasts of periods before
            ``evaluate_from``) are dropped from the horizon weights, with a warning.
        n_jobs : int, optional
            Parallel jobs (default: those of the search).

        Returns
        -------
        CovidRobustness
            One row per (specification, treatment).

        Raises
        ------
        ValueError
            If there is no search context, no valid specification, or nothing could be
            evaluated in the evaluation window.

        Warns
        -----
        DataQualityWarning
            If some weighted horizons were not evaluated in the window.

        Examples
        --------
        >>> rob = out.covid_robustness(top=2, evaluate_from="2019Q1")  # doctest: +SKIP
        >>> rob.best()  # doctest: +SKIP
        """
        context = self._robustness_context(backtest)
        selected = self._valid_top(top)
        periods = self.info.get("periods")
        if evaluate_from is not None:
            periods = {f"from {evaluate_from}": (str(evaluate_from), None)}
        plan = _robustness_plan(self, context, selected, resolve_treatments(treatments), periods)
        jobs = self.info.get("n_jobs") if n_jobs is None else n_jobs
        evaluations, backtests = _robustness_run(context, plan, jobs)
        valid = _valid(evaluations, self.max_missing)
        if not bool(valid.any()):
            reasons = evaluations["error"].dropna().astype(str)
            first = next((r for r in reasons if r), "unknown")
            raise ValueError(f"No (specification, treatment) could be evaluated: {first}")
        metrics = self.score_weights if score is None else _metric_weights(score)
        horizons = _resolve_horizons(
            self.horizon_weights if horizon_weights is None else _horizon_weights(horizon_weights),
            list(evaluations.loc[valid].dropna(axis=1, how="all").columns),
            metrics,
        )
        return CovidRobustness(
            target=self.target,
            treatments=tuple(dict.fromkeys(evaluations["treatment"])),
            periods=periods,
            evaluations=evaluations,
            models={key: model for key, (_, model, _) in plan.items()},
            score_weights=metrics,
            horizon_weights=horizons,
            normalize=self.normalize if normalize is None else normalize,
            max_missing=self.max_missing,
            backtests=backtests,
        )

    def _robustness_context(self, backtest: Mapping[str, Any] | None) -> _Context:
        context = self._context()
        if not backtest:
            return context
        merged = {**context.backtest, **dict(backtest)}
        kwargs = _backtest_kwargs(merged, context.data, context.target)
        return dataclasses.replace(context, backtest=kwargs)

    def _valid_top(self, top: int) -> pd.DataFrame:
        n = _check_int(top, "top", 1)
        table = self.table()
        table = table[np.isfinite(table["score"].to_numpy(dtype=float))]
        if table.empty:
            raise ValueError("No specification has a finite score.")
        return table.head(n)

    def summary(self, top: int = 10) -> str:
        """Plain-text summary: settings, counts and the best specifications.

        Parameters
        ----------
        top : int, default 10
            Number of specifications listed.

        Returns
        -------
        str
            Summary.

        Examples
        --------
        >>> "Specification search" in out.summary()  # doctest: +SKIP
        True
        """
        table = self.table(top=top)
        n_ok = int(_valid(self.evaluations, self.max_missing).sum())
        columns = ["rank", "draw", "score", *self.space.names, "coverage"]
        title = f"Specification search for {self.target}"
        lines = [
            title,
            "=" * len(title),
            f"space: {self.space!r}",
            f"specifications: {len(self.evaluations)} (valid {n_ok}; "
            f"duplicates {self.info.get('n_duplicates', 0)}; "
            f"resumed {self.info.get('n_resumed', 0)})",
            f"score: {self.score_weights}   horizons: {self.horizon_weights or 'equal'}   "
            f"normalize: {self.normalize}",
            "",
            table[columns].to_string(index=False, float_format=lambda v: f"{v:.4g}"),
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ====================================================================== robustness
_PlanItem = tuple[dict[str, Any], SpecifiedModel, _Task | None]


def _robustness_plan(
    results: SearchResults,
    context: _Context,
    selected: pd.DataFrame,
    treatments: Mapping[str, Treatment],
    periods: Any,
) -> dict[tuple[int, str], _PlanItem]:
    """Metadata, final model and backtest task of every (specification, treatment)."""
    user = context.backtest.get("preprocess")
    out: dict[tuple[int, str], _PlanItem] = {}
    for draw, search_rank in zip(selected["draw"], selected["rank"], strict=True):
        spec = results.specs[int(draw)]
        params, _, _ = _split_spec(spec)
        inner = configured_model(context.model, params)
        for label, treatment in treatments.items():
            plan: TreatmentPlan = treatment(inner, context.target)
            meta = {
                "draw": int(draw),
                "search_rank": int(search_rank),
                "treatment": label,
                "supported": bool(plan.supported),
                "note": plan.note,
            }
            model = _specified(
                context.model,
                spec,
                context.ranking,
                plan.params,
                _compose(user, plan.preprocess),
            )
            task = None
            if plan.supported:
                task = _Task(int(draw), spec, spec_key(spec), plan.params, plan.preprocess, periods)
            out[(int(draw), label)] = (meta, model, task)
    return out


def _robustness_run(
    context: _Context, plan: Mapping[tuple[int, str], _PlanItem], n_jobs: int | None
) -> tuple[pd.DataFrame, dict[tuple[int, str], BacktestResults]]:
    keys = [key for key, (_, _, task) in plan.items() if task is not None]
    tasks = [plan[key][2] for key in keys]
    work = functools.partial(_run_task, context)
    outputs = dict(zip(keys, _parallel(work, tasks, n_jobs), strict=True))
    rows = []
    for key, (meta, _, _) in plan.items():
        if key in outputs:
            record = {k: v for k, v in outputs[key][0].items() if k not in ("draw", "spec_key")}
        else:
            record = {"status": "skipped", "coverage": np.nan, "n_forecasts": 0, "error": ""}
        rows.append({**meta, **record})
    frame = pd.DataFrame.from_records(rows)
    return frame, {key: out[1] for key, out in outputs.items()}


@dataclass(frozen=True)
class CovidRobustness:
    """Accuracy of the best specifications under each treatment of the pandemic.

    Attributes
    ----------
    target : str
        Target series.
    treatments : tuple of str
        Treatment labels, in order.
    periods : object
        Target periods of the evaluation (sub-period of
        :meth:`~nowcastbox.evaluation.BacktestResults.split_periods`).
    evaluations : pandas.DataFrame
        One row per (specification, treatment): ``draw``, ``search_rank``,
        ``treatment``, ``supported``, ``note``, ``status`` (``"ok"``, ``"failed"``
        or ``"skipped"``), ``coverage``, criteria and run information.
    models : dict
        ``(draw, treatment)`` to the unfitted :class:`SpecifiedModel`.
    score_weights, horizon_weights, normalize, max_missing
        Score settings.
    backtests : dict
        ``(draw, treatment)`` to the backtest results.

    Examples
    --------
    >>> rob.table()  # doctest: +SKIP
    >>> rob.pivot()  # scores, one column per treatment  # doctest: +SKIP
    """

    target: str
    treatments: tuple[str, ...]
    periods: Any
    evaluations: pd.DataFrame
    models: dict[tuple[int, str], SpecifiedModel] = field(repr=False)
    score_weights: dict[str, float] = field(default_factory=lambda: {"rmsfe": 1.0})
    horizon_weights: dict[str, float] | None = None
    normalize: str = "rank"
    max_missing: float = 0.1
    backtests: dict[tuple[int, str], BacktestResults] = field(default_factory=dict, repr=False)

    def scores(
        self,
        score: Mapping[str, float] | str | None = None,
        horizon_weights: Mapping[Any, float] | None = None,
        normalize: str | None = None,
    ) -> pd.Series:
        """Score of every (specification, treatment) row (``inf`` when not evaluated).

        Parameters
        ----------
        score, horizon_weights, normalize : optional
            Override the score settings (see :func:`weighted_score`).

        Returns
        -------
        pandas.Series
            Aligned with :attr:`evaluations`.

        Raises
        ------
        ValueError
            On invalid score settings.

        Examples
        --------
        >>> rob.scores()  # doctest: +SKIP
        """
        return _scores(
            self.evaluations,
            _valid(self.evaluations, self.max_missing),
            self.score_weights if score is None else score,
            self.horizon_weights if horizon_weights is None else horizon_weights,
            self.normalize if normalize is None else normalize,
        )

    def table(
        self,
        score: Mapping[str, float] | str | None = None,
        horizon_weights: Mapping[Any, float] | None = None,
        normalize: str | None = None,
    ) -> pd.DataFrame:
        """(Specification, treatment) pairs sorted by score (best first).

        Parameters
        ----------
        score, horizon_weights, normalize : optional
            Override the score settings.

        Returns
        -------
        pandas.DataFrame
            Columns ``rank``, ``draw``, ``search_rank``, ``treatment``, ``score``,
            ``status``, ``note``, ``coverage``, criteria and run information.

        Raises
        ------
        ValueError
            On invalid score settings.

        Examples
        --------
        >>> rob.table().head()  # doctest: +SKIP
        """
        scores = self.scores(score, horizon_weights, normalize)
        first = ["draw", "search_rank", "treatment"]
        return _ordered(self.evaluations, scores.to_numpy(), first).reset_index(drop=True)

    def pivot(self) -> pd.DataFrame:
        """Scores with one row per specification and one column per treatment.

        Returns
        -------
        pandas.DataFrame
            Index ``draw`` (in search order), columns the treatments; NaN for the
            treatments a model cannot apply.

        Examples
        --------
        >>> rob.pivot()  # doctest: +SKIP
        """
        scores = self.scores().where(self.evaluations["status"] != "skipped")
        frame = self.evaluations.assign(score=scores.to_numpy())
        frame = frame.sort_values("search_rank", kind="mergesort")
        table = frame.pivot_table(
            index="draw", columns="treatment", values="score", aggfunc="first", dropna=False
        )
        table = table.reindex(index=list(dict.fromkeys(frame["draw"])), columns=self.treatments)
        table.columns.name = None
        return table

    def best(self) -> dict[str, Any]:
        """Best (specification, treatment) pair.

        Returns
        -------
        dict
            ``draw``, ``treatment``, ``score`` and ``note``.

        Raises
        ------
        ValueError
            If no pair has a finite score.

        Examples
        --------
        >>> rob.best()["treatment"]  # doctest: +SKIP
        'dummy'
        """
        table = self.table()
        row = table.iloc[0]
        if not np.isfinite(float(row["score"])):
            raise ValueError("No (specification, treatment) pair has a finite score.")
        return {
            "draw": int(row["draw"]),
            "treatment": str(row["treatment"]),
            "score": float(row["score"]),
            "note": str(row["note"]),
        }

    def best_model(self) -> SpecifiedModel:
        """Unfitted model of the best (specification, treatment) pair.

        Returns
        -------
        SpecifiedModel
            Model with the treatment's parameters (e.g. ``covid="dummy"``) or
            preprocessing.

        Raises
        ------
        ValueError
            If no pair has a finite score.

        Examples
        --------
        >>> rob.best_model().fit(data, "gdp")  # doctest: +SKIP
        """
        best = self.best()
        return self.models[(best["draw"], best["treatment"])].clone()

    def summary(self) -> str:
        """Plain-text summary of the robustness step.

        Returns
        -------
        str
            Evaluation window, treatments and the score table.

        Examples
        --------
        >>> print(rob.summary())  # doctest: +SKIP
        """
        title = f"Covid robustness for {self.target}"
        pivot = self.pivot().to_string(float_format=lambda v: f"{v:.4g}")
        notes = self.evaluations.drop_duplicates("treatment")[["treatment", "note"]]
        lines = [title, "=" * len(title), f"evaluation periods: {self.periods}", ""]
        lines += [f"  {t}: {n}" for t, n in zip(notes["treatment"], notes["note"], strict=True)]
        return "\n".join([*lines, "", pivot])

    def __str__(self) -> str:
        return self.summary()

r"""Block and variable selection validated in pseudo real time (innovation I7).

Which groups of predictors - the blocks of a dynamic factor model (real, nominal,
surveys, financial...) or individual series - should enter a nowcasting model? Bai &
Ng (2008) show that forecasting with factors extracted from *fewer, targeted*
predictors can beat the full panel, and Boivin & Ng (2006) that adding noisy or
redundant series can worsen the factor estimates. In nowcasting the relevant loss is
the accuracy along the real-time data flow, so the groups are chosen here by a greedy
search scored by the **pseudo real-time root mean squared forecast error** (RMSFE)
of the model (Bańbura, Giannone, Modugno & Reichlin, 2013), computed with
:class:`~nowcastbox.evaluation.PseudoRealTimeBacktest`:

* **forward** selection starts from the groups that are always included and adds, at
  each step, the group whose inclusion lowers the score the most;
* **backward** elimination starts from every group and removes, at each step, the
  group whose exclusion lowers the score the most;

both stop when no move improves the score by more than ``min_improvement`` (or when
``max_groups`` is reached). With :math:`G` groups the search evaluates at most
:math:`G(G+1)/2` sets; each set is scored once (scores are cached) and every set of a
step is evaluated in **one** backtest, so the pseudo real-time vintages are built once
per step and shared by all candidates. Within a backtest each candidate re-estimates
its parameters every ``refit_every`` vintages and only updates the information set in
between (``model.update`` / ``results.predict``), and re-estimations are
**warm-started** from the previous estimates when the model accepts ``init=`` (e.g.
:class:`~nowcastbox.models.MixedFreqDFM`). Keeping the number of vintages small
(default: 8 monthly vintages before the release of the last observed target period)
makes the search affordable.

A cheaper **information-criterion** option fits each candidate once on the full
sample and scores the in-sample fit of the target obtained *without the target's
own observations* (the target is masked and the nowcast is reconstructed from the
predictors only),

.. math::

    \mathrm{IC}(S) = n \ln \hat\sigma^2(S) + k(S)\, c_n, \qquad
    c_n = 2 \;(\text{AIC}), \; \ln n \;(\text{BIC}), \; 2 \ln\ln n \;(\text{HQ}),

with :math:`n` the number of observed target periods, :math:`\hat\sigma^2(S)` the mean
squared fitting error and :math:`k(S)` the number of selected groups (a heuristic
count of the extra parameters of the target equation; Bai & Ng, 2008, use the BIC in
the forecasting equation of targeted factors). It ignores the real-time data flow
and uses full-sample parameters, so it is a screening device; the RMSFE criterion is
the recommended one.

References
----------
Bai, J., & Ng, S. (2008). Forecasting economic time series using targeted predictors.
*Journal of Econometrics*, 146(2), 304-317.

Boivin, J., & Ng, S. (2006). Are more data always better for factor analysis?
*Journal of Econometrics*, 132(1), 169-194.

Bańbura, M., Giannone, D., Modugno, M., & Reichlin, L. (2013). Now-casting and the
real-time data flow. In *Handbook of Economic Forecasting*, vol. 2A, 195-237.

Bańbura, M., & Modugno, M. (2014). Maximum likelihood estimation of factor models on
datasets with arbitrary pattern of missing data. *Journal of Applied Econometrics*,
29(1), 133-160.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical
Learning*, 2nd ed., section 3.3.2 (forward and backward stepwise selection).
"""

from __future__ import annotations

import copy
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import FrequencySpec, MixedFrequencyData, as_mixed_frequency_data
from nowcastbox.core.formula import resolve_target
from nowcastbox.core.frequency import Frequency, base_to_native, is_period_end
from nowcastbox.core.results import NowcastResults
from nowcastbox.selection._plot import Backend, plot_lines

__all__ = [
    "SCORINGS",
    "SelectionPath",
    "ValidationSettings",
    "select_blocks",
    "select_variables",
]

logger = get_logger(__name__)

Direction = Literal["forward", "backward"]
Scoring = Literal["rmsfe", "mae", "aic", "bic", "hq"]
SCORINGS: tuple[str, ...] = ("rmsfe", "mae", "aic", "bic", "hq")
"""Available scores: pseudo real-time ``"rmsfe"``/``"mae"`` and the information
criteria ``"aic"``, ``"bic"``, ``"hq"``."""

_BACKTEST_SCORES = ("rmsfe", "mae")
_DIRECTIONS = ("forward", "backward")
GroupKey = frozenset[str]


# ====================================================================== settings
@dataclass(frozen=True)
class ValidationSettings:
    """Design of the pseudo real-time validation used by the RMSFE/MAE scores.

    Parameters
    ----------
    start, end : date-like, optional
        First and last vintage dates. Default: ``end`` = last day of the last
        observed target period (its nowcast is the last one evaluated) and ``start``
        = ``n_vintages - 1`` months earlier.
    n_vintages : int, default 8
        Number of monthly vintages when ``start`` is not given.
    step : str, default "M"
        Spacing of the vintages (:func:`nowcastbox.vintages.vintage_dates`).
    calendar, delay : optional
        Release rule of the pseudo real-time vintages (see
        :class:`~nowcastbox.evaluation.PseudoRealTimeBacktest`); default: the
        ``release_delays`` metadata of the data.
    target_offsets : sequence of int, default (-1, 0)
        Target periods evaluated at each vintage relative to the vintage's period
        (backcast while unreleased and nowcast).
    window : {"expanding", "rolling"}, default "expanding"
        Estimation window.
    window_length : int, optional
        Rolling window length in base periods.
    refit_every : int, default 3
        Re-estimate the parameters every ``refit_every`` vintages; in between only
        the information set is updated (warm start of the state).
    warm_start : bool, default True
        Start each re-estimation from the previous estimates of the same candidate
        when the model accepts ``init=`` (EM models).
    fit_kwargs : mapping, optional
        Extra keyword arguments of ``model.fit``.
    max_missing : float, default 0.1
        Largest share of missing forecasts (failed fits) of a candidate; above it the
        candidate's score is ``inf``.
    n_jobs : int, optional
        Parallel jobs of the backtest (blocks of ``refit_every`` vintages).

    Examples
    --------
    >>> ValidationSettings(n_vintages=6, refit_every=2).n_vintages
    6
    """

    start: Any = None
    end: Any = None
    n_vintages: int = 8
    step: Any = "M"
    calendar: Any = None
    delay: Any = None
    target_offsets: Sequence[int] = (-1, 0)
    window: str = "expanding"
    window_length: int | None = None
    refit_every: int = 3
    warm_start: bool = True
    fit_kwargs: Mapping[str, Any] | None = None
    max_missing: float = 0.1
    n_jobs: int | None = None

    def validate(self) -> None:
        """Check the settings.

        Raises
        ------
        ValueError
            If a setting is invalid.

        Examples
        --------
        >>> ValidationSettings(refit_every=0).validate()
        Traceback (most recent call last):
        ...
        ValueError: refit_every must be a positive integer, got 0.
        """
        for name in ("n_vintages", "refit_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if not 0.0 <= float(self.max_missing) < 1.0:
            raise ValueError(f"max_missing must lie in [0, 1), got {self.max_missing!r}.")

    def dates(self, data: MixedFrequencyData, target: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """First and last vintage dates (defaults resolved on ``data``).

        Parameters
        ----------
        data : MixedFrequencyData
            Final dataset.
        target : str
            Target series.

        Returns
        -------
        tuple of pandas.Timestamp
            ``(start, end)``.

        Raises
        ------
        ValueError
            If the target has no observation.

        Examples
        --------
        >>> from nowcastbox.models.two_step import simulate_two_step_example
        >>> data = simulate_two_step_example(random_state=0)
        >>> [str(d.date()) for d in ValidationSettings(n_vintages=3).dates(data, "gdp")]
        ['2014-07-31', '2014-09-30']
        """
        end = self.end
        if end is None:
            observed = data.to_native(target, dropna=True)
            if observed.empty:
                raise ValueError(f"The target {target!r} has no observations.")
            end = observed.index[-1].end_time.normalize()
        end_ts = pd.Timestamp(end)
        if self.start is not None:
            return pd.Timestamp(self.start), end_ts
        first = pd.Period(end_ts, freq="M") - (int(self.n_vintages) - 1)
        return first.end_time.normalize(), end_ts


def _as_settings(validation: ValidationSettings | Mapping[str, Any] | None) -> ValidationSettings:
    if validation is None:
        settings = ValidationSettings()
    elif isinstance(validation, ValidationSettings):
        settings = validation
    elif isinstance(validation, Mapping):
        settings = ValidationSettings(**dict(validation))
    else:
        raise TypeError(
            "validation must be a ValidationSettings, a mapping or None, got "
            f"{type(validation).__name__}."
        )
    settings.validate()
    return settings


# ====================================================================== result
@dataclass(frozen=True)
class SelectionPath:
    """Outcome of :func:`select_blocks` / :func:`select_variables`.

    Attributes
    ----------
    target : str
        Target series.
    kind : str
        ``"blocks"`` or ``"variables"``.
    direction : str
        ``"forward"`` or ``"backward"``.
    scoring : str
        Score minimised (``"rmsfe"``, ``"mae"``, ``"aic"``, ``"bic"``, ``"hq"``).
    groups : dict[str, tuple[str, ...]]
        Candidate groups and their series.
    chosen : tuple[str, ...]
        Selected groups (in candidate order).
    score : float
        Score of the selected set.
    path : pandas.DataFrame
        One row per accepted step: ``step``, ``action`` (``"start"``, ``"add"``,
        ``"remove"``), ``group``, ``n_groups``, ``score``, ``improvement``,
        ``selected``.
    evaluations : pandas.DataFrame
        Every evaluated set: ``step``, ``action``, ``group``, ``groups``,
        ``n_groups``, ``n_series``, ``score``, ``coverage`` (share of forecasts
        available; 1 for information criteria).
    fixed : tuple[str, ...]
        Predictors in no candidate group (always included).
    always_include : tuple[str, ...]
        Groups forced into every set.
    series_order : tuple[str, ...]
        Predictors in data order (used to order :attr:`selected_series`).
    info : dict
        Run information (number of evaluations, vintages, timing, warnings...).

    Examples
    --------
    >>> import warnings
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from nowcastbox.models import TwoStepDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> data = simulate_two_step_example(n_periods=90, n_series=4, random_state=0)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     path = select_variables(TwoStepDFM(n_factors=1), data, "gdp", scoring="bic")
    >>> path.path.columns.tolist()
    ['step', 'action', 'group', 'n_groups', 'score', 'improvement', 'selected']
    >>> path.select(data).columns == ["gdp", *path.selected_series]
    True
    >>> bool(path.scores.is_monotonic_decreasing)
    True
    >>> "Variables selection (forward, bic)" in path.summary()
    True
    >>> ax = path.plot()
    """

    target: str
    kind: str
    direction: str
    scoring: str
    groups: dict[str, tuple[str, ...]]
    chosen: tuple[str, ...]
    score: float
    path: pd.DataFrame
    evaluations: pd.DataFrame
    fixed: tuple[str, ...] = ()
    always_include: tuple[str, ...] = ()
    series_order: tuple[str, ...] = ()
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_series(self) -> list[str]:
        """Predictors of the selected set (fixed series + chosen groups), in data order.

        Examples
        --------
        >>> path.selected_series  # doctest: +SKIP
        ['x1', 'x2']
        """
        keep = set(self.fixed).union(*(self.groups[g] for g in self.chosen))
        return [s for s in self.series_order if s in keep]

    @property
    def scores(self) -> pd.Series:
        """Score after each accepted step (index ``step``).

        Examples
        --------
        >>> path.scores  # doctest: +SKIP
        """
        return self.path.set_index("step")["score"].rename(self.scoring)

    def select(self, data: MixedFrequencyData | pd.DataFrame) -> MixedFrequencyData:
        """Restrict a panel to the target and the selected predictors.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            Panel containing the target and the selected series.

        Returns
        -------
        MixedFrequencyData
            The target and :attr:`selected_series`.

        Examples
        --------
        >>> path.select(data).columns  # doctest: +SKIP
        """
        panel = as_mixed_frequency_data(data)
        return panel.select([self.target, *self.selected_series])

    def to_frame(self) -> pd.DataFrame:
        """The accepted path (copy of :attr:`path`).

        Returns
        -------
        pandas.DataFrame
            One row per step.

        Examples
        --------
        >>> path.to_frame().columns.tolist()  # doctest: +SKIP
        """
        return self.path.copy()

    def summary(self) -> str:
        """Plain-text summary.

        Returns
        -------
        str
            Settings, chosen groups and the path.

        Examples
        --------
        >>> "Selected" in path.summary()  # doctest: +SKIP
        True
        """
        title = f"{self.kind.capitalize()} selection ({self.direction}, {self.scoring})"
        lines = [
            title,
            "=" * len(title),
            f"Target: {self.target}   candidate groups: {len(self.groups)}   "
            f"evaluations: {len(self.evaluations)}",
            f"Selected ({len(self.chosen)}): {', '.join(self.chosen) or '-'}",
            f"Series ({len(self.selected_series)}): {', '.join(self.selected_series)}",
            f"Score: {self.score:.6g}",
        ]
        if self.fixed:
            lines.append(f"Always included series: {', '.join(self.fixed)}")
        table = self.path.drop(columns="selected").to_string(
            index=False, float_format=lambda v: f"{v:.6g}"
        )
        return "\n".join([*lines, "", table])

    def __str__(self) -> str:
        return self.summary()

    def plot(self, *, backend: Backend = "matplotlib", ax: Any = None) -> Any:
        """Plot the score along the accepted steps.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, default "matplotlib"
            Plotting library.
        ax : matplotlib.axes.Axes, optional
            Existing axes (Matplotlib only).

        Returns
        -------
        matplotlib.axes.Axes or plotly.graph_objects.Figure
            The plot.

        Examples
        --------
        >>> ax = path.plot()  # doctest: +SKIP
        """
        frame = self.path.set_index("step")[["score"]].replace([np.inf, -np.inf], np.nan)
        best = int(self.path["step"].iloc[-1])
        return plot_lines(
            frame,
            selected={"score": best},
            title=f"{self.kind.capitalize()} selection path ({self.direction})",
            xlabel="step",
            ylabel=self.scoring,
            backend=backend,
            ax=ax,
        )


# ====================================================================== candidates
def _clone_model(model: Any) -> Any:
    clone = getattr(model, "clone", None)
    return clone() if callable(clone) else copy.deepcopy(model)


def _native_estimates(frame: pd.DataFrame, target: str, freq: Frequency) -> pd.Series:
    """Target values in the storage slots of a base-grid frame, on native periods."""
    column = frame[target]
    index = column.index
    assert isinstance(index, pd.PeriodIndex)  # noqa: S101
    slots = is_period_end(index, freq)  # calendar aware (weekly/daily grids)
    out = column[slots]
    out.index = base_to_native(index[slots], freq)
    return out


def _information_update(
    estimator: Any, results: NowcastResults | None, panel: MixedFrequencyData
) -> pd.Series | None:
    """Estimates on ``panel`` with fixed parameters (``update`` or ``predict``)."""
    update = getattr(estimator, "update", None)
    if callable(update):
        out = update(panel)
        if isinstance(out, NowcastResults):
            return out.estimate
    predict = getattr(results, "predict", None)
    if results is not None and callable(predict):
        frame = predict(panel)
        if isinstance(frame, pd.DataFrame) and results.target in frame:
            return _native_estimates(frame, results.target, results.target_frequency)
    return None


@dataclass
class _WarmState:
    """State shared by the clones of one candidate along a backtest."""

    estimator: Any = None
    results: NowcastResults | None = None
    calls: int = 0


class _SubsetForecaster:
    """Forecaster restricted to a set of predictors (``fit``/``predict`` protocol).

    Clones share a :class:`_WarmState`, so the backtest's per-vintage clones can
    update the information set of the last estimate and warm-start re-estimations.
    """

    def __init__(
        self,
        model: Any,
        columns: Sequence[str],
        settings: ValidationSettings,
        state: _WarmState | None = None,
    ) -> None:
        self.model = model
        self.columns = list(columns)
        self.settings = settings
        self.state = _WarmState() if state is None else state
        self.name = "subset"
        self._estimate: pd.Series | None = None
        self._predictor: Any = None

    def clone(self) -> _SubsetForecaster:
        return _SubsetForecaster(self.model, self.columns, self.settings, self.state)

    def fit(self, data: MixedFrequencyData, target: str) -> _SubsetForecaster:
        panel = data.select([target, *self.columns])
        state = self.state
        refit = state.estimator is None or state.calls % int(self.settings.refit_every) == 0
        state.calls += 1
        if not refit:
            estimate = _information_update(state.estimator, state.results, panel)
            if estimate is not None:
                self._estimate = estimate
                return self
        self._full_fit(panel, target)
        return self

    def _full_fit(self, panel: MixedFrequencyData, target: str) -> None:
        state = self.state
        kwargs = dict(self.settings.fit_kwargs or {})
        estimator = _clone_model(self.model)
        warm = self._warm_start(estimator)
        try:
            out = estimator.fit(panel, target, **kwargs)
        except Exception:
            if not warm:
                raise
            estimator = _clone_model(self.model)
            out = estimator.fit(panel, target, **kwargs)
        state.estimator = estimator
        if isinstance(out, NowcastResults):
            state.results = out
            self._estimate = out.estimate
        else:
            state.results = None
            self._estimate = None
            self._predictor = estimator

    def _warm_start(self, estimator: Any) -> bool:
        previous = self.state.results
        get_params = getattr(estimator, "get_params", None)
        if not self.settings.warm_start or previous is None or not callable(get_params):
            return False
        params = get_params(deep=False)
        if not isinstance(params, Mapping) or "init" not in params:
            return False
        estimator.set_params(init=previous)
        return True

    def predict(self, periods: pd.PeriodIndex) -> pd.Series:
        if self._estimate is not None:
            return self._estimate.reindex(periods)
        values = self._predictor.predict(periods)
        return pd.Series(np.asarray(values, dtype=float), index=periods)


# ====================================================================== scorers
@dataclass
class _Problem:
    """Everything a scorer needs."""

    model: Any
    data: MixedFrequencyData
    target: str
    groups: dict[str, tuple[str, ...]]
    fixed: tuple[str, ...]
    order: tuple[str, ...]
    scoring: str
    settings: ValidationSettings
    n_warnings: int = 0
    n_failures: int = 0

    def columns(self, key: GroupKey) -> list[str]:
        keep = set(self.fixed).union(*(self.groups[g] for g in key))
        return [s for s in self.order if s in keep]


def _backtest_scores(
    problem: _Problem, keys: list[GroupKey]
) -> dict[GroupKey, tuple[float, float]]:
    """Score several sets in one pseudo real-time backtest."""
    from nowcastbox.evaluation import PseudoRealTimeBacktest

    s = problem.settings
    start, end = s.dates(problem.data, problem.target)
    labels = [f"candidate_{i}" for i in range(len(keys))]
    benchmarks = {
        lab: _SubsetForecaster(problem.model, problem.columns(k), s)
        for lab, k in zip(labels, keys, strict=True)
    }
    backtest = PseudoRealTimeBacktest(
        data=problem.data,
        target=problem.target,
        calendar=s.calendar,
        delay=s.delay,
        start=start,
        end=end,
        step=s.step,
        benchmarks=benchmarks,
        window=s.window,
        window_length=s.window_length,
        refit_every=int(s.refit_every),
        n_jobs=s.n_jobs,
        target_offsets=tuple(s.target_offsets),
        errors="warn",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = backtest.run()
    problem.n_warnings += len(caught)
    problem.n_failures += len(result.info.get("failures", []))
    frame = result.forecasts[np.isfinite(result.forecasts["actual"].to_numpy(dtype=float))]
    if frame.empty:
        raise ValueError(
            "No evaluable forecast in the validation window (no released target period); "
            "move the vintage dates or increase n_vintages."
        )
    out: dict[GroupKey, tuple[float, float]] = {}
    for lab, key in zip(labels, keys, strict=True):
        errors = frame.loc[frame["model"] == lab, "error"].to_numpy(dtype=float)
        coverage = float(np.isfinite(errors).mean()) if errors.size else 0.0
        out[key] = (_loss(errors, problem.scoring, coverage, s.max_missing), coverage)
    return out


def _loss(errors: np.ndarray, scoring: str, coverage: float, max_missing: float) -> float:
    finite = errors[np.isfinite(errors)]
    if finite.size == 0 or coverage < 1.0 - max_missing:
        return float("inf")
    if scoring == "mae":
        return float(np.mean(np.abs(finite)))
    return float(np.sqrt(np.mean(finite**2)))


def _target_free_fit(
    estimator: Any, results: NowcastResults, panel: MixedFrequencyData
) -> pd.Series:
    """Target estimates reconstructed from the predictors only (target masked)."""
    frame = panel.to_frame()
    frame[results.target] = np.nan
    masked = panel.with_data(frame)
    estimate = _information_update(estimator, results, masked)
    return results.estimate if estimate is None else estimate


def _ic_scores(problem: _Problem, keys: list[GroupKey]) -> dict[GroupKey, tuple[float, float]]:
    """Information criterion of each set (one full-sample fit per set)."""
    out: dict[GroupKey, tuple[float, float]] = {}
    observed = problem.data.to_native(problem.target, dropna=True)
    for key in keys:
        panel = problem.data.select([problem.target, *problem.columns(key)])
        estimator = _clone_model(problem.model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                results = estimator.fit(
                    panel, problem.target, **dict(problem.settings.fit_kwargs or {})
                )
                fitted = _target_free_fit(estimator, results, panel)
            except Exception as err:
                logger.info("Candidate %s failed: %s", sorted(key), err)
                problem.n_failures += 1
                fitted = None
        problem.n_warnings += len(caught)
        out[key] = (_information_criterion(observed, fitted, len(key), problem.scoring), 1.0)
    return out


def _information_criterion(
    observed: pd.Series, fitted: pd.Series | None, n_groups: int, scoring: str
) -> float:
    if fitted is None:
        return float("inf")
    errors = (observed - fitted.reindex(observed.index)).to_numpy(dtype=float)
    errors = errors[np.isfinite(errors)]
    n = errors.size
    if n < 3:
        return float("inf")
    sigma2 = max(float(np.mean(errors**2)), np.finfo(float).tiny)
    penalty = {"aic": 2.0, "bic": np.log(n), "hq": 2.0 * np.log(np.log(n))}[scoring]
    return float(n * np.log(sigma2) + n_groups * penalty)


class _Scorer:
    """Cached batch scoring of sets of groups."""

    def __init__(self, problem: _Problem) -> None:
        self.problem = problem
        self.cache: dict[GroupKey, tuple[float, float]] = {}
        self.batch: Callable[[_Problem, list[GroupKey]], dict[GroupKey, tuple[float, float]]] = (
            _backtest_scores if problem.scoring in _BACKTEST_SCORES else _ic_scores
        )

    def __call__(self, keys: Iterable[GroupKey]) -> dict[GroupKey, tuple[float, float]]:
        keys = list(dict.fromkeys(keys))
        todo = [k for k in keys if k not in self.cache and (k or self.problem.fixed)]
        if todo:
            self.cache.update(self.batch(self.problem, todo))
        return {k: self.cache.get(k, (float("inf"), 0.0)) for k in keys}


# ====================================================================== search
@dataclass
class _Search:
    """Greedy forward/backward search over groups."""

    scorer: _Scorer
    names: tuple[str, ...]
    always: frozenset[str]
    direction: str
    max_groups: int
    min_improvement: float
    path: list[dict[str, Any]] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)

    def _record(self, step: int, action: str, group: str | None, key: GroupKey, value: Any) -> None:
        score, coverage = value
        self.evaluations.append(
            {
                "step": step,
                "action": action,
                "group": group,
                "groups": self.ordered(key),
                "n_groups": len(key),
                "n_series": len(self.scorer.problem.columns(key)),
                "score": score,
                "coverage": coverage,
            }
        )

    def ordered(self, key: GroupKey) -> tuple[str, ...]:
        return tuple(g for g in self.names if g in key)

    def _accept(
        self,
        step: int,
        action: str,
        group: str | None,
        key: GroupKey,
        score: float,
        previous: float,
    ) -> None:
        improvement = previous - score if np.isfinite(previous) else np.nan
        self.path.append(
            {
                "step": step,
                "action": action,
                "group": group,
                "n_groups": len(key),
                "score": score,
                "improvement": improvement,
                "selected": ", ".join(self.ordered(key)),
            }
        )

    def _moves(self, current: GroupKey) -> list[tuple[str, GroupKey]]:
        if self.direction == "forward":
            if len(current) >= self.max_groups:
                return []
            return [(g, current | {g}) for g in self.names if g not in current]
        return [(g, current - {g}) for g in self.names if g in current and g not in self.always]

    def run(self) -> tuple[GroupKey, float]:
        start: GroupKey = (
            frozenset(self.always) if self.direction == "forward" else frozenset(self.names)
        )
        score = self.scorer([start])[start]
        self._record(0, "start", None, start, score)
        self._accept(0, "start", None, start, score[0], np.nan)
        current, best = start, score[0]
        action = "add" if self.direction == "forward" else "remove"
        step = 0
        while moves := self._moves(current):
            step += 1
            scores = self.scorer([k for _, k in moves])
            for group, key in moves:
                self._record(step, action, group, key, scores[key])
            group, key = min(moves, key=lambda m: scores[m[1]][0])
            candidate = scores[key][0]
            if not candidate < best - self.min_improvement and np.isfinite(best):
                break
            if not np.isfinite(candidate):
                break
            self._accept(step, action, group, key, candidate, best)
            current, best = key, candidate
        return current, best


# ====================================================================== public API
def _check_options(direction: str, scoring: str, min_improvement: float) -> None:
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}.")
    if scoring not in SCORINGS:
        raise ValueError(f"scoring must be one of {SCORINGS}, got {scoring!r}.")
    value = float(min_improvement)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"min_improvement must be >= 0, got {min_improvement!r}.")


def _check_model(model: Any) -> None:
    if not callable(getattr(model, "fit", None)):
        raise TypeError(f"model must have a fit(data, target) method, got {type(model).__name__}.")


def _groups_from_frame(membership: pd.DataFrame, universe: Sequence[str]) -> dict[str, list[str]]:
    frame = membership.reindex(index=list(universe)).fillna(False).astype(bool)
    return {str(b): [s for s in universe if bool(frame.loc[s, b])] for b in frame.columns}


def _block_groups(
    blocks: Any, data: MixedFrequencyData, universe: Sequence[str]
) -> dict[str, list[str]]:
    """Candidate blocks: data metadata (optionally a subset of names), mapping or frame."""
    if blocks is None or (isinstance(blocks, Sequence) and not isinstance(blocks, str | bytes)):
        membership = data.blocks
        if membership.shape[1] == 0:
            raise ValueError(
                "The data have no block metadata; pass blocks as a mapping or DataFrame."
            )
        if blocks is not None:
            names = [str(b) for b in blocks]
            unknown = [b for b in names if b not in membership.columns]
            if unknown:
                raise ValueError(
                    f"Unknown blocks {unknown}; available: {list(membership.columns)}."
                )
            membership = membership[names]
        return _groups_from_frame(membership, universe)
    if isinstance(blocks, pd.DataFrame):
        return _groups_from_frame(blocks, universe)
    if isinstance(blocks, Mapping):
        return {str(k): _as_list(v) for k, v in blocks.items()}
    raise TypeError(
        "blocks must be None, a sequence of block names, a mapping {block: series} or a "
        f"DataFrame (series x blocks), got {type(blocks).__name__}."
    )


def _variable_groups(candidates: Any, universe: Sequence[str]) -> dict[str, list[str]]:
    if candidates is None:
        return {s: [s] for s in universe}
    if isinstance(candidates, Mapping):
        return {str(k): _as_list(v) for k, v in candidates.items()}
    if isinstance(candidates, str) or not isinstance(candidates, Iterable):
        raise TypeError(
            "candidates must be None, a sequence of series names or a mapping "
            f"{{group: series}}, got {type(candidates).__name__}."
        )
    return {str(s): [str(s)] for s in candidates}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _finalise_groups(
    raw: dict[str, list[str]], universe: Sequence[str], always: Sequence[str]
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...], tuple[str, ...]]:
    """Validate groups; return groups, fixed series and always-included groups."""
    known = set(universe)
    unknown = sorted({s for members in raw.values() for s in members} - known)
    if unknown:
        raise ValueError(f"Unknown predictors {unknown} (not in the data or the formula).")
    groups = {g: tuple(s for s in universe if s in set(m)) for g, m in raw.items()}
    empty = [g for g, m in groups.items() if not m]
    if empty:
        logger.info("Dropping empty candidate groups %s.", empty)
    groups = {g: m for g, m in groups.items() if m}
    if not groups:
        raise ValueError("No non-empty candidate group of predictors.")
    always_t = tuple(str(g) for g in always)
    missing = [g for g in always_t if g not in groups]
    if missing:
        raise ValueError(f"always_include names unknown groups {missing}.")
    covered = set().union(*groups.values())
    fixed = tuple(s for s in universe if s not in covered)
    return groups, fixed, always_t


def _run_selection(
    kind: str,
    model: Any,
    data: MixedFrequencyData | pd.DataFrame,
    target: str,
    raw_groups: Callable[[MixedFrequencyData, Sequence[str]], dict[str, list[str]]],
    options: dict[str, Any],
) -> SelectionPath:
    started = time.perf_counter()
    _check_model(model)
    _check_options(options["direction"], options["scoring"], options["min_improvement"])
    settings = _as_settings(options["validation"])
    panel = as_mixed_frequency_data(data, options["frequency"])
    target_name, regressors = resolve_target(target, panel.columns)
    universe = tuple(regressors)
    groups, fixed, always = _finalise_groups(
        raw_groups(panel, universe), universe, options["always_include"]
    )
    max_groups = options["max_groups"]
    limit = len(groups) if max_groups is None else int(max_groups)
    if limit < max(1, len(always)):
        raise ValueError(f"max_groups={max_groups} is smaller than the forced groups.")
    problem = _Problem(
        model=model,
        data=panel.select([target_name, *universe]),
        target=target_name,
        groups=groups,
        fixed=fixed,
        order=universe,
        scoring=options["scoring"],
        settings=settings,
    )
    search = _Search(
        scorer=_Scorer(problem),
        names=tuple(groups),
        always=frozenset(always),
        direction=options["direction"],
        max_groups=limit,
        min_improvement=float(options["min_improvement"]),
    )
    chosen, score = search.run()
    info: dict[str, Any] = {
        "n_evaluations": len(search.scorer.cache),
        "n_warnings": problem.n_warnings,
        "n_failures": problem.n_failures,
        "time": time.perf_counter() - started,
        "settings": settings,
    }
    if options["scoring"] in _BACKTEST_SCORES:
        info["vintage_range"] = settings.dates(problem.data, target_name)
    logger.info("%s selection: chose %s (score %.4g).", kind, sorted(chosen), score)
    return SelectionPath(
        target=target_name,
        kind=kind,
        direction=options["direction"],
        scoring=options["scoring"],
        groups=groups,
        chosen=search.ordered(chosen),
        score=score,
        path=pd.DataFrame(search.path),
        evaluations=pd.DataFrame(search.evaluations),
        fixed=fixed,
        always_include=always,
        series_order=universe,
        info=info,
    )


def select_blocks(
    model: Any,
    data: MixedFrequencyData | pd.DataFrame,
    target: str,
    blocks: Any = None,
    *,
    direction: Direction = "forward",
    scoring: Scoring = "rmsfe",
    always_include: Sequence[str] = (),
    max_blocks: int | None = None,
    min_improvement: float = 0.0,
    validation: ValidationSettings | Mapping[str, Any] | None = None,
    frequency: FrequencySpec | None = None,
) -> SelectionPath:
    """Select the blocks of predictors of a nowcasting model (greedy search, I7).

    Parameters
    ----------
    model : BaseNowcaster or forecaster
        Unfitted model (e.g. :class:`~nowcastbox.models.TwoStepDFM`,
        :class:`~nowcastbox.models.MixedFreqDFM`); it is cloned for every fit. Any
        object whose ``fit(data, target)`` returns
        :class:`~nowcastbox.core.results.NowcastResults` (or follows the
        ``fit``/``predict`` benchmark protocol, RMSFE/MAE scores only).
    data : MixedFrequencyData or pandas.DataFrame
        Final dataset (pseudo real-time vintages are built from its release delays).
    target : str
        Target series, or a formula restricting the predictors (``"gdp ~ . - x9"``).
    blocks : None, sequence of str, mapping or pandas.DataFrame, optional
        Candidate blocks: ``None`` = the block metadata of ``data``; a sequence of
        block names = those metadata blocks; a mapping ``{block: [series]}``; or a
        boolean DataFrame (series x blocks). A series may belong to several blocks;
        predictors in no candidate block are always included.
    direction : {"forward", "backward"}, default "forward"
        Forward selection or backward elimination.
    scoring : {"rmsfe", "mae", "aic", "bic", "hq"}, default "rmsfe"
        Pseudo real-time loss or information criterion (see the module notes).
    always_include : sequence of str, default ()
        Blocks forced into every set.
    max_blocks : int, optional
        Largest number of selected blocks (forward search).
    min_improvement : float, default 0.0
        A step is accepted only if it lowers the score by more than this amount.
    validation : ValidationSettings or mapping, optional
        Vintage design of the RMSFE/MAE scores (:class:`ValidationSettings`).
    frequency : frequency specification, optional
        Per-series frequencies for DataFrame input.

    Returns
    -------
    SelectionPath
        Chosen blocks, scores along the path and every evaluated set.

    Raises
    ------
    ValueError
        Invalid options, unknown blocks/series or no evaluable forecast.
    TypeError
        Invalid ``model``, ``blocks`` or ``validation``.

    See Also
    --------
    select_variables : The same search over single series or user-defined groups.

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> from nowcastbox.models import TwoStepDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> data = simulate_two_step_example(n_periods=120, n_series=8, random_state=0)
    >>> frame = data.to_frame()
    >>> noise = ["noise1", "noise2", "noise3"]
    >>> frame[noise] = 3 * np.random.default_rng(1).normal(size=(120, 3))
    >>> freqs = {**data.frequencies.to_dict(), **dict.fromkeys(noise, "M")}
    >>> delays = {c: 20 for c in frame.columns} | {"gdp": 45}
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> panel = MixedFrequencyData(frame, freqs, release_delays=delays)
    >>> blocks = {
    ...     "real": [f"x{i}" for i in range(1, 5)],
    ...     "soft": [f"x{i}" for i in range(5, 9)],
    ...     "noise": noise,
    ... }
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     path = select_blocks(
    ...         TwoStepDFM(n_factors=1),
    ...         panel,
    ...         "gdp",
    ...         blocks,
    ...         validation={"n_vintages": 6, "refit_every": 3},
    ...     )
    >>> path.chosen
    ('real', 'soft')
    >>> path.selected_series == [f"x{i}" for i in range(1, 9)]
    True
    """
    return _run_selection(
        "blocks",
        model,
        data,
        target,
        lambda panel, universe: _block_groups(blocks, panel, universe),
        {
            "direction": direction,
            "scoring": scoring,
            "always_include": always_include,
            "max_groups": max_blocks,
            "min_improvement": min_improvement,
            "validation": validation,
            "frequency": frequency,
        },
    )


def select_variables(
    model: Any,
    data: MixedFrequencyData | pd.DataFrame,
    target: str,
    candidates: Any = None,
    *,
    direction: Direction = "forward",
    scoring: Scoring = "rmsfe",
    always_include: Sequence[str] = (),
    max_variables: int | None = None,
    min_improvement: float = 0.0,
    validation: ValidationSettings | Mapping[str, Any] | None = None,
    frequency: FrequencySpec | None = None,
) -> SelectionPath:
    """Select predictors (or groups of predictors) of a nowcasting model (I7).

    Same greedy search as :func:`select_blocks`, with single series (or user-defined
    groups of series) as candidates.

    Parameters
    ----------
    model : BaseNowcaster or forecaster
        Unfitted model, cloned for every fit (see :func:`select_blocks`).
    data : MixedFrequencyData or pandas.DataFrame
        Final dataset.
    target : str
        Target series or formula.
    candidates : None, sequence of str or mapping, optional
        ``None``: every predictor is a candidate; a sequence of series names: those
        series (the other predictors are always included); a mapping
        ``{group: [series]}``: groups of series added/removed together.
    direction : {"forward", "backward"}, default "forward"
        Forward selection or backward elimination.
    scoring : {"rmsfe", "mae", "aic", "bic", "hq"}, default "rmsfe"
        Pseudo real-time loss or information criterion.
    always_include : sequence of str, default ()
        Candidates forced into every set.
    max_variables : int, optional
        Largest number of selected candidates (forward search).
    min_improvement : float, default 0.0
        A step is accepted only if it lowers the score by more than this amount.
    validation : ValidationSettings or mapping, optional
        Vintage design of the RMSFE/MAE scores.
    frequency : frequency specification, optional
        Per-series frequencies for DataFrame input.

    Returns
    -------
    SelectionPath
        Chosen candidates, scores along the path and every evaluated set.

    Raises
    ------
    ValueError
        Invalid options, unknown series or no evaluable forecast.
    TypeError
        Invalid ``model``, ``candidates`` or ``validation``.

    Examples
    --------
    >>> import warnings
    >>> from nowcastbox.models import TwoStepDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> data = simulate_two_step_example(n_periods=120, n_series=6, random_state=0)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     path = select_variables(
    ...         TwoStepDFM(n_factors=1), data, "gdp", scoring="bic", always_include=["x1"]
    ...     )
    >>> path.chosen[0], path.scoring
    ('x1', 'bic')
    """
    return _run_selection(
        "variables",
        model,
        data,
        target,
        lambda _panel, universe: _variable_groups(candidates, universe),
        {
            "direction": direction,
            "scoring": scoring,
            "always_include": always_include,
            "max_groups": max_variables,
            "min_improvement": min_improvement,
            "validation": validation,
            "frequency": frequency,
        },
    )

r"""Pseudo real-time (and real-time) backtesting of nowcasting models.

:class:`PseudoRealTimeBacktest` replays the data flow: for each information date
:math:`v` in a sequence of vintages it builds the dataset available at :math:`v`,
re-estimates (or updates) the model and every benchmark on it, and records the
nowcasts of the target periods around :math:`v` - the previous quarter while it is
not yet released (*backcast*), the current quarter (*nowcast*) and, optionally, the
next ones (*forecast*) - together with horizon labels (months and days to the end of
the target period and days to its release). This is the evaluation design of
Giannone, Reichlin & Small (2008) and Bańbura, Giannone, Modugno & Reichlin (2013).

Two kinds of information sets are supported:

* **pseudo real-time** (default): the final dataset masked by a release calendar
  (:func:`nowcastbox.vintages.pseudo_real_time`; no revisions);
* **real time** (plan innovation I8): a :class:`~nowcastbox.vintages.VintageStore`
  with the data as actually published at each date, including revisions; forecasts
  are then compared with the latest, first or :math:`n`-th release of the target.

Estimation windows are expanding (all data since the start of the sample) or rolling
(the last ``window_length`` base periods). With ``refit_every = k > 1`` the model's
parameters are re-estimated every :math:`k` vintages and, in between, kept fixed
while the state is updated with the new data (``model.update(data)`` when available,
as in :class:`~nowcastbox.models.TwoStepDFM`, or ``results.predict(data)`` as in
:class:`~nowcastbox.models.MixedFreqDFM`); benchmarks are cheap and are re-estimated
at every vintage. Vintages are processed in parallel with :mod:`joblib` when it is
installed and ``n_jobs != 1`` (blocks of ``refit_every`` consecutive vintages are the
unit of work); otherwise sequentially.

:class:`BacktestResults` collects the forecasts in a long table and provides RMSFE /
MAE / bias by horizon, relative RMSFE, Diebold-Mariano and Giacomini-White tests
against a reference model and the Model Confidence Set.

**Directional accuracy.** Every row also stores ``previous_actual``, the value of the
target in the period before the target period *as known at the vintage date* (the
last released value when that period is not yet published), which is the reference
:math:`y^{p}_t` of the forecast directional accuracy (FDA, Linzenich & Meunier, 2024)
and of the Pesaran & Timmermann (1992) test
(:meth:`BacktestResults.directional_accuracy`, ``metrics=(..., "fda")``). With
``previous="final"`` the final value of the previous period is used instead (this also
works for tables saved before the column existed).

**Sub-periods.** Every accuracy table and test accepts ``periods=``: a mapping
``{label: (first, last)}`` of target periods (``None`` = open end; a list of such
pairs is a union), or the shortcuts ``"covid"`` (pre-Covid, Covid, post-Covid and
ex-Covid, see :func:`covid_periods`) and ``"ex-covid"``. The forecasts are filtered
by target period and the result gains an outer ``period`` level.

References
----------
Giannone, D., Reichlin, L. & Small, D. (2008). Nowcasting: The real-time informational
content of macroeconomic data. *Journal of Monetary Economics*, 55(4), 665-676.

Bańbura, M., Giannone, D., Modugno, M. & Reichlin, L. (2013). Now-casting and the
real-time data flow. In *Handbook of Economic Forecasting*, vol. 2A, 195-237.

Croushore, D. (2011). Frontiers of real-time data analysis. *Journal of Economic
Literature*, 49(1), 72-100.

Pesaran, M. H. & Timmermann, A. (1992). A simple nonparametric test of predictive
performance. *Journal of Business & Economic Statistics*, 10(4), 461-465.

Linzenich, J. & Meunier, B. (2024). Nowcasting made easier: a toolbox for economists.
ECB Working Paper No. 3004.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.benchmarks.sklearn_adapter import SklearnBenchmark
from nowcastbox.core.base import BaseBenchmark, BaseNowcaster
from nowcastbox.core.data import FrequencySpec, MixedFrequencyData, as_mixed_frequency_data
from nowcastbox.core.exceptions import DataQualityWarning, NowcastBoxWarning, NowcastDataError
from nowcastbox.core.formula import resolve_target
from nowcastbox.core.frequency import Frequency, base_to_native, is_period_end
from nowcastbox.core.results import NowcastResults
from nowcastbox.evaluation.metrics import (
    DIRECTIONAL_METRICS,
    LossLike,
    accuracy_by_horizon,
    directional_accuracy,
    directional_changes,
    loss_values,
    metric_by_horizon,
)
from nowcastbox.evaluation.tests import (
    ClarkWestResult,
    DieboldMarianoResult,
    GiacominiWhiteResult,
    ModelConfidenceSetResult,
    clark_west_from_differential,
    diebold_mariano,
    giacomini_white,
    model_confidence_set,
    pesaran_timmermann,
)
from nowcastbox.models.robust import COVID_WINDOW
from nowcastbox.vintages import ReleaseCalendar, VintageStore, pseudo_real_time, vintage_dates

__all__ = ["FORECAST_COLUMNS", "BacktestResults", "PseudoRealTimeBacktest", "covid_periods"]

logger = get_logger(__name__)

#: Columns of :attr:`BacktestResults.forecasts` (one row per vintage, target period, model).
#: ``previous_actual`` (value of the previous target period known at the vintage) is
#: optional when a :class:`BacktestResults` is built from an older table.
FORECAST_COLUMNS: tuple[str, ...] = (
    "vintage",
    "target_period",
    "offset",
    "kind",
    "months_to_end",
    "days_to_end",
    "days_to_release",
    "model",
    "forecast",
    "actual",
    "error",
    "previous_actual",
)
_KEY = ["vintage", "target_period"]
_VALUE_COLUMNS = ("model", "forecast", "actual", "error", "previous_actual")
_PREVIOUS = ("vintage", "final")
_AGGREGATES = (None, "target_period")
_PairTestResult = DieboldMarianoResult | GiacominiWhiteResult | ClarkWestResult
_WINDOWS = ("expanding", "rolling")
_ERRORS = ("raise", "warn")


# ====================================================================== information sets
def _calendar_for(panel: MixedFrequencyData, delay: Any, calendar: Any) -> ReleaseCalendar:
    """Release calendar from a delay specification (``None`` = panel metadata)."""
    if delay is not None and calendar is not None:
        raise ValueError("Pass either delay or calendar, not both.")
    spec = calendar if calendar is not None else delay
    if isinstance(spec, ReleaseCalendar):
        return spec
    if spec is None:
        return ReleaseCalendar.from_data(panel)
    if isinstance(spec, int | np.integer) and not isinstance(spec, bool):
        return ReleaseCalendar.from_data(panel, dict.fromkeys(panel.columns, int(spec)))
    if isinstance(spec, Mapping | pd.Series):
        return ReleaseCalendar.from_data(panel, spec)
    if isinstance(spec, Sequence | np.ndarray) and not isinstance(spec, str):
        values = list(spec)
        if len(values) != panel.n_series:
            raise ValueError("A delay sequence must have one value per series of the data.")
        return ReleaseCalendar.from_data(panel, dict(zip(panel.columns, values, strict=True)))
    raise ValueError(f"Invalid delay/calendar specification of type {type(spec).__name__}.")


@dataclass(frozen=True)
class _Source:
    """Where the information set of a date comes from (final data or a store)."""

    panel: MixedFrequencyData | None
    calendar: ReleaseCalendar | None
    store: VintageStore | None
    metadata: Mapping[str, Any]
    releases: dict[str, dict[pd.Period, pd.Timestamp]] = field(default_factory=dict)
    preprocess: Callable[[MixedFrequencyData], MixedFrequencyData] | None = None

    def raw(self, date: pd.Timestamp) -> MixedFrequencyData:
        """Information set at ``date`` as published (before ``preprocess``)."""
        if self.store is not None:
            out = self.store.as_of(date, as_mixed=True, **self.metadata)
        else:
            assert self.panel is not None  # noqa: S101
            out = pseudo_real_time(self.panel, calendar=self.calendar, vintage=date)
        assert isinstance(out, MixedFrequencyData)  # noqa: S101
        return out

    def prepare(self, panel: MixedFrequencyData) -> MixedFrequencyData:
        """Apply the per-vintage ``preprocess`` step (identity when absent)."""
        return panel if self.preprocess is None else self.preprocess(panel)

    def release_date(self, target: str, period: pd.Period) -> pd.Timestamp | None:
        if self.store is not None:
            if target not in self.releases:
                self.releases[target] = _first_release(self.store, target)
            return self.releases[target].get(period)
        assert self.calendar is not None  # noqa: S101
        try:
            return self.calendar.release_date(target, period)
        except NowcastDataError:
            return None


def _first_release(store: VintageStore, target: str) -> dict[pd.Period, pd.Timestamp]:
    """First release date of every period of ``target`` in a store."""
    records = store.records
    sub = records[(records["series"] == target) & records["value"].notna()]
    out: dict[pd.Period, pd.Timestamp] = {}
    for period, date in zip(sub["reference_period"], sub["vintage_date"], strict=True):
        key = pd.Period(period)
        if key not in out or date < out[key]:
            out[key] = pd.Timestamp(date)
    return out


# ====================================================================== forecasters
def _is_sklearn_estimator(obj: object) -> bool:
    module = type(obj).__module__ or ""
    return (
        hasattr(obj, "__sklearn_tags__")
        or hasattr(obj, "_estimator_type")
        or module.split(".")[0] == "sklearn"
    )


def _as_forecaster(obj: object) -> object:
    """Benchmark/model as given, or a scikit-learn regressor wrapped (I11)."""
    if isinstance(obj, BaseNowcaster | BaseBenchmark):
        return obj
    if _is_sklearn_estimator(obj):
        return SklearnBenchmark(obj)
    if callable(getattr(obj, "fit", None)) and callable(getattr(obj, "predict", None)):
        return obj
    raise TypeError(
        f"{type(obj).__name__} is neither a nowcaster, a benchmark nor an estimator with "
        "fit/predict."
    )


def _display_name(obj: object) -> str:
    name = getattr(obj, "name", None)
    return name if isinstance(name, str) and name else type(obj).__name__


def _clone(obj: Any) -> Any:
    clone = getattr(obj, "clone", None)
    if callable(clone):
        return clone()
    return copy.deepcopy(obj)


def _unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return out


def _estimates_from_frame(frame: pd.DataFrame, target: str, freq: Frequency) -> pd.Series:
    """Target values in the storage slots of a base-grid frame, on native periods."""
    column = frame[target]
    index = column.index
    assert isinstance(index, pd.PeriodIndex)  # noqa: S101
    # storage slot = last base period *ending* in the native period (calendar aware:
    # on a weekly grid not the week containing the last day of a quarter)
    slots = is_period_end(index, freq)
    out = column[slots]
    out.index = base_to_native(index[slots], freq)
    return out


class _Runner:
    """Fits (or updates) one forecaster along a block of vintages."""

    def __init__(self, template: Any, is_model: bool, fit_kwargs: Mapping[str, Any]) -> None:
        self.template = template
        self.is_model = is_model
        self.fit_kwargs = dict(fit_kwargs) if is_model else {}
        self.estimator: Any = None
        self.results: NowcastResults | None = None

    def forecast(
        self,
        panel: MixedFrequencyData,
        target: str,
        periods: pd.PeriodIndex,
        refit: bool,
    ) -> np.ndarray:
        if refit or self.estimator is None or not self.is_model:
            return self._fit(panel, target, periods)
        updated = self._update(panel)
        if updated is None:
            return self._fit(panel, target, periods)
        return updated.reindex(periods).to_numpy(dtype=float)

    def _fit(self, panel: MixedFrequencyData, target: str, periods: pd.PeriodIndex) -> np.ndarray:
        self.estimator = _clone(self.template)
        out = self.estimator.fit(panel, target, **self.fit_kwargs)
        if isinstance(out, NowcastResults):
            self.results = out
            return out.estimate.reindex(periods).to_numpy(dtype=float)
        self.results = None
        predicted = pd.Series(self.estimator.predict(periods))
        return predicted.to_numpy(dtype=float)

    def _update(self, panel: MixedFrequencyData) -> pd.Series | None:
        if self.results is None:
            return None
        update = getattr(self.estimator, "update", None)
        if callable(update):
            out = update(panel)
            if isinstance(out, NowcastResults):
                return out.estimate
        predict = getattr(self.results, "predict", None)
        if callable(predict):
            name = self.results.target
            frame = predict(panel)
            if isinstance(frame, pd.DataFrame) and name in frame:
                return _estimates_from_frame(frame, name, self.results.target_frequency)
        return None


# ====================================================================== run specification
@dataclass(frozen=True)
class _RunSpec:
    """Everything a worker needs to process a block of vintages."""

    source: _Source
    target: str
    target_name: str
    forecasters: list[tuple[str, Any, bool]]
    offsets: tuple[int, ...]
    include_released: bool
    window: str
    window_length: int | None
    fit_kwargs: Mapping[str, Any]
    errors: str


def _target_periods(
    spec: _RunSpec, panel: MixedFrequencyData, date: pd.Timestamp
) -> pd.PeriodIndex:
    freq = panel.metadata[spec.target_name].frequency.pandas_freq
    current = pd.Period(date, freq=freq)
    periods = pd.PeriodIndex([current + k for k in spec.offsets], freq=freq)
    if spec.include_released:
        return periods
    observed = panel.to_native(spec.target_name, dropna=True)
    released = periods.isin(observed.index)
    return periods[~released]


def _estimation_panel(
    spec: _RunSpec, panel: MixedFrequencyData, date: pd.Timestamp, periods: pd.PeriodIndex
) -> MixedFrequencyData:
    """Truncate to the estimation window and extend to the last target period."""
    base = panel.base_frequency.pandas_freq
    info_end = pd.Period(date, freq=base)
    needed = max(info_end, periods.asfreq(base, how="E").max())
    start = panel.start
    if spec.window == "rolling":
        assert spec.window_length is not None  # noqa: S101
        start = max(start, info_end - (spec.window_length - 1))
    out = panel.truncate(start, min(needed, panel.end))
    if needed > out.end:
        out = out.extend(int((needed - out.end).n))
    return out


def _labels(spec: _RunSpec, date: pd.Timestamp, period: pd.Period, offset: int) -> dict[str, Any]:
    month = pd.Period(date, freq="M")
    end = period.end_time.normalize()
    release = spec.source.release_date(spec.target_name, period)
    return {
        "vintage": date,
        "target_period": period,
        "offset": offset,
        "kind": "nowcast" if offset == 0 else ("backcast" if offset < 0 else "forecast"),
        "months_to_end": int(period.asfreq("M", how="E").ordinal - month.ordinal),
        "days_to_end": int((end - date).days),
        "days_to_release": np.nan if release is None else float((release - date).days),
    }


def _previous_values(
    panel: MixedFrequencyData, target: str, periods: pd.PeriodIndex
) -> list[float]:
    """Last value of ``target`` released at the vintage up to the period before each target."""
    observed = panel.to_native(target, dropna=True)
    out = []
    for period in periods:
        known = observed[observed.index <= period - 1]
        out.append(float(known.iloc[-1]) if len(known) else np.nan)
    return out


def _forecast_safely(
    runner: _Runner,
    spec: _RunSpec,
    args: tuple[MixedFrequencyData, str, pd.PeriodIndex, bool],
    failures: list[str],
    label: str,
) -> np.ndarray:
    try:
        return runner.forecast(*args)
    except Exception as err:
        if spec.errors == "raise":
            raise
        failures.append(f"{label}: {type(err).__name__}: {err}")
        runner.estimator = None
        return np.full(len(args[2]), np.nan)


def _run_block(
    spec: _RunSpec, dates: Sequence[pd.Timestamp]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Process consecutive vintages; the model is re-estimated at the first one."""
    runners = [
        (name, _Runner(est, is_model, spec.fit_kwargs)) for name, est, is_model in spec.forecasters
    ]
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    refit = True
    for date in dates:
        raw = spec.source.raw(date)
        panel = spec.source.prepare(raw)
        periods = _target_periods(spec, panel, date)
        if len(periods) == 0:
            continue
        window = _estimation_panel(spec, panel, date, periods)
        current = pd.Period(date, freq=periods.freqstr)
        previous = _previous_values(raw, spec.target_name, periods)
        labels = [
            {**_labels(spec, date, p, int((p - current).n)), "previous_actual": prev}
            for p, prev in zip(periods, previous, strict=True)
        ]
        for name, runner in runners:
            target = spec.target if runner.is_model else spec.target_name
            values = _forecast_safely(
                runner, spec, (window, target, periods, refit), failures, f"{name} @ {date.date()}"
            )
            records += [
                {**lab, "model": name, "forecast": float(v)}
                for lab, v in zip(labels, values, strict=True)
            ]
        refit = False
    return records, failures


def _run_parallel(
    func: Callable[[Sequence[pd.Timestamp]], Any],
    blocks: list[list[pd.Timestamp]],
    n_jobs: int | None,
) -> list[Any]:
    if n_jobs in (None, 1) or len(blocks) <= 1:
        return [func(block) for block in blocks]
    try:
        from joblib import Parallel, delayed
    except ImportError:  # pragma: no cover - joblib ships with scikit-learn/most stacks
        logger.info("joblib is not installed; running the backtest sequentially.")
        return [func(block) for block in blocks]
    return list(Parallel(n_jobs=n_jobs)(delayed(func)(block) for block in blocks))


# ====================================================================== backtest
class PseudoRealTimeBacktest:
    """Pseudo real-time (or real-time) evaluation of a nowcasting model and benchmarks.

    Parameters
    ----------
    model : BaseNowcaster or forecaster, optional
        Main model, e.g. :class:`~nowcastbox.models.MixedFreqDFM` (any object whose
        ``fit(data, target)`` returns :class:`~nowcastbox.core.results.NowcastResults`
        or that follows the benchmark ``fit``/``predict`` protocol). It is cloned at
        every re-estimation.
    data : MixedFrequencyData, pandas.DataFrame or VintageStore
        Final dataset (pseudo real-time) or a store of real-time vintages.
    target : str
        Target series, or a formula (``"gdp ~ ip + pmi"``) passed to the model;
        benchmarks receive the target name.
    calendar, delay : ReleaseCalendar or delay specification, optional
        Release rule of the pseudo real-time vintages (one of them; see
        :func:`nowcastbox.vintages.pseudo_real_time`). Default: the
        ``release_delays`` metadata. Not used with a :class:`VintageStore`.
    start, end : date-like
        First and last vintage dates.
    step : str, int, Timedelta or DateOffset, default "M"
        Spacing of the vintages (:func:`nowcastbox.vintages.vintage_dates`);
        ``"release"`` = one vintage per release (pseudo real time) or per store
        vintage date (real time).
    benchmarks : sequence or mapping of forecasters, optional
        Benchmarks (:mod:`nowcastbox.benchmarks`, any ``fit``/``predict`` forecaster or
        scikit-learn regressors, which are wrapped in
        :class:`~nowcastbox.benchmarks.SklearnBenchmark`). A mapping gives the names.
    window : {"expanding", "rolling"}, default "expanding"
        Estimation window.
    window_length : int, optional
        Length of the rolling window in base periods (months); required with
        ``window="rolling"``.
    refit_every : int, default 1
        Re-estimate the model's parameters every ``refit_every`` vintages (in between
        only the information is updated, see the module notes).
    n_jobs : int, optional
        Parallel jobs (joblib); ``None``/1 = sequential, -1 = all cores.
    frequency : frequency specification, optional
        Per-series frequencies for DataFrame input.
    target_offsets : sequence of int, default (-1, 0, 1)
        Target periods evaluated at each vintage, relative to the period containing
        the vintage date (-1 = backcast, 0 = nowcast, 1 = one-period-ahead forecast).
    include_released : bool, default False
        Also evaluate target periods already released at the vintage date.
    actual : {"final", "latest", "first"}, int or pandas.Series, optional
        Realisations: the final data / latest store release (default), the first
        release or the ``n``-th release (stores only), or explicit values indexed by
        target period.
    fit_kwargs : mapping, optional
        Extra keyword arguments of ``model.fit`` (e.g. ``{"horizon": 1}``).
    errors : {"raise", "warn"}, default "raise"
        Failure policy of a fit at one vintage: propagate, or record NaN and warn.
    metadata : mapping, optional
        Metadata keyword arguments of :class:`MixedFrequencyData` used with a
        :class:`VintageStore` (``release_delays``, ``blocks``...).
    model_name : str, optional
        Label of the main model in the results (default: its class name, e.g.
        ``"MixedFreqDFM"``); useful to compare several variants of one estimator.
    preprocess : callable, optional
        Function applied to each vintage's information set before estimation
        (e.g. outlier correction or interior-gap filling). Applying such cleaning
        per vintage, rather than once to the final data, avoids look-ahead bias.

    Raises
    ------
    ValueError
        If a parameter is invalid (raised by :meth:`run`).

    See Also
    --------
    BacktestResults : Accuracy tables and tests.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.benchmarks import AR, BridgeBenchmark
    >>> from nowcastbox.evaluation import PseudoRealTimeBacktest
    >>> rng = np.random.default_rng(0)
    >>> idx = pd.period_range("2000-01", periods=120, freq="M")
    >>> x = rng.standard_normal(120)
    >>> gdp = (0.5 + pd.Series(x, index=idx).rolling(3).mean()).where(idx.month % 3 == 0)
    >>> frame = pd.DataFrame({"x": x, "gdp": gdp}, index=idx)
    >>> bt = PseudoRealTimeBacktest(
    ...     data=frame,
    ...     target="gdp",
    ...     delay={"x": 20, "gdp": 45},
    ...     start="2008-01-15",
    ...     end="2009-12-15",
    ...     step="M",
    ...     benchmarks=[AR(p=1), BridgeBenchmark()],
    ...     frequency={"x": "M", "gdp": "Q"},
    ... )
    >>> res = bt.run()
    >>> res.models
    ['AR', 'BridgeBenchmark']
    >>> relative = res.relative_to("AR")  # RMSFE ratios by months to the quarter end
    >>> bool((relative.loc[[-1, 0], "BridgeBenchmark"] < 1).all())  # backcast and nowcast
    True
    """

    def __init__(
        self,
        model: Any = None,
        data: MixedFrequencyData | pd.DataFrame | VintageStore | None = None,
        target: str | None = None,
        *,
        calendar: Any = None,
        delay: Any = None,
        start: Any = None,
        end: Any = None,
        step: Any = "M",
        benchmarks: Sequence[Any] | Mapping[str, Any] | None = None,
        window: str = "expanding",
        window_length: int | None = None,
        refit_every: int = 1,
        n_jobs: int | None = None,
        frequency: FrequencySpec | None = None,
        target_offsets: Sequence[int] = (-1, 0, 1),
        include_released: bool = False,
        actual: str | int | pd.Series | None = None,
        fit_kwargs: Mapping[str, Any] | None = None,
        errors: str = "raise",
        metadata: Mapping[str, Any] | None = None,
        model_name: str | None = None,
        preprocess: Callable[[MixedFrequencyData], MixedFrequencyData] | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.target = target
        self.calendar = calendar
        self.delay = delay
        self.start = start
        self.end = end
        self.step = step
        self.benchmarks = benchmarks
        self.window = window
        self.window_length = window_length
        self.refit_every = refit_every
        self.n_jobs = n_jobs
        self.frequency = frequency
        self.target_offsets = target_offsets
        self.include_released = include_released
        self.actual = actual
        self.fit_kwargs = fit_kwargs
        self.errors = errors
        self.metadata = metadata
        self.model_name = model_name
        self.preprocess = preprocess

    # ------------------------------------------------------------------ validation
    def _check_int(self, name: str, minimum: int) -> int:
        value = getattr(self, name)
        if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
        return int(value)

    def _validate(self) -> None:
        if self.data is None or self.target is None:
            raise ValueError("data and target are required.")
        if self.start is None or self.end is None:
            raise ValueError("start and end are required.")
        if self.window not in _WINDOWS:
            raise ValueError(f"window must be one of {_WINDOWS}, got {self.window!r}.")
        if self.window == "rolling":
            self._check_int("window_length", 2)
        if self.errors not in _ERRORS:
            raise ValueError(f"errors must be one of {_ERRORS}, got {self.errors!r}.")
        self._check_int("refit_every", 1)
        offsets = list(self.target_offsets)
        if not offsets or not all(isinstance(o, int | np.integer) for o in offsets):
            raise ValueError("target_offsets must be a non-empty sequence of integers.")
        if self.model is None and not self.benchmarks:
            raise ValueError("Give a model and/or benchmarks to evaluate.")
        if self.model_name is not None and (
            not isinstance(self.model_name, str) or not self.model_name
        ):
            raise ValueError(f"model_name must be a non-empty string, got {self.model_name!r}.")

    # ------------------------------------------------------------------ setup
    def _source(self) -> _Source:
        meta = dict(self.metadata or {})
        if isinstance(self.data, VintageStore):
            if self.calendar is not None or self.delay is not None:
                raise ValueError("calendar/delay cannot be used with a VintageStore.")
            if self.frequency is not None:
                raise ValueError("frequency cannot be used with a VintageStore.")
            return _Source(None, None, self.data, meta, preprocess=self.preprocess)
        if meta:
            raise ValueError("metadata is only used with a VintageStore.")
        assert self.data is not None  # noqa: S101
        panel = as_mixed_frequency_data(self.data, self.frequency)
        calendar = _calendar_for(panel, self.delay, self.calendar)
        return _Source(panel, calendar, None, {}, preprocess=self.preprocess)

    def _forecasters(self) -> list[tuple[str, Any, bool]]:
        items: list[tuple[str, Any, bool]] = []
        if self.model is not None:
            model = _as_forecaster(self.model)
            items.append((self.model_name or _display_name(model), model, True))
        bench = self.benchmarks or []
        if isinstance(bench, Mapping):
            pairs = [(str(k), _as_forecaster(v)) for k, v in bench.items()]
        else:
            pairs = [(_display_name(b), b) for b in map(_as_forecaster, bench)]
        items += [(name, est, False) for name, est in pairs]
        names = _unique([name for name, _, _ in items])
        return [(n, est, is_model) for n, (_, est, is_model) in zip(names, items, strict=True)]

    def _dates(self, source: _Source) -> pd.DatetimeIndex:
        if source.store is not None and isinstance(self.step, str) and self.step == "release":
            dates = vintage_dates(self.start, self.end, "D")
            stored = source.store.vintage_dates()
            keep = stored[(stored > dates[0]) & (stored <= dates[-1])]
            return pd.DatetimeIndex([dates[0], *keep], name="vintage")
        return vintage_dates(
            self.start, self.end, self.step, calendar=source.calendar, data=source.panel
        )

    def _columns(self, source: _Source) -> list[str]:
        if source.store is not None:
            return source.store.series
        assert source.panel is not None  # noqa: S101
        return source.panel.columns

    # ------------------------------------------------------------------ actuals
    def _actuals(self, source: _Source, target: str, freq: Frequency) -> pd.Series:
        spec = self.actual
        if isinstance(spec, pd.Series):
            index = pd.PeriodIndex([pd.Period(p, freq=freq.pandas_freq) for p in spec.index])
            return pd.Series(spec.to_numpy(dtype=float), index=index)
        if source.store is None:
            if spec not in (None, "final", "latest", "first"):
                raise ValueError(
                    "With pseudo real-time data actual must be 'final', 'first' (no revisions) "
                    "or a Series."
                )
            assert source.panel is not None  # noqa: S101
            return source.panel.to_native(target, dropna=True)
        return self._store_actuals(source.store, target, freq)

    def _store_actuals(self, store: VintageStore, target: str, freq: Frequency) -> pd.Series:
        spec = self.actual
        if spec is None or (isinstance(spec, str) and spec in ("final", "latest")):
            frame = store.latest(series=target)
        elif isinstance(spec, str) and spec == "first":
            frame = store.nth_release(0, series=target)
        elif isinstance(spec, int | np.integer) and not isinstance(spec, bool):
            frame = store.nth_release(int(spec), series=target)
        else:
            raise ValueError(f"Invalid actual specification {spec!r}.")
        assert isinstance(frame, pd.DataFrame)  # noqa: S101
        return _estimates_from_frame(frame, target, freq).dropna()

    # ------------------------------------------------------------------ run
    def run(self, n_jobs: int | None = None) -> BacktestResults:
        """Run the backtest.

        Parameters
        ----------
        n_jobs : int, optional
            Overrides the ``n_jobs`` given at construction.

        Returns
        -------
        BacktestResults
            Forecasts of every model at every vintage, with accuracy tools.

        Raises
        ------
        ValueError
            If a parameter is invalid.
        NowcastDataError
            If the data are invalid or (with ``errors="raise"``) a model fails.

        Warns
        -----
        NowcastBoxWarning
            With ``errors="warn"``, if some fits failed.

        Examples
        --------
        >>> # see the class docstring
        """
        self._validate()
        source = self._source()
        assert self.target is not None  # noqa: S101
        target_name, _ = resolve_target(self.target, self._columns(source))
        forecasters = self._forecasters()
        dates = self._dates(source)
        spec = _RunSpec(
            source=source,
            target=self.target,
            target_name=target_name,
            forecasters=forecasters,
            offsets=tuple(int(o) for o in self.target_offsets),
            include_released=bool(self.include_released),
            window=self.window,
            window_length=None if self.window_length is None else int(self.window_length),
            fit_kwargs=dict(self.fit_kwargs or {}),
            errors=self.errors,
        )
        k = int(self.refit_every)
        blocks = [list(dates[i : i + k]) for i in range(0, len(dates), k)]
        jobs = self.n_jobs if n_jobs is None else n_jobs
        logger.info("Backtest: %d vintages, %d forecasters.", len(dates), len(forecasters))
        outputs = _run_parallel(functools.partial(_run_block, spec), blocks, jobs)
        records = [r for recs, _ in outputs for r in recs]
        failures = [f for _, fails in outputs for f in fails]
        if failures:
            warnings.warn(
                f"{len(failures)} forecaster fits failed (NaN recorded); first: {failures[0]}",
                NowcastBoxWarning,
                stacklevel=2,
            )
        freq = self._target_frequency(source, target_name)
        frame = _assemble(records, self._actuals(source, target_name, freq), freq)
        return BacktestResults(
            forecasts=frame,
            models=[name for name, _, _ in forecasters],
            target=target_name,
            reference=next((n for n, _, is_model in forecasters if not is_model), None),
            info={
                "n_vintages": len(dates),
                "window": self.window,
                "refit_every": k,
                "real_time": source.store is not None,
                "failures": failures,
            },
        )

    @staticmethod
    def _target_frequency(source: _Source, target: str) -> Frequency:
        if source.store is not None:
            return Frequency.from_value(source.store.frequencies[target])
        assert source.panel is not None  # noqa: S101
        return source.panel.metadata[target].frequency


def _assemble(records: list[dict[str, Any]], actuals: pd.Series, freq: Frequency) -> pd.DataFrame:
    recorded = [c for c in FORECAST_COLUMNS if c not in ("actual", "error")]
    frame = pd.DataFrame.from_records(records, columns=recorded)
    periods = pd.PeriodIndex(frame["target_period"], freq=freq.pandas_freq)
    frame["target_period"] = periods
    frame["actual"] = actuals.reindex(periods).to_numpy(dtype=float)
    frame["error"] = frame["actual"] - frame["forecast"]
    frame["vintage"] = pd.to_datetime(frame["vintage"])
    frame["previous_actual"] = frame["previous_actual"].astype(float)
    frame = frame[list(FORECAST_COLUMNS)]
    return frame.sort_values(["vintage", "target_period"], kind="stable").reset_index(drop=True)


# ====================================================================== sub-periods
_Interval = tuple[pd.Period | None, pd.Period | None]
_PERIOD_SHORTCUTS = ("covid", "ex-covid")


def _to_period(value: Any, freq: str | None) -> pd.Period:
    """A period of frequency ``freq`` from a period-like (a coarser period is converted)."""
    if isinstance(value, pd.Period):
        return value if freq is None else value.asfreq(freq)
    return pd.Period(value, freq=freq)


def covid_periods(
    freq: str = "Q", window: tuple[Any, Any] = COVID_WINDOW
) -> dict[str, list[tuple[str | None, str | None]]]:
    """Sub-periods around the Covid-19 pandemic, as target periods of frequency ``freq``.

    The pandemic window (default: March 2020 to December 2021, the
    :data:`~nowcastbox.models.robust.COVID_WINDOW` of the robust DFM) is converted to the
    target frequency: every target period that overlaps it is a Covid period (2020Q1 to
    2021Q4 for a quarterly target).

    Parameters
    ----------
    freq : str, default "Q"
        Frequency of the target periods (pandas alias).
    window : tuple of two period-likes, default ("2020-03", "2021-12")
        First and last (base) period of the pandemic.

    Returns
    -------
    dict
        ``{"pre-Covid", "Covid", "post-Covid", "ex-Covid"}`` mapped to lists of
        ``(first, last)`` target periods as strings (``None`` = open end); "ex-Covid"
        is the union of the pre- and post-Covid intervals. This is the
        ``periods="covid"`` shortcut of :class:`BacktestResults`.

    Raises
    ------
    ValueError
        If the window ends before it starts.

    Examples
    --------
    >>> from nowcastbox.evaluation import covid_periods
    >>> covid_periods("Q")["Covid"]
    [('2020Q1', '2021Q4')]
    >>> covid_periods("Q")["ex-Covid"]
    [(None, '2019Q4'), ('2022Q1', None)]
    """
    first, last = _to_period(window[0], freq), _to_period(window[1], freq)
    if first > last:
        raise ValueError(f"The Covid window ends before it starts: {window!r}.")
    pre, post = str(first - 1), str(last + 1)
    return {
        "pre-Covid": [(None, pre)],
        "Covid": [(str(first), str(last))],
        "post-Covid": [(post, None)],
        "ex-Covid": [(None, pre), (post, None)],
    }


def _is_pair(value: Any) -> bool:
    return (
        isinstance(value, tuple | list)
        and len(value) == 2
        and not any(isinstance(v, tuple | list) for v in value)
    )


def _intervals(label: str, value: Any, freq: str | None) -> list[_Interval]:
    """Validated ``(first, last)`` target-period intervals of one sub-period."""
    pairs = [value] if _is_pair(value) else value
    if not isinstance(pairs, tuple | list) or not pairs or not all(map(_is_pair, pairs)):
        raise ValueError(
            f"Sub-period {label!r} must be a (first, last) pair or a list of pairs, got {value!r}."
        )
    out: list[_Interval] = []
    for first, last in pairs:
        lo = None if first is None else _to_period(first, freq)
        hi = None if last is None else _to_period(last, freq)
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"Sub-period {label!r} ends before it starts: {(first, last)!r}.")
        out.append((lo, hi))
    return out


def _resolve_periods(periods: Any, freq: str | None) -> dict[str, list[_Interval]]:
    """Sub-period specification (shortcut or mapping) as intervals of target periods."""
    if isinstance(periods, str):
        key = periods.lower()
        if key not in _PERIOD_SHORTCUTS:
            raise ValueError(
                f"periods shortcut must be one of {_PERIOD_SHORTCUTS}, got {periods!r}."
            )
        spec: dict[Any, Any] = covid_periods(freq or "Q")
        if key == "ex-covid":
            spec = {"ex-Covid": spec["ex-Covid"]}
    elif isinstance(periods, Mapping):
        spec = dict(periods)
    else:
        raise ValueError(
            f"periods must be a mapping {{label: (first, last)}}, 'covid' or 'ex-covid', got "
            f"{type(periods).__name__}."
        )
    if not spec:
        raise ValueError("periods must define at least one sub-period.")
    return {str(label): _intervals(str(label), value, freq) for label, value in spec.items()}


def _period_mask(index: pd.PeriodIndex, intervals: list[_Interval]) -> np.ndarray:
    """Rows whose target period falls in the union of the intervals."""
    mask = np.zeros(len(index), dtype=bool)
    for first, last in intervals:
        inside = np.ones(len(index), dtype=bool)
        if first is not None:
            inside &= np.asarray(index >= first)
        if last is not None:
            inside &= np.asarray(index <= last)
        mask |= inside
    return mask


def _target_index(frame: pd.DataFrame) -> pd.PeriodIndex:
    """Target periods of a forecast table (also when stored as strings, e.g. Parquet)."""
    column = frame["target_period"]
    if isinstance(column.dtype, pd.PeriodDtype):
        return pd.PeriodIndex(column)
    return pd.PeriodIndex([pd.Period(str(v)) for v in column])


def _actual_by_period(frame: pd.DataFrame) -> pd.Series:
    """Realisation of every target period of a forecast table."""
    values = pd.Series(frame["actual"].to_numpy(dtype=float), index=_target_index(frame))
    return values.groupby(level=0).first()


def _final_previous(frame: pd.DataFrame, actuals: pd.Series | None = None) -> np.ndarray:
    """Final value of the period before each row's target period."""
    if "previous_final" in frame:
        return frame["previous_final"].to_numpy(dtype=float)
    if frame.empty:
        return np.empty(0)
    source = _actual_by_period(frame) if actuals is None else actuals
    return source.reindex(_target_index(frame) - 1).to_numpy(dtype=float)


def _check_previous(previous: str) -> None:
    if previous not in _PREVIOUS:
        raise ValueError(f"previous must be one of {_PREVIOUS}, got {previous!r}.")


# ====================================================================== results
@dataclass(frozen=True)
class BacktestResults:
    """Forecasts of a backtest and the tools to evaluate them.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Long table, columns :data:`FORECAST_COLUMNS` (``previous_actual`` is optional:
        tables written by older versions, without it, still work, and directional
        accuracy is then available with ``previous="final"``).
    models : list of str
        Model names (main model first, then benchmarks).
    target : str
        Target series.
    reference : str, optional
        Default reference model of the comparisons (the first benchmark).
    info : dict
        Run information.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame(
    ...     {
    ...         "vintage": pd.to_datetime(["2020-01-15"] * 2),
    ...         "target_period": pd.PeriodIndex(["2020Q1"] * 2, freq="Q"),
    ...         "offset": 0,
    ...         "kind": "nowcast",
    ...         "months_to_end": 2,
    ...         "days_to_end": 76,
    ...         "days_to_release": 120.0,
    ...         "model": ["M", "AR"],
    ...         "forecast": [1.0, 0.0],
    ...         "actual": 1.5,
    ...         "error": [0.5, 1.5],
    ...     }
    ... )
    >>> res = BacktestResults(frame, ["M", "AR"], "gdp", reference="AR")
    >>> res.relative_to().round(4).to_dict()
    {'M': {2: 0.3333}, 'AR': {2: 1.0}}
    """

    forecasts: pd.DataFrame
    models: list[str]
    target: str
    reference: str | None = None
    info: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ frames
    def to_frame(self, wide: bool = False) -> pd.DataFrame:
        """Forecast table (copy).

        Parameters
        ----------
        wide : bool, default False
            Long format (one row per vintage, target period and model) or wide format
            (one row per vintage and target period, one column per model plus
            ``actual`` and, when available, ``previous_actual``).

        Returns
        -------
        pandas.DataFrame
            Forecasts.

        Examples
        --------
        >>> res.to_frame(wide=True).columns.tolist()  # doctest: +SKIP
        ['M', 'AR', 'actual', 'previous_actual']
        """
        frame = self.forecasts.copy()
        if not wide:
            return frame
        labels = [c for c in FORECAST_COLUMNS if c not in _VALUE_COLUMNS]
        indexed = frame.set_index([*labels, "model"])
        table = indexed["forecast"].unstack("model")  # noqa: PD010
        table = table.reindex(columns=[m for m in self.models if m in table.columns])
        for column in ("actual", "previous_actual"):
            if column in indexed:
                values = indexed[column].groupby(level=labels, dropna=False).first()
                table[column] = values.reindex(table.index)
        table.columns.name = None
        return table

    def to_parquet(self, path: str | Path) -> Path:
        """Write the forecast table to Parquet (periods stored as strings).

        Parameters
        ----------
        path : str or pathlib.Path
            Output file.

        Returns
        -------
        pathlib.Path
            The written path.

        Raises
        ------
        ImportError
            If no Parquet engine (pyarrow) is installed.

        Examples
        --------
        >>> res.to_parquet("backtest.parquet")  # doctest: +SKIP
        """
        out = Path(path)
        frame = self.forecasts.copy()
        frame["target_period"] = frame["target_period"].astype(str)
        frame.to_parquet(out, index=False)
        return out

    def _models(self, models: Sequence[str] | None) -> list[str]:
        chosen = list(self.models if models is None else models)
        unknown = [m for m in chosen if m not in self.models]
        if unknown:
            raise ValueError(f"Unknown models {unknown}; available: {self.models}.")
        return chosen

    def _reference(self, reference: str | None) -> str:
        ref = self.reference if reference is None else reference
        if ref is None:
            raise ValueError("No reference model: pass reference=<model name>.")
        self._models([ref])
        return ref

    def evaluable(
        self, models: Sequence[str] | None = None, *, common_sample: bool = True
    ) -> pd.DataFrame:
        """Rows with an actual value and a forecast (optionally on a common sample).

        Parameters
        ----------
        models : sequence of str, optional
            Models to keep (default: all).
        common_sample : bool, default True
            Keep only (vintage, target period) pairs forecast by every selected model.

        Returns
        -------
        pandas.DataFrame
            Subset of the forecast table.

        Raises
        ------
        ValueError
            If a model is unknown.

        Examples
        --------
        >>> res.evaluable(["M"]).shape[0]  # doctest: +SKIP
        1
        """
        chosen = self._models(models)
        frame = self.forecasts[self.forecasts["model"].isin(chosen)]
        frame = frame[frame["error"].notna()]
        if common_sample and len(chosen) > 1:
            counts = frame.groupby(_KEY)["model"].transform("nunique")
            frame = frame[counts == len(chosen)]
        return frame.copy()

    # ------------------------------------------------------------------ sub-periods
    def split_periods(self, periods: str | Mapping[str, Any]) -> dict[str, BacktestResults]:
        """Results restricted to sub-periods of target periods.

        Parameters
        ----------
        periods : mapping or {"covid", "ex-covid"}
            ``{label: (first, last)}`` with target periods (strings, periods or dates;
            ``None`` = open end) or a list of such pairs (union); ``"covid"`` gives the
            pre-Covid, Covid, post-Covid and ex-Covid sub-periods of
            :func:`covid_periods` and ``"ex-covid"`` only the last one.

        Returns
        -------
        dict of str to BacktestResults
            One result per non-empty sub-period (in the given order). Directional
            accuracy with ``previous="final"`` still uses the actual value of a
            previous period that lies outside the sub-period.

        Raises
        ------
        ValueError
            If the specification is invalid or no forecast falls in any sub-period.

        Warns
        -----
        DataQualityWarning
            If a sub-period has no forecast (it is skipped).

        Examples
        --------
        >>> parts = res.split_periods({"2020": ("2020Q1", "2020Q4")})  # doctest: +SKIP
        >>> parts["2020"].metrics()  # doctest: +SKIP
        """
        index = _target_index(self.forecasts)
        spec = _resolve_periods(periods, index.freqstr)
        with_final = self.forecasts.assign(previous_final=_final_previous(self.forecasts))
        out: dict[str, BacktestResults] = {}
        for label, intervals in spec.items():
            mask = _period_mask(index, intervals)
            if not mask.any():
                warnings.warn(
                    f"No forecast has a target period in sub-period {label!r}; skipped.",
                    DataQualityWarning,
                    stacklevel=2,
                )
                continue
            out[label] = dataclasses.replace(self, forecasts=with_final[mask])
        if not out:
            raise ValueError("No forecast has a target period in any of the sub-periods.")
        return out

    def _by_periods(self, periods: Any, method: str, **kwargs: Any) -> pd.DataFrame:
        """Call a table method on every sub-period and stack the results (level ``period``)."""
        parts = {
            label: getattr(sub, method)(**kwargs)
            for label, sub in self.split_periods(periods).items()
        }
        return pd.concat(parts, names=["period"])

    def _evaluation_frame(
        self,
        models: Sequence[str] | None,
        common_sample: bool,
        metrics: Sequence[str],
        previous: str,
    ) -> pd.DataFrame:
        """Evaluable rows, with the previous values when a directional metric is asked."""
        _check_previous(previous)
        frame = self.evaluable(models, common_sample=common_sample)
        if not any(m in DIRECTIONAL_METRICS for m in metrics):
            return frame
        if previous == "final":
            actuals = _actual_by_period(self.forecasts)
            return frame.assign(previous_actual=_final_previous(frame, actuals))
        if "previous_actual" not in frame:
            raise ValueError(
                "The forecast table has no 'previous_actual' column (written by an older "
                "version?): use previous='final'."
            )
        return frame

    # ------------------------------------------------------------------ accuracy
    def metrics(
        self,
        horizon: str | None = "months_to_end",
        metrics: Sequence[str] = ("rmsfe", "mae", "bias", "n"),
        *,
        models: Sequence[str] | None = None,
        common_sample: bool = True,
        previous: str = "vintage",
        periods: str | Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Accuracy metrics by horizon (optionally by sub-period).

        Parameters
        ----------
        horizon : str or None, default "months_to_end"
            Grouping column (``"months_to_end"``, ``"days_to_end"``,
            ``"days_to_release"``, ``"offset"``, ``"kind"``...) or ``None`` (pooled).
        metrics : sequence of str, default ("rmsfe", "mae", "bias", "n")
            Metrics (see :data:`nowcastbox.evaluation.metrics.METRICS`) and directional
            metrics (``"fda"``, :data:`~nowcastbox.evaluation.metrics.DIRECTIONAL_METRICS`).
        models : sequence of str, optional
            Models (default: all).
        common_sample : bool, default True
            Use only target periods/vintages forecast by every model.
        previous : {"vintage", "final"}, default "vintage"
            Previous value of the directional metrics: the value of the previous target
            period known at the vintage (column ``previous_actual``; the last released
            value when that period is not yet published) or its final value.
        periods : mapping or {"covid", "ex-covid"}, optional
            Sub-periods of target periods (see :meth:`split_periods`); the index then
            gains an outer level ``period``.

        Returns
        -------
        pandas.DataFrame
            Index: horizon (or ``(period, horizon)``); columns ``(metric, model)``.

        Raises
        ------
        ValueError
            If a metric, model, ``previous`` or ``periods`` is invalid, or ``"fda"``
            is asked with ``previous="vintage"`` on a table without ``previous_actual``.

        Examples
        --------
        >>> res.metrics(horizon=None)  # doctest: +SKIP
        >>> res.metrics("kind", ("rmsfe", "fda"), periods="covid")  # doctest: +SKIP
        """
        if periods is not None:
            options = {"models": models, "common_sample": common_sample, "previous": previous}
            return self._by_periods(periods, "metrics", horizon=horizon, metrics=metrics, **options)
        frame = self._evaluation_frame(models, common_sample, metrics, previous)
        table = accuracy_by_horizon(frame, horizon, metrics)
        return table.reindex(columns=self._models(models), level="model")

    def _metric(
        self,
        metric: str,
        horizon: str | None,
        models: Sequence[str] | None,
        common: bool,
        previous: str = "vintage",
    ) -> pd.DataFrame:
        frame = self._evaluation_frame(models, common, (metric,), previous)
        table = metric_by_horizon(frame, metric, horizon)
        return table.reindex(columns=self._models(models))

    def rmsfe_by_horizon(
        self,
        horizon: str | None = "months_to_end",
        *,
        models: Sequence[str] | None = None,
        common_sample: bool = True,
        periods: str | Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """RMSFE of every model by nowcast horizon.

        Parameters
        ----------
        horizon : str or None, default "months_to_end"
            Grouping column (``None``: pooled).
        models : sequence of str, optional
            Models (default: all).
        common_sample : bool, default True
            Use only target periods/vintages forecast by every model.
        periods : mapping or {"covid", "ex-covid"}, optional
            Sub-periods of target periods (see :meth:`split_periods`).

        Returns
        -------
        pandas.DataFrame
            Index: horizon (or ``(period, horizon)``); columns: models.

        Examples
        --------
        >>> res.rmsfe_by_horizon()  # doctest: +SKIP
        """
        if periods is not None:
            return self._by_periods(
                periods,
                "rmsfe_by_horizon",
                horizon=horizon,
                models=models,
                common_sample=common_sample,
            )
        return self._metric("rmsfe", horizon, models, common_sample)

    def relative_to(
        self,
        reference: str | None = None,
        horizon: str | None = "months_to_end",
        *,
        metric: str = "rmsfe",
        models: Sequence[str] | None = None,
        periods: str | Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Metric of every model divided by that of a reference (common sample).

        Parameters
        ----------
        reference : str, optional
            Reference model (default: :attr:`reference`, the first benchmark).
        horizon : str or None, default "months_to_end"
            Grouping column (``None``: pooled).
        metric : str, default "rmsfe"
            Metric (``"rmsfe"``, ``"mae"``, ``"mse"``).
        models : sequence of str, optional
            Models (default: all; the reference is always included).
        periods : mapping or {"covid", "ex-covid"}, optional
            Sub-periods of target periods (see :meth:`split_periods`).

        Returns
        -------
        pandas.DataFrame
            Ratios (< 1: better than the reference).

        Raises
        ------
        ValueError
            If there is no reference or a model is unknown.

        Examples
        --------
        >>> res.relative_to("AR")  # doctest: +SKIP
        >>> res.relative_to("AR", "kind", periods="ex-covid")  # doctest: +SKIP
        """
        if periods is not None:
            options = {"metric": metric, "models": models}
            return self._by_periods(
                periods, "relative_to", reference=reference, horizon=horizon, **options
            )
        ref = self._reference(reference)
        chosen = self._models(models)
        if ref not in chosen:
            chosen.append(ref)
        table = self._metric(metric, horizon, chosen, True)
        return table.div(table[ref], axis=0)

    def directional_accuracy(
        self,
        horizon: str | None = "months_to_end",
        *,
        models: Sequence[str] | None = None,
        common_sample: bool = True,
        previous: str = "vintage",
        test: bool = True,
        alternative: str = "greater",
        periods: str | Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        r"""Forecast directional accuracy and Pesaran-Timmermann test by horizon.

        A forecast scores when it predicts correctly whether the target rises or falls
        with respect to the previous value, :math:`(y_t - y^p_t)(\hat y_t - y^p_t) > 0`
        (see :func:`~nowcastbox.evaluation.directional_accuracy`); the Pesaran &
        Timmermann (1992) test compares the hit rate with the one expected if the
        predicted and actual directions were independent.

        Parameters
        ----------
        horizon : str or None, default "months_to_end"
            Grouping column (``None``: pooled).
        models : sequence of str, optional
            Models (default: all).
        common_sample : bool, default True
            Use only target periods/vintages forecast by every model.
        previous : {"vintage", "final"}, default "vintage"
            Previous value: known at the vintage (``previous_actual``) or final.
        test : bool, default True
            Add the Pesaran-Timmermann statistic and p-value.
        alternative : {"greater", "two-sided", "less"}, default "greater"
            Alternative of the test (``"greater"``: better than chance).
        periods : mapping or {"covid", "ex-covid"}, optional
            Sub-periods of target periods (see :meth:`split_periods`).

        Returns
        -------
        pandas.DataFrame
            Index ``(model, horizon)`` (or ``(period, model, horizon)``); columns
            ``fda`` and ``n_obs`` and, with ``test=True``, ``statistic``, ``pvalue``
            and ``expected_hit_rate`` (NaN with fewer than 3 forecasts or when the
            directions do not vary).

        Raises
        ------
        ValueError
            If ``previous``, ``alternative`` or a model is invalid, or the table has no
            ``previous_actual`` column and ``previous="vintage"``.

        Examples
        --------
        >>> import pandas as pd
        >>> from nowcastbox.evaluation import BacktestResults
        >>> frame = pd.DataFrame(
        ...     {
        ...         "vintage": pd.to_datetime(["2020-01-15"] * 4),
        ...         "target_period": pd.PeriodIndex(["2019Q4"] * 2 + ["2020Q1"] * 2, freq="Q"),
        ...         "months_to_end": [-1, -1, 2, 2],
        ...         "model": ["M", "AR"] * 2,
        ...         "forecast": [1.2, 0.8, 0.4, 0.6],
        ...         "actual": [1.5, 1.5, 0.5, 0.5],
        ...         "previous_actual": [1.0, 1.0, 1.0, 1.0],
        ...     }
        ... ).assign(error=lambda f: f["actual"] - f["forecast"])
        >>> res = BacktestResults(frame, ["M", "AR"], "gdp")
        >>> res.directional_accuracy(horizon=None, test=False)["fda"].to_dict()
        {('M', 'all'): 1.0, ('AR', 'all'): 0.5}
        """
        if periods is not None:
            options = {"models": models, "common_sample": common_sample, "test": test}
            return self._by_periods(
                periods,
                "directional_accuracy",
                horizon=horizon,
                previous=previous,
                alternative=alternative,
                **options,
            )
        if alternative not in ("greater", "two-sided", "less"):
            raise ValueError(f"Invalid alternative {alternative!r}.")
        chosen = self._models(models)
        frame = self._evaluation_frame(chosen, common_sample, ("fda",), previous)
        work = frame.assign(_h="all") if horizon is None else frame.assign(_h=frame[horizon])
        rows = [
            {"model": model, "horizon": label, **_directional_row(group, test, alternative)}
            for model in chosen
            for label, group in work[work["model"] == model].groupby("_h", sort=True, dropna=False)
        ]
        columns = ["model", "horizon", "fda", "n_obs"]
        if test:
            columns += ["statistic", "pvalue", "expected_hit_rate"]
        return pd.DataFrame(rows, columns=columns).set_index(["model", "horizon"])

    # ------------------------------------------------------------------ tests
    def _paired(
        self, model: str, reference: str, horizon: str | None
    ) -> list[tuple[Any, pd.DataFrame]]:
        """Aligned errors (sorted by target period, vintage) per horizon group."""
        frame = self.evaluable([model, reference])
        wide = frame.pivot_table(
            index=[*_KEY, *([horizon] if horizon else [])],
            columns="model",
            values="error",
            aggfunc="first",
        ).dropna()
        wide = wide.sort_index(level=["target_period", "vintage"])
        if horizon is None:
            return [("all", wide)]
        return list(wide.groupby(level=horizon, sort=True))

    def _pairwise_table(
        self,
        reference: str | None,
        horizon: str | None,
        test: Callable[[np.ndarray, np.ndarray], _PairTestResult],
        pair_loss: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
        aggregate: str | None,
    ) -> pd.DataFrame:
        """Run ``test`` on the (optionally per-period averaged) losses of each model."""
        ref = self._reference(reference)
        _check_aggregate(aggregate)
        rows = []
        for model in self.models:
            if model == ref:
                continue
            for label, group in self._paired(model, ref, horizon):
                l_model, l_ref = pair_loss(group[model].to_numpy(), group[ref].to_numpy())
                if aggregate is not None:
                    losses = pd.DataFrame(
                        {"model": l_model, "ref": l_ref},
                        index=group.index.get_level_values(aggregate),
                    )
                    losses = losses.groupby(level=0, sort=True).mean()
                    l_model, l_ref = losses["model"].to_numpy(), losses["ref"].to_numpy()
                rows.append({"model": model, "horizon": label, **_test_row(test, l_model, l_ref)})
        columns = ["model", "horizon", "statistic", "pvalue", "mean_loss_differential", "n_obs"]
        return pd.DataFrame(rows, columns=columns).set_index(["model", "horizon"])

    def diebold_mariano(
        self,
        reference: str | None = None,
        horizon: str | None = "months_to_end",
        *,
        loss: LossLike = "squared",
        h: int = 1,
        hln: bool = True,
        alternative: str = "two-sided",
        aggregate: str | None = None,
        periods: str | Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        r"""Diebold-Mariano (HLN) tests of every model against a reference.

        The loss differential is :math:`L(e_{\text{model}}) - L(e_{\text{reference}})`
        (negative statistic: the model is more accurate), computed per horizon group
        over target periods (ordered in time).

        When a horizon group holds several vintages per target period (e.g. the three
        monthly nowcasts of a quarter grouped by ``"kind"``, or ``horizon=None``), the
        differentials of the same target period are strongly correlated and the
        truncated long-run variance with ``h=1`` overstates significance. Pass
        ``aggregate="target_period"`` to average the losses within each target period
        first (one differential per period) and set ``h`` to the number of target
        periods ahead of the longest forecast in the group.

        Parameters
        ----------
        reference : str, optional
            Reference model (default: the first benchmark).
        horizon : str or None, default "months_to_end"
            Grouping column (``None``: pooled).
        loss : {"squared", "absolute"} or callable, default "squared"
            Loss function.
        h : int, default 1
            Horizon of the long-run variance (see :func:`diebold_mariano`).
        hln : bool, default True
            Harvey-Leybourne-Newbold correction.
        alternative : {"two-sided", "less", "greater"}, default "two-sided"
            Alternative hypothesis.
        aggregate : {None, "target_period"}, default None
            Average the losses within each target period before testing.
        periods : mapping or {"covid", "ex-covid"}, optional
            Sub-periods of target periods (see :meth:`split_periods`); the index then
            starts with a ``period`` level.

        Returns
        -------
        pandas.DataFrame
            Index ``(model, horizon)`` (or ``(period, model, horizon)``); columns
            ``statistic``, ``pvalue``, ``mean_loss_differential``, ``n_obs`` (NaN when
            fewer than 3 pairs).

        Raises
        ------
        ValueError
            If ``aggregate`` or ``periods`` is invalid or the reference is unknown.

        Examples
        --------
        >>> res.diebold_mariano("AR")  # doctest: +SKIP
        >>> res.diebold_mariano("AR", "kind", aggregate="target_period", h=2)  # doctest: +SKIP
        """
        if periods is not None:
            options = {"loss": loss, "h": h, "hln": hln, "alternative": alternative}
            return self._by_periods(
                periods,
                "diebold_mariano",
                reference=reference,
                horizon=horizon,
                aggregate=aggregate,
                **options,
            )

        def test(l1: np.ndarray, l2: np.ndarray) -> DieboldMarianoResult:
            return diebold_mariano(l1, l2, h=h, loss=_identity, alternative=alternative, hln=hln)

        return self._pairwise_table(reference, horizon, test, _loss_pair(loss), aggregate)

    def clark_west(
        self,
        reference: str | None = None,
        horizon: str | None = "months_to_end",
        *,
        h: int = 1,
        alternative: str = "less",
        aggregate: str | None = None,
        periods: str | Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        r"""Clark-West tests of every model against a *nested* benchmark.

        Use it when the reference (e.g. an AR) is nested in the competing models, where
        the Diebold-Mariano test is undersized (Clark & West, 2007; see
        :func:`~nowcastbox.evaluation.clark_west`). Squared loss only; negative
        statistic: the larger model is more accurate.

        Parameters
        ----------
        reference : str, optional
            Nested benchmark (default: the first benchmark).
        horizon : str or None, default "months_to_end"
            Grouping column (``None``: pooled).
        h : int, default 1
            Horizon of the long-run variance.
        alternative : {"less", "greater", "two-sided"}, default "less"
            ``"less"``: the larger model is more accurate.
        aggregate : {None, "target_period"}, default None
            Average the adjusted losses within each target period before testing (see
            :meth:`diebold_mariano`).
        periods : mapping or {"covid", "ex-covid"}, optional
            Sub-periods of target periods (see :meth:`split_periods`).

        Returns
        -------
        pandas.DataFrame
            Index ``(model, horizon)`` (or ``(period, model, horizon)``); columns
            ``statistic``, ``pvalue``, ``mean_loss_differential`` (adjusted), ``n_obs``.

        Raises
        ------
        ValueError
            If ``aggregate`` or ``periods`` is invalid or the reference is unknown.

        Examples
        --------
        >>> res.clark_west("AR", "kind", aggregate="target_period")  # doctest: +SKIP
        """
        if periods is not None:
            options = {"h": h, "alternative": alternative, "aggregate": aggregate}
            return self._by_periods(
                periods, "clark_west", reference=reference, horizon=horizon, **options
            )

        def test(l1: np.ndarray, l2: np.ndarray) -> ClarkWestResult:
            # l1 - l2 is the Clark-West adjusted differential (see _clark_west_pair)
            return clark_west_from_differential(l1 - l2, h=h, alternative=alternative)

        return self._pairwise_table(reference, horizon, test, _clark_west_pair, aggregate)

    def giacomini_white(
        self,
        reference: str | None = None,
        horizon: str | None = "months_to_end",
        *,
        loss: LossLike = "squared",
        h: int = 1,
        aggregate: str | None = None,
        periods: str | Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Giacomini-White conditional tests of every model against a reference.

        Parameters
        ----------
        reference : str, optional
            Reference model (default: the first benchmark).
        horizon : str or None, default "months_to_end"
            Grouping column (``None``: pooled).
        loss : {"squared", "absolute"} or callable, default "squared"
            Loss function.
        h : int, default 1
            Forecast horizon (lag of the default instruments).
        aggregate : {None, "target_period"}, default None
            Average the losses within each target period before testing (see
            :meth:`diebold_mariano`).
        periods : mapping or {"covid", "ex-covid"}, optional
            Sub-periods of target periods (see :meth:`split_periods`).

        Returns
        -------
        pandas.DataFrame
            Index ``(model, horizon)`` (or ``(period, model, horizon)``); columns
            ``statistic``, ``pvalue``, ``mean_loss_differential``, ``n_obs``.

        Raises
        ------
        ValueError
            If ``aggregate`` or ``periods`` is invalid or the reference is unknown.

        Examples
        --------
        >>> res.giacomini_white("AR", horizon=None)  # doctest: +SKIP
        """
        if periods is not None:
            options = {"loss": loss, "h": h, "aggregate": aggregate}
            return self._by_periods(
                periods, "giacomini_white", reference=reference, horizon=horizon, **options
            )

        def test(l1: np.ndarray, l2: np.ndarray) -> GiacominiWhiteResult:
            return giacomini_white(l1, l2, h=h, loss=_identity)

        return self._pairwise_table(reference, horizon, test, _loss_pair(loss), aggregate)

    def loss_table(
        self, *, loss: LossLike = "squared", models: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """Losses of every model on the common sample (rows: vintage, target period, horizon).

        Parameters
        ----------
        loss : {"squared", "absolute"} or callable, default "squared"
            Loss function.
        models : sequence of str, optional
            Models (default: all).

        Returns
        -------
        pandas.DataFrame
            One column per model; index ``(vintage, target_period, months_to_end)``.

        Examples
        --------
        >>> res.loss_table().shape  # doctest: +SKIP
        """
        chosen = self._models(models)
        frame = self.evaluable(chosen)
        frame = frame.assign(loss=loss_values(frame["error"].to_numpy(), loss))
        table = frame.pivot_table(
            index=[*_KEY, "months_to_end"], columns="model", values="loss", aggfunc="first"
        )
        table = table.reindex(columns=chosen).dropna()
        table.columns.name = None
        return table.sort_index(level=["target_period", "vintage"])

    def mcs(
        self,
        alpha: float = 0.1,
        horizon: str | None = None,
        *,
        loss: LossLike = "squared",
        statistic: str = "max",
        n_bootstrap: int = 1000,
        block_length: int | None = None,
        random_state: int | np.random.Generator | None = 0,
        models: Sequence[str] | None = None,
        aggregate: str | None = None,
        periods: str | Mapping[str, Any] | None = None,
    ) -> ModelConfidenceSetResult | dict[Any, Any]:
        r"""Model Confidence Set (Hansen, Lunde & Nason, 2011) of the evaluated models.

        Parameters
        ----------
        alpha : float, default 0.1
            Level (the set has coverage :math:`1-\alpha`).
        horizon : str, optional
            Compute one set per value of this column (default: pooled).
        loss : {"squared", "absolute"} or callable, default "squared"
            Loss function.
        statistic : {"max", "range"}, default "max"
            MCS statistic.
        n_bootstrap : int, default 1000
            Bootstrap replications.
        block_length : int, optional
            Block length of the bootstrap.
        random_state : int, Generator or None, default 0
            Seed (fixed by default).
        models : sequence of str, optional
            Models (default: all).
        aggregate : {None, "target_period"}, default None
            Average the losses within each target period before bootstrapping (see
            :meth:`diebold_mariano`).
        periods : mapping or {"covid", "ex-covid"}, optional
            Sub-periods of target periods (see :meth:`split_periods`).

        Returns
        -------
        ModelConfidenceSetResult or dict
            One result, or ``{horizon value: result}``; with ``periods``, a dict
            ``{period label: <one of these>}``.

        Raises
        ------
        ValueError
            If ``aggregate`` or ``periods`` is invalid.

        Examples
        --------
        >>> res.mcs(alpha=0.25).included  # doctest: +SKIP
        >>> res.mcs(periods="covid")["ex-Covid"].included  # doctest: +SKIP
        """
        options = {
            "alpha": alpha,
            "statistic": statistic,
            "n_bootstrap": n_bootstrap,
            "block_length": block_length,
            "random_state": random_state,
        }
        if periods is not None:
            return {
                label: sub.mcs(
                    horizon=horizon, loss=loss, models=models, aggregate=aggregate, **options
                )
                for label, sub in self.split_periods(periods).items()
            }
        _check_aggregate(aggregate)
        table = self.loss_table(loss=loss, models=models)
        if horizon is None:
            return model_confidence_set(_per_period(table, aggregate), **options)
        frame = self.evaluable(models)
        labels = frame.set_index([*_KEY, "months_to_end"], drop=False)[horizon]
        labels = labels[~labels.index.duplicated()].reindex(table.index)
        return {
            key: model_confidence_set(
                _per_period(table.iloc[np.flatnonzero(labels == key)], aggregate), **options
            )
            for key in sorted(labels.dropna().unique())
        }

    # ------------------------------------------------------------------ report
    def summary(self, horizon: str | None = "months_to_end") -> str:
        """Plain-text report: pooled accuracy and RMSFE by horizon.

        Parameters
        ----------
        horizon : str or None, default "months_to_end"
            Grouping column of the by-horizon table (``None`` omits it).

        Returns
        -------
        str
            Report.

        Examples
        --------
        >>> print(res.summary())  # doctest: +SKIP
        """
        frame = self.forecasts
        lines = [
            "Pseudo real-time backtest"
            + (" (real-time vintages)" if self.info.get("real_time") else ""),
            f"Target: {self.target}    Models: {', '.join(self.models)}",
            f"Vintages: {frame['vintage'].nunique()} "
            f"({_fmt_date(frame['vintage'].min())} to {_fmt_date(frame['vintage'].max())})    "
            f"Target periods: {frame['target_period'].nunique()}",
            "",
            "Accuracy (common sample, all horizons)",
        ]
        pooled = self.metrics(horizon=None).T.unstack(level="metric")  # noqa: PD010
        pooled = pooled.droplevel(0, axis=1)
        if self.reference is not None and len(self.models) > 1:
            rel = self.relative_to(horizon=None).T
            pooled["rel_rmsfe"] = rel.iloc[:, 0]
        lines.append(pooled.to_string(float_format=lambda v: f"{v:.4f}"))
        if horizon is not None:
            lines += ["", f"RMSFE by {horizon}"]
            table = self.rmsfe_by_horizon(horizon)
            lines.append(table.to_string(float_format=lambda v: f"{v:.4f}"))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"BacktestResults(target={self.target!r}, models={self.models}, "
            f"n_forecasts={len(self.forecasts)})"
        )


def _directional_row(group: pd.DataFrame, test: bool, alternative: str) -> dict[str, float]:
    """FDA, number of complete forecasts and (optionally) the Pesaran-Timmermann test."""
    a, f, p = (group[c].to_numpy(dtype=float) for c in ("actual", "forecast", "previous_actual"))
    row = {
        "fda": directional_accuracy(a, f, p),
        "n_obs": float(directional_changes(a, f, p)[0].size),
    }
    if not test:
        return row
    nan = {"statistic": np.nan, "pvalue": np.nan, "expected_hit_rate": np.nan}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataQualityWarning)
            out = pesaran_timmermann(a, f, p, alternative=alternative)
    except NowcastDataError:
        return {**row, **nan}
    return {
        **row,
        "statistic": out.statistic,
        "pvalue": out.pvalue,
        "expected_hit_rate": out.expected_hit_rate,
    }


def _fmt_date(value: Any) -> str:
    return "-" if pd.isna(value) else str(pd.Timestamp(value).date())


def _check_aggregate(aggregate: str | None) -> None:
    if aggregate not in _AGGREGATES:
        raise ValueError(f"aggregate must be one of {_AGGREGATES}, got {aggregate!r}.")


def _per_period(table: pd.DataFrame, aggregate: str | None) -> pd.DataFrame:
    """Average a loss table within each target period (no-op when ``aggregate`` is None)."""
    if aggregate is None:
        return table
    return table.groupby(level=aggregate, sort=True).mean()


def _identity(values: np.ndarray) -> np.ndarray:
    return values


def _loss_pair(
    loss: LossLike,
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    def pair(e_model: np.ndarray, e_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return loss_values(e_model, loss), loss_values(e_ref, loss)

    return pair


def _clark_west_pair(e_model: np.ndarray, e_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Adjusted squared loss of the larger model and squared loss of the nested one."""
    return e_model**2 - (e_model - e_ref) ** 2, e_ref**2


def _test_row(
    test: Callable[[np.ndarray, np.ndarray], _PairTestResult],
    e_model: np.ndarray,
    e_ref: np.ndarray,
) -> dict[str, float]:
    try:
        out = test(e_model, e_ref)
    except NowcastDataError:
        return {
            "statistic": np.nan,
            "pvalue": np.nan,
            "mean_loss_differential": np.nan,
            "n_obs": float(e_model.size),
        }
    return {
        "statistic": out.statistic,
        "pvalue": out.pvalue,
        "mean_loss_differential": out.mean_loss_differential,
        "n_obs": float(out.n_obs),
    }

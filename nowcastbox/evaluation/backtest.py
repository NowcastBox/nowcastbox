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

References
----------
Giannone, D., Reichlin, L. & Small, D. (2008). Nowcasting: The real-time informational
content of macroeconomic data. *Journal of Monetary Economics*, 55(4), 665-676.

Bańbura, M., Giannone, D., Modugno, M. & Reichlin, L. (2013). Now-casting and the
real-time data flow. In *Handbook of Economic Forecasting*, vol. 2A, 195-237.

Croushore, D. (2011). Frontiers of real-time data analysis. *Journal of Economic
Literature*, 49(1), 72-100.
"""

from __future__ import annotations

import copy
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
from nowcastbox.core.exceptions import NowcastBoxWarning, NowcastDataError
from nowcastbox.core.formula import resolve_target
from nowcastbox.core.frequency import Frequency, base_to_native, is_period_end
from nowcastbox.core.results import NowcastResults
from nowcastbox.evaluation.metrics import (
    LossLike,
    accuracy_by_horizon,
    loss_values,
    metric_by_horizon,
)
from nowcastbox.evaluation.tests import (
    DieboldMarianoResult,
    GiacominiWhiteResult,
    ModelConfidenceSetResult,
    diebold_mariano,
    giacomini_white,
    model_confidence_set,
)
from nowcastbox.vintages import ReleaseCalendar, VintageStore, pseudo_real_time, vintage_dates

__all__ = ["FORECAST_COLUMNS", "BacktestResults", "PseudoRealTimeBacktest"]

logger = get_logger(__name__)

#: Columns of :attr:`BacktestResults.forecasts` (one row per vintage, target period, model).
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
)
_KEY = ["vintage", "target_period"]
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

    def vintage(self, date: pd.Timestamp) -> MixedFrequencyData:
        if self.store is not None:
            out = self.store.as_of(date, as_mixed=True, **self.metadata)
        else:
            assert self.panel is not None  # noqa: S101
            out = pseudo_real_time(self.panel, calendar=self.calendar, vintage=date)
        assert isinstance(out, MixedFrequencyData)  # noqa: S101
        if self.preprocess is not None:
            out = self.preprocess(out)
        return out

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
        panel = spec.source.vintage(date)
        periods = _target_periods(spec, panel, date)
        if len(periods) == 0:
            continue
        window = _estimation_panel(spec, panel, date, periods)
        current = pd.Period(date, freq=periods.freqstr)
        labels = [_labels(spec, date, p, int((p - current).n)) for p in periods]
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
    frame = pd.DataFrame.from_records(records, columns=list(FORECAST_COLUMNS[:-2]))
    periods = pd.PeriodIndex(frame["target_period"], freq=freq.pandas_freq)
    frame["target_period"] = periods
    frame["actual"] = actuals.reindex(periods).to_numpy(dtype=float)
    frame["error"] = frame["actual"] - frame["forecast"]
    frame["vintage"] = pd.to_datetime(frame["vintage"])
    return frame.sort_values(["vintage", "target_period"], kind="stable").reset_index(drop=True)


# ====================================================================== results
@dataclass(frozen=True)
class BacktestResults:
    """Forecasts of a backtest and the tools to evaluate them.

    Parameters
    ----------
    forecasts : pandas.DataFrame
        Long table, columns :data:`FORECAST_COLUMNS`.
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
            ``actual``).

        Returns
        -------
        pandas.DataFrame
            Forecasts.

        Examples
        --------
        >>> res.to_frame(wide=True).columns.tolist()  # doctest: +SKIP
        ['M', 'AR', 'actual']
        """
        frame = self.forecasts.copy()
        if not wide:
            return frame
        labels = [c for c in FORECAST_COLUMNS if c not in ("model", "forecast", "actual", "error")]
        indexed = frame.set_index([*labels, "model"])
        table = indexed["forecast"].unstack("model")  # noqa: PD010
        table = table.reindex(columns=[m for m in self.models if m in table.columns])
        actual = indexed["actual"].groupby(level=labels, dropna=False).first()
        table["actual"] = actual.reindex(table.index)
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

    # ------------------------------------------------------------------ accuracy
    def metrics(
        self,
        horizon: str | None = "months_to_end",
        metrics: Sequence[str] = ("rmsfe", "mae", "bias", "n"),
        *,
        models: Sequence[str] | None = None,
        common_sample: bool = True,
    ) -> pd.DataFrame:
        """Accuracy metrics by horizon.

        Parameters
        ----------
        horizon : str or None, default "months_to_end"
            Grouping column (``"months_to_end"``, ``"days_to_end"``,
            ``"days_to_release"``, ``"offset"``, ``"kind"``...) or ``None`` (pooled).
        metrics : sequence of str, default ("rmsfe", "mae", "bias", "n")
            Metrics (see :data:`nowcastbox.evaluation.metrics.METRICS`).
        models : sequence of str, optional
            Models (default: all).
        common_sample : bool, default True
            Use only target periods/vintages forecast by every model.

        Returns
        -------
        pandas.DataFrame
            Index: horizon; columns ``(metric, model)``.

        Examples
        --------
        >>> res.metrics(horizon=None)  # doctest: +SKIP
        """
        frame = self.evaluable(models, common_sample=common_sample)
        table = accuracy_by_horizon(frame, horizon, metrics)
        return table.reindex(columns=self._models(models), level="model")

    def _metric(
        self, metric: str, horizon: str | None, models: Sequence[str] | None, common: bool
    ) -> pd.DataFrame:
        frame = self.evaluable(models, common_sample=common)
        table = metric_by_horizon(frame, metric, horizon)
        return table.reindex(columns=self._models(models))

    def rmsfe_by_horizon(
        self,
        horizon: str | None = "months_to_end",
        *,
        models: Sequence[str] | None = None,
        common_sample: bool = True,
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

        Returns
        -------
        pandas.DataFrame
            Index: horizon; columns: models.

        Examples
        --------
        >>> res.rmsfe_by_horizon()  # doctest: +SKIP
        """
        return self._metric("rmsfe", horizon, models, common_sample)

    def relative_to(
        self,
        reference: str | None = None,
        horizon: str | None = "months_to_end",
        *,
        metric: str = "rmsfe",
        models: Sequence[str] | None = None,
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
        """
        ref = self._reference(reference)
        chosen = self._models(models)
        if ref not in chosen:
            chosen.append(ref)
        table = self._metric(metric, horizon, chosen, True)
        return table.div(table[ref], axis=0)

    # ------------------------------------------------------------------ tests
    def _paired(
        self, model: str, reference: str, horizon: str | None
    ) -> list[tuple[Any, np.ndarray, np.ndarray]]:
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
            groups: list[tuple[Any, pd.DataFrame]] = [("all", wide)]
        else:
            groups = list(wide.groupby(level=horizon, sort=True))
        return [(label, g[model].to_numpy(), g[reference].to_numpy()) for label, g in groups]

    def _pairwise_table(
        self,
        reference: str | None,
        horizon: str | None,
        test: Callable[[np.ndarray, np.ndarray], DieboldMarianoResult | GiacominiWhiteResult],
    ) -> pd.DataFrame:
        ref = self._reference(reference)
        rows = []
        for model in self.models:
            if model == ref:
                continue
            for label, e_model, e_ref in self._paired(model, ref, horizon):
                rows.append({"model": model, "horizon": label, **_test_row(test, e_model, e_ref)})
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
    ) -> pd.DataFrame:
        r"""Diebold-Mariano (HLN) tests of every model against a reference.

        The loss differential is :math:`L(e_{\text{model}}) - L(e_{\text{reference}})`
        (negative statistic: the model is more accurate), computed per horizon group
        over target periods (ordered in time).

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

        Returns
        -------
        pandas.DataFrame
            Index ``(model, horizon)``; columns ``statistic``, ``pvalue``,
            ``mean_loss_differential``, ``n_obs`` (NaN when fewer than 3 pairs).

        Examples
        --------
        >>> res.diebold_mariano("AR")  # doctest: +SKIP
        """

        def test(e1: np.ndarray, e2: np.ndarray) -> DieboldMarianoResult:
            return diebold_mariano(e1, e2, h=h, loss=loss, alternative=alternative, hln=hln)

        return self._pairwise_table(reference, horizon, test)

    def giacomini_white(
        self,
        reference: str | None = None,
        horizon: str | None = "months_to_end",
        *,
        loss: LossLike = "squared",
        h: int = 1,
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

        Returns
        -------
        pandas.DataFrame
            Index ``(model, horizon)``; columns ``statistic``, ``pvalue``,
            ``mean_loss_differential``, ``n_obs``.

        Examples
        --------
        >>> res.giacomini_white("AR", horizon=None)  # doctest: +SKIP
        """

        def test(e1: np.ndarray, e2: np.ndarray) -> GiacominiWhiteResult:
            return giacomini_white(e1, e2, h=h, loss=loss)

        return self._pairwise_table(reference, horizon, test)

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
    ) -> ModelConfidenceSetResult | dict[Any, ModelConfidenceSetResult]:
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

        Returns
        -------
        ModelConfidenceSetResult or dict
            One result, or ``{horizon value: result}``.

        Examples
        --------
        >>> res.mcs(alpha=0.25).included  # doctest: +SKIP
        """
        table = self.loss_table(loss=loss, models=models)
        options = {
            "alpha": alpha,
            "statistic": statistic,
            "n_bootstrap": n_bootstrap,
            "block_length": block_length,
            "random_state": random_state,
        }
        if horizon is None:
            return model_confidence_set(table, **options)
        frame = self.evaluable(models)
        labels = frame.set_index([*_KEY, "months_to_end"], drop=False)[horizon]
        labels = labels[~labels.index.duplicated()].reindex(table.index)
        return {
            key: model_confidence_set(table.iloc[np.flatnonzero(labels == key)], **options)
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


def _fmt_date(value: Any) -> str:
    return "-" if pd.isna(value) else str(pd.Timestamp(value).date())


def _test_row(
    test: Callable[[np.ndarray, np.ndarray], DieboldMarianoResult | GiacominiWhiteResult],
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

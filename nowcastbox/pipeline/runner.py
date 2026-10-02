"""Execute a :class:`~nowcastbox.pipeline.spec.NowcastSpec` end to end (innovation I10).

:func:`run_pipeline` chains the whole production workflow:

1. load the data (built-in dataset, file or connectors) and keep the information set
   of the spec's ``vintage`` (pseudo real-time release rule);
2. preprocess (:func:`~nowcastbox.preprocessing.prepare_panel`);
3. estimate the model through :func:`nowcastbox.nowcast` (Bai & Ng, 2002 selection
   when ``factors: auto``);
4. produce the requested outputs: density nowcast (I5), news decomposition against the
   previous snapshot or an earlier vintage (Bańbura & Modugno, 2014; I6), DFM
   diagnostics (I9), pseudo real-time backtest (accuracy by horizon and sub-period,
   directional accuracy), empirical error bands, the indicator z-score heatmap, the
   nowcasts of alternative models without one or two groups of indicators, the HTML
   report and an Excel workbook;
5. freeze everything in a versioned snapshot (:class:`~nowcastbox.pipeline.snapshots.SnapshotStore`).

The result is a :class:`PipelineRun` holding every intermediate object.
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.results import NowcastResults
from nowcastbox.pipeline.data import apply_vintage, data_hash, load_data, preprocess
from nowcastbox.pipeline.snapshots import Snapshot, SnapshotStore, headline_period, jsonable
from nowcastbox.pipeline.spec import (
    AlternativesOutput,
    BacktestOutput,
    DensityOutput,
    DiagnosticsOutput,
    EmpiricalBandsOutput,
    ExcelOutput,
    HeatmapOutput,
    NewsOutput,
    NowcastSpec,
    ReportOutput,
    SpecError,
    load_spec,
)

__all__ = ["PipelineRun", "run_pipeline"]

logger = get_logger(__name__)

_NEWS_ERRORS = (NowcastDataError, ValueError, TypeError, NotImplementedError, KeyError)


@dataclasses.dataclass(frozen=True)
class PipelineRun:
    """Everything produced by :func:`run_pipeline`.

    Parameters
    ----------
    spec : NowcastSpec
        The executed spec.
    vintage : pandas.Timestamp
        Information-set date.
    data : MixedFrequencyData
        Data vintage in levels (after the vintage rule).
    panel : MixedFrequencyData
        Model-ready panel (after preprocessing).
    data_hash : str
        SHA-256 fingerprint of ``data``.
    results : NowcastResults
        Fitted results (with density columns when requested).
    headline_period : pandas.Period or None
        Target period of the headline nowcast.
    headline : float
        Headline nowcast.
    distribution : NowcastDistribution, optional
        Predictive distribution (``density`` output).
    news : NewsResults, optional
        News decomposition (``news`` output).
    news_reference : str, optional
        What the news are measured against (snapshot id or vintage date).
    diagnostics : DiagnosticsReport, optional
        DFM diagnostics (``diagnostics`` output).
    backtest : BacktestResults, optional
        Pseudo real-time backtest (``backtest`` output).
    backtest_metrics : pandas.DataFrame, optional
        Accuracy table of the backtest (``outputs.backtest.metrics`` by horizon and,
        with ``outputs.backtest.periods``, by sub-period).
    empirical_bands : EmpiricalGaussianDistribution or EmpiricalQuantileDistribution, optional
        Empirical error bands around the nowcast (``empirical_bands`` output).
    heatmap : IndicatorZScores, optional
        Z-scores of the indicators (``heatmap`` output).
    alternatives : AlternativeNowcasts, optional
        Nowcasts of the alternative models (``alternatives`` output).
    report_html : str, optional
        HTML report (``report_html`` output).
    report_path : pathlib.Path, optional
        Where the report was written (snapshot or ``outputs.report_html.path``).
    excel_paths : tuple of pathlib.Path
        Excel workbooks written (``excel`` output: ``outputs.excel.path`` and the
        snapshot's ``results.xlsx``).
    snapshot : Snapshot, optional
        Snapshot written for this run.
    previous : Snapshot, optional
        Latest snapshot of the same spec name before this run.
    warnings : tuple of str
        Warnings raised during the run (also logged).
    timings : dict
        Seconds spent per stage.

    Examples
    --------
    >>> from nowcastbox.pipeline import run_pipeline
    >>> run = run_pipeline(
    ...     {
    ...         "target": "gdp",
    ...         "data": {"source": "simulated_dfm", "columns": ["x01", "x02", "x03", "x04"]},
    ...         "preprocessing": False,
    ...         "model": {"type": "TwoStepDFM", "factors": 1},
    ...     }
    ... )
    >>> str(run.headline_period), run.snapshot is None
    ('2019Q4', True)
    """

    spec: NowcastSpec
    vintage: pd.Timestamp
    data: MixedFrequencyData
    panel: MixedFrequencyData
    data_hash: str
    results: NowcastResults
    headline_period: pd.Period | None
    headline: float
    distribution: Any = None
    news: Any = None
    news_reference: str | None = None
    diagnostics: Any = None
    backtest: Any = None
    backtest_metrics: pd.DataFrame | None = None
    empirical_bands: Any = None
    heatmap: Any = None
    alternatives: Any = None
    report_html: str | None = None
    report_path: Path | None = None
    excel_paths: tuple[Path, ...] = ()
    snapshot: Snapshot | None = None
    previous: Snapshot | None = None
    warnings: tuple[str, ...] = ()
    timings: dict[str, float] = dataclasses.field(default_factory=dict)

    @property
    def nowcast(self) -> pd.DataFrame:
        """Nowcast table of the results (copy)."""
        return self.results.nowcast.copy()

    def summary(self) -> str:
        """Human-readable report of the run.

        Returns
        -------
        str
            Multi-line text (data, model, headline, news, diagnostics, backtest,
            snapshot, warnings).

        Examples
        --------
        >>> from nowcastbox.pipeline import run_pipeline
        >>> run = run_pipeline(
        ...     {
        ...         "target": "gdp",
        ...         "data": {"source": "simulated_dfm", "columns": ["x01", "x02", "x03"]},
        ...         "preprocessing": False,
        ...         "model": {"type": "TwoStepDFM", "factors": 1},
        ...     }
        ... )
        >>> print(run.summary())  # doctest: +ELLIPSIS
        Nowcast pipeline 'gdp' ...
          target      : gdp
        ...
        """
        lines = [
            f"Nowcast pipeline {self.spec.name!r} (vintage {self.vintage.date()})",
            f"  target      : {self.spec.target_name}",
            f"  data        : {self.spec.data.source}, {self.data.n_series} series, "
            f"{self.data.start} to {self.data.end} (hash {self.data_hash[:12]})",
            f"  panel       : {self.panel.n_series} series after preprocessing",
            f"  model       : {self.results.model_name} ({_model_line(self.results)})",
            f"  nowcast     : {self.headline_period} = {_fmt(self.headline)}{self._band_text()}",
        ]
        lines += self._news_lines()
        if self.diagnostics is not None:
            flagged = self.diagnostics.flagged_series
            lines.append(f"  diagnostics : {len(flagged)} flagged series")
        if self.backtest is not None:
            table = self.backtest.rmsfe_by_horizon()
            lines.append("  backtest    : RMSFE by horizon")
            lines += ["    " + row for row in table.to_string(float_format=_fmt).splitlines()]
        lines += self._phase_lines()
        if self.snapshot is not None:
            lines.append(f"  snapshot    : {self.snapshot.path}")
        if self.report_path is not None:
            lines.append(f"  report      : {self.report_path}")
        lines += [f"  warning     : {w}" for w in self.warnings]
        return "\n".join(lines)

    def _phase_lines(self) -> list[str]:
        """Summary lines of the empirical bands, heatmap and alternative models."""
        lines = [*self._empirical_band_line(), *self._alternatives_line()]
        if self.heatmap is not None:
            lines.append(f"  heatmap     : z-scores of {len(self.heatmap.series)} indicators")
        if self.excel_paths:
            lines.append(f"  excel       : {', '.join(str(p) for p in self.excel_paths)}")
        return lines

    def _empirical_band_line(self) -> list[str]:
        bands = self.empirical_bands
        if bands is None or self.headline_period is None:
            return []
        level = max(bands.levels)
        table = bands.interval(level)
        labels = [str(p) for p in table.index]
        if str(self.headline_period) not in labels:
            return []
        row = table.iloc[labels.index(str(self.headline_period))]
        return [f"  emp. bands  : {100 * level:g}% [{_fmt(row['lower'])}, {_fmt(row['upper'])}]"]

    def _alternatives_line(self) -> list[str]:
        if self.alternatives is None or self.headline_period is None:
            return []
        table = self.alternatives.range()
        labels = [str(p) for p in table.index]
        if str(self.headline_period) not in labels:
            return []
        row = table.iloc[labels.index(str(self.headline_period))]
        return [
            f"  alternatives: {self.alternatives.n_models} models, range "
            f"[{_fmt(row['min'])}, {_fmt(row['max'])}]"
        ]

    def _band_text(self) -> str:
        frame = self.results.nowcast
        if self.headline_period is None or "lower_68" not in frame.columns:
            return ""
        key: Any = self.headline_period
        row = frame.loc[key]
        if pd.isna(row.get("lower_68")):
            return ""
        return (
            f"  [68%: {_fmt(row['lower_68'])}, {_fmt(row['upper_68'])}; "
            f"90%: {_fmt(row['lower_90'])}, {_fmt(row['upper_90'])}]"
        )

    def _news_lines(self) -> list[str]:
        if self.news is None:
            return []
        news = self.news
        lines = [
            f"  news        : {_fmt(news.old_nowcast)} -> {_fmt(news.new_nowcast)} "
            f"against {self.news_reference} (news {_fmt(news.news_effect)}, "
            f"revisions {_fmt(news.revisions_effect)}, re-estimation "
            f"{_fmt(news.reestimation_effect)})"
        ]
        top = news.top_releases(5)
        if not top.empty and "impact" in top.columns:
            for series, impact in zip(top["series"], top["impact"], strict=False):
                lines.append(f"    {series}: {_fmt(impact)}")
        return lines

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary of the run.

        Returns
        -------
        dict
            Name, target, vintage, data hash, model, headline, news effects, snapshot
            id and paths, warnings and timings.

        Examples
        --------
        >>> from nowcastbox.pipeline import run_pipeline
        >>> run = run_pipeline(
        ...     {
        ...         "target": "gdp",
        ...         "data": {"source": "simulated_dfm", "columns": ["x01", "x02", "x03"]},
        ...         "preprocessing": False,
        ...         "model": {"type": "TwoStepDFM", "factors": 1},
        ...     }
        ... )
        >>> run.to_dict()["headline_period"]
        '2019Q4'
        """
        news = None
        if self.news is not None:
            news = _news_info(self.news, self.news_reference)
        return jsonable(
            {
                "name": self.spec.name,
                "target": self.spec.target_name,
                "vintage": self.vintage.date(),
                "data_hash": self.data_hash,
                "model": self.results.model_name,
                "headline_period": self.headline_period,
                "headline": self.headline,
                "news": news,
                "snapshot": None if self.snapshot is None else self.snapshot.id,
                "snapshot_path": None if self.snapshot is None else self.snapshot.path,
                "report_path": self.report_path,
                "excel_paths": list(self.excel_paths),
                "outputs": self.spec.outputs.names,
                "warnings": self.warnings,
                "timings": self.timings,
            }
        )


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if np.isnan(number) else f"{number:.4g}"


def _model_line(results: NowcastResults) -> str:
    parts = [f"factors={results.model_params.get('n_factors')}"]
    if results.n_iter is not None:
        parts.append(f"iterations={results.n_iter}")
    if results.converged is not None:
        parts.append(f"converged={results.converged}")
    return ", ".join(parts)


def _news_info(news: Any, reference: str | None) -> dict[str, Any]:
    return {
        "against": reference,
        "target_period": str(news.target_period),
        "old_nowcast": news.old_nowcast,
        "new_nowcast": news.new_nowcast,
        "news_effect": news.news_effect,
        "revisions_effect": news.revisions_effect,
        "reestimation_effect": news.reestimation_effect,
        "n_releases": news.n_releases,
    }


# ---------------------------------------------------------------------------- run
class _Timer:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        logger.info("pipeline stage: %s", name)
        try:
            yield
        finally:
            self.timings[name] = round(time.perf_counter() - start, 4)


@dataclasses.dataclass
class _State:
    """Mutable state shared by the stages of one run."""

    spec: NowcastSpec
    vintage: pd.Timestamp
    loaded: MixedFrequencyData
    data: MixedFrequencyData
    panel: MixedFrequencyData
    results: NowcastResults
    notes: list[str]
    previous: Snapshot | None = None
    distribution: Any = None
    news: Any = None
    news_reference: str | None = None
    diagnostics: Any = None
    backtest: Any = None
    backtest_metrics: pd.DataFrame | None = None
    bands: Any = None
    zscores: Any = None
    alternatives: Any = None
    report_html: str | None = None
    report_path: Path | None = None


def run_pipeline(
    spec: NowcastSpec | Mapping[str, Any] | str | Path,
    *,
    snapshot: bool = True,
    snapshot_dir: str | Path | None = None,
    vintage: str | None = None,
    today: pd.Timestamp | str | None = None,
) -> PipelineRun:
    """Run a nowcast spec end to end.

    Parameters
    ----------
    spec : NowcastSpec, mapping, str or pathlib.Path
        Spec, parsed mapping, YAML file or YAML text (see
        :func:`~nowcastbox.pipeline.spec.load_spec`).
    snapshot : bool, default True
        Write a snapshot when the spec has a ``snapshot_dir`` (or ``snapshot_dir`` is
        given). With ``False`` nothing is written, but an existing snapshot folder is
        still read, so news are computed against the latest stored snapshot.
    snapshot_dir : str or pathlib.Path, optional
        Override of the spec's ``snapshot_dir``.
    vintage : str, optional
        Override of the spec's ``vintage`` (``"today"`` or a date).
    today : pandas.Timestamp or str, optional
        Date used for ``vintage: today`` (default: the current date).

    Returns
    -------
    PipelineRun
        Results, outputs, snapshot and warnings.

    Raises
    ------
    SpecError
        If the spec is invalid or inconsistent with the data (e.g. unknown blocks).
    NowcastDataError
        If the data cannot support the model.

    Notes
    -----
    News against the previous snapshot use the parameters of the current model on both
    information sets (old vintage stored in the snapshot, new vintage of this run), so
    the revision of the headline nowcast splits exactly into data revisions and the
    news of each release (Bańbura & Modugno, 2014). Failures of optional outputs
    (news, diagnostics, backtest, empirical bands, heatmap, alternative models, report,
    Excel workbook) do not stop the run: they are recorded in
    :attr:`PipelineRun.warnings`.

    Examples
    --------
    >>> import tempfile
    >>> from nowcastbox.pipeline import run_pipeline
    >>> spec = {
    ...     "name": "demo",
    ...     "target": "gdp",
    ...     "data": {"source": "simulated_dfm", "columns": ["x01", "x02", "x03"]},
    ...     "vintage": "2019-11-15",
    ...     "preprocessing": False,
    ...     "model": {"type": "MixedFreqDFM", "factors": 1, "max_iter": 20},
    ...     "outputs": ["nowcast", "news", "density"],
    ...     "snapshot_dir": tempfile.mkdtemp(),
    ... }
    >>> first = run_pipeline(spec)
    >>> second = run_pipeline({**spec, "vintage": "2019-12-20"})
    >>> second.previous.id == first.snapshot.id, second.news.check_identity()
    (True, True)
    """
    parsed = load_spec(spec)
    if vintage is not None:
        parsed = NowcastSpec.from_dict(
            {**parsed.to_dict(), "vintage": vintage}, base_dir=parsed.base_dir, source=parsed.source
        )
    if snapshot_dir is not None:
        parsed = parsed.replace(snapshot_dir=Path(snapshot_dir).expanduser())
    store = SnapshotStore(parsed.snapshot_dir) if snapshot and parsed.snapshot_dir else None
    history = store
    if history is None and parsed.snapshot_dir is not None and parsed.snapshot_dir.is_dir():
        history = SnapshotStore(parsed.snapshot_dir)  # read-only: previous snapshot for news
    timer = _Timer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        state = _estimate(parsed, timer, None if today is None else pd.Timestamp(today))
        if history is not None:
            state.previous = history.latest(parsed.name)
        _outputs(state, timer)
    notes = _dedupe([*state.notes, *(f"{w.category.__name__}: {w.message}" for w in caught)])
    for note in notes:
        logger.warning("%s", note)
    period = headline_period(state.results.nowcast)
    headline = float("nan") if period is None else float(_estimate_at(state.results, period))
    snap = None
    if store is not None:
        with timer.stage("snapshot"):
            snap = _write_snapshot(store, state, period, notes)
            if state.report_html is not None and state.report_path is None:
                state.report_path = snap.report_path
    run = PipelineRun(
        spec=parsed,
        vintage=state.vintage,
        data=state.data,
        panel=state.panel,
        data_hash=data_hash(state.data),
        results=state.results,
        headline_period=period,
        headline=headline,
        distribution=state.distribution,
        news=state.news,
        news_reference=state.news_reference,
        diagnostics=state.diagnostics,
        backtest=state.backtest,
        backtest_metrics=state.backtest_metrics,
        empirical_bands=state.bands,
        heatmap=state.zscores,
        alternatives=state.alternatives,
        report_html=state.report_html,
        report_path=state.report_path,
        snapshot=snap,
        previous=state.previous,
        warnings=tuple(notes),
        timings=timer.timings,
    )
    if parsed.outputs.excel is not None:
        with timer.stage("excel"):
            run = _excel(run, parsed.outputs.excel)
    return run


def _excel(run: PipelineRun, options: ExcelOutput) -> PipelineRun:
    """Write the run's workbook(s); a failure is recorded as a warning of the run."""
    from nowcastbox.pipeline.data import write_run_excel

    targets = [options.path] if options.path is not None else []
    if run.snapshot is not None:
        targets.append(run.snapshot.path / "results.xlsx")
    written: list[Path] = []
    try:
        written = [write_run_excel(run, target) for target in targets]
    except ImportError as err:
        note = f"excel output skipped: {err}"
        logger.warning("%s", note)
        return dataclasses.replace(run, warnings=(*run.warnings, note))
    if not targets:
        note = "excel output skipped: give outputs.excel.path or a snapshot_dir"
        logger.warning("%s", note)
        return dataclasses.replace(run, warnings=(*run.warnings, note))
    return dataclasses.replace(run, excel_paths=tuple(written))


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)


def _estimate_at(results: NowcastResults, period: pd.Period) -> float:
    frame = results.nowcast
    key: Any = period
    value = frame["in_sample"].combine_first(frame["out_of_sample"]).loc[key]
    return float(value)


def _estimate(spec: NowcastSpec, timer: _Timer, today: pd.Timestamp | None) -> _State:
    """Data, vintage, preprocessing and estimation stages."""
    with timer.stage("data"):
        loaded = load_data(spec)
        vintage = spec.vintage_timestamp(today)
        data = apply_vintage(loaded, vintage, explicit=spec.vintage != "today")
    with timer.stage("preprocessing"):
        panel = preprocess(data, spec)
    with timer.stage("estimation"):
        results = _fit(spec, panel)
    return _State(spec, vintage, loaded, data, panel, results, notes=[])


def _fit(spec: NowcastSpec, panel: MixedFrequencyData) -> NowcastResults:
    from nowcastbox.api import nowcast

    model = spec.model
    return nowcast(
        panel,
        target=spec.target,
        method=model.method,
        n_factors=None if isinstance(model.n_factors, str) else model.n_factors,
        factor_lags=model.factor_lags,
        n_shocks=cast("Any", model.n_shocks),
        blocks=_resolve_blocks(model.n_factors, model.blocks, panel),
        horizon=model.horizon,
        rmax=model.rmax,
        criterion=model.criterion,
        **model.options,
    )


def _resolve_blocks(n_factors: Any, blocks: Any, panel: MixedFrequencyData) -> Any:
    """Restrict the data's block structure to the blocks listed in ``model.factors``.

    With ``factors: {global: 1, real: 1}`` and ``blocks: data`` (the default for a
    mapping), blocks of the metadata that are not listed are ignored, so a dataset with
    many blocks can be used with a few of them (plan §6.2 example).
    """
    if not isinstance(n_factors, dict) or blocks not in ("data", "auto"):
        return blocks
    available = list(panel.block_names)
    unknown = [b for b in n_factors if b not in available]
    if unknown:
        raise SpecError(
            [
                (
                    "model.factors",
                    f"blocks {unknown} have no series in the data (available: {available})",
                )
            ]
        )
    membership = {c: [b for b in panel.metadata[c].blocks if b in n_factors] for c in panel.columns}
    orphans = [c for c, b in membership.items() if not b]
    if orphans:
        raise SpecError(
            [
                (
                    "model.factors",
                    f"series {orphans} load on none of the blocks {list(n_factors)}; add one "
                    "of their blocks or drop them with data.columns",
                )
            ]
        )
    return membership


# ---------------------------------------------------------------------------- outputs
def _outputs(state: _State, timer: _Timer) -> None:
    outputs = state.spec.outputs
    stages: list[tuple[str, Any, Callable[[_State, Any], None]]] = [
        ("density", outputs.density, _density),
        ("news", outputs.news, _news),
        ("diagnostics", outputs.diagnostics, _diagnostics),
        ("backtest", outputs.backtest, _backtest),
        ("empirical_bands", outputs.empirical_bands, _empirical_bands),
        ("heatmap", outputs.heatmap, _heatmap),
        ("alternatives", outputs.alternatives, _alternatives),
        ("report_html", outputs.report_html, _report),
    ]
    for name, options, stage in stages:
        if options is None:
            continue
        with timer.stage(name):
            try:
                stage(state, options)
            except SpecError:
                raise
            except Exception as err:  # optional outputs never abort a run
                state.notes.append(f"{name} output skipped: {type(err).__name__}: {err}")


def _density(state: _State, options: DensityOutput) -> None:
    from nowcastbox.api import add_density

    state.results = add_density(
        state.results, n_boot=options.n_boot, random_state=state.spec.random_state
    )
    state.distribution = state.results.info.get("distribution")


def _news(state: _State, options: NewsOutput) -> None:
    old, reference = _old_vintage(state, options.against)
    if old is None:
        state.notes.append("news output skipped: no previous snapshot of this spec yet (first run)")
        return
    period = headline_period(state.results.nowcast)
    state.news = state.results.news(old=old, new=state.panel, target_period=period)
    state.news_reference = reference


def _old_vintage(state: _State, against: str) -> tuple[Any, str | None]:
    if against == "previous_snapshot":
        prev = state.previous
        if prev is None:
            return None, None
        panel = prev.panel()
        return (None, None) if panel is None else (panel, prev.id)
    date = pd.Timestamp(against)
    if date >= state.vintage:
        raise SpecError(
            [
                (
                    "outputs.news.against",
                    f"{against} is not before the vintage {state.vintage.date()}",
                )
            ]
        )
    old = preprocess(apply_vintage(state.loaded, date), state.spec)
    return old, str(date.date())


def _diagnostics(state: _State, options: DiagnosticsOutput) -> None:
    state.diagnostics = state.results.diagnostics(**options.options)


def _backtest(state: _State, options: BacktestOutput) -> None:
    from nowcastbox import benchmarks as bench_module
    from nowcastbox.evaluation import PseudoRealTimeBacktest

    params = {**state.results.model_params, **options.model}
    model = state.spec.model.estimator_class(**params)
    benches = [getattr(bench_module, name)(**kwargs) for name, kwargs in options.benchmarks]
    # Backtest on the uncleaned vintage data and re-run the spec's preprocessing on
    # each vintage, so outlier rules and gap filling never see future observations.
    delays = state.data.release_delays
    if delays.isna().any():
        raise SpecError(
            [("outputs.backtest", "the backtest needs publication delays for every series")]
        )
    fit_kwargs = {"horizon": state.spec.model.horizon} if state.spec.model.method == "em" else None
    backtest = PseudoRealTimeBacktest(
        model=model,
        data=state.data,
        target=state.spec.target_name,
        delay={k: int(v) for k, v in delays.items()},
        start=options.start,
        end=options.end,
        step=options.step,
        benchmarks=benches,
        window=options.window,
        window_length=options.window_length,
        refit_every=options.refit_every,
        target_offsets=options.target_offsets,
        fit_kwargs=fit_kwargs,
        preprocess=lambda vintage: preprocess(vintage, state.spec),
    )
    state.backtest = backtest.run()
    state.backtest_metrics = state.backtest.metrics(
        metrics=options.metrics, periods=options.periods
    )


def _empirical_bands(state: _State, options: EmpiricalBandsOutput) -> None:
    from nowcastbox.density import empirical_bands

    if state.backtest is None:
        state.notes.append("empirical_bands output skipped: the backtest output failed")
        return
    state.bands = empirical_bands(
        state.results,
        state.backtest,
        vintage=state.vintage,
        window=options.window,
        method=cast("Any", options.method),
        levels=options.levels,
        outliers=cast("Any", options.outliers),
        min_errors=options.min_errors,
        availability=cast("Any", options.availability),
    )


def _heatmap(state: _State, options: HeatmapOutput) -> None:
    from nowcastbox.diagnostics import indicator_zscores

    panel = state.panel
    target = state.spec.target_name
    state.zscores = indicator_zscores(
        panel,
        smooth=options.smooth,
        window=options.window,
        by=options.by,
        series=[c for c in panel.columns if c != target],
    )


def _alternatives(state: _State, options: AlternativesOutput) -> None:
    from nowcastbox.experiment import alternative_models

    data = state.results.data if state.results.data is not None else state.panel
    if options.refit:
        model: Any = state.spec.model.estimator_class(**state.results.model_params)
    else:
        model = state.results
    fit_kwargs = {"horizon": state.spec.model.horizon} if state.spec.model.method == "em" else None
    state.alternatives = alternative_models(
        model,
        data,
        state.spec.target_name,
        by=options.by,
        drop=options.drop,
        refit=options.refit,
        fit_kwargs=fit_kwargs,
    )


def _report(state: _State, options: ReportOutput) -> None:
    from nowcastbox.reports import NowcastReport

    heatmap = state.spec.outputs.heatmap
    report = NowcastReport(
        state.results,
        title=options.title or f"Nowcast of {state.spec.target_name} ({state.spec.name})",
        news=state.news,
        quantiles=state.distribution,
        backtest=state.backtest,
        backtest_metrics=state.backtest_metrics,
        bands=state.bands,
        alternatives=state.alternatives,
        heatmap=state.zscores,
        heatmap_last=24 if heatmap is None else heatmap.last,
        diagnostics=state.diagnostics if state.diagnostics is not None else False,
        author=options.author,
        notes=options.notes,
        n_periods=options.n_periods,
        plotlyjs="cdn" if options.plotlyjs == "cdn" else "inline",
    )
    state.report_html = report.to_html()
    if options.path is not None:
        options.path.parent.mkdir(parents=True, exist_ok=True)
        options.path.write_text(state.report_html, encoding="utf-8")
        state.report_path = options.path


# ---------------------------------------------------------------------------- snapshot
def _write_snapshot(
    store: SnapshotStore, state: _State, period: pd.Period | None, notes: list[str]
) -> Snapshot:
    results = state.results
    tables: dict[str, pd.DataFrame] = {}
    texts: dict[str, str] = {}
    if results.loadings is not None:
        tables["loadings"] = results.loadings
    if results.factors is not None:
        tables["factors"] = results.factors
    if state.distribution is not None:
        tables["density"] = state.distribution.to_frame()
    if state.diagnostics is not None:
        tables["diagnostics"] = state.diagnostics.series_overview()
        texts["diagnostics.txt"] = state.diagnostics.summary()
    if state.backtest is not None:
        tables["backtest"] = state.backtest.to_frame()
        tables["backtest_rmsfe"] = state.backtest.rmsfe_by_horizon()
    tables.update(_phase_tables(state))
    if state.report_html is not None:
        texts["report.html"] = state.report_html
    news_table = None
    news_info = None
    if state.news is not None:
        options = state.spec.outputs.news
        news_table = state.news.to_frame(by=options.by if options is not None else "series")
        news_info = _news_info(state.news, state.news_reference)
    return store.write(
        name=state.spec.name,
        target=state.spec.target_name,
        nowcast=results.nowcast,
        vintage=str(state.vintage.date()),
        data_hash=data_hash(state.data),
        model=results.model_name,
        headline_period=period,
        spec=state.spec.to_dict(),
        panel=state.panel.to_frame(),
        params=_params(results),
        news=news_table,
        news_info=news_info,
        tables=tables,
        texts=texts,
        warnings=notes,
        metadata={"nowcastbox_version": _version(), "outputs": list(state.spec.outputs.names)},
    )


def _phase_tables(state: _State) -> dict[str, pd.DataFrame]:
    """Tables of the accuracy, bands, heatmap and alternative-model outputs."""
    tables: dict[str, pd.DataFrame] = {}
    if state.backtest_metrics is not None:
        tables["backtest_metrics"] = state.backtest_metrics
    if state.bands is not None:
        tables["empirical_bands"] = state.bands.to_frame()
    if state.zscores is not None:
        tables["heatmap"] = state.zscores.zscores
        if state.zscores.groups is not None:
            tables["heatmap_groups"] = state.zscores.groups
    if state.alternatives is not None:
        tables["alternatives"] = state.alternatives.table()
        tables["alternatives_range"] = state.alternatives.range()
    return tables


def _params(results: NowcastResults) -> dict[str, Any]:
    return {
        "model_name": results.model_name,
        "model_params": results.model_params,
        "n_factors": results.n_factors,
        "loglikelihood": results.loglikelihood,
        "n_iter": results.n_iter,
        "converged": results.converged,
        "estimates": results.params,
    }


def _version() -> str:
    from nowcastbox.__version__ import __version__

    return __version__

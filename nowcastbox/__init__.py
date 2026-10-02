"""NowcastBox: nowcasting with dynamic factor models in Python.

The most used objects are available at the top level (``nb.MixedFreqDFM``,
``nb.prepare_panel``, ``nb.select_factors``, ``nb.pseudo_real_time``, ``nb.nowcast``,
``nb.news_decomposition``, ``nb.PseudoRealTimeBacktest``, ``nb.nowcast_distribution``,
``nb.run_diagnostics``, ``nb.NowcastReport``, ``nb.load_brazil_nowcast``,
``nb.select_blocks``, ``nb.run_pipeline``...); the subpackages (``nb.preprocessing``,
``nb.statespace``, ``nb.models``, ``nb.selection``, ``nb.vintages``,
``nb.data_sources``, ``nb.datasets``, ``nb.news``, ``nb.density``, ``nb.evaluation``,
``nb.benchmarks``, ``nb.diagnostics``, ``nb.visualization``, ``nb.reports``,
``nb.experiment``, ``nb.pipeline``, ``nb.cli``, ``nb.simulate``) hold the complete API.

Production use (innovation I10): ``nb.run_pipeline("nowcast.yaml")`` or the
``nowcastbox run nowcast.yaml`` command line, with versioned snapshots
(``nb.SnapshotStore``).

Fitted results also expose the wave-2 tools as methods: ``res.news(old, new, period)``,
``res.nowcast_tracker(...)``, ``res.level_contributions()``, ``res.distribution()``,
``res.diagnostics()`` and ``res.plot(kind)``.

ECB-toolbox parity (0.2.0): ``nb.empirical_bands`` (Reifschneider-Tulip error bands),
``nb.indicator_zscores`` (z-score heatmap), ``nb.alternative_models`` (nowcasts without
one or two groups of indicators), ``nb.pesaran_timmermann`` (directional accuracy test),
``MixedFrequencyData.released_share`` and ``BacktestResults.metrics(..., periods=...)``.

ECB-toolbox parity (0.3.0, model building): ``nb.preselect`` (t-stat, SIS and LARS
rankings of the indicators), ``nb.SpecificationSearch`` (random or grid search of
specifications scored in pseudo real time, with a Covid robustness step) and
``nb.BridgeCombination`` (combination of all small bridge equations).

ECB-toolbox parity (0.4.0): ``nb.LargeBVAR`` (large mixed-frequency Bayesian VAR with
blocking, GLP hierarchical prior, conditional forecasts, posterior densities and news)
and ``nb.BVARExtrapolator`` (``BridgeCombination(extrapolation="bvar")``).

Examples
--------
>>> import nowcastbox as nb
>>> isinstance(nb.__version__, str)
True
>>> from nowcastbox.models.two_step import simulate_two_step_example
>>> data = simulate_two_step_example(random_state=0)
>>> res = nb.TwoStepDFM(n_factors=1).fit(data, target="gdp")
>>> list(res.nowcast.columns)[:3]
['observed', 'in_sample', 'out_of_sample']
"""

from nowcastbox import (
    benchmarks,
    cli,
    core,
    data_sources,
    datasets,
    density,
    diagnostics,
    evaluation,
    experiment,
    models,
    news,
    pipeline,
    preprocessing,
    reports,
    selection,
    simulate,
    statespace,
    utils,
    vintages,
    visualization,
)
from nowcastbox.__version__ import __version__
from nowcastbox.api import nowcast
from nowcastbox.core import (
    AggregationType,
    BaseBenchmark,
    BaseNowcaster,
    ConvergenceWarning,
    DataQualityWarning,
    FactorResults,
    FormulaError,
    Frequency,
    MixedFrequencyData,
    ModelNotFittedError,
    NowcastBoxError,
    NowcastBoxWarning,
    NowcastDataError,
    NowcastResults,
    SeriesCategory,
    SeriesMetadata,
    as_mixed_frequency_data,
    parse_formula,
)
from nowcastbox.datasets import (
    Dataset,
    list_datasets,
    load_brazil_calendar,
    load_brazil_nowcast,
    load_brazil_vintages,
    load_dataset,
    load_nyfed,
    load_simulated_dfm,
    load_us_fred_md,
    load_us_grs_like,
)
from nowcastbox.density import NowcastDistribution, empirical_bands, nowcast_distribution
from nowcastbox.diagnostics import DiagnosticsReport, indicator_zscores, run_diagnostics
from nowcastbox.evaluation import (
    BacktestResults,
    PseudoRealTimeBacktest,
    pesaran_timmermann,
    scoring,
)
from nowcastbox.experiment import AlternativeNowcasts, NowcastExperiment, alternative_models
from nowcastbox.models import (
    BridgeCombination,
    BridgeCombinationResults,
    BridgeEquation,
    BridgeResults,
    BVARExtrapolator,
    LargeBVAR,
    LargeBVARResults,
    MixedFreqDFM,
    MixedFreqDFMResults,
    TwoStepDFM,
    TwoStepResults,
)
from nowcastbox.news import (
    LevelContributions,
    NewsResults,
    NowcastTracker,
    level_contributions,
    news_decomposition,
    nowcast_tracker,
)
from nowcastbox.pipeline import NowcastSpec, PipelineRun, SnapshotStore, run_pipeline
from nowcastbox.preprocessing import (
    CalendarAggregation,
    Compose,
    Diff,
    Identity,
    Log,
    PctChange,
    Scale,
    TemporalAggregation,
    Transform,
    apply_transforms,
    calendar_aggregation,
    invert_transforms,
    month_to_quarter,
    prepare_panel,
    quarter_to_month,
)
from nowcastbox.reports import NowcastReport
from nowcastbox.selection import (
    CovidRobustness,
    FactorSelectionResult,
    PreselectionResult,
    SearchResults,
    SelectionPath,
    ShockSelectionResult,
    SpecificationSearch,
    SpecifiedModel,
    ValidationSettings,
    hard_threshold,
    preselect,
    select_blocks,
    select_factors,
    select_shocks,
    select_targeted_predictors,
    select_variables,
    soft_threshold,
)
from nowcastbox.statespace import (
    StateSpace,
    kalman_filter,
    kalman_smoother,
    loglikelihood,
    simulate_state_space,
)
from nowcastbox.vintages import (
    ReleaseCalendar,
    Vintage,
    VintageStore,
    generate_vintages,
    pseudo_real_time,
)

__all__ = [
    "AggregationType",
    "AlternativeNowcasts",
    "BVARExtrapolator",
    "BacktestResults",
    "BaseBenchmark",
    "BaseNowcaster",
    "BridgeCombination",
    "BridgeCombinationResults",
    "BridgeEquation",
    "BridgeResults",
    "CalendarAggregation",
    "Compose",
    "ConvergenceWarning",
    "CovidRobustness",
    "DataQualityWarning",
    "Dataset",
    "DiagnosticsReport",
    "Diff",
    "FactorResults",
    "FactorSelectionResult",
    "FormulaError",
    "Frequency",
    "Identity",
    "LargeBVAR",
    "LargeBVARResults",
    "LevelContributions",
    "Log",
    "MixedFreqDFM",
    "MixedFreqDFMResults",
    "MixedFrequencyData",
    "ModelNotFittedError",
    "NewsResults",
    "NowcastBoxError",
    "NowcastBoxWarning",
    "NowcastDataError",
    "NowcastDistribution",
    "NowcastExperiment",
    "NowcastReport",
    "NowcastResults",
    "NowcastSpec",
    "NowcastTracker",
    "PctChange",
    "PipelineRun",
    "PreselectionResult",
    "PseudoRealTimeBacktest",
    "ReleaseCalendar",
    "Scale",
    "SearchResults",
    "SelectionPath",
    "SeriesCategory",
    "SeriesMetadata",
    "ShockSelectionResult",
    "SnapshotStore",
    "SpecificationSearch",
    "SpecifiedModel",
    "StateSpace",
    "TemporalAggregation",
    "Transform",
    "TwoStepDFM",
    "TwoStepResults",
    "ValidationSettings",
    "Vintage",
    "VintageStore",
    "__version__",
    "alternative_models",
    "apply_transforms",
    "as_mixed_frequency_data",
    "benchmarks",
    "calendar_aggregation",
    "cli",
    "core",
    "data_sources",
    "datasets",
    "density",
    "diagnostics",
    "empirical_bands",
    "evaluation",
    "experiment",
    "generate_vintages",
    "hard_threshold",
    "indicator_zscores",
    "invert_transforms",
    "kalman_filter",
    "kalman_smoother",
    "level_contributions",
    "list_datasets",
    "load_brazil_calendar",
    "load_brazil_nowcast",
    "load_brazil_vintages",
    "load_dataset",
    "load_nyfed",
    "load_simulated_dfm",
    "load_us_fred_md",
    "load_us_grs_like",
    "loglikelihood",
    "models",
    "month_to_quarter",
    "news",
    "news_decomposition",
    "nowcast",
    "nowcast_distribution",
    "nowcast_tracker",
    "parse_formula",
    "pesaran_timmermann",
    "pipeline",
    "prepare_panel",
    "preprocessing",
    "preselect",
    "pseudo_real_time",
    "quarter_to_month",
    "reports",
    "run_diagnostics",
    "run_pipeline",
    "scoring",
    "select_blocks",
    "select_factors",
    "select_shocks",
    "select_targeted_predictors",
    "select_variables",
    "selection",
    "simulate",
    "simulate_state_space",
    "soft_threshold",
    "statespace",
    "utils",
    "vintages",
    "visualization",
]

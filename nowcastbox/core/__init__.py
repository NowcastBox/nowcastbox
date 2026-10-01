"""Core contracts shared by every nowcastbox module.

* :mod:`~nowcastbox.core.exceptions` - errors and warnings.
* :mod:`~nowcastbox.core.frequency` - :class:`Frequency`, :class:`AggregationType` and
  calendar helpers.
* :mod:`~nowcastbox.core.data` - :class:`MixedFrequencyData`, the panel container.
* :mod:`~nowcastbox.core.results` - :class:`NowcastResults` / :class:`FactorResults`.
* :mod:`~nowcastbox.core.base` - :class:`BaseNowcaster` and :class:`BaseBenchmark`.
* :mod:`~nowcastbox.core.formula` - ``"y ~ ."`` formulas.

See ``desenvolvimento/CONTRATOS.md`` for the conventions all modules rely on.
"""

from nowcastbox.core.base import BaseBenchmark, BaseNowcaster, BenchmarkForecaster, ParamsMixin
from nowcastbox.core.data import (
    FrequencySpec,
    MixedFrequencyData,
    SeriesCategory,
    SeriesMetadata,
    StandardizationStats,
    as_mixed_frequency_data,
)
from nowcastbox.core.exceptions import (
    ConvergenceWarning,
    DataQualityWarning,
    FormulaError,
    ModelNotFittedError,
    NowcastBoxError,
    NowcastBoxWarning,
    NowcastDataError,
)
from nowcastbox.core.formula import ParsedFormula, is_formula, parse_formula, resolve_target
from nowcastbox.core.frequency import (
    AggregationType,
    Frequency,
    aggregation_ratio,
    base_to_native,
    calendar_position,
    infer_frequency,
    is_fixed_ratio,
    is_period_end,
    max_periods_per,
    native_period_bounds,
    native_to_base,
    period_to_base,
)
from nowcastbox.core.results import (
    NOWCAST_COLUMNS,
    FactorResults,
    NowcastResults,
    available_plots,
    build_nowcast_frame,
    register_plot,
)

__all__ = [
    "NOWCAST_COLUMNS",
    "AggregationType",
    "BaseBenchmark",
    "BaseNowcaster",
    "BenchmarkForecaster",
    "ConvergenceWarning",
    "DataQualityWarning",
    "FactorResults",
    "FormulaError",
    "Frequency",
    "FrequencySpec",
    "MixedFrequencyData",
    "ModelNotFittedError",
    "NowcastBoxError",
    "NowcastBoxWarning",
    "NowcastDataError",
    "NowcastResults",
    "ParamsMixin",
    "ParsedFormula",
    "SeriesCategory",
    "SeriesMetadata",
    "StandardizationStats",
    "aggregation_ratio",
    "as_mixed_frequency_data",
    "available_plots",
    "base_to_native",
    "build_nowcast_frame",
    "calendar_position",
    "infer_frequency",
    "is_fixed_ratio",
    "is_formula",
    "is_period_end",
    "max_periods_per",
    "native_period_bounds",
    "native_to_base",
    "parse_formula",
    "period_to_base",
    "register_plot",
    "resolve_target",
]

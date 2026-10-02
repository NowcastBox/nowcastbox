"""Real-time forecast evaluation.

Pseudo real-time (and real-time, with a :class:`~nowcastbox.vintages.VintageStore`)
backtesting, accuracy metrics by nowcast horizon and forecast comparison tests
(Diebold-Mariano with the Harvey-Leybourne-Newbold correction, Clark-West for nested
models, Giacomini-White, Model Confidence Set) and density scores
(:mod:`nowcastbox.evaluation.scoring`: CRPS, log score, PIT, Berkowitz/KS uniformity,
interval coverage and Christoffersen tests, quantile scores; innovation I5).

Examples
--------
>>> import numpy as np
>>> from nowcastbox.evaluation import diebold_mariano, rmsfe
>>> rmsfe([3.0, -4.0])
3.5355339059327378
>>> rng = np.random.default_rng(0)
>>> diebold_mariano(rng.normal(0, 1, 100), rng.normal(0, 3, 100)).pvalue < 0.01
True
"""

from nowcastbox.evaluation import scoring
from nowcastbox.evaluation.backtest import (
    FORECAST_COLUMNS,
    BacktestResults,
    PseudoRealTimeBacktest,
)
from nowcastbox.evaluation.metrics import (
    METRICS,
    accuracy_by_horizon,
    bias,
    forecast_errors,
    loss_values,
    mae,
    metric_by_horizon,
    mse,
    relative_rmsfe,
    rmsfe,
)
from nowcastbox.evaluation.scoring import (
    BerkowitzTestResult,
    CoverageTestResult,
    UniformityTestResult,
    berkowitz_test,
    christoffersen_test,
    crps,
    crps_gaussian,
    crps_mixture,
    crps_sample,
    interval_coverage,
    interval_hits,
    interval_score,
    ks_uniformity_test,
    log_score,
    log_score_gaussian,
    log_score_sample,
    pit,
    quantile_score,
    weighted_quantile_score,
)
from nowcastbox.evaluation.tests import (
    ClarkWestResult,
    DieboldMarianoResult,
    GiacominiWhiteResult,
    ModelConfidenceSetResult,
    clark_west,
    clark_west_differential,
    clark_west_from_differential,
    diebold_mariano,
    giacomini_white,
    model_confidence_set,
)

__all__ = [
    "FORECAST_COLUMNS",
    "METRICS",
    "BacktestResults",
    "BerkowitzTestResult",
    "ClarkWestResult",
    "CoverageTestResult",
    "DieboldMarianoResult",
    "GiacominiWhiteResult",
    "ModelConfidenceSetResult",
    "PseudoRealTimeBacktest",
    "UniformityTestResult",
    "accuracy_by_horizon",
    "berkowitz_test",
    "bias",
    "christoffersen_test",
    "clark_west",
    "clark_west_differential",
    "clark_west_from_differential",
    "crps",
    "crps_gaussian",
    "crps_mixture",
    "crps_sample",
    "diebold_mariano",
    "forecast_errors",
    "giacomini_white",
    "interval_coverage",
    "interval_hits",
    "interval_score",
    "ks_uniformity_test",
    "log_score",
    "log_score_gaussian",
    "log_score_sample",
    "loss_values",
    "mae",
    "metric_by_horizon",
    "model_confidence_set",
    "mse",
    "pit",
    "quantile_score",
    "relative_rmsfe",
    "rmsfe",
    "scoring",
    "weighted_quantile_score",
]

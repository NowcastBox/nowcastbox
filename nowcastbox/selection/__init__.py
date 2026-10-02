"""Model selection.

* Number of static factors - Bai & Ng (2002) ``IC_p`` and ``PC_p`` criteria
  (:func:`select_factors`).
* Number of primitive dynamic shocks - Bai & Ng (2007) (:func:`select_shocks`).
* Block / variable selection validated in pseudo real time (innovation I7): greedy
  forward/backward search scored by pseudo real-time RMSFE or an information criterion
  (:func:`select_blocks`, :func:`select_variables`, :class:`SelectionPath`).
* Targeted predictors - Bai & Ng (2008) hard and soft (elastic-net) thresholding,
  sure independence screening (Fan & Lv, 2008) and least angle regression (Efron et al.,
  2004) (:func:`hard_threshold`, :func:`soft_threshold`, :func:`sis`,
  :func:`lars_select`, :func:`lars_path`, :func:`select_targeted_predictors`).
* Pre-selection of indicators as in the ECB toolbox: t-stat, SIS and LARS rankings
  of the indicators aggregated to the target frequency, with leads/lags and a weighted
  aggregated score (:func:`preselect`, :class:`PreselectionResult`).
* Specification search as in the ECB toolbox: random or grid search of model
  specifications (factors, lags, sample start, number of indicators taken from the
  pre-selection ranking - funnel strategy) scored in pseudo real time by a weighted
  score over horizons and metrics, with checkpointing and a Covid robustness step
  (:class:`SpecificationSearch`, :class:`SearchResults`, :class:`CovidRobustness`).
"""

from nowcastbox.selection._lars import LarsPath, lars_path
from nowcastbox.selection.bai_ng_factors import (
    CRITERIA,
    MIN_RELIABLE_DIM,
    FactorSelectionResult,
    bai_ng_penalty,
    factor_criteria,
    select_factors,
)
from nowcastbox.selection.bai_ng_shocks import (
    DEFAULT_M,
    ShockSelectionResult,
    select_shocks,
    shock_bound,
    shock_statistics,
)
from nowcastbox.selection.blocks import (
    SCORINGS,
    SelectionPath,
    ValidationSettings,
    select_blocks,
    select_variables,
)
from nowcastbox.selection.preselection import (
    PRESELECTION_METHODS,
    PreselectionResult,
    align_to_target,
    preselect,
    rank_scores,
)
from nowcastbox.selection.search import (
    COVID_TREATMENTS,
    NORMALIZATIONS,
    SCORE_METRICS,
    SPECIAL_SETTINGS,
    CovidRobustness,
    OutlierCorrection,
    ParameterSpace,
    SearchResults,
    SpecificationSearch,
    SpecifiedModel,
    TreatmentPlan,
    weighted_score,
)
from nowcastbox.selection.targeted import (
    TargetedPredictorsResult,
    elastic_net,
    elastic_net_path,
    hard_threshold,
    lars_select,
    newey_west_lags,
    select_targeted_predictors,
    sis,
    soft_threshold,
)

__all__ = [
    "COVID_TREATMENTS",
    "CRITERIA",
    "DEFAULT_M",
    "MIN_RELIABLE_DIM",
    "NORMALIZATIONS",
    "PRESELECTION_METHODS",
    "SCORE_METRICS",
    "SCORINGS",
    "SPECIAL_SETTINGS",
    "CovidRobustness",
    "FactorSelectionResult",
    "LarsPath",
    "OutlierCorrection",
    "ParameterSpace",
    "PreselectionResult",
    "SearchResults",
    "SelectionPath",
    "ShockSelectionResult",
    "SpecificationSearch",
    "SpecifiedModel",
    "TargetedPredictorsResult",
    "TreatmentPlan",
    "ValidationSettings",
    "align_to_target",
    "bai_ng_penalty",
    "elastic_net",
    "elastic_net_path",
    "factor_criteria",
    "hard_threshold",
    "lars_path",
    "lars_select",
    "newey_west_lags",
    "preselect",
    "rank_scores",
    "select_blocks",
    "select_factors",
    "select_shocks",
    "select_targeted_predictors",
    "select_variables",
    "shock_bound",
    "shock_statistics",
    "sis",
    "soft_threshold",
    "weighted_score",
]

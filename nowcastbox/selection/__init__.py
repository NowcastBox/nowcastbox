"""Model selection.

* Number of static factors - Bai & Ng (2002) ``IC_p`` and ``PC_p`` criteria
  (:func:`select_factors`).
* Number of primitive dynamic shocks - Bai & Ng (2007) (:func:`select_shocks`).
* Block / variable selection validated in pseudo real time (innovation I7): greedy
  forward/backward search scored by pseudo real-time RMSFE or an information criterion
  (:func:`select_blocks`, :func:`select_variables`, :class:`SelectionPath`).
* Targeted predictors - Bai & Ng (2008) hard and soft (elastic-net) thresholding
  (:func:`hard_threshold`, :func:`soft_threshold`, :func:`select_targeted_predictors`).
"""

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
from nowcastbox.selection.targeted import (
    TargetedPredictorsResult,
    elastic_net,
    elastic_net_path,
    hard_threshold,
    newey_west_lags,
    select_targeted_predictors,
    soft_threshold,
)

__all__ = [
    "CRITERIA",
    "DEFAULT_M",
    "MIN_RELIABLE_DIM",
    "SCORINGS",
    "FactorSelectionResult",
    "SelectionPath",
    "ShockSelectionResult",
    "TargetedPredictorsResult",
    "ValidationSettings",
    "bai_ng_penalty",
    "elastic_net",
    "elastic_net_path",
    "factor_criteria",
    "hard_threshold",
    "newey_west_lags",
    "select_blocks",
    "select_factors",
    "select_shocks",
    "select_targeted_predictors",
    "select_variables",
    "shock_bound",
    "shock_statistics",
    "soft_threshold",
]

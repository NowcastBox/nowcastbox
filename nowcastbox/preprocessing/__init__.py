"""Pre-processing of mixed-frequency panels.

Transformations (codes 0-7 and named, invertible transforms), outlier detection and
correction, missing-value treatment outside the ragged edge, standardisation and
temporal aggregation (Mariano-Murasawa, month-to-quarter, quarter-to-month).
See plan section 3.1.

Modules
-------
transforms
    :class:`Transform` and its subclasses (``Diff``, ``PctChange``, ``Log``, ``Scale``,
    ``Identity``, ``Compose`` via ``|``), :func:`get_transform`, legacy codes 0-7.
outliers
    IQR rule and centred moving-median replacement.
missing
    Ragged-edge / leading / interior masks and gap filling.
aggregation
    Temporal aggregation weights and loading constraints (any frequency pair),
    :func:`month_to_quarter`, :func:`quarter_to_month`.
panel
    :func:`prepare_panel`, :func:`standardize`, :func:`destandardize`.
"""

from nowcastbox.preprocessing.aggregation import (
    CalendarAggregation,
    TemporalAggregation,
    aggregate_panel,
    aggregation_weights,
    calendar_aggregation,
    calendar_weight_matrix,
    loading_constraints,
    mariano_murasawa_weights,
    month_to_quarter,
    quarter_to_month,
    rolling_aggregate,
    temporal_aggregation,
    to_higher_frequency,
    to_lower_frequency,
)
from nowcastbox.preprocessing.missing import (
    fill_missing,
    interior_missing_mask,
    leading_missing_mask,
    missing_proportion,
    ragged_edge_mask,
)
from nowcastbox.preprocessing.outliers import (
    detect_outliers,
    iqr_outlier_mask,
    moving_median,
    replace_outliers,
)
from nowcastbox.preprocessing.panel import PanelReport, destandardize, prepare_panel, standardize
from nowcastbox.preprocessing.transforms import (
    NAMED_TRANSFORMS,
    TRANSFORM_CODES,
    Compose,
    Diff,
    Identity,
    Log,
    PctChange,
    Scale,
    Transform,
    apply_transforms,
    get_transform,
    invert_transforms,
    resolve_lag,
    resolve_transforms,
    transform_from_code,
)

__all__ = [
    "NAMED_TRANSFORMS",
    "TRANSFORM_CODES",
    "CalendarAggregation",
    "Compose",
    "Diff",
    "Identity",
    "Log",
    "PanelReport",
    "PctChange",
    "Scale",
    "TemporalAggregation",
    "Transform",
    "aggregate_panel",
    "aggregation_weights",
    "apply_transforms",
    "calendar_aggregation",
    "calendar_weight_matrix",
    "destandardize",
    "detect_outliers",
    "fill_missing",
    "get_transform",
    "interior_missing_mask",
    "invert_transforms",
    "iqr_outlier_mask",
    "leading_missing_mask",
    "loading_constraints",
    "mariano_murasawa_weights",
    "missing_proportion",
    "month_to_quarter",
    "moving_median",
    "prepare_panel",
    "quarter_to_month",
    "ragged_edge_mask",
    "replace_outliers",
    "resolve_lag",
    "resolve_transforms",
    "rolling_aggregate",
    "standardize",
    "temporal_aggregation",
    "to_higher_frequency",
    "to_lower_frequency",
    "transform_from_code",
]

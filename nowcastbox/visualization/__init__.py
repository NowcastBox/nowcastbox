"""Visualisation: Plotly (interactive, default) and Matplotlib (publication) charts.

Every plotting function takes ``backend="plotly" | "matplotlib"`` and ``theme=``
(see :mod:`~nowcastbox.visualization.themes`) and returns the figure
(``plotly.graph_objects.Figure`` or ``matplotlib.figure.Figure``).

Importing this package registers the plots with
:func:`nowcastbox.core.results.register_plot`, so ``results.plot("forecast")``,
``"factors"``, ``"eigenvalues"``, ``"loadings"``, ``"fan"``, ``"data_availability"``
and ``"loglikelihood"`` work on fitted ``TwoStepResults`` / ``MixedFreqDFMResults``
(see :mod:`~nowcastbox.visualization.registry`).

News waterfalls, nowcast trackers, fan charts, information-criterion curves and
backtest RMSFE charts take plain pandas objects with documented columns.
"""

from nowcastbox.visualization.data_flow import (
    AVAILABILITY_STATES,
    data_availability,
    plot_data_availability,
    release_table,
)
from nowcastbox.visualization.diagnostics import plot_loglikelihood
from nowcastbox.visualization.evaluation import plot_rmsfe_by_horizon, rmsfe_frame
from nowcastbox.visualization.factors import (
    panel_eigenvalues,
    plot_eigenvalues,
    plot_factors,
    plot_loadings,
)
from nowcastbox.visualization.forecast import (
    interval_levels,
    plot_fan_chart,
    plot_forecast,
    quantiles_from_nowcast,
)
from nowcastbox.visualization.news import (
    NEWS_HOVER_COLUMNS,
    news_waterfall_table,
    plot_news_waterfall,
    plot_nowcast_tracker,
    tracker_table,
)
from nowcastbox.visualization.registry import REGISTERED_PLOTS
from nowcastbox.visualization.selection import plot_factor_selection
from nowcastbox.visualization.themes import (
    Theme,
    get_theme,
    list_themes,
    register_theme,
    set_default_theme,
)

__all__ = [
    "AVAILABILITY_STATES",
    "NEWS_HOVER_COLUMNS",
    "REGISTERED_PLOTS",
    "Theme",
    "data_availability",
    "get_theme",
    "interval_levels",
    "list_themes",
    "news_waterfall_table",
    "panel_eigenvalues",
    "plot_data_availability",
    "plot_eigenvalues",
    "plot_factor_selection",
    "plot_factors",
    "plot_fan_chart",
    "plot_forecast",
    "plot_loadings",
    "plot_loglikelihood",
    "plot_news_waterfall",
    "plot_nowcast_tracker",
    "plot_rmsfe_by_horizon",
    "quantiles_from_nowcast",
    "register_theme",
    "release_table",
    "rmsfe_frame",
    "set_default_theme",
    "tracker_table",
]

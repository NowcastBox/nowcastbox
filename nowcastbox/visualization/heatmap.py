"""Heatmap of indicator z-scores (conjunctural "dashboard", ECB WP 3004 §3.4).

:func:`plot_indicator_heatmap` draws the z-scores of
:func:`nowcastbox.diagnostics.indicator_zscores` - by series or by group - on a diverging
colour scale centred at zero (the long-run mean of each indicator). Importing this
module registers it as ``results.plot("indicator_heatmap")``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.frequency import FrequencyLike
from nowcastbox.core.results import NowcastResults, register_plot
from nowcastbox.diagnostics.zscores import IndicatorZScores, indicator_zscores
from nowcastbox.visualization._common import (
    finish_mpl,
    finish_plotly,
    mpl_context,
    new_axes,
    new_plotly_figure,
    resolve,
)
from nowcastbox.visualization.themes import Theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["heatmap_table", "plot_indicator_heatmap"]

_LEVELS = ("auto", "series", "group")


def heatmap_table(
    source: IndicatorZScores | MixedFrequencyData | NowcastResults | pd.DataFrame,
    *,
    level: str = "auto",
    last: int | None = 24,
    frequency: FrequencyLike | None = None,
    **zscore_kwargs: Any,
) -> pd.DataFrame:
    """Z-score table drawn by :func:`plot_indicator_heatmap` (rows x periods).

    Parameters
    ----------
    source : IndicatorZScores, MixedFrequencyData, NowcastResults or pandas.DataFrame
        Computed z-scores, or data passed to
        :func:`~nowcastbox.diagnostics.indicator_zscores` with ``zscore_kwargs``.
    level : {"auto", "series", "group"}, default "auto"
        Rows: groups when the z-scores have a grouping (``"auto"``), else series.
    last : int or None, default 24
        Number of most recent periods (``None``: all).
    frequency : Frequency or str, optional
        Show one column per period of this lower frequency (e.g. ``"Q"``).
    **zscore_kwargs
        ``smooth``, ``window``, ``by``, ``as_of``, ``series``... when ``source`` is not
        already an :class:`~nowcastbox.diagnostics.IndicatorZScores`.

    Returns
    -------
    pandas.DataFrame
        Rows = series or groups, columns = periods.

    Raises
    ------
    ValueError
        Unknown ``level`` or invalid z-score options.
    TypeError
        If ``zscore_kwargs`` are given with an ``IndicatorZScores`` source.

    Examples
    --------
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> data = simulate_two_step_example(random_state=0)
    >>> heatmap_table(data, by="frequency", last=3).shape
    (2, 3)
    """
    if level not in _LEVELS:
        raise ValueError(f"level must be one of {_LEVELS}; got {level!r}.")
    if isinstance(source, IndicatorZScores):
        if zscore_kwargs:
            raise TypeError(f"Unexpected arguments {sorted(zscore_kwargs)} for IndicatorZScores.")
        z = source
    else:
        z = indicator_zscores(source, **zscore_kwargs)
    if level == "auto":
        level = "series" if z.groups is None else "group"
    return z.table(level, last=last, frequency=frequency)


def _colorscale(theme: Theme) -> list[str]:
    return [theme.negative_color, theme.plot_background_color, theme.positive_color]


def plot_indicator_heatmap(
    source: IndicatorZScores | MixedFrequencyData | NowcastResults | pd.DataFrame,
    *,
    level: str = "auto",
    last: int | None = 24,
    frequency: FrequencyLike | None = None,
    zmax: float = 2.0,
    annotate: bool = False,
    backend: str = "plotly",
    theme: Theme | str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    **zscore_kwargs: Any,
) -> Any:
    """Heatmap of indicator z-scores (rows = series or groups, columns = periods).

    Parameters
    ----------
    source : IndicatorZScores, MixedFrequencyData, NowcastResults or pandas.DataFrame
        Computed z-scores, or data (results: their panel without the target) passed to
        :func:`~nowcastbox.diagnostics.indicator_zscores` with ``zscore_kwargs``.
    level : {"auto", "series", "group"}, default "auto"
        Rows: groups when there is a grouping (``"auto"``), else series.
    last : int or None, default 24
        Number of most recent periods shown (``None``: all).
    frequency : Frequency or str, optional
        One column per period of this lower frequency (e.g. ``"Q"``).
    zmax : float, default 2.0
        The colour scale saturates at ``-zmax`` and ``+zmax``.
    annotate : bool, default False
        Print the z-scores in the cells.
    backend : {"plotly", "matplotlib"}, default "plotly"
        Plotting library.
    theme : Theme or str, optional
        Visual theme (negative / background / positive colours).
    title : str, optional
        Title (default ``"Indicator z-scores"``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (Matplotlib only).
    figsize : tuple of float, optional
        Matplotlib figure size.
    **zscore_kwargs
        ``smooth``, ``window``, ``by``, ``as_of``, ``series``... (see
        :func:`~nowcastbox.diagnostics.indicator_zscores`).

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        The figure.

    Raises
    ------
    ValueError
        Non-positive ``zmax``, unknown backend or level, invalid z-score options.
    TypeError
        If ``zscore_kwargs`` are given with an ``IndicatorZScores`` source.

    Examples
    --------
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> data = simulate_two_step_example(random_state=0)
    >>> fig = plot_indicator_heatmap(data, by="frequency", last=12)
    >>> fig.data[0].type, len(fig.data[0].y)
    ('heatmap', 2)
    """
    if not zmax > 0:
        raise ValueError(f"zmax must be positive; got {zmax}.")
    th, be = resolve(theme, backend, ax)
    table = heatmap_table(source, level=level, last=last, frequency=frequency, **zscore_kwargs)
    title = "Indicator z-scores" if title is None else title
    if be == "plotly":
        return _heatmap_plotly(table, th, title, zmax, annotate)
    return _heatmap_mpl(table, th, title, zmax, annotate, ax, figsize)


def _heatmap_plotly(
    table: pd.DataFrame, theme: Theme, title: str, zmax: float, annotate: bool
) -> Any:
    import plotly.graph_objects as go

    colors = _colorscale(theme)
    fig = new_plotly_figure()
    fig.add_trace(
        go.Heatmap(
            z=table.to_numpy(dtype=float),
            x=[str(p) for p in table.columns],
            y=[str(s) for s in table.index],
            zmin=-zmax,
            zmax=zmax,
            zmid=0.0,
            colorscale=[[0.0, colors[0]], [0.5, colors[1]], [1.0, colors[2]]],
            xgap=2,
            ygap=2,
            texttemplate="%{z:.1f}" if annotate else None,
            hovertemplate="%{y} %{x}: %{z:.2f}<extra></extra>",
            hoverongaps=False,
            colorbar={"title": {"text": "z"}},
        )
    )
    fig = finish_plotly(fig, theme, title=title, hovermode="closest", showlegend=False)
    fig.update_layout(height=max(theme.height, 24 * table.shape[0] + 160))
    fig.update_xaxes(showgrid=False, type="category")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def _annotate_mpl(axes: Any, values: np.ndarray, theme: Theme) -> None:
    for (i, j), value in np.ndenumerate(values):
        if np.isfinite(value):
            axes.text(j, i, f"{value:.1f}", ha="center", va="center", color=theme.text_color)


def _heatmap_mpl(
    table: pd.DataFrame,
    theme: Theme,
    title: str,
    zmax: float,
    annotate: bool,
    ax: Axes | None,
    figsize: tuple[float, float] | None,
) -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("nowcastbox_zscore", _colorscale(theme))
    cmap.set_bad(theme.background_color)
    values = table.to_numpy(dtype=float)
    size = figsize or (
        max(6.0, 0.36 * table.shape[1] + 3.5),
        max(2.5, 0.32 * table.shape[0] + 1.6),
    )
    with mpl_context(theme):
        fig, axes = new_axes(ax, theme, size)
        image = axes.imshow(
            np.ma.masked_invalid(values), cmap=cmap, vmin=-zmax, vmax=zmax, aspect="auto"
        )
        step = max(1, table.shape[1] // 12)
        positions = list(range(0, table.shape[1], step))
        axes.set_xticks(positions, [str(table.columns[i]) for i in positions], rotation=45)
        axes.set_yticks(range(table.shape[0]), [str(s) for s in table.index])
        axes.grid(False)
        if annotate:
            _annotate_mpl(axes, values, theme)
        fig.colorbar(image, ax=axes, label="z")
        finish_mpl(axes, theme, title=title, legend=False)
    return fig


register_plot("indicator_heatmap")(plot_indicator_heatmap)

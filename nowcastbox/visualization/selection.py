"""Information-criterion curves for the choice of the number of factors / shocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from nowcastbox.visualization._common import (
    as_float_array,
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

__all__ = ["plot_factor_selection"]


def _selection_input(
    selection: Any,
) -> tuple[pd.DataFrame, str | None, float | None, str]:
    """Criteria table, selected criterion, threshold line and x-axis label."""
    if isinstance(selection, pd.DataFrame):
        return selection.astype(float), None, None, str(selection.index.name or "r")
    criteria = getattr(selection, "criteria", None)
    if isinstance(criteria, pd.DataFrame):  # FactorSelectionResult
        return criteria.astype(float), getattr(selection, "criterion", None), None, "r"
    statistics = getattr(selection, "statistics", None)
    if isinstance(statistics, pd.DataFrame):  # ShockSelectionResult
        bound = getattr(selection, "bound", None)
        return (
            statistics.astype(float),
            getattr(selection, "statistic", None),
            None if bound is None else float(bound),
            "k",
        )
    raise TypeError(
        "Expected a FactorSelectionResult, ShockSelectionResult or a DataFrame of criteria; "
        f"got {type(selection).__name__}."
    )


def plot_factor_selection(
    selection: Any,
    *,
    criteria: list[str] | None = None,
    normalize: bool = False,
    backend: str = "plotly",
    theme: Theme | str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Plot information criteria against the number of factors and mark each minimum.

    Parameters
    ----------
    selection : FactorSelectionResult, ShockSelectionResult or pandas.DataFrame
        Output of :func:`nowcastbox.selection.select_factors` /
        :func:`~nowcastbox.selection.select_shocks`, or a table with one column per
        criterion indexed by the number of factors. For shock statistics the bound is
        drawn as a horizontal line and the marked point is the first ``k`` below it.
    criteria : list of str, optional
        Criteria to draw. Default: the ``IC`` criteria when present (they share a
        scale), else every column.
    normalize : bool, default False
        Rescale each curve to ``[0, 1]`` so criteria on different scales (``IC`` vs
        ``PC``) can share one axis.
    backend : {"plotly", "matplotlib"}, default "plotly"
        Plotting library.
    theme : Theme or str, optional
        Visual theme.
    title : str, optional
        Title (default ``"Information criteria"``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (Matplotlib only).
    figsize : tuple of float, optional
        Matplotlib figure size.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        The figure.

    Raises
    ------
    ValueError
        Unknown criteria, empty table or unknown backend.
    TypeError
        Unsupported input type.

    Examples
    --------
    >>> import pandas as pd
    >>> ic = pd.DataFrame({"IC1": [0.0, -0.4, -0.3], "IC2": [0.0, -0.3, -0.35]})
    >>> fig = plot_factor_selection(ic)
    >>> [t.name for t in fig.data]
    ['IC1', 'IC2', 'IC1 min', 'IC2 min']
    """
    th, be = resolve(theme, backend, ax)
    table, _selected, bound, xlabel = _selection_input(selection)
    if table.empty:
        raise ValueError("The criteria table is empty.")
    if criteria is None:
        ic = [c for c in table.columns if str(c).upper().startswith("IC")]
        criteria = ic or [str(c) for c in table.columns]
    unknown = [c for c in criteria if c not in table.columns]
    if unknown:
        raise ValueError(f"Unknown criteria {unknown}; available: {list(table.columns)}.")
    table = table[criteria]
    if normalize:
        span = (table.max() - table.min()).replace(0.0, 1.0)
        table = (table - table.min()) / span
    if bound is not None:
        below = {c: table.index[table[c] < bound] for c in criteria}
        best = {c: (idx[0] if len(idx) else table[c].idxmin()) for c, idx in below.items()}
    else:
        best = {c: table[c].idxmin() for c in criteria}
    title = "Information criteria" if title is None else title
    ylabel = "Criterion (rescaled)" if normalize else "Criterion"
    if be == "plotly":
        return _selection_plotly(table, best, bound, th, title, xlabel, ylabel)
    return _selection_mpl(table, best, bound, th, title, xlabel, ylabel, ax, figsize)


def _selection_plotly(
    table: pd.DataFrame,
    best: dict[str, Any],
    bound: float | None,
    theme: Theme,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Any:
    import plotly.graph_objects as go

    fig = new_plotly_figure()
    for i, col in enumerate(table.columns):
        fig.add_trace(
            go.Scatter(
                x=list(table.index),
                y=table[col],
                name=str(col),
                mode="lines+markers",
                line={"color": theme.color(i), "width": theme.line_width},
                marker={"size": theme.marker_size * 0.7},
            )
        )
    for i, col in enumerate(table.columns):
        x = best[col]
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[table.loc[x, col]],
                name=f"{col} min",
                mode="markers",
                showlegend=False,
                marker={
                    "size": theme.marker_size * 1.8,
                    "color": theme.color(i),
                    "symbol": "star",
                    "line": {"color": theme.background_color, "width": 2},
                },
                hovertemplate=f"{col}: selected {xlabel}=%{{x}}<extra></extra>",
            )
        )
    if bound is not None:
        fig.add_hline(y=bound, line={"color": theme.muted_text_color, "dash": "dash", "width": 1})
    fig.update_xaxes(dtick=1)
    return finish_plotly(fig, theme, title=title, xlabel=xlabel, ylabel=ylabel, hovermode="closest")


def _selection_mpl(
    table: pd.DataFrame,
    best: dict[str, Any],
    bound: float | None,
    theme: Theme,
    title: str,
    xlabel: str,
    ylabel: str,
    ax: Axes | None,
    figsize: tuple[float, float] | None,
) -> Any:
    with mpl_context(theme):
        fig, axes = new_axes(ax, theme, figsize)
        x = list(table.index)
        for i, col in enumerate(table.columns):
            axes.plot(
                x,
                as_float_array(table[col]),
                color=theme.color(i),
                marker="o",
                markersize=theme.marker_size / 2,
                label=str(col),
            )
            axes.plot(
                [best[col]],
                as_float_array(table[col].loc[[best[col]]]),
                marker="*",
                linestyle="none",
                color=theme.color(i),
                markersize=theme.marker_size * 1.6,
                markeredgecolor=theme.background_color,
            )
        if bound is not None:
            axes.axhline(bound, color=theme.muted_text_color, linestyle="--", linewidth=1)
        axes.set_xticks(x)
        finish_mpl(axes, theme, title=title, xlabel=xlabel, ylabel=ylabel)
    return fig

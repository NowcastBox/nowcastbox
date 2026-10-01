"""Forecast-evaluation charts: RMSFE by nowcast horizon."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
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

__all__ = ["plot_rmsfe_by_horizon", "rmsfe_frame"]


def rmsfe_frame(rmsfe: Any, *, relative_to: str | None = None) -> pd.DataFrame:
    """Normalise RMSFE input to a DataFrame (rows: horizons, columns: models).

    Parameters
    ----------
    rmsfe : pandas.DataFrame, pandas.Series or backtest results
        Table of RMSFE by horizon (index) and model (columns); a Series for one
        model; or an object with a ``rmsfe_by_horizon()`` method returning such a
        table (e.g. the output of a pseudo-real-time backtest).
    relative_to : str, optional
        Divide every column by this model's RMSFE (relative RMSFE; values below 1
        beat the reference).

    Returns
    -------
    pandas.DataFrame
        Float table.

    Raises
    ------
    ValueError
        Empty table, unknown reference model or non-positive reference values.
    TypeError
        Unsupported input type.

    Examples
    --------
    >>> import pandas as pd
    >>> t = pd.DataFrame({"DFM": [0.5, 0.8], "AR": [1.0, 1.0]}, index=[0, 1])
    >>> rmsfe_frame(t, relative_to="AR")["DFM"].tolist()
    [0.5, 0.8]
    """
    if hasattr(rmsfe, "rmsfe_by_horizon") and not isinstance(rmsfe, (pd.DataFrame, pd.Series)):
        rmsfe = rmsfe.rmsfe_by_horizon()
    if isinstance(rmsfe, pd.Series):
        rmsfe = rmsfe.to_frame(name=str(rmsfe.name) if rmsfe.name is not None else "RMSFE")
    if not isinstance(rmsfe, pd.DataFrame):
        raise TypeError(
            "rmsfe must be a DataFrame, a Series or have a rmsfe_by_horizon() method; "
            f"got {type(rmsfe).__name__}."
        )
    if rmsfe.empty:
        raise ValueError("The RMSFE table is empty.")
    table = rmsfe.astype(float)
    if relative_to is not None:
        if relative_to not in table.columns:
            raise ValueError(f"relative_to={relative_to!r} is not a model column.")
        ref = table[relative_to]
        if (ref <= 0).any():
            raise ValueError("Reference RMSFE must be positive to compute relative RMSFE.")
        table = table.div(ref, axis=0)
    return table


def plot_rmsfe_by_horizon(
    rmsfe: Any,
    *,
    relative_to: str | None = None,
    style: Literal["line", "bar"] = "line",
    backend: str = "plotly",
    theme: Theme | str | None = None,
    title: str | None = None,
    xlabel: str = "Horizon",
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """RMSFE of each model by nowcast horizon (backtest summary chart).

    Parameters
    ----------
    rmsfe : pandas.DataFrame, pandas.Series or backtest results
        See :func:`rmsfe_frame`.
    relative_to : str, optional
        Plot RMSFE relative to this model (a reference line is drawn at 1).
    style : {"line", "bar"}, default "line"
        Lines with markers or grouped bars.
    backend : {"plotly", "matplotlib"}, default "plotly"
        Plotting library.
    theme : Theme or str, optional
        Visual theme.
    title : str, optional
        Title (default ``"RMSFE by horizon"`` or ``"Relative RMSFE (vs <model>)"``).
    xlabel : str, default "Horizon"
        X-axis label.
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
        See :func:`rmsfe_frame`; invalid ``style`` or backend.
    TypeError
        Unsupported input type.

    Examples
    --------
    >>> import pandas as pd
    >>> t = pd.DataFrame({"DFM": [0.5, 0.8], "AR": [1.0, 1.1]}, index=[0, 1])
    >>> [trace.name for trace in plot_rmsfe_by_horizon(t).data]
    ['DFM', 'AR']
    """
    th, be = resolve(theme, backend, ax)
    if style not in ("line", "bar"):
        raise ValueError(f"style must be 'line' or 'bar'; got {style!r}.")
    table = rmsfe_frame(rmsfe, relative_to=relative_to)
    if title is None:
        title = "RMSFE by horizon" if relative_to is None else f"Relative RMSFE (vs {relative_to})"
    ylabel = "RMSFE" if relative_to is None else "Relative RMSFE"
    x = [str(h) for h in table.index]
    if be == "plotly":
        import plotly.graph_objects as go

        fig = new_plotly_figure()
        for i, col in enumerate(table.columns):
            if style == "line":
                trace: Any = go.Scatter(
                    x=x,
                    y=table[col],
                    name=str(col),
                    mode="lines+markers",
                    line={"color": th.color(i), "width": th.line_width},
                    marker={"size": th.marker_size},
                )
            else:
                trace = go.Bar(x=x, y=table[col], name=str(col), marker={"color": th.color(i)})
            fig.add_trace(trace)
        if relative_to is not None:
            fig.add_hline(y=1.0, line={"color": th.muted_text_color, "dash": "dash", "width": 1})
        fig.update_xaxes(type="category")
        return finish_plotly(fig, th, title=title, xlabel=xlabel, ylabel=ylabel)
    with mpl_context(th):
        fig, axes = new_axes(ax, th, figsize)
        positions = np.arange(len(x))
        width = 0.8 / table.shape[1]
        for i, col in enumerate(table.columns):
            values = as_float_array(table[col])
            if style == "line":
                axes.plot(
                    positions,
                    values,
                    color=th.color(i),
                    marker="o",
                    markersize=th.marker_size / 1.5,
                    label=str(col),
                )
            else:
                offset = (i - (table.shape[1] - 1) / 2) * width
                axes.bar(
                    positions + offset, values, width=width * 0.9, color=th.color(i), label=str(col)
                )
        if relative_to is not None:
            axes.axhline(1.0, color=th.muted_text_color, linestyle="--", linewidth=1)
        axes.set_xticks(positions, x)
        finish_mpl(axes, th, title=title, xlabel=xlabel, ylabel=ylabel)
    return fig

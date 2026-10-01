"""Plot helpers for selection results (Matplotlib and Plotly backends)."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

__all__ = ["Backend", "plot_lines"]

Backend = Literal["matplotlib", "plotly"]


def plot_lines(
    frame: pd.DataFrame,
    *,
    selected: dict[str, int],
    title: str,
    xlabel: str,
    ylabel: str,
    hline: float | None = None,
    hline_label: str = "",
    backend: Backend = "matplotlib",
    ax: Any = None,
) -> Any:
    """Plot each column of ``frame`` against its index and mark selected points.

    Parameters
    ----------
    frame : pandas.DataFrame
        One line per column, x-axis = index.
    selected : dict[str, int]
        ``{column: x}`` points to highlight (the selected number of factors/shocks).
    title, xlabel, ylabel : str
        Labels.
    hline : float, optional
        Horizontal reference line (e.g. a threshold).
    hline_label : str
        Legend label of the reference line.
    backend : {"matplotlib", "plotly"}
        Plotting library.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (Matplotlib only).

    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure
        The drawn object.

    Raises
    ------
    ValueError
        Unknown backend, or ``ax`` passed with the Plotly backend.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import pandas as pd
    >>> ax = plot_lines(
    ...     pd.DataFrame({"a": [3.0, 1.0, 2.0]}),
    ...     selected={"a": 1},
    ...     title="t",
    ...     xlabel="x",
    ...     ylabel="y",
    ... )
    >>> ax.get_title()
    't'
    """
    if backend == "matplotlib":
        return _plot_matplotlib(frame, selected, title, xlabel, ylabel, hline, hline_label, ax)
    if backend == "plotly":
        if ax is not None:
            raise ValueError("'ax' is only supported by the matplotlib backend.")
        return _plot_plotly(frame, selected, title, xlabel, ylabel, hline, hline_label)
    raise ValueError(f"backend must be 'matplotlib' or 'plotly'; got {backend!r}.")


def _plot_matplotlib(
    frame: pd.DataFrame,
    selected: dict[str, int],
    title: str,
    xlabel: str,
    ylabel: str,
    hline: float | None,
    hline_label: str,
    ax: Any,
) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for col in frame.columns:
        (line,) = ax.plot(frame.index, frame[col], marker="o", label=str(col))
        if col in selected and selected[col] in frame.index:
            x_sel = selected[col]
            ax.plot(
                [x_sel],
                [frame.loc[x_sel, col]],
                marker="*",
                markersize=16,
                color=line.get_color(),
                linestyle="none",
            )
    if hline is not None:
        ax.axhline(hline, color="grey", linestyle="--", label=hline_label or None)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax


def _plot_plotly(
    frame: pd.DataFrame,
    selected: dict[str, int],
    title: str,
    xlabel: str,
    ylabel: str,
    hline: float | None,
    hline_label: str,
) -> Any:
    import plotly.graph_objects as go

    fig = go.Figure()
    for col in frame.columns:
        fig.add_trace(
            go.Scatter(x=list(frame.index), y=frame[col], mode="lines+markers", name=str(col))
        )
        if col in selected and selected[col] in frame.index:
            x_sel = selected[col]
            fig.add_trace(
                go.Scatter(
                    x=[x_sel],
                    y=[frame.loc[x_sel, col]],
                    mode="markers",
                    marker={"symbol": "star", "size": 16},
                    name=f"{col} selected",
                    showlegend=False,
                )
            )
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", annotation_text=hline_label)
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    return fig

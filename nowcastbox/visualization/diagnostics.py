"""Estimation diagnostics: convergence of the EM algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

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

__all__ = ["plot_loglikelihood"]


def plot_loglikelihood(
    results: Any,
    *,
    backend: str = "plotly",
    theme: Theme | str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Log-likelihood at each EM iteration (convergence diagnostic).

    Parameters
    ----------
    results : MixedFreqDFMResults or array-like
        Results with a ``loglikelihood_path`` attribute (first entry: initial
        parameters) or the path itself.
    backend : {"plotly", "matplotlib"}, default "plotly"
        Plotting library.
    theme : Theme or str, optional
        Visual theme.
    title : str, optional
        Title (default ``"EM convergence"``).
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
        No log-likelihood path or unknown backend.

    Examples
    --------
    >>> fig = plot_loglikelihood([-120.0, -100.0, -99.5, -99.4])
    >>> list(fig.data[0].x)
    [0, 1, 2, 3]
    """
    th, be = resolve(theme, backend, ax)
    path = getattr(results, "loglikelihood_path", results)
    if path is None:
        raise ValueError("Results carry no log-likelihood path.")
    values = np.asarray(path, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("The log-likelihood path is empty.")
    iterations = list(range(values.size))
    title = "EM convergence" if title is None else title
    if be == "plotly":
        import plotly.graph_objects as go

        fig = new_plotly_figure()
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=values,
                mode="lines+markers",
                name="Log-likelihood",
                line={"color": th.color(0), "width": th.line_width},
                marker={"size": th.marker_size * 0.6},
            )
        )
        return finish_plotly(
            fig,
            th,
            title=title,
            xlabel="Iteration",
            ylabel="Log-likelihood",
            hovermode="closest",
            showlegend=False,
        )
    with mpl_context(th):
        fig, axes = new_axes(ax, th, figsize)
        axes.plot(iterations, values, color=th.color(0), marker="o", markersize=th.marker_size / 2)
        finish_mpl(axes, th, title=title, xlabel="Iteration", ylabel="Log-likelihood", legend=False)
    return fig

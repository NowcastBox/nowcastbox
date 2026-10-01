"""Backend plumbing shared by the plotting functions (internal)."""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from nowcastbox.visualization.themes import Theme, get_theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from plotly.graph_objects import Figure as PlotlyFigure

__all__ = [
    "BACKENDS",
    "Backend",
    "as_float_array",
    "check_backend",
    "finish_mpl",
    "finish_plotly",
    "mpl_context",
    "new_axes",
    "new_plotly_figure",
    "resolve",
    "rgba",
    "to_plot_index",
]

Backend = Literal["plotly", "matplotlib"]
BACKENDS: tuple[str, str] = ("plotly", "matplotlib")
_HEX = re.compile(r"^#([0-9a-fA-F]{6})$")


def check_backend(backend: str, ax: Any = None) -> Backend:
    """Validate ``backend`` (and that ``ax`` is only used with Matplotlib)."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'; got {backend!r}.")
    if backend == "plotly" and ax is not None:
        raise ValueError("'ax' is only supported by the matplotlib backend.")
    return cast("Backend", backend)


def resolve(theme: Theme | str | None, backend: str, ax: Any = None) -> tuple[Theme, Backend]:
    """Resolve theme and backend arguments in one call."""
    return get_theme(theme), check_backend(backend, ax)


def to_plot_index(index: pd.Index) -> pd.Index:
    """Convert a PeriodIndex to timestamps at the end of each period (normalised).

    Using the period end puts quarterly values in the last month of the quarter, the
    storage convention of :class:`~nowcastbox.core.data.MixedFrequencyData`, so that
    monthly and quarterly series line up on a common time axis.
    """
    if isinstance(index, pd.PeriodIndex):
        return index.to_timestamp(how="end").normalize()
    return index


def rgba(color: str, alpha: float) -> str:
    """CSS ``rgba()`` string of a ``#rrggbb`` colour with transparency ``alpha``."""
    match = _HEX.match(color)
    if match is None:
        raise ValueError(f"Expected a '#rrggbb' colour; got {color!r}.")
    value = match.group(1)
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:g})"


# ---------------------------------------------------------------------- plotly
def new_plotly_figure() -> PlotlyFigure:
    """Empty Plotly figure."""
    import plotly.graph_objects as go

    return go.Figure()


def finish_plotly(
    fig: PlotlyFigure,
    theme: Theme,
    *,
    title: str | None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    hovermode: str | None = None,
    showlegend: bool = True,
) -> PlotlyFigure:
    """Apply the theme, title and axis labels to a Plotly figure."""
    layout = theme.plotly_layout()
    if title is not None:
        layout["title"] = {**layout["title"], "text": title}
    if hovermode is not None:
        layout["hovermode"] = hovermode
    layout["showlegend"] = showlegend
    fig.update_layout(**layout)
    fig.update_xaxes(**theme.plotly_axis())
    fig.update_yaxes(**theme.plotly_axis())
    if xlabel is not None:
        fig.update_xaxes(title_text=xlabel)
    if ylabel is not None:
        fig.update_yaxes(title_text=ylabel)
    return fig


# ---------------------------------------------------------------------- matplotlib
@contextlib.contextmanager
def mpl_context(theme: Theme) -> Iterator[None]:
    """Context in which Matplotlib artists are created with the theme's rcParams."""
    import matplotlib as mpl

    with mpl.rc_context(theme.matplotlib_rc()):
        yield


def new_axes(
    ax: Axes | None, theme: Theme, figsize: tuple[float, float] | None
) -> tuple[Figure, Axes]:
    """Return ``(figure, axes)``: the given axes or a new single-axes figure."""
    if ax is not None:
        return cast("Figure", ax.get_figure()), ax
    import matplotlib.pyplot as plt

    fig, new_ax = plt.subplots(figsize=figsize or theme.figsize, layout="constrained")
    return fig, new_ax


def finish_mpl(
    ax: Axes,
    theme: Theme,
    *,
    title: str | None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    dates: bool = False,
) -> None:
    """Apply title, labels, legend (and a concise date axis) to Matplotlib axes."""
    handles, _ = ax.get_legend_handles_labels()
    show_legend = legend and len(handles) > 1
    if dates:
        import matplotlib.dates as mdates

        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    if title is not None:
        rows = (len(handles) - 1) // 4 + 1 if show_legend else 0
        pad = 6.0 + rows * 1.9 * theme.font_size
        ax.set_title(title, loc="left", color=theme.text_color, pad=pad)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if show_legend:
        # Legend in a row between the title and the plot: never covers the data.
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=min(len(handles), 4),
            labelcolor=theme.text_color,
            borderaxespad=0.3,
        )


def as_float_array(values: Any) -> np.ndarray:
    """``float64`` NumPy array of ``values``."""
    return np.asarray(values, dtype=float)

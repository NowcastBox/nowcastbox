"""Data-flow charts: the ragged edge of a mixed-frequency panel.

Each cell of the panel (series x base period) is classified as

``observed``
    a value is available;
``missing``
    a storage slot inside the sample without a value (a genuine gap);
``pending``
    a storage slot after the series' last observation - a release not yet published
    (the *ragged edge*, Wallis 1986; Giannone, Reichlin & Small 2008);
``no_slot``
    not a storage slot of the series (e.g. the first two months of a quarter for a
    quarterly series).

:func:`released_share_table` and :func:`plot_released_share` summarise the same
information for one target period: the share of its observations already released at
the vintage of the panel (ECB WP 3004, Linzenich & Meunier 2024), by series or group
(``results.plot("released_share")``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.results import NowcastResults, register_plot
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

__all__ = [
    "AVAILABILITY_STATES",
    "data_availability",
    "plot_data_availability",
    "plot_released_share",
    "release_table",
    "released_share_table",
]

AVAILABILITY_STATES: tuple[str, ...] = ("no_slot", "observed", "missing", "pending")
"""Codes ``0..3`` of :func:`data_availability`, in order."""

_LABELS = {
    "no_slot": "No release slot",
    "observed": "Observed",
    "missing": "Missing",
    "pending": "Not yet released",
}


def _panel(data: MixedFrequencyData | NowcastResults) -> MixedFrequencyData:
    if isinstance(data, NowcastResults):
        if data.data is None:
            raise ValueError("Results do not carry their estimation data.")
        return data.data
    if isinstance(data, MixedFrequencyData):
        return data
    raise TypeError(f"Expected MixedFrequencyData or NowcastResults; got {type(data).__name__}.")


def data_availability(
    data: MixedFrequencyData | NowcastResults,
    *,
    n_periods: int | None = 24,
    series: list[str] | None = None,
) -> pd.DataFrame:
    """Classify every cell of the panel (see module docstring).

    Parameters
    ----------
    data : MixedFrequencyData or NowcastResults
        Panel, or results carrying their estimation data.
    n_periods : int or None, default 24
        Keep only the last ``n_periods`` base periods (``None``: all).
    series : list of str, optional
        Subset of series (default: all, in panel order).

    Returns
    -------
    pandas.DataFrame
        Integer codes (index of :data:`AVAILABILITY_STATES`), rows = series, columns =
        base periods.

    Raises
    ------
    ValueError
        Unknown series, ``n_periods < 1``, or results without data.
    TypeError
        Unsupported input type.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> df = pd.DataFrame(
    ...     {
    ...         "x": [1.0, np.nan, 1.0, 1.0, np.nan, np.nan],
    ...         "q": [np.nan, np.nan, 2.0, np.nan, np.nan, np.nan],
    ...     },
    ...     index=idx,
    ... )
    >>> mfd = MixedFrequencyData(df, {"x": "M", "q": "Q"})
    >>> data_availability(mfd).loc["x"].tolist()
    [1, 2, 1, 1, 3, 3]
    >>> data_availability(mfd).loc["q"].tolist()
    [0, 0, 1, 0, 0, 3]
    """
    panel = _panel(data)
    if series is not None:
        unknown = [s for s in series if s not in panel.columns]
        if unknown:
            raise ValueError(f"Unknown series {unknown}.")
        panel = panel.select(series)
    if n_periods is not None and n_periods < 1:
        raise ValueError(f"n_periods must be >= 1 or None; got {n_periods}.")
    codes = np.zeros(panel.shape, dtype=int)
    codes[panel.observation_mask().to_numpy()] = 1
    codes[panel.missing_mask().to_numpy()] = 2
    codes[panel.ragged_edge_mask().to_numpy()] = 3
    frame = pd.DataFrame(codes, index=panel.index, columns=panel.columns).T
    if n_periods is not None:
        frame = frame.iloc[:, -n_periods:]
    return frame


def release_table(data: MixedFrequencyData | NowcastResults) -> pd.DataFrame:
    """Per-series release status: frequency, last observation, delay and backlog.

    Parameters
    ----------
    data : MixedFrequencyData or NowcastResults
        Panel, or results carrying their estimation data.

    Returns
    -------
    pandas.DataFrame
        Index = series; columns ``frequency``, ``category``, ``last_observed``
        (native period), ``release_delay`` (days, may be missing), ``n_observations``
        and ``pending`` (number of storage slots after the last observation).

    Raises
    ------
    ValueError
        Results without data.
    TypeError
        Unsupported input type.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> idx = pd.period_range("2020-01", periods=3, freq="M")
    >>> mfd = MixedFrequencyData(pd.DataFrame({"x": [1.0, 2.0, np.nan]}, index=idx), "M")
    >>> int(release_table(mfd).loc["x", "pending"])
    1
    """
    panel = _panel(data)
    last = panel.last_observed()
    native = []
    for col in panel.columns:
        period = last[col]
        freq = panel.metadata[col].frequency
        native.append("-" if pd.isna(period) else str(period.asfreq(freq.pandas_freq)))
    categories = [
        "-" if m.category is None else str(getattr(m.category, "value", m.category))
        for m in panel.metadata.values()
    ]
    return pd.DataFrame(
        {
            "frequency": [panel.metadata[c].frequency.label for c in panel.columns],
            "category": categories,
            "last_observed": native,
            "release_delay": panel.release_delays.to_numpy(),
            "n_observations": panel.n_observations().to_numpy(),
            "pending": panel.ragged_edge_mask().sum(axis=0).to_numpy(),
        },
        index=pd.Index(panel.columns, name="series"),
    )


def _state_colors(theme: Theme) -> list[str]:
    return ["#f0efec", theme.in_sample_color, theme.neutral_color, theme.out_of_sample_color]


def plot_data_availability(
    data: MixedFrequencyData | NowcastResults,
    *,
    n_periods: int | None = 24,
    series: list[str] | None = None,
    backend: str = "plotly",
    theme: Theme | str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Heatmap of data availability (ragged edge) over the last base periods.

    Parameters
    ----------
    data : MixedFrequencyData or NowcastResults
        Panel, or results carrying their estimation data.
    n_periods : int or None, default 24
        Number of most recent base periods shown (``None``: all).
    series : list of str, optional
        Subset of series.
    backend : {"plotly", "matplotlib"}, default "plotly"
        Plotting library.
    theme : Theme or str, optional
        Visual theme.
    title : str, optional
        Title (default ``"Data availability"``).
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
        See :func:`data_availability`; unknown backend.
    TypeError
        Unsupported input type.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> idx = pd.period_range("2020-01", periods=4, freq="M")
    >>> mfd = MixedFrequencyData(pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan]}, index=idx), "M")
    >>> fig = plot_data_availability(mfd)
    >>> fig.data[0].type
    'heatmap'
    """
    th, be = resolve(theme, backend, ax)
    codes = data_availability(data, n_periods=n_periods, series=series)
    title = "Data availability" if title is None else title
    colors = _state_colors(th)
    if be == "plotly":
        return _availability_plotly(codes, colors, th, title)
    return _availability_mpl(codes, colors, th, title, ax, figsize)


def _availability_plotly(codes: pd.DataFrame, colors: list[str], theme: Theme, title: str) -> Any:
    import plotly.graph_objects as go

    n = len(AVAILABILITY_STATES)
    scale: list[list[Any]] = []
    for i, color in enumerate(colors):
        scale += [[i / n, color], [(i + 1) / n, color]]
    text = np.vectorize(lambda c: _LABELS[AVAILABILITY_STATES[c]])(codes.to_numpy())
    fig = new_plotly_figure()
    fig.add_trace(
        go.Heatmap(
            z=codes.to_numpy(),
            x=[str(p) for p in codes.columns],
            y=[str(s) for s in codes.index],
            zmin=-0.5,
            zmax=n - 0.5,
            colorscale=scale,
            xgap=2,
            ygap=2,
            text=text,
            hovertemplate="%{y} %{x}: %{text}<extra></extra>",
            colorbar={
                "tickvals": list(range(n)),
                "ticktext": [_LABELS[s] for s in AVAILABILITY_STATES],
                "title": {"text": ""},
            },
        )
    )
    fig = finish_plotly(fig, theme, title=title, hovermode="closest", showlegend=False)
    fig.update_layout(height=max(theme.height, 22 * codes.shape[0] + 160))
    fig.update_xaxes(showgrid=False, type="category")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def _availability_mpl(
    codes: pd.DataFrame,
    colors: list[str],
    theme: Theme,
    title: str,
    ax: Axes | None,
    figsize: tuple[float, float] | None,
) -> Any:
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), cmap.N)
    size = figsize or (
        max(6.0, 0.32 * codes.shape[1] + 3.0),
        max(2.5, 0.3 * codes.shape[0] + 1.6),
    )
    with mpl_context(theme):
        fig, axes = new_axes(ax, theme, size)
        axes.imshow(codes.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
        step = max(1, codes.shape[1] // 12)
        positions = list(range(0, codes.shape[1], step))
        axes.set_xticks(positions, [str(codes.columns[i]) for i in positions], rotation=45)
        axes.set_yticks(range(codes.shape[0]), [str(s) for s in codes.index])
        axes.grid(False)
        handles = [
            Patch(facecolor=c, edgecolor=theme.grid_color, label=_LABELS[s])
            for c, s in zip(colors, AVAILABILITY_STATES, strict=True)
        ]
        axes.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        finish_mpl(axes, theme, title=title, legend=False)
    return fig


# ---------------------------------------------------------------------- released share
def _results_period(results: NowcastResults) -> pd.Period:
    """First target period after the last observation of the target (current nowcast)."""
    estimate = results.estimate
    last = results.nowcast["observed"].last_valid_index()
    later = estimate.index if last is None else estimate.index[estimate.index > last]
    if len(later) == 0:
        raise ValueError("No target period after the last observation; pass period=.")
    return later[0]


def released_share_table(
    data: MixedFrequencyData | NowcastResults,
    period: pd.Period | str | None = None,
    *,
    by: str | Mapping[str, str | Sequence[str]] | None = "category",
    weights: Mapping[str, float] | pd.Series | None = None,
    as_of: pd.Timestamp | str | None = None,
    series: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Share of the observations of a target period already released, by group.

    Thin wrapper of :meth:`~nowcastbox.core.data.MixedFrequencyData.released_share` that
    also accepts fitted results: then the panel is the estimation data, the target is
    left out (unless ``series`` lists it) and ``period`` defaults to the current nowcast
    period (the first target period after its last observation).

    Parameters
    ----------
    data : MixedFrequencyData or NowcastResults
        Panel (a vintage), or results carrying their estimation data.
    period : pandas.Period or str, optional
        Target period (required for a panel).
    by : {"series", "category", "block", "frequency"}, mapping or None, default "category"
        Grouping of the rows.
    weights : mapping or pandas.Series, optional
        Non-negative weights of the series.
    as_of : Timestamp or str, optional
        Information date (applies ``as_of`` to the panel first).
    series : sequence of str, optional
        Subset of series.

    Returns
    -------
    pandas.DataFrame
        Columns ``released``, ``expected``, ``weight``, ``share``; last row ``"total"``.

    Raises
    ------
    ValueError
        Missing ``period`` for a panel, results without data or without a period after
        the last observation; see ``released_share``.
    TypeError
        Unsupported input type.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan]}, index=idx)
    >>> table = released_share_table(MixedFrequencyData(df, "M"), "2020Q2", by=None)
    >>> round(float(table.loc["total", "share"]), 4)
    0.3333
    """
    panel = _panel(data)
    if isinstance(data, NowcastResults):
        period = _results_period(data) if period is None else period
        if series is None:
            series = [c for c in panel.columns if c != data.target]
    if period is None:
        raise ValueError("period is required for a MixedFrequencyData panel.")
    return panel.released_share(period, by=by, weights=weights, series=series, as_of=as_of)


def plot_released_share(
    data: MixedFrequencyData | NowcastResults,
    period: pd.Period | str | None = None,
    *,
    by: str | Mapping[str, str | Sequence[str]] | None = "category",
    weights: Mapping[str, float] | pd.Series | None = None,
    as_of: pd.Timestamp | str | None = None,
    series: Sequence[str] | None = None,
    backend: str = "plotly",
    theme: Theme | str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Bar chart of the share of a target period's data already released, by group.

    Parameters
    ----------
    data : MixedFrequencyData or NowcastResults
        Panel (a vintage), or results carrying their estimation data.
    period : pandas.Period or str, optional
        Target period (default for results: the current nowcast period).
    by : {"series", "category", "block", "frequency"}, mapping or None, default "category"
        Grouping of the bars.
    weights : mapping or pandas.Series, optional
        Non-negative weights of the series.
    as_of : Timestamp or str, optional
        Information date.
    series : sequence of str, optional
        Subset of series.
    backend : {"plotly", "matplotlib"}, default "plotly"
        Plotting library.
    theme : Theme or str, optional
        Visual theme.
    title : str, optional
        Title (default: ``"Data released for <period>: <total>%"``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (Matplotlib only).
    figsize : tuple of float, optional
        Matplotlib figure size.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        The figure (one bar per group, in percent; the total as a dashed line).

    Raises
    ------
    ValueError
        See :func:`released_share_table`; unknown backend.
    TypeError
        Unsupported input type.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan]}, index=idx)
    >>> fig = plot_released_share(MixedFrequencyData(df, "M"), "2020Q2", by=None)
    >>> fig.data[0].type
    'bar'
    """
    th, be = resolve(theme, backend, ax)
    if period is None and isinstance(data, NowcastResults):
        period = _results_period(data)
    table = released_share_table(data, period, by=by, weights=weights, as_of=as_of, series=series)
    total = float(table["share"].to_numpy(dtype=float)[-1])
    bars = table.drop(index="total")
    if title is None:
        title = f"Data released for {period}: {100 * total:.0f}%"
    if be == "plotly":
        return _released_plotly(bars, total, th, title)
    return _released_mpl(bars, total, th, title, ax, figsize)


def _released_plotly(bars: pd.DataFrame, total: float, theme: Theme, title: str) -> Any:
    import plotly.graph_objects as go

    labels = [f"{r}/{e}" for r, e in zip(bars["released"], bars["expected"], strict=True)]
    fig = new_plotly_figure()
    fig.add_trace(
        go.Bar(
            x=100 * bars["share"].to_numpy(dtype=float),
            y=[str(i) for i in bars.index],
            orientation="h",
            marker_color=theme.color(0),
            text=labels,
            textposition="auto",
            name="Released",
            hovertemplate="%{y}: %{x:.0f}% (%{text})<extra></extra>",
        )
    )
    if np.isfinite(total):
        fig.add_vline(x=100 * total, line_dash="dash", line_color=theme.neutral_color)
    fig = finish_plotly(fig, theme, title=title, xlabel="% released", showlegend=False)
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(autorange="reversed")
    return fig


def _released_mpl(
    bars: pd.DataFrame,
    total: float,
    theme: Theme,
    title: str,
    ax: Axes | None,
    figsize: tuple[float, float] | None,
) -> Any:
    size = figsize or (7.0, max(2.5, 0.4 * len(bars) + 1.5))
    with mpl_context(theme):
        fig, axes = new_axes(ax, theme, size)
        positions = np.arange(len(bars))
        axes.barh(positions, 100 * bars["share"].to_numpy(dtype=float), color=theme.color(0))
        axes.set_yticks(positions, [str(i) for i in bars.index])
        axes.invert_yaxis()
        axes.set_xlim(0, 100)
        if np.isfinite(total):
            axes.axvline(100 * total, linestyle="--", color=theme.neutral_color)
        finish_mpl(axes, theme, title=title, xlabel="% released", legend=False)
    return fig


register_plot("released_share")(plot_released_share)

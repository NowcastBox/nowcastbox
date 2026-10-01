r"""News decomposition charts: waterfall of a nowcast revision and the nowcast tracker.

These functions work on **plain DataFrames** with documented columns, so they do not
depend on the internals of :mod:`nowcastbox.news` (plan innovation I6). Following
Bańbura & Modugno (2014), the revision of the nowcast between two vintages is
decomposed as

.. math::

    \mathbb{E}[y_t \mid \Omega_{v+1}] - \mathbb{E}[y_t \mid \Omega_v]
    = \sum_{j} b_j \bigl(x_j - \mathbb{E}[x_j \mid \Omega_v]\bigr)
    \; (+ \text{revision and re-estimation effects}),

and each term :math:`b_j (x_j - \mathbb{E}[x_j \mid \Omega_v])` is an *impact*.

News table (input of :func:`plot_news_waterfall`)
-------------------------------------------------
One row per release (or per series). Columns:

``impact`` (required)
    Contribution to the nowcast revision, in target units.
``series`` (optional; else the index is used)
    Label of the release.
``category``, ``block``, ``release_type`` (optional)
    Grouping keys (``release_type`` is e.g. ``"news"``/``"revision"``) usable with
    ``group_by``.
``actual``, ``expected``, ``news``, ``weight`` (optional)
    Released value, its model forecast, the surprise and the weight :math:`b_j`;
    shown in the hover text when present.

Objects with a ``releases`` or ``impacts`` DataFrame attribute (or a ``to_frame()``
method returning one with an ``impact`` column) and optional ``old_nowcast``,
``new_nowcast``, ``target_period``, ``revisions_effect``, ``removal_effect`` and
``reestimation_effect`` attributes are accepted too, so
:class:`nowcastbox.news.NewsResults` can be passed directly; the non-zero effects are
shown as separate bars.

Tracker table (input of :func:`plot_nowcast_tracker`)
-----------------------------------------------------
Wide format: one row per vintage (index sorted in time), a ``nowcast`` column with the
nowcast level after the vintage, and one column per group (e.g. ``hard``, ``soft``,
``financial``) with the impact of that group's releases at that vintage.
Long format: columns ``vintage``, ``impact``, a group column (default ``category``) and
optionally ``nowcast`` (constant within a vintage); it is pivoted to wide format.
Objects with ``contributions`` and ``path`` DataFrames (``path["nowcast"]``), such as
:class:`nowcastbox.news.NowcastTracker`, are accepted too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
    to_plot_index,
)
from nowcastbox.visualization.themes import Theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "NEWS_HOVER_COLUMNS",
    "news_waterfall_table",
    "plot_news_waterfall",
    "plot_nowcast_tracker",
    "tracker_table",
]

NEWS_HOVER_COLUMNS: tuple[str, ...] = ("actual", "expected", "news", "weight")
"""Optional numeric columns of the news table shown in hover texts."""

_OTHER = "Other"
_RESIDUAL = "Other effects"


# ---------------------------------------------------------------------- inputs
_EFFECT_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("revisions_effect", "Data revisions"),
    ("removal_effect", "Removed data"),
    ("reestimation_effect", "Re-estimation"),
)


@dataclass(frozen=True)
class _NewsInput:
    table: pd.DataFrame
    old: float | None = None
    new: float | None = None
    period: Any = None
    effects: dict[str, float] = field(default_factory=dict)


def _news_object_table(news: Any) -> pd.DataFrame:
    for attribute in ("releases", "impacts"):
        value = getattr(news, attribute, None)
        if isinstance(value, pd.DataFrame):
            return value.copy()
    if callable(getattr(news, "to_frame", None)):
        value = news.to_frame()
        if isinstance(value, pd.DataFrame):
            return value.copy()
    raise TypeError(
        "news must be a DataFrame, a Series of impacts or an object with a 'releases' or "
        f"'impacts' DataFrame; got {type(news).__name__}."
    )


def _news_input(news: Any) -> _NewsInput:
    """Read a news table, Series or news-results object."""
    if isinstance(news, pd.Series):
        table = news.rename("impact").to_frame()
        parsed = _NewsInput(table)
    elif isinstance(news, pd.DataFrame):
        parsed = _NewsInput(news.copy())
    else:
        effects = {}
        for attribute, label in _EFFECT_ATTRIBUTES:
            value = getattr(news, attribute, None)
            if value is not None and float(value) != 0.0:
                effects[label] = float(value)
        old = getattr(news, "old_nowcast", None)
        new = getattr(news, "new_nowcast", None)
        parsed = _NewsInput(
            _news_object_table(news),
            None if old is None else float(old),
            None if new is None else float(new),
            getattr(news, "target_period", None),
            effects,
        )
    if "impact" not in parsed.table.columns:
        raise ValueError("The news table needs an 'impact' column.")
    return parsed


def _impact_rows(table: pd.DataFrame, group_by: str | None, top_n: int | None) -> pd.DataFrame:
    """Impact bars (largest absolute first), grouped and truncated as requested."""
    if table.empty:
        raise ValueError("The news table is empty.")
    labels = table["series"].astype(str) if "series" in table.columns else table.index.astype(str)
    table = table.assign(label=labels.to_numpy(), impact=table["impact"].astype(float))
    if group_by is not None:
        if group_by not in table.columns:
            raise ValueError(f"group_by={group_by!r} is not a column of the news table.")
        grouped = table.groupby(table[group_by].astype(str), sort=False)["impact"].sum()
        table = pd.DataFrame({"label": grouped.index, "impact": grouped.to_numpy()})
    keep = ["label", "impact"] + [c for c in NEWS_HOVER_COLUMNS if c in table.columns]
    table = table[keep]
    table = table.iloc[np.argsort(-np.abs(table["impact"].to_numpy()), kind="stable")]
    if top_n is not None:
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1; got {top_n}.")
        if len(table) > top_n:
            rest = float(table["impact"].iloc[top_n:].sum())
            other = pd.DataFrame({"label": [_OTHER], "impact": [rest]})
            table = pd.concat([table.iloc[:top_n], other], ignore_index=True)
    return table.rename(columns={"impact": "value"}).assign(measure="relative")


def news_waterfall_table(
    news: Any,
    *,
    group_by: str | None = None,
    top_n: int | None = None,
    old_nowcast: float | None = None,
    new_nowcast: float | None = None,
) -> pd.DataFrame:
    """Bars of a news waterfall: start, impacts (largest first), residual and end.

    Parameters
    ----------
    news : pandas.DataFrame, pandas.Series or news-results object
        News table (see module docstring).
    group_by : str, optional
        Aggregate impacts by this column (e.g. ``"category"``, ``"block"``).
    top_n : int, optional
        Keep the ``top_n`` largest absolute impacts and sum the rest into ``"Other"``.
    old_nowcast, new_nowcast : float, optional
        Nowcast before and after the new information (override the attributes of a
        news-results object). When both are given, ``new - old - sum(impacts)`` is
        shown as ``"Other effects"`` (data revisions, re-estimation).

    Returns
    -------
    pandas.DataFrame
        Columns ``label``, ``value``, ``measure`` (``"absolute"``, ``"relative"`` or
        ``"total"``), ``start`` and ``end`` (running level), plus the hover columns
        present in the input (relative rows only, ungrouped tables).

    Raises
    ------
    ValueError
        Missing ``impact``/``group_by`` column, empty table or ``top_n < 1``.
    TypeError
        Unsupported input type.

    Examples
    --------
    >>> import pandas as pd
    >>> news = pd.DataFrame({"series": ["ip", "pmi"], "impact": [0.2, -0.05]})
    >>> t = news_waterfall_table(news, old_nowcast=1.0, new_nowcast=1.2)
    >>> t["label"].tolist()
    ['Previous nowcast', 'ip', 'pmi', 'Other effects', 'New nowcast']
    >>> round(float(t["end"].iloc[-1]), 6)
    1.2
    """
    parsed = _news_input(news)
    old = parsed.old if old_nowcast is None else old_nowcast
    new = parsed.new if new_nowcast is None else new_nowcast
    rows = _impact_rows(parsed.table, group_by, top_n)
    if parsed.effects:
        effects = pd.DataFrame(
            {"label": list(parsed.effects), "value": list(parsed.effects.values())}
        ).assign(measure="relative")
        rows = pd.concat([rows, effects], ignore_index=True)
    start_level = 0.0 if old is None else float(old)
    parts = []
    if old is not None:
        parts.append(
            pd.DataFrame(
                {"label": ["Previous nowcast"], "value": [start_level], "measure": ["absolute"]}
            )
        )
    parts.append(rows)
    total = start_level + float(rows["value"].sum())
    if old is not None and new is not None and not np.isclose(float(new), total, atol=1e-12):
        parts.append(
            pd.DataFrame(
                {"label": [_RESIDUAL], "value": [float(new) - total], "measure": ["relative"]}
            )
        )
        total = float(new)
    end_label = "New nowcast" if old is not None else "Total revision"
    parts.append(pd.DataFrame({"label": [end_label], "value": [total], "measure": ["total"]}))
    out = pd.concat(parts, ignore_index=True)
    rel = np.where(out["measure"] == "relative", out["value"].to_numpy(dtype=float), 0.0)
    level = np.where(out["measure"] == "relative", np.nan, out["value"].to_numpy(dtype=float))
    end = pd.Series(level).fillna(pd.Series(np.cumsum(rel) + start_level)).to_numpy()
    begin = np.where(out["measure"] == "relative", end - rel, 0.0)
    return out.assign(start=begin, end=end)


# ---------------------------------------------------------------------- waterfall
def _waterfall_range(bars: pd.DataFrame) -> tuple[float, float] | None:
    """Y-range focused on the running levels when the chart starts from a level.

    Level bars (previous/new nowcast) start at zero; starting the axis at zero would
    flatten the impacts, so the range covers the running levels only (the level bars
    are then cut at the bottom of the axis, as is usual in waterfall charts).
    """
    if not (bars["measure"] == "absolute").any():
        return None
    points = np.concatenate(
        [
            bars["start"].to_numpy(dtype=float)[bars["measure"] == "relative"],
            bars["end"].to_numpy(dtype=float),
        ]
    )
    lo, hi = float(points.min()), float(points.max())
    pad = 0.15 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.1
    return lo - pad, hi + pad


def plot_news_waterfall(
    news: Any,
    *,
    group_by: str | None = None,
    top_n: int | None = 15,
    old_nowcast: float | None = None,
    new_nowcast: float | None = None,
    backend: str = "plotly",
    theme: Theme | str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Waterfall chart of the impacts of new releases on the nowcast.

    Parameters
    ----------
    news : pandas.DataFrame, pandas.Series or news-results object
        News table (see module docstring).
    group_by : str, optional
        Aggregate impacts by this column (``"category"``, ``"block"``,
        ``"release_type"``...).
    top_n : int or None, default 15
        Show the largest ``top_n`` impacts; the rest are summed into ``"Other"``.
    old_nowcast, new_nowcast : float, optional
        Levels before/after the update (see :func:`news_waterfall_table`).
    backend : {"plotly", "matplotlib"}, default "plotly"
        Plotting library.
    theme : Theme or str, optional
        Visual theme (positive impacts use ``positive_color``, negative ones
        ``negative_color``, levels ``neutral_color``).
    title : str, optional
        Title (default ``"News decomposition"``, with the target period if known).
    ylabel : str, optional
        Y-axis label.
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
        See :func:`news_waterfall_table`; unknown backend.
    TypeError
        Unsupported input type.

    Examples
    --------
    >>> import pandas as pd
    >>> news = pd.DataFrame(
    ...     {"impact": [0.10, -0.04, 0.02], "category": ["hard", "soft", "hard"]},
    ...     index=["ip", "pmi", "retail"],
    ... )
    >>> fig = plot_news_waterfall(news, group_by="category", old_nowcast=1.5)
    >>> list(fig.data[0].x)
    ['Previous nowcast', 'hard', 'soft', 'New nowcast']
    """
    th, be = resolve(theme, backend, ax)
    bars = news_waterfall_table(
        news, group_by=group_by, top_n=top_n, old_nowcast=old_nowcast, new_nowcast=new_nowcast
    )
    if title is None:
        period = _news_input(news).period
        title = "News decomposition" + ("" if period is None else f" - {period}")
    if be == "plotly":
        return _waterfall_plotly(bars, th, title, ylabel)
    return _waterfall_mpl(bars, th, title, ylabel, ax, figsize)


def _waterfall_plotly(bars: pd.DataFrame, theme: Theme, title: str, ylabel: str | None) -> Any:
    import plotly.graph_objects as go

    fig = new_plotly_figure()
    fig.add_trace(
        go.Waterfall(
            x=bars["label"].tolist(),
            y=bars["value"].tolist(),
            measure=bars["measure"].tolist(),
            increasing={"marker": {"color": theme.positive_color}},
            decreasing={"marker": {"color": theme.negative_color}},
            totals={"marker": {"color": theme.neutral_color}},
            connector={"line": {"color": theme.grid_color, "width": 1}},
            customdata=bars["end"].to_numpy(),
            hovertemplate="%{x}<br>Impact: %{y:+.3f}<br>Level: %{customdata:.3f}<extra></extra>",
            name="News",
        )
    )
    fig = finish_plotly(
        fig, theme, title=title, ylabel=ylabel, hovermode="closest", showlegend=False
    )
    fig.update_xaxes(type="category")
    y_range = _waterfall_range(bars)
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    return fig


def _waterfall_mpl(
    bars: pd.DataFrame,
    theme: Theme,
    title: str,
    ylabel: str | None,
    ax: Axes | None,
    figsize: tuple[float, float] | None,
) -> Any:
    colors = []
    for measure, value in zip(bars["measure"], bars["value"], strict=True):
        if measure != "relative":
            colors.append(theme.neutral_color)
        else:
            colors.append(theme.positive_color if value >= 0 else theme.negative_color)
    heights = (bars["end"] - bars["start"]).to_numpy(dtype=float)
    size = figsize or (max(6.0, 0.6 * len(bars) + 2.0), theme.figsize[1])
    with mpl_context(theme):
        fig, axes = new_axes(ax, theme, size)
        positions = np.arange(len(bars))
        axes.bar(
            positions, heights, bottom=bars["start"].to_numpy(dtype=float), color=colors, width=0.7
        )
        ends = bars["end"].to_numpy(dtype=float)
        for i in range(len(bars) - 1):
            axes.plot([i + 0.35, i + 0.65], [ends[i], ends[i]], color=theme.grid_color, linewidth=1)
        axes.set_xticks(positions, bars["label"].tolist(), rotation=45, ha="right")
        y_range = _waterfall_range(bars)
        if y_range is None:
            axes.axhline(0.0, color=theme.muted_text_color, linewidth=0.6)
        else:
            axes.set_ylim(*y_range)
        finish_mpl(axes, theme, title=title, ylabel=ylabel, legend=False)
    return fig


# ---------------------------------------------------------------------- tracker
def tracker_table(
    tracker: Any,
    *,
    nowcast_column: str = "nowcast",
    vintage_column: str = "vintage",
    group_column: str = "category",
    impact_column: str = "impact",
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Normalise a tracker table to ``(contributions, nowcast)`` in wide format.

    Parameters
    ----------
    tracker : pandas.DataFrame or tracker object
        Wide or long tracker table (see module docstring).
    nowcast_column, vintage_column, group_column, impact_column : str
        Column names.

    Returns
    -------
    contributions : pandas.DataFrame
        Rows = vintages, columns = groups (missing combinations are 0).
    nowcast : pandas.Series or None
        Nowcast level after each vintage, if available.

    Raises
    ------
    ValueError
        Empty table, missing group column in long format, or no contribution columns.
    TypeError
        ``tracker`` is neither a DataFrame nor a tracker object.

    Examples
    --------
    >>> import pandas as pd
    >>> long = pd.DataFrame(
    ...     {
    ...         "vintage": ["2024-05-01", "2024-05-01", "2024-06-01"],
    ...         "category": ["hard", "soft", "hard"],
    ...         "impact": [0.1, -0.2, 0.05],
    ...         "nowcast": [1.0, 1.0, 1.05],
    ...     }
    ... )
    >>> contrib, level = tracker_table(long)
    >>> float(contrib.loc["2024-06-01", "soft"]), float(level.iloc[-1])
    (0.0, 1.05)
    """
    contributions = getattr(tracker, "contributions", None)
    path = getattr(tracker, "path", None)
    if isinstance(contributions, pd.DataFrame) and isinstance(path, pd.DataFrame):
        tracker = contributions.join(path[[nowcast_column]], how="left")
    if not isinstance(tracker, pd.DataFrame):
        raise TypeError(f"tracker must be a DataFrame; got {type(tracker).__name__}.")
    if tracker.empty:
        raise ValueError("The tracker table is empty.")
    if vintage_column in tracker.columns and impact_column in tracker.columns:
        if group_column not in tracker.columns:
            raise ValueError(f"Long tracker table needs a {group_column!r} column.")
        contrib = tracker.pivot_table(
            index=vintage_column,
            columns=group_column,
            values=impact_column,
            aggfunc="sum",
            fill_value=0.0,
            sort=False,
        )
        contrib.columns = pd.Index([str(c) for c in contrib.columns])
        contrib.index.name = None
        level = None
        if nowcast_column in tracker.columns:
            level = tracker.groupby(vintage_column, sort=False)[nowcast_column].last()
            level = level.reindex(contrib.index).astype(float)
            level.index.name = None
        return contrib.astype(float), level
    level = tracker[nowcast_column].astype(float) if nowcast_column in tracker.columns else None
    contrib = tracker.drop(columns=[nowcast_column], errors="ignore").select_dtypes("number")
    if contrib.shape[1] == 0:
        raise ValueError("The tracker table has no numeric contribution columns.")
    return contrib.astype(float).fillna(0.0), level


def plot_nowcast_tracker(
    tracker: Any,
    *,
    nowcast_column: str = "nowcast",
    vintage_column: str = "vintage",
    group_column: str = "category",
    impact_column: str = "impact",
    actual: float | None = None,
    backend: str = "plotly",
    theme: Theme | str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Nowcast tracker: nowcast level by vintage over stacked contributions by group.

    Each vintage shows the impact of the releases of each group (e.g. hard, soft,
    financial data) as stacked bars (positive above zero, negative below) and the
    resulting nowcast as a line, the "nowcast tracker" popularised by the New York
    Fed Staff Nowcast (Bok et al., 2018).

    Parameters
    ----------
    tracker : pandas.DataFrame or tracker object
        Wide or long tracker table (see module docstring).
    nowcast_column, vintage_column, group_column, impact_column : str
        Column names.
    actual : float, optional
        Realised value of the target, drawn as a horizontal reference line.
    backend : {"plotly", "matplotlib"}, default "plotly"
        Plotting library.
    theme : Theme or str, optional
        Visual theme.
    title : str, optional
        Title (default ``"Nowcast tracker"``).
    ylabel : str, optional
        Y-axis label.
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
        See :func:`tracker_table`; unknown backend.
    TypeError
        ``tracker`` is neither a DataFrame nor a tracker object.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.period_range("2024-04", periods=3, freq="M")
    >>> wide = pd.DataFrame(
    ...     {"hard": [0.1, 0.0, -0.1], "soft": [0.05, 0.2, 0.0], "nowcast": [1.0, 1.2, 1.1]},
    ...     index=idx,
    ... )
    >>> fig = plot_nowcast_tracker(wide, actual=1.15)
    >>> [t.name for t in fig.data]
    ['hard', 'soft', 'Nowcast']
    """
    th, be = resolve(theme, backend, ax)
    contrib, level = tracker_table(
        tracker,
        nowcast_column=nowcast_column,
        vintage_column=vintage_column,
        group_column=group_column,
        impact_column=impact_column,
    )
    title = "Nowcast tracker" if title is None else title
    if be == "plotly":
        return _tracker_plotly(contrib, level, actual, th, title, ylabel)
    return _tracker_mpl(contrib, level, actual, th, title, ylabel, ax, figsize)


def _x_values(index: pd.Index) -> Any:
    if isinstance(index, (pd.PeriodIndex, pd.DatetimeIndex)):
        return to_plot_index(index)
    return [str(v) for v in index]


def _tracker_plotly(
    contrib: pd.DataFrame,
    level: pd.Series | None,
    actual: float | None,
    theme: Theme,
    title: str,
    ylabel: str | None,
) -> Any:
    import plotly.graph_objects as go

    fig = new_plotly_figure()
    x = _x_values(contrib.index)
    for i, col in enumerate(contrib.columns):
        fig.add_trace(
            go.Bar(
                x=x,
                y=contrib[col],
                name=str(col),
                marker={
                    "color": theme.color(i),
                    "line": {"color": theme.background_color, "width": 1},
                },
                hovertemplate=f"{col}: %{{y:+.3f}}<extra></extra>",
            )
        )
    if level is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=level,
                name="Nowcast",
                mode="lines+markers",
                line={"color": theme.observed_color, "width": theme.line_width},
                marker={
                    "size": theme.marker_size,
                    "line": {"color": theme.background_color, "width": 2},
                },
                hovertemplate="Nowcast: %{y:.3f}<extra></extra>",
            )
        )
    if actual is not None:
        fig.add_hline(
            y=actual,
            line={"color": theme.muted_text_color, "dash": "dash", "width": 1},
            annotation_text="Actual",
            annotation_position="top left",
        )
    fig.update_layout(barmode="relative")
    return finish_plotly(fig, theme, title=title, ylabel=ylabel)


def _tracker_mpl(
    contrib: pd.DataFrame,
    level: pd.Series | None,
    actual: float | None,
    theme: Theme,
    title: str,
    ylabel: str | None,
    ax: Axes | None,
    figsize: tuple[float, float] | None,
) -> Any:
    positions = np.arange(len(contrib))
    labels = [str(v) for v in contrib.index]
    with mpl_context(theme):
        fig, axes = new_axes(ax, theme, figsize)
        pos_bottom = np.zeros(len(contrib))
        neg_bottom = np.zeros(len(contrib))
        for i, col in enumerate(contrib.columns):
            values = as_float_array(contrib[col])
            bottom = np.where(values >= 0, pos_bottom, neg_bottom)
            axes.bar(
                positions,
                values,
                bottom=bottom,
                color=theme.color(i),
                width=0.7,
                edgecolor=theme.background_color,
                linewidth=1,
                label=str(col),
            )
            pos_bottom += np.clip(values, 0.0, None)
            neg_bottom += np.clip(values, None, 0.0)
        if level is not None:
            axes.plot(
                positions,
                as_float_array(level),
                color=theme.observed_color,
                marker="o",
                markersize=theme.marker_size / 1.5,
                label="Nowcast",
            )
        if actual is not None:
            axes.axhline(
                actual, color=theme.muted_text_color, linestyle="--", linewidth=1, label="Actual"
            )
        axes.axhline(0.0, color=theme.muted_text_color, linewidth=0.6)
        step = max(1, len(labels) // 12)
        axes.set_xticks(positions[::step], labels[::step], rotation=45, ha="right")
        finish_mpl(axes, theme, title=title, ylabel=ylabel)
    return fig

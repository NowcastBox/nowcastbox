"""Nowcast path and density (fan) charts.

* :func:`plot_forecast` - observed target, in-sample fit and out-of-sample
  (backcast / nowcast / forecast) estimates, with prediction intervals when the
  ``nowcast`` frame carries ``lower_<p>``/``upper_<p>`` columns.
* :func:`plot_fan_chart` - fan chart of a predictive distribution given as a table of
  quantiles (density nowcasts, plan innovation I5).
* :func:`quantiles_from_nowcast` - Gaussian quantiles from the ``std`` column of a
  ``nowcast`` frame, to feed :func:`plot_fan_chart`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from nowcastbox.core.results import NOWCAST_COLUMNS, NowcastResults
from nowcastbox.visualization._common import (
    as_float_array,
    finish_mpl,
    finish_plotly,
    mpl_context,
    new_axes,
    new_plotly_figure,
    resolve,
    rgba,
    to_plot_index,
)
from nowcastbox.visualization.themes import Theme

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "interval_levels",
    "plot_empirical_bands",
    "plot_fan_chart",
    "plot_forecast",
    "quantiles_from_nowcast",
]

_INTERVAL = re.compile(r"^(lower|upper)_(\d+(?:\.\d+)?)$")
_LEVEL_PATTERNS = (
    re.compile(r"^q_?(\d+(?:\.\d+)?)$"),
    re.compile(r"^(\d+(?:\.\d+)?)%$"),
)


# ---------------------------------------------------------------------- inputs
def _nowcast_frame(results: NowcastResults | pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Return the nowcast frame and the target name (if known)."""
    if isinstance(results, NowcastResults):
        return results.nowcast.copy(), results.target
    if isinstance(results, pd.DataFrame):
        missing = [c for c in NOWCAST_COLUMNS if c not in results.columns]
        if missing:
            raise ValueError(f"nowcast frame is missing the columns {missing}.")
        return results.copy(), None
    raise TypeError(
        f"Expected NowcastResults or a nowcast DataFrame; got {type(results).__name__}."
    )


def interval_levels(frame: pd.DataFrame) -> list[str]:
    """Coverage labels with both ``lower_<p>`` and ``upper_<p>`` columns, widest first.

    Parameters
    ----------
    frame : pandas.DataFrame
        A ``nowcast`` frame.

    Returns
    -------
    list of str
        Labels such as ``["90", "68"]``.

    Examples
    --------
    >>> import pandas as pd
    >>> cols = ["lower_68", "upper_68", "lower_90", "upper_90", "lower_50"]
    >>> interval_levels(pd.DataFrame(columns=cols))
    ['90', '68']
    """
    found: dict[str, set[str]] = {}
    for col in frame.columns:
        match = _INTERVAL.match(str(col))
        if match:
            found.setdefault(match.group(2), set()).add(match.group(1))
    levels = [lvl for lvl, sides in found.items() if sides == {"lower", "upper"}]
    return sorted(levels, key=float, reverse=True)


def _slice(frame: pd.DataFrame | pd.Series, start: Any, end: Any) -> Any:
    if start is None and end is None:
        return frame
    return frame.loc[start:end]


def _runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """``[start, stop)`` bounds of the runs of consecutive True values."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(valid):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(valid)))
    return runs


def add_band(
    fig: Any,
    x: pd.Index,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    color: str,
    name: str,
    hover: bool = True,
) -> None:
    """Add a shaded band to a Plotly figure, one closed polygon per run without gaps.

    Filling between two traces (``fill="tonexty"``) would bridge missing periods, so each
    contiguous run is drawn as its own polygon; isolated periods (a polygon needs two
    points) are drawn as thick vertical error bars.
    """
    import plotly.graph_objects as go

    valid = np.isfinite(lower) & np.isfinite(upper)
    xs = np.asarray(x)
    runs = _runs(valid)
    single = [a for a, b in runs if b - a == 1]
    for k, (a, b) in enumerate(r for r in runs if r[1] - r[0] > 1):
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([xs[a:b], xs[a:b][::-1]]),
                y=np.concatenate([upper[a:b], lower[a:b][::-1]]),
                mode="lines",
                line={"width": 0},
                fill="toself",
                fillcolor=color,
                name=name,
                legendgroup=name,
                showlegend=k == 0,
                hoverinfo="y+name" if hover else "skip",
            )
        )
    if single:
        idx = np.asarray(single)
        half = (upper[idx] - lower[idx]) / 2.0
        fig.add_trace(
            go.Scatter(
                x=xs[idx],
                y=lower[idx] + half,
                mode="markers",
                marker={"size": 0, "opacity": 0, "color": color},
                error_y={
                    "type": "data",
                    "array": half,
                    "visible": True,
                    "width": 0,
                    "thickness": 8,
                    "color": color,
                },
                name=name,
                legendgroup=name,
                showlegend=len(single) == len(runs),
                hoverinfo="skip",
            )
        )


# ---------------------------------------------------------------------- forecast
def plot_forecast(
    results: NowcastResults | pd.DataFrame,
    *,
    backend: str = "plotly",
    theme: Theme | str | None = None,
    title: str | None = None,
    start: Any = None,
    end: Any = None,
    show_intervals: bool = True,
    ylabel: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Plot observed values, in-sample fit and out-of-sample nowcasts of the target.

    Parameters
    ----------
    results : NowcastResults or pandas.DataFrame
        Fitted results, or a ``nowcast`` frame with columns ``observed``,
        ``in_sample`` and ``out_of_sample`` on a PeriodIndex. Optional
        ``lower_<p>``/``upper_<p>`` columns (e.g. ``lower_90``) are drawn as shaded
        prediction intervals.
    backend : {"plotly", "matplotlib"}, default "plotly"
        Interactive (Plotly) or publication (Matplotlib) output.
    theme : Theme or str, optional
        Visual theme (default: current default theme).
    title : str, optional
        Title (default: ``"Nowcast of <target>"``).
    start, end : period-like, optional
        Restrict the plotted periods (e.g. ``"2015Q1"``).
    show_intervals : bool, default True
        Draw the prediction intervals when available.
    ylabel : str, optional
        Y-axis label.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (Matplotlib only).
    figsize : tuple of float, optional
        Matplotlib figure size in inches.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        The figure.

    Raises
    ------
    ValueError
        Unknown backend, missing columns, or no periods in ``[start, end]``.
    TypeError
        ``results`` of an unsupported type.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.results import build_nowcast_frame
    >>> idx = pd.period_range("2020Q1", periods=4, freq="Q")
    >>> frame = build_nowcast_frame(
    ...     pd.Series([1.0, 2.0, 1.5, np.nan], index=idx),
    ...     pd.Series([1.1, 1.8, 1.6, 1.2], index=idx),
    ... )
    >>> fig = plot_forecast(frame)
    >>> [trace.name for trace in fig.data]
    ['Observed', 'In-sample', 'Out-of-sample']
    """
    th, be = resolve(theme, backend, ax)
    frame, target = _nowcast_frame(results)
    frame = _slice(frame, start, end)
    if frame.empty:
        raise ValueError("No periods to plot in the requested range.")
    if title is None:
        title = f"Nowcast of {target}" if target else "Nowcast"
    levels = interval_levels(frame) if show_intervals else []
    if be == "plotly":
        return _forecast_plotly(frame, levels, th, title, ylabel)
    return _forecast_mpl(frame, levels, th, title, ylabel, ax, figsize)


def _band_alpha(position: int, n_bands: int) -> float:
    """Transparency of nested bands: widest is lightest."""
    return 0.12 + 0.16 * (position / max(n_bands - 1, 1)) if n_bands > 1 else 0.2


def _forecast_plotly(
    frame: pd.DataFrame, levels: list[str], theme: Theme, title: str, ylabel: str | None
) -> Any:
    import plotly.graph_objects as go

    fig = new_plotly_figure()
    x = to_plot_index(frame.index)
    labels = [str(p) for p in frame.index]
    for pos, lvl in enumerate(levels):
        lower = frame[f"lower_{lvl}"]
        upper = frame[f"upper_{lvl}"]
        add_band(
            fig,
            x,
            lower.to_numpy(dtype=float),
            upper.to_numpy(dtype=float),
            color=rgba(theme.interval_color, _band_alpha(pos, len(levels))),
            name=f"{lvl}% interval",
        )
    specs = (
        ("observed", "Observed", theme.observed_color, "lines+markers", "solid"),
        ("in_sample", "In-sample", theme.in_sample_color, "lines", "solid"),
        ("out_of_sample", "Out-of-sample", theme.out_of_sample_color, "lines+markers", "dot"),
    )
    for col, name, color, mode, dash in specs:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=frame[col],
                name=name,
                mode=mode,
                customdata=labels,
                line={"color": color, "width": theme.line_width, "dash": dash},
                marker={"size": theme.marker_size * (0.6 if col == "observed" else 1.0)},
                hovertemplate=f"%{{customdata}}<br>{name}: %{{y:.3f}}<extra></extra>",
            )
        )
    return finish_plotly(fig, theme, title=title, ylabel=ylabel)


def _forecast_mpl(
    frame: pd.DataFrame,
    levels: list[str],
    theme: Theme,
    title: str,
    ylabel: str | None,
    ax: Axes | None,
    figsize: tuple[float, float] | None,
) -> Any:
    x = to_plot_index(frame.index)
    with mpl_context(theme):
        fig, axes = new_axes(ax, theme, figsize)
        for pos, lvl in enumerate(levels):
            axes.fill_between(
                x,
                as_float_array(frame[f"lower_{lvl}"]),
                as_float_array(frame[f"upper_{lvl}"]),
                color=theme.interval_color,
                alpha=_band_alpha(pos, len(levels)),
                linewidth=0,
                label=f"{lvl}% interval",
            )
            # Vertical segments keep isolated interval periods visible.
            axes.vlines(
                x,
                as_float_array(frame[f"lower_{lvl}"]),
                as_float_array(frame[f"upper_{lvl}"]),
                colors=theme.interval_color,
                alpha=_band_alpha(pos, len(levels)),
                linewidth=6,
            )
        ms = theme.marker_size / 1.5
        axes.plot(
            x,
            as_float_array(frame["observed"]),
            color=theme.observed_color,
            marker="o",
            markersize=ms * 0.6,
            label="Observed",
        )
        axes.plot(
            x, as_float_array(frame["in_sample"]), color=theme.in_sample_color, label="In-sample"
        )
        axes.plot(
            x,
            as_float_array(frame["out_of_sample"]),
            color=theme.out_of_sample_color,
            linestyle=":",
            marker="D",
            markersize=ms,
            label="Out-of-sample",
        )
        finish_mpl(axes, theme, title=title, ylabel=ylabel, dates=True)
    return fig


# ---------------------------------------------------------------------- fan chart
def _parse_level(column: object) -> float:
    """Quantile level of a column label (``0.05``, ``"0.05"``, ``"q05"``, ``"5%"``)."""
    if isinstance(column, (int, float, np.floating, np.integer)) and not isinstance(column, bool):
        value = float(column)
    else:
        text = str(column).strip().lower()
        value = float("nan")
        for pattern in _LEVEL_PATTERNS:
            match = pattern.match(text)
            if match:
                digits = match.group(1)
                value = float(digits)
                if "%" in text or "." not in digits or value >= 1.0:
                    value /= 100.0
                break
        else:
            try:
                value = float(text)
            except ValueError:
                raise ValueError(f"Cannot read a quantile level from column {column!r}.") from None
    if not 0.0 < value < 1.0:
        raise ValueError(f"Quantile level of column {column!r} must lie in (0, 1); got {value}.")
    return value


FAN_LEVELS: tuple[float, ...] = (0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95)
"""Quantile levels drawn when the fan chart gets a predictive distribution object."""


def _is_distribution(obj: object) -> bool:
    """Duck type of :class:`nowcastbox.density.NowcastDistribution` (``quantiles(q)``)."""
    return not isinstance(obj, pd.DataFrame) and callable(getattr(obj, "quantiles", None))


def _quantile_table(quantiles: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(quantiles, pd.DataFrame):
        raise TypeError(f"quantiles must be a DataFrame; got {type(quantiles).__name__}.")
    if quantiles.empty:
        raise ValueError("quantiles is empty.")
    table = quantiles.copy()
    table.columns = pd.Index([_parse_level(c) for c in quantiles.columns])
    if table.columns.has_duplicates:
        raise ValueError("Duplicate quantile levels in the columns of quantiles.")
    return table.sort_index(axis=1).astype(float)


def _quantile_pairs(levels: Sequence[float]) -> list[tuple[float, float]]:
    """Symmetric ``(lower, upper)`` pairs, widest first."""
    pairs: list[tuple[float, float]] = []
    for low in levels:
        if low >= 0.5:
            continue
        match = [q for q in levels if abs(q - (1.0 - low)) < 1e-9]
        if not match:
            raise ValueError(f"Quantile {low:g} has no symmetric counterpart {1 - low:g}.")
        pairs.append((low, match[0]))
    if not pairs:
        raise ValueError("quantiles needs at least one symmetric pair of levels (q, 1 - q).")
    return sorted(pairs, key=lambda p: p[0])


def quantiles_from_nowcast(
    results: NowcastResults | pd.DataFrame,
    levels: Sequence[float] = (0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95),
) -> pd.DataFrame:
    """Gaussian predictive quantiles from the ``std`` column of a ``nowcast`` frame.

    For out-of-sample periods with a standard deviation, quantile ``q`` is
    ``out_of_sample + std * Phi^{-1}(q)``.

    Parameters
    ----------
    results : NowcastResults or pandas.DataFrame
        Results or ``nowcast`` frame with a ``std`` column.
    levels : sequence of float
        Quantile levels in (0, 1).

    Returns
    -------
    pandas.DataFrame
        Index = out-of-sample periods with ``std``; columns = ``levels``.

    Raises
    ------
    ValueError
        No ``std`` column, no period with both estimate and ``std``, or invalid levels.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.results import build_nowcast_frame
    >>> idx = pd.period_range("2020Q1", periods=2, freq="Q")
    >>> f = build_nowcast_frame(
    ...     pd.Series([1.0, np.nan], index=idx),
    ...     pd.Series([1.0, 2.0], index=idx),
    ...     extra={"std": pd.Series([np.nan, 0.5], index=idx)},
    ... )
    >>> round(float(quantiles_from_nowcast(f, [0.5, 0.975]).iloc[0, 1]), 2)
    2.98
    """
    frame, _ = _nowcast_frame(results)
    if "std" not in frame.columns:
        raise ValueError("The nowcast frame has no 'std' column.")
    lv = [_parse_level(q) for q in levels]
    rows = frame[frame["out_of_sample"].notna() & frame["std"].notna()]
    if rows.empty:
        raise ValueError("No out-of-sample period with a standard deviation.")
    mean = rows["out_of_sample"].to_numpy(dtype=float)[:, None]
    sd = rows["std"].to_numpy(dtype=float)[:, None]
    values = mean + sd * stats.norm.ppf(np.asarray(lv))[None, :]
    return pd.DataFrame(values, index=rows.index, columns=pd.Index(lv))


def plot_fan_chart(
    quantiles: pd.DataFrame | NowcastResults | Any,
    *,
    observed: pd.Series | None = None,
    backend: str = "plotly",
    theme: Theme | str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    history: int | None = 12,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Fan chart of a predictive distribution given by quantiles.

    Parameters
    ----------
    quantiles : pandas.DataFrame, NowcastResults or NowcastDistribution
        Index = target periods; columns = quantile levels as floats in (0, 1) or
        labels such as ``"0.05"``, ``"q05"``, ``"5%"``. Every level below 0.5 needs its
        symmetric counterpart (``q`` and ``1 - q``); the median (0.5), when present, is
        drawn as a line. A :class:`NowcastResults` with a ``std`` column is converted
        with :func:`quantiles_from_nowcast`, keeping the periods after the last
        observation (its observed values are added as history). A
        :class:`~nowcastbox.density.NowcastDistribution` (Gaussian or bootstrap mixture,
        innovation I5) is converted with its exact quantiles at :data:`FAN_LEVELS`,
        keeping the periods after the last value of ``observed`` when given.
    observed : pandas.Series, optional
        Realised values to draw as history (same kind of index).
    backend : {"plotly", "matplotlib"}, default "plotly"
        Plotting library.
    theme : Theme or str, optional
        Visual theme.
    title : str, optional
        Title (default ``"Predictive distribution"``).
    ylabel : str, optional
        Y-axis label.
    history : int or None, default 12
        Number of most recent observed values to show (``None``: all).
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
        Bad quantile labels, unpaired levels or unknown backend.
    TypeError
        ``quantiles`` of an unsupported type.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.period_range("2024Q3", periods=2, freq="Q")
    >>> q = pd.DataFrame({"q05": [0.0, -0.5], "q50": [1.0, 1.0], "q95": [2.0, 2.5]}, index=idx)
    >>> fig = plot_fan_chart(q)
    >>> fig.data[-1].name
    'Median'
    """
    th, be = resolve(theme, backend, ax)
    if isinstance(quantiles, NowcastResults):
        if observed is None:
            observed = quantiles.observed.dropna()
        last = quantiles.observed.last_valid_index()
        q_frame = quantiles_from_nowcast(quantiles)
        if last is not None and (q_frame.index > last).any():
            # Only the periods after the last observation (nowcast and forecasts).
            q_frame = q_frame[q_frame.index > last]
    elif _is_distribution(quantiles):
        q_frame = quantiles.quantiles(FAN_LEVELS)  # type: ignore[union-attr]
        last = None if observed is None else observed.last_valid_index()
        if last is not None and (q_frame.index > last).any():
            q_frame = q_frame[q_frame.index > last]
    else:
        q_frame = quantiles
    table = _quantile_table(q_frame)
    pairs = _quantile_pairs([float(c) for c in table.columns])
    if observed is not None:
        observed = observed.dropna()
        if history is not None:
            observed = observed.iloc[-history:] if history > 0 else observed.iloc[:0]
    title = "Predictive distribution" if title is None else title
    if be == "plotly":
        return _fan_plotly(table, pairs, observed, th, title, ylabel)
    return _fan_mpl(table, pairs, observed, th, title, ylabel, ax, figsize)


def _fan_alpha(position: int, n_pairs: int) -> float:
    return 0.15 + 0.35 * (position + 1) / n_pairs


def _fan_plotly(
    table: pd.DataFrame,
    pairs: list[tuple[float, float]],
    observed: pd.Series | None,
    theme: Theme,
    title: str,
    ylabel: str | None,
) -> Any:
    import plotly.graph_objects as go

    fig = new_plotly_figure()
    x = to_plot_index(table.index)
    for pos, (low, high) in enumerate(pairs):
        add_band(
            fig,
            x,
            table[low].to_numpy(dtype=float),
            table[high].to_numpy(dtype=float),
            color=rgba(theme.interval_color, _fan_alpha(pos, len(pairs))),
            name=f"{round(100 * (high - low))}% band",
        )
    if observed is not None and len(observed):
        fig.add_trace(
            go.Scatter(
                x=to_plot_index(observed.index),
                y=observed,
                name="Observed",
                mode="lines+markers",
                line={"color": theme.observed_color, "width": theme.line_width},
                marker={"size": theme.marker_size * 0.6},
            )
        )
    if 0.5 in table.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=table[0.5],
                name="Median",
                mode="lines+markers",
                line={"color": theme.out_of_sample_color, "width": theme.line_width},
                marker={"size": theme.marker_size},
            )
        )
    return finish_plotly(fig, theme, title=title, ylabel=ylabel)


def _fan_mpl(
    table: pd.DataFrame,
    pairs: list[tuple[float, float]],
    observed: pd.Series | None,
    theme: Theme,
    title: str,
    ylabel: str | None,
    ax: Axes | None,
    figsize: tuple[float, float] | None,
) -> Any:
    x = to_plot_index(table.index)
    with mpl_context(theme):
        fig, axes = new_axes(ax, theme, figsize)
        for pos, (low, high) in enumerate(pairs):
            axes.fill_between(
                x,
                as_float_array(table[low]),
                as_float_array(table[high]),
                color=theme.interval_color,
                alpha=_fan_alpha(pos, len(pairs)),
                linewidth=0,
                label=f"{round(100 * (high - low))}% band",
            )
        if observed is not None and len(observed):
            axes.plot(
                to_plot_index(observed.index),
                as_float_array(observed),
                color=theme.observed_color,
                marker="o",
                markersize=theme.marker_size / 2.5,
                label="Observed",
            )
        if 0.5 in table.columns:
            axes.plot(
                x,
                as_float_array(table[0.5]),
                color=theme.out_of_sample_color,
                marker="D",
                markersize=theme.marker_size / 1.5,
                label="Median",
            )
        finish_mpl(axes, theme, title=title, ylabel=ylabel, dates=True)
    return fig


def plot_empirical_bands(
    bands: Any,
    *,
    observed: pd.Series | None = None,
    backend: str = "plotly",
    theme: Theme | str | None = None,
    title: str | None = None,
) -> Any:
    """Fan chart of empirical error bands at their own levels (e.g. 57.5 %, 68 %, 90 %).

    Parameters
    ----------
    bands : EmpiricalGaussianDistribution or EmpiricalQuantileDistribution
        Output of :func:`~nowcastbox.density.empirical_bands` (any object with
        ``levels`` and ``quantiles(q)``).
    observed : pandas.Series, optional
        Realised values drawn as history.
    backend : {"plotly", "matplotlib"}, default "plotly"
        Plotting library.
    theme : Theme or str, optional
        Visual theme.
    title : str, optional
        Title (default ``"Empirical error bands"``).

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        The fan chart.

    Examples
    --------
    >>> from nowcastbox.density import EmpiricalQuantileDistribution
    >>> bands = EmpiricalQuantileDistribution(["2020Q1"], [0.0], [[-1.0, 0.5, 1.0]])
    >>> fig = plot_empirical_bands(bands)
    >>> len(fig.data)
    4
    """
    probs = {0.5}
    for level in bands.levels:
        probs.update({round((1 - level) / 2, 10), round((1 + level) / 2, 10)})
    quantiles = bands.quantiles(sorted(probs))
    return plot_fan_chart(
        quantiles,
        observed=observed,
        backend=backend,
        theme=theme,
        title=title or "Empirical error bands",
    )

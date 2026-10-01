"""Plot hooks of the news results (waterfall, tracker path, level contributions).

The news objects (:class:`~nowcastbox.news.NewsResults`,
:class:`~nowcastbox.news.NowcastTracker`, :class:`~nowcastbox.news.LevelContributions`)
are not :class:`~nowcastbox.core.results.NowcastResults`, so they have their own plot
registry with the same semantics as :func:`nowcastbox.core.results.register_plot`:
``obj.plot(kind)`` looks the kind up along the MRO of the object's class; when nothing
is registered it imports :mod:`nowcastbox.visualization` (which may register richer
plots, e.g. Plotly) and tries again. Matplotlib defaults are registered here.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["available_news_plots", "plot_news_object", "register_news_plot"]

PlotFunction = Callable[..., Any]

_REGISTRY: dict[tuple[type, str], PlotFunction] = {}


def register_news_plot(kind: str, results_type: type) -> Callable[[PlotFunction], PlotFunction]:
    """Register a plotting function for ``obj.plot(kind)`` on a news object.

    Parameters
    ----------
    kind : str
        Plot name (``"waterfall"``, ``"path"``, ``"contributions"``, ``"bar"``...).
    results_type : type
        Class the plot applies to (subclasses inherit it).

    Returns
    -------
    callable
        Decorator returning the function unchanged.

    Examples
    --------
    >>> from nowcastbox.news import NewsResults, register_news_plot
    >>> @register_news_plot("my_waterfall", NewsResults)
    ... def _plot(news, **kwargs):
    ...     return "figure"
    """

    def decorator(func: PlotFunction) -> PlotFunction:
        _REGISTRY[(results_type, kind)] = func
        return func

    return decorator


def _lookup(results_type: type, kind: str) -> PlotFunction | None:
    for klass in results_type.__mro__:
        func = _REGISTRY.get((klass, kind))
        if func is not None:
            return func
    return None


def available_news_plots(obj: object) -> list[str]:
    """Plot kinds registered for a news object (or class).

    Parameters
    ----------
    obj : object or type
        News results object or class.

    Returns
    -------
    list of str
        Sorted plot kinds.

    Examples
    --------
    >>> from nowcastbox.news import NewsResults, available_news_plots
    >>> "waterfall" in available_news_plots(NewsResults)
    True
    """
    klass = obj if isinstance(obj, type) else type(obj)
    mro = set(klass.__mro__)
    return sorted({kind for (k, kind) in _REGISTRY if k in mro})


def plot_news_object(obj: object, kind: str, **kwargs: Any) -> Any:
    """Dispatch ``obj.plot(kind, **kwargs)`` through the registry.

    Parameters
    ----------
    obj : object
        News results object.
    kind : str
        Plot kind.
    **kwargs
        Passed to the plotting function.

    Returns
    -------
    object
        Figure returned by the plotting function.

    Raises
    ------
    NotImplementedError
        If no plot of that kind is registered.

    Examples
    --------
    >>> news.plot("waterfall")  # doctest: +SKIP
    """
    func = _lookup(type(obj), kind)
    if func is None:
        with contextlib.suppress(ImportError):
            importlib.import_module("nowcastbox.visualization")
        func = _lookup(type(obj), kind)
    if func is None:
        raise NotImplementedError(
            f"No plot {kind!r} registered for {type(obj).__name__}; "
            f"available: {available_news_plots(obj)}."
        )
    return func(obj, **kwargs)


# ====================================================================== matplotlib defaults
def _axes(ax: Any, figsize: tuple[float, float]) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.figure, ax


def waterfall_chart(
    steps: pd.Series,
    *,
    ax: Any = None,
    title: str | None = None,
    figsize: tuple[float, float] = (9.0, 4.5),
    colors: tuple[str, str, str] = ("#4c72b0", "#55a868", "#c44e52"),
) -> Any:
    """Matplotlib waterfall chart of a decomposition.

    Parameters
    ----------
    steps : pandas.Series
        First and last entries are levels (start and end); the entries in between are
        increments.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (a new figure is created otherwise).
    title : str, optional
        Axes title.
    figsize : tuple of float, default (9, 4.5)
        Size of a new figure.
    colors : tuple of str
        Colours of levels, positive and negative increments.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Examples
    --------
    >>> import pandas as pd
    >>> fig = waterfall_chart(pd.Series([1.0, 0.2, -0.1, 1.1], index=list("abcd")))
    >>> len(fig.axes)
    1
    """
    fig, ax = _axes(ax, figsize)
    values = steps.to_numpy(dtype=float)
    n = len(values)
    bottoms = np.zeros(n)
    heights = values.copy()
    colour = [colors[0]] * n
    running = values[0]
    for i in range(1, n - 1):
        bottoms[i] = running if values[i] >= 0 else running + values[i]
        heights[i] = abs(values[i])
        colour[i] = colors[1] if values[i] >= 0 else colors[2]
        running += values[i]
    positions = np.arange(n)
    ax.bar(positions, heights, bottom=bottoms, color=colour, edgecolor="black", linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(i) for i in steps.index], rotation=30, ha="right")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def tracker_chart(
    path: pd.Series,
    contributions: pd.DataFrame,
    *,
    ax: Any = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 4.5),
) -> Any:
    """Nowcast path with stacked per-vintage contributions.

    Parameters
    ----------
    path : pandas.Series
        Nowcast at each vintage.
    contributions : pandas.DataFrame
        Change of the nowcast at each vintage by group (rows aligned with ``path``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    title : str, optional
        Axes title.
    figsize : tuple of float, default (10, 4.5)
        Size of a new figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2020-01-01", periods=3)
    >>> path = pd.Series([1.0, 1.2, 1.1], index=idx)
    >>> contrib = pd.DataFrame({"hard": [0.0, 0.2, -0.1]}, index=idx)
    >>> fig = tracker_chart(path, contrib)
    >>> len(fig.axes)
    1
    """
    fig, ax = _axes(ax, figsize)
    x = np.arange(len(path))
    pos = np.zeros(len(path))
    neg = np.zeros(len(path))
    for column in contributions.columns:
        vals = contributions[column].to_numpy(dtype=float)
        base = np.where(vals >= 0, pos, neg)
        ax.bar(x, vals, bottom=base, label=str(column), alpha=0.8)
        pos = pos + np.maximum(vals, 0.0)
        neg = neg + np.minimum(vals, 0.0)
    ax.plot(x, path.to_numpy(dtype=float), color="black", marker="o", label="nowcast")
    labels = [i.strftime("%Y-%m-%d") if isinstance(i, pd.Timestamp) else str(i) for i in path.index]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.legend(loc="best", fontsize="small")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def bar_chart(
    values: pd.Series,
    *,
    ax: Any = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 4.5),
) -> Any:
    """Horizontal bar chart of contributions.

    Parameters
    ----------
    values : pandas.Series
        Contributions indexed by label.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    title : str, optional
        Axes title.
    figsize : tuple of float, default (8, 4.5)
        Size of a new figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Examples
    --------
    >>> import pandas as pd
    >>> fig = bar_chart(pd.Series({"a": 0.3, "b": -0.1}))
    >>> len(fig.axes)
    1
    """
    fig, ax = _axes(ax, figsize)
    vals = values.to_numpy(dtype=float)
    colour = np.where(vals >= 0, "#55a868", "#c44e52")
    ax.barh([str(i) for i in values.index], vals, color=colour)
    ax.axvline(0.0, color="grey", linewidth=0.8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig

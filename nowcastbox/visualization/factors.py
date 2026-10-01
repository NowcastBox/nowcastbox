"""Factor-model charts: factors, scree plot (eigenvalues) and loadings.

All functions accept a fitted :class:`~nowcastbox.core.results.FactorResults`
(``TwoStepResults``, ``MixedFreqDFMResults``) or the corresponding plain pandas
objects, so they can also be used on factors estimated elsewhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.results import NowcastResults
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

__all__ = ["panel_eigenvalues", "plot_eigenvalues", "plot_factors", "plot_loadings"]


# ---------------------------------------------------------------------- factors
def _select_factors(frame: pd.DataFrame, factors: list[str] | None) -> pd.DataFrame:
    if factors is None:
        return frame
    unknown = [f for f in factors if f not in frame.columns]
    if unknown:
        raise ValueError(f"Unknown factors {unknown}; available: {list(frame.columns)}.")
    return frame[factors]


def _factor_frame(
    results: NowcastResults | pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Period | None]:
    """Factors and the last period of the estimation sample (for shading)."""
    if isinstance(results, NowcastResults):
        if results.factors is None:
            raise ValueError(f"{type(results).__name__} has no factors to plot.")
        end = results.data.end if results.data is not None else None
        return results.factors.copy(), end
    if isinstance(results, pd.DataFrame):
        return results.copy(), None
    raise TypeError(f"Expected NowcastResults or a DataFrame; got {type(results).__name__}.")


def plot_factors(
    results: NowcastResults | pd.DataFrame,
    *,
    factors: list[str] | None = None,
    backend: str = "plotly",
    theme: Theme | str | None = None,
    title: str | None = None,
    shade_forecast: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Plot the smoothed factors over time.

    Parameters
    ----------
    results : NowcastResults or pandas.DataFrame
        Results with ``factors`` or a DataFrame of factors (columns) on a
        PeriodIndex.
    factors : list of str, optional
        Subset of factor columns (default: all).
    backend : {"plotly", "matplotlib"}, default "plotly"
        Plotting library.
    theme : Theme or str, optional
        Visual theme.
    title : str, optional
        Title (default ``"Estimated factors"``).
    shade_forecast : bool, default True
        Shade the periods after the end of the estimation sample (factor forecasts),
        when ``results`` carries its data.
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
        No factors, unknown factor names, or unknown backend.
    TypeError
        Unsupported input type.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> idx = pd.period_range("2020-01", periods=12, freq="M")
    >>> f = pd.DataFrame({"f1": np.arange(12.0), "f2": np.ones(12)}, index=idx)
    >>> len(plot_factors(f).data)
    2
    """
    th, be = resolve(theme, backend, ax)
    frame, end = _factor_frame(results)
    frame = _select_factors(frame, factors)
    if frame.shape[1] == 0 or frame.empty:
        raise ValueError("No factors to plot.")
    shade = None
    if shade_forecast and end is not None and isinstance(frame.index, pd.PeriodIndex):
        after = frame.index[frame.index > end]
        if len(after):
            shade = (to_plot_index(frame.index[frame.index <= end])[-1], to_plot_index(after)[-1])
    title = "Estimated factors" if title is None else title
    if be == "plotly":
        return _factors_plotly(frame, shade, th, title)
    return _factors_mpl(frame, shade, th, title, ax, figsize)


def _factors_plotly(frame: pd.DataFrame, shade: Any, theme: Theme, title: str) -> Any:
    import plotly.graph_objects as go

    fig = new_plotly_figure()
    x = to_plot_index(frame.index)
    labels = [str(p) for p in frame.index]
    for i, col in enumerate(frame.columns):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=frame[col],
                name=str(col),
                mode="lines",
                customdata=labels,
                line={"color": theme.color(i), "width": theme.line_width},
                hovertemplate=f"%{{customdata}}<br>{col}: %{{y:.3f}}<extra></extra>",
            )
        )
    if shade is not None:
        fig.add_vrect(
            x0=shade[0],
            x1=shade[1],
            fillcolor=rgba(theme.neutral_color, 0.15),
            line_width=0,
            annotation_text="forecast",
            annotation_position="top left",
        )
    fig.add_hline(y=0, line={"color": theme.grid_color, "width": 1})
    return finish_plotly(fig, theme, title=title, ylabel="Factor value")


def _factors_mpl(
    frame: pd.DataFrame,
    shade: Any,
    theme: Theme,
    title: str,
    ax: Axes | None,
    figsize: tuple[float, float] | None,
) -> Any:
    x = to_plot_index(frame.index)
    with mpl_context(theme):
        fig, axes = new_axes(ax, theme, figsize)
        for i, col in enumerate(frame.columns):
            axes.plot(x, as_float_array(frame[col]), color=theme.color(i), label=str(col))
        if shade is not None:
            axes.axvspan(shade[0], shade[1], color=theme.neutral_color, alpha=0.15, linewidth=0)
        axes.axhline(0.0, color=theme.muted_text_color, linewidth=0.6)
        finish_mpl(axes, theme, title=title, ylabel="Factor value", dates=True)
    return fig


# ---------------------------------------------------------------------- eigenvalues
def panel_eigenvalues(data: MixedFrequencyData | pd.DataFrame) -> pd.Series:
    """Eigenvalues of the correlation matrix of the balanced base-frequency panel.

    Lower-frequency series are dropped (they are observed only in their storage
    slots) as well as every period with a missing value, so the result matches the
    principal-components input of the two-step estimator (Stock & Watson, 2002).

    Parameters
    ----------
    data : MixedFrequencyData or pandas.DataFrame
        Panel. For a DataFrame every column is used.

    Returns
    -------
    pandas.Series
        Eigenvalues in decreasing order, index ``1..N`` named ``component``.

    Raises
    ------
    ValueError
        Fewer than two series or two complete periods, or a constant series.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> x = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 4)))
    >>> ev = panel_eigenvalues(x)
    >>> len(ev), round(float(ev.sum()), 6)
    (4, 4.0)
    """
    if isinstance(data, MixedFrequencyData):
        frame = data.data[data.columns_with_frequency(data.base_frequency)]
    else:
        frame = pd.DataFrame(data)
    balanced = frame.dropna(axis=0, how="any")
    if balanced.shape[1] < 2 or balanced.shape[0] < 2:
        raise ValueError(
            "Need at least two base-frequency series and two complete periods to compute "
            f"eigenvalues; balanced panel has shape {balanced.shape}."
        )
    values = balanced.to_numpy(dtype=float)
    sd = values.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        raise ValueError("Cannot compute eigenvalues: a series is constant on the balanced panel.")
    corr = np.corrcoef(values, rowvar=False)
    eig = np.sort(np.linalg.eigvalsh(corr))[::-1]
    index = pd.RangeIndex(1, eig.size + 1, name="component")
    return pd.Series(np.clip(eig, 0.0, None), index=index, name="eigenvalue")


def _eigen_input(results: Any) -> tuple[pd.Series, int | None]:
    """Eigenvalues and the selected number of factors from several input types."""
    if isinstance(results, pd.Series):
        return results.astype(float), None
    if isinstance(results, (np.ndarray, list, tuple)):
        arr = np.asarray(results, dtype=float).ravel()
        return pd.Series(arr, index=pd.RangeIndex(1, arr.size + 1, name="component")), None
    eigen = getattr(results, "eigenvalues", None)
    n_selected = getattr(results, "r_star", None)
    if isinstance(results, NowcastResults):
        n_selected = results.n_factors
        if eigen is None:
            if results.data is None:
                raise ValueError("Results carry neither eigenvalues nor data.")
            eigen = panel_eigenvalues(results.data)
    if isinstance(eigen, pd.Series):
        return eigen.astype(float), n_selected
    raise TypeError(
        "Expected results with eigenvalues, a FactorSelectionResult, a Series or an array; "
        f"got {type(results).__name__}."
    )


def plot_eigenvalues(
    results: Any,
    *,
    n_factors: int | None = None,
    max_components: int | None = 20,
    backend: str = "plotly",
    theme: Theme | str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Scree plot: eigenvalues of the panel in decreasing order.

    The bars of the retained components (``n_factors``) are highlighted; the hover
    text (Plotly) shows the share of variance and the cumulative share.

    Parameters
    ----------
    results : FactorResults, FactorSelectionResult, pandas.Series or array-like
        ``TwoStepResults`` (uses its ``eigenvalues``), any factor results with data
        (eigenvalues of the balanced panel, see :func:`panel_eigenvalues`), a
        :class:`~nowcastbox.selection.FactorSelectionResult` or the eigenvalues
        themselves.
    n_factors : int, optional
        Number of retained components to highlight (default: from ``results``).
    max_components : int or None, default 20
        Show at most this many components.
    backend : {"plotly", "matplotlib"}, default "plotly"
        Plotting library.
    theme : Theme or str, optional
        Visual theme.
    title : str, optional
        Title (default ``"Scree plot"``).
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
        No eigenvalues available, or unknown backend.
    TypeError
        Unsupported input type.

    Examples
    --------
    >>> fig = plot_eigenvalues([4.0, 1.0, 0.5, 0.3], n_factors=1)
    >>> list(fig.data[0].x)
    [1, 2, 3, 4]
    """
    th, be = resolve(theme, backend, ax)
    eigen, selected = _eigen_input(results)
    if eigen.empty:
        raise ValueError("No eigenvalues to plot.")
    k = selected if n_factors is None else n_factors
    total = float(eigen.sum())
    share = eigen / total if total > 0 else eigen * 0.0
    if max_components is not None:
        eigen, share = eigen.iloc[:max_components], share.iloc[:max_components]
    comps = list(range(1, len(eigen) + 1))
    colors = [
        th.color(0) if (k is not None and i < k) else th.neutral_color for i in range(len(comps))
    ]
    title = "Scree plot" if title is None else title
    if be == "plotly":
        import plotly.graph_objects as go

        fig = new_plotly_figure()
        fig.add_trace(
            go.Bar(
                x=comps,
                y=eigen.to_numpy(),
                marker={"color": colors},
                name="Eigenvalue",
                customdata=np.column_stack([share.to_numpy(), share.cumsum().to_numpy()]),
                hovertemplate=(
                    "Component %{x}<br>Eigenvalue: %{y:.3f}<br>Share: %{customdata[0]:.1%}"
                    "<br>Cumulative: %{customdata[1]:.1%}<extra></extra>"
                ),
            )
        )
        fig.update_xaxes(dtick=1)
        return finish_plotly(
            fig,
            th,
            title=title,
            xlabel="Component",
            ylabel="Eigenvalue",
            hovermode="closest",
            showlegend=False,
        )
    with mpl_context(th):
        fig, axes = new_axes(ax, th, figsize)
        axes.bar(comps, as_float_array(eigen), color=colors, width=0.7)
        axes.set_xticks(comps)
        finish_mpl(axes, th, title=title, xlabel="Component", ylabel="Eigenvalue", legend=False)
    return fig


# ---------------------------------------------------------------------- loadings
def _loading_frame(results: NowcastResults | pd.DataFrame) -> pd.DataFrame:
    if isinstance(results, NowcastResults):
        if results.loadings is None:
            raise ValueError(f"{type(results).__name__} has no loadings to plot.")
        return results.loadings.copy()
    if isinstance(results, pd.DataFrame):
        return results.copy()
    raise TypeError(f"Expected NowcastResults or a DataFrame; got {type(results).__name__}.")


def _diverging_scale(theme: Theme) -> list[list[Any]]:
    return [[0.0, theme.negative_color], [0.5, "#f0efec"], [1.0, theme.positive_color]]


def plot_loadings(
    results: NowcastResults | pd.DataFrame,
    *,
    style: Literal["heatmap", "bar"] = "heatmap",
    factors: list[str] | None = None,
    sort_by: str | None = None,
    backend: str = "plotly",
    theme: Theme | str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Plot factor loadings as a heatmap (series x factors) or horizontal bars.

    Parameters
    ----------
    results : NowcastResults or pandas.DataFrame
        Results with ``loadings`` or the loadings themselves (rows: series, columns:
        factors).
    style : {"heatmap", "bar"}, default "heatmap"
        Heatmap with a diverging scale centred on zero, or one bar panel per factor
        (small multiples sharing the series axis).
    factors : list of str, optional
        Subset of factors.
    sort_by : str, optional
        Sort series by the loading on this factor.
    backend : {"plotly", "matplotlib"}, default "plotly"
        Plotting library.
    theme : Theme or str, optional
        Visual theme.
    title : str, optional
        Title (default ``"Factor loadings"``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (Matplotlib only; heatmap or a single factor).
    figsize : tuple of float, optional
        Matplotlib figure size.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        The figure.

    Raises
    ------
    ValueError
        No loadings, unknown factors, invalid ``style`` or ``ax`` with several bar panels.
    TypeError
        Unsupported input type.

    Examples
    --------
    >>> import pandas as pd
    >>> lam = pd.DataFrame({"f1": [0.9, -0.2, 0.5]}, index=["ip", "pmi", "sales"])
    >>> fig = plot_loadings(lam, style="bar", sort_by="f1")
    >>> list(fig.data[0].y)
    ['pmi', 'sales', 'ip']
    """
    th, be = resolve(theme, backend, ax)
    if style not in ("heatmap", "bar"):
        raise ValueError(f"style must be 'heatmap' or 'bar'; got {style!r}.")
    frame = _loading_frame(results)
    frame = _select_factors(frame, factors)
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("No loadings to plot.")
    if sort_by is not None:
        if sort_by not in frame.columns:
            raise ValueError(f"sort_by={sort_by!r} is not a factor column.")
        frame = frame.sort_values(sort_by)
    frame = frame.astype(float)
    title = "Factor loadings" if title is None else title
    if style == "bar" and ax is not None and frame.shape[1] > 1:
        raise ValueError("'ax' can only be used with style='bar' for a single factor.")
    if be == "plotly":
        draw_plotly = _heatmap_plotly if style == "heatmap" else _bars_plotly
        return draw_plotly(frame, th, title)
    draw_mpl = _heatmap_mpl if style == "heatmap" else _bars_mpl
    return draw_mpl(frame, th, title, ax, figsize)


def _heatmap_plotly(frame: pd.DataFrame, theme: Theme, title: str) -> Any:
    import plotly.graph_objects as go

    bound = float(np.nanmax(np.abs(frame.to_numpy()))) or 1.0
    fig = new_plotly_figure()
    fig.add_trace(
        go.Heatmap(
            z=frame.to_numpy(),
            x=[str(c) for c in frame.columns],
            y=[str(i) for i in frame.index],
            zmin=-bound,
            zmax=bound,
            colorscale=_diverging_scale(theme),
            xgap=2,
            ygap=2,
            colorbar={"title": {"text": "Loading"}},
            hovertemplate="%{y} on %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig = finish_plotly(fig, theme, title=title, hovermode="closest", showlegend=False)
    fig.update_layout(height=max(theme.height, 22 * frame.shape[0] + 160))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def _bars_plotly(frame: pd.DataFrame, theme: Theme, title: str) -> Any:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    k = frame.shape[1]
    fig = make_subplots(rows=1, cols=k, shared_yaxes=True, subplot_titles=[str(c) for c in frame])
    names = [str(i) for i in frame.index]
    for j, col in enumerate(frame.columns):
        values = frame[col].to_numpy()
        colors = [theme.positive_color if v >= 0 else theme.negative_color for v in values]
        fig.add_trace(
            go.Bar(
                x=values,
                y=names,
                orientation="h",
                marker={"color": colors},
                name=str(col),
                hovertemplate=f"%{{y}}: %{{x:.3f}}<extra>{col}</extra>",
            ),
            row=1,
            col=j + 1,
        )
    fig = finish_plotly(fig, theme, title=title, hovermode="closest", showlegend=False)
    fig.update_layout(height=max(theme.height, 22 * frame.shape[0] + 160))
    return fig


def _heatmap_mpl(
    frame: pd.DataFrame,
    theme: Theme,
    title: str,
    ax: Axes | None,
    figsize: tuple[float, float] | None,
) -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "nowcastbox_diverging", [theme.negative_color, "#f0efec", theme.positive_color]
    )
    bound = float(np.nanmax(np.abs(frame.to_numpy()))) or 1.0
    height = max(3.0, 0.28 * frame.shape[0] + 1.5)
    size = figsize or (max(4.0, 1.2 * frame.shape[1] + 3.0), height)
    with mpl_context(theme):
        fig, axes = new_axes(ax, theme, size)
        image = axes.imshow(frame.to_numpy(), cmap=cmap, vmin=-bound, vmax=bound, aspect="auto")
        axes.set_xticks(range(frame.shape[1]), [str(c) for c in frame.columns])
        axes.set_yticks(range(frame.shape[0]), [str(i) for i in frame.index])
        axes.grid(False)
        fig.colorbar(image, ax=axes, label="Loading")
        finish_mpl(axes, theme, title=title, legend=False)
    return fig


def _bars_mpl(
    frame: pd.DataFrame,
    theme: Theme,
    title: str,
    ax: Axes | None,
    figsize: tuple[float, float] | None,
) -> Any:
    import matplotlib.pyplot as plt

    k = frame.shape[1]
    height = max(3.0, 0.28 * frame.shape[0] + 1.5)
    size = figsize or (max(4.0, 3.2 * k + 1.5), height)
    names = [str(i) for i in frame.index]
    with mpl_context(theme):
        if ax is not None:
            fig, axes_list = ax.get_figure(), [ax]
        else:
            fig, grid = plt.subplots(
                1, k, figsize=size, sharey=True, squeeze=False, layout="constrained"
            )
            axes_list = list(grid[0])
        for axes, col in zip(axes_list, frame.columns, strict=True):
            values = as_float_array(frame[col])
            colors = [theme.positive_color if v >= 0 else theme.negative_color for v in values]
            axes.barh(names, values, color=colors, height=0.7)
            axes.axvline(0.0, color=theme.muted_text_color, linewidth=0.6)
            axes.set_title(str(col), color=theme.text_color)
        if fig is not None and ax is None:
            fig.suptitle(title, x=0.02, ha="left", color=theme.text_color)
        elif ax is not None:
            finish_mpl(ax, theme, title=title, legend=False)
    return fig

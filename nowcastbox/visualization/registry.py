"""Registration of the plots with :func:`nowcastbox.core.results.register_plot`.

Importing :mod:`nowcastbox.visualization` runs this module, after which
``results.plot(kind)`` works for every results class:

=====================  =====================================  ==========================
kind                   function                               results classes
=====================  =====================================  ==========================
``"forecast"``         :func:`~.forecast.plot_forecast`        all ``NowcastResults``
``"fan"``              :func:`~.forecast.plot_fan_chart`       all (needs ``std``)
``"data_availability"`` :func:`~.data_flow.plot_data_availability` all (needs ``data``)
``"ragged_edge"``      alias of ``"data_availability"``
``"factors"``          :func:`~.factors.plot_factors`          ``FactorResults``
``"eigenvalues"``      :func:`~.factors.plot_eigenvalues`      ``FactorResults``
``"loadings"``         :func:`~.factors.plot_loadings`         ``FactorResults``
``"loglikelihood"``    :func:`~.diagnostics.plot_loglikelihood` ``MixedFreqDFMResults``
``"density"``          :func:`~.forecast.plot_fan_chart` of    all (needs ``std``)
                       ``results.distribution(...)`` (I5)
=====================  =====================================  ==========================

``TwoStepResults`` and ``MixedFreqDFMResults`` inherit from ``FactorResults`` and so get
every factor plot through the registry's MRO lookup.

The news objects (:class:`~nowcastbox.news.NewsResults`,
:class:`~nowcastbox.news.NowcastTracker`) have their own registry
(:func:`~nowcastbox.news.register_news_plot`); this module makes their ``"waterfall"`` and
``"path"`` kinds backend-aware: ``news.plot("waterfall")`` keeps the default Matplotlib
chart of :mod:`nowcastbox.news`, while ``backend="plotly"`` (or ``"matplotlib"`` with a
``theme``) draws :func:`~.news.plot_news_waterfall` / :func:`~.news.plot_nowcast_tracker`
with the themes of this package.
"""

from __future__ import annotations

from typing import Any

from nowcastbox.core.results import FactorResults, NowcastResults, register_plot
from nowcastbox.models.em import MixedFreqDFMResults
from nowcastbox.news import NewsResults, NowcastTracker, register_news_plot
from nowcastbox.news.decomposition import plot_waterfall_default as _news_waterfall_default
from nowcastbox.news.tracker import plot_path_default as _tracker_path_default
from nowcastbox.visualization.data_flow import plot_data_availability
from nowcastbox.visualization.diagnostics import plot_loglikelihood
from nowcastbox.visualization.factors import plot_eigenvalues, plot_factors, plot_loadings
from nowcastbox.visualization.forecast import plot_fan_chart, plot_forecast
from nowcastbox.visualization.news import plot_news_waterfall, plot_nowcast_tracker

__all__ = ["REGISTERED_PLOTS", "plot_density", "plot_news_results", "plot_tracker_results"]

_NEWS_GROUPS = ("category", "block")
"""``by`` values of :meth:`NewsResults.plot` that are columns of the release table."""


def plot_density(results: NowcastResults, **kwargs: Any) -> Any:
    """Fan chart of the predictive distribution of ``results`` (``kind="density"``).

    Parameters
    ----------
    results : NowcastResults
        Fitted results with a ``std`` column.
    **kwargs
        ``n_boot``, ``method``, ``random_state``, ``n_jobs``, ``periods`` go to
        :meth:`~nowcastbox.core.results.NowcastResults.distribution`; the rest to
        :func:`~nowcastbox.visualization.plot_fan_chart`.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        The fan chart.

    Examples
    --------
    >>> from nowcastbox.models import TwoStepDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> res = TwoStepDFM(n_factors=1).fit(simulate_two_step_example(random_state=0), "gdp")
    >>> fig = res.plot("density")
    >>> fig.data[-1].name
    'Median'
    """
    keys = ("n_boot", "method", "random_state", "n_jobs", "periods", "block_length")
    dist_kwargs = {k: kwargs.pop(k) for k in keys if k in kwargs}
    dist = results.distribution(**dist_kwargs)
    kwargs.setdefault("observed", results.observed.dropna())
    kwargs.setdefault("title", f"{results.target}: predictive distribution")
    return plot_fan_chart(dist, **kwargs)


def plot_news_results(
    news: NewsResults, by: str = "category", backend: str | None = None, **kwargs: Any
) -> Any:
    """Waterfall of a news decomposition, on either backend (``news.plot("waterfall")``).

    Parameters
    ----------
    news : NewsResults
        News decomposition.
    by : str, default "category"
        Grouping (``"series"``, ``"block"``, ``"category"``, ``"release"``).
    backend : {"matplotlib", "plotly"}, optional
        ``None`` (default) keeps the Matplotlib chart of :mod:`nowcastbox.news`
        (unless a ``theme`` is given); otherwise
        :func:`~nowcastbox.visualization.plot_news_waterfall` with that backend.
    **kwargs
        Passed to the plotting function.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        The figure.

    Examples
    --------
    >>> news.plot("waterfall", backend="plotly", by="category")  # doctest: +SKIP
    """
    if backend is None and "theme" not in kwargs:
        return _news_waterfall_default(news, by=by, **kwargs)
    group = by if by in _NEWS_GROUPS else None
    return plot_news_waterfall(news, group_by=group, backend=backend or "matplotlib", **kwargs)


def plot_tracker_results(tracker: NowcastTracker, backend: str | None = None, **kwargs: Any) -> Any:
    """Nowcast tracker chart on either backend (``tracker.plot("path")``).

    Parameters
    ----------
    tracker : NowcastTracker
        Tracker.
    backend : {"matplotlib", "plotly"}, optional
        ``None`` (default) keeps the Matplotlib chart of :mod:`nowcastbox.news`
        (unless a ``theme`` is given); otherwise
        :func:`~nowcastbox.visualization.plot_nowcast_tracker` with that backend.
    **kwargs
        Passed to the plotting function.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
        The figure.

    Examples
    --------
    >>> tracker.plot("path", backend="plotly")  # doctest: +SKIP
    """
    if backend is None and "theme" not in kwargs:
        return _tracker_path_default(tracker, **kwargs)
    return plot_nowcast_tracker(tracker, backend=backend or "matplotlib", **kwargs)


REGISTERED_PLOTS: dict[str, tuple[type, object]] = {
    "forecast": (NowcastResults, plot_forecast),
    "fan": (NowcastResults, plot_fan_chart),
    "data_availability": (NowcastResults, plot_data_availability),
    "ragged_edge": (NowcastResults, plot_data_availability),
    "factors": (FactorResults, plot_factors),
    "eigenvalues": (FactorResults, plot_eigenvalues),
    "loadings": (FactorResults, plot_loadings),
    "loglikelihood": (MixedFreqDFMResults, plot_loglikelihood),
    "density": (NowcastResults, plot_density),
}
"""``{kind: (results class, plotting function)}`` registered on import."""

for _kind, (_cls, _func) in REGISTERED_PLOTS.items():
    register_plot(_kind, _cls)(_func)  # type: ignore[arg-type]

register_news_plot("waterfall", NewsResults)(plot_news_results)
register_news_plot("path", NowcastTracker)(plot_tracker_results)

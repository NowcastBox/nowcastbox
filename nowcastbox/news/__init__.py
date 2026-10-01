"""News decomposition (Bańbura & Modugno, 2014) and nowcast explainability (I6).

* :func:`news_decomposition` - revision of a nowcast between two vintages split into
  the impact of each new release (news x weight), data revisions and re-estimation;
  :class:`NewsResults` with ``summary()``, ``to_frame(by="series"|"block"|"category")``,
  ``waterfall()`` and ``plot("waterfall")``.
* :func:`nowcast_tracker` - the nowcast through the vintages of a quarter with
  cumulative contributions by group (:class:`NowcastTracker`).
* :func:`level_contributions` - contribution of each series to the level of the
  nowcast (:class:`LevelContributions`).
* :func:`data_revisions` / :func:`compare_vintages` - released, revised and removed
  cells between vintages.
* Every :class:`~nowcastbox.core.results.NowcastResults` has native ``news``,
  ``nowcast_tracker`` and ``level_contributions`` methods delegating here
  (``res.news(old, new, "2020Q2")``); :func:`attach` adds the same methods to other
  (duck-typed) results classes.

Supported models: :class:`~nowcastbox.models.MixedFreqDFM` (monthly grids and
weekly/daily grids with calendar aggregation, whose time-varying observation equation is
rebuilt for each vintage) and :class:`~nowcastbox.models.TwoStepDFM` (``aggregate=
"factors"``, or ``"variables"``: vintages are passed through the fitted predictor filters,
a raw release is attributed to the filtered observation it completes and a revision of a
raw value to the data revisions).

Examples
--------
>>> import nowcastbox as nb
>>> from nowcastbox.news import news_decomposition
>>> from nowcastbox.models.two_step import simulate_two_step_example
>>> data = simulate_two_step_example(random_state=0)
>>> res = nb.MixedFreqDFM(n_factors=1, max_iter=20).fit(data, "gdp")
>>> news = news_decomposition(res, data.truncate(end=data.index[-3]), data, "2014Q4")
>>> bool(abs(news.old_nowcast + news.revisions_effect + news.news_effect - news.new_nowcast) < 1e-8)
True
"""

from __future__ import annotations

from typing import Any

from nowcastbox.core.results import NowcastResults
from nowcastbox.news.contributions import LevelContributions, level_contributions
from nowcastbox.news.decomposition import NewsResults, news_decomposition, news_weights
from nowcastbox.news.plotting import available_news_plots, register_news_plot
from nowcastbox.news.revisions import VintageDiff, compare_vintages, data_revisions
from nowcastbox.news.tracker import NowcastTracker, nowcast_tracker

__all__ = [
    "LevelContributions",
    "NewsResults",
    "NowcastTracker",
    "VintageDiff",
    "attach",
    "available_news_plots",
    "compare_vintages",
    "data_revisions",
    "detach",
    "level_contributions",
    "news_decomposition",
    "news_weights",
    "nowcast_tracker",
    "register_news_plot",
]

_MARK = "_nowcastbox_news_method"


def _news_method(
    self: NowcastResults,
    old: object,
    new: object,
    target_period: object = None,
    **kwargs: Any,
) -> NewsResults:
    """News decomposition of these results between two vintages.

    See :func:`nowcastbox.news.news_decomposition` (``old``/``new`` are the old and new
    vintages).
    """
    return news_decomposition(self, old, new, target_period, **kwargs)


def _tracker_method(
    self: NowcastResults,
    data: object,
    calendar: object = None,
    target_period: object = None,
    start: Any = None,
    end: Any = None,
    **kwargs: Any,
) -> NowcastTracker:
    """Nowcast tracker with these (fixed) parameters.

    See :func:`nowcastbox.news.nowcast_tracker`.
    """
    return nowcast_tracker(self, data, calendar, target_period, start, end, **kwargs)


def _level_method(
    self: NowcastResults, data: object = None, target_period: object = None, **kwargs: Any
) -> LevelContributions:
    """Contribution of each series to the level of the nowcast.

    See :func:`nowcastbox.news.level_contributions`.
    """
    return level_contributions(self, data, target_period, **kwargs)


_METHODS = {
    "news": _news_method,
    "nowcast_tracker": _tracker_method,
    "level_contributions": _level_method,
}
for _func in _METHODS.values():
    setattr(_func, _MARK, True)


def attach(results_type: type = NowcastResults) -> list[str]:
    """Add ``news``, ``nowcast_tracker`` and ``level_contributions`` methods to a class.

    The methods delegate to :func:`news_decomposition`, :func:`nowcast_tracker` and
    :func:`level_contributions` (they raise ``TypeError`` for results that are not
    state-space factor models). Existing attributes that were not added by this
    function are never overwritten, so native methods take precedence: since the
    wave-2 integration :class:`~nowcastbox.core.results.NowcastResults` defines them
    natively and ``attach()`` with the default class adds nothing. Use it for other
    (duck-typed) results classes. Calling it twice is harmless.

    Parameters
    ----------
    results_type : type, default NowcastResults
        Class to extend (subclasses inherit the methods).

    Returns
    -------
    list of str
        Names of the methods now provided by this module.

    Examples
    --------
    >>> from nowcastbox.news import attach, detach
    >>> attach()  # NowcastResults already has the methods
    []
    >>> class MyResults:
    ...     pass
    >>> sorted(attach(MyResults))
    ['level_contributions', 'news', 'nowcast_tracker']
    >>> detach(MyResults)
    """
    attached = []
    for name, func in _METHODS.items():
        current = results_type.__dict__.get(name)
        if current is not None and not getattr(current, _MARK, False):
            continue
        setattr(results_type, name, func)
        attached.append(name)
    return attached


def detach(results_type: type = NowcastResults) -> None:
    """Remove the methods added by :func:`attach`.

    Parameters
    ----------
    results_type : type, default NowcastResults
        Class previously extended.

    Examples
    --------
    >>> from nowcastbox.news import attach, detach
    >>> class MyResults:
    ...     pass
    >>> _ = attach(MyResults)
    >>> detach(MyResults)
    >>> hasattr(MyResults, "news")
    False
    """
    for name in _METHODS:
        current = results_type.__dict__.get(name)
        if current is not None and getattr(current, _MARK, False):
            delattr(results_type, name)

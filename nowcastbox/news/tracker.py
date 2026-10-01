r"""Nowcast tracker: the nowcast through the data flow of a quarter (innovation I6).

The tracker evaluates the model on a sequence of vintages
:math:`\Omega_{v_0} \subseteq \Omega_{v_1} \subseteq \dots` (pseudo real-time vintages
built from a release calendar, or real vintages from a
:class:`~nowcastbox.vintages.VintageStore`) and chains the news decompositions of
consecutive vintages (Bańbura & Modugno, 2014; Bańbura, Giannone, Modugno & Reichlin,
2013), so that

.. math::

    \hat y_\tau(v_k) = \hat y_\tau(v_0) + \sum_{l=1}^{k}\Bigl(\sum_{g} \text{news}_{g,l}
    + \text{revisions}_l + \text{re-estimation}_l\Bigr)

for every group :math:`g` (series, block or category). This is the "nowcast tracker"
chart of central-bank nowcasting reports.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.base import BaseNowcaster
from nowcastbox.core.data import MixedFrequencyData, as_mixed_frequency_data
from nowcastbox.core.frequency import Frequency
from nowcastbox.core.results import NowcastResults
from nowcastbox.news._model import to_period
from nowcastbox.news.decomposition import GROUPINGS, NewsResults, news_decomposition
from nowcastbox.news.plotting import plot_news_object, register_news_plot, tracker_chart
from nowcastbox.vintages import ReleaseCalendar, VintageStore, pseudo_real_time

__all__ = ["NowcastTracker", "nowcast_tracker", "plot_path_default"]

logger = get_logger(__name__)

DateLike = str | dt.date | pd.Timestamp

_PATH_COLUMNS = ["nowcast", "change", "news", "revisions", "reestimation", "n_releases"]


@dataclass(frozen=True, kw_only=True, eq=False)
class NowcastTracker:
    """Nowcast path through a sequence of vintages with its news decomposition.

    Parameters
    ----------
    target : str
        Target series.
    target_period : pandas.Period
        Target period.
    by : str
        Grouping of :attr:`contributions` (``"series"``, ``"block"`` or ``"category"``).
    path : pandas.DataFrame
        Index = vintage dates; columns ``nowcast``, ``change`` (vs. previous vintage),
        ``news``, ``revisions``, ``reestimation`` and ``n_releases``.
    contributions : pandas.DataFrame
        Change of the nowcast at each vintage: news by group plus ``revisions`` and
        ``re-estimation`` columns (first row zero). Row sums equal ``path["change"]``.
    steps : tuple of NewsResults
        Decomposition between consecutive vintages.

    Examples
    --------
    >>> tr = nowcast_tracker(res, data, None, "2020Q2")  # doctest: +SKIP
    >>> tr.cumulative_contributions()  # doctest: +SKIP
    """

    target: str
    target_period: pd.Period
    by: str
    path: pd.DataFrame
    contributions: pd.DataFrame
    steps: tuple[NewsResults, ...]

    @property
    def nowcasts(self) -> pd.Series:
        """Nowcast at each vintage."""
        return self.path["nowcast"].copy()

    def cumulative_contributions(self) -> pd.DataFrame:
        """Cumulative contributions since the first vintage.

        Returns
        -------
        pandas.DataFrame
            Running sums of :attr:`contributions`; the first nowcast plus each row sum
            gives the nowcast of that vintage.

        Examples
        --------
        >>> tr.cumulative_contributions().iloc[-1]  # doctest: +SKIP
        """
        return self.contributions.cumsum()

    def check_identity(self, atol: float = 1e-8) -> bool:
        """Whether first nowcast plus cumulative contributions reproduces the path.

        Parameters
        ----------
        atol : float, default 1e-8
            Absolute tolerance.

        Returns
        -------
        bool
            True when every vintage closes within ``atol``.

        Examples
        --------
        >>> tr.check_identity()  # doctest: +SKIP
        True
        """
        start = float(self.path["nowcast"].iloc[0])
        rebuilt = start + self.cumulative_contributions().sum(axis=1)
        return bool(np.all(np.abs(rebuilt - self.path["nowcast"]) <= atol))

    def releases(self) -> pd.DataFrame:
        """All releases of the period with the vintage in which they arrived.

        Returns
        -------
        pandas.DataFrame
            Concatenation of the ``releases`` tables of :attr:`steps` with a leading
            ``vintage`` column.

        Examples
        --------
        >>> tr.releases().head()  # doctest: +SKIP
        """
        parts = []
        for vintage, step in zip(self.path.index[1:], self.steps, strict=True):
            part = step.releases.copy()
            part.insert(0, "vintage", vintage)
            parts.append(part)
        return pd.concat(parts, ignore_index=True)

    def to_frame(self) -> pd.DataFrame:
        """Path and contributions side by side.

        Returns
        -------
        pandas.DataFrame
            :attr:`path` joined with :attr:`contributions` (prefixed ``contrib:``).

        Examples
        --------
        >>> tr.to_frame().columns[:3].tolist()  # doctest: +SKIP
        ['nowcast', 'change', 'news']
        """
        contrib = self.contributions.add_prefix("contrib:")
        return self.path.join(contrib)

    def summary(self) -> str:
        """Formatted text summary.

        Returns
        -------
        str
            One line per vintage.

        Examples
        --------
        >>> print(tr.summary())  # doctest: +SKIP
        """
        width = 78
        lines = [
            "=" * width,
            f"Nowcast tracker: {self.target} {self.target_period} ({len(self.path)} vintages)",
            "=" * width,
            f"  {'vintage':<14}{'nowcast':>12}{'change':>12}{'news':>12}"
            f"{'revisions':>12}{'releases':>10}",
        ]
        for vintage, row in self.path.iterrows():
            label = vintage.strftime("%Y-%m-%d") if isinstance(vintage, pd.Timestamp) else vintage
            lines.append(
                f"  {label!s:<14}{row['nowcast']:>12.4f}{row['change']:>12.4f}"
                f"{row['news']:>12.4f}{row['revisions']:>12.4f}{int(row['n_releases']):>10d}"
            )
        lines.append("=" * width)
        return "\n".join(lines)

    def plot(self, kind: str = "path", **kwargs: Any) -> Any:
        """Plot through the news plot registry.

        Parameters
        ----------
        kind : str, default "path"
            Plot kind (``"path"``: nowcast path with stacked contributions).
        **kwargs
            Passed to the plotting function.

        Returns
        -------
        object
            Figure.

        Examples
        --------
        >>> tr.plot()  # doctest: +SKIP
        """
        return plot_news_object(self, kind, **kwargs)


@register_news_plot("path", NowcastTracker)
def plot_path_default(tracker: NowcastTracker, **kwargs: Any) -> Any:
    """Default Matplotlib tracker chart (``kind="path"``).

    :mod:`nowcastbox.visualization` wraps it so that ``tracker.plot("path")`` keeps
    this chart and ``backend="plotly"`` switches to the themed charts.

    Parameters
    ----------
    tracker : NowcastTracker
        Nowcast tracker.
    **kwargs
        Passed to the chart (``ax``, ``title``, ``figsize``...).

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Examples
    --------
    >>> fig = plot_path_default(tracker)  # doctest: +SKIP
    """
    kwargs.setdefault("title", f"{tracker.target} {tracker.target_period}: nowcast tracker")
    return tracker_chart(tracker.path["nowcast"], tracker.contributions, **kwargs)


# ====================================================================== helpers
def _calendar(data: MixedFrequencyData, calendar: object) -> ReleaseCalendar:
    if isinstance(calendar, ReleaseCalendar):
        return calendar
    if calendar is None:
        return ReleaseCalendar.from_data(data)
    if isinstance(calendar, Integral) and not isinstance(calendar, bool):
        return ReleaseCalendar.from_data(data, dict.fromkeys(data.columns, int(calendar)))
    if isinstance(calendar, Mapping | pd.Series):
        return ReleaseCalendar.from_data(data, calendar)  # type: ignore[arg-type]
    raise ValueError(
        "calendar must be a ReleaseCalendar, a mapping/Series of delays in days, an int or "
        f"None (release_delays metadata); got {type(calendar).__name__}."
    )


def _dates(
    source: object,
    calendar: ReleaseCalendar | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dates: Sequence[DateLike] | pd.DatetimeIndex | None,
) -> pd.DatetimeIndex:
    if dates is not None:
        out = pd.DatetimeIndex([pd.Timestamp(d) for d in dates]).unique().sort_values()
    elif isinstance(source, VintageStore):
        stored = source.vintage_dates()
        out = stored[(stored > start) & (stored <= end)].insert(0, start)
    else:
        assert calendar is not None  # noqa: S101
        assert isinstance(source, MixedFrequencyData)  # noqa: S101
        out = calendar.release_dates(start, end, source).insert(0, start)
    if len(out) < 2:
        raise ValueError("The tracker needs at least two vintage dates.")
    return pd.DatetimeIndex(out, name="vintage")


def _panel(source: object, calendar: ReleaseCalendar | None, date: pd.Timestamp) -> Any:
    if isinstance(source, VintageStore):
        return source.as_of(date, as_mixed=True)
    return pseudo_real_time(source, delay=calendar, vintage=date)  # type: ignore[arg-type]


def _resolve_model(model_or_results: object, target: str | None) -> tuple[Any, Any, str]:
    """(fixed results or None, estimator or None, target)."""
    if isinstance(model_or_results, NowcastResults):
        return model_or_results, None, model_or_results.target
    if isinstance(model_or_results, BaseNowcaster):
        fitted = model_or_results.results_ if model_or_results.is_fitted else None
        name = target if target is not None else (fitted.target if fitted else None)
        if name is None:
            raise ValueError("target is required when an unfitted model is given.")
        return fitted, model_or_results, name
    raise TypeError(
        "model_or_results must be fitted results or a BaseNowcaster estimator, got "
        f"{type(model_or_results).__name__}."
    )


def _target_frequency(source: object, target: str, fixed: Any) -> Frequency:
    if fixed is not None:
        return fixed.target_frequency
    if isinstance(source, VintageStore):
        if target not in source.series:
            raise ValueError(f"The vintage store has no series {target!r}.")
        return Frequency.from_value(source.frequencies[target])
    assert isinstance(source, MixedFrequencyData)  # noqa: S101
    if target not in source.columns:
        raise ValueError(f"The data have no series {target!r}.")
    return source.metadata[target].frequency


def _default_window(
    period: pd.Period, start: DateLike | None, end: DateLike | None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    t0 = pd.Timestamp(start) if start is not None else period.start_time.normalize()
    t1 = pd.Timestamp(end) if end is not None else period.end_time.normalize()
    if t0 >= t1:
        raise ValueError(f"start ({t0.date()}) must be before end ({t1.date()}).")
    return t0, t1


# ====================================================================== main
def nowcast_tracker(
    model_or_results: object,
    data: object,
    calendar: object = None,
    target_period: object = None,
    start: DateLike | None = None,
    end: DateLike | None = None,
    *,
    dates: Sequence[DateLike] | pd.DatetimeIndex | None = None,
    target: str | None = None,
    refit: bool = False,
    by: str = "category",
    categories: Mapping[str, object] | None = None,
) -> NowcastTracker:
    """Track the nowcast of a target period through a sequence of vintages.

    Parameters
    ----------
    model_or_results : NowcastResults or BaseNowcaster
        Fitted results (parameters kept fixed), a fitted estimator (its ``results_``) or
        an unfitted estimator (fitted on the first vintage; then ``target`` is
        required). With ``refit=True`` an estimator is re-estimated at every vintage.
    data : MixedFrequencyData, pandas.DataFrame or VintageStore
        Final dataset from which pseudo real-time vintages are cut with ``calendar``,
        or a store of real vintages (then ``calendar`` is not used).
    calendar : ReleaseCalendar, mapping, Series or int, optional
        Release calendar or delays in days (default: ``release_delays`` metadata).
    target_period : period-like
        Target period (e.g. ``"2020Q2"``).
    start, end : date-like, optional
        Window of information dates: the first vintage is ``start`` and the following
        ones are the release dates in ``(start, end]`` (default: the first and last day
        of the target period).
    dates : sequence of date-like, optional
        Explicit vintage dates (override the release dates).
    target : str, optional
        Target of an unfitted estimator.
    refit : bool, default False
        Re-estimate the model at every vintage (re-estimation effect reported).
    by : {"series", "block", "category"}, default "category"
        Grouping of the contributions.
    categories : mapping of str to category, optional
        Override of the series categories.

    Returns
    -------
    NowcastTracker
        Nowcast path and contributions; ``check_identity()`` holds to rounding.

    Raises
    ------
    ValueError
        On a missing target period, an invalid window or grouping, or fewer than two
        vintage dates.
    TypeError
        If ``model_or_results`` is neither results nor an estimator, or ``refit=True``
        without an estimator.

    Examples
    --------
    >>> import nowcastbox as nb
    >>> from nowcastbox.news import nowcast_tracker
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> data = simulate_two_step_example(random_state=0)
    >>> res = nb.MixedFreqDFM(n_factors=1, max_iter=20).fit(data, "gdp")
    >>> delays = {c: 30 for c in data.columns} | {"gdp": 45}
    >>> tr = nowcast_tracker(res, data, delays, "2014Q4", "2014-10-01", "2015-02-28")
    >>> tr.check_identity()
    True
    """
    if by not in GROUPINGS:
        raise ValueError(f"by must be one of {GROUPINGS}, got {by!r}.")
    if target_period is None:
        raise ValueError("target_period is required.")
    fixed, estimator, name = _resolve_model(model_or_results, target)
    if refit and estimator is None:
        raise TypeError("refit=True needs an estimator (BaseNowcaster), not results.")
    source: object = data
    cal: ReleaseCalendar | None = None
    if not isinstance(data, VintageStore):
        source = as_mixed_frequency_data(data)  # type: ignore[arg-type]
        cal = _calendar(source, calendar)  # type: ignore[arg-type]
    period = to_period(target_period, _target_frequency(source, name, fixed))
    t0, t1 = _default_window(period, start, end)
    vintages = _dates(source, cal, t0, t1, dates)
    previous_panel = _panel(source, cal, vintages[0])
    if fixed is None:
        fixed = estimator.clone().fit(previous_panel, name)  # type: ignore[union-attr]
    previous_results = fixed
    steps: list[NewsResults] = []
    for date in vintages[1:]:
        panel = _panel(source, cal, date)
        new_results = estimator.clone().fit(panel, name) if refit else None  # type: ignore[union-attr]
        step = news_decomposition(
            previous_results,
            previous_panel,
            panel,
            period,
            new_results=new_results,
            categories=categories,
        )
        steps.append(step)
        previous_panel = panel
        previous_results = new_results if new_results is not None else previous_results
        logger.debug("tracker %s: nowcast %.4f", date.date(), step.new_nowcast)
    return _assemble(name, period, by, vintages, steps)


def _assemble(
    target: str,
    period: pd.Period,
    by: str,
    vintages: pd.DatetimeIndex,
    steps: list[NewsResults],
) -> NowcastTracker:
    rows = [
        {
            "nowcast": steps[0].old_nowcast,
            "change": 0.0,
            "news": 0.0,
            "revisions": 0.0,
            "reestimation": 0.0,
            "n_releases": 0,
        }
    ]
    contrib_rows: list[dict[str, float]] = [{}]
    for step in steps:
        rows.append(
            {
                "nowcast": step.new_nowcast,
                "change": step.new_nowcast - step.old_nowcast,
                "news": step.news_effect,
                "revisions": step.revisions_effect,
                "reestimation": step.reestimation_effect,
                "n_releases": step.n_releases,
            }
        )
        news = step.to_frame(by)["news"] if step.n_releases else pd.Series(dtype=float)
        entry = {str(k): float(v) for k, v in news.items()}
        entry["revisions"] = step.revisions_effect
        entry["re-estimation"] = step.reestimation_effect
        contrib_rows.append(entry)
    path = pd.DataFrame(rows, index=vintages, columns=_PATH_COLUMNS)
    path["n_releases"] = path["n_releases"].astype(int)
    contributions = pd.DataFrame(contrib_rows, index=vintages).fillna(0.0).astype(float)
    groups = [c for c in contributions.columns if c not in ("revisions", "re-estimation")]
    contributions = contributions[[*groups, "revisions", "re-estimation"]]
    return NowcastTracker(
        target=target,
        target_period=period,
        by=by,
        path=path,
        contributions=contributions,
        steps=tuple(steps),
    )

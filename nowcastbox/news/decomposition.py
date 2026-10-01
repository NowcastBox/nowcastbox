r"""News decomposition of nowcast revisions (Bańbura & Modugno, 2014).

Let :math:`\Omega_v` and :math:`\Omega_{v+1}` be two vintages and :math:`y_\tau` the
target. With fixed parameters the change of the nowcast splits exactly into

.. math::

    \mathbb{E}[y_\tau \mid \Omega_{v+1}] - \mathbb{E}[y_\tau \mid \Omega_v]
    = \underbrace{\mathbb{E}[y_\tau \mid \Omega^\ast] - \mathbb{E}[y_\tau \mid \Omega_v]}
      _{\text{data revisions}}
    + \underbrace{\sum_{j \in I_{v+1}} b_j \bigl(x_j - \mathbb{E}[x_j \mid \Omega^\ast]\bigr)}
      _{\text{news}},

where :math:`\Omega^\ast` holds the *old* observation pattern with the *new* (revised)
values, :math:`I_{v+1}` the newly released cells, :math:`x_j - \mathbb{E}[x_j \mid
\Omega^\ast]` the *news* of release :math:`j` and :math:`b_j` its weight,

.. math::

    b = \operatorname{Cov}(y_\tau, I \mid \Omega^\ast)\,
        \operatorname{Var}(I \mid \Omega^\ast)^{-1},

the Kalman-smoother projection coefficients of the target on the innovations
(Bańbura & Modugno, 2014, eq. 12). Because the smoother is linear in the observations
for a given missing-data pattern, :math:`b_j` equals the derivative
:math:`\partial\, \mathbb{E}[y_\tau \mid \Omega_{v+1}] / \partial x_j`; it is computed
exactly by smoothing unit vectors on the pattern of :math:`\Omega_{v+1}` (one smoother
pass per release, no state augmentation). When the new vintage comes with re-estimated
parameters, a third term (*re-estimation effect*) closes the identity

.. math::

    \hat y^{new} = \hat y^{old} + \text{revisions} + \sum_j \text{impact}_j
                   + \text{re-estimation}.

References
----------
Bańbura, M., & Modugno, M. (2014). Maximum likelihood estimation of factor models on
datasets with arbitrary pattern of missing data. *Journal of Applied Econometrics*,
29(1), 133-160 (section 2.3, "news").

Bańbura, M., Giannone, D., Modugno, M., & Reichlin, L. (2013). Now-casting and the
real-time data flow. In *Handbook of Economic Forecasting* 2A, 195-237.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from nowcastbox._logging import get_logger
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.results import NowcastResults
from nowcastbox.news._model import (
    LinearNowcastModel,
    linear_model,
    position,
    to_frame,
    to_period,
)
from nowcastbox.news.plotting import (
    plot_news_object,
    register_news_plot,
    waterfall_chart,
)
from nowcastbox.news.revisions import compare_vintages

__all__ = [
    "GROUPINGS",
    "NewsResults",
    "news_decomposition",
    "news_weights",
    "plot_waterfall_default",
]

logger = get_logger(__name__)

FloatArray = NDArray[np.float64]

GROUPINGS = ("series", "block", "category")
_CATEGORY_ORDER = ("hard", "soft", "financial", "other", "uncategorized")
_RELEASE_COLUMNS = [
    "series",
    "reference_period",
    "slot",
    "actual",
    "expected",
    "news",
    "weight",
    "impact",
    "category",
    "block",
]


# ====================================================================== results
@dataclass(frozen=True, kw_only=True, eq=False)
class NewsResults:
    r"""Decomposition of the revision of a nowcast between two vintages.

    Parameters
    ----------
    target : str
        Target series.
    target_period : pandas.Period
        Target period (native frequency).
    model_name : str
        Model of the results.
    old_nowcast : float
        :math:`\mathbb{E}[y_\tau \mid \Omega_v]` (original units).
    new_nowcast : float
        Nowcast with the new vintage (and new parameters, if given).
    revisions_effect : float
        Effect of revised (and removed) values of previously released data.
    reestimation_effect : float
        Effect of the change of parameters (0 when they are fixed).
    releases : pandas.DataFrame
        One row per new release: ``series``, ``reference_period`` (native period),
        ``slot`` (base period), ``actual``, ``expected`` (:math:`\mathbb{E}[x_j \mid
        \Omega^\ast]`), ``news`` (``actual - expected``, units of the series), ``weight``
        (impact per unit of news, target units per series unit), ``impact``
        (``weight * news``, target units), ``category`` and ``block``.
    revisions : pandas.DataFrame
        One row per revised series: ``n_revised``, ``impact``, ``category``, ``block``
        (index: series). Their sum plus :attr:`removal_effect` is the revisions effect up
        to rounding.
    removal_effect : float, default 0.0
        Effect of values present in the old vintage but missing in the new one.
    residual : float, default 0.0
        Numerical residual ``new - old - revisions - news - re-estimation`` (rounding
        error of the linear algebra, ~1e-12).
    info : dict
        Diagnostics (``n_released``, ``n_revised``, ``n_removed``, the nowcast with
        fixed parameters, ...).

    Examples
    --------
    >>> news = news_decomposition(res, old, new, "2020Q2")  # doctest: +SKIP
    >>> news.summary()  # doctest: +SKIP
    >>> news.to_frame(by="category")  # doctest: +SKIP
    """

    target: str
    target_period: pd.Period
    model_name: str
    old_nowcast: float
    new_nowcast: float
    revisions_effect: float
    reestimation_effect: float
    releases: pd.DataFrame
    revisions: pd.DataFrame
    removal_effect: float = 0.0
    residual: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ totals
    @property
    def news_effect(self) -> float:
        """Sum of the impacts of the new releases."""
        return float(self.releases["impact"].sum())

    @property
    def total_change(self) -> float:
        """``new_nowcast - old_nowcast``."""
        return self.new_nowcast - self.old_nowcast

    @property
    def n_releases(self) -> int:
        """Number of new releases."""
        return len(self.releases)

    def check_identity(self, atol: float = 1e-8) -> bool:
        """Whether old + revisions + news + re-estimation reproduces the new nowcast.

        Parameters
        ----------
        atol : float, default 1e-8
            Absolute tolerance.

        Returns
        -------
        bool
            True when the decomposition closes within ``atol``.

        Examples
        --------
        >>> news.check_identity()  # doctest: +SKIP
        True
        """
        total = self.old_nowcast + self.revisions_effect + self.news_effect
        return bool(abs(total + self.reestimation_effect - self.new_nowcast) <= atol)

    # ------------------------------------------------------------------ frames
    def _group_column(self, by: str) -> str:
        if by not in GROUPINGS:
            raise ValueError(f"by must be one of {GROUPINGS} or 'release', got {by!r}.")
        return by

    def _order(self, by: str, labels: list[str]) -> list[str]:
        if by == "series":
            order = list(self.info.get("series", []))
        elif by == "category":
            order = list(_CATEGORY_ORDER)
        else:
            order = list(dict.fromkeys(self.info.get("block_labels", [])))
        known = [g for g in order if g in labels]
        return known + sorted(g for g in labels if g not in known)

    def to_frame(self, by: str = "series") -> pd.DataFrame:
        """News and revision effects aggregated by series, block or category.

        Parameters
        ----------
        by : {"series", "block", "category", "release"}, default "series"
            Grouping. ``"release"`` returns a copy of :attr:`releases`.

        Returns
        -------
        pandas.DataFrame
            Index = groups with any release or revision; columns ``news`` (sum of the
            impacts of new releases), ``revisions`` (effect of revised values) and
            ``total``. The column sums plus :attr:`removal_effect` and
            :attr:`reestimation_effect` add up to :attr:`total_change`.

        Raises
        ------
        ValueError
            On an unknown grouping.

        Examples
        --------
        >>> news.to_frame(by="category")  # doctest: +SKIP
        """
        if by == "release":
            return self.releases.copy()
        column = self._group_column(by)
        rel = self.releases
        news = rel.groupby(rel[column], sort=False)["impact"].sum()
        revs = self.revisions.reset_index()
        rev = revs.groupby(revs[column], sort=False)["impact"].sum()
        labels = [str(g) for g in news.index.union(rev.index)]
        index = pd.Index(self._order(by, labels), name=by)
        frame = pd.DataFrame(
            {
                "news": news.reindex(index, fill_value=0.0),
                "revisions": rev.reindex(index, fill_value=0.0),
            },
            index=index,
        ).astype(float)
        frame["total"] = frame["news"] + frame["revisions"]
        return frame

    def waterfall(self, by: str = "category") -> pd.Series:
        """Steps from the old to the new nowcast.

        Parameters
        ----------
        by : {"series", "block", "category"}, default "category"
            Grouping of the news.

        Returns
        -------
        pandas.Series
            ``"old nowcast"`` (level), the news impact of every group, ``"revisions"``
            (when there are revised or removed values), ``"re-estimation"`` (when
            parameters changed) and ``"new nowcast"`` (level). The increments add up
            to the change of the nowcast (up to :attr:`residual`).

        Raises
        ------
        ValueError
            On an unknown grouping.

        Examples
        --------
        >>> news.waterfall("block")  # doctest: +SKIP
        """
        self._group_column(by)
        rel = self.releases
        news = rel.groupby(rel[by], sort=False)["impact"].sum()
        order = self._order(by, [str(g) for g in news.index])
        steps: dict[str, float] = {"old nowcast": self.old_nowcast}
        steps.update({f"news: {g}": float(news[g]) for g in order})
        if not self.revisions.empty or self.removal_effect != 0.0:
            steps["revisions"] = self.revisions_effect
        if self.info.get("reestimated", False):
            steps["re-estimation"] = self.reestimation_effect
        steps["new nowcast"] = self.new_nowcast
        return pd.Series(steps, name=self.target, dtype=float)

    def top_releases(self, n: int = 10) -> pd.DataFrame:
        """Releases with the largest absolute impact.

        Parameters
        ----------
        n : int, default 10
            Number of rows.

        Returns
        -------
        pandas.DataFrame
            Subset of :attr:`releases` sorted by ``|impact|``.

        Examples
        --------
        >>> news.top_releases(5)  # doctest: +SKIP
        """
        order = self.releases["impact"].abs().sort_values(ascending=False, kind="stable")
        return self.releases.loc[order.index[:n]].copy()

    # ------------------------------------------------------------------ text
    def summary(self, by: str = "category", n_releases: int = 10) -> str:
        """Formatted text summary.

        Parameters
        ----------
        by : {"series", "block", "category"}, default "category"
            Grouping of the decomposition table.
        n_releases : int, default 10
            Number of releases (largest absolute impact) listed.

        Returns
        -------
        str
            Multi-line summary.

        Examples
        --------
        >>> print(news.summary())  # doctest: +SKIP
        """
        width = 78
        lines = [
            "=" * width,
            f"News decomposition: {self.target} {self.target_period} ({self.model_name})",
            "=" * width,
            f"  {'Old nowcast':<26}{self.old_nowcast:>14.6f}",
            f"  {'Data revisions':<26}{self.revisions_effect:>14.6f}",
            f"  {'News (new releases)':<26}{self.news_effect:>14.6f}",
            f"  {'Re-estimation':<26}{self.reestimation_effect:>14.6f}",
            f"  {'New nowcast':<26}{self.new_nowcast:>14.6f}",
            f"  {'Releases / revised':<26}{self.n_releases:>7d} / {len(self.revisions):d}",
            "-" * width,
            f"  {by:<26}{'news':>14}{'revisions':>14}{'total':>14}",
        ]
        for label, row in self.to_frame(by).iterrows():
            lines.append(
                f"  {str(label)[:25]:<26}{row['news']:>14.6f}{row['revisions']:>14.6f}"
                f"{row['total']:>14.6f}"
            )
        if self.n_releases:
            lines += [
                "-" * width,
                f"  {'release':<26}{'actual':>12}{'expected':>12}{'weight':>12}{'impact':>12}",
            ]
            for _, row in self.top_releases(n_releases).iterrows():
                label = f"{row['series']} {row['reference_period']}"
                lines.append(
                    f"  {label[:25]:<26}{row['actual']:>12.4f}{row['expected']:>12.4f}"
                    f"{row['weight']:>12.4f}{row['impact']:>12.4f}"
                )
        lines.append("=" * width)
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Short representation."""
        return (
            f"NewsResults(target={self.target!r}, period={self.target_period}, "
            f"old={self.old_nowcast:.4f}, new={self.new_nowcast:.4f}, "
            f"releases={self.n_releases})"
        )

    def plot(self, kind: str = "waterfall", **kwargs: Any) -> Any:
        """Plot the decomposition through the news plot registry.

        Parameters
        ----------
        kind : str, default "waterfall"
            Plot kind (``"waterfall"``; see :func:`available_news_plots`).
        **kwargs
            Passed to the plotting function (``by``, ``ax``, ``title``...).

        Returns
        -------
        object
            Figure.

        Examples
        --------
        >>> news.plot("waterfall", by="block")  # doctest: +SKIP
        """
        return plot_news_object(self, kind, **kwargs)


@register_news_plot("waterfall", NewsResults)
def plot_waterfall_default(news: NewsResults, by: str = "category", **kwargs: Any) -> Any:
    """Default Matplotlib waterfall of :meth:`NewsResults.waterfall` (``kind="waterfall"``).

    :mod:`nowcastbox.visualization` wraps it so that ``news.plot("waterfall")`` keeps
    this chart and ``backend="plotly"`` switches to the themed charts.

    Parameters
    ----------
    news : NewsResults
        News decomposition.
    by : str, default "category"
        Grouping of the news bars (``"series"``, ``"block"``, ``"category"``).
    **kwargs
        Passed to the waterfall chart (``ax``, ``title``, ``figsize``...).

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Examples
    --------
    >>> fig = plot_waterfall_default(news, by="block")  # doctest: +SKIP
    """
    kwargs.setdefault("title", f"{news.target} {news.target_period}: nowcast revision")
    return waterfall_chart(news.waterfall(by), **kwargs)


# ====================================================================== engine
def _default_period(lin: LinearNowcastModel, frame: pd.DataFrame) -> pd.Period:
    """First target period after the last observation of the target in ``frame``."""
    freq = lin.target_frequency.pandas_freq
    if lin.target in frame.columns:
        observed = frame[lin.target].dropna()
        if len(observed):
            return observed.index[-1].asfreq(freq) + 1
    return frame.index[-1].asfreq(freq)


def _unit_weights(
    lin: LinearNowcastModel, pattern: NDArray[np.bool_], cells: NDArray[np.intp], pos: int
) -> FloatArray:
    """Derivatives of the standardised nowcast w.r.t. the observations in ``cells``.

    One batched smoother pass on ``pattern``: a zero data set (intercept) plus one unit
    data set per cell; by linearity the differences are the weights.
    """
    n_cells = len(cells)
    data = np.zeros((*pattern.shape, n_cells + 1))
    data[cells[:, 0], cells[:, 1], np.arange(1, n_cells + 1)] = 1.0
    values = lin.evaluate_many(pattern, data, pos)
    return values[1:] - values[0]


def news_weights(
    results: NowcastResults, data: object, target_period: object, cells: Any = None
) -> pd.DataFrame:
    r"""Weights of observations on the nowcast for a given information set.

    The nowcast is :math:`\hat y_\tau = c + \sum_k w_k x_k` over the observed cells
    :math:`x_k` of ``data`` (standardised); this returns :math:`w_k` in target units per
    unit of each series (``scale * w_k / std_k``). For newly released cells these are
    the news weights :math:`b_j` of Bańbura & Modugno (2014).

    Parameters
    ----------
    results : NowcastResults
        :class:`~nowcastbox.models.MixedFreqDFMResults` or ``TwoStepResults``.
    data : MixedFrequencyData or pandas.DataFrame
        Information set (its observation pattern).
    target_period : period-like
        Target period.
    cells : sequence of (series, base period), optional
        Observed cells whose weights are wanted (default: every observed cell of the
        last 12 base periods of the data).

    Returns
    -------
    pandas.DataFrame
        Columns ``series``, ``slot`` and ``weight``.

    Raises
    ------
    NowcastDataError
        If a requested cell is not observed in ``data``.

    Examples
    --------
    >>> news_weights(res, vintage, "2020Q2")  # doctest: +SKIP
    """
    lin = linear_model(results)
    frame = to_frame(data)
    slot = lin.target_slot(target_period)
    grid = lin.grid(slot, frame)
    values = lin.standardize(frame, grid)
    pattern = ~np.isnan(values)
    if cells is None:
        start = max(len(grid) - 12, 0)
        mask = np.zeros_like(pattern)
        mask[start:] = pattern[start:]
        idx = np.argwhere(mask)
    else:
        pairs = []
        for name, period in cells:
            t = position(grid, pd.Period(period, freq=grid.freqstr))
            i = lin.series.index(name)
            if t < 0 or not pattern[t, i]:
                raise NowcastDataError(f"Cell ({name!r}, {period}) is not observed in the data.")
            pairs.append((t, i))
        idx = np.array(pairs, dtype=np.intp).reshape(-1, 2)
    w = _unit_weights(lin, pattern, idx, position(grid, slot))
    return pd.DataFrame(
        {
            "series": [lin.series[i] for _, i in idx],
            "slot": [grid[t] for t, _ in idx],
            "weight": lin.scale * w / lin.std[idx[:, 1]],
        }
    )


@dataclass(frozen=True)
class _Panels:
    grid: pd.PeriodIndex
    position: int
    old: FloatArray
    new: FloatArray


def _panels(
    lin: LinearNowcastModel, old_data: object, new_data: object, target_period: object
) -> tuple[_Panels, pd.Period]:
    old_frame = to_frame(old_data)
    new_frame = to_frame(new_data)
    period = (
        _default_period(lin, old_frame)
        if target_period is None
        else to_period(target_period, lin.target_frequency)
    )
    slot = lin.target_slot(period)
    grid = lin.grid(slot, old_frame, new_frame)
    panels = _Panels(
        grid=grid,
        position=position(grid, slot),
        old=lin.standardize(old_frame, grid),
        new=lin.standardize(new_frame, grid),
    )
    return panels, period


def _revision_table(
    lin: LinearNowcastModel, p: _Panels, revised: NDArray[np.bool_]
) -> pd.DataFrame:
    """Effect of the revisions of each series (old pattern, linearity of the smoother)."""
    columns = np.flatnonzero(revised.any(axis=0))
    pattern = ~np.isnan(p.old)
    data = np.repeat(np.nan_to_num(p.old)[:, :, None], len(columns) + 1, axis=2)
    for k, i in enumerate(columns, start=1):
        cells = revised[:, i]
        data[cells, i, k] = p.new[cells, i]
    values = lin.evaluate_many(pattern, data, p.position)
    rows = [
        {
            "series": lin.series[i],
            "n_revised": int(revised[:, i].sum()),
            "impact": lin.scale * float(values[k] - values[0]),
            "category": lin.categories[lin.series[i]],
            "block": lin.blocks[lin.series[i]],
        }
        for k, i in enumerate(columns, start=1)
    ]
    frame = pd.DataFrame(rows, columns=["series", "n_revised", "impact", "category", "block"])
    return frame.set_index("series")


def _release_table(
    lin: LinearNowcastModel,
    p: _Panels,
    cells: NDArray[np.intp],
    expected: FloatArray,
    weights: FloatArray,
) -> pd.DataFrame:
    rows = []
    for (t, i), e, b in zip(cells, expected, weights, strict=True):
        name = lin.series[i]
        std = float(lin.std[i])
        slot = p.grid[t]
        news = (p.new[t, i] - e) * std
        rows.append(
            {
                "series": name,
                "reference_period": slot.asfreq(lin.frequencies[name].pandas_freq),
                "slot": slot,
                "actual": float(lin.mean[i] + std * p.new[t, i]),
                "expected": float(lin.mean[i] + std * e),
                "news": float(news),
                "weight": float(lin.scale * b / std),
                "impact": float(lin.scale * b * (p.new[t, i] - e)),
                "category": lin.categories[name],
                "block": lin.blocks[name],
            }
        )
    frame = pd.DataFrame(rows, columns=_RELEASE_COLUMNS)
    if frame.empty:
        return frame.astype(
            dict.fromkeys(("actual", "expected", "news", "weight", "impact"), float)
        )
    order = {name: k for k, name in enumerate(lin.series)}
    frame["_o"] = frame["series"].map(order)
    frame = frame.sort_values(["_o", "slot"], kind="stable").drop(columns="_o")
    return frame.reset_index(drop=True)


def _reestimation(
    lin: LinearNowcastModel,
    new_results: NowcastResults | None,
    new_data: object,
    period: pd.Period,
    fixed: float,
    categories: Mapping[str, object] | None,
) -> tuple[float, float, bool]:
    """(new nowcast, re-estimation effect, whether parameters changed)."""
    if new_results is None:
        return fixed, 0.0, False
    lin_new = linear_model(new_results, categories)
    if lin_new.target != lin.target:
        raise ValueError(
            f"new_results has target {lin_new.target!r}, the old results {lin.target!r}."
        )
    if lin_new.same_parameters(lin):
        return fixed, 0.0, False
    value = lin_new.nowcast(to_frame(new_data), period)
    return value, value - fixed, True


def news_decomposition(
    results: NowcastResults,
    old_data: object,
    new_data: object,
    target_period: object = None,
    *,
    new_results: NowcastResults | None = None,
    categories: Mapping[str, object] | None = None,
) -> NewsResults:
    r"""Decompose the revision of a nowcast between two vintages into news and revisions.

    Parameters
    ----------
    results : NowcastResults
        Fitted :class:`~nowcastbox.models.MixedFreqDFM` or
        :class:`~nowcastbox.models.TwoStepDFM` results; its parameters define the
        expectations under both information sets. With ``aggregate="variables"`` the
        releases are the filtered predictor values that the new vintage completes.
    old_data, new_data : MixedFrequencyData or pandas.DataFrame
        Old and new vintages (base-frequency PeriodIndex) containing every model series.
        Values may be revised between them; cells may also disappear (with a warning).
    target_period : period-like, optional
        Target period (e.g. ``"2020Q2"``). Default: the first target period after the
        last target observation in ``old_data``.
    new_results : NowcastResults, optional
        Results re-estimated on the new vintage. When their parameters differ, the
        difference between the nowcast with new and old parameters (both on
        ``new_data``) is reported as the re-estimation effect.
    categories : mapping of str to category, optional
        Override of the series categories used for grouping (default: the
        ``SeriesMetadata.category`` of the estimation panel; ``"uncategorized"`` when
        unset).

    Returns
    -------
    NewsResults
        Decomposition with ``old_nowcast + revisions_effect + news_effect +
        reestimation_effect == new_nowcast`` (up to rounding, ~1e-12).

    Raises
    ------
    TypeError
        If ``results`` come from an unsupported model.
    NowcastDataError
        If a vintage lacks model series or has another base frequency.
    ValueError
        If the target period is before the model grid or ``new_results`` has another
        target.

    Warns
    -----
    DataQualityWarning
        When values of the old vintage are missing in the new one.

    Notes
    -----
    The weights :math:`b_j` are computed exactly from the linearity of the Kalman
    smoother in the data (one smoother pass per release); they coincide with
    :math:`\operatorname{Cov}(y_\tau, I \mid \Omega^\ast)\operatorname{Var}(I \mid
    \Omega^\ast)^{-1}` of Bańbura & Modugno (2014). Series outside the model (e.g. the
    target of a two-step model, which enters only through the bridge equation) are
    ignored.

    References
    ----------
    Bańbura, M., & Modugno, M. (2014). *Journal of Applied Econometrics*, 29(1), 133-160.

    Examples
    --------
    >>> import nowcastbox as nb
    >>> from nowcastbox.news import news_decomposition
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> data = simulate_two_step_example(random_state=0)
    >>> res = nb.MixedFreqDFM(n_factors=1, max_iter=20).fit(data, "gdp")
    >>> old = data.truncate(end=data.index[-3])
    >>> news = news_decomposition(res, old, data, "2014Q4")
    >>> news.check_identity()
    True
    """
    lin = linear_model(results, categories)
    p, period = _panels(lin, old_data, new_data, target_period)
    diff = compare_vintages(p.old, p.new)
    if diff.n_removed:
        warnings.warn(
            f"{diff.n_removed} values of the old vintage are missing in the new one; their "
            "removal is counted with the data revisions.",
            DataQualityWarning,
            stacklevel=2,
        )
    f_old = lin.evaluate(p.old, p.position)
    revised = p.old.copy()
    revised[diff.revised] = p.new[diff.revised]
    f_rev = lin.evaluate(revised, p.position) if diff.n_revised else f_old
    mid = revised.copy()
    mid[diff.removed] = np.nan
    smoothed_mid = lin.smooth(mid)
    f_mid = lin.functional(smoothed_mid, p.position)

    cells = np.argwhere(diff.released)
    signal = smoothed_mid.smoothed_signal()
    expected = signal[cells[:, 0], cells[:, 1]] if len(cells) else np.empty(0)
    pattern = ~np.isnan(p.new)
    weights = _unit_weights(lin, pattern, cells, p.position)
    f_new = lin.evaluate(p.new, p.position) if len(cells) else f_mid
    news_std = float(weights @ (p.new[cells[:, 0], cells[:, 1]] - expected)) if len(cells) else 0.0

    fixed_new = lin.to_original(f_new)
    new_nowcast, reestimation, reestimated = _reestimation(
        lin, new_results, new_data, period, fixed_new, categories
    )
    releases = _release_table(lin, p, cells, expected, weights)
    info: dict[str, Any] = {
        "n_released": diff.n_released,
        "n_revised": diff.n_revised,
        "n_removed": diff.n_removed,
        "fixed_parameter_nowcast": fixed_new,
        "reestimated": reestimated,
        "series": list(lin.series),
        "block_labels": [lin.blocks[s] for s in lin.series],
        "grid": p.grid,
    }
    logger.debug(
        "news %s %s: %d releases, %d revised", lin.target, period, len(cells), diff.n_revised
    )
    return NewsResults(
        target=lin.target,
        target_period=period,
        model_name=lin.model_name,
        old_nowcast=lin.to_original(f_old),
        new_nowcast=new_nowcast,
        revisions_effect=lin.scale * (f_mid - f_old),
        reestimation_effect=reestimation,
        releases=releases,
        revisions=_revision_table(lin, p, diff.revised),
        removal_effect=lin.scale * (f_mid - f_rev),
        residual=lin.scale * (f_new - f_mid - news_std),
        info=info,
    )

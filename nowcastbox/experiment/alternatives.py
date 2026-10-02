r"""Nowcasts of alternative models that leave out one or two groups of indicators.

The *range of alternative nowcasts* of Linzenich & Meunier (2024, ECB WP 3004, §3.5)
measures how much the headline nowcast depends on specific groups of variables: the
model is estimated again (or only re-run with the base parameters) after removing every
group, and every pair of groups, of predictors, and the dispersion of the resulting
nowcasts is reported next to the base nowcast. A wide range signals that the nowcast
hinges on a few indicators; a narrow one that the information is spread over the panel,
in the spirit of the forecast-combination literature (Granger & Jeon, 2004;
Timmermann, 2006).

Groups come from the panel metadata (``by="category"``: hard/soft/financial/other;
``by="block"``: factor blocks) or from an explicit mapping. Two modes are available:

* ``refit=True`` (default) - every alternative model is a fresh clone of the base
  estimator fitted on the panel without the dropped series (runs in parallel with
  :mod:`joblib` when ``n_jobs != 1``, as :class:`~nowcastbox.evaluation.PseudoRealTimeBacktest`);
* ``refit=False`` - the parameters of the base model are kept and the Kalman smoother is
  re-run with the dropped series treated as missing (exact and fast; for the state-space
  models :class:`~nowcastbox.models.MixedFreqDFM` and :class:`~nowcastbox.models.TwoStepDFM`).

Examples
--------
>>> import warnings
>>> from nowcastbox.models import TwoStepDFM
>>> from nowcastbox.models.two_step import simulate_two_step_example
>>> data = simulate_two_step_example(n_series=6, random_state=0)
>>> groups = {"real": ["x1", "x2"], "soft": ["x3", "x4"], "fin": ["x5", "x6"]}
>>> alt = alternative_models(TwoStepDFM(n_factors=1), data, "gdp", by=groups, refit=False)
>>> alt.n_models
6
>>> alt.range().columns[:4].tolist()
['base', 'min', 'max', 'median']
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import MixedFrequencyData, as_mixed_frequency_data
from nowcastbox.core.formula import resolve_target
from nowcastbox.core.results import NowcastResults

if TYPE_CHECKING:
    from nowcastbox.visualization.themes import Theme

__all__ = [
    "GROUPINGS",
    "AlternativeNowcasts",
    "alternative_models",
    "combination_label",
    "resolve_groups",
]

logger = get_logger(__name__)

GROUPINGS: tuple[str, ...] = ("category", "block")
"""Metadata groupings accepted by ``by`` (besides an explicit mapping)."""

BASE_LABEL = "base"
"""Row label of the base model in :meth:`AlternativeNowcasts.table`."""


# ---------------------------------------------------------------------------- groups
def resolve_groups(
    data: MixedFrequencyData,
    target: str,
    by: str | Mapping[str, Any] = "category",
    groups: Sequence[str] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Groups of predictors (``{group: series}``) used to build alternative models.

    Parameters
    ----------
    data : MixedFrequencyData
        Panel (target and predictors).
    target : str
        Target series; it never belongs to a group.
    by : {"category", "block"} or mapping, default "category"
        ``"category"``: the :attr:`SeriesMetadata.category` of each series;
        ``"block"``: the factor blocks (a series in several blocks belongs to each of
        them); a mapping ``{group: [series, ...]}`` or ``{series: group}`` (read as
        ``{group: series}`` when every value is a predictor name).
        Series without a group are never dropped.
    groups : sequence of str, optional
        Restrict the droppable groups to these names (in this order).

    Returns
    -------
    dict of str to tuple of str
        Members of each group (panel order), in order of first appearance.

    Raises
    ------
    ValueError
        If ``by`` is unknown, no series has a group, a mapping names series that are
        not predictors of the panel, or ``groups`` names unknown groups.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> idx = pd.period_range("2020-01", periods=3, freq="M")
    >>> frame = pd.DataFrame({"y": [1.0, 2, 3], "a": [1.0, 2, 4], "b": [2.0, 1, 0]}, idx)
    >>> mfd = MixedFrequencyData(frame, "M", categories={"a": "hard", "b": "soft"})
    >>> resolve_groups(mfd, "y")
    {'hard': ('a',), 'soft': ('b',)}
    """
    predictors = [c for c in data.columns if c != target]
    if isinstance(by, Mapping):
        found = _groups_from_mapping(by, predictors)
    elif by == "category":
        found = _groups_from_metadata(data, predictors, _category_of)
    elif by == "block":
        found = _groups_from_metadata(data, predictors, lambda d, c: d.metadata[c].blocks)
    else:
        raise ValueError(f"by must be 'category', 'block' or a mapping; got {by!r}.")
    if not found:
        raise ValueError(f"No predictor of the panel has a group (by={_by_name(by)!r}).")
    if groups is None:
        return found
    unknown = [g for g in groups if g not in found]
    if unknown:
        raise ValueError(f"Unknown groups {unknown}; available: {list(found)}.")
    return {g: found[g] for g in dict.fromkeys(groups)}


def _category_of(data: MixedFrequencyData, column: str) -> tuple[str, ...]:
    category = data.metadata[column].category
    return () if category is None else (category.value,)


def _groups_from_metadata(
    data: MixedFrequencyData,
    predictors: list[str],
    labels: Callable[[MixedFrequencyData, str], Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for column in predictors:
        for label in labels(data, column):
            out.setdefault(str(label), []).append(column)
    return {k: tuple(v) for k, v in out.items()}


def _groups_from_mapping(
    mapping: Mapping[str, Any], predictors: list[str]
) -> dict[str, tuple[str, ...]]:
    values = list(mapping.values())
    if values and all(isinstance(v, str) for v in values) and not set(values) <= set(predictors):
        # {series: group}; {group: "series"} (every value a predictor) is kept as is
        inverted: dict[str, list[str]] = {}
        for series, group in mapping.items():
            inverted.setdefault(str(group), []).append(str(series))
        mapping = inverted
    out: dict[str, tuple[str, ...]] = {}
    unknown: list[str] = []
    for group, members in mapping.items():
        names = [members] if isinstance(members, str) else [str(m) for m in members]
        unknown += [n for n in names if n not in predictors]
        chosen = tuple(c for c in predictors if c in names)
        if chosen:
            out[str(group)] = chosen
    if unknown:
        raise ValueError(f"The group mapping names series that are not predictors: {unknown}.")
    return out


def _by_name(by: str | Mapping[str, Any]) -> str:
    return "mapping" if isinstance(by, Mapping) else str(by)


def _drop_sizes(drop: int | Iterable[int], n_groups: int) -> tuple[int, ...]:
    sizes = (drop,) if isinstance(drop, int | np.integer) else tuple(drop)
    if not sizes:
        raise ValueError("drop must name at least one number of groups to remove.")
    for k in sizes:
        if isinstance(k, bool) or not isinstance(k, int | np.integer) or k < 1:
            raise ValueError(f"drop must contain integers >= 1; got {k!r}.")
    valid = sorted({int(k) for k in sizes if k <= n_groups})
    if not valid:
        raise ValueError(f"Cannot drop {min(sizes)} groups: only {n_groups} groups are available.")
    if len(valid) < len(set(sizes)):
        logger.info("drop sizes above the %d available groups ignored", n_groups)
    return tuple(valid)


def _combinations(
    groups: Mapping[str, tuple[str, ...]], sizes: tuple[int, ...], predictors: Sequence[str]
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """``(kept, skipped)`` combinations; skipped ones would leave no predictor."""
    kept: list[tuple[str, ...]] = []
    skipped: list[tuple[str, ...]] = []
    for k in sizes:
        for combo in itertools.combinations(groups, k):
            dropped = {s for g in combo for s in groups[g]}
            if any(p not in dropped for p in predictors):
                kept.append(combo)
            else:
                skipped.append(combo)
    if skipped:
        logger.info("alternative models skipped (no predictor left): %s", skipped)
    if not kept:
        raise ValueError("Every combination of dropped groups removes all the predictors.")
    return kept, skipped


def combination_label(combo: Sequence[str]) -> str:
    """Row label of an alternative model (``"-hard -soft"``).

    Parameters
    ----------
    combo : sequence of str
        Dropped groups.

    Returns
    -------
    str
        Each group prefixed with ``-`` (empty sequence: ``"base"``).

    Examples
    --------
    >>> combination_label(("hard", "soft"))
    '-hard -soft'
    """
    return " ".join(f"-{g}" for g in combo) if combo else BASE_LABEL


# ---------------------------------------------------------------------------- results
@dataclass(frozen=True)
class AlternativeNowcasts:
    """Nowcasts of the base model and of the alternatives without 1-2 groups.

    Parameters
    ----------
    target : str
        Target series.
    by : str
        Grouping (``"category"``, ``"block"`` or ``"mapping"``).
    refit : bool
        Whether the alternatives were re-estimated (``False``: base parameters).
    groups : dict of str to tuple of str
        Members of each droppable group.
    combinations : tuple of tuple of str
        Dropped groups of each alternative model, in the order of :attr:`nowcasts`.
    nowcasts : pandas.DataFrame
        Rows: ``"base"`` then one row per alternative (:func:`combination_label`);
        columns: target periods (``PeriodIndex``).
    base : NowcastResults
        Results of the base model.
    skipped : tuple of tuple of str
        Combinations not estimated because they removed every predictor.
    info : dict
        Extras (model name, seconds spent).

    Examples
    --------
    >>> from nowcastbox.models import TwoStepDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> data = simulate_two_step_example(n_series=6, random_state=0)
    >>> groups = {"real": ["x1", "x2"], "soft": ["x3", "x4"], "fin": ["x5", "x6"]}
    >>> alt = alternative_models(TwoStepDFM(n_factors=1), data, "gdp", by=groups)
    >>> alt.n_models, alt.combinations[:2]
    (6, (('real',), ('soft',)))
    """

    target: str
    by: str
    refit: bool
    groups: dict[str, tuple[str, ...]]
    combinations: tuple[tuple[str, ...], ...]
    nowcasts: pd.DataFrame
    base: NowcastResults
    skipped: tuple[tuple[str, ...], ...] = ()
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def periods(self) -> pd.PeriodIndex:
        """Target periods of the nowcasts."""
        index = self.nowcasts.columns
        assert isinstance(index, pd.PeriodIndex)  # noqa: S101
        return index

    @property
    def n_models(self) -> int:
        """Number of alternative models (the base model excluded)."""
        return len(self.combinations)

    @property
    def alternatives(self) -> pd.DataFrame:
        """Nowcasts of the alternative models only (copy, without the base row)."""
        return self.nowcasts.iloc[1:].copy()

    def table(self, deviation: bool = False) -> pd.DataFrame:
        """Nowcast of every model (base and alternatives) per target period.

        Parameters
        ----------
        deviation : bool, default False
            Report the difference to the base nowcast instead of the level.

        Returns
        -------
        pandas.DataFrame
            Index ``model`` (``"base"``, ``"-hard"``, ``"-hard -soft"`` ...); columns
            ``dropped`` (comma-separated groups), ``n_dropped`` (groups),
            ``n_series_dropped`` and one column per target period (``str``).

        Examples
        --------
        >>> from nowcastbox.models import TwoStepDFM
        >>> from nowcastbox.models.two_step import simulate_two_step_example
        >>> data = simulate_two_step_example(n_series=6, random_state=0)
        >>> groups = {"real": ["x1", "x2"], "soft": ["x3", "x4"], "fin": ["x5", "x6"]}
        >>> alt = alternative_models(TwoStepDFM(n_factors=1), data, "gdp", by=groups)
        >>> row = alt.table().loc["-real -soft"]
        >>> row["dropped"], int(row["n_series_dropped"])
        ('real, soft', 4)
        """
        values = self.nowcasts.copy()
        if deviation:
            values = values - values.iloc[0]
        values.columns = [str(p) for p in self.periods]
        combos: list[tuple[str, ...]] = [(), *self.combinations]
        meta = pd.DataFrame(
            {
                "dropped": [", ".join(c) for c in combos],
                "n_dropped": [len(c) for c in combos],
                "n_series_dropped": [len(self.dropped_series(c)) for c in combos],
            },
            index=values.index,
        )
        out = pd.concat([meta, values], axis=1)
        out.index.name = "model"
        return out

    def dropped_series(self, combo: Sequence[str]) -> tuple[str, ...]:
        """Series removed by a combination of groups.

        Parameters
        ----------
        combo : sequence of str
            Group names.

        Returns
        -------
        tuple of str
            Union of the members of the groups (group order).

        Raises
        ------
        KeyError
            If a group is unknown.

        Examples
        --------
        >>> from nowcastbox.models import TwoStepDFM
        >>> from nowcastbox.models.two_step import simulate_two_step_example
        >>> data = simulate_two_step_example(n_series=6, random_state=0)
        >>> groups = {"real": ["x1", "x2"], "soft": ["x3", "x4"], "fin": ["x5", "x6"]}
        >>> alt = alternative_models(TwoStepDFM(n_factors=1), data, "gdp", by=groups)
        >>> alt.dropped_series(("real", "fin"))
        ('x1', 'x2', 'x5', 'x6')
        """
        out: dict[str, None] = {}
        for group in combo:
            if group not in self.groups:
                raise KeyError(f"Unknown group {group!r}; groups: {list(self.groups)}.")
            out.update(dict.fromkeys(self.groups[group]))
        return tuple(out)

    def range(self) -> pd.DataFrame:
        """Dispersion of the alternative nowcasts per target period.

        Returns
        -------
        pandas.DataFrame
            Indexed by target period; columns ``base``, ``min``, ``max``, ``median``,
            ``mean``, ``spread`` (``max - min``), ``min_model``, ``max_model`` (labels
            of the extreme alternatives) and ``n_models``.

        Examples
        --------
        >>> from nowcastbox.models import TwoStepDFM
        >>> from nowcastbox.models.two_step import simulate_two_step_example
        >>> data = simulate_two_step_example(n_series=6, random_state=0)
        >>> groups = {"real": ["x1", "x2"], "soft": ["x3", "x4"], "fin": ["x5", "x6"]}
        >>> alt = alternative_models(TwoStepDFM(n_factors=1), data, "gdp", by=groups)
        >>> rng = alt.range()
        >>> bool((rng["spread"] >= 0).all()), int(rng["n_models"].iloc[0])
        (True, 6)
        """
        alt = self.alternatives
        rows = {
            "base": self.nowcasts.iloc[0],
            "min": alt.min(),
            "max": alt.max(),
            "median": alt.median(),
            "mean": alt.mean(),
        }
        out = pd.DataFrame(rows)
        out["spread"] = out["max"] - out["min"]
        out["min_model"] = [_arg(alt[p], "min") for p in alt.columns]
        out["max_model"] = [_arg(alt[p], "max") for p in alt.columns]
        out["n_models"] = alt.notna().sum().astype(int)
        out.index = self.periods
        out.index.name = "period"
        return out

    def summary(self) -> str:
        """Text summary: groups, models and the range of each target period.

        Returns
        -------
        str
            Multi-line report.

        Examples
        --------
        >>> from nowcastbox.models import TwoStepDFM
        >>> from nowcastbox.models.two_step import simulate_two_step_example
        >>> data = simulate_two_step_example(n_series=6, random_state=0)
        >>> groups = {"real": ["x1", "x2"], "soft": ["x3", "x4"], "fin": ["x5", "x6"]}
        >>> alt = alternative_models(TwoStepDFM(n_factors=1), data, "gdp", by=groups)
        >>> print(alt.summary())  # doctest: +ELLIPSIS
        Alternative models for 'gdp' (TwoStepDFM)
          groups (mapping): real (2); soft (2); fin (2)
        ...
        """
        mode = "re-estimated" if self.refit else "base parameters (re-filtered)"
        lines = [
            f"Alternative models for {self.target!r} ({self.info.get('model', 'model')})",
            f"  groups ({self.by}): "
            + "; ".join(f"{g} ({len(s)})" for g, s in self.groups.items()),
            f"  models       : {self.n_models} alternatives, {mode}",
        ]
        if self.skipped:
            lines.append(f"  skipped      : {[combination_label(c) for c in self.skipped]}")
        table = self.range()[["base", "min", "median", "max", "spread"]]
        lines += ["", table.to_string(float_format=lambda v: f"{v:.4g}")]
        return "\n".join(lines)

    def plot(
        self,
        backend: str = "plotly",
        *,
        theme: Theme | str | None = None,
        title: str | None = None,
        ax: Any = None,
    ) -> Any:
        """Alternative nowcasts around the base nowcast, per target period.

        Each alternative is a dot, the vertical bar spans the min-max range and the
        large marker is the base nowcast.

        Parameters
        ----------
        backend : {"plotly", "matplotlib"}, default "plotly"
            Plotting backend.
        theme : Theme or str, optional
            Visual theme (:mod:`nowcastbox.visualization.themes`).
        title : str, optional
            Figure title.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on (Matplotlib only).

        Returns
        -------
        plotly.graph_objects.Figure or matplotlib.figure.Figure
            The chart.

        Raises
        ------
        ValueError
            If the backend is unknown.

        Examples
        --------
        >>> from nowcastbox.models import TwoStepDFM
        >>> from nowcastbox.models.two_step import simulate_two_step_example
        >>> data = simulate_two_step_example(n_series=6, random_state=0)
        >>> groups = {"real": ["x1", "x2"], "soft": ["x3", "x4"], "fin": ["x5", "x6"]}
        >>> alt = alternative_models(TwoStepDFM(n_factors=1), data, "gdp", by=groups)
        >>> fig = alt.plot()
        >>> len(fig.data)  # range bar, 6 alternatives, base
        8
        """
        from nowcastbox.visualization._common import resolve

        resolved, kind = resolve(theme, backend, ax)
        text = title or f"Alternative nowcasts of {self.target} (without 1-2 groups)"
        if kind == "plotly":
            return _plot_plotly(self, resolved, text)
        return _plot_mpl(self, resolved, text, ax)


def _arg(column: pd.Series, how: str) -> str | None:
    values = column.dropna()
    if values.empty:
        return None
    return str(values.idxmin() if how == "min" else values.idxmax())


def _plot_plotly(alt: AlternativeNowcasts, theme: Theme, title: str) -> Any:
    import plotly.graph_objects as go

    from nowcastbox.visualization._common import finish_plotly, new_plotly_figure, rgba

    fig = new_plotly_figure()
    x = [str(p) for p in alt.periods]
    rng = alt.range()
    fig.add_trace(
        go.Bar(
            x=x,
            y=(rng["max"] - rng["min"]).to_numpy(),
            base=rng["min"].to_numpy(),
            width=0.25,
            marker_color=rgba(theme.interval_color, 0.25),
            name="range",
            hoverinfo="skip",
        )
    )
    others = alt.alternatives
    for label, row in others.iterrows():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=row.to_numpy(dtype=float),
                mode="markers",
                marker={"color": theme.neutral_color, "size": theme.marker_size},
                name=str(label),
                legendgroup="alternatives",
                showlegend=False,
                hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=alt.nowcasts.iloc[0].to_numpy(dtype=float),
            mode="markers",
            marker={"color": theme.out_of_sample_color, "size": 2 * theme.marker_size},
            name="base nowcast",
        )
    )
    return finish_plotly(fig, theme, title=title, xlabel="target period", ylabel=alt.target)


def _plot_mpl(alt: AlternativeNowcasts, theme: Theme, title: str, ax: Any) -> Any:
    from nowcastbox.visualization._common import finish_mpl, mpl_context, new_axes

    with mpl_context(theme):
        fig, axes = new_axes(ax, theme, None)
        x = np.arange(len(alt.periods), dtype=float)
        rng = alt.range()
        axes.vlines(
            x, rng["min"], rng["max"], color=theme.interval_color, alpha=0.4, lw=6, label="range"
        )
        others = alt.alternatives.to_numpy(dtype=float)
        for row in others:
            axes.plot(x, row, "o", color=theme.neutral_color, ms=theme.marker_size / 1.5)
        axes.plot(
            x,
            alt.nowcasts.iloc[0].to_numpy(dtype=float),
            "D",
            color=theme.out_of_sample_color,
            ms=theme.marker_size,
            label="base nowcast",
        )
        axes.set_xticks(x, [str(p) for p in alt.periods])
        finish_mpl(axes, theme, title=title, xlabel="target period", ylabel=alt.target)
    return fig


# ---------------------------------------------------------------------------- estimation
def _refit_one(
    model: Any,
    panel: MixedFrequencyData,
    target: str,
    dropped: tuple[str, ...],
    periods: pd.PeriodIndex,
    fit_kwargs: Mapping[str, Any],
) -> np.ndarray:
    """Out-of-sample nowcasts of a clone of ``model`` fitted without ``dropped``."""
    results = model.clone().fit(panel.drop(list(dropped)), target, **fit_kwargs)
    return results.out_of_sample.reindex(periods).to_numpy(dtype=float)


def _refilter_one(
    linear: Any, frame: pd.DataFrame, dropped: tuple[str, ...], periods: pd.PeriodIndex
) -> np.ndarray:
    """Nowcasts with the base parameters and the ``dropped`` series set to missing."""
    from nowcastbox.news._model import position

    masked = frame.copy()
    masked.loc[:, list(dropped)] = np.nan
    slots = [linear.target_slot(p) for p in periods]
    grid = linear.grid(max(slots), masked)
    smoothed = linear.smooth(linear.standardize(masked, grid))
    return np.array(
        [linear.to_original(linear.functional(smoothed, position(grid, s))) for s in slots]
    )


def _linear(results: NowcastResults) -> Any:
    from nowcastbox.news._model import linear_model

    try:
        return linear_model(results)
    except (TypeError, NotImplementedError) as err:
        raise TypeError(
            "refit=False re-runs the Kalman smoother with the base parameters and needs a "
            "state-space model (MixedFreqDFM, or TwoStepDFM with aggregate='factors' or "
            f"'variables'); got {results.model_name} results. Use refit=True. ({err})"
        ) from err


def _run_parallel(
    func: Callable[[tuple[str, ...]], np.ndarray],
    items: Sequence[tuple[str, ...]],
    n_jobs: int | None,
) -> list[np.ndarray]:
    if n_jobs in (None, 1) or len(items) <= 1:
        return [func(item) for item in items]
    try:
        from joblib import Parallel, delayed
    except ImportError:  # pragma: no cover - joblib ships with scikit-learn/most stacks
        logger.info("joblib is not installed; estimating the alternatives sequentially.")
        return [func(item) for item in items]
    return cast("list[np.ndarray]", list(Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in items)))


def _resolve_periods(base: NowcastResults, periods: Any) -> pd.PeriodIndex:
    """Target periods with an out-of-sample nowcast (default: all of them)."""
    frame = base.nowcast
    index = frame.index
    assert isinstance(index, pd.PeriodIndex)  # noqa: S101
    available = index[(frame["out_of_sample"].notna() & frame["observed"].isna()).to_numpy()]
    if periods is None:
        if available.empty:
            raise ValueError("The base model has no out-of-sample nowcast of the target.")
        return pd.PeriodIndex(available)
    items = [periods] if isinstance(periods, str | pd.Period) else list(periods)
    chosen = pd.PeriodIndex([pd.Period(p, freq=index.freqstr) for p in items])
    missing = [str(p) for p in chosen if p not in available]
    if missing:
        raise ValueError(
            f"No out-of-sample nowcast for the periods {missing}; available: "
            f"{[str(p) for p in available]}."
        )
    return chosen


def _base_and_panel(
    model: Any, data: Any, target: str, fit_kwargs: Mapping[str, Any], refit: bool
) -> tuple[NowcastResults, MixedFrequencyData, str]:
    """Base results, the target/predictor panel and the target name."""
    if isinstance(model, NowcastResults):
        if refit:
            raise ValueError(
                "refit=True needs an estimator; fitted results only allow refit=False."
            )
        source = model.data if data is None else data
        if source is None:
            raise ValueError("No data given and none stored in the results.")
        panel = as_mixed_frequency_data(source)
        name, regressors = resolve_target(target, panel.columns)
        if name != model.target:
            raise ValueError(f"The results are for {model.target!r}, not for {name!r}.")
        return model, _select(panel, name, regressors), name
    if not hasattr(model, "fit") or not hasattr(model, "clone"):
        raise TypeError(f"model must be a nowcastbox estimator or results; got {model!r}.")
    if data is None:
        raise ValueError("data is required when model is an estimator.")
    panel = as_mixed_frequency_data(data)
    name, regressors = resolve_target(target, panel.columns)
    panel = _select(panel, name, regressors)
    base = model.clone().fit(panel, name, **fit_kwargs)
    return base, panel, name


def _model_series(
    panel: MixedFrequencyData, base: NowcastResults, linear: Any, name: str
) -> MixedFrequencyData:
    """Keep the series of the fitted model (others cannot move a re-filtered nowcast)."""
    fitted = base.data.columns if base.data is not None else linear.series
    keep = {*fitted, *linear.series, name}
    extra = [c for c in panel.columns if c not in keep]
    if extra:
        logger.info("series outside the fitted model ignored with refit=False: %s", extra)
    return panel.select([c for c in panel.columns if c in keep])


def _select(panel: MixedFrequencyData, name: str, regressors: Sequence[str]) -> MixedFrequencyData:
    keep = set(regressors) | {name}
    return panel.select([c for c in panel.columns if c in keep])


def alternative_models(
    model: Any,
    data: MixedFrequencyData | pd.DataFrame | None,
    target: str,
    *,
    by: str | Mapping[str, Any] = "category",
    drop: int | Iterable[int] = (1, 2),
    refit: bool = True,
    groups: Sequence[str] | None = None,
    periods: Any = None,
    n_jobs: int | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> AlternativeNowcasts:
    """Nowcasts of alternative models without one or two groups of variables.

    Implements the range of alternative nowcasts of Linzenich & Meunier (2024, ECB WP
    3004, §3.5): for every combination of ``drop`` groups, the model is estimated again
    without the series of those groups (``refit=True``) or re-run with the base
    parameters treating them as missing (``refit=False``).

    Parameters
    ----------
    model : estimator or NowcastResults
        Unfitted (or fitted) nowcastbox estimator, e.g. ``TwoStepDFM(n_factors=2)``;
        the base model is a clone fitted on ``data``. Fitted :class:`NowcastResults`
        are accepted with ``refit=False`` (their parameters are used as given).
    data : MixedFrequencyData or pandas.DataFrame or None
        Panel (one information set). ``None`` only with results that store their data.
        With results (``refit=False``) it may be another vintage: the base and the
        alternative nowcasts are all re-filtered on it, and series outside the fitted
        model are ignored.
    target : str
        Target name or formula (``"gdp ~ . - x3"``).
    by : {"category", "block"} or mapping, default "category"
        How predictors are grouped (see :func:`resolve_groups`).
    drop : int or iterable of int, default (1, 2)
        Numbers of groups removed simultaneously (sizes above the number of groups are
        ignored).
    refit : bool, default True
        Re-estimate every alternative (``False``: base parameters, dropped series as
        missing; state-space models only).
    groups : sequence of str, optional
        Only these groups are dropped.
    periods : period-like or sequence, optional
        Target periods (default: every out-of-sample period of the base nowcast).
    n_jobs : int, optional
        Parallel jobs for ``refit=True`` (joblib; ``None``/1 sequential, -1 all cores).
    fit_kwargs : mapping, optional
        Extra keyword arguments of ``fit`` (e.g. ``{"horizon": 3}``).

    Returns
    -------
    AlternativeNowcasts
        ``table()``, ``range()``, ``summary()`` and ``plot()``.

    Raises
    ------
    ValueError
        On an unknown grouping or group, invalid ``drop``, periods without an
        out-of-sample nowcast, or combinations removing every predictor.
    TypeError
        If ``refit=False`` is used with a model that is not a state-space model.

    Notes
    -----
    With ``refit=False`` the nowcast is the linear Kalman-smoother functional of the
    base model (the one used by the news decomposition, Bańbura & Modugno, 2014), so the
    alternatives differ from the base only through the information set. Combinations
    that would leave no predictor (e.g. a global block containing every series) are
    skipped and listed in :attr:`AlternativeNowcasts.skipped`.

    Examples
    --------
    >>> from nowcastbox.models import TwoStepDFM
    >>> from nowcastbox.models.two_step import simulate_two_step_example
    >>> data = simulate_two_step_example(n_series=4, random_state=0)
    >>> alt = alternative_models(
    ...     TwoStepDFM(n_factors=1),
    ...     data,
    ...     "gdp",
    ...     by={"a": ["x1", "x2"], "b": ["x3", "x4"]},
    ...     drop=1,
    ... )
    >>> alt.table().index.tolist()
    ['base', '-a', '-b']
    """
    start = time.perf_counter()
    kwargs = dict(fit_kwargs or {})
    base, panel, name = _base_and_panel(model, data, target, kwargs, refit)
    chosen = _resolve_periods(base, periods)
    linear = None if refit else _linear(base)
    if linear is not None:
        panel = _model_series(panel, base, linear, name)
    found = resolve_groups(panel, name, by, groups)
    predictors = [c for c in panel.columns if c != name]
    combos, skipped = _combinations(found, _drop_sizes(drop, len(found)), predictors)

    def members(combo: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s for g in combo for s in found[g]))

    if linear is None:
        base_row = base.out_of_sample.reindex(chosen).to_numpy(dtype=float)
        rows = _run_parallel(
            lambda c: _refit_one(model, panel, name, members(c), chosen, kwargs), combos, n_jobs
        )
    else:
        # the base row is re-filtered too, so that base and alternatives share the
        # information set of ``data`` (which may be a later vintage than the fit)
        frame = panel.to_frame()
        base_row = _refilter_one(linear, frame, (), chosen)
        rows = [_refilter_one(linear, frame, members(c), chosen) for c in combos]
    nowcasts = pd.DataFrame(
        np.vstack([base_row, *rows]),
        index=[BASE_LABEL, *(combination_label(c) for c in combos)],
        columns=chosen,
    )
    return AlternativeNowcasts(
        target=name,
        by=_by_name(by),
        refit=bool(refit),
        groups=found,
        combinations=tuple(combos),
        nowcasts=nowcasts,
        base=base,
        skipped=tuple(skipped),
        info={"model": base.model_name, "seconds": round(time.perf_counter() - start, 4)},
    )

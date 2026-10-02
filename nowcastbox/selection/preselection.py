r"""Pre-selection of indicators for a nowcasting model (ECB toolbox workflow).

Before a model is specified, the candidate indicators are ranked against the target
with several complementary criteria and the rankings are combined into one score
(Linzenich & Meunier, 2024, ECB WP 3004, §2.1; applied in Chinn, Meunier & Stumpner,
2023, ECB WP 2808):

* ``"tstat"`` - :math:`|t|` of the indicator in a regression of the target on a
  constant, ``y_lags`` lags of the target and the indicator, with HAC standard errors
  (Bai & Ng, 2008, hard thresholding; :func:`~nowcastbox.selection.hard_threshold`);
* ``"sis"`` - absolute marginal correlation, *sure independence screening* (Fan & Lv,
  2008; :func:`~nowcastbox.selection.sis`);
* ``"lars"`` - order of entry on the least angle regression path (Efron, Hastie,
  Johnstone & Tibshirani, 2004; :func:`~nowcastbox.selection.lars_select`).

**Mixed frequency.** The rankings are computed at the frequency of the target. Every
candidate observed at a higher frequency (monthly indicators for a quarterly target) is
first aggregated to target periods with the weights of its aggregation rule (the
series' ``SeriesMetadata.aggregation`` if set, otherwise the ``aggregation`` argument,
``"average"`` by default, i.e. the quarterly mean of the three months; the
Mariano-Murasawa rule uses :math:`\tfrac13(1,2,3,2,1)`), exactly as a bridge equation
would use it. Candidates at the target frequency are used as they are; candidates at a
lower frequency than the target are skipped.

**Leads and lags.** ``x_lags`` adds shifted copies of every aggregated candidate:
``l > 0`` uses :math:`\bar x_{t-l}` (named ``<series>_lag<l>``), ``l < 0`` uses the
lead :math:`\bar x_{t+|l|}` (``<series>_lead<|l|>``). Each (series, lag) pair is ranked
as a separate candidate; a series is then as good as its best lag.

**Aggregated score.** For method :math:`m` with rank :math:`r_m(i)\in\{1,\dots,N\}`
(1 = best; candidates a method could not rank share the remaining positions, i.e. get
their average), the normalised score is :math:`s_m(i) = 1 - (r_m(i) - 1)/(N - 1)` and
the aggregated score is the weighted mean

.. math::

    S(i) = \frac{\sum_m w_m\, s_m(i)}{\sum_m w_m} \in [0, 1].

Series are ordered by :math:`S` (ties broken by the method ranks in the order of
``methods``) and the ``top`` best are selected.

**No look-ahead.** With ``as_of`` the panel is first replaced by its pseudo real-time
view :meth:`MixedFrequencyData.as_of`, so that only data released by that date enter the
rankings (the right choice inside a pseudo real-time evaluation).

References
----------
Linzenich, J., & Meunier, B. (2024). Nowcasting made easier: a toolbox for economists.
ECB Working Paper No. 3004.

Chinn, M. D., Meunier, B., & Stumpner, S. (2023). Nowcasting world trade with machine
learning: a three-step approach. ECB Working Paper No. 2808.

Bai, J., & Ng, S. (2008). Forecasting economic time series using targeted predictors.
*Journal of Econometrics*, 146(2), 304-317.

Fan, J., & Lv, J. (2008). Sure independence screening for ultrahigh dimensional feature
space. *Journal of the Royal Statistical Society B*, 70(5), 849-911.

Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression.
*The Annals of Statistics*, 32(2), 407-499.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import MixedFrequencyData, as_mixed_frequency_data
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.formula import resolve_target
from nowcastbox.core.frequency import AggregationType, Frequency
from nowcastbox.models.bridge import aggregate_to_target, resolve_aggregation_weights
from nowcastbox.selection.targeted import (
    TargetedPredictorsResult,
    hard_threshold,
    lars_select,
    sis,
)

__all__ = [
    "PRESELECTION_METHODS",
    "PreselectionResult",
    "align_to_target",
    "preselect",
    "rank_scores",
]

logger = get_logger(__name__)

PRESELECTION_METHODS: tuple[str, ...] = ("tstat", "sis", "lars")
"""Ranking methods understood by :func:`preselect`."""

_STAT_NAMES = {"tstat": "abs_t", "sis": "abs_corr", "lars": "lars_entry"}

AggregationSpec = str | AggregationType | Sequence[float]


# ---------------------------------------------------------------------------
# Alignment to the target frequency
# ---------------------------------------------------------------------------
def _lag_name(series: str, lag: int) -> str:
    if lag == 0:
        return series
    return f"{series}_lag{lag}" if lag > 0 else f"{series}_lead{-lag}"


def _is_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | np.integer)


def _check_lags(x_lags: int | Iterable[int]) -> tuple[int, ...]:
    if _is_int(x_lags):
        if int(x_lags) < 0:  # type: ignore[arg-type]
            raise ValueError(f"An integer x_lags must be non-negative; got {x_lags!r}.")
        return tuple(range(int(x_lags) + 1))  # type: ignore[arg-type]
    if isinstance(x_lags, bool | str) or not isinstance(x_lags, Iterable):
        raise ValueError(f"x_lags must be an integer or a sequence of integers; got {x_lags!r}.")
    lags = list(x_lags)
    if not all(_is_int(v) for v in lags) or not lags or len(set(lags)) != len(lags):
        raise ValueError(f"x_lags must be distinct integers; got {x_lags!r}.")
    return tuple(int(v) for v in lags)


def _aggregation_for(
    data: MixedFrequencyData, column: str, aggregation: AggregationSpec | Mapping[str, Any]
) -> Any:
    """Aggregation rule of ``column``: mapping entry > series metadata > default."""
    if isinstance(aggregation, Mapping):
        if column in aggregation:
            return aggregation[column]
        default: Any = "average"
    else:
        default = aggregation
    meta = data.metadata[column].aggregation
    return default if meta is None else meta


def _aggregated(
    data: MixedFrequencyData,
    column: str,
    target_freq: Frequency,
    periods: pd.PeriodIndex,
    spec: Any,
) -> pd.Series | None:
    """Candidate ``column`` on target periods, or None when it cannot be aligned."""
    freq = data.metadata[column].frequency
    native = data.to_native(column)
    if freq == target_freq:
        return native.reindex(periods)
    if freq.is_lower_than(target_freq):
        return None
    try:
        weights = resolve_aggregation_weights(spec, freq, target_freq, normalize="ratio")
    except ValueError:
        return None
    grid = pd.period_range(native.index[0], native.index[-1], freq=freq.pandas_freq)
    return aggregate_to_target(native.reindex(grid), weights, periods)


def _candidate_frame(
    data: MixedFrequencyData,
    target: str,
    candidates: Sequence[str],
    lags: tuple[int, ...],
    aggregation: AggregationSpec | Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Aggregated/shifted candidates, target and the (series, lag) key of each column."""
    target_freq = data.metadata[target].frequency
    pf = target_freq.pandas_freq
    periods = pd.period_range(data.start.asfreq(pf), data.end.asfreq(pf), freq=pf)
    y = data.to_native(target).reindex(periods).astype(float)
    columns: dict[str, pd.Series] = {}
    keys: list[tuple[str, str, int]] = []
    skipped: list[str] = []
    for col in candidates:
        agg = _aggregated(data, col, target_freq, periods, _aggregation_for(data, col, aggregation))
        if agg is None:
            skipped.append(col)
            continue
        for lag in lags:
            name = _lag_name(col, lag)
            columns[name] = agg.shift(lag)
            keys.append((name, col, lag))
    if skipped:
        warnings.warn(
            f"{len(skipped)} candidates cannot be aligned to the {target_freq.label} target "
            f"(lower or non-fixed frequency) and are skipped: {skipped}.",
            DataQualityWarning,
            stacklevel=3,
        )
    if not columns:
        raise NowcastDataError("No candidate indicator can be aligned to the target.")
    frame = pd.DataFrame(columns, index=periods)
    key_frame = pd.DataFrame(keys, columns=["candidate", "series", "lag"]).set_index("candidate")
    return frame, y, key_frame


def align_to_target(
    data: MixedFrequencyData | pd.DataFrame,
    target: str,
    *,
    x_lags: int | Iterable[int] = (0,),
    aggregation: AggregationSpec | Mapping[str, Any] = "average",
    candidates: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Aggregate the candidate indicators to the target frequency and add leads/lags.

    Parameters
    ----------
    data : MixedFrequencyData or DataFrame
        Panel (a DataFrame is wrapped with :func:`as_mixed_frequency_data`).
    target : str
        Target series name or formula (``"gdp ~ ."``, ``"gdp ~ x1 + x2"``).
    x_lags : int or iterable of int, default (0,)
        Shifts in target periods; positive = lags, negative = leads. An integer ``L``
        means ``(0, 1, ..., L)``.
    aggregation : str, AggregationType, weights or mapping, default "average"
        Aggregation of higher-frequency candidates for series whose metadata has no
        ``aggregation`` (a mapping ``{series: rule}`` overrides per series).
    candidates : sequence of str, optional
        Candidate series (default: the regressors of ``target``).

    Returns
    -------
    frame : pandas.DataFrame
        Candidates on the target's native PeriodIndex (one column per series and lag).
    y : pandas.Series
        Target on the same index.

    Raises
    ------
    NowcastDataError
        If no candidate can be aligned or a candidate is unknown.
    ValueError
        Invalid ``x_lags``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> df = pd.DataFrame(
    ...     {"x": np.arange(1.0, 7.0), "y": [np.nan, np.nan, 1.0, np.nan, np.nan, 2.0]}, index=idx
    ... )
    >>> mfd = MixedFrequencyData(df, {"x": "M", "y": "Q"})
    >>> frame, y = align_to_target(mfd, "y", x_lags=(0, 1))
    >>> frame.round(6).to_dict("list")
    {'x': [2.0, 5.0], 'x_lag1': [nan, 2.0]}
    """
    panel = as_mixed_frequency_data(data)
    name, regressors = resolve_target(target, panel.columns)
    cands = _resolve_candidates(panel, name, regressors, candidates)
    frame, y, _ = _candidate_frame(panel, name, cands, _check_lags(x_lags), aggregation)
    return frame, y


def _resolve_candidates(
    data: MixedFrequencyData, target: str, regressors: Sequence[str], candidates: Any
) -> list[str]:
    if candidates is None:
        return list(regressors)
    cands = [str(c) for c in candidates]
    unknown = [c for c in cands if c not in data.columns or c == target]
    if unknown or not cands:
        raise NowcastDataError(f"Invalid candidate series {unknown or cands}.")
    return cands


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Settings:
    horizon: int
    y_lags: int
    cov_type: Literal["hac", "nonrobust"]
    hac_lags: int | None
    min_obs: int
    lars_method: Literal["lar", "lasso"]
    missing: Literal["drop", "mean"]


def _rank_tstat(frame: pd.DataFrame, y: pd.Series, st: _Settings) -> TargetedPredictorsResult:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No predictor has", category=DataQualityWarning)
        return hard_threshold(
            frame,
            y,
            horizon=st.horizon,
            y_lags=st.y_lags,
            threshold=0.0,
            cov_type=st.cov_type,
            hac_lags=st.hac_lags,
            min_obs=st.min_obs,
        )


def _rank_sis(frame: pd.DataFrame, y: pd.Series, st: _Settings) -> TargetedPredictorsResult:
    return sis(frame, y, horizon=st.horizon, n_predictors=1, min_obs=st.min_obs)


def _pair_counts(frame: pd.DataFrame, y: pd.Series, horizon: int) -> pd.Series:
    """Observations of each candidate paired with an observed target ``horizon`` ahead."""
    observed = y.index[y.notna().to_numpy()]
    rows = observed[: max(len(observed) - horizon, 0)]
    return frame.loc[rows].notna().sum()


def _rank_lars(frame: pd.DataFrame, y: pd.Series, st: _Settings) -> TargetedPredictorsResult:
    # As for the t-stat and SIS, candidates with fewer than ``min_obs`` observations are
    # not ranked (mean imputation would otherwise let them enter the path).
    counts = _pair_counts(frame, y, st.horizon)
    usable = frame.loc[:, (counts >= st.min_obs).to_numpy()]
    if usable.shape[1] == 0:
        raise NowcastDataError("No candidate has enough observations for LARS.")
    # Lags/leads leave expected gaps at the sample borders: imputation is not news here.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*replaced by the mean", category=DataQualityWarning
        )
        res = lars_select(
            usable, y, horizon=st.horizon, n_predictors=1, method=st.lars_method, missing=st.missing
        )
    names = [str(c) for c in frame.columns]
    return replace(res, scores=res.scores.reindex(names), ranking=res.ranking.reindex(names))


_RANKERS: dict[str, Callable[[pd.DataFrame, pd.Series, _Settings], TargetedPredictorsResult]] = {
    "tstat": _rank_tstat,
    "sis": _rank_sis,
    "lars": _rank_lars,
}


def _check_methods(methods: str | Iterable[str]) -> tuple[str, ...]:
    out = (methods,) if isinstance(methods, str) else tuple(methods)
    unknown = [m for m in out if m not in _RANKERS]
    if unknown or not out or len(set(out)) != len(out):
        raise ValueError(
            f"methods must be distinct names from {PRESELECTION_METHODS}; got {methods!r}."
        )
    return out


def _check_weights(
    weights: Mapping[str, float] | None, methods: tuple[str, ...]
) -> dict[str, float]:
    if weights is None:
        return dict.fromkeys(methods, 1.0)
    unknown = sorted(set(weights) - set(methods))
    if unknown:
        raise ValueError(f"weights given for methods not in {methods}: {unknown}.")
    out = {m: float(weights.get(m, 1.0)) for m in methods}
    if any(not np.isfinite(w) or w < 0 for w in out.values()) or sum(out.values()) <= 0:
        raise ValueError(f"weights must be non-negative, finite and not all zero; got {out}.")
    return out


def rank_scores(ranks: pd.DataFrame) -> pd.DataFrame:
    """Normalised scores ``1 - (r - 1)/(N - 1)`` of rank columns (1 = best).

    Entries a method could not rank (NaN) receive the average of the positions left
    free, ``(n_ranked + 1 + N) / 2``.

    Parameters
    ----------
    ranks : pandas.DataFrame
        One column of ranks per method, one row per candidate.

    Returns
    -------
    pandas.DataFrame
        Scores in ``[0, 1]`` (1 = best); all ones when ``N = 1``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rank_scores(pd.DataFrame({"m": [1.0, 3.0, 2.0, np.nan, np.nan]}))["m"].tolist()
    [1.0, 0.5, 0.75, 0.125, 0.125]
    """
    n = len(ranks)
    filled = ranks.apply(lambda col: col.fillna((col.notna().sum() + 1 + n) / 2.0))
    if n <= 1:
        return filled * 0.0 + 1.0
    return 1.0 - (filled - 1.0) / (n - 1.0)


def _aggregate_score(ranks: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    scores = rank_scores(ranks)
    w = pd.Series(weights)
    return (scores[list(w.index)] * w).sum(axis=1) / w.sum()


def _order(table: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    """Sort by score (desc), ties broken by the method ranks; add the final rank."""
    keys = table[[f"rank_{m}" for m in methods]].fillna(np.inf)
    keys.insert(0, "_neg_score", -table["score"])
    order = keys.sort_values(list(keys.columns), kind="mergesort").index
    out = table.loc[order].copy()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PreselectionResult:
    """Outcome of :func:`preselect`.

    Attributes
    ----------
    target : str
        Target series.
    methods : tuple[str, ...]
        Ranking methods used.
    weights : dict[str, float]
        Weight of each method in the aggregated score.
    selected : list[str]
        The ``top`` series by aggregated score, best first.
    x_lags : tuple[int, ...]
        Leads (< 0) and lags (> 0) of the candidates, in target periods.
    horizon : int
        Forecast horizon in target periods.
    as_of : pandas.Timestamp or None
        Information date of the pseudo real-time view (None: all data).
    n_obs : int
        Number of target observations available.
    series_table : pandas.DataFrame
        One row per series, sorted by rank (see :meth:`table`).
    candidate_table : pandas.DataFrame
        One row per (series, lag) candidate.
    results : dict[str, TargetedPredictorsResult]
        Candidate-level output of each method.
    """

    target: str
    methods: tuple[str, ...]
    weights: dict[str, float]
    selected: list[str]
    x_lags: tuple[int, ...]
    horizon: int
    as_of: pd.Timestamp | None
    n_obs: int
    series_table: pd.DataFrame
    candidate_table: pd.DataFrame
    results: dict[str, TargetedPredictorsResult] = field(default_factory=dict)

    @property
    def n_selected(self) -> int:
        """Number of selected series."""
        return len(self.selected)

    def table(self, level: Literal["series", "candidate"] = "series") -> pd.DataFrame:
        """Ranking table.

        Parameters
        ----------
        level : {"series", "candidate"}, default "series"
            ``"series"``: one row per series with the final ``rank``, aggregated
            ``score``, ``selected`` flag, ``best_lag``, the rank of each method
            (``rank_<method>``, best over lags), the statistic of each method
            (``abs_t``, ``abs_corr``, ``lars_entry``) at the method's own best lag, and the
            ``frequency``, ``release_delay`` (days), ``category`` and ``blocks`` of the
            series. ``"candidate"``: one row per series and lag.

        Returns
        -------
        pandas.DataFrame
            Copy of the table, sorted by rank.

        Raises
        ------
        ValueError
            Unknown ``level``.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> pre = preselect(nb.load_simulated_dfm().data, "gdp", top=3)
        >>> pre.table().columns[:4].tolist()
        ['rank', 'score', 'selected', 'best_lag']
        """
        if level == "series":
            return self.series_table.copy()
        if level == "candidate":
            return self.candidate_table.copy()
        raise ValueError(f"level must be 'series' or 'candidate'; got {level!r}.")

    def ranking(self, method: str | None = None) -> pd.Series:
        """Series ranks by one method, or the final ranking.

        Parameters
        ----------
        method : str, optional
            One of :attr:`methods`; None gives the aggregated ranking.

        Returns
        -------
        pandas.Series
            Ranks (1 = best) indexed by series, sorted ascending (unranked last).

        Raises
        ------
        ValueError
            If ``method`` was not used.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> pre = preselect(nb.load_simulated_dfm().data, "gdp", methods=("sis",), top=3)
        >>> pre.ranking("sis").index[:3].tolist() == pre.selected
        True
        """
        if method is None:
            col = "rank"
        elif method in self.methods:
            col = f"rank_{method}"
        else:
            raise ValueError(f"method must be one of {self.methods}; got {method!r}.")
        return self.series_table[col].sort_values(kind="mergesort").rename(method or "rank")

    def transform(
        self, data: MixedFrequencyData, *, keep_target: bool = True
    ) -> MixedFrequencyData:
        """Restrict a panel to the selected series (and the target).

        Parameters
        ----------
        data : MixedFrequencyData
            Panel containing the selected series.
        keep_target : bool, default True
            Keep the target column as well.

        Returns
        -------
        MixedFrequencyData
            The reduced panel.

        Raises
        ------
        NowcastDataError
            If a selected series is missing from ``data``.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> data = nb.load_simulated_dfm().data
        >>> preselect(data, "gdp", top=2).transform(data).n_series
        3
        """
        cols = [*self.selected, self.target] if keep_target else list(self.selected)
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise NowcastDataError(f"Series not found in data: {missing}.")
        return data.select(cols)

    def summary(self) -> str:
        """Plain-text summary.

        Returns
        -------
        str
            Settings and the selected series with their scores.

        Examples
        --------
        >>> import nowcastbox as nb
        >>> pre = preselect(nb.load_simulated_dfm().data, "gdp", top=2)
        >>> pre.summary().splitlines()[0]
        'Pre-selection of indicators for gdp'
        """
        weights = ", ".join(f"{m}={w:g}" for m, w in self.weights.items())
        as_of = "-" if self.as_of is None else str(self.as_of.date())
        lines = [
            f"Pre-selection of indicators for {self.target}",
            "=" * 60,
            f"methods (weights): {weights}   x_lags: {list(self.x_lags)}",
            f"horizon: {self.horizon}   as_of: {as_of}   target observations: {self.n_obs}",
            f"Selected {self.n_selected} of {len(self.series_table)} series:",
        ]
        for name in self.selected:
            row = self.series_table.loc[name]
            lines.append(
                f"  {int(row['rank']):>3}. {name}  score={row['score']:.3f}  "
                f"lag={int(row['best_lag'])}"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def plot(
        self,
        *,
        top: int | None = None,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        ax: Any = None,
    ) -> Any:
        """Horizontal bar chart of the aggregated score (selected series highlighted).

        Parameters
        ----------
        top : int, optional
            Number of series shown (default: ``max(n_selected, 10)``, at most all).
        backend : {"matplotlib", "plotly"}, default "matplotlib"
            Plotting library.
        ax : matplotlib.axes.Axes, optional
            Existing axes (Matplotlib only).

        Returns
        -------
        matplotlib.axes.Axes or plotly.graph_objects.Figure
            The chart.

        Raises
        ------
        ValueError
            Unknown backend, or ``ax`` passed with the Plotly backend.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import nowcastbox as nb
        >>> ax = preselect(nb.load_simulated_dfm().data, "gdp", top=3).plot()
        >>> ax.get_xlabel()
        'aggregated score'
        """
        k = max(self.n_selected, 10) if top is None else int(top)
        rows = self.series_table.iloc[: max(1, k)]
        if backend == "matplotlib":
            return _bars_matplotlib(rows, self.target, ax)
        if backend == "plotly":
            if ax is not None:
                raise ValueError("'ax' is only supported by the matplotlib backend.")
            return _bars_plotly(rows, self.target)
        raise ValueError(f"backend must be 'matplotlib' or 'plotly'; got {backend!r}.")


_SELECTED_COLOR = "#1f6feb"
_OTHER_COLOR = "#b8b8b8"


def _bars_matplotlib(rows: pd.DataFrame, target: str, ax: Any) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(2.5, 0.3 * len(rows) + 1)))
    colors = [_SELECTED_COLOR if s else _OTHER_COLOR for s in rows["selected"]]
    ax.barh(list(rows.index)[::-1], rows["score"].to_numpy()[::-1], color=colors[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("aggregated score")
    ax.set_title(f"Pre-selection for {target}")
    return ax


def _bars_plotly(rows: pd.DataFrame, target: str) -> Any:
    import plotly.graph_objects as go

    colors = [_SELECTED_COLOR if s else _OTHER_COLOR for s in rows["selected"]]
    fig = go.Figure(
        go.Bar(
            x=rows["score"].to_numpy()[::-1],
            y=list(rows.index)[::-1],
            orientation="h",
            marker={"color": colors[::-1]},
        )
    )
    fig.update_layout(
        title=f"Pre-selection for {target}",
        xaxis_title="aggregated score",
        xaxis_range=[0, 1],
    )
    return fig


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _candidate_table(
    keys: pd.DataFrame,
    results: dict[str, TargetedPredictorsResult],
    weights: dict[str, float],
) -> pd.DataFrame:
    table = keys.copy()
    for m, res in results.items():
        table[f"rank_{m}"] = res.ranking.reindex(table.index).to_numpy()
        table[_STAT_NAMES[m]] = res.scores.reindex(table.index).to_numpy()
    ranks = table[[f"rank_{m}" for m in results]].rename(columns=lambda c: c[5:])
    table["score"] = _aggregate_score(ranks, weights)
    return _order(table, tuple(results))


def _best_by(candidates: pd.DataFrame, column: str, ascending: bool) -> pd.Series:
    """Candidate (row label) with the best ``column`` within each series."""
    ordered = candidates.sort_values(column, ascending=ascending, kind="mergesort")
    return ordered.reset_index().groupby("series", sort=False)["candidate"].first()


def _series_table(
    candidates: pd.DataFrame,
    data: MixedFrequencyData,
    methods: tuple[str, ...],
    weights: dict[str, float],
) -> pd.DataFrame:
    best = _best_by(candidates, "rank", ascending=True)
    table = pd.DataFrame(index=pd.Index(best.index, name="series"))
    table["best_lag"] = candidates.loc[best.to_numpy(), "lag"].to_numpy()
    for m in methods:
        col, stat = f"rank_{m}", _STAT_NAMES[m]
        picked = candidates.loc[_best_by(candidates, col, ascending=True)]
        picked.index = pd.Index(picked["series"])
        table[col] = picked[col].reindex(table.index).rank(method="first")
        table[stat] = picked[stat].reindex(table.index)
    ranks = table[[f"rank_{m}" for m in methods]].rename(columns=lambda c: c[5:])
    table["score"] = _aggregate_score(ranks, weights)
    table = _order(table, methods)
    meta = data.metadata
    table["frequency"] = [meta[s].frequency.value for s in table.index]
    table["release_delay"] = [meta[s].release_delay for s in table.index]
    table["category"] = [_category(meta[s].category) for s in table.index]
    table["blocks"] = [", ".join(meta[s].blocks) for s in table.index]
    return table


def _category(value: Any) -> str | None:
    return None if value is None else str(getattr(value, "value", value))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def preselect(
    data: MixedFrequencyData | pd.DataFrame,
    target: str,
    methods: str | Iterable[str] = PRESELECTION_METHODS,
    *,
    x_lags: int | Iterable[int] = (0,),
    weights: Mapping[str, float] | None = None,
    top: int | None = None,
    as_of: pd.Timestamp | str | None = None,
    release_delays: Mapping[str, int] | None = None,
    horizon: int = 0,
    y_lags: int = 0,
    aggregation: AggregationSpec | Mapping[str, Any] = "average",
    candidates: Sequence[str] | None = None,
    cov_type: Literal["hac", "nonrobust"] = "hac",
    hac_lags: int | None = None,
    min_obs: int = 10,
    lars_method: Literal["lar", "lasso"] = "lar",
    missing: Literal["drop", "mean"] = "mean",
) -> PreselectionResult:
    """Rank candidate indicators against a target and keep the best (ECB WP 3004, §2.1).

    See the module description for the alignment of mixed frequencies, the leads/lags
    and the aggregated score.

    Parameters
    ----------
    data : MixedFrequencyData or DataFrame
        Panel with the target and the candidates.
    target : str
        Target series, or a formula restricting the candidates (``"gdp ~ x1 + x2"``).
    methods : str or iterable of str, default ("tstat", "sis", "lars")
        Rankings to combine (see :data:`PRESELECTION_METHODS`).
    x_lags : int or iterable of int, default (0,)
        Leads (< 0) and lags (> 0) of every candidate in target periods; an integer
        ``L`` means ``(0, ..., L)``.
    weights : mapping of str to float, optional
        Weight of each method in the aggregated score (default 1 for every method;
        methods not in the mapping get 1).
    top : int, optional
        Number of series selected (default ``min(30, N)``; capped at ``N``).
    as_of : Timestamp or str, optional
        Use only the data released by this date (:meth:`MixedFrequencyData.as_of`).
    release_delays : mapping of str to int, optional
        Release-delay overrides (days) for ``as_of``.
    horizon : int, default 0
        Forecast horizon in target periods.
    y_lags : int, default 0
        Target lags used as controls in the t-stat regressions.
    aggregation : str, AggregationType, weights or mapping, default "average"
        Aggregation of higher-frequency candidates without an ``aggregation`` in their
        metadata (a mapping ``{series: rule}`` overrides per series).
    candidates : sequence of str, optional
        Candidate series (default: every series other than the target, or the
        regressors of the formula).
    cov_type : {"hac", "nonrobust"}, default "hac"
        Standard errors of the t-stat regressions.
    hac_lags : int, optional
        HAC bandwidth (default :func:`~nowcastbox.selection.newey_west_lags`).
    min_obs : int, default 10
        Minimum number of observations per candidate (all methods); candidates with
        fewer are not ranked.
    lars_method : {"lar", "lasso"}, default "lar"
        LARS variant.
    missing : {"drop", "mean"}, default "mean"
        Missing aggregated values in LARS: drop incomplete periods or impute the mean.

    Returns
    -------
    PreselectionResult
        Rankings, aggregated score, selection and the summary table.

    Raises
    ------
    ValueError
        Invalid methods, weights, lags or ``top``.
    NowcastDataError
        Invalid data, unknown candidates, or missing release delays with ``as_of``.

    Examples
    --------
    >>> import nowcastbox as nb
    >>> data = nb.load_simulated_dfm().data
    >>> pre = preselect(data, "gdp", x_lags=(0, 1), top=5)
    >>> pre.n_selected, list(pre.methods)
    (5, ['tstat', 'sis', 'lars'])
    >>> small = data.select([*pre.selected, "gdp"])
    """
    meths = _check_methods(methods)
    w = _check_weights(weights, meths)
    lags = _check_lags(x_lags)
    panel = as_mixed_frequency_data(data)
    name, regressors = resolve_target(target, panel.columns)
    cands = _resolve_candidates(panel, name, regressors, candidates)
    stamp = None if as_of is None else pd.Timestamp(as_of)
    if stamp is not None:
        panel = panel.as_of(stamp, release_delays)
    frame, y, keys = _candidate_frame(panel, name, cands, lags, aggregation)
    st = _Settings(horizon, y_lags, cov_type, hac_lags, min_obs, lars_method, missing)
    results = {m: _RANKERS[m](frame, y, st) for m in meths}
    cand_table = _candidate_table(keys, results, w)
    series = _series_table(cand_table, panel, meths, w)
    k = _check_top(top, len(series))
    series["selected"] = np.arange(len(series)) < k
    front = ["rank", "score", "selected", "best_lag"]
    series = series[front + [c for c in series.columns if c not in front]]
    selected = [str(s) for s in series.index[:k]]
    logger.info("Pre-selection for %s: %d of %d series kept", name, k, len(series))
    return PreselectionResult(
        target=name,
        methods=meths,
        weights=w,
        selected=selected,
        x_lags=lags,
        horizon=horizon,
        as_of=stamp,
        n_obs=int(y.notna().sum()),
        series_table=series,
        candidate_table=cand_table,
        results=results,
    )


def _check_top(top: int | None, n: int) -> int:
    if top is None:
        return min(30, n)
    if isinstance(top, bool) or not isinstance(top, int | np.integer) or top < 1:
        raise ValueError(f"top must be a positive integer; got {top!r}.")
    return min(int(top), n)

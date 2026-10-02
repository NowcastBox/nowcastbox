"""Result containers returned by every nowcasting estimator.

:class:`NowcastResults` is a frozen (immutable) dataclass. Estimators may subclass it
(e.g. :class:`FactorResults` for factor models) and add fields; all fields are
keyword-only, so subclasses can add required fields freely.

The ``nowcast`` frame
---------------------
``nowcast`` is indexed by the **native** periods of the target (e.g. a quarterly
``PeriodIndex`` for GDP) and has at least the columns

``observed``
    Target value as observed in the estimation data (NaN when not available).
``in_sample``
    Model estimate for periods where the target is observed (fitted values).
``out_of_sample``
    Model estimate for periods where the target is **not** observed: backcasts,
    nowcasts and forecasts.

Exactly one of ``in_sample``/``out_of_sample`` is filled for each period with an
estimate. Values are in the **original units** of the target as supplied to ``fit``
(i.e. after preprocessing transformations, before standardisation). Further columns
(``lower_68``, ``upper_90``, ``std`` ...) may be added by density nowcasts (I5).

Plot hook
---------
``NowcastResults.plot(kind)`` dispatches to functions registered with
:func:`register_plot` (done by ``nowcastbox.visualization`` at import time). Until a
plot is registered it raises :class:`NotImplementedError`.

Analysis methods
----------------
:meth:`NowcastResults.news`, :meth:`~NowcastResults.nowcast_tracker`,
:meth:`~NowcastResults.level_contributions` (:mod:`nowcastbox.news`, I6),
:meth:`~NowcastResults.distribution` (:mod:`nowcastbox.density`, I5) and
:meth:`~NowcastResults.diagnostics` (:mod:`nowcastbox.diagnostics`, I9) delegate to the
analysis subpackages, imported lazily (they depend on ``core``, not the reverse).
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import pickle
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pandas as pd

from nowcastbox.core.data import MixedFrequencyData, StandardizationStats
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import Frequency

if TYPE_CHECKING:  # analysis modules depend on core, never the reverse at runtime
    from nowcastbox.density import EmpiricalQuantileDistribution, NowcastDistribution
    from nowcastbox.diagnostics import DiagnosticsReport
    from nowcastbox.news import LevelContributions, NewsResults, NowcastTracker

__all__ = [
    "NOWCAST_COLUMNS",
    "FactorResults",
    "NowcastResults",
    "available_plots",
    "build_nowcast_frame",
    "register_plot",
]

NOWCAST_COLUMNS: tuple[str, str, str] = ("observed", "in_sample", "out_of_sample")
"""Mandatory columns of :attr:`NowcastResults.nowcast`."""

_R = TypeVar("_R", bound="NowcastResults")
PlotFunction = Callable[..., Any]
_PLOT_REGISTRY: dict[tuple[type, str], PlotFunction] = {}


def register_plot(
    kind: str, results_type: type[NowcastResults] | None = None
) -> Callable[[PlotFunction], PlotFunction]:
    """Register a plotting function for ``results.plot(kind)``.

    The function is called as ``func(results, **kwargs)`` and should return the figure
    object it creates. Registrations are looked up along the MRO of the results class,
    so a plot registered for :class:`NowcastResults` works for every subclass, and a
    subclass may override it.

    Parameters
    ----------
    kind : str
        Plot name, e.g. ``"forecast"``, ``"factors"``, ``"eigenvalues"``.
    results_type : type, optional
        Results class the plot applies to (default :class:`NowcastResults`).

    Returns
    -------
    callable
        Decorator returning the function unchanged.

    Examples
    --------
    >>> from nowcastbox.core.results import register_plot, NowcastResults
    >>> @register_plot("my_kind")
    ... def _plot(results, **kwargs):
    ...     return "figure"
    """
    cls = NowcastResults if results_type is None else results_type

    def decorator(func: PlotFunction) -> PlotFunction:
        _PLOT_REGISTRY[(cls, kind)] = func
        return func

    return decorator


def _lookup_plot(results_type: type, kind: str) -> PlotFunction | None:
    for klass in results_type.__mro__:
        func = _PLOT_REGISTRY.get((klass, kind))
        if func is not None:
            return func
    return None


def available_plots(results_type: type[NowcastResults] | NowcastResults) -> list[str]:
    """List plot kinds registered for a results class (or instance).

    Parameters
    ----------
    results_type : type or NowcastResults
        Results class or object.

    Returns
    -------
    list of str
        Sorted plot kinds.

    Examples
    --------
    >>> from nowcastbox.core.results import available_plots, NowcastResults
    >>> isinstance(available_plots(NowcastResults), list)
    True
    """
    klass = results_type if isinstance(results_type, type) else type(results_type)
    mro = set(klass.__mro__)
    return sorted({kind for (k, kind) in _PLOT_REGISTRY if k in mro})


def build_nowcast_frame(
    observed: pd.Series,
    estimate: pd.Series,
    *,
    extra: Mapping[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Assemble a ``nowcast`` frame from observed values and model estimates.

    Periods with an observed value receive the estimate in ``in_sample``; the others
    in ``out_of_sample``. The index is the union of both indexes.

    Parameters
    ----------
    observed : pandas.Series
        Target values on its native :class:`pandas.PeriodIndex` (NaN if missing).
    estimate : pandas.Series
        Model estimates on the same kind of index.
    extra : mapping of str to pandas.Series, optional
        Additional columns (e.g. interval bounds).

    Returns
    -------
    pandas.DataFrame
        Frame with columns :data:`NOWCAST_COLUMNS` (+ extras).

    Raises
    ------
    NowcastDataError
        If the indexes are not PeriodIndexes of the same frequency.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.results import build_nowcast_frame
    >>> idx = pd.period_range("2020Q1", periods=3, freq="Q")
    >>> obs = pd.Series([1.0, 2.0, np.nan], index=idx)
    >>> est = pd.Series([1.1, 1.9, 2.5], index=idx)
    >>> build_nowcast_frame(obs, est)["out_of_sample"].tolist()
    [nan, nan, 2.5]
    """
    obs_index, est_index = observed.index, estimate.index
    if not isinstance(obs_index, pd.PeriodIndex) or not isinstance(est_index, pd.PeriodIndex):
        raise NowcastDataError("observed and estimate must be indexed by a PeriodIndex.")
    if obs_index.freqstr != est_index.freqstr:
        raise NowcastDataError("observed and estimate must have the same frequency.")
    index = observed.index.union(estimate.index).sort_values()
    obs = observed.reindex(index).astype(float)
    est = estimate.reindex(index).astype(float)
    has_obs = obs.notna()
    frame = pd.DataFrame(
        {
            "observed": obs,
            "in_sample": est.where(has_obs),
            "out_of_sample": est.where(~has_obs),
        },
        index=index,
    )
    for name, col in (extra or {}).items():
        frame[name] = col.reindex(index).astype(float)
    frame.index.name = "period"
    return frame


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float | np.floating):
        return "nan" if np.isnan(value) else f"{float(value):.4f}"
    return str(value)


def _validated_nowcast(nowcast: object) -> pd.DataFrame:
    """Check the ``nowcast`` frame contract and return a copy."""
    if not isinstance(nowcast, pd.DataFrame):
        raise NowcastDataError("nowcast must be a pandas DataFrame.")
    if not isinstance(nowcast.index, pd.PeriodIndex):
        raise NowcastDataError("nowcast must be indexed by a pandas PeriodIndex.")
    missing = [c for c in NOWCAST_COLUMNS if c not in nowcast.columns]
    if missing:
        raise NowcastDataError(f"nowcast is missing the columns {missing}.")
    return nowcast.copy()


@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class NowcastResults:
    """Immutable container of the output of a fitted nowcasting model.

    Parameters
    ----------
    target : str
        Name of the target series.
    nowcast : pandas.DataFrame
        Target-frequency frame with columns ``observed``, ``in_sample`` and
        ``out_of_sample`` (see module docstring) indexed by a PeriodIndex.
    model_name : str, default ""
        Name of the estimator (e.g. ``"TwoStepDFM"``).
    model_params : dict, default {}
        Estimator hyper-parameters (``estimator.get_params()``).
    factors : pandas.DataFrame, optional
        Smoothed factors on the base grid (columns ``f1, f2, ...``).
    loadings : pandas.DataFrame, optional
        Factor loadings (rows: series, columns: factors).
    params : dict, default {}
        Estimated parameters (matrices, coefficients...), model specific.
    loglikelihood : float, optional
        Log-likelihood at the estimate.
    n_iter : int, optional
        Number of iterations of the estimation algorithm.
    converged : bool, optional
        Whether the algorithm converged.
    data : MixedFrequencyData, optional
        Estimation panel (as used by the model).
    standardization : StandardizationStats, optional
        Statistics used to standardise the data.
    info : dict, default {}
        Free-form extra information (timings, settings, diagnostics).

    Raises
    ------
    NowcastDataError
        If ``nowcast`` is not a DataFrame with a PeriodIndex and the mandatory columns.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.results import NowcastResults, build_nowcast_frame
    >>> idx = pd.period_range("2020Q1", periods=3, freq="Q")
    >>> frame = build_nowcast_frame(
    ...     pd.Series([1.0, 2.0, np.nan], index=idx), pd.Series([1.1, 1.9, 2.5], index=idx)
    ... )
    >>> res = NowcastResults(target="gdp", nowcast=frame, model_name="Demo")
    >>> res.get_nowcast()
    2.5
    >>> res.target_frequency
    <Frequency.QUARTERLY: 'Q'>
    """

    target: str
    nowcast: pd.DataFrame
    model_name: str = ""
    model_params: dict[str, Any] = field(default_factory=dict[str, Any])
    factors: pd.DataFrame | None = None
    loadings: pd.DataFrame | None = None
    params: dict[str, Any] = field(default_factory=dict[str, Any])
    loglikelihood: float | None = None
    n_iter: int | None = None
    converged: bool | None = None
    data: MixedFrequencyData | None = None
    standardization: StandardizationStats | None = None
    info: dict[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        object.__setattr__(self, "nowcast", _validated_nowcast(self.nowcast))
        for name in ("factors", "loadings"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, pd.DataFrame):
                raise NowcastDataError(f"{name} must be a pandas DataFrame or None.")
            if value is not None:
                object.__setattr__(self, name, value.copy())
        for name in ("model_params", "params", "info"):
            object.__setattr__(self, name, dict(getattr(self, name)))
        if self.loglikelihood is not None:
            object.__setattr__(self, "loglikelihood", float(self.loglikelihood))

    # ------------------------------------------------------------------ accessors
    @property
    def target_frequency(self) -> Frequency:
        """Native frequency of the target (frequency of the ``nowcast`` index)."""
        index = self.nowcast.index
        assert isinstance(index, pd.PeriodIndex)  # noqa: S101
        return Frequency.from_index(index)

    @property
    def observed(self) -> pd.Series:
        """Observed target values (copy of ``nowcast["observed"]``)."""
        return self.nowcast["observed"].copy()

    @property
    def in_sample(self) -> pd.Series:
        """Fitted values where the target is observed."""
        return self.nowcast["in_sample"].copy()

    @property
    def out_of_sample(self) -> pd.Series:
        """Backcasts, nowcasts and forecasts where the target is not observed."""
        return self.nowcast["out_of_sample"].copy()

    @property
    def estimate(self) -> pd.Series:
        """Model estimate for every period (``in_sample`` combined with ``out_of_sample``)."""
        return (
            self.nowcast["in_sample"]
            .combine_first(self.nowcast["out_of_sample"])
            .rename("estimate")
        )

    @property
    def n_factors(self) -> int | None:
        """Number of factors (from ``factors`` or ``loadings``), ``None`` if not a factor model."""
        if self.factors is not None:
            return self.factors.shape[1]
        if self.loadings is not None:
            return self.loadings.shape[1]
        return None

    def get_nowcast(self, period: pd.Period | str | None = None) -> float:
        """Return the model estimate for one target period.

        Parameters
        ----------
        period : pandas.Period or str, optional
            Target period (e.g. ``"2020Q2"``). Default: the first period after the last
            observed value of the target (the current nowcast).

        Returns
        -------
        float
            Estimate (NaN if the model produced none for that period).

        Raises
        ------
        KeyError
            If the period is outside the ``nowcast`` index or there is no period after
            the last observation.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2020Q1", periods=2, freq="Q")
        >>> f = build_nowcast_frame(
        ...     pd.Series([1.0, np.nan], index=idx), pd.Series([0.9, 2.0], index=idx)
        ... )
        >>> NowcastResults(target="y", nowcast=f).get_nowcast("2020Q1")
        0.9
        """
        estimate = self.estimate
        if period is None:
            last = self.nowcast["observed"].last_valid_index()
            later = estimate.index if last is None else estimate.index[estimate.index > last]
            if len(later) == 0:
                raise KeyError("No period after the last observation of the target.")
            key = later[0]
        else:
            key = pd.Period(period, freq=self.target_frequency.pandas_freq)
        matches = np.flatnonzero(estimate.index == key)
        if matches.size == 0:
            raise KeyError(f"Period {key} is not in the nowcast index.")
        return float(estimate.to_numpy(dtype=float)[matches[0]])

    def replace(self: _R, **changes: Any) -> _R:
        """Return a copy with some fields replaced.

        Parameters
        ----------
        **changes
            Field values.

        Returns
        -------
        NowcastResults
            New results object of the same class.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020Q1", periods=1, freq="Q")
        >>> f = build_nowcast_frame(pd.Series([1.0], index=idx), pd.Series([1.0], index=idx))
        >>> NowcastResults(target="y", nowcast=f).replace(model_name="X").model_name
        'X'
        """
        return dataclasses.replace(self, **changes)

    # ------------------------------------------------------------------ output
    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the ``nowcast`` frame.

        Returns
        -------
        pandas.DataFrame
            Nowcast table.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020Q1", periods=1, freq="Q")
        >>> f = build_nowcast_frame(pd.Series([1.0], index=idx), pd.Series([1.0], index=idx))
        >>> list(NowcastResults(target="y", nowcast=f).to_frame().columns)
        ['observed', 'in_sample', 'out_of_sample']
        """
        return self.nowcast.copy()

    def _summary_header(self) -> list[tuple[str, str]]:
        data = self.data
        sample = f"{data.start} .. {data.end}" if data is not None else "-"
        n_series = str(data.n_series) if data is not None else "-"
        return [
            ("Target", f"{self.target} ({self.target_frequency.label})"),
            ("Sample", sample),
            ("Series", n_series),
            ("Factors", _fmt(self.n_factors)),
            ("Log-likelihood", _fmt(self.loglikelihood)),
            ("Iterations", _fmt(self.n_iter)),
            ("Converged", _fmt(self.converged)),
        ]

    def _summary_sections(self) -> list[tuple[str, list[str]]]:
        """Extra summary sections ``(title, lines)``; subclasses extend this hook."""
        return []

    def summary(self, *, n_periods: int = 8) -> str:
        """Formatted text summary of the results.

        Parameters
        ----------
        n_periods : int, default 8
            Number of most recent target periods shown in the nowcast table.

        Returns
        -------
        str
            Multi-line summary.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020Q1", periods=1, freq="Q")
        >>> f = build_nowcast_frame(pd.Series([1.0], index=idx), pd.Series([1.0], index=idx))
        >>> "Nowcast Results" in NowcastResults(target="y", nowcast=f).summary()
        True
        """
        width = 78
        title = f"Nowcast Results: {self.model_name}" if self.model_name else "Nowcast Results"
        lines = ["=" * width, title.center(width).rstrip(), "=" * width]
        for key, value in self._summary_header():
            lines.append(f"{key + ':':<24}{value}")
        if self.model_params:
            lines += ["-" * width, "Model parameters"]
            key_width = max(22, *(len(str(k)) + 2 for k in self.model_params))
            lines += [f"  {k:<{key_width}}{_fmt(v)}" for k, v in self.model_params.items()]
        for section_title, section_lines in self._summary_sections():
            lines += ["-" * width, section_title, *section_lines]
        table = self.nowcast.loc[:, list(NOWCAST_COLUMNS)].tail(n_periods)
        lines += ["-" * width, "Nowcast (most recent periods)"]
        lines += table.to_string(float_format=lambda x: f"{x:.4f}", na_rep="").splitlines()
        lines.append("=" * width)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        index = self.nowcast.index
        span = f"{index[0]}..{index[-1]}" if len(index) else "empty"
        name = self.model_name or type(self).__name__
        return f"{type(self).__name__}(model={name!r}, target={self.target!r}, periods={span})"

    def plot(self, kind: str = "forecast", **kwargs: Any) -> Any:
        """Plot the results using a registered plotting function.

        Parameters
        ----------
        kind : str, default "forecast"
            Plot kind (``"forecast"``, ``"factors"``, ``"eigenvalues"``, ``"loadings"``...,
            see :func:`available_plots`).
        **kwargs
            Passed to the plotting function.

        Returns
        -------
        object
            Figure returned by the plotting function.

        Raises
        ------
        NotImplementedError
            If no plot of that kind is registered (the visualization module is not
            available yet or does not provide it).

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020Q1", periods=1, freq="Q")
        >>> f = build_nowcast_frame(pd.Series([1.0], index=idx), pd.Series([1.0], index=idx))
        >>> NowcastResults(target="y", nowcast=f).plot("forecast")  # doctest: +SKIP
        """
        func = _lookup_plot(type(self), kind)
        if func is None:
            with contextlib.suppress(ImportError):
                importlib.import_module("nowcastbox.visualization")
            func = _lookup_plot(type(self), kind)
        if func is None:
            raise NotImplementedError(
                f"No plot {kind!r} registered for {type(self).__name__}; "
                f"available: {available_plots(type(self))}."
            )
        return func(self, **kwargs)

    # ------------------------------------------------------------------ analysis
    def news(self, old: Any, new: Any, target_period: Any = None, **kwargs: Any) -> NewsResults:
        """News decomposition of the nowcast between two vintages (I6).

        Delegates to :func:`nowcastbox.news.news_decomposition` (Bańbura & Modugno,
        2014): the revision of the nowcast is split into data revisions, the impact of
        each new release (news x weight) and re-estimation.

        Parameters
        ----------
        old, new : MixedFrequencyData or pandas.DataFrame
            Old and new vintages.
        target_period : period-like, optional
            Target period (default: the nowcast period of ``new``).
        **kwargs
            ``new_results`` and ``categories`` (see
            :func:`~nowcastbox.news.news_decomposition`).

        Returns
        -------
        NewsResults
            Decomposition with ``summary()``, ``to_frame(by=...)`` and ``plot()``.

        Raises
        ------
        TypeError
            If the results are not from a supported state-space factor model.

        Examples
        --------
        >>> news = res.news(old=v_old, new=v_new, target_period="2015Q2")  # doctest: +SKIP
        >>> news.summary()  # doctest: +SKIP
        """
        from nowcastbox.news import news_decomposition

        return news_decomposition(self, old, new, target_period, **kwargs)

    def nowcast_tracker(
        self,
        data: Any,
        calendar: Any = None,
        target_period: Any = None,
        start: Any = None,
        end: Any = None,
        **kwargs: Any,
    ) -> NowcastTracker:
        """Nowcast path through the vintages of a period, with contributions (I6).

        Delegates to :func:`nowcastbox.news.nowcast_tracker` with these (fixed)
        parameters.

        Parameters
        ----------
        data : MixedFrequencyData, pandas.DataFrame or VintageStore
            Final data (pseudo real-time vintages are built with ``calendar``) or a
            vintage store.
        calendar : ReleaseCalendar, int or mapping, optional
            Publication delays.
        target_period : period-like
            Target period (required, e.g. ``"2015Q2"``; ``None`` raises ``ValueError``).
        start, end : date-like, optional
            First and last vintage date.
        **kwargs
            Further options of :func:`~nowcastbox.news.nowcast_tracker` (``dates``,
            ``by``, ``refit``...).

        Returns
        -------
        NowcastTracker
            Nowcast path and cumulative contributions.

        Examples
        --------
        >>> tracker = res.nowcast_tracker(data, calendar, "2015Q2")  # doctest: +SKIP
        """
        from nowcastbox.news import nowcast_tracker

        return nowcast_tracker(self, data, calendar, target_period, start, end, **kwargs)

    def level_contributions(
        self, data: Any = None, target_period: Any = None, **kwargs: Any
    ) -> LevelContributions:
        """Contribution of every series to the level of the nowcast (I6).

        Delegates to :func:`nowcastbox.news.level_contributions` (Koopman & Harvey,
        2003 observation weights).

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame, optional
            Vintage (default: the estimation data).
        target_period : period-like, optional
            Target period.
        **kwargs
            ``categories`` override.

        Returns
        -------
        LevelContributions
            Baseline plus one exact contribution per series.

        Examples
        --------
        >>> res.level_contributions().to_frame("category")  # doctest: +SKIP
        """
        from nowcastbox.news import level_contributions

        return level_contributions(self, data, target_period, **kwargs)

    def distribution(self, **kwargs: Any) -> NowcastDistribution | EmpiricalQuantileDistribution:
        """Predictive distribution of the nowcast (density nowcast, I5).

        Delegates to :func:`nowcastbox.density.nowcast_distribution`: Gaussian
        filtering uncertainty, plus parameter uncertainty from a bootstrap when
        ``n_boot > 0``. With ``method="empirical"`` it delegates instead to
        :func:`nowcastbox.density.empirical_bands`: bands from the past errors of a
        backtest at the same horizon (Reifschneider-Tulip / ECB style).

        Parameters
        ----------
        **kwargs
            Options of :func:`~nowcastbox.density.nowcast_distribution` (``n_boot``,
            ``method``, ``periods``, ``random_state``, ``n_jobs``...). With
            ``method="empirical"``: ``backtest`` (required), ``empirical_method``
            (``"mae"``, ``"rmse"`` or ``"quantile"``, passed as ``method``) and the
            other options of :func:`~nowcastbox.density.empirical_bands` (``vintage``,
            ``window``, ``levels``, ``outliers``...).

        Returns
        -------
        NowcastDistribution or EmpiricalQuantileDistribution
            Distribution with quantiles, intervals, sampling and ``plot()``
            (an :class:`~nowcastbox.density.EmpiricalQuantileDistribution` only for
            ``method="empirical", empirical_method="quantile"``).

        Raises
        ------
        NowcastDataError
            If the results have no standard deviation for the requested periods (or
            no period has enough past errors).
        ValueError
            If ``method="empirical"`` is given without ``backtest``.

        Examples
        --------
        >>> res.distribution(n_boot=199, random_state=0).interval(0.9)  # doctest: +SKIP
        >>> res.distribution(method="empirical", backtest=bt).interval(0.9)  # doctest: +SKIP
        """
        if kwargs.get("method") == "empirical":
            from nowcastbox.density import empirical_bands

            options = {k: v for k, v in kwargs.items() if k != "method"}
            backtest = options.pop("backtest", None)
            if backtest is None:
                raise ValueError('distribution(method="empirical") requires backtest=...')
            options["method"] = options.pop("empirical_method", "mae")
            return empirical_bands(self, backtest, **options)
        from nowcastbox.density import nowcast_distribution

        return nowcast_distribution(self, **kwargs)

    def diagnostics(self, data: Any = None, **kwargs: Any) -> DiagnosticsReport:
        """Diagnostics of the factor model (I9).

        Delegates to :func:`nowcastbox.diagnostics.run_diagnostics`: loading
        stability (Breitung & Eickmeier, 2011), EM convergence, factor contributions,
        data quality and residual tests.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame, optional
            Panel (default: the estimation data).
        **kwargs
            Options of :func:`~nowcastbox.diagnostics.run_diagnostics`.

        Returns
        -------
        DiagnosticsReport
            Report with ``summary()``, ``to_frames()`` and ``series_overview()``.

        Examples
        --------
        >>> print(res.diagnostics().summary())  # doctest: +SKIP
        """
        from nowcastbox.diagnostics import run_diagnostics

        return run_diagnostics(self, data, **kwargs)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        """Save the results with :mod:`pickle`.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file.

        Returns
        -------
        pathlib.Path
            The path written.

        Examples
        --------
        >>> import pandas as pd, tempfile, pathlib
        >>> idx = pd.period_range("2020Q1", periods=1, freq="Q")
        >>> f = build_nowcast_frame(pd.Series([1.0], index=idx), pd.Series([1.0], index=idx))
        >>> res = NowcastResults(target="y", nowcast=f)
        >>> p = res.save(pathlib.Path(tempfile.mkdtemp()) / "res.pkl")
        >>> NowcastResults.load(p).target
        'y'
        """
        out = Path(path)
        with out.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return out

    @classmethod
    def load(cls: type[_R], path: str | Path) -> _R:
        """Load results saved with :meth:`save`.

        Only load files from trusted sources: unpickling can execute arbitrary code.

        Parameters
        ----------
        path : str or pathlib.Path
            File written by :meth:`save`.

        Returns
        -------
        NowcastResults
            The loaded results (instance of ``cls`` or a subclass).

        Raises
        ------
        TypeError
            If the file does not contain an instance of ``cls``.

        Examples
        --------
        >>> NowcastResults.load("results.pkl")  # doctest: +SKIP
        """
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__} object.")
        return obj


@dataclass(frozen=True, kw_only=True, eq=False, repr=False)
class FactorResults(NowcastResults):
    r"""Results of a dynamic factor model (common base of two-step and EM results).

    Notation follows plan section 4.1: :math:`x_t = \Lambda f_t + \varepsilon_t`,
    :math:`f_t = \sum_{i=1}^p A_i f_{t-i} + B u_t`.

    Parameters
    ----------
    transition : numpy.ndarray, optional
        Stacked VAR matrices :math:`[A_1, \dots, A_p]` of shape ``(r, r * p)``.
    shock_loadings : numpy.ndarray, optional
        :math:`B` of shape ``(r, q)``.
    factor_lags : int, optional
        VAR order :math:`p`.
    n_shocks : int, optional
        Number of dynamic shocks :math:`q`.
    idiosyncratic_variance : pandas.Series, optional
        Diagonal of :math:`\Psi`, indexed by series.

    Notes
    -----
    Every field of :class:`NowcastResults` is also accepted (keyword-only).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.results import FactorResults, build_nowcast_frame
    >>> idx = pd.period_range("2020Q1", periods=1, freq="Q")
    >>> f = build_nowcast_frame(pd.Series([1.0], index=idx), pd.Series([1.0], index=idx))
    >>> res = FactorResults(target="y", nowcast=f, transition=np.eye(2), factor_lags=1)
    >>> res.transition_matrices()[0].shape
    (2, 2)
    """

    transition: np.ndarray | None = None
    shock_loadings: np.ndarray | None = None
    factor_lags: int | None = None
    n_shocks: int | None = None
    idiosyncratic_variance: pd.Series | None = None

    def transition_matrices(self) -> list[np.ndarray]:
        r"""Split :attr:`transition` into the list :math:`[A_1, \dots, A_p]`.

        Returns
        -------
        list of numpy.ndarray
            ``p`` matrices of shape ``(r, r)``.

        Raises
        ------
        ValueError
            If ``transition`` or ``factor_lags`` is missing or inconsistent.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2020Q1", periods=1, freq="Q")
        >>> f = build_nowcast_frame(pd.Series([1.0], index=idx), pd.Series([1.0], index=idx))
        >>> A = np.hstack([np.eye(2), 0.5 * np.eye(2)])
        >>> res = FactorResults(target="y", nowcast=f, transition=A, factor_lags=2)
        >>> [float(a[0, 0]) for a in res.transition_matrices()]
        [1.0, 0.5]
        """
        if self.transition is None or self.factor_lags is None:
            raise ValueError("transition and factor_lags are required.")
        r, cols = self.transition.shape
        if cols != r * self.factor_lags:
            raise ValueError(
                f"transition has shape {self.transition.shape}, expected ({r}, {r * self.factor_lags})."
            )
        return [self.transition[:, i * r : (i + 1) * r].copy() for i in range(self.factor_lags)]

    def _summary_sections(self) -> list[tuple[str, list[str]]]:
        sections = super()._summary_sections()
        lines = [
            f"  {'Factor lags (p)':<22}{_fmt(self.factor_lags)}",
            f"  {'Dynamic shocks (q)':<22}{_fmt(self.n_shocks)}",
        ]
        if self.transition is not None:
            eig = np.abs(np.linalg.eigvals(_companion(self.transition)))
            lines.append(f"  {'Max |eigenvalue| VAR':<22}{_fmt(float(eig.max()))}")
        sections.append(("Factor dynamics", lines))
        return sections


def _companion(transition: np.ndarray) -> np.ndarray:
    """Companion matrix of a stacked VAR coefficient matrix ``(r, r * p)``."""
    r, rp = transition.shape
    if rp % r != 0:
        raise ValueError(f"transition shape {transition.shape} is not (r, r * p).")
    comp = np.zeros((rp, rp))
    comp[:r, :] = transition
    comp[r:, :-r] = np.eye(rp - r)
    return comp

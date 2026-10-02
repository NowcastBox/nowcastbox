"""Model-building stages of a nowcast spec (``selection``; ECB toolbox parity, 0.3.0).

The optional ``selection`` section of a :class:`~nowcastbox.pipeline.NowcastSpec`
chains the model-building workflow of the ECB Nowcasting Toolbox (Linzenich & Meunier,
2024, ECB WP 3004, §2) in front of the production nowcast:

* ``selection.preselect`` ranks the indicators against the target with
  :func:`~nowcastbox.selection.preselect` (t-stat, SIS and LARS rankings, weighted
  score) on the information set of the run's vintage. With ``apply: true`` (default)
  the model is fitted on the selected indicators only, and the ``backtest`` output
  repeats the pre-selection on every pseudo real-time vintage (no look-ahead).
* ``selection.search`` runs a :class:`~nowcastbox.selection.SpecificationSearch` of
  the spec's model (its fitted parameters are the template; ``space`` lists the
  searched settings), optionally followed by the Covid robustness step
  (``covid_robustness``). With ``apply: true`` the best specification (or the best
  specification-treatment pair of the robustness step) replaces the spec's model.

Example::

    selection:
      preselect: {methods: [tstat, sis, lars], x_lags: [0, 1], top: 20}
      search:
        space: {n_factors: [1, 2], n_series: [10, 20], start: [2005-01, 2010-01]}
        n_draws: 20
        backtest: {start: 2017-01-15, end: 2019-12-15}
        score: {rmsfe: 0.7, fda: 0.3}
        covid_robustness: {top: 3, evaluate_from: 2022Q1}
        apply: true
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.pipeline._common import plain as _plain

if TYPE_CHECKING:
    from nowcastbox.core.data import MixedFrequencyData
    from nowcastbox.pipeline.spec import NowcastSpec
    from nowcastbox.selection import CovidRobustness, PreselectionResult, SearchResults

__all__ = [
    "PRESELECT_EXCLUDED",
    "ROBUSTNESS_KEYS",
    "SEARCH_BACKTEST_KEYS",
    "SEARCH_KEYS",
    "SEARCH_OPTIONS",
    "PreselectSpec",
    "SearchSpec",
    "SelectionSpec",
]

PRESELECT_EXCLUDED: tuple[str, ...] = ("data", "target", "as_of", "release_delays")
"""Arguments of :func:`~nowcastbox.selection.preselect` set by the pipeline."""

SEARCH_KEYS: tuple[str, ...] = (
    "space",
    "n_draws",
    "ranking",
    "backtest",
    "score",
    "horizon_weights",
    "random_state",
    "n_jobs",
    "checkpoint",
    "normalize",
    "periods",
    "horizon",
    "metrics",
    "max_missing",
    "covid_robustness",
    "apply",
)
"""Keys of ``selection.search``."""

SEARCH_BACKTEST_KEYS: tuple[str, ...] = (
    "start",
    "end",
    "step",
    "target_offsets",
    "window",
    "window_length",
    "refit_every",
    "n_vintages",
    "benchmarks",
)
"""Keys of ``selection.search.backtest`` (the delays and preprocessing come from the spec)."""

ROBUSTNESS_KEYS: tuple[str, ...] = ("top", "treatments", "evaluate_from")
"""Keys of ``selection.search.covid_robustness``."""
SEARCH_OPTIONS: tuple[str, ...] = (
    "score",
    "horizon_weights",
    "random_state",
    "n_jobs",
    "normalize",
    "periods",
    "horizon",
    "metrics",
    "max_missing",
)
"""Keys of ``selection.search`` passed unchanged to ``SpecificationSearch``."""


@dataclasses.dataclass(frozen=True)
class PreselectSpec:
    """Pre-selection of the indicators (``selection.preselect``).

    Parameters
    ----------
    options : dict
        Keyword arguments of :func:`~nowcastbox.selection.preselect` (``methods``,
        ``x_lags``, ``weights``, ``top``, ``horizon``, ``aggregation``...); the data,
        the target and the vintage come from the spec.
    apply : bool, default True
        Fit the model on the selected indicators (and the target) only.

    Examples
    --------
    >>> PreselectSpec(options={"top": 10}).to_dict()
    {'top': 10, 'apply': True}
    """

    options: dict[str, Any] = dataclasses.field(default_factory=dict)
    apply: bool = True

    @property
    def ranking_options(self) -> dict[str, Any]:
        """Options of the per-vintage ranking of a funnel (``top`` excluded).

        Returns
        -------
        dict
            :attr:`options` without ``top`` (the funnel size is set by ``n_series``).

        Examples
        --------
        >>> PreselectSpec(options={"top": 10, "x_lags": [0, 1]}).ranking_options
        {'x_lags': [0, 1]}
        """
        return {k: v for k, v in self.options.items() if k != "top"}

    def to_dict(self) -> dict[str, Any]:
        """YAML-ready mapping.

        Returns
        -------
        dict
            The ``selection.preselect`` section.

        Examples
        --------
        >>> PreselectSpec(apply=False).to_dict()
        {'apply': False}
        """
        return {**_plain(self.options), "apply": self.apply}


@dataclasses.dataclass(frozen=True)
class SearchSpec:
    """Specification search (``selection.search``).

    Parameters
    ----------
    space : dict
        Searched settings (:class:`~nowcastbox.selection.ParameterSpace`): model
        parameters, ``start`` and ``n_series``.
    n_draws : int or None, default 100
        Random draws; ``None`` (``n_draws: null``) evaluates the exhaustive grid.
    ranking : str, list or dict, optional
        Funnel ranking: ``"preselect"``, :func:`~nowcastbox.selection.preselect`
        options or a list of names. Default with ``n_series`` in ``space``: the
        options of ``selection.preselect`` (recomputed on every vintage).
    backtest : dict
        Vintage window of the search (``start``, ``end``, ``step``, ``refit_every``,
        ``n_vintages``...); delays and preprocessing come from the spec.
    benchmarks : tuple of (str, dict)
        Benchmarks of the search backtest (needed by ``normalize: relative``).
    options : dict
        Other arguments of :class:`~nowcastbox.selection.SpecificationSearch`
        (``score``, ``horizon_weights``, ``random_state``, ``n_jobs``, ``normalize``,
        ``periods``, ``horizon``, ``metrics``, ``max_missing``).
    checkpoint : pathlib.Path, optional
        Parquet checkpoint of the search (resumed when the settings match).
    robustness : dict, optional
        Arguments of :meth:`~nowcastbox.selection.SearchResults.covid_robustness`
        (``top``, ``treatments``, ``evaluate_from``); ``None`` skips the step.
    apply : bool, default False
        Replace the spec's model by the best specification (of the robustness step
        when it runs).

    Examples
    --------
    >>> SearchSpec(space={"n_factors": [1, 2]}, n_draws=None).to_dict()["n_draws"] is None
    True
    """

    space: dict[str, Any]
    n_draws: int | None = 100
    ranking: Any = None
    backtest: dict[str, Any] = dataclasses.field(default_factory=dict)
    benchmarks: tuple[tuple[str, dict[str, Any]], ...] = ()
    options: dict[str, Any] = dataclasses.field(default_factory=dict)
    checkpoint: Path | None = None
    robustness: dict[str, Any] | None = None
    apply: bool = False

    def to_dict(self) -> dict[str, Any]:
        """YAML-ready mapping.

        Returns
        -------
        dict
            The ``selection.search`` section.

        Examples
        --------
        >>> sorted(SearchSpec(space={"n_factors": [1, 2]}).to_dict())
        ['apply', 'backtest', 'n_draws', 'space']
        """
        backtest = dict(self.backtest)
        if self.benchmarks:
            backtest["benchmarks"] = [{"type": n, **kw} for n, kw in self.benchmarks]
        out: dict[str, Any] = {
            "space": _plain(self.space),
            "n_draws": self.n_draws,
            "backtest": _plain(backtest),
        }
        if self.ranking is not None:
            out["ranking"] = _plain(self.ranking)
        out.update(_plain(self.options))
        if self.checkpoint is not None:
            out["checkpoint"] = str(self.checkpoint)
        if self.robustness is not None:
            out["covid_robustness"] = _plain(self.robustness)
        out["apply"] = self.apply
        return out


@dataclasses.dataclass(frozen=True)
class SelectionSpec:
    """Model-building stages (``selection``).

    Parameters
    ----------
    preselect : PreselectSpec, optional
        Pre-selection of the indicators.
    search : SearchSpec, optional
        Specification search.

    Examples
    --------
    >>> SelectionSpec().to_dict(), SelectionSpec().empty
    ({}, True)
    """

    preselect: PreselectSpec | None = None
    search: SearchSpec | None = None

    @property
    def empty(self) -> bool:
        """``True`` when no stage is requested."""
        return self.preselect is None and self.search is None

    def to_dict(self) -> dict[str, Any]:
        """YAML-ready mapping.

        Returns
        -------
        dict
            The ``selection`` section (empty when no stage is requested).

        Examples
        --------
        >>> SelectionSpec(preselect=PreselectSpec()).to_dict()
        {'preselect': {'apply': True}}
        """
        out: dict[str, Any] = {}
        if self.preselect is not None:
            out["preselect"] = self.preselect.to_dict()
        if self.search is not None:
            out["search"] = self.search.to_dict()
        return out


# ---------------------------------------------------------------------------- running
def run_preselection(spec: NowcastSpec, panel: MixedFrequencyData) -> PreselectionResult:
    """Pre-select the indicators on the run's panel (``selection.preselect``).

    Parameters
    ----------
    spec : NowcastSpec
        Spec with a ``selection.preselect`` stage.
    panel : MixedFrequencyData
        Model-ready panel of the vintage (only data released by then).

    Returns
    -------
    PreselectionResult
        Rankings, aggregated score and selected indicators.

    Raises
    ------
    ValueError
        If the spec has no pre-selection stage.

    Examples
    --------
    >>> from nowcastbox.pipeline.spec import NowcastSpec
    >>> from nowcastbox.datasets import load_simulated_dfm
    >>> spec = NowcastSpec.from_dict(
    ...     {
    ...         "target": "gdp",
    ...         "data": {"source": "simulated_dfm"},
    ...         "selection": {"preselect": {"methods": ["sis"], "top": 3}},
    ...     }
    ... )
    >>> run_preselection(spec, load_simulated_dfm().data).n_selected
    3
    """
    stage = spec.selection.preselect
    if stage is None:
        raise ValueError("The spec has no selection.preselect stage.")
    from nowcastbox.selection import preselect

    return preselect(panel, spec.target, **stage.options)


def funnel_model(spec: NowcastSpec, estimator: Any, preselection: PreselectionResult) -> Any:
    """Model that repeats the pre-selection on each vintage it is fitted on.

    Parameters
    ----------
    spec : NowcastSpec
        Spec with a ``selection.preselect`` stage.
    estimator : nowcaster
        Unfitted model of the spec (with its fitted parameters).
    preselection : PreselectionResult
        Pre-selection of the run (sets the number of indicators kept).

    Returns
    -------
    SpecifiedModel
        The estimator behind a per-vintage funnel of ``preselection.n_selected``
        indicators (used by the backtest output: no look-ahead).

    Examples
    --------
    >>> import nowcastbox as nb
    >>> from nowcastbox.pipeline.spec import NowcastSpec
    >>> spec = NowcastSpec.from_dict(
    ...     {
    ...         "target": "gdp",
    ...         "data": {"source": "simulated_dfm"},
    ...         "selection": {"preselect": {"methods": ["sis"], "top": 3}},
    ...     }
    ... )
    >>> pre = run_preselection(spec, nb.load_simulated_dfm().data)
    >>> funnel_model(spec, nb.TwoStepDFM(n_factors=1), pre).n_series
    3
    """
    from nowcastbox.selection import SpecifiedModel

    stage = spec.selection.preselect
    options = {} if stage is None else stage.ranking_options
    return SpecifiedModel(
        type(estimator),
        params=estimator.get_params(deep=False),
        n_series=preselection.n_selected,
        ranking=options,
    )


def search_ranking(spec: NowcastSpec) -> Any:
    """Funnel ranking of the search (explicit, or the pre-selection options).

    Parameters
    ----------
    spec : NowcastSpec
        Spec with a ``selection.search`` stage.

    Returns
    -------
    str, list, dict or None
        ``selection.search.ranking`` when given; otherwise, when ``n_series`` is
        searched, the ``selection.preselect`` options (recomputed on every vintage) or
        ``"preselect"``; ``None`` without a funnel.

    Examples
    --------
    >>> from nowcastbox.pipeline.spec import NowcastSpec
    >>> spec = NowcastSpec.from_dict(
    ...     {
    ...         "target": "gdp",
    ...         "data": {"source": "simulated_dfm"},
    ...         "selection": {"search": {"space": {"n_series": [2, 4]}}},
    ...     }
    ... )
    >>> search_ranking(spec)
    'preselect'
    """
    stage = spec.selection.search
    if stage is None:
        return None
    if stage.ranking is not None:
        return stage.ranking
    if "n_series" not in stage.space:
        return None
    pre = spec.selection.preselect
    return "preselect" if pre is None or not pre.ranking_options else pre.ranking_options


def run_search(
    spec: NowcastSpec,
    data: MixedFrequencyData,
    template: Any,
    preprocess: Callable[[MixedFrequencyData], MixedFrequencyData] | None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> tuple[SearchResults, CovidRobustness | None]:
    """Run the specification search (and the Covid robustness step) of a spec.

    Parameters
    ----------
    spec : NowcastSpec
        Spec with a ``selection.search`` stage.
    data : MixedFrequencyData
        Data of the run's vintage, before preprocessing (the backtest builds its
        pseudo real-time vintages from the publication delays).
    template : nowcaster
        Unfitted model whose parameters the searched settings override.
    preprocess : callable, optional
        Preprocessing applied to every vintage (the spec's ``preprocessing``).
    fit_kwargs : mapping, optional
        Extra ``fit`` arguments of the model (e.g. ``horizon`` of ``MixedFreqDFM``).

    Returns
    -------
    (SearchResults, CovidRobustness or None)
        The search and, with ``covid_robustness``, the robustness step.

    Raises
    ------
    ValueError
        If the spec has no search stage or a search setting is invalid.
    NowcastDataError
        If a series has no publication delay.

    Examples
    --------
    >>> import warnings
    >>> import nowcastbox as nb
    >>> from nowcastbox.pipeline.spec import NowcastSpec
    >>> spec = NowcastSpec.from_dict(
    ...     {
    ...         "target": "gdp",
    ...         "data": {"source": "simulated_dfm"},
    ...         "selection": {
    ...             "search": {
    ...                 "space": {"n_factors": [1, 2]},
    ...                 "n_draws": None,
    ...                 "backtest": {"start": "2019-01-15", "end": "2019-06-15"},
    ...             }
    ...         },
    ...     }
    ... )
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     out, rob = run_search(spec, nb.load_simulated_dfm().data, nb.TwoStepDFM(), None)
    >>> len(out.table()), rob is None
    (2, True)
    """
    stage = spec.selection.search
    if stage is None:
        raise ValueError("The spec has no selection.search stage.")
    from nowcastbox import benchmarks as bench_module
    from nowcastbox.selection import SpecificationSearch

    delays = data.release_delays
    if delays.isna().any():
        missing = list(delays.index[delays.isna()])
        raise NowcastDataError(f"The search needs publication delays for every series {missing}.")
    backtest: dict[str, Any] = {
        **stage.backtest,
        "delay": {k: int(v) for k, v in delays.items()},
        "preprocess": preprocess,
    }
    if fit_kwargs:
        backtest["fit_kwargs"] = dict(fit_kwargs)
    if stage.benchmarks:
        backtest["benchmarks"] = [getattr(bench_module, n)(**kw) for n, kw in stage.benchmarks]
    options = {"random_state": spec.random_state, **stage.options}
    search = SpecificationSearch(
        template,
        data,
        spec.target_name,
        stage.space,
        n_draws=stage.n_draws,
        ranking=search_ranking(spec),
        backtest=backtest,
        checkpoint=stage.checkpoint,
        **options,
    )
    out = search.run()
    robustness = None if stage.robustness is None else out.covid_robustness(**stage.robustness)
    return out, robustness


def best_model(search: SearchResults, robustness: CovidRobustness | None) -> Any:
    """Unfitted best model: of the robustness step when it ran, else of the search.

    Parameters
    ----------
    search : SearchResults
        Results of the specification search.
    robustness : CovidRobustness or None
        Results of the robustness step.

    Returns
    -------
    SpecifiedModel
        Best specification (with the search's preprocessing).

    Examples
    --------
    >>> best_model(out, None)  # doctest: +SKIP
    """
    return robustness.best_model() if robustness is not None else search.best_model()

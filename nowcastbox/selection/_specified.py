"""A model specification found by the search: estimator settings, sample start and funnel.

:class:`SpecifiedModel` wraps an estimator with the non-parameter settings of a
specification - the first period of the estimation sample and the number of
indicators kept from a ranking (the "funnel" strategy of the ECB toolbox) - so that a
specification can be backtested and used like any other nowcaster. When the ranking
is a set of :func:`~nowcastbox.selection.preselect` options it is recomputed on the
data passed to ``fit``: inside a pseudo real-time backtest that is the information set
of the vintage, so the selection never uses data released later.
"""

from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from nowcastbox.core.base import BaseNowcaster
from nowcastbox.core.data import MixedFrequencyData, as_mixed_frequency_data
from nowcastbox.core.exceptions import ModelNotFittedError
from nowcastbox.core.frequency import Frequency, base_to_native, is_period_end
from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.selection.preselection import PreselectionResult, preselect

__all__ = ["SpecifiedModel", "configured_model", "ranking_names"]

_CACHE_SIZE = 256
_preselect_cache: OrderedDict[str, list[str]] = OrderedDict()

Ranking = Sequence[str] | Mapping[str, Any] | PreselectionResult | str | None


def configured_model(model: Any, params: Mapping[str, Any] | None = None) -> Any:
    """Unfitted copy of an estimator (or instance of a class) with parameters set.

    Parameters
    ----------
    model : estimator or estimator class
        Template.
    params : mapping, optional
        Parameters passed to ``set_params``.

    Returns
    -------
    estimator
        New estimator.

    Raises
    ------
    ValueError
        If parameters are given but the estimator has no ``set_params``, or a name
        is invalid.

    Examples
    --------
    >>> from nowcastbox.models import TwoStepDFM
    >>> configured_model(TwoStepDFM, {"n_factors": 3}).n_factors
    3
    """
    if isinstance(model, type):
        estimator = model()
    else:
        clone = getattr(model, "clone", None)
        estimator = clone() if callable(clone) else copy.deepcopy(model)
    if params:
        set_params = getattr(estimator, "set_params", None)
        if not callable(set_params):
            raise ValueError(f"{type(estimator).__name__} has no set_params; cannot set {params}.")
        set_params(**dict(params))
    return estimator


def ranking_names(ranking: Sequence[str] | PreselectionResult) -> tuple[str, ...]:
    """Series names of a static ranking, best first.

    Parameters
    ----------
    ranking : sequence of str or PreselectionResult
        Ordered names, or a pre-selection (its aggregated ranking, unranked series
        last).

    Returns
    -------
    tuple of str
        Names.

    Raises
    ------
    TypeError
        If ``ranking`` is not a sequence of strings or a pre-selection.

    Examples
    --------
    >>> ranking_names(["b", "a"])
    ('b', 'a')
    """
    if isinstance(ranking, PreselectionResult):
        return tuple(str(s) for s in ranking.ranking().index)
    if isinstance(ranking, str) or not all(isinstance(s, str) for s in ranking):
        raise TypeError("A ranking must be a sequence of series names or a PreselectionResult.")
    return tuple(ranking)


def _native(frame: pd.DataFrame, target: str, freq: Frequency) -> pd.Series:
    """Target values in their storage slots of a base-grid frame, on native periods."""
    column = frame[target]
    index = column.index
    assert isinstance(index, pd.PeriodIndex)  # noqa: S101
    slots = is_period_end(index, freq)
    out = column[slots]
    out.index = base_to_native(index[slots], freq)
    return out


def _panel_hash(panel: MixedFrequencyData, target: str, options: Mapping[str, Any]) -> str:
    hashed = pd.util.hash_pandas_object(panel.data, index=True).to_numpy()
    digest = hashlib.sha256(hashed.tobytes())
    metadata = [repr(panel.metadata[c]) for c in panel.columns]
    digest.update(repr((list(panel.columns), metadata, target, sorted(options.items()))).encode())
    return digest.hexdigest()


def preselect_ranking(
    panel: MixedFrequencyData, target: str, options: Mapping[str, Any]
) -> list[str]:
    """Pre-selection ranking of a panel (cached on the panel's content).

    Parameters
    ----------
    panel : MixedFrequencyData
        Information set (e.g. a pseudo real-time vintage).
    target : str
        Target series.
    options : mapping
        Keyword arguments of :func:`~nowcastbox.selection.preselect`.

    Returns
    -------
    list of str
        Series ordered by the aggregated score (unranked series last).

    Examples
    --------
    >>> import nowcastbox as nb
    >>> data = nb.load_simulated_dfm().data
    >>> preselect_ranking(data, "gdp", {"methods": ("sis",)})[:3]
    ['x10', 'x19', 'x12']
    """
    key = _panel_hash(panel, target, options)
    if key in _preselect_cache:
        _preselect_cache.move_to_end(key)
        return list(_preselect_cache[key])
    names = [str(s) for s in preselect(panel, target, **dict(options)).ranking().index]
    _preselect_cache[key] = names
    if len(_preselect_cache) > _CACHE_SIZE:
        _preselect_cache.popitem(last=False)
    return list(names)


class SpecifiedModel(BaseNowcaster):
    """An estimator with a sample start and a funnel selection of indicators.

    ``fit(data, target)`` applies ``preprocess``, keeps the target and the first
    ``n_series`` indicators of the ranking, drops the periods before ``start``, and
    fits a copy of ``model`` with ``params``; it returns the inner model's results.

    Parameters
    ----------
    model : estimator or estimator class
        Nowcasting model (e.g. :class:`~nowcastbox.models.MixedFreqDFM`).
    params : mapping, optional
        Parameters of ``model`` (``set_params``).
    start : period-like, optional
        First base period of the estimation sample (``None``: the whole sample).
    n_series : int, optional
        Number of indicators kept from ``ranking`` (``None``: every series). Capped at
        the number of ranked series available in the data.
    ranking : sequence of str, PreselectionResult, "preselect" or mapping, optional
        Static ranking (names, best first; names absent from the data are ignored), or
        ``"preselect"`` / the keyword arguments of
        :func:`~nowcastbox.selection.preselect`, recomputed on the data given to
        ``fit`` (no look-ahead in a backtest). Required with ``n_series``.
    preprocess : callable, optional
        Function applied to the panel before anything else (e.g.
        :class:`~nowcastbox.selection.OutlierCorrection`).

    Attributes
    ----------
    estimator_ : estimator
        The fitted inner model.
    columns_ : list of str
        Indicators used in the last fit.

    Raises
    ------
    ValueError
        If the settings are invalid (raised by ``fit``).

    Examples
    --------
    >>> import nowcastbox as nb
    >>> data = nb.load_simulated_dfm().data
    >>> spec = SpecifiedModel(
    ...     nb.TwoStepDFM,
    ...     params={"n_factors": 1},
    ...     start="2005-01",
    ...     n_series=4,
    ...     ranking={"methods": ("sis",)},
    ... )
    >>> res = spec.fit(data, "gdp")
    >>> spec.columns_
    ['x10', 'x19', 'x12', 'x03']
    >>> str(res.nowcast.index[0])
    '2005Q1'
    >>> again = spec.update(data)
    >>> bool(abs(again.get_nowcast() - res.get_nowcast()) < 1e-10)
    True
    """

    def __init__(
        self,
        model: Any,
        params: Mapping[str, Any] | None = None,
        start: Any = None,
        n_series: int | None = None,
        ranking: Ranking = None,
        preprocess: Callable[[MixedFrequencyData], MixedFrequencyData] | None = None,
    ) -> None:
        self.model = model
        self.params = params
        self.start = start
        self.n_series = n_series
        self.ranking = ranking
        self.preprocess = preprocess

    estimator_: Any = None
    columns_: list[str] | None = None

    def _validate_params(self) -> None:
        if not isinstance(self.model, type) and not callable(getattr(self.model, "fit", None)):
            raise ValueError("model must be an estimator (with fit) or an estimator class.")
        if self.n_series is not None:
            n = self.n_series
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ValueError(f"n_series must be a positive integer, got {n!r}.")
            if self.ranking is None:
                raise ValueError("n_series needs a ranking.")
        if isinstance(self.ranking, str) and self.ranking != "preselect":
            raise ValueError(
                f"A string ranking must be 'preselect', got {self.ranking!r} (use a list of names)."
            )

    def _reset(self) -> None:
        super()._reset()
        self.estimator_ = None
        self.columns_ = None

    # ------------------------------------------------------------------ data
    def _prepared(self, data: MixedFrequencyData) -> MixedFrequencyData:
        return data if self.preprocess is None else self.preprocess(data)

    def _ordered(self, panel: MixedFrequencyData, target: str) -> list[str]:
        """Indicators in ranking order (every indicator without a funnel)."""
        candidates = [c for c in panel.columns if c != target]
        if self.n_series is None or self.ranking is None:
            return candidates
        if isinstance(self.ranking, str):
            names = preselect_ranking(panel, target, {})
        elif isinstance(self.ranking, Mapping):
            names = preselect_ranking(panel, target, self.ranking)
        else:
            names = list(ranking_names(self.ranking))
        return [s for s in names if s in candidates]

    def _restricted(
        self, panel: MixedFrequencyData, target: str, columns: Sequence[str]
    ) -> MixedFrequencyData:
        out = panel.select([target, *columns])
        if self.start is not None:
            first = pd.Period(self.start, freq=panel.base_frequency.pandas_freq)
            if first > out.start:
                out = out.truncate(first, None)
        return out

    # ------------------------------------------------------------------ fit
    def _fit(self, data: MixedFrequencyData, target: str, **fit_kwargs: Any) -> NowcastResults:
        panel = self._prepared(data)
        ordered = self._ordered(panel, target)
        columns = ordered if self.n_series is None else ordered[: self.n_series]
        if not columns:
            raise ValueError("The specification keeps no indicator.")
        estimator = configured_model(self.model, self.params)
        results = estimator.fit(self._restricted(panel, target, columns), target, **fit_kwargs)
        self.estimator_ = estimator
        self.columns_ = list(columns)
        return results

    def update(self, data: MixedFrequencyData | pd.DataFrame) -> NowcastResults | None:
        """Results on new data with the fitted parameters and indicators kept fixed.

        Uses the inner model's ``update`` when it has one, otherwise its results'
        ``predict`` (e.g. :class:`~nowcastbox.models.MixedFreqDFM`).

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            New information set containing the indicators used in the fit.

        Returns
        -------
        NowcastResults or None
            Updated results; ``None`` when the inner model supports neither.

        Raises
        ------
        ModelNotFittedError
            If the model has not been fitted.

        Examples
        --------
        >>> spec.update(data).target  # doctest: +SKIP
        'gdp'
        """
        if self.estimator_ is None or self.columns_ is None:
            raise ModelNotFittedError("This SpecifiedModel is not fitted yet; call fit() first.")
        fitted = self.results_
        target = fitted.target
        panel = self._restricted(
            self._prepared(as_mixed_frequency_data(data)), target, self.columns_
        )
        update = getattr(self.estimator_, "update", None)
        if callable(update):
            out = update(panel)
            return out if isinstance(out, NowcastResults) else None
        predict = getattr(fitted, "predict", None)
        if not callable(predict):
            return None
        frame = predict(panel)
        if not isinstance(frame, pd.DataFrame) or target not in frame:
            return None
        estimate = _native(frame, target, fitted.target_frequency)
        observed = panel.to_native(target).reindex(estimate.index)
        return NowcastResults(
            target=target,
            nowcast=build_nowcast_frame(observed, estimate),
            model_name=fitted.model_name,
        )

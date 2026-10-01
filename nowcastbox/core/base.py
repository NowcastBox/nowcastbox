"""Abstract base classes shared by all estimators.

* :class:`BaseNowcaster` - contract of every nowcasting model (``TwoStepDFM``,
  ``MixedFreqDFM``, ``BridgeEquation`` ...): scikit-learn-like hyper-parameters set in
  ``__init__``, ``fit(data, target) -> NowcastResults``.
* :class:`BaseBenchmark` and the :class:`BenchmarkForecaster` protocol - contract of
  the simple forecasters used in real-time evaluation (AR, random walk, MIDAS...).

Implementing a new estimator
----------------------------
1. Subclass :class:`BaseNowcaster`; store **every** ``__init__`` argument as an
   attribute with the same name and do no validation in ``__init__`` (as in
   scikit-learn), so that :meth:`~BaseNowcaster.get_params`/``set_params``/``clone``
   work.
2. Validate hyper-parameters in :meth:`~BaseNowcaster._validate_params`.
3. Implement :meth:`~BaseNowcaster._fit`, which receives a validated
   :class:`~nowcastbox.core.data.MixedFrequencyData` restricted to the target and its
   predictors and returns a :class:`~nowcastbox.core.results.NowcastResults`.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Protocol, TypeVar, runtime_checkable

import pandas as pd

from nowcastbox.core.data import FrequencySpec, MixedFrequencyData, as_mixed_frequency_data
from nowcastbox.core.exceptions import ModelNotFittedError, NowcastDataError
from nowcastbox.core.formula import resolve_target
from nowcastbox.core.frequency import Frequency
from nowcastbox.core.results import NowcastResults

__all__ = [
    "BaseBenchmark",
    "BaseNowcaster",
    "BenchmarkForecaster",
    "ParamsMixin",
]

_P = TypeVar("_P", bound="ParamsMixin")


class ParamsMixin:
    """scikit-learn-like hyper-parameter handling.

    Parameters are the arguments of ``__init__`` (excluding ``self``, ``*args`` and
    ``**kwargs``), each stored as an attribute of the same name.

    Examples
    --------
    >>> from nowcastbox.core.base import ParamsMixin
    >>> class Demo(ParamsMixin):
    ...     def __init__(self, n_factors=1, factor_lags=2):
    ...         self.n_factors = n_factors
    ...         self.factor_lags = factor_lags
    >>> Demo().get_params()
    {'n_factors': 1, 'factor_lags': 2}
    >>> Demo().set_params(n_factors=3).n_factors
    3
    """

    @classmethod
    def _get_param_names(cls) -> list[str]:
        init = cls.__init__
        if init is object.__init__:
            return []
        signature = inspect.signature(init)
        names: list[str] = []
        for p in signature.parameters.values():
            if p.name == "self":
                continue
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                raise TypeError(
                    f"{cls.__name__}.__init__ must not use *args/**kwargs "
                    "(hyper-parameters must be explicit)."
                )
            names.append(p.name)
        return names

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the estimator's hyper-parameters.

        Parameters
        ----------
        deep : bool, default True
            Also return the parameters of nested estimators (objects with
            ``get_params``) as ``"<name>__<param>"``.

        Returns
        -------
        dict
            Parameter name to value.

        Raises
        ------
        AttributeError
            If a parameter was not stored as an attribute in ``__init__``.

        Examples
        --------
        >>> class Demo(ParamsMixin):
        ...     def __init__(self, alpha=1.0):
        ...         self.alpha = alpha
        >>> Demo(alpha=2.0).get_params()
        {'alpha': 2.0}
        """
        out: dict[str, Any] = {}
        for name in self._get_param_names():
            if not hasattr(self, name):
                raise AttributeError(
                    f"{type(self).__name__} must store the __init__ argument {name!r} "
                    "as an attribute of the same name."
                )
            value = getattr(self, name)
            if deep and isinstance(value, ParamsMixin):
                out.update({f"{name}__{k}": v for k, v in value.get_params(deep=True).items()})
            out[name] = value
        return out

    def set_params(self: _P, **params: Any) -> _P:
        """Set hyper-parameters (nested ones as ``"<name>__<param>"``).

        Setting parameters discards any fitted state.

        Parameters
        ----------
        **params
            Parameter values.

        Returns
        -------
        self
            The estimator.

        Raises
        ------
        ValueError
            If a parameter name is not valid for this estimator.

        Examples
        --------
        >>> class Demo(ParamsMixin):
        ...     def __init__(self, alpha=1.0):
        ...         self.alpha = alpha
        >>> Demo().set_params(alpha=0.5).alpha
        0.5
        """
        valid = set(self._get_param_names())
        nested: dict[str, dict[str, Any]] = {}
        for key, value in params.items():
            name, sep, sub = key.partition("__")
            if name not in valid:
                raise ValueError(
                    f"Invalid parameter {name!r} for {type(self).__name__}; "
                    f"valid parameters are {sorted(valid)}."
                )
            if sep:
                nested.setdefault(name, {})[sub] = value
            else:
                setattr(self, name, value)
        for name, sub_params in nested.items():
            child = getattr(self, name)
            if not isinstance(child, ParamsMixin):
                raise ValueError(f"Parameter {name!r} is not an estimator with parameters.")
            child.set_params(**sub_params)
        self._reset()
        return self

    def _reset(self) -> None:
        """Discard fitted state (hook; overridden by fitted estimators)."""

    def clone(self: _P) -> _P:
        """Return an unfitted copy with the same hyper-parameters.

        Returns
        -------
        same type
            New estimator.

        Examples
        --------
        >>> class Demo(ParamsMixin):
        ...     def __init__(self, alpha=1.0):
        ...         self.alpha = alpha
        >>> Demo(alpha=3.0).clone().alpha
        3.0
        """
        return type(self)(**self.get_params(deep=False))

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params(deep=False).items())
        return f"{type(self).__name__}({params})"


class BaseNowcaster(ParamsMixin, ABC):
    """Abstract base class of nowcasting estimators.

    Subclasses implement :meth:`_fit` (and optionally :meth:`_validate_params`).
    :meth:`fit` takes care of input coercion, target/formula resolution and fitted
    state.

    Attributes
    ----------
    results_ : NowcastResults
        Results of the last call to :meth:`fit` (raises before fitting).
    is_fitted : bool
        Whether :meth:`fit` has been called successfully.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.base import BaseNowcaster
    >>> from nowcastbox.core.results import NowcastResults, build_nowcast_frame
    >>> class MeanNowcaster(BaseNowcaster):
    ...     def __init__(self, scale=1.0):
    ...         self.scale = scale
    ...
    ...     def _fit(self, data, target, **kwargs):
    ...         y = data.to_native(target)
    ...         estimate = pd.Series(self.scale * y.mean(), index=y.index)
    ...         frame = build_nowcast_frame(y, estimate)
    ...         return NowcastResults(target=target, nowcast=frame, model_name="Mean")
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> df = pd.DataFrame(
    ...     {"x": np.arange(6.0), "y": [np.nan, np.nan, 1.0, np.nan, np.nan, np.nan]},
    ...     index=idx,
    ... )
    >>> res = MeanNowcaster().fit(df, target="y", frequency={"x": "M", "y": "Q"})
    >>> res.get_nowcast()
    1.0
    """

    _results: NowcastResults | None = None

    def fit(
        self,
        data: MixedFrequencyData | pd.DataFrame,
        target: str,
        *,
        frequency: FrequencySpec | None = None,
        **fit_kwargs: Any,
    ) -> NowcastResults:
        """Estimate the model and compute nowcasts of ``target``.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            Panel on the base grid (see :class:`~nowcastbox.core.data.MixedFrequencyData`).
        target : str
            Target series name, or a formula (``"gdp ~ ."``, ``"gdp ~ ip + pmi"``)
            selecting the predictors.
        frequency : frequency specification, optional
            Per-series frequencies when ``data`` is a DataFrame.
        **fit_kwargs
            Estimator-specific fit options, passed to :meth:`_fit`.

        Returns
        -------
        NowcastResults
            Estimation results (also stored in :attr:`results_`).

        Raises
        ------
        NowcastDataError
            If the data are invalid.
        FormulaError
            If ``target`` is an invalid formula or an unknown series.
        TypeError
            If :meth:`_fit` does not return a :class:`NowcastResults`.
        """
        self._reset()
        self._validate_params()
        panel = as_mixed_frequency_data(data, frequency)
        target_name, regressors = resolve_target(target, panel.columns)
        keep = {*regressors, target_name}
        if len(keep) != panel.n_series:
            panel = panel.select([c for c in panel.columns if c in keep])
        results = self._fit(panel, target_name, **fit_kwargs)
        if not isinstance(results, NowcastResults):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"{type(self).__name__}._fit must return a NowcastResults, "
                f"got {type(results).__name__}."
            )
        self._results = results
        return results

    @abstractmethod
    def _fit(self, data: MixedFrequencyData, target: str, **fit_kwargs: Any) -> NowcastResults:
        """Estimate the model on a validated panel (implemented by subclasses).

        Parameters
        ----------
        data : MixedFrequencyData
            Panel containing the target and its predictors only.
        target : str
            Target series name (a column of ``data``).
        **fit_kwargs
            Estimator-specific options.

        Returns
        -------
        NowcastResults
            Estimation results.
        """

    def _validate_params(self) -> None:
        """Validate hyper-parameters before fitting (hook; raise ``ValueError``)."""

    def _reset(self) -> None:
        self._results = None

    @property
    def is_fitted(self) -> bool:
        """Whether the estimator has been fitted."""
        return self._results is not None

    @property
    def results_(self) -> NowcastResults:
        """Results of the last fit.

        Raises
        ------
        ModelNotFittedError
            If the estimator has not been fitted.
        """
        self._check_fitted()
        assert self._results is not None  # noqa: S101
        return self._results

    def _check_fitted(self) -> None:
        """Raise :class:`ModelNotFittedError` if the estimator is not fitted."""
        if self._results is None:
            raise ModelNotFittedError(
                f"This {type(self).__name__} instance is not fitted yet; call fit() first."
            )


@runtime_checkable
class BenchmarkForecaster(Protocol):
    """Structural interface of forecasters usable in real-time evaluation.

    Any object with these two methods (including adapters around scikit-learn
    estimators, plan innovation I11) can be used as a benchmark by
    ``nowcastbox.evaluation``.

    Examples
    --------
    >>> from nowcastbox.core.base import BenchmarkForecaster
    >>> class Naive:
    ...     def fit(self, data, target):
    ...         return self
    ...
    ...     def predict(self, periods):
    ...         return None
    >>> isinstance(Naive(), BenchmarkForecaster)
    True
    """

    def fit(self, data: MixedFrequencyData, target: str) -> Any:
        """Fit on the information set ``data`` (one vintage)."""
        ...

    def predict(self, periods: pd.PeriodIndex) -> pd.Series:
        """Predict the target for ``periods`` (native target frequency)."""
        ...


class BaseBenchmark(ParamsMixin, ABC):
    """Abstract base class of simple benchmark forecasters (AR, RW, MIDAS...).

    Contract: ``fit(data, target)`` uses only the information in ``data`` (a vintage),
    and ``predict(periods)`` returns predictions of the target for the given periods
    of the target's native frequency, in the units of the data.

    Attributes
    ----------
    target_ : str
        Target series of the last fit.
    target_frequency_ : Frequency
        Native frequency of the target.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.base import BaseBenchmark
    >>> class HistoricalMean(BaseBenchmark):
    ...     def __init__(self):
    ...         pass
    ...
    ...     def _fit(self, data, target):
    ...         self.mean_ = float(data.to_native(target).mean())
    ...
    ...     def _predict(self, periods):
    ...         return pd.Series(self.mean_, index=periods)
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> df = pd.DataFrame({"y": [1.0, 2, 3, 4, 5, 6]}, index=idx)
    >>> HistoricalMean().fit(df, "y", frequency="M").predict(["2020-07"]).tolist()
    [3.5]
    """

    _target: str | None = None
    _target_frequency: Frequency | None = None

    @property
    def name(self) -> str:
        """Display name of the benchmark (class name by default)."""
        return type(self).__name__

    @property
    def is_fitted(self) -> bool:
        """Whether the benchmark has been fitted."""
        return self._target is not None

    @property
    def target_(self) -> str:
        """Target series of the last fit."""
        self._check_fitted()
        assert self._target is not None  # noqa: S101
        return self._target

    @property
    def target_frequency_(self) -> Frequency:
        """Native frequency of the target of the last fit."""
        self._check_fitted()
        assert self._target_frequency is not None  # noqa: S101
        return self._target_frequency

    def _reset(self) -> None:
        self._target = None
        self._target_frequency = None

    def _check_fitted(self) -> None:
        """Raise :class:`ModelNotFittedError` if the benchmark is not fitted."""
        if self._target is None:
            raise ModelNotFittedError(
                f"This {type(self).__name__} instance is not fitted yet; call fit() first."
            )

    def fit(
        self,
        data: MixedFrequencyData | pd.DataFrame,
        target: str,
        *,
        frequency: FrequencySpec | None = None,
    ) -> BaseBenchmark:
        """Fit the benchmark on one information set.

        Parameters
        ----------
        data : MixedFrequencyData or pandas.DataFrame
            Panel (the vintage available at the forecast origin).
        target : str
            Target series name.
        frequency : frequency specification, optional
            Per-series frequencies when ``data`` is a DataFrame.

        Returns
        -------
        BaseBenchmark
            ``self``.

        Raises
        ------
        NowcastDataError
            If ``target`` is not a column of ``data``.
        """
        self._reset()
        panel = as_mixed_frequency_data(data, frequency)
        if target not in panel:
            raise NowcastDataError(f"Target {target!r} is not a column of the data.")
        self._fit(panel, target)
        self._target = target
        self._target_frequency = panel.metadata[target].frequency
        return self

    def predict(self, periods: pd.PeriodIndex | Iterable[pd.Period | str]) -> pd.Series:
        """Predict the target for some periods of its native frequency.

        Parameters
        ----------
        periods : pandas.PeriodIndex or iterable of Period/str
            Target periods (e.g. ``["2020Q2"]``); converted to the target frequency.

        Returns
        -------
        pandas.Series
            Predictions indexed by ``periods``.

        Raises
        ------
        ModelNotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``_predict`` returns a series with a different index.
        """
        self._check_fitted()
        freq = self.target_frequency_.pandas_freq
        if isinstance(periods, pd.PeriodIndex):
            index = periods.asfreq(freq)
        else:
            index = pd.PeriodIndex([pd.Period(p, freq=freq) for p in periods], freq=freq)
        out = self._predict(index)
        if not out.index.equals(index):
            raise ValueError(f"{type(self).__name__}._predict returned a misaligned index.")
        return out.rename(self.target_)

    @abstractmethod
    def _fit(self, data: MixedFrequencyData, target: str) -> None:
        """Fit on a validated panel (implemented by subclasses).

        Parameters
        ----------
        data : MixedFrequencyData
            Information set.
        target : str
            Target series name.
        """

    @abstractmethod
    def _predict(self, periods: pd.PeriodIndex) -> pd.Series:
        """Return predictions indexed exactly by ``periods`` (implemented by subclasses).

        Parameters
        ----------
        periods : pandas.PeriodIndex
            Target periods at the target's native frequency.

        Returns
        -------
        pandas.Series
            Predictions.
        """

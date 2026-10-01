r"""Named, composable and invertible stationarity transformations.

A :class:`Transform` maps a series in levels :math:`x_t` to a (typically stationary)
series :math:`y_t`, and its :meth:`Transform.inverse` rebuilds levels from transformed
values plus a short history of levels — e.g. to turn a nowcast of a growth rate back
into a nowcast of the level.

Elementary transforms
---------------------
==================  ==========================================  ==========================
Class               Forward :math:`y_t`                         Inverse :math:`x_t`
==================  ==========================================  ==========================
:class:`Identity`   :math:`x_t`                                 :math:`y_t`
:class:`Log`        :math:`\log x_t`                            :math:`e^{y_t}`
:class:`Scale`      :math:`c\,x_t`                              :math:`y_t / c`
:class:`Diff`       :math:`x_t - x_{t-k}`                       :math:`x_{t-k} + y_t`
:class:`PctChange`  :math:`(x_t - x_{t-k}) / x_{t-k}`           :math:`x_{t-k}(1 + y_t)`
==================  ==========================================  ==========================

Transforms compose left to right with ``|``: ``Log() | Diff(1)`` is the log
difference. The inverse of a composition applies the inverses right to left, using the
history of levels pushed through the leading steps.

Frequency-aware lags
--------------------
A lag is either an integer number of **native periods** of the series or a calendar
span (``"month"``, ``"quarter"``, ``"year"``, ...) resolved with the series frequency
(:func:`resolve_lag`): ``PctChange("year")`` uses lag 12 for a monthly series and 4 for
a quarterly one. Series stored on a finer base grid (quarterly values in the third
month of each quarter) are transformed on their native grid, so ``Diff(1)`` of such a
series is the quarter-on-quarter difference.

Legacy codes
------------
The numeric codes 0-7 used in the nowcasting literature (e.g. the panels of
Giannone, Reichlin & Small, 2008) are accepted as shortcuts (:data:`TRANSFORM_CODES`):

====  =======================  =============================================================
Code  Name                     Formula (lags in native periods of the series)
====  =======================  =============================================================
0     ``level``                :math:`x_t`
1     ``pct_change``           :math:`(x_t - x_{t-1})/x_{t-1}`
2     ``diff``                 :math:`x_t - x_{t-1}`
3     ``diff_of_yoy``          :math:`\Delta_1[(x_t - x_{t-s})/x_{t-s}]`, :math:`s` = 1 year
4     ``diff_of_annual_diff``  :math:`\Delta_1 \Delta_s x_t`
5     ``annual_diff``          :math:`x_t - x_{t-s}`
6     ``yoy``                  :math:`(x_t - x_{t-s})/x_{t-s}`
7     ``qoq``                  :math:`(x_t - x_{t-q})/x_{t-q}`, :math:`q` = 1 quarter
====  =======================  =============================================================

For monthly series these coincide with the original monthly definitions
(:math:`s = 12`, :math:`q = 3`).
"""

from __future__ import annotations

import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.data import FrequencySpec, MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.frequency import Frequency, FrequencyLike
from nowcastbox.preprocessing._utils import (
    NativeView,
    PanelLike,
    as_panel,
    map_native,
    per_series,
    return_like,
    series_frequency,
)

__all__ = [
    "NAMED_TRANSFORMS",
    "TRANSFORM_CODES",
    "Compose",
    "Diff",
    "Identity",
    "LagLike",
    "Log",
    "PctChange",
    "Scale",
    "Transform",
    "TransformLike",
    "apply_transforms",
    "get_transform",
    "invert_transforms",
    "resolve_lag",
    "resolve_transforms",
    "transform_from_code",
]

logger = get_logger(__name__)

LagLike: TypeAlias = int | str
"""A lag: integer number of native periods, or a calendar span such as ``"year"``."""


def resolve_lag(lag: LagLike, frequency: FrequencyLike | None = None) -> int:
    """Resolve a lag specification into a number of native periods.

    Parameters
    ----------
    lag : int or str
        Positive integer number of periods, or a calendar span (``"day"``, ``"week"``,
        ``"month"``, ``"quarter"``, ``"year"`` or any alias accepted by
        :meth:`~nowcastbox.core.frequency.Frequency.from_value`).
    frequency : Frequency, str or int, optional
        Native frequency of the series; required when ``lag`` is a span.

    Returns
    -------
    int
        Number of native periods (>= 1).

    Raises
    ------
    ValueError
        If ``lag`` is not a positive integer or a known span, if ``frequency`` is
        missing for a span, if the span is shorter than one period of the series
        (e.g. a monthly lag of a quarterly series) or is not an integer number of
        periods (e.g. a month of a weekly series).

    Examples
    --------
    >>> from nowcastbox.preprocessing.transforms import resolve_lag
    >>> resolve_lag("year", "M"), resolve_lag("year", "Q"), resolve_lag("quarter", "M")
    (12, 4, 3)
    >>> resolve_lag(2)
    2
    """
    if isinstance(lag, bool):
        raise ValueError(f"lag must be a positive integer or a calendar span, got {lag!r}.")
    if isinstance(lag, Integral):
        if int(lag) < 1:
            raise ValueError(f"lag must be >= 1, got {lag}.")
        return int(lag)
    if not isinstance(lag, str):
        raise ValueError(f"lag must be a positive integer or a calendar span, got {lag!r}.")
    if lag.strip().isdigit():
        return resolve_lag(int(lag.strip()))
    try:
        span = Frequency.from_value(lag)
    except ValueError:
        raise ValueError(f"Unknown lag span {lag!r}.") from None
    if frequency is None:
        raise ValueError(f"The frequency of the series is needed to resolve the lag {lag!r}.")
    freq = Frequency.from_value(frequency)
    if span.is_higher_than(freq):
        raise ValueError(
            f"A {span.label} lag is undefined for a {freq.label} series "
            "(it is shorter than one period)."
        )
    ratio = freq.periods_per_year / span.periods_per_year
    if not float(ratio).is_integer():
        raise ValueError(f"A {span.label} span is not an integer number of {freq.label} periods.")
    return int(ratio)


def _lag_spec(lag: LagLike) -> str:
    return str(int(lag)) if isinstance(lag, Integral) else str(lag).strip().lower()


def _validate_lag(lag: object) -> LagLike:
    if isinstance(lag, bool) or not isinstance(lag, Integral | str):
        raise ValueError(f"lag must be a positive integer or a calendar span, got {lag!r}.")
    if isinstance(lag, Integral):
        resolve_lag(int(lag))
        return int(lag)
    text = lag.strip().lower()
    if text.isdigit():
        return resolve_lag(int(text))
    try:
        Frequency.from_value(text)
    except ValueError:
        raise ValueError(f"Unknown lag span {lag!r}.") from None
    return text


class Transform(ABC):
    """Base class of invertible series transformations.

    Subclasses implement :meth:`_forward` and :meth:`_backward` on NumPy arrays laid
    out on the contiguous native grid of a series, plus :meth:`to_spec`. Everything
    else - pandas handling, base-grid storage slots, panels, composition and the
    inverse - is provided here.

    Notes
    -----
    Composition with ``|`` is left to right: ``a | b`` first applies ``a`` and then
    ``b``. Two transforms are equal when their :meth:`to_spec` strings are equal.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.preprocessing.transforms import Diff, Log
    >>> x = pd.Series([100.0, 110.0, 121.0], index=pd.period_range("2020-01", periods=3, freq="M"))
    >>> dlog = Log() | Diff(1)
    >>> dlog.apply(x).round(4).tolist()
    [nan, 0.0953, 0.0953]
    >>> dlog.inverse(dlog.apply(x).iloc[1:], history=x.iloc[:1]).round(6).tolist()
    [100.0, 110.0, 121.0]
    """

    # ------------------------------------------------------------------ to implement
    @abstractmethod
    def _forward(self, x: np.ndarray, frequency: Frequency) -> np.ndarray:
        """Transform levels on the contiguous native grid."""

    @abstractmethod
    def _backward(self, y: np.ndarray, history: np.ndarray, frequency: Frequency) -> np.ndarray:
        """Rebuild levels: known values of ``history`` win; NaN elsewhere is rebuilt."""

    @abstractmethod
    def to_spec(self) -> str:
        """Canonical string specification, parseable by :func:`get_transform`.

        Returns
        -------
        str
            Specification such as ``"log|diff(1)"``.

        Examples
        --------
        >>> from nowcastbox.preprocessing.transforms import PctChange
        >>> PctChange("year").to_spec()
        'pct_change(year)'
        """

    def n_lags(self, frequency: FrequencyLike | None = None) -> int:
        """Number of leading periods lost by the transformation.

        Parameters
        ----------
        frequency : Frequency, str or int, optional
            Native frequency of the series (needed for calendar-span lags).

        Returns
        -------
        int
            Number of leading native periods without a transformed value; also the
            minimum length of the history needed by :meth:`inverse`.

        Examples
        --------
        >>> from nowcastbox.preprocessing.transforms import get_transform
        >>> get_transform(4).n_lags("M")
        13
        """
        return 0

    # ------------------------------------------------------------------ composition
    @property
    def steps(self) -> tuple[Transform, ...]:
        """Elementary steps of the transform, in application order."""
        return (self,)

    def __or__(self, other: object) -> Compose:
        if not isinstance(other, Transform):
            other = get_transform(other)  # type: ignore[arg-type]
        return Compose(self, other)

    def __ror__(self, other: object) -> Compose:
        return Compose(get_transform(other), self)  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transform):
            return NotImplemented
        return self.to_spec() == other.to_spec()

    def __hash__(self) -> int:
        return hash(self.to_spec())

    def __repr__(self) -> str:
        return f"Transform({self.to_spec()!r})"

    def __str__(self) -> str:
        return self.to_spec()

    # ------------------------------------------------------------------ application
    def apply(self, data: Any, frequency: FrequencySpec | None = None) -> Any:
        """Apply the transformation.

        Parameters
        ----------
        data : pandas.Series, pandas.DataFrame or MixedFrequencyData
            Levels. A Series must have a :class:`pandas.PeriodIndex`; it may be on its
            native grid or on a finer base grid (lower-frequency values stored in the
            last base period). Panels are transformed series by series on each
            series' native grid.
        frequency : frequency or frequency specification, optional
            Native frequency of a Series, or per-series frequencies of a DataFrame.
            Inferred from the observation pattern when omitted. Not allowed with a
            :class:`MixedFrequencyData` (which carries its frequencies).

        Returns
        -------
        pandas.Series, pandas.DataFrame or MixedFrequencyData
            Transformed data, of the same type and on the same index as ``data``.

        Raises
        ------
        NowcastDataError
            On invalid input, or values outside the domain of the transformation
            (e.g. non-positive values for :class:`Log`).

        Warns
        -----
        DataQualityWarning
            When a percentage change divides by zero (the result is set to NaN).

        Examples
        --------
        >>> import pandas as pd
        >>> from nowcastbox.preprocessing.transforms import Diff
        >>> x = pd.Series([1.0, 3.0, 6.0], index=pd.period_range("2020Q1", periods=3, freq="Q"))
        >>> Diff(1).apply(x).tolist()
        [nan, 2.0, 3.0]
        """
        if isinstance(data, pd.Series):
            view = NativeView(data, frequency)  # type: ignore[arg-type]
            values = self._forward(view.native.to_numpy(dtype=float), view.frequency)
            return view.to_original(values)
        panel, was_frame = as_panel(data, frequency)
        frame = map_native(panel, lambda _n, s, f: self._forward(s.to_numpy(float), f))
        return return_like(panel.with_data(frame), was_frame)

    def __call__(self, data: Any, frequency: FrequencySpec | None = None) -> Any:
        """Alias of :meth:`apply`.

        Parameters
        ----------
        data : pandas.Series, pandas.DataFrame or MixedFrequencyData
            Levels.
        frequency : frequency or frequency specification, optional
            Native frequency (frequencies) of the data.

        Returns
        -------
        same type as ``data``
            Transformed data.

        Examples
        --------
        >>> import pandas as pd
        >>> from nowcastbox.preprocessing.transforms import Diff
        >>> x = pd.Series([1.0, 4.0], index=pd.period_range("2020-01", periods=2, freq="M"))
        >>> Diff()(x).tolist()
        [nan, 3.0]
        """
        return self.apply(data, frequency)

    def inverse(
        self,
        transformed: pd.Series,
        history: pd.Series | None = None,
        frequency: FrequencyLike | None = None,
    ) -> pd.Series:
        """Rebuild levels from transformed values and a history of levels.

        Parameters
        ----------
        transformed : pandas.Series
            Transformed values (e.g. a nowcast of a growth rate), indexed by a
            :class:`pandas.PeriodIndex` on the native or the base grid.
        history : pandas.Series, optional
            Levels on the same kind of index. Must contain at least
            :meth:`n_lags` observations right before the first period to rebuild.
            Optional only for transforms without lags (log, scale, identity).
        frequency : Frequency, str or int, optional
            Native frequency of the series (inferred when omitted).

        Returns
        -------
        pandas.Series
            Levels on the union of both period ranges. Where ``history`` is known it
            is returned unchanged; elsewhere levels are rebuilt recursively from
            ``transformed`` (NaN where the required lagged level is unknown).

        Raises
        ------
        NowcastDataError
            If the indexes are invalid or have different frequencies.
        ValueError
            If ``history`` is missing but the transform has lags.

        Examples
        --------
        >>> import pandas as pd
        >>> from nowcastbox.preprocessing.transforms import PctChange
        >>> idx = pd.period_range("2020Q1", periods=4, freq="Q")
        >>> hist = pd.Series([100.0], index=idx[:1])
        >>> growth = pd.Series([0.1, 0.0, -0.5], index=idx[1:])
        >>> PctChange(1).inverse(growth, hist).round(6).tolist()
        [100.0, 110.0, 110.0, 55.0]
        """
        y = _validated_series(transformed, "transformed")
        h = (
            pd.Series(np.nan, index=y.index[:0], dtype=float)
            if history is None
            else _validated_series(history, "history")
        )
        y_freq = cast("pd.PeriodIndex", y.index).freqstr
        h_freq = cast("pd.PeriodIndex", h.index).freqstr
        if h_freq != y_freq:
            raise NowcastDataError(
                f"transformed ({y_freq}) and history ({h_freq}) must have the same index frequency."
            )
        union = y.index.union(h.index)
        if len(union) == 0:
            raise NowcastDataError("transformed and history are both empty.")
        full = pd.period_range(union[0], union[-1], freq=y_freq)
        y_full = y.reindex(full)
        h_full = h.reindex(full)
        freq = series_frequency(y_full.combine_first(h_full), frequency)
        needed = self.n_lags(freq)
        if history is None and needed > 0:
            raise ValueError(
                f"{self.to_spec()!r} needs a history of at least {needed} level(s) to be inverted."
            )
        y_view = NativeView(y_full, freq, name="transformed")
        h_view = NativeView(h_full, freq, name="history")
        levels = self._backward(
            y_view.native.to_numpy(dtype=float), h_view.native.to_numpy(dtype=float), freq
        )
        result = h_view.to_original(levels)
        result.name = y.name if y.name is not None else h.name
        return result


def _validated_series(series: object, what: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise NowcastDataError(f"{what} must be a pandas Series, got {type(series).__name__}.")
    if not isinstance(series.index, pd.PeriodIndex):
        raise NowcastDataError(f"{what} must be indexed by a pandas.PeriodIndex.")
    if series.index.has_duplicates:
        raise NowcastDataError(f"{what} has duplicated periods.")
    return series.astype(float).sort_index()


# ---------------------------------------------------------------------- elementary
class Identity(Transform):
    """Identity (levels, legacy code 0).

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.preprocessing.transforms import Identity
    >>> x = pd.Series([1.0, 2.0], index=pd.period_range("2020-01", periods=2, freq="M"))
    >>> Identity().apply(x).tolist()
    [1.0, 2.0]
    """

    def _forward(self, x: np.ndarray, frequency: Frequency) -> np.ndarray:
        return x.copy()

    def _backward(self, y: np.ndarray, history: np.ndarray, frequency: Frequency) -> np.ndarray:
        return np.where(np.isnan(history), y, history)

    def to_spec(self) -> str:
        """Return ``"level"``.

        Returns
        -------
        str
            Specification.

        Examples
        --------
        >>> Identity().to_spec()
        'level'
        """
        return "level"


class Log(Transform):
    r"""Natural logarithm :math:`y_t = \log x_t`.

    Raises :class:`~nowcastbox.core.exceptions.NowcastDataError` on non-positive values.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.preprocessing.transforms import Log
    >>> x = pd.Series([1.0, np.e], index=pd.period_range("2020-01", periods=2, freq="M"))
    >>> Log().apply(x).tolist()
    [0.0, 1.0]
    """

    def _forward(self, x: np.ndarray, frequency: Frequency) -> np.ndarray:
        obs = x[~np.isnan(x)]
        if bool((obs <= 0).any()):
            raise NowcastDataError(
                "The log transformation needs strictly positive values; "
                f"found {int((obs <= 0).sum())} non-positive value(s)."
            )
        return np.log(x)

    def _backward(self, y: np.ndarray, history: np.ndarray, frequency: Frequency) -> np.ndarray:
        return np.where(np.isnan(history), np.exp(y), history)

    def to_spec(self) -> str:
        """Return ``"log"``.

        Returns
        -------
        str
            Specification.

        Examples
        --------
        >>> Log().to_spec()
        'log'
        """
        return "log"


class Scale(Transform):
    r"""Multiplication by a non-zero constant, :math:`y_t = c\,x_t`.

    Parameters
    ----------
    factor : float
        Non-zero finite factor (e.g. 100 to express rates in percent).

    Raises
    ------
    ValueError
        If ``factor`` is zero or not finite.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.preprocessing.transforms import PctChange, Scale
    >>> x = pd.Series([100.0, 102.0], index=pd.period_range("2020-01", periods=2, freq="M"))
    >>> (PctChange(1) | Scale(100)).apply(x).round(6).tolist()
    [nan, 2.0]
    """

    def __init__(self, factor: float) -> None:
        if isinstance(factor, bool) or not isinstance(factor, Real):
            raise ValueError(f"factor must be a real number, got {factor!r}.")
        if not np.isfinite(float(factor)) or float(factor) == 0.0:
            raise ValueError(f"factor must be finite and non-zero, got {factor!r}.")
        self.factor = float(factor)

    def _forward(self, x: np.ndarray, frequency: Frequency) -> np.ndarray:
        return x * self.factor

    def _backward(self, y: np.ndarray, history: np.ndarray, frequency: Frequency) -> np.ndarray:
        return np.where(np.isnan(history), y / self.factor, history)

    def to_spec(self) -> str:
        """Return ``"scale(<factor>)"``.

        Returns
        -------
        str
            Specification.

        Examples
        --------
        >>> Scale(100).to_spec()
        'scale(100)'
        """
        return f"scale({self.factor:g})"


class _Lagged(Transform):
    """Common code of lag-based transforms."""

    _name: str = ""

    def __init__(self, lag: LagLike = 1) -> None:
        self.lag = _validate_lag(lag)

    def n_lags(self, frequency: FrequencyLike | None = None) -> int:
        """Number of leading periods lost (the resolved lag).

        Parameters
        ----------
        frequency : Frequency, str or int, optional
            Native frequency (needed for calendar-span lags).

        Returns
        -------
        int
            Lag in native periods.

        Examples
        --------
        >>> from nowcastbox.preprocessing.transforms import Diff
        >>> Diff("year").n_lags("Q")
        4
        """
        return resolve_lag(self.lag, frequency)

    def to_spec(self) -> str:
        """Return ``"<name>(<lag>)"``.

        Returns
        -------
        str
            Specification.

        Examples
        --------
        >>> from nowcastbox.preprocessing.transforms import Diff
        >>> Diff(12).to_spec()
        'diff(12)'
        """
        return f"{self._name}({_lag_spec(self.lag)})"

    @staticmethod
    def _rebuild(
        y: np.ndarray, history: np.ndarray, k: int, step: Any
    ) -> np.ndarray:  # step(previous_level, transformed) -> level
        x = history.astype(float).copy()
        for t in range(k, len(x)):
            if np.isnan(x[t]):
                x[t] = step(x[t - k], y[t])
        return x


class Diff(_Lagged):
    r"""Lagged difference :math:`y_t = x_t - x_{t-k}`.

    Parameters
    ----------
    lag : int or str, default 1
        Lag :math:`k` in native periods, or a calendar span (``"year"`` gives
        :math:`k = 12` for monthly and :math:`k = 4` for quarterly series).

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.preprocessing.transforms import Diff
    >>> x = pd.Series([1.0, 2.0, 4.0, 7.0], index=pd.period_range("2020-01", periods=4, freq="M"))
    >>> Diff(2).apply(x).tolist()
    [nan, nan, 3.0, 5.0]
    """

    _name = "diff"

    def _forward(self, x: np.ndarray, frequency: Frequency) -> np.ndarray:
        k = resolve_lag(self.lag, frequency)
        out = np.full_like(x, np.nan)
        out[k:] = x[k:] - x[:-k]
        return out

    def _backward(self, y: np.ndarray, history: np.ndarray, frequency: Frequency) -> np.ndarray:
        k = resolve_lag(self.lag, frequency)
        return self._rebuild(y, history, k, lambda prev, val: prev + val)


class PctChange(_Lagged):
    r"""Lagged relative change :math:`y_t = (x_t - x_{t-k}) / x_{t-k}`.

    Parameters
    ----------
    lag : int or str, default 1
        Lag :math:`k` in native periods, or a calendar span (``"month"``,
        ``"quarter"``, ``"year"``...).

    Notes
    -----
    A zero denominator yields NaN and a
    :class:`~nowcastbox.core.exceptions.DataQualityWarning`.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.preprocessing.transforms import PctChange
    >>> x = pd.Series([100.0, 105.0], index=pd.period_range("2020-01", periods=2, freq="M"))
    >>> PctChange(1).apply(x).round(6).tolist()
    [nan, 0.05]
    """

    _name = "pct_change"

    def _forward(self, x: np.ndarray, frequency: Frequency) -> np.ndarray:
        k = resolve_lag(self.lag, frequency)
        out = np.full_like(x, np.nan)
        prev = x[:-k]
        zero = prev == 0
        if bool(zero.any()):
            warnings.warn(
                f"pct_change({_lag_spec(self.lag)}): {int(zero.sum())} division(s) by zero; "
                "the results were set to NaN.",
                DataQualityWarning,
                stacklevel=4,
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            out[k:] = np.where(zero, np.nan, (x[k:] - prev) / np.where(zero, 1.0, prev))
        return out

    def _backward(self, y: np.ndarray, history: np.ndarray, frequency: Frequency) -> np.ndarray:
        k = resolve_lag(self.lag, frequency)
        return self._rebuild(y, history, k, lambda prev, val: prev * (1.0 + val))


class Compose(Transform):
    """Sequential composition of transforms (applied left to right).

    Usually built with ``|``: ``Log() | Diff(1)``. Nested compositions are flattened.

    Parameters
    ----------
    *steps : Transform
        At least one transform.

    Raises
    ------
    ValueError
        If no step is given or a step is not a :class:`Transform`.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.preprocessing.transforms import Diff, PctChange
    >>> code3 = PctChange("year") | Diff(1)
    >>> code3.to_spec()
    'pct_change(year)|diff(1)'
    >>> code3.n_lags("M")
    13
    """

    def __init__(self, *steps: Transform) -> None:
        if not steps:
            raise ValueError("Compose needs at least one step.")
        flat: list[Transform] = []
        for s in steps:
            if not isinstance(s, Transform):
                raise ValueError(f"Compose steps must be Transform objects, got {s!r}.")
            flat.extend(s.steps)
        self._steps = tuple(flat)

    @property
    def steps(self) -> tuple[Transform, ...]:
        """Elementary steps, in application order."""
        return self._steps

    def n_lags(self, frequency: FrequencyLike | None = None) -> int:
        """Total number of leading periods lost (sum over the steps).

        Parameters
        ----------
        frequency : Frequency, str or int, optional
            Native frequency (needed for calendar-span lags).

        Returns
        -------
        int
            Sum of the steps' lags.

        Examples
        --------
        >>> from nowcastbox.preprocessing.transforms import Diff, Log
        >>> (Log() | Diff(1) | Diff(12)).n_lags()
        13
        """
        return sum(s.n_lags(frequency) for s in self._steps)

    def _forward(self, x: np.ndarray, frequency: Frequency) -> np.ndarray:
        for s in self._steps:
            x = s._forward(x, frequency)
        return x

    def _backward(self, y: np.ndarray, history: np.ndarray, frequency: Frequency) -> np.ndarray:
        # histories of every intermediate stage: h_0 = levels, h_j = step_j(h_{j-1})
        stages = [history]
        for s in self._steps[:-1]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DataQualityWarning)
                stages.append(s._forward(stages[-1], frequency))
        out = y
        for s, h in zip(reversed(self._steps), reversed(stages), strict=True):
            out = s._backward(out, h, frequency)
        return out

    def to_spec(self) -> str:
        """Return the ``|``-joined specifications of the steps.

        Returns
        -------
        str
            Specification.

        Examples
        --------
        >>> from nowcastbox.preprocessing.transforms import Diff, Log
        >>> (Log() | Diff(1)).to_spec()
        'log|diff(1)'
        """
        return "|".join(s.to_spec() for s in self._steps)


# ---------------------------------------------------------------------- registry
NAMED_TRANSFORMS: dict[str, str] = {
    "level": "level",
    "levels": "level",
    "identity": "level",
    "none": "level",
    "log": "log",
    "diff": "diff(1)",
    "pct_change": "pct_change(1)",
    "growth": "pct_change(1)",
    "dlog": "log|diff(1)",
    "log_diff": "log|diff(1)",
    "mom": "pct_change(month)",
    "qoq": "pct_change(quarter)",
    "yoy": "pct_change(year)",
    "annual_diff": "diff(year)",
    "dlog_yoy": "log|diff(year)",
    "diff_of_yoy": "pct_change(year)|diff(1)",
    "diff_of_annual_diff": "diff(year)|diff(1)",
}
"""Named shortcuts and their canonical specifications."""

TRANSFORM_CODES: dict[int, str] = {
    0: "level",
    1: "pct_change",
    2: "diff",
    3: "diff_of_yoy",
    4: "diff_of_annual_diff",
    5: "annual_diff",
    6: "yoy",
    7: "qoq",
}
"""Legacy numeric codes 0-7 and the named transforms they stand for."""

_STEP_RE = re.compile(r"^\s*([a-z_]+)\s*(?:\(\s*([^()]*?)\s*\))?\s*$")

TransformLike: TypeAlias = Transform | str | int | None
"""Anything :func:`get_transform` accepts."""


def transform_from_code(code: int) -> Transform:
    """Return the transform of a legacy numeric code (0-7).

    Parameters
    ----------
    code : int
        Code between 0 and 7 (see :data:`TRANSFORM_CODES`). Integral floats such as
        ``2.0`` (common in legend tables) are accepted.

    Returns
    -------
    Transform
        The corresponding transform.

    Raises
    ------
    ValueError
        If ``code`` is not one of 0-7.

    Examples
    --------
    >>> from nowcastbox.preprocessing.transforms import transform_from_code
    >>> transform_from_code(3).to_spec()
    'pct_change(year)|diff(1)'
    """
    if isinstance(code, bool) or not isinstance(code, Real):
        raise ValueError(f"Transformation code must be an integer 0-7, got {code!r}.")
    if not float(code).is_integer() or int(code) not in TRANSFORM_CODES:
        raise ValueError(f"Unknown transformation code {code!r}; expected 0-7.")
    return get_transform(TRANSFORM_CODES[int(code)])


def _parse_step(text: str) -> Transform:
    m = _STEP_RE.match(text.lower())
    if m is None:
        raise ValueError(f"Cannot parse transformation {text!r}.")
    name, arg = m.group(1), m.group(2)
    if arg is None:
        if name in NAMED_TRANSFORMS:
            spec = NAMED_TRANSFORMS[name]
            return _parse_spec(spec) if spec != text.strip().lower() else _elementary(name, None)
        return _elementary(name, None)
    return _elementary(name, arg)


def _elementary(name: str, arg: str | None) -> Transform:
    if name in {"level", "identity"} and not arg:
        return Identity()
    if name == "log" and not arg:
        return Log()
    if name == "diff":
        return Diff(1 if arg is None else arg)
    if name == "pct_change":
        return PctChange(1 if arg is None else arg)
    if name == "scale" and arg:
        try:
            return Scale(float(arg))
        except ValueError as err:
            raise ValueError(f"Invalid scale factor {arg!r}: {err}") from None
    raise ValueError(
        f"Unknown transformation {name!r}"
        + ("" if arg is None else f" with argument {arg!r}")
        + f"; known names: {sorted(NAMED_TRANSFORMS)} and diff(k), pct_change(k), scale(c)."
    )


def _parse_spec(spec: str) -> Transform:
    parts = spec.split("|")
    if any(not p.strip() for p in parts):
        raise ValueError(f"Empty step in transformation specification {spec!r}.")
    steps = [_parse_step(p) for p in parts]
    return steps[0] if len(steps) == 1 else Compose(*steps)


def get_transform(spec: TransformLike) -> Transform:
    """Build a :class:`Transform` from a flexible specification.

    Parameters
    ----------
    spec : Transform, str, int or None
        * a :class:`Transform` (returned unchanged);
        * ``None`` - the identity;
        * a legacy code 0-7 (int, integral float or digit string);
        * a name from :data:`NAMED_TRANSFORMS` (``"dlog"``, ``"yoy"``, ...);
        * a specification string such as ``"diff(12)"``, ``"pct_change(year)"``,
          ``"scale(100)"`` or a ``|``-separated pipeline (``"log|diff|diff(12)"``).

    Returns
    -------
    Transform
        Parsed transform.

    Raises
    ------
    ValueError
        If ``spec`` cannot be parsed.

    Examples
    --------
    >>> from nowcastbox.preprocessing.transforms import get_transform
    >>> get_transform("dlog")
    Transform('log|diff(1)')
    >>> get_transform(7)
    Transform('pct_change(quarter)')
    >>> get_transform("log | diff(year)").to_spec()
    'log|diff(year)'
    """
    if spec is None:
        return Identity()
    if isinstance(spec, Transform):
        return spec
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            raise ValueError("Empty transformation specification.")
        if re.fullmatch(r"\d+(\.0+)?", text):
            return transform_from_code(int(float(text)))
        return _parse_spec(text)
    if isinstance(spec, Real) and not isinstance(spec, bool):
        if isinstance(spec, float) and np.isnan(spec):
            return Identity()
        return transform_from_code(spec)  # type: ignore[arg-type]
    raise ValueError(f"Cannot interpret {spec!r} as a transformation.")


# ---------------------------------------------------------------------- panels
def resolve_transforms(
    panel: MixedFrequencyData, transforms: Any, *, include_applied: bool = False
) -> dict[str, Any]:
    """Resolve a transformation specification into one specification per series.

    Parameters
    ----------
    panel : MixedFrequencyData
        Panel whose ``transform`` metadata provide the defaults.
    transforms : transform-like, mapping, Series, sequence or None
        One specification for all series, a mapping / Series by name (series not
        covered fall back to their metadata), a sequence aligned with the columns,
        or ``None`` (metadata for every series).
    include_applied : bool, default False
        Whether metadata defaults of series flagged ``transform_applied=True`` are
        returned. ``False`` (forward transformation): already-transformed series
        default to identity, so a panel is never transformed twice from its
        metadata. ``True`` (inversion): the recorded specification is returned.

    Returns
    -------
    dict
        ``{series: specification}`` (``None`` means identity).

    Raises
    ------
    NowcastDataError
        For unknown series names or a misaligned sequence.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> from nowcastbox.preprocessing.transforms import resolve_transforms
    >>> idx = pd.period_range("2020-01", periods=2, freq="M")
    >>> mfd = MixedFrequencyData(
    ...     pd.DataFrame({"a": [1.0, 2.0]}, index=idx), "M", transforms={"a": 2}
    ... )
    >>> resolve_transforms(mfd, None), resolve_transforms(mfd, "log")
    ({'a': 2}, {'a': 'log'})
    """

    def default(c: str) -> Any:
        meta = panel.metadata[c]
        return meta.transform if include_applied or not meta.transform_applied else None

    if transforms is None:
        return {c: default(c) for c in panel.columns}
    if isinstance(transforms, Transform | str | Integral | float) and not isinstance(
        transforms, bool
    ):
        return dict.fromkeys(panel.columns, transforms)
    if isinstance(transforms, Mapping | pd.Series):
        given = per_series(transforms, panel.columns, "transforms", default=None)
        return {c: (default(c) if given[c] is None else given[c]) for c in panel.columns}
    return per_series(transforms, panel.columns, "transforms")


def apply_transforms(
    data: PanelLike,
    transforms: Any = None,
    *,
    frequency: FrequencySpec | None = None,
) -> PanelLike:
    """Apply (possibly different) transformations to every series of a panel.

    Parameters
    ----------
    data : pandas.DataFrame or MixedFrequencyData
        Panel in levels on a base grid.
    transforms : transform-like, mapping, Series or sequence, optional
        One specification (see :func:`get_transform`) for all series, a mapping /
        Series keyed by series name, or a sequence aligned with the columns (e.g. a
        legend column of codes 0-7). Series not covered - or all series when omitted
        - use the ``transform`` metadata of a :class:`MixedFrequencyData` (identity
        if unset).
    frequency : frequency specification, optional
        Per-series native frequencies of a DataFrame (inferred when omitted).

    Returns
    -------
    pandas.DataFrame or MixedFrequencyData
        Transformed panel of the input type. For a :class:`MixedFrequencyData` the
        ``transform`` metadata of each series records the specification applied, so
        that :func:`invert_transforms` can undo it.

    Raises
    ------
    NowcastDataError
        On invalid data or specifications for unknown series.
    ValueError
        On an unknown transformation.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.preprocessing.transforms import apply_transforms
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> df = pd.DataFrame(
    ...     {
    ...         "ip": [100.0, 101, 102, 103, 104, 105],
    ...         "gdp": [np.nan, np.nan, 50.0, np.nan, np.nan, 55.0],
    ...     },
    ...     index=idx,
    ... )
    >>> out = apply_transforms(df, {"ip": "diff", "gdp": 1}, frequency={"ip": "M", "gdp": "Q"})
    >>> out["gdp"].round(6).tolist()
    [nan, nan, nan, nan, nan, 0.1]
    """
    panel, was_frame = as_panel(data, frequency)
    specs = resolve_transforms(panel, transforms)
    objs = {c: get_transform(s) for c, s in specs.items()}
    frame = map_native(
        panel,
        lambda n, s, f: objs[n]._forward(s.to_numpy(float), f),  # pyright: ignore[reportPrivateUsage]
    )
    out = panel.with_data(frame)
    if not was_frame:
        out = _record_applied(out, specs, objs)
    logger.debug("Applied transforms %s", {c: t.to_spec() for c, t in objs.items()})
    return return_like(out, was_frame)


def _record_applied(
    panel: MixedFrequencyData, specs: dict[str, Any], objs: dict[str, Transform]
) -> MixedFrequencyData:
    """Flag the series transformed by :func:`apply_transforms` in their metadata."""
    out = panel
    for c in panel.columns:
        if specs[c] is None:
            continue
        meta = panel.metadata[c]
        if meta.transform_applied and not isinstance(objs[c], Identity):
            warnings.warn(
                f"Series {c!r} was already transformed ({meta.transform!r}); applying "
                f"{specs[c]!r} on top of it. The metadata records only the last "
                "transformation, so invert_transforms will not undo both.",
                DataQualityWarning,
                stacklevel=3,
            )
        out = out.with_metadata(c, transform=specs[c], transform_applied=True)
    return out


def invert_transforms(
    transformed: PanelLike,
    history: PanelLike,
    transforms: Any = None,
    *,
    frequency: FrequencySpec | None = None,
) -> PanelLike:
    """Rebuild levels of every series of a transformed panel.

    Parameters
    ----------
    transformed : pandas.DataFrame or MixedFrequencyData
        Transformed values (e.g. a panel completed with nowcasts and forecasts).
    history : pandas.DataFrame or MixedFrequencyData
        Levels of the same series (at least :meth:`Transform.n_lags` observations
        before the first period to rebuild).
    transforms : transform-like, mapping, Series or sequence, optional
        Transformations that produced ``transformed``. Defaults to the ``transform``
        metadata of ``transformed`` (or of ``history``) when they are
        :class:`MixedFrequencyData`.
    frequency : frequency specification, optional
        Per-series native frequencies for DataFrame input.

    Returns
    -------
    pandas.DataFrame or MixedFrequencyData
        Levels on the union of both period ranges, of the type of ``transformed``.
        Known levels in ``history`` are kept unchanged.

    Raises
    ------
    NowcastDataError
        If the panels hold different series or base frequencies.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.preprocessing.transforms import apply_transforms, invert_transforms
    >>> idx = pd.period_range("2020-01", periods=4, freq="M")
    >>> lev = pd.DataFrame({"a": [1.0, 2.0, 4.0, 8.0]}, index=idx)
    >>> tr = apply_transforms(lev, "pct_change", frequency="M")
    >>> invert_transforms(tr, lev.iloc[:1], "pct_change", frequency="M")["a"].tolist()
    [1.0, 2.0, 4.0, 8.0]
    """
    t_panel, was_frame = as_panel(transformed, frequency)
    h_panel, _ = as_panel(
        history, frequency if not isinstance(history, MixedFrequencyData) else None
    )
    if set(t_panel.columns) != set(h_panel.columns):
        raise NowcastDataError(
            "transformed and history must contain the same series, got "
            f"{sorted(t_panel.columns)} and {sorted(h_panel.columns)}."
        )
    if t_panel.base_frequency != h_panel.base_frequency:
        raise NowcastDataError("transformed and history must share the base frequency.")
    source = t_panel
    if transforms is None and all(t_panel.metadata[c].transform is None for c in t_panel.columns):
        source = h_panel
    specs = resolve_transforms(source, transforms, include_applied=True)
    frames: dict[str, pd.Series] = {}
    for c in t_panel.columns:
        freq = t_panel.metadata[c].frequency
        frames[c] = get_transform(specs[c]).inverse(t_panel[c], h_panel[c], freq)
    frame = pd.DataFrame(frames)[t_panel.columns]
    if was_frame:
        return frame
    return MixedFrequencyData(
        frame,
        metadata={
            c: t_panel.metadata[c].replace(transform=None, transform_applied=False)
            for c in t_panel.columns
        },
        base_frequency=t_panel.base_frequency,
    )

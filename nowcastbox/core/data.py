"""Mixed-frequency panel container.

:class:`MixedFrequencyData` is the data contract shared by every nowcastbox module.
It wraps a :class:`pandas.DataFrame` whose index is a contiguous
:class:`pandas.PeriodIndex` on the *base grid* (monthly in v0.1) together with
per-series metadata (:class:`SeriesMetadata`).

Conventions
-----------
* **Base grid** - rows are consecutive periods of the base frequency (``freq="M"``
  for the monthly + quarterly case). Gaps are filled with NaN rows (with a
  :class:`~nowcastbox.core.exceptions.DataQualityWarning`).
* **Lower-frequency storage** - a series of lower frequency is stored on the last
  base period of each of its periods: quarterly values in March, June, September and
  December; annual values in December. Every other cell of that column is
  structurally NaN; a value there raises
  :class:`~nowcastbox.core.exceptions.NowcastDataError`.
* **Values** are ``float64``; missing data are ``NaN``. Infinite values are rejected.
* **Immutability** - every method returns a new object; the wrapped frame is never
  exposed by reference (``to_frame`` and ``data`` return copies).
* **Standardisation** - ``(x - mean) / std`` column-wise over observed values, with
  ``ddof=1`` by default; the statistics are returned as
  :class:`StandardizationStats` so that results can be mapped back to original units.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Literal, TypeAlias, TypeVar

import numpy as np
import pandas as pd

from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.frequency import (
    AggregationType,
    Frequency,
    FrequencyLike,
    aggregation_ratio,
    infer_frequency,
    is_period_end,
    native_to_base,
    period_to_base,
)

__all__ = [
    "FrequencySpec",
    "MixedFrequencyData",
    "SeriesCategory",
    "SeriesMetadata",
    "StandardizationStats",
    "as_mixed_frequency_data",
]

FrequencySpec: TypeAlias = (
    Mapping[str, FrequencyLike] | pd.Series | Sequence[FrequencyLike] | FrequencyLike
)
"""Accepted specifications of per-series frequencies (see :class:`MixedFrequencyData`)."""

_T = TypeVar("_T")


class SeriesCategory(str, Enum):
    """Economic category of a series, used to aggregate news (plan innovation I6).

    Examples
    --------
    >>> from nowcastbox.core.data import SeriesCategory
    >>> SeriesCategory.from_value("Soft")
    <SeriesCategory.SOFT: 'soft'>
    """

    HARD = "hard"
    SOFT = "soft"
    FINANCIAL = "financial"
    OTHER = "other"

    @classmethod
    def from_value(cls, value: SeriesCategory | str) -> SeriesCategory:
        """Parse a category from a member or a case-insensitive string.

        Parameters
        ----------
        value : SeriesCategory or str
            Category (``"hard"``, ``"soft"``, ``"financial"`` or ``"other"``).

        Returns
        -------
        SeriesCategory
            Parsed member.

        Raises
        ------
        ValueError
            If the value is not a known category.

        Examples
        --------
        >>> SeriesCategory.from_value("HARD")
        <SeriesCategory.HARD: 'hard'>
        """
        if isinstance(value, SeriesCategory):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        raise ValueError(
            f"Unknown series category {value!r}; expected one of {[c.value for c in cls]}."
        )


def _is_missing_scalar(value: object) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _coerce_flag(value: object) -> bool:
    """Coerce a boolean metadata flag (``None``/NaN -> ``False``; no strings)."""
    if _is_missing_scalar(value) or value is pd.NA:
        return False
    if isinstance(value, bool | np.bool_) or (isinstance(value, Real) and value in (0, 1)):
        return bool(value)
    raise ValueError(f"transform_applied must be a boolean, got {value!r}.")


_MIN_RELEASE_DELAY = -366
"""Smallest (exclusive) release delay: at most one year before the end of the period."""


def _coerce_delay(value: object) -> int | None:
    if _is_missing_scalar(value) or value is pd.NA:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"release_delay must be an integer number of days, got {value!r}.")
    as_float = float(value)
    if not as_float.is_integer() or as_float <= _MIN_RELEASE_DELAY:
        raise ValueError(
            f"release_delay must be an integer larger than {_MIN_RELEASE_DELAY} days "
            f"(negative: released before the end of the period), got {value!r}."
        )
    return int(as_float)


def _coerce_blocks(value: object) -> tuple[str, ...]:
    if _is_missing_scalar(value):
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        blocks = tuple(str(b) for b in value)  # pyright: ignore[reportUnknownVariableType]
        if len(set(blocks)) != len(blocks):
            raise ValueError(f"Duplicated block names in {blocks!r}.")
        return blocks
    raise ValueError(f"blocks must be a string or a sequence of strings, got {value!r}.")


@dataclass(frozen=True)
class SeriesMetadata:
    """Metadata of one series of a :class:`MixedFrequencyData` panel.

    Parameters
    ----------
    name : str
        Series name (column of the data frame).
    frequency : Frequency or str or int
        Native observation frequency (coerced with :meth:`Frequency.from_value`).
    transform : int or str, optional
        Transformation code (0-7) or name applied by ``preprocessing``; stored as given.
    release_delay : int, optional
        Publication lag in **days** after the end of the reference period. Used by
        :meth:`MixedFrequencyData.as_of` and the vintages module.
    blocks : tuple of str, default ()
        Factor blocks the series loads on (e.g. ``("global", "real")``).
    category : SeriesCategory or str, optional
        ``"hard"``, ``"soft"``, ``"financial"`` or ``"other"``.
    description : str, default ""
        Free-text description.
    aggregation : AggregationType or str, optional
        Link to the latent high-frequency series for lower-frequency series
        (``None`` lets each model use its documented default).
    units : str, default ""
        Units of measurement.
    transform_applied : bool, default False
        Whether ``transform`` has **already been applied** to the stored values.
        ``False`` (default): ``transform`` is the transformation *to apply* (the values
        are in levels), as in a dataset legend. ``True``: the values are already
        transformed and ``transform`` records what was applied (set by
        ``preprocessing.apply_transforms``/``prepare_panel``), so the metadata default
        is not applied a second time and ``invert_transforms`` can undo it.

    Examples
    --------
    >>> from nowcastbox.core.data import SeriesMetadata
    >>> m = SeriesMetadata("gdp", "Q", release_delay=60, blocks=["global"])
    >>> m.frequency, m.blocks
    (<Frequency.QUARTERLY: 'Q'>, ('global',))
    >>> m.replace(release_delay=45).release_delay
    45
    """

    name: str
    frequency: Frequency
    transform: int | str | None = None
    release_delay: int | None = None
    blocks: tuple[str, ...] = ()
    category: SeriesCategory | None = None
    description: str = ""
    aggregation: AggregationType | None = None
    units: str = ""
    transform_applied: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"Series name must be a non-empty string, got {self.name!r}.")
        object.__setattr__(self, "frequency", Frequency.from_value(self.frequency))
        object.__setattr__(self, "release_delay", _coerce_delay(self.release_delay))
        object.__setattr__(self, "blocks", _coerce_blocks(self.blocks))
        if _is_missing_scalar(self.transform):
            object.__setattr__(self, "transform", None)
        cat: Any = self.category
        object.__setattr__(
            self, "category", None if _is_missing_scalar(cat) else SeriesCategory.from_value(cat)
        )
        agg: Any = self.aggregation
        object.__setattr__(
            self,
            "aggregation",
            None if _is_missing_scalar(agg) else AggregationType.from_value(agg),
        )
        if _is_missing_scalar(self.description):
            object.__setattr__(self, "description", "")
        object.__setattr__(self, "transform_applied", _coerce_flag(self.transform_applied))

    def replace(self, **changes: Any) -> SeriesMetadata:
        """Return a copy with some fields replaced (validated again).

        Parameters
        ----------
        **changes
            Field values to replace.

        Returns
        -------
        SeriesMetadata
            New metadata object.

        Examples
        --------
        >>> SeriesMetadata("ip", "M").replace(category="hard").category
        <SeriesCategory.HARD: 'hard'>
        """
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return the metadata as a plain dictionary (enums as their string values).

        Returns
        -------
        dict
            Field name to value.

        Examples
        --------
        >>> SeriesMetadata("ip", "M").to_dict()["frequency"]
        'M'
        """
        out = dataclasses.asdict(self)
        out["frequency"] = self.frequency.value
        out["category"] = None if self.category is None else self.category.value
        out["aggregation"] = None if self.aggregation is None else self.aggregation.value
        out["blocks"] = list(self.blocks)
        return out


@dataclass(frozen=True, eq=False)
class StandardizationStats:
    """Column means and standard deviations used to standardise a panel.

    Parameters
    ----------
    mean : pandas.Series
        Mean of each series (indexed by series name).
    std : pandas.Series
        Standard deviation of each series (strictly positive).
    ddof : int, default 1
        Delta degrees of freedom used for ``std``.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.core.data import StandardizationStats
    >>> stats = StandardizationStats(pd.Series({"a": 1.0}), pd.Series({"a": 2.0}))
    >>> stats.inverse_series(0.5, "a")
    2.0
    """

    mean: pd.Series
    std: pd.Series
    ddof: int = 1

    def __post_init__(self) -> None:
        mean = pd.Series(self.mean, dtype=float).copy()
        std = pd.Series(self.std, dtype=float).copy()
        if not mean.index.equals(std.index):
            raise ValueError("mean and std must be indexed by the same series names.")
        if bool((~np.isfinite(std.to_numpy()) | (std.to_numpy() <= 0)).any()):
            raise ValueError("Standard deviations must be finite and strictly positive.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @property
    def columns(self) -> list[str]:
        """Series names covered by the statistics."""
        return [str(c) for c in self.mean.index]

    def _check_columns(self, columns: Iterable[object]) -> list[str]:
        cols = [str(c) for c in columns]
        unknown = sorted(set(cols) - set(self.columns))
        if unknown:
            raise NowcastDataError(f"No standardisation statistics for series {unknown}.")
        return cols

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Standardise ``frame``: ``(x - mean) / std`` column by column.

        Parameters
        ----------
        frame : pandas.DataFrame
            Data whose columns are a subset of :attr:`columns`.

        Returns
        -------
        pandas.DataFrame
            Standardised copy.

        Raises
        ------
        NowcastDataError
            If ``frame`` has columns without statistics.

        Examples
        --------
        >>> import pandas as pd
        >>> s = StandardizationStats(pd.Series({"a": 1.0}), pd.Series({"a": 2.0}))
        >>> s.transform(pd.DataFrame({"a": [3.0]}))["a"].tolist()
        [1.0]
        """
        cols = self._check_columns(frame.columns)
        return (frame - self.mean[cols]) / self.std[cols]

    def inverse_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Map standardised data back to original units: ``z * std + mean``.

        Parameters
        ----------
        frame : pandas.DataFrame
            Standardised data whose columns are a subset of :attr:`columns`.

        Returns
        -------
        pandas.DataFrame
            Copy in original units.

        Raises
        ------
        NowcastDataError
            If ``frame`` has columns without statistics.

        Examples
        --------
        >>> import pandas as pd
        >>> s = StandardizationStats(pd.Series({"a": 1.0}), pd.Series({"a": 2.0}))
        >>> s.inverse_transform(pd.DataFrame({"a": [1.0]}))["a"].tolist()
        [3.0]
        """
        cols = self._check_columns(frame.columns)
        return frame * self.std[cols] + self.mean[cols]

    def inverse_series(self, values: _T, column: str, *, scale_only: bool = False) -> _T:
        """Map standardised values of one series back to original units.

        Parameters
        ----------
        values : float, numpy.ndarray or pandas.Series/DataFrame
            Standardised values.
        column : str
            Series whose statistics to use.
        scale_only : bool, default False
            Only multiply by the standard deviation (for standard deviations,
            forecast errors or news impacts, which do not carry the mean).

        Returns
        -------
        same type as ``values``
            Values in original units.

        Raises
        ------
        NowcastDataError
            If ``column`` has no statistics.

        Examples
        --------
        >>> import pandas as pd
        >>> s = StandardizationStats(pd.Series({"a": 1.0}), pd.Series({"a": 2.0}))
        >>> s.inverse_series(1.0, "a", scale_only=True)
        2.0
        """
        self._check_columns([column])
        sd = float(self.std[column])
        mu = 0.0 if scale_only else float(self.mean[column])
        return values * sd + mu  # type: ignore[operator, return-value]


@dataclass(frozen=True)
class _MetaSources:
    """Per-series keyword overrides passed to :class:`MixedFrequencyData`."""

    frequencies: Mapping[str, Any] = field(default_factory=dict[str, Any])
    transforms: Mapping[str, Any] = field(default_factory=dict[str, Any])
    release_delays: Mapping[str, Any] = field(default_factory=dict[str, Any])
    blocks: Mapping[str, Any] = field(default_factory=dict[str, Any])
    categories: Mapping[str, Any] = field(default_factory=dict[str, Any])
    descriptions: Mapping[str, Any] = field(default_factory=dict[str, Any])
    aggregations: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def field_map(self) -> dict[str, Mapping[str, Any]]:
        return {
            "frequency": self.frequencies,
            "transform": self.transforms,
            "release_delay": self.release_delays,
            "blocks": self.blocks,
            "category": self.categories,
            "description": self.descriptions,
            "aggregation": self.aggregations,
        }


def _as_name_mapping(spec: object, columns: Sequence[str], what: str) -> dict[str, Any]:
    """Normalise a per-series specification to ``{name: value}``.

    Accepts None, a mapping, a Series indexed by names (or positional when its index is a
    RangeIndex of matching length), a list/tuple/ndarray aligned with ``columns`` or a
    scalar applied to every column.
    """
    if spec is None:
        return {}
    if isinstance(spec, pd.Series):
        series: pd.Series = spec  # pyright: ignore[reportUnknownVariableType]
        if isinstance(series.index, pd.RangeIndex) and len(series) == len(columns):
            return dict(zip(columns, series.tolist(), strict=True))
        return {str(k): v for k, v in series.items()}
    if isinstance(spec, Mapping):
        return {str(k): v for k, v in spec.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    if isinstance(spec, list | tuple | np.ndarray):
        values = list(spec)  # pyright: ignore[reportUnknownArgumentType]
        if len(values) != len(columns):
            raise NowcastDataError(
                f"{what}: got {len(values)} values for {len(columns)} series; pass a mapping "
                "{series: value} or a sequence aligned with the columns."
            )
        return dict(zip(columns, values, strict=True))
    return dict.fromkeys(columns, spec)


def _blocks_mapping(spec: object, columns: Sequence[str]) -> tuple[dict[str, Any], list[str]]:
    """Normalise a block specification; returns the mapping and the ordered block names."""
    if isinstance(spec, pd.DataFrame):
        frame: pd.DataFrame = spec
        names = [str(c) for c in frame.columns]
        mapping: dict[str, Any] = {}
        for row_name, row in frame.iterrows():
            flags = row.fillna(0).astype(bool).to_numpy()
            mapping[str(row_name)] = tuple(n for n, f in zip(names, flags, strict=True) if f)
        return mapping, names
    mapping = _as_name_mapping(spec, columns, "blocks")
    names: list[str] = []
    for value in mapping.values():
        for b in _coerce_blocks(value):
            if b not in names:
                names.append(b)
    return mapping, names


_GROUPINGS = ("series", "category", "block", "frequency")
"""Named groupings of :meth:`MixedFrequencyData.released_share` (plus mappings)."""


def _mapping_groups(
    names: Sequence[str], mapping: Mapping[str, str | Sequence[str]]
) -> dict[str, list[str]]:
    unknown = sorted(set(mapping) - set(names))
    if unknown:
        raise ValueError(f"Grouping given for unknown series {unknown}.")
    groups: dict[str, list[str]] = {}
    for name in names:
        labels = mapping.get(name, "unassigned")
        for label in [labels] if isinstance(labels, str) else list(labels):
            groups.setdefault(str(label), []).append(name)
    return groups


def _series_groups(
    metadata: Mapping[str, SeriesMetadata], by: str | Mapping[str, str | Sequence[str]] | None
) -> dict[str, list[str]]:
    """``{group: [series, ...]}`` in order of first appearance (internal, shared helper).

    ``by`` is ``None``/``"series"`` (one group per series), ``"category"``
    (``"uncategorized"`` when unset), ``"block"`` (a series in each of its blocks;
    ``"unassigned"`` without blocks), ``"frequency"`` or a mapping
    ``{series: group or [groups]}``.
    """
    names = list(metadata)
    if by is None or by == "series":
        return {name: [name] for name in names}
    if isinstance(by, Mapping):
        return _mapping_groups(names, by)
    if by == "category":
        labels = {
            n: m.category.value if m.category else "uncategorized" for n, m in metadata.items()
        }
    elif by == "block":
        labels = {n: list(m.blocks) or "unassigned" for n, m in metadata.items()}
    elif by == "frequency":
        labels = {n: m.frequency.label for n, m in metadata.items()}
    else:
        raise ValueError(f"by must be one of {_GROUPINGS}, a mapping or None; got {by!r}.")
    return _mapping_groups(names, labels)


def _period_counts(
    panel: MixedFrequencyData, period: pd.Period | str
) -> tuple[pd.Series, pd.Series]:
    """Released and expected observations of each series inside ``period``."""
    target = period if isinstance(period, pd.Period) else pd.Period(period)
    base = panel.base_frequency
    if Frequency.from_value(target.freqstr).is_higher_than(base):
        raise NowcastDataError(
            f"period {target} is of a higher frequency than the base grid ({base.label})."
        )
    grid = pd.period_range(
        pd.Period(target.start_time, freq=base.pandas_freq),
        pd.Period(target.end_time, freq=base.pandas_freq),
        freq=base.pandas_freq,
    )
    grid = grid[np.asarray(grid.asfreq(target.freqstr, how="E") == target)]
    observed = panel.to_frame().reindex(grid).notna()
    expected, released = {}, {}
    for name, meta in panel.metadata.items():
        slots = is_period_end(grid, meta.frequency)
        expected[name] = int(slots.sum())
        released[name] = int((slots & observed[name].to_numpy()).sum())
    return pd.Series(released, dtype=int), pd.Series(expected, dtype=int)


def _share_weights(
    expected: pd.Series, weights: Mapping[str, float] | pd.Series | None
) -> pd.Series:
    """Weights of each series (default: the number of expected observations)."""
    if weights is None:
        return expected.astype(float)
    given = pd.Series(weights, dtype=float)
    unknown = sorted(set(given.index) - set(expected.index))
    if unknown:
        raise ValueError(f"weights given for unknown series {unknown}.")
    if bool((given < 0).any()) or not bool(np.isfinite(given).all()):
        raise ValueError("weights must be finite and non-negative.")
    return given.reindex(expected.index, fill_value=0.0)


def _share_row(released: pd.Series, expected: pd.Series, weight: pd.Series) -> dict[str, float]:
    """One row of :meth:`MixedFrequencyData.released_share`."""
    has_slots = expected > 0
    share = released[has_slots] / expected[has_slots]
    w = weight[has_slots]
    total = float(w.sum())
    return {
        "released": int(released.sum()),
        "expected": int(expected.sum()),
        "weight": total,
        "share": float((share * w).sum() / total) if total > 0 else np.nan,
    }


def _to_period(
    value: pd.Period | str | pd.Timestamp, base: Frequency, how: Literal["S", "E"]
) -> pd.Period:
    period = value if isinstance(value, pd.Period) else pd.Period(value)
    own = Frequency.from_value(period.freqstr)
    if own.is_lower_than(base):
        return period_to_base(period, base, how="start" if how == "S" else "end")
    return period.asfreq(base.pandas_freq)


class MixedFrequencyData:
    """Panel of time series of mixed frequencies on a common base period grid.

    Parameters
    ----------
    data : pandas.DataFrame
        Numeric data, one column per series. The index must be a
        :class:`pandas.PeriodIndex` (its frequency is the base frequency) or a
        :class:`pandas.DatetimeIndex` (converted to periods of ``base_frequency``).
        Lower-frequency series must only have values on the last base period of their
        own periods (quarterly values in the third month of the quarter).
    frequencies : mapping, Series, sequence or scalar, optional
        Native frequency of each series: ``{"gdp": "Q", ...}``, a Series indexed by
        series name (e.g. a legend column of 12/4 codes), a sequence aligned with the
        columns, or a single value for every series. Series not covered here nor in
        ``metadata`` have their frequency inferred with
        :func:`~nowcastbox.core.frequency.infer_frequency`.
    metadata : mapping, optional
        ``{name: SeriesMetadata or dict}`` with any :class:`SeriesMetadata` fields.
    transforms, release_delays, categories, descriptions, aggregations : optional
        Per-series values of the corresponding :class:`SeriesMetadata` fields, in any
        form accepted by ``frequencies``. They override ``metadata``.
    blocks : mapping or pandas.DataFrame, optional
        Block memberships: ``{name: ["global", "real"]}`` or a 0/1 DataFrame with one
        row per series and one column per block (NY Fed layout).
    base_frequency : Frequency or str, optional
        Base grid frequency. Required for a DatetimeIndex without an inferable
        frequency; must agree with the frequency of a PeriodIndex.

    Raises
    ------
    NowcastDataError
        On an invalid index, duplicated/non-string column names, non-numeric or
        infinite values, a series of higher frequency than the base grid, values of a
        lower-frequency series outside its storage slots, or metadata for unknown
        series.

    Warns
    -----
    DataQualityWarning
        When the index has gaps (filled with NaN rows) or a series has no observation.

    See Also
    --------
    nowcastbox.core.frequency.Frequency : Supported frequencies.
    SeriesMetadata : Per-series metadata.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from nowcastbox.core.data import MixedFrequencyData
    >>> idx = pd.period_range("2020-01", periods=6, freq="M")
    >>> df = pd.DataFrame(
    ...     {
    ...         "ip": [0.1, 0.2, -0.1, 0.3, 0.0, np.nan],
    ...         "gdp": [np.nan, np.nan, 0.5, np.nan, np.nan, np.nan],
    ...     },
    ...     index=idx,
    ... )
    >>> mfd = MixedFrequencyData(df, frequencies={"ip": "M", "gdp": "Q"})
    >>> mfd.n_series, mfd.quarterly_columns
    (2, ['gdp'])
    >>> mfd.ragged_edge_mask()["gdp"].tolist()
    [False, False, False, False, False, True]
    """

    __slots__ = ("_base", "_block_names", "_frame", "_meta")

    _frame: pd.DataFrame
    _meta: dict[str, SeriesMetadata]
    _base: Frequency
    _block_names: tuple[str, ...]

    def __init__(
        self,
        data: pd.DataFrame,
        frequencies: FrequencySpec | None = None,
        *,
        metadata: Mapping[str, SeriesMetadata | Mapping[str, Any]] | None = None,
        transforms: Mapping[str, Any] | pd.Series | Sequence[Any] | None = None,
        release_delays: Mapping[str, Any] | pd.Series | Sequence[Any] | None = None,
        blocks: Mapping[str, Any] | pd.DataFrame | None = None,
        categories: Mapping[str, Any] | pd.Series | Sequence[Any] | None = None,
        descriptions: Mapping[str, Any] | pd.Series | Sequence[Any] | None = None,
        aggregations: Mapping[str, Any] | pd.Series | Sequence[Any] | None = None,
        base_frequency: FrequencyLike | None = None,
    ) -> None:
        if not isinstance(data, pd.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise NowcastDataError(f"data must be a pandas DataFrame, got {type(data).__name__}.")
        frame, base = self._prepare_frame(data, base_frequency)
        columns = [str(c) for c in frame.columns]
        block_map, block_names = _blocks_mapping(blocks, columns)
        sources = _MetaSources(
            frequencies=_as_name_mapping(frequencies, columns, "frequencies"),
            transforms=_as_name_mapping(transforms, columns, "transforms"),
            release_delays=_as_name_mapping(release_delays, columns, "release_delays"),
            blocks=block_map,
            categories=_as_name_mapping(categories, columns, "categories"),
            descriptions=_as_name_mapping(descriptions, columns, "descriptions"),
            aggregations=_as_name_mapping(aggregations, columns, "aggregations"),
        )
        meta = self._build_metadata(frame, base, metadata or {}, sources)
        self._validate_slots(frame, meta, base)
        self._frame = frame
        self._meta = meta
        self._base = base
        self._block_names = self._resolve_block_names(meta, block_names)
        self._warn_empty_series()

    # ------------------------------------------------------------------ construction
    @staticmethod
    def _resolve_base(index: pd.Index, base_frequency: FrequencyLike | None) -> Frequency:
        requested = None if base_frequency is None else Frequency.from_value(base_frequency)
        if isinstance(index, pd.PeriodIndex):
            own = Frequency.from_index(index)
            if requested is not None and requested != own:
                raise NowcastDataError(
                    f"base_frequency={requested.value!r} disagrees with the PeriodIndex "
                    f"frequency {own.value!r}."
                )
            return own
        if isinstance(index, pd.DatetimeIndex):
            if requested is not None:
                return requested
            inferred = pd.infer_freq(index) if len(index) >= 3 else None
            if inferred is None:
                raise NowcastDataError(
                    "Cannot infer the base frequency of the DatetimeIndex; pass base_frequency."
                )
            return Frequency.from_value(inferred)
        raise NowcastDataError(
            f"The index must be a pandas PeriodIndex or DatetimeIndex, got {type(index).__name__}."
        )

    @classmethod
    def _prepare_frame(
        cls, data: pd.DataFrame, base_frequency: FrequencyLike | None
    ) -> tuple[pd.DataFrame, Frequency]:
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise NowcastDataError("data must have at least one row and one column.")
        bad_names = [c for c in data.columns if not isinstance(c, str) or not c]
        if bad_names:
            raise NowcastDataError(f"Column names must be non-empty strings, got {bad_names}.")
        if data.columns.has_duplicates:
            dup = sorted(set(data.columns[data.columns.duplicated()]))
            raise NowcastDataError(f"Duplicated column names: {dup}.")
        base = cls._resolve_base(data.index, base_frequency)
        frame = data.copy()
        if isinstance(frame.index, pd.DatetimeIndex):
            frame.index = frame.index.to_period(base.pandas_freq)
        frame = cls._coerce_numeric(frame)
        frame = cls._regularize_index(frame, base)
        return frame, base

    @staticmethod
    def _coerce_numeric(frame: pd.DataFrame) -> pd.DataFrame:
        try:
            out = frame.astype(float)
        except (TypeError, ValueError) as err:
            raise NowcastDataError(f"All series must be numeric: {err}") from err
        values = out.to_numpy()
        if bool(np.isinf(values).any()):
            cols = out.columns[np.isinf(values).any(axis=0)].tolist()
            raise NowcastDataError(f"Infinite values in series {cols}.")
        return out

    @staticmethod
    def _regularize_index(frame: pd.DataFrame, base: Frequency) -> pd.DataFrame:
        if frame.index.has_duplicates:
            dup = frame.index[frame.index.duplicated()].astype(str).tolist()[:5]
            raise NowcastDataError(f"Duplicated periods in the index, e.g. {dup}.")
        frame = frame.sort_index()
        index = frame.index
        assert isinstance(index, pd.PeriodIndex)  # noqa: S101 - guaranteed by _resolve_base
        full = pd.period_range(index[0], index[-1], freq=base.pandas_freq)
        if len(full) != len(index):
            warnings.warn(
                f"The index has {len(full) - len(index)} missing {base.label} periods; "
                "they were inserted as rows of NaN.",
                DataQualityWarning,
                stacklevel=4,
            )
            frame = frame.reindex(full)
        frame.index.name = frame.index.name or "period"
        frame.columns = pd.Index([str(c) for c in frame.columns])
        return frame

    @staticmethod
    def _build_metadata(
        frame: pd.DataFrame,
        base: Frequency,
        metadata: Mapping[str, SeriesMetadata | Mapping[str, Any]],
        sources: _MetaSources,
    ) -> dict[str, SeriesMetadata]:
        columns = [str(c) for c in frame.columns]
        known = set(columns)
        fields = sources.field_map()
        for what, mapping in [("metadata", metadata), *fields.items()]:
            unknown = sorted(set(mapping) - known)
            if unknown:
                raise NowcastDataError(f"{what} given for unknown series {unknown}.")
        return {
            name: MixedFrequencyData._series_metadata(frame, name, metadata.get(name), fields)
            for name in columns
        }

    @staticmethod
    def _series_metadata(
        frame: pd.DataFrame,
        name: str,
        given: SeriesMetadata | Mapping[str, Any] | None,
        fields: Mapping[str, Mapping[str, Any]],
    ) -> SeriesMetadata:
        values: dict[str, Any] = (
            given.to_dict() if isinstance(given, SeriesMetadata) else dict(given or {})
        )
        values["name"] = name
        values.update({f: mapping[name] for f, mapping in fields.items() if name in mapping})
        if _is_missing_scalar(values.get("frequency")):
            values["frequency"] = infer_frequency(frame[name])
        try:
            return SeriesMetadata(**values)
        except (TypeError, ValueError) as err:
            raise NowcastDataError(f"Invalid metadata for series {name!r}: {err}") from err

    @staticmethod
    def _validate_slots(
        frame: pd.DataFrame, meta: Mapping[str, SeriesMetadata], base: Frequency
    ) -> None:
        index = frame.index
        assert isinstance(index, pd.PeriodIndex)  # noqa: S101
        slots: dict[Frequency, np.ndarray] = {}
        for name, m in meta.items():
            if m.frequency.is_higher_than(base):
                raise NowcastDataError(
                    f"Series {name!r} is {m.frequency.label}, higher than the {base.label} base "
                    "grid; aggregate it first or use a finer base grid."
                )
            if m.frequency == base:
                continue
            if m.frequency not in slots:
                slots[m.frequency] = is_period_end(index, m.frequency)
            off = frame[name].notna().to_numpy() & ~slots[m.frequency]
            if bool(off.any()):
                examples = index[off].astype(str).tolist()[:3]
                raise NowcastDataError(
                    f"Series {name!r} is {m.frequency.label} but has values outside the last "
                    f"{base.label} period of its periods, e.g. at {examples}."
                )

    @staticmethod
    def _resolve_block_names(
        meta: Mapping[str, SeriesMetadata], preferred: Sequence[str]
    ) -> tuple[str, ...]:
        names = list(preferred)
        for m in meta.values():
            for b in m.blocks:
                if b not in names:
                    names.append(b)
        return tuple(names)

    def _warn_empty_series(self) -> None:
        empty = self._frame.columns[self._frame.isna().all().to_numpy()].tolist()
        if empty:
            warnings.warn(
                f"Series without any observation: {empty}.", DataQualityWarning, stacklevel=3
            )

    @classmethod
    def from_series(
        cls,
        series: Mapping[str, pd.Series],
        *,
        base_frequency: FrequencyLike = Frequency.MONTHLY,
        **metadata_kwargs: Any,
    ) -> MixedFrequencyData:
        """Build a panel from series indexed at their own (native) frequencies.

        Parameters
        ----------
        series : mapping of str to pandas.Series
            Each series must have a :class:`pandas.PeriodIndex` of frequency lower than
            or equal to ``base_frequency``; its frequency becomes the series frequency.
        base_frequency : Frequency or str, default "M"
            Frequency of the base grid.
        **metadata_kwargs
            Any keyword argument of :class:`MixedFrequencyData` except ``frequencies``
            and ``base_frequency`` (e.g. ``release_delays``, ``blocks``).

        Returns
        -------
        MixedFrequencyData
            Panel covering the union of all sample periods.

        Raises
        ------
        NowcastDataError
            If ``series`` is empty or a series is not indexed by a PeriodIndex.

        Examples
        --------
        >>> import pandas as pd
        >>> from nowcastbox.core.data import MixedFrequencyData
        >>> gdp = pd.Series([1.0, 2.0], index=pd.period_range("2020Q1", periods=2, freq="Q"))
        >>> ip = pd.Series(range(6), index=pd.period_range("2020-01", periods=6, freq="M"))
        >>> mfd = MixedFrequencyData.from_series({"gdp": gdp, "ip": ip})
        >>> mfd["gdp"].dropna().index.astype(str).tolist()
        ['2020-03', '2020-06']
        """
        if not series:
            raise NowcastDataError("from_series needs at least one series.")
        if "frequencies" in metadata_kwargs:
            raise TypeError("frequencies are taken from the series indexes; do not pass them.")
        base = Frequency.from_value(base_frequency)
        placed: dict[str, pd.Series] = {}
        freqs: dict[str, Frequency] = {}
        for name, s in series.items():
            if not isinstance(s.index, pd.PeriodIndex):
                raise NowcastDataError(f"Series {name!r} must be indexed by a pandas.PeriodIndex.")
            freqs[name] = Frequency.from_index(s.index)
            try:
                new_index = native_to_base(s.index, base)
            except ValueError as err:
                raise NowcastDataError(f"Series {name!r}: {err}") from err
            placed[name] = pd.Series(s.to_numpy(dtype=float), index=new_index, name=name)
        start = min(p.index.min() for p in placed.values())
        end = max(p.index.max() for p in placed.values())
        grid = pd.period_range(start, end, freq=base.pandas_freq, name="period")
        frame = pd.DataFrame({name: p.reindex(grid) for name, p in placed.items()}, index=grid)
        return cls(frame, freqs, base_frequency=base, **metadata_kwargs)

    # ------------------------------------------------------------------ basic properties
    @property
    def base_frequency(self) -> Frequency:
        """Frequency of the base grid (rows)."""
        return self._base

    @property
    def index(self) -> pd.PeriodIndex:
        """Base-grid :class:`pandas.PeriodIndex` (immutable)."""
        index = self._frame.index
        assert isinstance(index, pd.PeriodIndex)  # noqa: S101
        return index

    @property
    def columns(self) -> list[str]:
        """Series names, in column order."""
        return [str(c) for c in self._frame.columns]

    @property
    def n_series(self) -> int:
        """Number of series (columns)."""
        return self._frame.shape[1]

    @property
    def n_periods(self) -> int:
        """Number of base-grid periods (rows)."""
        return self._frame.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_periods, n_series)``."""
        return (self.n_periods, self.n_series)

    @property
    def start(self) -> pd.Period:
        """First base period."""
        return self.index[0]

    @property
    def end(self) -> pd.Period:
        """Last base period."""
        return self.index[-1]

    @property
    def data(self) -> pd.DataFrame:
        """Copy of the underlying data frame (same as :meth:`to_frame`)."""
        return self._frame.copy()

    @property
    def values(self) -> np.ndarray:
        """Copy of the data as a ``(n_periods, n_series)`` float array."""
        return self._frame.to_numpy(dtype=float, copy=True)

    @property
    def metadata(self) -> dict[str, SeriesMetadata]:
        """Mapping of series name to its (frozen) :class:`SeriesMetadata`."""
        return dict(self._meta)

    @property
    def frequencies(self) -> pd.Series:
        """Native frequency of each series (Series of :class:`Frequency`)."""
        return pd.Series(
            {n: m.frequency for n, m in self._meta.items()}, name="frequency", dtype=object
        )

    @property
    def transforms(self) -> pd.Series:
        """Transformation code/name of each series (``None`` when unset)."""
        return pd.Series(
            {n: m.transform for n, m in self._meta.items()}, name="transform", dtype=object
        )

    @property
    def release_delays(self) -> pd.Series:
        """Publication delay in days of each series (nullable ``Int64``)."""
        return pd.Series(
            {n: m.release_delay for n, m in self._meta.items()}, name="release_delay", dtype="Int64"
        )

    @property
    def categories(self) -> pd.Series:
        """Category of each series (:class:`SeriesCategory` or ``None``)."""
        return pd.Series(
            {n: m.category for n, m in self._meta.items()}, name="category", dtype=object
        )

    @property
    def block_names(self) -> list[str]:
        """Names of all blocks, in order of declaration."""
        return list(self._block_names)

    @property
    def blocks(self) -> pd.DataFrame:
        """Boolean block-membership matrix (series x blocks)."""
        matrix = pd.DataFrame(
            False, index=pd.Index(self.columns, name="series"), columns=self.block_names
        )
        for name, m in self._meta.items():
            for b in m.blocks:
                matrix.loc[name, b] = True
        return matrix

    @property
    def is_mixed_frequency(self) -> bool:
        """True if the panel contains more than one frequency."""
        return len({m.frequency for m in self._meta.values()}) > 1

    @property
    def monthly_columns(self) -> list[str]:
        """Names of the monthly series."""
        return self.columns_with_frequency(Frequency.MONTHLY)

    @property
    def quarterly_columns(self) -> list[str]:
        """Names of the quarterly series."""
        return self.columns_with_frequency(Frequency.QUARTERLY)

    def columns_with_frequency(self, frequency: FrequencyLike) -> list[str]:
        """Names of the series with a given native frequency.

        Parameters
        ----------
        frequency : Frequency, str or int
            Frequency to select.

        Returns
        -------
        list of str
            Series names, in column order.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=3, freq="M")
        >>> mfd = MixedFrequencyData(pd.DataFrame({"a": [1.0, 2, 3]}, index=idx), "M")
        >>> mfd.columns_with_frequency("monthly")
        ['a']
        """
        freq = Frequency.from_value(frequency)
        return [n for n, m in self._meta.items() if m.frequency == freq]

    def series_by_frequency(self) -> dict[Frequency, list[str]]:
        """Group series names by native frequency, from highest to lowest frequency.

        Returns
        -------
        dict of Frequency to list of str
            Only frequencies present in the panel are included.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2020-01", periods=3, freq="M")
        >>> df = pd.DataFrame({"a": [1.0, 2, 3], "q": [np.nan, np.nan, 1.0]}, index=idx)
        >>> MixedFrequencyData(df, {"a": "M", "q": "Q"}).series_by_frequency()
        {<Frequency.MONTHLY: 'M'>: ['a'], <Frequency.QUARTERLY: 'Q'>: ['q']}
        """
        present = sorted(
            {m.frequency for m in self._meta.values()},
            key=lambda f: -f.periods_per_year,
        )
        return {f: self.columns_with_frequency(f) for f in present}

    def frequency_ratio(self, column: str) -> int:
        """Number of base periods per native period of ``column`` (3 for Q on a M grid).

        Only defined for fixed-ratio pairs; for calendar pairs (weekly series of a
        daily grid excepted, monthly or quarterly series on a weekly or daily grid) use
        :func:`~nowcastbox.core.frequency.calendar_position` on :attr:`index`.

        Parameters
        ----------
        column : str
            Series name.

        Returns
        -------
        int
            Aggregation ratio.

        Raises
        ------
        KeyError
            If ``column`` is unknown.
        ValueError
            If the number of base periods varies over the calendar.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2020-01", periods=3, freq="M")
        >>> df = pd.DataFrame({"q": [np.nan, np.nan, 1.0]}, index=idx)
        >>> MixedFrequencyData(df, "Q").frequency_ratio("q")
        3
        """
        return aggregation_ratio(self._base, self._get_meta(column).frequency)

    def _get_meta(self, column: str) -> SeriesMetadata:
        try:
            return self._meta[column]
        except KeyError:
            raise KeyError(f"Unknown series {column!r}.") from None

    # ------------------------------------------------------------------ masks
    def observation_mask(self) -> pd.DataFrame:
        """Boolean frame, True where a value is observed.

        Returns
        -------
        pandas.DataFrame
            Same shape as the data.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2020-01", periods=2, freq="M")
        >>> mfd = MixedFrequencyData(pd.DataFrame({"a": [1.0, np.nan]}, index=idx), "M")
        >>> mfd.observation_mask()["a"].tolist()
        [True, False]
        """
        return self._frame.notna()

    def slot_mask(self) -> pd.DataFrame:
        """Boolean frame, True on the cells where each series *can* be observed.

        All cells for base-frequency series; only the last base period of each native
        period for lower-frequency series.

        Returns
        -------
        pandas.DataFrame
            Same shape as the data.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2020-01", periods=3, freq="M")
        >>> df = pd.DataFrame({"q": [np.nan, np.nan, 1.0]}, index=idx)
        >>> MixedFrequencyData(df, "Q").slot_mask()["q"].tolist()
        [False, False, True]
        """
        index = self.index
        by_frequency: dict[Frequency, np.ndarray] = {self.base_frequency: np.ones(len(index), bool)}
        cols: dict[str, np.ndarray] = {}
        for name, m in self._meta.items():
            if m.frequency not in by_frequency:
                by_frequency[m.frequency] = is_period_end(index, m.frequency)
            cols[name] = by_frequency[m.frequency]
        return pd.DataFrame(cols, index=index, columns=pd.Index(self.columns))

    def missing_mask(self) -> pd.DataFrame:
        """Boolean frame, True on storage slots without a value (genuine missing data).

        Returns
        -------
        pandas.DataFrame
            ``slot_mask() & ~observation_mask()``.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2020-01", periods=6, freq="M")
        >>> df = pd.DataFrame({"q": [np.nan] * 5 + [1.0]}, index=idx)
        >>> MixedFrequencyData(df, "Q").missing_mask()["q"].tolist()
        [False, False, True, False, False, False]
        """
        return self.slot_mask() & ~self.observation_mask()

    def last_observed(self) -> pd.Series:
        """Last base period with an observation, for each series (NaT if none).

        Returns
        -------
        pandas.Series
            Indexed by series name.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2020-01", periods=3, freq="M")
        >>> mfd = MixedFrequencyData(pd.DataFrame({"a": [1.0, 2.0, np.nan]}, index=idx), "M")
        >>> str(mfd.last_observed()["a"])
        '2020-02'
        """
        out = {n: self._frame[n].last_valid_index() for n in self.columns}
        return pd.Series(out, name="last_observed", dtype=f"period[{self._base.pandas_freq}]")

    def ragged_edge_mask(self) -> pd.DataFrame:
        """Boolean frame flagging the ragged (jagged) edge of the panel.

        A cell is True if it is a storage slot of its series that lies **after** the
        series' last observation, i.e. a value not yet released. Series without
        observations are True on every slot.

        Returns
        -------
        pandas.DataFrame
            Same shape as the data.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2020-01", periods=3, freq="M")
        >>> df = pd.DataFrame({"a": [1.0, np.nan, np.nan]}, index=idx)
        >>> MixedFrequencyData(df, "M").ragged_edge_mask()["a"].tolist()
        [False, True, True]
        """
        observed = self._frame.notna().to_numpy()
        # number of observations from t to the end; zero => after the last observation
        remaining = np.flip(np.cumsum(np.flip(observed, axis=0), axis=0), axis=0)
        after_last = remaining == 0
        slots = self.slot_mask().to_numpy()
        mask = after_last & slots
        return pd.DataFrame(mask, index=self.index, columns=self._frame.columns)

    def n_observations(self) -> pd.Series:
        """Number of observed values per series.

        Returns
        -------
        pandas.Series
            Integer counts indexed by series name.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2020-01", periods=2, freq="M")
        >>> mfd = MixedFrequencyData(pd.DataFrame({"a": [1.0, np.nan]}, index=idx), "M")
        >>> int(mfd.n_observations()["a"])
        1
        """
        return self._frame.notna().sum().rename("n_observations")

    def released_share(
        self,
        period: pd.Period | str,
        *,
        by: str | Mapping[str, str | Sequence[str]] | None = None,
        weights: Mapping[str, float] | pd.Series | None = None,
        series: Sequence[str] | None = None,
        as_of: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Share of the observations of a target period that are already released.

        For every series, the *expected* observations are its storage slots inside
        ``period`` (three for a monthly series in a quarter, one for a quarterly series,
        none for an annual series outside the fourth quarter) and the *released* ones are
        the slots holding a value in this panel (a vintage). The share of a group is the
        weighted mean of the shares of its series; with the default weights (the number
        of expected observations) it is simply ``released / expected``. Base periods of
        ``period`` beyond the end of the grid count as expected and not released.

        Parameters
        ----------
        period : pandas.Period or str
            Target period, at the base frequency or a lower one (e.g. ``"2020Q2"``).
        by : {None, "series", "category", "block", "frequency"} or mapping, optional
            Grouping of the rows: one row per series (``None``/``"series"``), per
            ``SeriesMetadata.category`` (``"uncategorized"`` when unset), per factor
            block (a series counts in each of its blocks; ``"unassigned"`` when it has
            none), per frequency, or a mapping ``{series: group or [groups]}`` (series
            left out go to ``"unassigned"``).
        weights : mapping or pandas.Series, optional
            Non-negative weight of each series in the group shares, e.g. the absolute
            weights of the series in the model's nowcast; series left out weigh zero.
        series : sequence of str, optional
            Subset of series (e.g. every indicator but the target).
        as_of : Timestamp or str, optional
            Information date: apply :meth:`as_of` first (needs release delays).

        Returns
        -------
        pandas.DataFrame
            Index = series or groups plus a final ``"total"`` row (each series counted
            once); columns ``released`` and ``expected`` (number of observations),
            ``weight`` (sum of the weights of the series with expected observations) and
            ``share`` (in ``[0, 1]``; NaN when nothing is expected or the weights sum to
            zero).

        Raises
        ------
        NowcastDataError
            If ``period`` has a higher frequency than the base grid, or ``as_of`` is
            given and release delays are missing.
        ValueError
            On unknown series, an invalid grouping, negative weights or a series or group
            named ``"total"``.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2020-01", periods=6, freq="M")
        >>> df = pd.DataFrame(
        ...     {
        ...         "ip": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
        ...         "pmi": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ...         "gdp": [np.nan, np.nan, 1.0, np.nan, np.nan, np.nan],
        ...     },
        ...     index=idx,
        ... )
        >>> mfd = MixedFrequencyData(df, {"ip": "M", "pmi": "M", "gdp": "Q"})
        >>> share = mfd.released_share("2020Q2", series=["ip", "pmi"])
        >>> share["released"].tolist(), share["expected"].tolist()
        ([2, 3, 5], [3, 3, 6])
        >>> round(float(share.loc["total", "share"]), 4)
        0.8333
        """
        panel = self if as_of is None else self.as_of(as_of)
        if series is not None:
            panel = panel.select(list(series))
        released, expected = _period_counts(panel, period)
        weight = _share_weights(expected, weights)
        members = _series_groups(panel.metadata, by)
        rows = {
            group: _share_row(released[list(cols)], expected[list(cols)], weight[list(cols)])
            for group, cols in members.items()
        }
        if "total" in rows:
            raise ValueError("A series or group named 'total' clashes with the total row.")
        rows["total"] = _share_row(released, expected, weight)
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "series" if by in (None, "series") else "group"
        return frame.astype({"released": int, "expected": int, "weight": float, "share": float})

    # ------------------------------------------------------------------ conversions
    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the data as a :class:`pandas.DataFrame` on the base grid.

        Returns
        -------
        pandas.DataFrame
            Copy of the data.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=2, freq="M")
        >>> MixedFrequencyData(pd.DataFrame({"a": [1.0, 2.0]}, index=idx), "M").to_frame().shape
        (2, 1)
        """
        return self._frame.copy()

    def to_native(self, column: str, *, dropna: bool = False) -> pd.Series:
        """Return one series on its native period index (e.g. quarterly ``2020Q1``).

        Parameters
        ----------
        column : str
            Series name.
        dropna : bool, default False
            Drop missing periods.

        Returns
        -------
        pandas.Series
            Values on the storage slots, indexed by native periods.

        Raises
        ------
        KeyError
            If ``column`` is unknown.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> idx = pd.period_range("2020-01", periods=6, freq="M")
        >>> df = pd.DataFrame({"q": [np.nan, np.nan, 1.0, np.nan, np.nan, 2.0]}, index=idx)
        >>> MixedFrequencyData(df, "Q").to_native("q").to_dict()
        {Period('2020Q1', 'Q-DEC'): 1.0, Period('2020Q2', 'Q-DEC'): 2.0}
        """
        freq = self._get_meta(column).frequency
        slots = is_period_end(self.index, freq)
        s = self._frame.loc[slots, column]
        s.index = self.index[slots].asfreq(freq.pandas_freq)
        s.index.name = "period"
        return s.dropna() if dropna else s.copy()

    def metadata_frame(self) -> pd.DataFrame:
        """Metadata of all series as a data frame (one row per series).

        Returns
        -------
        pandas.DataFrame
            Columns are the :class:`SeriesMetadata` fields except ``name``.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=2, freq="M")
        >>> mfd = MixedFrequencyData(pd.DataFrame({"a": [1.0, 2.0]}, index=idx), "M")
        >>> mfd.metadata_frame().loc["a", "frequency"]
        'M'
        """
        rows = {n: m.to_dict() for n, m in self._meta.items()}
        frame = pd.DataFrame.from_dict(rows, orient="index").drop(columns="name")
        frame.index.name = "series"
        return frame

    # ------------------------------------------------------------------ derived panels
    def _derive(
        self,
        frame: pd.DataFrame,
        meta: Mapping[str, SeriesMetadata] | None = None,
    ) -> MixedFrequencyData:
        """Build a new panel from validated metadata (no empty-series warning)."""
        meta = self._meta if meta is None else meta
        frame, _ = self._prepare_frame(frame, self._base)
        cols = [str(c) for c in frame.columns]
        new_meta = {c: meta[c] for c in cols}
        self._validate_slots(frame, new_meta, self._base)
        block_names = [
            b for b in self._block_names if any(b in m.blocks for m in new_meta.values())
        ]
        out = object.__new__(MixedFrequencyData)
        out._frame = frame
        out._meta = new_meta
        out._base = self._base
        out._block_names = self._resolve_block_names(new_meta, block_names)
        return out

    def copy(self) -> MixedFrequencyData:
        """Return a copy of the panel.

        Returns
        -------
        MixedFrequencyData
            Independent copy.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=2, freq="M")
        >>> mfd = MixedFrequencyData(pd.DataFrame({"a": [1.0, 2.0]}, index=idx), "M")
        >>> mfd.copy().equals(mfd)
        True
        """
        return self._derive(self._frame.copy())

    def select(self, columns: Sequence[str]) -> MixedFrequencyData:
        """Keep only some series (in the given order).

        Parameters
        ----------
        columns : sequence of str
            Series to keep.

        Returns
        -------
        MixedFrequencyData
            Sub-panel with the same metadata.

        Raises
        ------
        KeyError
            If a series is unknown.
        NowcastDataError
            If ``columns`` is empty or has duplicates.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=2, freq="M")
        >>> df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=idx)
        >>> MixedFrequencyData(df, "M").select(["b"]).columns
        ['b']
        """
        cols = [columns] if isinstance(columns, str) else list(columns)
        if not cols:
            raise NowcastDataError("select needs at least one series.")
        if len(set(cols)) != len(cols):
            raise NowcastDataError(f"Duplicated series in {cols}.")
        for c in cols:
            self._get_meta(c)
        return self._derive(self._frame[cols].copy())

    def drop(self, columns: Sequence[str] | str) -> MixedFrequencyData:
        """Remove some series.

        Parameters
        ----------
        columns : str or sequence of str
            Series to remove.

        Returns
        -------
        MixedFrequencyData
            Panel without those series.

        Raises
        ------
        KeyError
            If a series is unknown.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=2, freq="M")
        >>> df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=idx)
        >>> MixedFrequencyData(df, "M").drop("a").columns
        ['b']
        """
        to_drop = {columns} if isinstance(columns, str) else set(columns)
        for c in to_drop:
            self._get_meta(c)
        return self.select([c for c in self.columns if c not in to_drop])

    def truncate(
        self,
        start: pd.Period | str | pd.Timestamp | None = None,
        end: pd.Period | str | pd.Timestamp | None = None,
    ) -> MixedFrequencyData:
        """Restrict the panel to a range of periods (inclusive).

        Parameters
        ----------
        start, end : pandas.Period, str or Timestamp, optional
            Bounds. Lower-frequency periods are expanded to the base grid (``"2020Q1"``
            as ``start`` means ``2020-01``; as ``end`` it means ``2020-03``).

        Returns
        -------
        MixedFrequencyData
            Truncated panel.

        Raises
        ------
        NowcastDataError
            If the resulting panel is empty.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=6, freq="M")
        >>> mfd = MixedFrequencyData(pd.DataFrame({"a": range(6)}, index=idx, dtype=float), "M")
        >>> mfd.truncate(end="2020Q1").n_periods
        3
        """
        lo = self.start if start is None else _to_period(start, self._base, "S")
        hi = self.end if end is None else _to_period(end, self._base, "E")
        keep = (self.index >= lo) & (self.index <= hi)
        if not bool(keep.any()):
            raise NowcastDataError(f"No periods between {lo} and {hi}.")
        return self._derive(self._frame.loc[keep].copy())

    def extend(self, n_periods: int) -> MixedFrequencyData:
        """Append ``n_periods`` empty base periods at the end (forecast horizon).

        Parameters
        ----------
        n_periods : int
            Number of periods to append (non-negative).

        Returns
        -------
        MixedFrequencyData
            Extended panel.

        Raises
        ------
        ValueError
            If ``n_periods`` is negative.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=2, freq="M")
        >>> mfd = MixedFrequencyData(pd.DataFrame({"a": [1.0, 2.0]}, index=idx), "M")
        >>> str(mfd.extend(3).end)
        '2020-05'
        """
        if n_periods < 0:
            raise ValueError(f"n_periods must be non-negative, got {n_periods}.")
        grid = pd.period_range(
            self.start, periods=self.n_periods + n_periods, freq=self._base.pandas_freq
        )
        frame = self._frame.reindex(grid)
        frame.index.name = self._frame.index.name
        return self._derive(frame)

    def with_data(self, data: pd.DataFrame) -> MixedFrequencyData:
        """Return a panel with new values but the same metadata.

        Used e.g. by transformations. ``data`` may hold a subset of the series and a
        different (contiguous) period range.

        Parameters
        ----------
        data : pandas.DataFrame
            New values, indexed by base-grid periods.

        Returns
        -------
        MixedFrequencyData
            New panel (validated).

        Raises
        ------
        NowcastDataError
            If ``data`` contains unknown series or violates the storage convention.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=2, freq="M")
        >>> mfd = MixedFrequencyData(pd.DataFrame({"a": [1.0, 2.0]}, index=idx), "M")
        >>> mfd.with_data(mfd.to_frame() * 2)["a"].tolist()
        [2.0, 4.0]
        """
        unknown = sorted(set(map(str, data.columns)) - set(self.columns))
        if unknown:
            raise NowcastDataError(f"with_data got unknown series {unknown}.")
        return self._derive(data.copy())

    def with_metadata(self, column: str, **changes: Any) -> MixedFrequencyData:
        """Return a panel where the metadata of one series are updated.

        Parameters
        ----------
        column : str
            Series name.
        **changes
            :class:`SeriesMetadata` fields to change (e.g. ``release_delay=30``).

        Returns
        -------
        MixedFrequencyData
            New panel (validated, so a frequency change is checked against the data).

        Raises
        ------
        KeyError
            If ``column`` is unknown.
        NowcastDataError
            If the new metadata are invalid.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=2, freq="M")
        >>> mfd = MixedFrequencyData(pd.DataFrame({"a": [1.0, 2.0]}, index=idx), "M")
        >>> int(mfd.with_metadata("a", release_delay=15).release_delays["a"])
        15
        """
        old = self._get_meta(column)
        if "name" in changes:
            raise NowcastDataError("Renaming series through with_metadata is not supported.")
        try:
            new = old.replace(**changes)
        except (TypeError, ValueError) as err:
            raise NowcastDataError(f"Invalid metadata for series {column!r}: {err}") from err
        meta = dict(self._meta)
        meta[column] = new
        return self._derive(self._frame.copy(), meta)

    def as_of(
        self,
        vintage: pd.Timestamp | str,
        release_delays: Mapping[str, int] | None = None,
    ) -> MixedFrequencyData:
        """Pseudo real-time view: mask values not yet released at date ``vintage``.

        The observation of period :math:`p` (native frequency) is released on
        ``end_date(p) + release_delay`` days and kept iff that date is on or before
        ``vintage``. Data revisions are not modelled here (see ``nowcastbox.vintages``
        for real vintages). The period grid is unchanged.

        Parameters
        ----------
        vintage : Timestamp or str
            Information date.
        release_delays : mapping of str to int, optional
            Delays (days) overriding the series metadata.

        Returns
        -------
        MixedFrequencyData
            Panel with unreleased values set to NaN.

        Raises
        ------
        NowcastDataError
            If some series have no release delay.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=3, freq="M")
        >>> mfd = MixedFrequencyData(
        ...     pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx), "M", release_delays={"a": 10}
        ... )
        >>> mfd.as_of("2020-03-10")["a"].tolist()
        [1.0, 2.0, nan]
        """
        overrides = dict(release_delays) if release_delays is not None else {}
        unknown = sorted(set(overrides) - set(self._meta))
        if unknown:
            raise NowcastDataError(f"release_delays given for unknown series {unknown}.")
        delays = {n: m.release_delay for n, m in self._meta.items()}
        delays.update({n: _coerce_delay(d) for n, d in overrides.items()})
        missing = sorted(n for n, d in delays.items() if d is None)
        if missing:
            raise NowcastDataError(f"No release delay for series {missing}; cannot build vintage.")
        date = pd.Timestamp(vintage).normalize()
        frame = self._frame.copy()
        for name, m in self._meta.items():
            delay = delays[name]
            assert delay is not None  # noqa: S101
            native = self.index.asfreq(m.frequency.pandas_freq)
            ends = native.to_timestamp(how="end").normalize()
            released = ends + pd.Timedelta(days=int(delay)) <= date
            frame.loc[~np.asarray(released), name] = np.nan
        return self._derive(frame)

    # ------------------------------------------------------------------ standardisation
    def standardization_stats(self, *, ddof: int = 1) -> StandardizationStats:
        """Compute column means and standard deviations over observed values.

        Parameters
        ----------
        ddof : int, default 1
            Delta degrees of freedom of the standard deviation.

        Returns
        -------
        StandardizationStats
            Statistics for every series.

        Raises
        ------
        NowcastDataError
            If a series has fewer than ``ddof + 1`` observations or zero variance.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=3, freq="M")
        >>> mfd = MixedFrequencyData(pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx), "M")
        >>> float(mfd.standardization_stats().std["a"])
        1.0
        """
        mean = self._frame.mean(skipna=True)
        std = self._frame.std(skipna=True, ddof=ddof)
        bad = std.index[~np.isfinite(std.to_numpy()) | (std.to_numpy() <= 0)].tolist()
        if bad:
            raise NowcastDataError(
                f"Cannot standardise series {bad}: too few observations or zero variance."
            )
        return StandardizationStats(mean=mean, std=std, ddof=ddof)

    def standardize(
        self, stats: StandardizationStats | None = None, *, ddof: int = 1
    ) -> tuple[MixedFrequencyData, StandardizationStats]:
        """Standardise every series: ``(x - mean) / std``.

        Parameters
        ----------
        stats : StandardizationStats, optional
            Statistics to use (e.g. estimated on a training sample). Computed from
            this panel when omitted.
        ddof : int, default 1
            Delta degrees of freedom (only used when ``stats`` is None).

        Returns
        -------
        standardized : MixedFrequencyData
            Standardised panel.
        stats : StandardizationStats
            Statistics needed by :meth:`destandardize`.

        Raises
        ------
        NowcastDataError
            If statistics cannot be computed or do not cover every series.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=3, freq="M")
        >>> mfd = MixedFrequencyData(pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx), "M")
        >>> z, stats = mfd.standardize()
        >>> z["a"].tolist()
        [-1.0, 0.0, 1.0]
        >>> z.destandardize(stats)["a"].tolist()
        [1.0, 2.0, 3.0]
        """
        stats = self.standardization_stats(ddof=ddof) if stats is None else stats
        return self._derive(stats.transform(self._frame)), stats

    def destandardize(self, stats: StandardizationStats) -> MixedFrequencyData:
        """Map a standardised panel back to original units.

        Parameters
        ----------
        stats : StandardizationStats
            Statistics returned by :meth:`standardize`.

        Returns
        -------
        MixedFrequencyData
            Panel in original units.

        Raises
        ------
        NowcastDataError
            If ``stats`` does not cover every series.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=2, freq="M")
        >>> mfd = MixedFrequencyData(pd.DataFrame({"a": [1.0, 3.0]}, index=idx), "M")
        >>> z, s = mfd.standardize()
        >>> z.destandardize(s).equals(mfd)
        True
        """
        return self._derive(stats.inverse_transform(self._frame))

    # ------------------------------------------------------------------ dunder / misc
    def equals(self, other: object, *, atol: float = 1e-12) -> bool:
        """Test equality of values (NaN-aware, tolerance ``atol``), index and metadata.

        Parameters
        ----------
        other : object
            Object to compare with.
        atol : float, default 1e-12
            Absolute tolerance on values.

        Returns
        -------
        bool
            True if both panels are equal.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.period_range("2020-01", periods=2, freq="M")
        >>> a = MixedFrequencyData(pd.DataFrame({"a": [1.0, 2.0]}, index=idx), "M")
        >>> a.equals(a.copy())
        True
        """
        if not isinstance(other, MixedFrequencyData):
            return False
        if (
            self._base != other._base
            or not self.index.equals(other.index)
            or self.columns != other.columns
            or self._meta != other._meta
        ):
            return False
        return bool(np.allclose(self.values, other.values, atol=atol, rtol=0.0, equal_nan=True))

    def __getitem__(self, column: str) -> pd.Series:
        if column not in self._meta:
            raise KeyError(f"Unknown series {column!r}.")
        return self._frame[column].copy()

    def __contains__(self, column: object) -> bool:
        return column in self._meta

    def __iter__(self) -> Iterator[str]:
        return iter(self.columns)

    def __len__(self) -> int:
        return self.n_periods

    def __repr__(self) -> str:
        groups = ", ".join(f"{f.label}={len(c)}" for f, c in self.series_by_frequency().items())
        return (
            f"MixedFrequencyData(n_periods={self.n_periods}, n_series={self.n_series}, "
            f"base={self._base.label}, {self.start}..{self.end}, {groups})"
        )

    def __getstate__(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)


def as_mixed_frequency_data(
    data: MixedFrequencyData | pd.DataFrame,
    frequency: FrequencySpec | None = None,
    **kwargs: Any,
) -> MixedFrequencyData:
    """Coerce user input to :class:`MixedFrequencyData`.

    This is the canonical entry point used by every estimator's ``fit``.

    Parameters
    ----------
    data : MixedFrequencyData or pandas.DataFrame
        Panel. A DataFrame is wrapped with ``MixedFrequencyData(data, frequency, **kwargs)``.
    frequency : frequency specification, optional
        Per-series frequencies (see :class:`MixedFrequencyData`). Must be omitted when
        ``data`` is already a :class:`MixedFrequencyData`.
    **kwargs
        Further keyword arguments of :class:`MixedFrequencyData` (DataFrame input only).

    Returns
    -------
    MixedFrequencyData
        The panel (returned unchanged when already a :class:`MixedFrequencyData`).

    Raises
    ------
    NowcastDataError
        If metadata are passed together with a :class:`MixedFrequencyData` or ``data``
        has an unsupported type.

    Examples
    --------
    >>> import pandas as pd
    >>> from nowcastbox.core.data import as_mixed_frequency_data
    >>> idx = pd.period_range("2020-01", periods=2, freq="M")
    >>> as_mixed_frequency_data(pd.DataFrame({"a": [1.0, 2.0]}, index=idx), "M").n_series
    1
    """
    if isinstance(data, MixedFrequencyData):
        if frequency is not None or kwargs:
            raise NowcastDataError(
                "data is already a MixedFrequencyData; frequencies/metadata cannot be "
                "passed again (use with_metadata to change them)."
            )
        return data
    if isinstance(data, pd.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance]
        return MixedFrequencyData(data, frequency, **kwargs)
    raise NowcastDataError(
        f"data must be a MixedFrequencyData or a pandas DataFrame, got {type(data).__name__}."
    )

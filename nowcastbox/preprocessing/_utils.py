"""Private helpers shared by the preprocessing modules.

Every preprocessing operation works on the **native** grid of each series: a quarterly
series stored in the third month of each quarter on a monthly base grid is first
extracted as a quarterly series (one value per quarter), processed, and written back
to its storage slots. These helpers implement that round trip for single series and
for panels (:class:`pandas.DataFrame` or
:class:`~nowcastbox.core.data.MixedFrequencyData`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd

from nowcastbox.core.data import FrequencySpec, MixedFrequencyData, as_mixed_frequency_data
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import (
    Frequency,
    FrequencyLike,
    infer_frequency,
    is_period_end,
    native_to_base,
)

__all__ = [
    "NativeView",
    "PanelLike",
    "as_panel",
    "map_native",
    "per_series",
    "return_like",
    "series_frequency",
]

PanelLike: TypeAlias = pd.DataFrame | MixedFrequencyData


def _require_period_series(series: object, what: str = "series") -> pd.Series:
    if not isinstance(series, pd.Series):
        raise NowcastDataError(f"{what} must be a pandas Series, got {type(series).__name__}.")
    if not isinstance(series.index, pd.PeriodIndex):
        raise NowcastDataError(
            f"{what} must be indexed by a pandas.PeriodIndex, got {type(series.index).__name__}."
        )
    if series.index.has_duplicates:
        raise NowcastDataError(f"{what} has duplicated periods.")
    try:
        out = series.astype(float)
    except (TypeError, ValueError) as err:
        raise NowcastDataError(f"{what} must be numeric: {err}") from err
    if bool(np.isinf(out.to_numpy()).any()):
        raise NowcastDataError(f"{what} contains infinite values.")
    return out.sort_index()


def series_frequency(series: pd.Series, frequency: FrequencyLike | None) -> Frequency:
    """Native frequency of a series (explicit, or inferred from the observation pattern).

    Parameters
    ----------
    series : pandas.Series
        Series indexed by a :class:`pandas.PeriodIndex` (native or base grid).
    frequency : Frequency, str or int, optional
        Explicit native frequency; inferred with
        :func:`~nowcastbox.core.frequency.infer_frequency` when omitted.

    Returns
    -------
    Frequency
        Native frequency of the series.

    Raises
    ------
    NowcastDataError
        If ``frequency`` is higher than the index frequency.
    """
    index_freq = Frequency.from_index(series.index)  # type: ignore[arg-type]
    if frequency is None:
        return infer_frequency(series)
    freq = Frequency.from_value(frequency)
    if freq.is_higher_than(index_freq):
        raise NowcastDataError(
            f"frequency={freq.value!r} is higher than the {index_freq.label} index of the series."
        )
    return freq


class NativeView:
    """A series seen on its native, contiguous period grid.

    Parameters
    ----------
    series : pandas.Series
        Series indexed by a :class:`pandas.PeriodIndex`, either native or on a finer
        base grid (lower-frequency values stored in the last base period).
    frequency : Frequency, str or int, optional
        Native frequency (inferred when omitted).
    name : str, default "series"
        Name used in error messages.

    Attributes
    ----------
    native : pandas.Series
        Values on the contiguous native grid.
    frequency : Frequency
        Native frequency.
    """

    def __init__(
        self, series: pd.Series, frequency: FrequencyLike | None = None, name: str = "series"
    ) -> None:
        s = _require_period_series(series, name)
        self.original_index: pd.PeriodIndex = s.index  # type: ignore[assignment]
        self.index_frequency = Frequency.from_index(self.original_index)
        self.frequency = series_frequency(s, frequency)
        self.name = series.name
        if self.frequency == self.index_frequency:
            if len(s) == 0:
                self.native = s.copy()
            else:
                full = pd.period_range(s.index[0], s.index[-1], freq=self.frequency.pandas_freq)
                self.native = s.reindex(full)
            self._slots: np.ndarray | None = None
        else:
            slots = is_period_end(self.original_index, self.frequency)
            off = s.notna().to_numpy() & ~slots
            if bool(off.any()):
                examples = self.original_index[off].astype(str).tolist()[:3]
                raise NowcastDataError(
                    f"{name} is {self.frequency.label} but has values outside the last "
                    f"{self.index_frequency.label} period of its periods, e.g. at {examples}."
                )
            native = s[slots]
            native.index = self.original_index[slots].asfreq(self.frequency.pandas_freq)
            if len(native):
                full = pd.period_range(
                    native.index[0], native.index[-1], freq=self.frequency.pandas_freq
                )
                native = native.reindex(full)
            self.native = native
            self._slots = slots
        self.native.name = self.name

    def to_original(self, values: pd.Series | np.ndarray) -> pd.Series:
        """Write native-grid values back on the original index.

        Parameters
        ----------
        values : pandas.Series or numpy.ndarray
            Values aligned with :attr:`native`.

        Returns
        -------
        pandas.Series
            Series on the original index (NaN outside the storage slots).
        """
        arr = np.asarray(values, dtype=float)
        native = pd.Series(arr, index=self.native.index, name=self.name)
        if self._slots is None:
            return native.reindex(self.original_index)
        out = pd.Series(np.nan, index=self.original_index, name=self.name)
        native_index = cast("pd.PeriodIndex", native.index)
        base_slots = native_to_base(native_index, self.index_frequency)
        out.loc[base_slots] = native.to_numpy()
        return out


def as_panel(
    data: PanelLike, frequency: FrequencySpec | None = None
) -> tuple[MixedFrequencyData, bool]:
    """Coerce a panel to :class:`MixedFrequencyData`.

    Parameters
    ----------
    data : pandas.DataFrame or MixedFrequencyData
        Panel on a base grid.
    frequency : frequency specification, optional
        Per-series frequencies for DataFrame input (inferred when omitted).

    Returns
    -------
    panel : MixedFrequencyData
        The panel.
    was_frame : bool
        Whether the input was a DataFrame (so the output should be one too).
    """
    if isinstance(data, MixedFrequencyData):
        if frequency is not None:
            raise NowcastDataError(
                "frequency cannot be passed with a MixedFrequencyData (it carries its own)."
            )
        return data, False
    return as_mixed_frequency_data(data, frequency), True


def return_like(panel: MixedFrequencyData, was_frame: bool) -> PanelLike:
    """Return ``panel`` as a DataFrame when the input was one.

    Parameters
    ----------
    panel : MixedFrequencyData
        Result panel.
    was_frame : bool
        Output type flag from :func:`as_panel`.

    Returns
    -------
    pandas.DataFrame or MixedFrequencyData
        Output in the caller's type.
    """
    return panel.to_frame() if was_frame else panel


def per_series(spec: Any, columns: list[str], what: str, *, default: Any = None) -> dict[str, Any]:
    """Expand a scalar-or-mapping specification to one value per column.

    Parameters
    ----------
    spec : scalar, mapping, pandas.Series or sequence
        ``None`` (default for all), a scalar applied to every column, a mapping /
        Series keyed by column name, or a sequence aligned with ``columns``.
    columns : list of str
        Column names.
    what : str
        Name used in error messages.
    default : object, optional
        Value for columns not covered by a mapping.

    Returns
    -------
    dict
        ``{column: value}``.

    Raises
    ------
    NowcastDataError
        On unknown column names or a misaligned sequence.
    """
    if spec is None:
        return dict.fromkeys(columns, default)
    if isinstance(spec, pd.Series):
        spec = spec.to_dict()
    if isinstance(spec, Mapping):
        unknown = sorted(set(map(str, spec)) - set(columns))
        if unknown:
            raise NowcastDataError(f"{what} given for unknown series {unknown}.")
        return {c: spec.get(c, default) for c in columns}
    if isinstance(spec, list | tuple | np.ndarray):
        if len(spec) != len(columns):
            raise NowcastDataError(
                f"{what} has {len(spec)} elements but the panel has {len(columns)} series."
            )
        return dict(zip(columns, list(spec), strict=True))
    return dict.fromkeys(columns, spec)


def map_native(
    panel: MixedFrequencyData,
    func: Callable[[str, pd.Series, Frequency], pd.Series | np.ndarray],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Apply ``func(name, native_series, frequency)`` to each series on its native grid.

    Parameters
    ----------
    panel : MixedFrequencyData
        Input panel.
    func : callable
        Receives the series name, its contiguous native series and its frequency;
        returns values aligned with the native series.
    columns : list of str, optional
        Columns to process (others are copied unchanged).

    Returns
    -------
    pandas.DataFrame
        New frame on the base grid.
    """
    frame = panel.to_frame()
    cols = panel.columns if columns is None else columns
    for name in cols:
        view = NativeView(frame[name], panel.metadata[name].frequency, name=f"Series {name!r}")
        frame[name] = view.to_original(func(name, view.native, view.frequency)).to_numpy()
    return frame

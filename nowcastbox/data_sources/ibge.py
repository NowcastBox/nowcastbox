"""IBGE — SIDRA (Sistema IBGE de Recuperação Automática).

Endpoint (HTTPS, public, no key)::

    https://apisidra.ibge.gov.br/values/t/{table}/{level}/{territories}/v/{variables}
        /p/{periods}/c{classification}/{categories}...

The answer is a JSON list whose first element is a header mapping field codes
(``"V"``, ``"D1C"``, ``"D1N"``, ...) to labels such as ``"Mês (Código)"``; the other
elements are one observation each. The period dimension is located through its label
(``Mês``, ``Trimestre``, ``Trimestre Móvel``, ``Ano``) and determines the native
frequency. Period codes are ``yyyymm`` (months, and the last month of a rolling
quarter), ``yyyyqq`` (quarters, ``qq`` in ``01..04``) or ``yyyy`` (years).

IBGE special symbols: ``-`` is an absolute zero, ``..`` (not applicable), ``...``
(not available) and ``X`` (suppressed) are missing values.
"""

from __future__ import annotations

import unicodedata
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.frequency import Frequency, FrequencyLike
from nowcastbox.data_sources._http import RequestOptions, get_json
from nowcastbox.data_sources._parsing import (
    DateLike,
    check_range,
    combine_series,
    filter_period_range,
    to_timestamp,
)
from nowcastbox.data_sources.cache import DiskCache
from nowcastbox.data_sources.exceptions import DataSourceError

__all__ = [
    "SIDRA_URL",
    "fetch_sidra",
    "fetch_sidra_raw",
    "parse_sidra_period",
    "sidra_path",
]

logger = get_logger(__name__)

SIDRA_URL = "https://apisidra.ibge.gov.br/values"
"""Base URL of the SIDRA values endpoint."""

_ZERO_TOKENS = frozenset({"-"})
_MISSING_TOKENS = frozenset({"..", "...", "X", "x", ""})

_PERIOD_LABELS: dict[str, Frequency] = {
    "mes": Frequency.MONTHLY,
    "trimestre movel": Frequency.MONTHLY,
    "trimestre": Frequency.QUARTERLY,
    "ano": Frequency.ANNUAL,
}
_UNSUPPORTED_PERIOD_LABELS = frozenset({"semestre", "semana", "dia", "periodo", "bimestre"})

SelectionLike = int | str | Sequence[int | str] | None


def _plain(text: str) -> str:
    """Lower-case ASCII version of a label, without the "(Código)" suffix."""
    text = text.replace("(Código)", "").strip()
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower().strip()


def _selection(value: SelectionLike, what: str) -> str:
    if value is None:
        return "all"
    if isinstance(value, str | int) and not isinstance(value, bool):
        text = str(value).strip()
    elif isinstance(value, Sequence):
        if len(value) == 0:
            raise ValueError(f"Empty {what} selection.")
        text = ",".join(str(v).strip() for v in value)
    else:
        raise TypeError(f"Invalid {what} selection {value!r}.")
    if not text or "/" in text:
        raise ValueError(f"Invalid {what} selection {value!r}.")
    return text


def _classification_id(key: int | str) -> str:
    text = str(key).strip().lower()
    text = text[1:] if text.startswith("c") else text
    if not text.isdigit():
        raise ValueError(f"Classification ids are integers such as 315 or 'c315', got {key!r}.")
    return f"c{int(text)}"


def sidra_path(
    table: int | str,
    variable: SelectionLike = None,
    classifications: Mapping[int | str, SelectionLike] | None = None,
    *,
    territorial_level: str = "n1",
    territories: SelectionLike = "all",
    periods: SelectionLike = "all",
) -> str:
    """Build the SIDRA query path (everything after ``/values``).

    Parameters
    ----------
    table : int or str
        SIDRA table number (e.g. ``1620``, quarterly GDP volume index).
    variable : int, str, sequence or None
        Variable code(s); ``None`` selects all variables.
    classifications : mapping, optional
        ``{classification: categories}``, e.g. ``{"c544": 129314}`` or
        ``{315: [7169, 7170]}``; categories ``None`` means ``"all"``.
    territorial_level : str, default "n1"
        Territorial level (``n1`` Brazil, ``n2`` regions, ``n3`` states, ...).
    territories : int, str or sequence, default "all"
        Territory codes within the level.
    periods : int, str or sequence, default "all"
        Period selection: ``"all"``, ``"last 12"``, ``"202001-202312"`` or codes.

    Returns
    -------
    str
        Path such as ``"/t/1620/n1/all/v/583/p/all"``.

    Raises
    ------
    ValueError
        On empty or malformed selections.

    Examples
    --------
    >>> from nowcastbox.data_sources import sidra_path
    >>> sidra_path(8888, 12606, {"c544": 129314}, periods="last 12")
    '/t/8888/n1/all/v/12606/p/last 12/c544/129314'
    """
    table_text = str(table).strip()
    if not table_text.isdigit():
        raise ValueError(f"SIDRA table numbers are integers, got {table!r}.")
    level = str(territorial_level).strip().lower()
    if not (level.startswith("n") and level[1:].isdigit()):
        raise ValueError(f"territorial_level must look like 'n1', 'n3', got {territorial_level!r}.")
    parts = [
        f"/t/{int(table_text)}",
        f"/{level}/{_selection(territories, 'territory')}",
        f"/v/{_selection(variable, 'variable')}",
        f"/p/{_selection(periods, 'period')}",
    ]
    for key, categories in (classifications or {}).items():
        parts.append(f"/{_classification_id(key)}/{_selection(categories, 'category')}")
    return "".join(parts)


def parse_sidra_period(code: str, frequency: FrequencyLike) -> pd.Period:
    """Convert a SIDRA period code to a :class:`pandas.Period`.

    Parameters
    ----------
    code : str
        ``yyyymm`` (monthly), ``yyyyqq`` (quarterly) or ``yyyy`` (annual).
    frequency : Frequency, str or int
        Frequency of the period dimension.

    Returns
    -------
    pandas.Period
        The period.

    Raises
    ------
    NowcastDataError
        If the code does not match the frequency.

    Examples
    --------
    >>> from nowcastbox.data_sources import parse_sidra_period
    >>> parse_sidra_period("202302", "Q")
    Period('2023Q2', 'Q-DEC')
    >>> parse_sidra_period("202311", "M")
    Period('2023-11', 'M')
    """
    freq = Frequency.from_value(frequency)
    text = str(code).strip()
    if freq == Frequency.ANNUAL and len(text) == 4 and text.isdigit():
        return pd.Period(year=int(text), freq="Y")
    if len(text) == 6 and text.isdigit():
        year, sub = int(text[:4]), int(text[4:])
        if freq == Frequency.MONTHLY and 1 <= sub <= 12:
            return pd.Period(year=year, month=sub, freq="M")
        if freq == Frequency.QUARTERLY and 1 <= sub <= 4:
            return pd.Period(year=year, quarter=sub, freq="Q")
    raise NowcastDataError(f"Invalid SIDRA period code {code!r} for {freq.label} data.")


def _sidra_value(raw: object, dash_as: float = 0.0) -> float:
    if raw is None:
        return float("nan")
    text = str(raw).strip()
    if text in _ZERO_TOKENS:
        return float(dash_as)
    if text in _MISSING_TOKENS:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        warnings.warn(
            f"Unrecognized SIDRA value {text!r} set to NaN.", DataQualityWarning, stacklevel=4
        )
        return float("nan")


def _period_field(header: Mapping[str, str]) -> tuple[str, Frequency]:
    """Locate the ``DkC`` field holding the period and its frequency."""
    for field_code, label in header.items():
        if not (field_code.startswith("D") and field_code.endswith("C")):
            continue
        plain = _plain(str(label))
        if plain in _PERIOD_LABELS:
            return field_code, _PERIOD_LABELS[plain]
        if plain in _UNSUPPORTED_PERIOD_LABELS:
            raise NowcastDataError(f"SIDRA period dimension {label!r} is not supported.")
    raise DataSourceError(f"No period dimension found in the SIDRA header {dict(header)!r}.")


def _payload_rows(payload: Any) -> tuple[dict[str, str], list[dict[str, Any]]]:
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise DataSourceError("Unexpected SIDRA answer (expected a JSON list with a header row).")
    header = {str(k): str(v) for k, v in payload[0].items()}
    if "V" not in header:
        raise DataSourceError("The SIDRA header has no 'V' (value) field.")
    return header, payload[1:]


def fetch_sidra_raw(
    table: int | str,
    variable: SelectionLike = None,
    classifications: Mapping[int | str, SelectionLike] | None = None,
    *,
    territorial_level: str = "n1",
    territories: SelectionLike = "all",
    periods: SelectionLike = "all",
    native_frequency: FrequencyLike | None = None,
    dash_as: float = 0.0,
    cache: DiskCache | bool | None = None,
    session: Any = None,
    timeout: float = 60.0,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
) -> pd.DataFrame:
    """Download a SIDRA query as a long (tidy) table.

    Parameters
    ----------
    table, variable, classifications, territorial_level, territories, periods
        Query, see :func:`sidra_path`.
    native_frequency : Frequency, str or int, optional
        Overrides the frequency deduced from the period dimension label.
    dash_as : float, default 0.0
        Value of the SIDRA token ``-`` ("zero, not resulting from rounding"). SIDRA
        also prints ``-`` where an index series is not available (e.g. before its
        base period); pass ``float("nan")`` for such series so that the missing
        values do not become zeros (which break log transformations).
    cache : DiskCache, bool or None, default None
        Response cache (``None``: default cache; ``False``: disabled).
    session : requests.Session, optional
        HTTP session to reuse.
    timeout : float, default 60.0
        Timeout of each request, in seconds.
    max_retries : int, default 3
        Retries for connection errors, timeouts and HTTP 429/5xx.
    backoff_factor : float, default 1.0
        Exponential back-off factor (seconds).

    Returns
    -------
    pandas.DataFrame
        One row per observation with columns ``period`` (Period), ``value``
        (float; ``-`` → ``dash_as``, ``..``/``...``/``X`` → NaN) and the original fields
        renamed by their header labels (e.g. ``"Variável (Código)"``).
        ``attrs["frequency"]`` holds the native :class:`Frequency` and
        ``attrs["dimensions"]`` the ``(code_label, name_label)`` pairs of the
        non-period dimensions.

    Raises
    ------
    DataSourceError
        Network failure, invalid query (HTTP 400 with the SIDRA message) or
        unexpected answer.
    NowcastDataError
        Unsupported period dimension or invalid period codes.

    Examples
    --------
    >>> from nowcastbox.data_sources import fetch_sidra_raw
    >>> raw = fetch_sidra_raw(1620, 583, {"c11255": 90707}, periods="last 4")  # doctest: +SKIP
    """
    path = sidra_path(
        table,
        variable,
        classifications,
        territorial_level=territorial_level,
        territories=territories,
        periods=periods,
    )
    options = RequestOptions(
        cache=cache,
        session=session,
        timeout=timeout,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
    )
    header, rows = _payload_rows(get_json(SIDRA_URL + path, None, options=options))
    period_code, freq = _period_field(header)
    if native_frequency is not None:
        freq = Frequency.from_value(native_frequency)
    dims = [
        (header[f], header.get(f[:-1] + "N", header[f]))
        for f in header
        if f.startswith("D") and f.endswith("C") and f != period_code
    ]
    frame = pd.DataFrame(rows, columns=list(header), dtype="object")
    frame.insert(0, "period", [parse_sidra_period(c, freq) for c in frame[period_code]])
    frame.insert(1, "value", [_sidra_value(v, dash_as) for v in frame["V"]])
    frame = frame.rename(columns=header)
    frame.attrs["frequency"] = freq
    frame.attrs["dimensions"] = dims
    frame.attrs["path"] = path
    return frame


def _column_labels(raw: pd.DataFrame, name: str | None, default: str) -> pd.Series:
    """One label per row identifying its (non-period) dimension combination."""
    varying = [(c, n) for c, n in raw.attrs["dimensions"] if len(set(raw[c])) > 1]
    if not varying:
        return pd.Series(name or default, index=raw.index)
    labels = raw[[n for _, n in varying]].astype(str).agg(" | ".join, axis=1)
    codes = raw[[c for c, _ in varying]].astype(str).agg("|".join, axis=1)
    pairs = pd.DataFrame({"label": labels, "code": codes}).drop_duplicates()
    if pairs["label"].duplicated().any():  # different codes sharing a description
        labels = labels + " [" + codes + "]"
    return labels if name is None else name + ": " + labels


def fetch_sidra(
    table: int | str,
    variable: SelectionLike = None,
    classifications: Mapping[int | str, SelectionLike] | None = None,
    *,
    territorial_level: str = "n1",
    territories: SelectionLike = "all",
    periods: SelectionLike = "all",
    start: DateLike | None = None,
    end: DateLike | None = None,
    name: str | None = None,
    native_frequency: FrequencyLike | None = None,
    base_frequency: FrequencyLike | None = None,
    dash_as: float = 0.0,
    cache: DiskCache | bool | None = None,
    session: Any = None,
    timeout: float = 60.0,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
) -> pd.DataFrame:
    """Download a SIDRA table as a wide, period-indexed DataFrame.

    Parameters
    ----------
    table : int or str
        SIDRA table (e.g. ``1620`` quarterly GDP index, ``8888`` PIM-PF).
    variable : int, str, sequence or None
        Variable code(s); ``None`` selects all variables.
    classifications : mapping, optional
        ``{classification: categories}``, e.g. ``{"c544": 129314}``.
    territorial_level : str, default "n1"
        Territorial level (``n1`` Brazil, ``n3`` states, ...).
    territories : int, str or sequence, default "all"
        Territory codes within the level.
    periods : int, str or sequence, default "all"
        Period selection sent to SIDRA (``"all"``, ``"last 12"``, ``"202001-202312"``).
    start, end : date-like, optional
        Inclusive bounds applied after download.
    name : str, optional
        Column name when the query yields a single series (default
        ``"sidra_<table>"`` or ``"sidra_<table>_v<variable>"``); with several series
        it prefixes the generated labels.
    native_frequency : Frequency, str or int, optional
        Overrides the frequency deduced from the period dimension.
    base_frequency : Frequency, str or int, optional
        Output grid (e.g. ``"M"`` stores quarterly values in the third month).
    dash_as : float, default 0.0
        Value of the SIDRA token ``-`` ("zero, not resulting from rounding"). SIDRA
        also prints ``-`` where an index series is not available (e.g. before its
        base period); pass ``float("nan")`` for such series so that the missing
        values do not become zeros (which break log transformations).
    cache : DiskCache, bool or None, default None
        Response cache (``None``: default cache; ``False``: disabled).
    session : requests.Session, optional
        HTTP session to reuse.
    timeout : float, default 60.0
        Timeout of each request, in seconds.
    max_retries : int, default 3
        Retries for connection errors, timeouts and HTTP 429/5xx.
    backoff_factor : float, default 1.0
        Exponential back-off factor (seconds).

    Returns
    -------
    pandas.DataFrame
        ``float64`` frame indexed by a PeriodIndex; one column per combination of the
        dimensions that vary in the answer (labelled ``"<name> | <name>"``).

    Raises
    ------
    DataSourceError
        Network failure, invalid query or unexpected answer.
    NowcastDataError
        Empty answer, unsupported period dimension or duplicated observations.

    Examples
    --------
    >>> from nowcastbox.data_sources import fetch_sidra
    >>> pim = fetch_sidra(8888, 12606, {"c544": 129314}, name="pim")  # doctest: +SKIP
    >>> pim.columns.tolist()  # doctest: +SKIP
    ['pim']
    """
    raw = fetch_sidra_raw(
        table,
        variable,
        classifications,
        territorial_level=territorial_level,
        territories=territories,
        periods=periods,
        native_frequency=native_frequency,
        dash_as=dash_as,
        cache=cache,
        session=session,
        timeout=timeout,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
    )
    if raw.empty:
        raise NowcastDataError(f"SIDRA query {raw.attrs['path']} returned no observations.")
    first, last = to_timestamp(start), to_timestamp(end, end=True)
    check_range(first, last)
    default = f"sidra_{int(str(table).strip())}"
    if isinstance(variable, int | str) and str(variable).strip().isdigit():
        default += f"_v{int(str(variable).strip())}"
    raw = raw.assign(_column=_column_labels(raw, name, default))
    if raw.duplicated(["_column", "period"]).any():
        raise NowcastDataError("SIDRA answer has duplicated (series, period) observations.")
    freq = raw.attrs["frequency"]
    series = {}
    for label, group in raw.groupby("_column", sort=False):
        index = pd.PeriodIndex(list(group["period"]), freq=freq.pandas_freq)
        values = pd.Series(group["value"].to_numpy(dtype="float64"), index=index, name=str(label))
        series[str(label)] = filter_period_range(values.sort_index(), first, last)
    return combine_series(series, base_frequency=base_frequency)

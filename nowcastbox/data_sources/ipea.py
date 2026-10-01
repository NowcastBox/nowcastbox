"""IPEADATA — OData v4 API of the Instituto de Pesquisa Econômica Aplicada.

Endpoints (public, no key)::

    http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{code}')
    http://www.ipeadata.gov.br/api/odata4/Metadados('{code}')

Values come as ``{"value": [{"SERCODIGO": ..., "VALDATA": "2020-01-01T00:00:00-03:00",
"VALVALOR": 1.2, "NIVNOME": "", "TERCODIGO": ""}, ...]}``. ``VALDATA`` marks the
start of the reference period; only its calendar date is used (the UTC offset is
ignored, so that midnight dates never shift to the previous day). Regional series
carry one row per territory (``NIVNOME`` level, ``TERCODIGO`` code) and must be
filtered with ``territorial_level``/``territory_code``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import FrequencyLike
from nowcastbox.data_sources._http import RequestOptions, get_json
from nowcastbox.data_sources._parsing import (
    DateLike,
    build_series,
    check_range,
    combine_series,
    filter_period_range,
    normalize_codes,
    resolve_native_frequency,
    to_timestamp,
)
from nowcastbox.data_sources.cache import DiskCache
from nowcastbox.data_sources.exceptions import DataSourceError

__all__ = ["IPEADATA_URL", "fetch_ipeadata", "fetch_ipeadata_metadata", "fetch_ipeadata_series"]

logger = get_logger(__name__)

IPEADATA_URL = "http://www.ipeadata.gov.br/api/odata4"
"""Base URL of the IPEADATA OData v4 service."""

_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _ipea_code(code: object) -> str:
    text = str(code).strip()
    if not _CODE_PATTERN.match(text):
        raise ValueError(f"Invalid IPEADATA series code {code!r} (letters, digits, '_', '.', '-').")
    return text


def _odata_rows(payload: Any, what: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise DataSourceError(
            f"{what}: unexpected IPEADATA answer of type {type(payload).__name__}."
        )
    if "error" in payload:
        raise DataSourceError(f"{what}: IPEADATA error: {payload['error']}")
    rows = payload.get("value")
    if not isinstance(rows, list):
        raise DataSourceError(f"{what}: IPEADATA answer has no 'value' list.")
    return rows


def fetch_ipeadata_metadata(
    code: str,
    *,
    base_url: str = IPEADATA_URL,
    cache: DiskCache | bool | None = None,
    session: Any = None,
    timeout: float = 60.0,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
) -> dict[str, Any]:
    """Metadata of an IPEADATA series (name, unit, periodicity, source, ...).

    Parameters
    ----------
    code : str
        IPEADATA series code (``SERCODIGO``), e.g. ``"SCN104_PIBPM104"``.
    base_url : str, default IPEADATA_URL
        OData service root.
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
    dict
        Metadata fields as returned by the API (e.g. ``SERNOME``, ``PERNOME``,
        ``UNINOME``, ``FNTNOME``).

    Raises
    ------
    DataSourceError
        Network failure, unknown code or unexpected answer.

    Examples
    --------
    >>> from nowcastbox.data_sources import fetch_ipeadata_metadata
    >>> fetch_ipeadata_metadata("SCN104_PIBPM104")["PERNOME"]  # doctest: +SKIP
    'Trimestral'
    """
    sercode = _ipea_code(code)
    options = RequestOptions(
        cache=cache,
        session=session,
        timeout=timeout,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
    )
    rows = _odata_rows(
        get_json(f"{base_url}/Metadados('{sercode}')", None, options=options), sercode
    )
    if not rows:
        raise DataSourceError(f"IPEADATA series {sercode!r} not found.")
    return dict(rows[0])


def _select_territory(
    rows: list[dict[str, Any]], code: str, territorial_level: str | None, territory_code: str | None
) -> list[dict[str, Any]]:
    if territorial_level is not None:
        rows = [r for r in rows if str(r.get("NIVNOME") or "") == territorial_level]
    if territory_code is not None:
        rows = [r for r in rows if str(r.get("TERCODIGO") or "") == str(territory_code)]
    territories = sorted(
        {(str(r.get("NIVNOME") or ""), str(r.get("TERCODIGO") or "")) for r in rows}
    )
    if len(territories) > 1:
        shown = ", ".join(f"{lvl or '-'}/{ter or '-'}" for lvl, ter in territories[:8])
        raise NowcastDataError(
            f"IPEADATA series {code!r} has {len(territories)} territories ({shown}, ...); "
            "select one with territorial_level and/or territory_code."
        )
    return rows


def fetch_ipeadata_series(
    code: str,
    start: DateLike | None = None,
    end: DateLike | None = None,
    *,
    name: str | None = None,
    territorial_level: str | None = None,
    territory_code: str | None = None,
    native_frequency: FrequencyLike | None = None,
    base_url: str = IPEADATA_URL,
    cache: DiskCache | bool | None = None,
    session: Any = None,
    timeout: float = 60.0,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
) -> pd.Series:
    """Download one IPEADATA series with a native-frequency PeriodIndex.

    Parameters
    ----------
    code : str
        IPEADATA series code (``SERCODIGO``).
    start, end : date-like, optional
        Inclusive bounds applied after download.
    name : str, optional
        Series name (default: the code).
    territorial_level : str, optional
        Keep only rows with this ``NIVNOME`` (e.g. ``"Brasil"``, ``"Estados"``).
    territory_code : str, optional
        Keep only rows with this ``TERCODIGO`` (e.g. ``"35"`` for São Paulo).
    native_frequency : Frequency, str or int, optional
        Native frequency; inferred from the dates when omitted.
    base_url : str, default IPEADATA_URL
        OData service root.
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
    pandas.Series
        ``float64`` series.

    Raises
    ------
    DataSourceError
        Network failure or unexpected answer.
    NowcastDataError
        Unknown/empty series or several territories without a filter.

    Examples
    --------
    >>> from nowcastbox.data_sources import fetch_ipeadata_series
    >>> selic = fetch_ipeadata_series("BM12_TJOVER12", start="2020-01")  # doctest: +SKIP
    """
    sercode = _ipea_code(code)
    first, last = to_timestamp(start), to_timestamp(end, end=True)
    check_range(first, last)
    options = RequestOptions(
        cache=cache,
        session=session,
        timeout=timeout,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
    )
    payload = get_json(f"{base_url}/ValoresSerie(SERCODIGO='{sercode}')", None, options=options)
    rows = _odata_rows(payload, sercode)
    rows = _select_territory(rows, sercode, territorial_level, territory_code)
    if not rows:
        raise NowcastDataError(
            f"IPEADATA series {sercode!r} returned no observations (unknown code or empty territory filter)."
        )
    try:
        dates = pd.to_datetime([str(r["VALDATA"])[:10] for r in rows], format="%Y-%m-%d")
        values = [r.get("VALVALOR") for r in rows]
    except (KeyError, ValueError) as exc:
        raise DataSourceError(f"IPEADATA series {sercode!r}: malformed records ({exc}).") from None
    series = build_series(dates, values, name or sercode, native_frequency=native_frequency)
    return filter_period_range(series, first, last)


def fetch_ipeadata(
    codes: str | Sequence[str] | Mapping[str, str],
    start: DateLike | None = None,
    end: DateLike | None = None,
    *,
    territorial_level: str | None = None,
    territory_code: str | None = None,
    native_frequency: FrequencyLike | Mapping[str, FrequencyLike] | None = None,
    base_frequency: FrequencyLike | None = None,
    base_url: str = IPEADATA_URL,
    cache: DiskCache | bool | None = None,
    session: Any = None,
    timeout: float = 60.0,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
) -> pd.DataFrame:
    """Download one or more IPEADATA series.

    Parameters
    ----------
    codes : str, sequence or mapping
        Series code(s); a mapping ``{name: code}`` names the columns (default name:
        the code).
    start, end : date-like, optional
        Inclusive bounds applied after download.
    territorial_level, territory_code : str, optional
        Territory filters applied to every series (see :func:`fetch_ipeadata_series`).
    native_frequency : Frequency, str, int or mapping, optional
        Native frequency (scalar or per series name); inferred when omitted.
    base_frequency : Frequency, str or int, optional
        Output grid; required when the series have different native frequencies.
    base_url : str, default IPEADATA_URL
        OData service root.
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
        ``float64`` frame indexed by a PeriodIndex, one column per series.

    Raises
    ------
    DataSourceError
        Network failure or unexpected answer.
    NowcastDataError
        Empty series, ambiguous territories, or mixed frequencies without
        ``base_frequency``.

    Examples
    --------
    >>> from nowcastbox.data_sources import fetch_ipeadata
    >>> df = fetch_ipeadata({"selic": "BM12_TJOVER12"}, start="2015-01")  # doctest: +SKIP
    """
    names = normalize_codes(codes, default_name=_ipea_code)
    series = {
        name: fetch_ipeadata_series(
            code,
            start,
            end,
            name=name,
            territorial_level=territorial_level,
            territory_code=territory_code,
            native_frequency=resolve_native_frequency(native_frequency, name),
            base_url=base_url,
            cache=cache,
            session=session,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )
        for name, code in names.items()
    }
    return combine_series(series, base_frequency=base_frequency)

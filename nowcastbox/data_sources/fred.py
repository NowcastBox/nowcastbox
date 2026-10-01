"""FRED / ALFRED — Federal Reserve Bank of St. Louis (optional connector).

Endpoint (HTTPS, free API key required)::

    https://api.stlouisfed.org/fred/series/observations
        ?series_id=GDPC1&api_key=...&file_type=json
        &observation_start=YYYY-MM-DD&observation_end=YYYY-MM-DD
        &realtime_start=YYYY-MM-DD&realtime_end=YYYY-MM-DD

The key is read from the ``api_key`` argument or the ``FRED_API_KEY`` environment
variable. It is never stored in the cache, in logs or in error messages. Passing
``vintage_date`` sets ``realtime_start = realtime_end = vintage_date``, i.e. the data
as they were known on that day (ALFRED). Missing values are coded ``"."``.
"""

from __future__ import annotations

import os
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
    normalize_codes,
    resolve_native_frequency,
    to_timestamp,
)
from nowcastbox.data_sources.cache import DiskCache
from nowcastbox.data_sources.exceptions import DataSourceError, MissingAPIKeyError

__all__ = ["FRED_API_KEY_ENV", "FRED_URL", "fetch_fred", "fetch_fred_series", "get_fred_api_key"]

logger = get_logger(__name__)

FRED_URL = "https://api.stlouisfed.org/fred"
"""Root of the FRED web service."""

FRED_API_KEY_ENV = "FRED_API_KEY"
"""Environment variable read when no ``api_key`` is passed."""

_SERIES_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+$")
_ISO = "%Y-%m-%d"


def get_fred_api_key(api_key: str | None = None) -> str:
    """Return the FRED API key (argument first, then ``FRED_API_KEY``).

    Parameters
    ----------
    api_key : str, optional
        Explicit key.

    Returns
    -------
    str
        The key.

    Raises
    ------
    MissingAPIKeyError
        If no key is available.

    Examples
    --------
    >>> from nowcastbox.data_sources import get_fred_api_key
    >>> get_fred_api_key("abcdef0123456789abcdef0123456789")
    'abcdef0123456789abcdef0123456789'
    """
    key = api_key if api_key is not None else os.environ.get(FRED_API_KEY_ENV)
    if key is None or not str(key).strip():
        raise MissingAPIKeyError(
            "FRED requires a free API key (https://fred.stlouisfed.org/docs/api/api_key.html); "
            f"pass api_key=... or set the {FRED_API_KEY_ENV} environment variable."
        )
    return str(key).strip()


def _series_id(series_id: object) -> str:
    text = str(series_id).strip()
    if not _SERIES_PATTERN.match(text):
        raise ValueError(f"Invalid FRED series id {series_id!r}.")
    return text.upper()


def fetch_fred_series(
    series_id: str,
    start: DateLike | None = None,
    end: DateLike | None = None,
    *,
    api_key: str | None = None,
    vintage_date: DateLike | None = None,
    name: str | None = None,
    native_frequency: FrequencyLike | None = None,
    cache: DiskCache | bool | None = None,
    session: Any = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
) -> pd.Series:
    """Download one FRED series (optionally as of an ALFRED vintage).

    Parameters
    ----------
    series_id : str
        FRED series id (e.g. ``"GDPC1"``, ``"INDPRO"``).
    start, end : date-like, optional
        Inclusive observation bounds.
    api_key : str, optional
        FRED API key (default: ``FRED_API_KEY`` environment variable).
    vintage_date : date-like, optional
        Real-time date: return the data as published on that day (ALFRED).
    name : str, optional
        Series name (default: the series id).
    native_frequency : Frequency, str or int, optional
        Native frequency; inferred from the dates when omitted.
    cache : DiskCache, bool or None, default None
        Response cache (``None``: default cache; ``False``: disabled).
    session : requests.Session, optional
        HTTP session to reuse.
    timeout : float, default 30.0
        Timeout of each request, in seconds.
    max_retries : int, default 3
        Retries for connection errors, timeouts and HTTP 429/5xx.
    backoff_factor : float, default 1.0
        Exponential back-off factor (seconds).

    Returns
    -------
    pandas.Series
        ``float64`` series indexed by a native-frequency PeriodIndex.

    Raises
    ------
    MissingAPIKeyError
        No API key available.
    DataSourceError
        Network failure, invalid series id or unexpected answer.
    NowcastDataError
        If the series has no observation in the requested range.

    Examples
    --------
    >>> from nowcastbox.data_sources import fetch_fred_series
    >>> gdp = fetch_fred_series("GDPC1", start="2000-01-01")  # doctest: +SKIP
    >>> gdp.index.freqstr  # doctest: +SKIP
    'Q-DEC'
    """
    sid = _series_id(series_id)
    key = get_fred_api_key(api_key)
    first, last = to_timestamp(start), to_timestamp(end, end=True)
    check_range(first, last)
    params: dict[str, str] = {"series_id": sid, "api_key": key, "file_type": "json"}
    if first is not None:
        params["observation_start"] = first.strftime(_ISO)
    if last is not None:
        params["observation_end"] = last.strftime(_ISO)
    if vintage_date is not None:
        vintage = to_timestamp(vintage_date).strftime(_ISO)  # type: ignore[union-attr]
        params["realtime_start"] = params["realtime_end"] = vintage
    options = RequestOptions(
        cache=cache,
        session=session,
        timeout=timeout,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        secret_params=("api_key",),
    )
    payload = get_json(f"{FRED_URL}/series/observations", params, options=options)
    if not isinstance(payload, dict) or "error_message" in payload:
        message = payload.get("error_message") if isinstance(payload, dict) else payload
        raise DataSourceError(f"FRED series {sid}: API error: {message}")
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise DataSourceError(f"FRED series {sid}: answer has no 'observations' list.")
    if not observations:
        raise NowcastDataError(f"FRED series {sid} has no observations in the requested range.")
    try:
        dates = pd.to_datetime([o["date"] for o in observations], format=_ISO)
        values = [o["value"] for o in observations]
    except (KeyError, TypeError, ValueError) as exc:
        raise DataSourceError(f"FRED series {sid}: malformed observations ({exc}).") from None
    return build_series(
        dates, values, name or sid, native_frequency=native_frequency, missing_tokens=(".", "")
    )


def fetch_fred(
    series: str | Sequence[str] | Mapping[str, str],
    start: DateLike | None = None,
    end: DateLike | None = None,
    *,
    api_key: str | None = None,
    vintage_date: DateLike | None = None,
    native_frequency: FrequencyLike | Mapping[str, FrequencyLike] | None = None,
    base_frequency: FrequencyLike | None = None,
    cache: DiskCache | bool | None = None,
    session: Any = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
) -> pd.DataFrame:
    """Download one or more FRED series.

    Parameters
    ----------
    series : str, sequence or mapping
        FRED id(s); a mapping ``{name: id}`` names the columns (default: the id).
    start, end : date-like, optional
        Inclusive observation bounds.
    api_key : str, optional
        FRED API key (default: ``FRED_API_KEY`` environment variable).
    vintage_date : date-like, optional
        ALFRED real-time date applied to every series.
    native_frequency : Frequency, str, int or mapping, optional
        Native frequency (scalar or per series name); inferred when omitted.
    base_frequency : Frequency, str or int, optional
        Output grid; required when the series have different native frequencies.
    cache : DiskCache, bool or None, default None
        Response cache (``None``: default cache; ``False``: disabled).
    session : requests.Session, optional
        HTTP session to reuse.
    timeout : float, default 30.0
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
    MissingAPIKeyError
        No API key available.
    DataSourceError
        Network failure or unexpected answer.
    NowcastDataError
        Empty series or mixed frequencies without ``base_frequency``.

    Examples
    --------
    >>> from nowcastbox.data_sources import fetch_fred
    >>> df = fetch_fred({"gdp": "GDPC1", "ip": "INDPRO"}, base_frequency="M")  # doctest: +SKIP
    """
    key = get_fred_api_key(api_key)
    names = normalize_codes(series, default_name=_series_id)
    frames = {
        name: fetch_fred_series(
            code,
            start,
            end,
            api_key=key,
            vintage_date=vintage_date,
            name=name,
            native_frequency=resolve_native_frequency(native_frequency, name),
            cache=cache,
            session=session,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )
        for name, code in names.items()
    }
    return combine_series(frames, base_frequency=base_frequency)

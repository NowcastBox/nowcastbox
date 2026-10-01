"""Banco Central do Brasil — SGS (Sistema Gerenciador de Séries Temporais).

Endpoint (HTTPS, public, no key)::

    https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados
        ?formato=json&dataInicial=dd/mm/aaaa&dataFinal=dd/mm/aaaa

The API answers ``[{"data": "dd/mm/aaaa", "valor": "1.23"}, ...]``. Daily series can
only be queried in windows of at most ten years (HTTP 406 otherwise), and a window
without observations answers HTTP 404 ("Value(s) not found"). :func:`fetch_sgs`
therefore splits every dated request in windows of at most ten years, treats empty
windows as empty, and — when no ``start`` is given — first tries an undated request
(fine for monthly/quarterly/annual series) and falls back to walking back in ten-year
windows from ``end`` until a window comes back empty.

SGS dates mark the *start* of the reference period (``01/04/2020`` is 2020Q2 for a
quarterly series); the native frequency is inferred from the dates.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
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
from nowcastbox.data_sources.exceptions import DataSourceError, HTTPStatusError

__all__ = ["SGS_URL", "SGS_WINDOW_YEARS", "fetch_sgs", "fetch_sgs_series", "sgs_windows"]

logger = get_logger(__name__)

SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
"""URL template of the SGS data endpoint."""

SGS_WINDOW_YEARS = 10
"""Maximum length, in years, of one SGS query for daily series."""

_MAX_BACKWARD_WINDOWS = 15
_DATE_FORMAT = "%d/%m/%Y"


def sgs_windows(
    start: DateLike, end: DateLike, *, years: int = SGS_WINDOW_YEARS
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split ``[start, end]`` into consecutive windows of at most ``years`` years.

    Window ``k`` covers ``[s_k, s_k + years - 1 day]`` (inclusive), so windows never
    overlap and their union is exactly ``[start, end]``.

    Parameters
    ----------
    start, end : date-like
        Inclusive bounds.
    years : int, default 10
        Maximum window length in years.

    Returns
    -------
    list of (Timestamp, Timestamp)
        Inclusive ``(first_day, last_day)`` pairs in chronological order.

    Raises
    ------
    ValueError
        If ``start > end`` or ``years < 1``.

    Examples
    --------
    >>> from nowcastbox.data_sources import sgs_windows
    >>> [
    ...     (a.date().isoformat(), b.date().isoformat())
    ...     for a, b in sgs_windows("2000-01-01", "2021-06-30")
    ... ]
    [('2000-01-01', '2009-12-31'), ('2010-01-01', '2019-12-31'), ('2020-01-01', '2021-06-30')]
    """
    if isinstance(years, bool) or not isinstance(years, int) or years < 1:
        raise ValueError(f"years must be a positive int, got {years!r}.")
    first = to_timestamp(start)
    last = to_timestamp(end, end=True)
    if first is None or last is None:
        raise ValueError("sgs_windows requires both start and end.")
    check_range(first, last)
    windows = []
    current = first
    while current <= last:
        stop = min(current + pd.DateOffset(years=years) - pd.Timedelta(days=1), last)
        windows.append((current, stop))
        current = stop + pd.Timedelta(days=1)
    return windows


def _sgs_code(code: Hashable) -> str:
    text = str(code).strip()
    if not text.isdigit():
        raise ValueError(f"SGS codes are positive integers, got {code!r}.")
    return str(int(text))


def _parse_records(payload: Any, code: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        message = payload.get("error") or payload.get("erro") or payload
        raise DataSourceError(f"SGS series {code}: API error: {message}")
    if not isinstance(payload, list):
        raise DataSourceError(
            f"SGS series {code}: unexpected response of type {type(payload).__name__}."
        )
    for record in payload:
        if not isinstance(record, dict) or "data" not in record or "valor" not in record:
            raise DataSourceError(f"SGS series {code}: malformed record {record!r}.")
    return payload


def _request(
    code: str,
    first: pd.Timestamp | None,
    last: pd.Timestamp | None,
    options: RequestOptions,
) -> list[dict[str, Any]]:
    """One SGS query; an HTTP 404 ("Value(s) not found") is an empty window."""
    params = {"formato": "json"}
    if first is not None:
        params["dataInicial"] = first.strftime(_DATE_FORMAT)
    if last is not None:
        params["dataFinal"] = last.strftime(_DATE_FORMAT)
    try:
        payload = get_json(SGS_URL.format(code=code), params, options=options)
    except HTTPStatusError as err:
        if err.status_code == 404 and first is not None:
            return []
        raise
    return _parse_records(payload, code)


def _forward(
    code: str, first: pd.Timestamp, last: pd.Timestamp, options: RequestOptions
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for lo, hi in sgs_windows(first, last):
        records.extend(_request(code, lo, hi, options))
    return records


def _backward(code: str, last: pd.Timestamp, options: RequestOptions) -> list[dict[str, Any]]:
    """Walk back from ``last`` in ten-year windows until a window is empty."""
    chunks: list[list[dict[str, Any]]] = []
    hi = last
    for _ in range(_MAX_BACKWARD_WINDOWS):
        lo = hi - pd.DateOffset(years=SGS_WINDOW_YEARS) + pd.Timedelta(days=1)
        chunk = _request(code, lo, hi, options)
        if not chunk:
            break
        chunks.append(chunk)
        hi = lo - pd.Timedelta(days=1)
    return [record for chunk in reversed(chunks) for record in chunk]


def _is_window_error(err: HTTPStatusError) -> bool:
    return err.status_code in (400, 406) and "janela" in err.body.lower()


def _download(
    code: str, first: pd.Timestamp | None, last: pd.Timestamp | None, options: RequestOptions
) -> list[dict[str, Any]]:
    if first is not None:
        return _forward(code, first, last or pd.Timestamp.today().normalize(), options)
    if last is None:
        try:
            return _request(code, None, None, options)
        except HTTPStatusError as err:
            if not _is_window_error(err):
                raise
            logger.info(
                "SGS series %s is daily; downloading in %d-year windows.", code, SGS_WINDOW_YEARS
            )
    return _backward(code, last or pd.Timestamp.today().normalize(), options)


def fetch_sgs_series(
    code: int | str,
    start: DateLike | None = None,
    end: DateLike | None = None,
    *,
    name: str | None = None,
    native_frequency: FrequencyLike | None = None,
    cache: DiskCache | bool | None = None,
    session: Any = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
) -> pd.Series:
    """Download one SGS series as a series with a native-frequency PeriodIndex.

    Parameters
    ----------
    code : int or str
        SGS series code (e.g. ``433`` for IPCA monthly inflation).
    start, end : date-like, optional
        Inclusive bounds (``"2020-01-01"``, ``"2020-03"``, ``"2020Q1"``, dates or
        periods). Without ``start`` the whole history up to ``end`` is downloaded.
    name : str, optional
        Series name (default ``"sgs_<code>"``).
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
        ``float64`` series indexed by a PeriodIndex at the native frequency.

    Raises
    ------
    DataSourceError
        Network failure, invalid code or unexpected answer.
    NowcastDataError
        If the series has no observation in the requested range.

    Examples
    --------
    >>> from nowcastbox.data_sources import fetch_sgs_series
    >>> ipca = fetch_sgs_series(433, start="2020-01", end="2020-12")  # doctest: +SKIP
    >>> ipca.index.freqstr  # doctest: +SKIP
    'M'
    """
    sgs_code = _sgs_code(code)
    label = name or f"sgs_{sgs_code}"
    first = to_timestamp(start)
    last = to_timestamp(end, end=True)
    check_range(first, last)
    options = RequestOptions(
        cache=cache,
        session=session,
        timeout=timeout,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
    )
    records = _download(sgs_code, first, last, options)
    if not records:
        raise NowcastDataError(f"SGS series {sgs_code} has no observations in the requested range.")
    dates = pd.to_datetime([r["data"] for r in records], format=_DATE_FORMAT)
    series = build_series(
        dates, [r["valor"] for r in records], label, native_frequency=native_frequency
    )
    return filter_period_range(series, first, last)


def fetch_sgs(
    codes: int | str | Sequence[int | str] | Mapping[str, int | str],
    start: DateLike | None = None,
    end: DateLike | None = None,
    *,
    native_frequency: FrequencyLike | Mapping[str, FrequencyLike] | None = None,
    base_frequency: FrequencyLike | None = None,
    cache: DiskCache | bool | None = None,
    session: Any = None,
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
) -> pd.DataFrame:
    """Download one or more series from the BCB SGS.

    Parameters
    ----------
    codes : int, str, sequence or mapping
        SGS code(s). A mapping ``{name: code}`` names the columns; otherwise columns
        are called ``"sgs_<code>"``.
    start, end : date-like, optional
        Inclusive bounds. Requests are split in windows of at most ten years
        (:func:`sgs_windows`), the SGS limit for daily series.
    native_frequency : Frequency, str, int or mapping, optional
        Native frequency of every series (scalar) or per series name (mapping);
        inferred from the observation dates when omitted.
    base_frequency : Frequency, str or int, optional
        Output grid. When omitted every series must share one native frequency and
        the frame is indexed at that frequency. With e.g. ``"M"``, lower-frequency
        series are stored in the last month of their period (quarterly → third
        month), ready for :class:`~nowcastbox.core.data.MixedFrequencyData`.
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
        ``float64`` frame indexed by a PeriodIndex (named ``"period"``), one column
        per series in input order.

    Raises
    ------
    DataSourceError
        Network failure, invalid code or unexpected answer.
    NowcastDataError
        Empty series, or mixed native frequencies without ``base_frequency``.

    Examples
    --------
    >>> from nowcastbox.data_sources import fetch_sgs
    >>> df = fetch_sgs({"ipca": 433, "ibc_br": 24363}, start="2015-01")  # doctest: +SKIP
    >>> df.index.freqstr  # doctest: +SKIP
    'M'
    >>> mixed = fetch_sgs({"ipca": 433, "gdp": 22099}, base_frequency="M")  # doctest: +SKIP
    """
    names = normalize_codes(codes, default_name=lambda c: f"sgs_{_sgs_code(c)}")
    series = {
        name: fetch_sgs_series(
            code,
            start,
            end,
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
    return combine_series(series, base_frequency=base_frequency)

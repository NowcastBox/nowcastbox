"""HTTP plumbing shared by the connectors: retries, back-off, caching, error mapping.

Only successful (HTTP 200) responses are cached. Query parameters listed in
``secret_params`` (API keys) are sent to the server but are never written to the
cache key, log messages or exception messages.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import requests

from nowcastbox.__version__ import __version__
from nowcastbox._logging import get_logger
from nowcastbox.data_sources.cache import DiskCache, resolve_cache
from nowcastbox.data_sources.exceptions import DataSourceError, HTTPStatusError

__all__ = ["RETRY_STATUS_CODES", "RequestOptions", "get_json", "get_text"]

logger = get_logger(__name__)

RETRY_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
"""Status codes that are retried with exponential back-off."""

USER_AGENT = f"nowcastbox/{__version__} (+https://github.com/NowcastBox/nowcastbox)"
_MAX_BACKOFF = 60.0
_BODY_SNIPPET = 500


def _sleep(seconds: float) -> None:
    """Sleep between retries (patched in tests)."""
    time.sleep(seconds)


@dataclass(frozen=True)
class RequestOptions:
    """Network options shared by every connector call.

    Parameters
    ----------
    cache : DiskCache, bool or None, default None
        ``None``/``True``: default on-disk cache; ``False``: disabled; or a cache.
    session : requests.Session, optional
        Session used for the requests (a temporary one is created otherwise).
    timeout : float, default 30.0
        Timeout in seconds of each request.
    max_retries : int, default 3
        Retries after the first attempt for connection errors, timeouts and status
        codes in :data:`RETRY_STATUS_CODES`.
    backoff_factor : float, default 1.0
        Waiting time before retry ``k`` (1-based) is ``backoff_factor * 2**(k-1)``
        seconds (capped at 60 s); a numeric ``Retry-After`` header takes precedence.
    """

    cache: DiskCache | bool | None = None
    session: Any = None
    timeout: float = 30.0
    max_retries: int = 3
    backoff_factor: float = 1.0
    secret_params: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise TypeError(f"max_retries must be an int, got {self.max_retries!r}.")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}.")
        if not self.timeout > 0:
            raise ValueError(f"timeout must be positive, got {self.timeout!r}.")
        if not self.backoff_factor >= 0:
            raise ValueError(f"backoff_factor must be >= 0, got {self.backoff_factor!r}.")


def _public_params(
    params: Mapping[str, Any] | None, secret: Sequence[str]
) -> list[tuple[str, str]]:
    if not params:
        return []
    return sorted((str(k), str(v)) for k, v in params.items() if k not in secret)


def describe_url(url: str, params: Mapping[str, Any] | None, secret: Sequence[str] = ()) -> str:
    """URL with its query string, secrets redacted (for keys, logs and errors)."""
    public = _public_params(params, secret)
    hidden = [k for k in (params or {}) if k in secret]
    query = urlencode(public + [(k, "***") for k in sorted(hidden)], safe="/:',()*")
    return f"{url}?{query}" if query else url


def _retry_delay(response: Any, attempt: int, backoff_factor: float) -> float:
    headers = getattr(response, "headers", None) or {}
    try:
        retry_after = float(str(headers.get("Retry-After")))
    except (TypeError, ValueError, AttributeError):
        retry_after = -1.0
    if retry_after >= 0:
        return min(retry_after, _MAX_BACKOFF)
    return min(backoff_factor * (2.0 ** (attempt - 1)), _MAX_BACKOFF)


def _attempt(
    session: Any, url: str, params: Mapping[str, Any] | None, options: RequestOptions, label: str
) -> tuple[Any, str | None]:
    """One GET; returns ``(response, problem)``, ``problem`` set when a retry is due."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    try:
        logger.debug("GET %s", label)
        response = session.get(
            url, params=dict(params or {}), timeout=options.timeout, headers=headers
        )
    except (requests.ConnectionError, requests.Timeout) as exc:
        return None, type(exc).__name__
    except requests.RequestException as exc:
        raise DataSourceError(f"Request to {label} failed: {type(exc).__name__}.") from None
    status = int(response.status_code)
    if status in RETRY_STATUS_CODES:
        return response, f"HTTP {status}"
    if status != 200:
        body = (response.text or "")[:_BODY_SNIPPET]
        raise HTTPStatusError(
            f"Request to {label} failed with HTTP {status}: {body.strip()[:200]}",
            status_code=status,
            body=body,
        )
    return response, None


def _perform(
    url: str, params: Mapping[str, Any] | None, options: RequestOptions, label: str
) -> str:
    """Run the request with retries; return the body of an HTTP 200 response."""
    owns_session = options.session is None
    session = requests.Session() if owns_session else options.session
    problem = "unknown error"
    response = None
    try:
        for attempt in range(options.max_retries + 1):
            if attempt:
                delay = _retry_delay(response, attempt, options.backoff_factor)
                logger.info(
                    "Retrying %s in %.1f s (attempt %d): %s", label, delay, attempt + 1, problem
                )
                _sleep(delay)
            response, failure = _attempt(session, url, params, options, label)
            if failure is None:
                return response.text
            problem = failure
    finally:
        if owns_session:
            session.close()
    raise DataSourceError(
        f"Request to {label} failed after {options.max_retries + 1} attempt(s): {problem}."
    )


def get_text(url: str, params: Mapping[str, Any] | None = None, *, options: RequestOptions) -> str:
    """GET ``url`` and return the response body, using the cache when enabled.

    Parameters
    ----------
    url : str
        Endpoint (without query string).
    params : mapping, optional
        Query parameters.
    options : RequestOptions
        Network options.

    Returns
    -------
    str
        Body of the HTTP 200 response.

    Raises
    ------
    HTTPStatusError
        Non-retryable HTTP error status (e.g. 400, 404).
    DataSourceError
        Retries exhausted or unexpected request failure.
    """
    label = describe_url(url, params, options.secret_params)
    cache = resolve_cache(options.cache)
    key = f"GET {label}"
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
    text = _perform(url, params, options, label)
    if cache is not None:
        cache.set(key, text)
    return text


def _decode(text: str) -> Any | None:
    """Decode JSON, returning ``None`` (never a valid API payload here) on failure."""
    try:
        return json.loads(text)
    except ValueError:
        return None


def get_json(url: str, params: Mapping[str, Any] | None = None, *, options: RequestOptions) -> Any:
    """GET ``url`` and decode the JSON body, using the cache when enabled.

    Some public APIs occasionally answer HTTP 200 with an HTML error page (e.g. the
    BCB gateway under load). Such non-JSON answers are never cached and are retried
    with the same back-off schedule as transient HTTP errors.

    Parameters
    ----------
    url : str
        Endpoint.
    params : mapping, optional
        Query parameters.
    options : RequestOptions
        Network options.

    Returns
    -------
    Any
        Decoded JSON document.

    Raises
    ------
    HTTPStatusError
        Non-retryable HTTP error status.
    DataSourceError
        If the body is still not valid JSON after all retries (e.g. an unknown
        series code answered with an HTML page), or retries are exhausted.
    """
    label = describe_url(url, params, options.secret_params)
    cache = resolve_cache(options.cache)
    key = f"GET {label}"
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            decoded = _decode(cached)
            if decoded is not None:
                return decoded
            cache.delete(key)
    text = ""
    for attempt in range(options.max_retries + 1):
        if attempt:
            delay = _retry_delay(None, attempt, options.backoff_factor)
            logger.info("Retrying %s in %.1f s: answer was not JSON.", label, delay)
            _sleep(delay)
        text = _perform(url, params, options, label)
        decoded = _decode(text)
        if decoded is not None:
            if cache is not None:
                cache.set(key, text)
            return decoded
    snippet = " ".join(text.split())[:120]
    raise DataSourceError(
        f"{label} did not return JSON (check the series/table code). "
        f"Response starts with: {snippet!r}"
    )

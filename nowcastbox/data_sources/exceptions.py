"""Errors raised by the data-source connectors.

All connector errors derive from :class:`DataSourceError`, itself a
:class:`~nowcastbox.core.exceptions.NowcastBoxError`, so that callers can catch every
download problem with a single ``except`` clause. Problems with the *content* of a
successful response that make it unusable as a time series raise
:class:`~nowcastbox.core.exceptions.NowcastDataError` instead.
"""

from __future__ import annotations

from nowcastbox.core.exceptions import NowcastBoxError

__all__ = ["DataSourceError", "HTTPStatusError", "MissingAPIKeyError"]


class DataSourceError(NowcastBoxError, RuntimeError):
    """A remote data source could not be queried or returned an unusable answer.

    Examples
    --------
    >>> from nowcastbox.data_sources import DataSourceError
    >>> from nowcastbox.core.exceptions import NowcastBoxError
    >>> issubclass(DataSourceError, NowcastBoxError)
    True
    """


class HTTPStatusError(DataSourceError):
    """The server answered with a non-success HTTP status code.

    Parameters
    ----------
    message : str
        Human-readable description (never contains secrets such as API keys).
    status_code : int
        HTTP status code returned by the server.
    body : str, default ""
        First characters of the response body (useful for API error messages).

    Attributes
    ----------
    status_code : int
        HTTP status code.
    body : str
        Truncated response body.

    Examples
    --------
    >>> from nowcastbox.data_sources import HTTPStatusError
    >>> err = HTTPStatusError("not found", status_code=404, body="{}")
    >>> err.status_code
    404
    """

    def __init__(self, message: str, *, status_code: int, body: str = "") -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.body = body


class MissingAPIKeyError(DataSourceError):
    """A data source that requires an API key was called without one.

    Examples
    --------
    >>> from nowcastbox.data_sources import MissingAPIKeyError, DataSourceError
    >>> issubclass(MissingAPIKeyError, DataSourceError)
    True
    """

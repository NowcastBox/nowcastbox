"""Fake HTTP session/response objects for the data-source tests (no network)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeResponse:
    status_code: int = 200
    text: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def json_body(cls, payload: Any, status_code: int = 200, **kw: Any) -> FakeResponse:
        return cls(status_code=status_code, text=json.dumps(payload), **kw)


@dataclass
class Call:
    url: str
    params: dict[str, Any]
    timeout: float
    headers: dict[str, str]


class FakeSession:
    """Session whose ``get`` delegates to ``handler(url, params)``.

    ``handler`` may return a FakeResponse or raise an exception. A list of
    responses/exceptions is also accepted and consumed in order.
    """

    def __init__(self, handler: Callable[[str, dict[str, Any]], FakeResponse] | list[Any]):
        self.handler = handler
        self.calls: list[Call] = []
        self.closed = False

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        timeout: float = 0.0,
        headers: dict | None = None,
    ) -> FakeResponse:
        params = dict(params or {})
        self.calls.append(Call(url, params, timeout, dict(headers or {})))
        if isinstance(self.handler, list):
            item = self.handler.pop(0)
        else:
            item = self.handler(url, params)
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self.closed = True

"""Fixtures for the data_sources tests: isolated cache and no sleeping.

The ``--run-network`` option and the skip hook for ``network`` tests live in the root
``tests/conftest.py``.
"""

from __future__ import annotations

import pytest

from nowcastbox.data_sources import _http, set_default_cache


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Default cache in a temporary directory; restores the default afterwards."""
    monkeypatch.setenv("NOWCASTBOX_CACHE_DIR", str(tmp_path / "default-cache"))
    monkeypatch.delenv("NOWCASTBOX_CACHE_TTL", raising=False)
    monkeypatch.delenv("NOWCASTBOX_DISABLE_CACHE", raising=False)
    set_default_cache(True)
    yield
    set_default_cache(True)


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """Record retry delays instead of sleeping."""
    recorded: list[float] = []
    monkeypatch.setattr(_http, "_sleep", recorded.append)
    return recorded


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(_http, "_sleep", lambda seconds: None)

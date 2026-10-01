"""Shared pytest configuration for the nowcastbox test-suite.

Fixtures defined here are available to every test package. Module-specific fixtures
belong in ``tests/<module>/conftest.py``.

- ``--run-network`` (or ``NOWCASTBOX_RUN_NETWORK=1``) enables the tests marked
  ``network`` (real BCB/IBGE/IPEA/FRED APIs); they are skipped otherwise.
- BLAS is limited to one thread for every test (``threadpoolctl``): the suite works
  with many small/medium matrices, for which multi-threaded OpenBLAS is often much
  slower (thread oversubscription, notably under WSL or with ``pytest -n``).
  Set ``NOWCASTBOX_TEST_BLAS_THREADS`` to another value to override.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - threadpoolctl is a test dependency
    threadpool_limits = None

settings.register_profile(
    "ci", max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile("dev", max_examples=50, deadline=None)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--run-network``."""
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked 'network' (real BCB/IBGE/IPEA/FRED APIs)",
    )


def network_enabled(config: pytest.Config) -> bool:
    """Whether tests marked ``network`` should run."""
    return bool(config.getoption("--run-network")) or (
        os.environ.get("NOWCASTBOX_RUN_NETWORK", "") == "1"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``network`` tests unless network runs are enabled."""
    if network_enabled(config):
        return
    skip = pytest.mark.skip(reason="network test: use --run-network or NOWCASTBOX_RUN_NETWORK=1")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True, scope="session")
def _limited_blas_threads() -> Iterator[None]:
    """Limit BLAS threads for the whole session (see module docstring)."""
    if threadpool_limits is None:  # pragma: no cover
        yield
        return
    n_threads = int(os.environ.get("NOWCASTBOX_TEST_BLAS_THREADS", "1"))
    with threadpool_limits(limits=n_threads, user_api="blas"):
        yield


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator (reproducible tests)."""
    return np.random.default_rng(20261001)


def _timing_unreliable() -> str | None:
    """Why wall-clock assertions are not meaningful in this run, or ``None``."""
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return "tests are running in parallel (xdist)"
    try:
        import coverage
    except ImportError:  # pragma: no cover - coverage is a test dependency
        return None
    if coverage.Coverage.current() is not None:
        return "coverage measurement inflates the run time"
    return None


@pytest.fixture
def timing_reliable() -> bool:
    """Whether wall-clock bounds should be asserted.

    Under coverage or ``pytest-xdist`` the machine is loaded and timings are not
    meaningful; timing tests then still check numerical results but skip their
    time bounds. Run ``pytest tests/performance -m slow`` alone to check speed.
    """
    return _timing_unreliable() is None

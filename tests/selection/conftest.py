"""Fixtures for the selection tests.

The tests factorise many medium-sized panels; multi-threaded BLAS can be dramatically
slower than a single thread for such sizes on some machines (thread oversubscription),
so BLAS is limited to one thread here when ``threadpoolctl`` is available.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    threadpool_limits = None


@pytest.fixture(autouse=True)
def _single_threaded_blas() -> Iterator[None]:
    if threadpool_limits is None:  # pragma: no cover
        yield
        return
    with threadpool_limits(limits=1, user_api="blas"):
        yield

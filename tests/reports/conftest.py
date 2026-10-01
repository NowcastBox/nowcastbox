"""Fixtures for tests/reports."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from tests.visualization import _fixtures


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def two_step():
    return _fixtures.two_step_results()


@pytest.fixture
def em():
    return _fixtures.em_results()


@pytest.fixture
def panel():
    return _fixtures.panel()

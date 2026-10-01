"""Run the doctests of nowcastbox.diagnostics."""

from __future__ import annotations

import doctest
import importlib
import warnings

import pytest

MODULES = [
    "nowcastbox.diagnostics",
    "nowcastbox.diagnostics._common",
    "nowcastbox.diagnostics.contribution",
    "nowcastbox.diagnostics.convergence",
    "nowcastbox.diagnostics.data_quality",
    "nowcastbox.diagnostics.report",
    "nowcastbox.diagnostics.residuals",
    "nowcastbox.diagnostics.stability",
]


@pytest.mark.parametrize("name", MODULES)
def test_doctests(name):
    module = importlib.import_module(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = doctest.testmod(
            module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
        )
    assert result.failed == 0
    assert result.attempted > 0

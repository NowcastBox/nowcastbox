"""Run the doctests of nowcastbox.pipeline and nowcastbox.cli."""

from __future__ import annotations

import doctest
import importlib
import warnings

import pytest

MODULES = [
    "nowcastbox.pipeline",
    "nowcastbox.pipeline.spec",
    "nowcastbox.pipeline.data",
    "nowcastbox.pipeline.examples",
    "nowcastbox.pipeline.snapshots",
    "nowcastbox.pipeline.runner",
    "nowcastbox.pipeline.selection",
    "nowcastbox.pipeline._common",
    "nowcastbox.cli.main",
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

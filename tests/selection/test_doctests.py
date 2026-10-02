"""Run the docstring examples of the selection modules as tests."""

from __future__ import annotations

import doctest
import importlib

import matplotlib
import pytest

matplotlib.use("Agg")

MODULES = [
    "nowcastbox.selection._lars",
    "nowcastbox.selection._panel",
    "nowcastbox.selection._plot",
    "nowcastbox.selection._search_checkpoint",
    "nowcastbox.selection._search_space",
    "nowcastbox.selection._search_treatments",
    "nowcastbox.selection._specified",
    "nowcastbox.selection.bai_ng_factors",
    "nowcastbox.selection.bai_ng_shocks",
    "nowcastbox.selection.blocks",
    "nowcastbox.selection.preselection",
    "nowcastbox.selection.search",
    "nowcastbox.selection.targeted",
]


@pytest.mark.parametrize("name", MODULES)
def test_docstring_examples(name):
    module = importlib.import_module(name)
    flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    result = doctest.testmod(module, optionflags=flags, verbose=False)
    assert result.attempted > 0
    assert result.failed == 0

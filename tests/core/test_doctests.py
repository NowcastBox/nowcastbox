"""Run the docstring examples of the core modules as tests."""

from __future__ import annotations

import doctest
import importlib

import pytest

MODULES = [
    "nowcastbox.core.exceptions",
    "nowcastbox.core.frequency",
    "nowcastbox.core.data",
    "nowcastbox.core.formula",
    "nowcastbox.core.results",
    "nowcastbox.core.base",
    "nowcastbox._logging",
]


@pytest.mark.parametrize("name", MODULES)
def test_docstring_examples(name):
    module = importlib.import_module(name)
    flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    result = doctest.testmod(module, optionflags=flags, verbose=False)
    assert result.failed == 0

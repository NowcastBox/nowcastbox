"""Run the docstring examples of the evaluation modules as tests."""

from __future__ import annotations

import doctest
import importlib

import pytest

MODULES = [
    "nowcastbox.evaluation",
    "nowcastbox.evaluation.backtest",
    "nowcastbox.evaluation.metrics",
    "nowcastbox.evaluation.tests",
]


@pytest.mark.parametrize("name", MODULES)
def test_docstring_examples(name: str) -> None:
    module = importlib.import_module(name)
    flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    result = doctest.testmod(module, optionflags=flags, verbose=False)
    assert result.attempted > 0
    assert result.failed == 0

"""Run the docstring examples of the benchmark modules as tests."""

from __future__ import annotations

import doctest
import importlib

import pytest

MODULES = [
    "nowcastbox.benchmarks",
    "nowcastbox.benchmarks._utils",
    "nowcastbox.benchmarks.ar",
    "nowcastbox.benchmarks.bridge",
    "nowcastbox.benchmarks.mean",
    "nowcastbox.benchmarks.midas",
    "nowcastbox.benchmarks.random_walk",
    "nowcastbox.benchmarks.sklearn_adapter",
]


@pytest.mark.parametrize("name", MODULES)
def test_docstring_examples(name: str) -> None:
    module = importlib.import_module(name)
    flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    result = doctest.testmod(module, optionflags=flags, verbose=False)
    assert result.attempted > 0
    assert result.failed == 0

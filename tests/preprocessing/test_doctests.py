"""Run the docstring examples of the preprocessing modules as tests."""

from __future__ import annotations

import doctest
import importlib

import pytest

MODULES = [
    "nowcastbox.preprocessing",
    "nowcastbox.preprocessing._utils",
    "nowcastbox.preprocessing.transforms",
    "nowcastbox.preprocessing.outliers",
    "nowcastbox.preprocessing.missing",
    "nowcastbox.preprocessing.aggregation",
    "nowcastbox.preprocessing.panel",
]


@pytest.mark.parametrize("name", MODULES)
def test_docstring_examples(name):
    module = importlib.import_module(name)
    flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    result = doctest.testmod(module, optionflags=flags, verbose=False)
    assert result.failed == 0


def test_public_api_is_exported():
    import nowcastbox.preprocessing as pp

    for name in pp.__all__:
        assert hasattr(pp, name), name
        obj = getattr(pp, name)
        if callable(obj):
            assert obj.__doc__, name

"""Package-level smoke tests (import, version, entry point)."""

from __future__ import annotations

import importlib
import re

import pytest

import nowcastbox
from nowcastbox.cli import main

SUBPACKAGES = [
    "core",
    "preprocessing",
    "statespace",
    "models",
    "selection",
    "vintages",
    "news",
    "benchmarks",
    "evaluation",
    "data_sources",
    "datasets",
    "visualization",
    "reports",
    "experiment",
    "pipeline",
    "simulate",
    "cli",
]


def test_version():
    assert re.match(r"^\d+\.\d+\.\d+", nowcastbox.__version__)


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import_and_are_documented(name):
    module = importlib.import_module(f"nowcastbox.{name}")
    assert module.__doc__


def test_cli_version(capsys):
    assert main(["--version"]) == 0
    assert nowcastbox.__version__ in capsys.readouterr().out


def test_cli_missing_spec(capsys):
    assert main(["run", "x.yaml"]) == 2
    assert "Spec file not found" in capsys.readouterr().err

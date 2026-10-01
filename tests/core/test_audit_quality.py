"""Regression tests from the final software-quality audit (core)."""

from __future__ import annotations

import pytest

from nowcastbox.core.exceptions import FormulaError
from nowcastbox.core.formula import is_formula, resolve_target


@pytest.mark.parametrize("target", [None, 3, ["gdp"]])
def test_resolve_target_rejects_non_string(target: object) -> None:
    with pytest.raises(FormulaError, match="target must be a series name or a formula"):
        resolve_target(target, ["gdp", "ip"])  # type: ignore[arg-type]


def test_is_formula_non_string_is_false() -> None:
    assert is_formula(None) is False  # type: ignore[arg-type]

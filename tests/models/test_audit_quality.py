"""Regression tests from the final software-quality audit (models)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd


def test_import_does_not_load_statsmodels() -> None:
    # statsmodels.api costs ~0.7 s; it is only needed when a bridge is estimated
    code = "import sys, nowcastbox; print('statsmodels' in sys.modules)"
    out = subprocess.run(  # noqa: S603 - fixed command
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_bridge_still_estimates_with_lazy_statsmodels() -> None:
    from nowcastbox.models.bridge import fit_bridge_regression

    rng = np.random.default_rng(0)
    idx = pd.period_range("2000Q1", periods=40, freq="Q")
    x = pd.DataFrame({"f1": rng.standard_normal(40)}, index=idx)
    y = pd.Series(1.0 + 2.0 * x["f1"] + 0.01 * rng.standard_normal(40), name="y")
    fit = fit_bridge_regression(y, x)
    assert np.isclose(fit.params["f1"], 2.0, atol=0.05)


def test_positive_parameter_message_is_consistent() -> None:
    import pytest

    from nowcastbox.models.em import _check_positive

    for bad in (-1, 0, float("nan"), "2"):
        with pytest.raises(ValueError, match="df must be a finite positive number"):
            _check_positive(bad, "df")
    _check_positive(2.5, "df")

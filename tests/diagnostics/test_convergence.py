"""Tests of the EM convergence diagnostics."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.exceptions import ConvergenceWarning, NowcastDataError
from nowcastbox.diagnostics import ConvergenceDiagnostics, em_convergence, loglikelihood_path


def test_path_table_and_relative_change():
    path = [-120.0, -101.0, -100.5, -100.49]
    diag = em_convergence(path, tol=1e-3)
    t = diag.to_frame()
    assert t.index.name == "iteration"
    assert t.columns.tolist() == ["loglikelihood", "change", "relative_change", "decrease"]
    np.testing.assert_allclose(t["change"].iloc[1:], [19.0, 0.5, 0.01])
    expected = 0.01 / ((100.49 + 100.5) / 2)
    assert t["relative_change"].iloc[-1] == pytest.approx(expected, rel=1e-9)
    assert diag.converged and diag.monotone and diag.n_iter == 3
    assert diag.largest_decrease == 0.0
    assert diag.last_relative_change == pytest.approx(expected)


def test_non_monotone_path_warns():
    with pytest.warns(ConvergenceWarning, match="decreased in 2 iteration"):
        diag = em_convergence([-10.0, -9.0, -9.5, -9.0, -9.2, -9.19999], tol=1e-3)
    assert diag.n_decreases == 2
    assert not diag.monotone
    assert diag.path["decrease"].tolist() == [False, False, True, False, True, False]
    assert diag.largest_decrease == pytest.approx(0.5 / 9.25)
    assert diag.converged


def test_not_converged_warns_and_decrease_tolerance():
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        diag = em_convergence([-10.0, -9.0], tol=1e-6)
    assert not diag.converged
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet = em_convergence([-10.0, -9.0, -9.0000001], tol=1e-3, decrease_tol=1e-6)
        assert quiet.n_decreases == 0
        em_convergence([-10.0, -9.0], tol=1e-6, warn=False)
        single = em_convergence(np.array([-5.0]))
    assert single.n_iter == 0 and not single.converged
    assert np.isnan(single.last_relative_change)


def test_without_tolerance_is_not_converged():
    diag = em_convergence([-10.0, -9.0], warn=False)
    assert diag.tol is None and not diag.converged
    assert "tol=-" in diag.summary()


def test_from_em_results(em_results):
    diag = em_convergence(em_results)
    assert diag.converged is bool(em_results.converged)
    assert diag.tol == em_results.model_params["tol"]
    assert diag.n_iter == em_results.loglikelihood_path.size - 1
    assert diag.path["loglikelihood"].iloc[-1] == pytest.approx(em_results.loglikelihood)
    row = diag.summary_frame().iloc[0]
    assert row["final_loglikelihood"] == pytest.approx(em_results.loglikelihood)
    text = diag.summary()
    assert text.startswith("EM convergence") and "Non-monotone steps" in text


def test_from_results_info_and_flag(em_results):
    path = np.array([-50.0, -40.0, -45.0])
    fake = em_results.replace(
        loglikelihood_path=None, info={"loglikelihood_path": path}, converged=False
    )
    assert loglikelihood_path(fake).tolist() == path.tolist()
    with pytest.warns(ConvergenceWarning):
        diag = em_convergence(fake)
    assert not diag.converged and diag.n_decreases == 1
    assert "largest" in diag.summary()


def test_loglikelihood_path_sources(two_step_results):
    assert loglikelihood_path(two_step_results) is None
    assert loglikelihood_path(pd.Series([1.0, 2.0])).tolist() == [1.0, 2.0]
    assert loglikelihood_path((1, 2)).tolist() == [1.0, 2.0]
    assert loglikelihood_path("abc") is None


@pytest.mark.parametrize("bad", [None, [], [1.0, np.nan], [np.inf]])
def test_invalid_paths(bad):
    with pytest.raises(NowcastDataError, match="log-likelihood path"):
        em_convergence(bad)


def test_dataclass_type():
    assert isinstance(em_convergence([1.0, 2.0], warn=False), ConvergenceDiagnostics)

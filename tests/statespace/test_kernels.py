"""Sparse (CSR) and dense (BLAS) transition paths of the Kalman kernels."""

from __future__ import annotations

import numpy as np
import pytest

from nowcastbox.statespace import (
    StateSpace,
    kalman_smoother,
    random_missing,
    random_state_space,
    simulate_state_space,
)
from nowcastbox.statespace.kalman import transition_csr

from ._dfm import dfm_state_space, ragged_panel
from ._reference import statsmodels_smoother

pytestmark = pytest.mark.reference_validation


def _compare(model: StateSpace, y: np.ndarray, method: str = "auto") -> None:
    ours = kalman_smoother(model, y, method=method)  # type: ignore[arg-type]
    ref = statsmodels_smoother(model, y)
    assert ours.loglikelihood == pytest.approx(float(np.sum(ref.llf_obs)), rel=1e-9)
    np.testing.assert_allclose(ours.smoothed_state, ref.smoothed_state.T, atol=1e-7)
    np.testing.assert_allclose(
        ours.smoothed_state_cov, np.moveaxis(ref.smoothed_state_cov, -1, 0), atol=1e-7
    )
    np.testing.assert_allclose(
        ours.filter_result.predicted_state_cov,
        np.moveaxis(ref.predicted_state_cov, -1, 0),
        atol=1e-7,
    )


def test_transition_csr():
    t_mat = np.zeros((20, 20))
    t_mat[0, 0] = 0.5
    t_mat[np.arange(1, 20), np.arange(19)] = 1.0
    ptr, idx, _val, sparse = transition_csr(t_mat)
    assert sparse and ptr[-1] == 20 and idx.dtype == np.int64
    dense = np.random.default_rng(0).normal(size=(20, 20))
    assert not transition_csr(dense)[3]
    assert transition_csr(dense[:4, :4])[3]  # small matrices always use the loops


@pytest.mark.parametrize("method", ["univariate", "multivariate"])
def test_dense_transition_blas_path(method):
    model = random_state_space(6, 20, random_state=3)  # m > 16, dense T -> BLAS
    assert not transition_csr(model.T)[3]
    y, _ = simulate_state_space(model, 25, random_state=4)
    _compare(model, random_missing(y, 0.2, random_state=5), method)


@pytest.mark.parametrize("method", ["univariate", "multivariate"])
def test_sparse_transition_path(rng, method):
    model = dfm_state_space(8, 2, rng=rng, obs_var=0.05)  # m = 23, sparse T -> CSR
    assert transition_csr(model.T)[3] and model.n_states > 16
    y = ragged_panel(model, 30, rng, n_quarterly=2)
    _compare(model, y, method)


def test_intercepts_with_sparse_transition(rng):
    model = dfm_state_space(6, 2, rng=rng, intercepts=True, initial_mean=True, obs_var=0.1)
    y = ragged_panel(model, 24, rng, n_quarterly=2)
    _compare(model, y)

"""Tests for nowcastbox.statespace.collapse (Jungbacker & Koopman, 2015)."""

from __future__ import annotations

import numpy as np
import pytest

from nowcastbox.statespace import (
    StateSpace,
    collapse_observations,
    kalman_filter,
    kalman_smoother,
    random_missing,
    simulate_state_space,
)


def _factor_model(n_obs=40, r=2, seed=0, full_h=False, intercept=False):
    rng = np.random.default_rng(seed)
    t_mat = np.diag(rng.uniform(0.3, 0.9, r))
    z = rng.normal(size=(n_obs, r))
    if full_h:
        a = rng.normal(size=(n_obs, n_obs))
        h = a @ a.T / n_obs + 0.5 * np.eye(n_obs)
    else:
        h = rng.uniform(0.3, 1.0, n_obs)
    d = rng.normal(size=n_obs) if intercept else None
    return StateSpace(t_mat, z, np.eye(r), h, obs_intercept=d)


def test_dimensions_and_patterns():
    ssm = _factor_model()
    y, _ = simulate_state_space(ssm, 30, random_state=1)
    y[-1, :10] = np.nan
    y[-2, :5] = np.nan
    col = collapse_observations(ssm, y)
    assert col.n_patterns == 3
    assert col.max_dimension == 2
    assert col.n_periods == 30
    np.testing.assert_array_equal(col.n_collapsed, np.full(30, 2))
    assert col.design.shape == (3, 2, 2)
    assert np.isfinite(col.observations).all()


@pytest.mark.parametrize("full_h", [False, True])
@pytest.mark.parametrize("intercept", [False, True])
def test_exact_equivalence_with_full_filter(full_h, intercept):
    ssm = _factor_model(full_h=full_h, intercept=intercept, seed=3)
    y, _ = simulate_state_space(ssm, 50, random_state=4)
    y = random_missing(y, 0.3, random_state=5)
    y[7] = np.nan
    y[8, 1:] = np.nan  # single observation (p_t < rank)
    full = kalman_smoother(ssm, y)
    col = kalman_smoother(ssm, y, collapse=True)
    assert col.filter_result.collapsed
    np.testing.assert_allclose(
        col.filter_result.loglikelihood_obs, full.filter_result.loglikelihood_obs, atol=1e-9
    )
    np.testing.assert_allclose(
        col.filter_result.filtered_state, full.filter_result.filtered_state, atol=1e-9
    )
    np.testing.assert_allclose(col.smoothed_state, full.smoothed_state, atol=1e-9)
    np.testing.assert_allclose(col.smoothed_state_cov, full.smoothed_state_cov, atol=1e-9)
    np.testing.assert_allclose(col.smoothed_state_autocov, full.smoothed_state_autocov, atol=1e-9)


def test_rank_deficient_design_with_lag_states():
    """Z = [Lambda, 0]: rank r < m, collapsed dimension is r."""
    rng = np.random.default_rng(6)
    r, n = 2, 25
    t_mat = np.zeros((4, 4))
    t_mat[:2, :2] = [[0.6, 0.1], [0.0, 0.4]]
    t_mat[2:, :2] = np.eye(2)
    sel = np.vstack([np.eye(2), np.zeros((2, 2))])
    z = np.hstack([rng.normal(size=(n, r)), np.zeros((n, r))])
    ssm = StateSpace(t_mat, z, np.eye(2), np.ones(n), selection=sel)
    y, _ = simulate_state_space(ssm, 40, random_state=7)
    y = random_missing(y, 0.2, random_state=8)
    col = collapse_observations(ssm, y)
    assert col.max_dimension == 2
    np.testing.assert_allclose(
        kalman_filter(ssm, y, collapse=True).loglikelihood,
        kalman_filter(ssm, y).loglikelihood,
        atol=1e-9,
    )


def test_all_missing_period_and_padding():
    ssm = _factor_model(n_obs=5, r=2)
    y, _ = simulate_state_space(ssm, 6, random_state=9)
    y[2] = np.nan
    y[3, 1:] = np.nan
    col = collapse_observations(ssm, y)
    assert col.n_collapsed[2] == 0
    assert col.n_collapsed[3] == 1
    assert np.isnan(col.observations[2]).all()
    assert np.isnan(col.observations[3, 1])
    assert col.loglikelihood_adjustment[2] == 0.0


def test_all_missing_everywhere():
    ssm = _factor_model(n_obs=3, r=1)
    col = collapse_observations(ssm, np.full((4, 3), np.nan))
    assert col.max_dimension == 1
    assert np.isnan(col.observations).all()


def test_adjustment_formula_single_period():
    """Check the log-likelihood decomposition by direct Gaussian density evaluation."""
    ssm = _factor_model(n_obs=6, r=1, seed=10, intercept=True)
    y = np.random.default_rng(11).normal(size=(1, 6))
    f = ssm.Z @ ssm.P0 @ ssm.Z.T + np.diag(ssm.H)
    v = y[0] - ssm.d - ssm.Z @ ssm.a0
    direct = -0.5 * (6 * np.log(2 * np.pi) + np.linalg.slogdet(f)[1] + v @ np.linalg.solve(f, v))
    assert kalman_filter(ssm, y, collapse=True).loglikelihood == pytest.approx(direct, abs=1e-10)


def test_rank_tol():
    ssm = _factor_model(n_obs=10, r=2)
    y, _ = simulate_state_space(ssm, 5, random_state=12)
    col = collapse_observations(ssm, y, rank_tol=1e6)
    assert col.max_dimension == 1  # padded; all singular values dropped
    np.testing.assert_array_equal(col.n_collapsed, 0)
    with pytest.raises(ValueError, match="rank_tol"):
        collapse_observations(ssm, y, rank_tol=-1.0)


def test_singular_h_raises():
    ssm = StateSpace([[0.5]], np.ones((3, 1)), [[1.0]], [1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="positive definite"):
        collapse_observations(ssm, np.zeros((2, 3)))


def test_singular_full_h_raises():
    h = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    ssm = StateSpace([[0.5]], np.ones((3, 1)), [[1.0]], h)
    with pytest.raises(ValueError, match="positive definite"):
        collapse_observations(ssm, np.zeros((2, 3)))
    # dropping one of the perfectly correlated series makes the block PD
    col = collapse_observations(ssm, np.array([[0.0, np.nan, 1.0]]))
    assert col.n_collapsed[0] == 1

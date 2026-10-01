"""Tests for nowcastbox.statespace.smoother."""

from __future__ import annotations

import numpy as np
import pytest

from nowcastbox.statespace import (
    StateSpace,
    companion_matrix,
    kalman_filter,
    kalman_smoother,
    random_missing,
    random_state_space,
    simulate_state_space,
    smooth,
)
from tests.statespace._reference import naive_rts, statsmodels_smoother

TOL = 1e-8


def _model_and_data(seed, n_obs=6, n_states=3, n_periods=60, frac=0.35, **kw):
    ssm = random_state_space(n_obs, n_states, random_state=seed, **kw)
    y, _ = simulate_state_space(ssm, n_periods, random_state=seed + 100)
    y = random_missing(y, frac, random_state=seed + 200)
    y[-3:, : n_obs // 2] = np.nan  # ragged edge
    return ssm, y


def _assert_matches_statsmodels(res, ref):
    np.testing.assert_allclose(res.smoothed_state, ref.smoothed_state.T, atol=TOL, rtol=0)
    np.testing.assert_allclose(
        res.smoothed_state_cov, ref.smoothed_state_cov.transpose(2, 0, 1), atol=TOL, rtol=0
    )
    np.testing.assert_allclose(
        res.smoothed_state_autocov,
        ref.smoothed_state_autocov.transpose(2, 0, 1),
        atol=TOL,
        rtol=0,
    )


@pytest.mark.reference_validation
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize(
    ("method", "collapse"), [("univariate", False), ("multivariate", False), ("univariate", True)]
)
def test_matches_statsmodels(seed, method, collapse):
    ssm, y = _model_and_data(seed, n_disturbances=1 if seed % 2 else None, intercepts=seed % 3 == 1)
    ref = statsmodels_smoother(ssm, y)
    res = kalman_smoother(ssm, y, method=method, collapse=collapse)
    _assert_matches_statsmodels(res, ref)
    assert res.loglikelihood == pytest.approx(ref.llf_obs.sum(), abs=TOL)


@pytest.mark.reference_validation
@pytest.mark.parametrize("seed", range(3))
def test_matches_statsmodels_full_h(seed):
    ssm, y = _model_and_data(seed, diagonal_obs_cov=False)
    ref = statsmodels_smoother(ssm, y)
    _assert_matches_statsmodels(kalman_smoother(ssm, y), ref)
    _assert_matches_statsmodels(kalman_smoother(ssm, y, collapse=True), ref)


@pytest.mark.reference_validation
def test_dfm_with_lagged_states_and_mariano_murasawa_loadings():
    """Factor VAR(1) in companion form with 5 lags (singular P_{t+1}); quarterly rows."""
    r, n_m, n_q = 2, 8, 2
    rng = np.random.default_rng(1)
    a1 = np.array([[0.7, 0.1], [0.0, 0.5]])
    t_mat = companion_matrix([a1], n_lags=5)
    r_mat = np.zeros((10, r))
    r_mat[:r] = np.eye(r)
    lam = rng.normal(size=(n_m + n_q, r))
    z = np.zeros((n_m + n_q, 10))
    z[:n_m, :r] = lam[:n_m]
    for j, w in enumerate([1, 2, 3, 2, 1]):
        z[n_m:, j * r : (j + 1) * r] = w / 3 * lam[n_m:]
    ssm = StateSpace(t_mat, z, np.eye(r), np.full(n_m + n_q, 0.3), selection=r_mat)
    y, _ = simulate_state_space(ssm, 90, random_state=2)
    mask = np.ones(90, dtype=bool)
    mask[2::3] = False
    y[mask, n_m:] = np.nan  # quarterly in the 3rd month only
    y[-4:, :4] = np.nan
    ref = statsmodels_smoother(ssm, y)
    for collapse in (False, True):
        res = kalman_smoother(ssm, y, collapse=collapse)
        _assert_matches_statsmodels(res, ref)
        assert res.loglikelihood == pytest.approx(ref.llf_obs.sum(), abs=TOL)


def test_matches_naive_rts():
    ssm, y = _model_and_data(9, n_obs=4, n_states=2)
    a_s, v_s, autocov = naive_rts(ssm, y)
    res = kalman_smoother(ssm, y)
    np.testing.assert_allclose(res.smoothed_state, a_s, atol=1e-9)
    np.testing.assert_allclose(res.smoothed_state_cov, v_s, atol=1e-9)
    np.testing.assert_allclose(res.smoothed_state_autocov, autocov, atol=1e-9)


def test_last_period_equals_filter():
    ssm, y = _model_and_data(10)
    res = kalman_smoother(ssm, y)
    fres = res.filter_result
    np.testing.assert_allclose(res.smoothed_state[-1], fres.filtered_state[-1], atol=1e-12)
    np.testing.assert_allclose(res.smoothed_state_cov[-1], fres.filtered_state_cov[-1], atol=1e-12)
    np.testing.assert_allclose(res.smoothed_state_autocov[-1], ssm.T @ fres.filtered_state_cov[-1])


def test_smoothed_cov_not_larger_than_filtered():
    ssm, y = _model_and_data(12)
    res = kalman_smoother(ssm, y)
    diff = res.filter_result.filtered_state_cov - res.smoothed_state_cov
    for mat in diff:
        assert np.linalg.eigvalsh(mat).min() > -1e-10


def test_all_missing_returns_prior():
    ssm = random_state_space(2, 2, random_state=13, intercepts=True)
    res = kalman_smoother(ssm, np.full((4, 2), np.nan))
    a, p = ssm.a0.copy(), ssm.P0.copy()
    for t in range(4):
        np.testing.assert_allclose(res.smoothed_state[t], a, atol=1e-12)
        np.testing.assert_allclose(res.smoothed_state_cov[t], p, atol=1e-12)
        a = ssm.T @ a + ssm.c
        p = ssm.T @ p @ ssm.T.T + ssm.state_disturbance_cov


def test_signal_helpers():
    ssm, y = _model_and_data(14, intercepts=True)
    res = kalman_smoother(ssm, y)
    np.testing.assert_allclose(res.smoothed_signal(), res.smoothed_state @ ssm.Z.T + ssm.d)
    np.testing.assert_allclose(
        res.smoothed_signal_cov(2), ssm.Z @ res.smoothed_state_cov[2] @ ssm.Z.T
    )
    np.testing.assert_allclose(res.smoothed_signal_cov(-1), res.smoothed_signal_cov(y.shape[0] - 1))
    with pytest.raises(IndexError):
        res.smoothed_signal_cov(-y.shape[0] - 1)
    assert res.model is ssm
    assert res.n_periods == y.shape[0]


def test_smooth_requires_stored_quantities():
    ssm, y = _model_and_data(15)
    with pytest.raises(ValueError, match="store_smoother=False"):
        smooth(kalman_filter(ssm, y, store_smoother=False))


def test_univariate_equals_multivariate_smoother():
    ssm, y = _model_and_data(16, n_obs=10, n_states=4)
    uni = kalman_smoother(ssm, y, method="univariate")
    multi = kalman_smoother(ssm, y, method="multivariate")
    np.testing.assert_allclose(uni.smoothed_state, multi.smoothed_state, atol=1e-10)
    np.testing.assert_allclose(uni.smoothed_state_cov, multi.smoothed_state_cov, atol=1e-10)
    np.testing.assert_allclose(uni.smoothed_state_autocov, multi.smoothed_state_autocov, atol=1e-10)

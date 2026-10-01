"""Approximate diffuse initialization (Durbin & Koopman, 2012, sec. 5.1)."""

from __future__ import annotations

import numpy as np
import pytest

from nowcastbox.statespace import (
    StateSpace,
    approximate_diffuse_initial_cov,
    kalman_filter,
    stationary_initial_cov,
)


def test_local_level_matches_exact_diffuse_limit():
    # y_t = mu_t + eps_t, mu_{t+1} = mu_t + eta_t: the exact diffuse filter gives
    # a_2 = y_1 and P_2 = h + q (Durbin & Koopman, 2012, sec. 2.9)
    h, q = 0.7, 0.3
    # kappa trades truncation error (h^2 / kappa) against cancellation (eps kappa)
    p0 = approximate_diffuse_initial_cov([[1.0]], [[q]], kappa=1e7)
    ssm = StateSpace([[1.0]], [[1.0]], [[q]], [h], initial_state_cov=p0)
    res = kalman_filter(ssm, np.array([[2.5], [1.0], [np.nan]]))
    assert res.predicted_state[1, 0] == pytest.approx(2.5, rel=1e-6)
    assert res.predicted_state_cov[1, 0, 0] == pytest.approx(h + q, rel=1e-6)
    # the first log-likelihood term is dominated by kappa; the rest is proper
    assert res.loglikelihood_obs[0] < -5.0
    f = (h + q) + h  # F_2 = P_2 + H
    v = 1.0 - res.predicted_state[1, 0]
    expected = -0.5 * (np.log(2 * np.pi) + np.log(f) + v * v / f)
    assert res.loglikelihood_obs[1] == pytest.approx(expected, rel=1e-6)


def test_nonstationary_blocks_only():
    t_mat = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.2], [0.0, 0.0, 0.3]])
    rqr = np.diag([1.0, 0.75, 0.5])
    p0 = approximate_diffuse_initial_cov(t_mat, rqr)
    assert p0[0, 0] == 1e6
    np.testing.assert_array_equal(p0[0, 1:], 0.0)
    sub = np.ix_([1, 2], [1, 2])
    np.testing.assert_allclose(p0[sub], stationary_initial_cov(t_mat[sub], rqr[sub]))


def test_all_and_explicit_states():
    t_mat = np.diag([0.5, 0.4])
    rqr = np.eye(2)
    np.testing.assert_array_equal(
        approximate_diffuse_initial_cov(t_mat, rqr, kappa=10.0, states="all"), 10.0 * np.eye(2)
    )
    p0 = approximate_diffuse_initial_cov(t_mat, rqr, kappa=5.0, states=[1])
    assert p0[1, 1] == 5.0 and p0[0, 0] == pytest.approx(1.0 / 0.75)
    mask = approximate_diffuse_initial_cov(t_mat, rqr, kappa=5.0, states=np.array([True, False]))
    assert mask[0, 0] == 5.0
    none = approximate_diffuse_initial_cov(t_mat, rqr, states=np.array([], dtype=int))
    np.testing.assert_allclose(none, stationary_initial_cov(t_mat, rqr))


def test_stationary_model_is_unchanged():
    p0 = approximate_diffuse_initial_cov([[0.5]], [[0.75]])
    np.testing.assert_allclose(p0, [[1.0]])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kappa": 0.0}, "kappa"),
        ({"kappa": np.inf}, "kappa"),
        ({"states": "some"}, "states must be"),
        ({"states": np.array([True])}, "shape"),
        ({"states": [0.5]}, "integer"),
        ({"states": [5]}, "out of range"),
        ({"states": [0]}, "stationary sub-system"),
    ],
)
def test_errors(kwargs, match):
    t_mat = np.diag([0.5, 1.0])
    with pytest.raises(ValueError, match=match):
        approximate_diffuse_initial_cov(t_mat, np.eye(2), **kwargs)


def test_shape_errors():
    with pytest.raises(ValueError, match="transition"):
        approximate_diffuse_initial_cov(np.ones((2, 3)), np.eye(2))
    with pytest.raises(ValueError, match="state_disturbance_cov"):
        approximate_diffuse_initial_cov(np.eye(2) * 0.5, np.eye(3))


def test_model_method():
    rw = StateSpace(
        [[1.0, 0.0], [0.0, 0.5]], np.eye(2), np.eye(2), [1.0, 1.0],
        initial_state=[3.0, 1.0], initial_state_cov=np.eye(2),
    )  # fmt: skip
    diffuse = rw.with_approximate_diffuse(1e7)
    assert diffuse.P0[0, 0] == 1e7
    assert diffuse.P0[1, 1] == pytest.approx(1.0 / 0.75)
    np.testing.assert_array_equal(diffuse.a0, [0.0, 1.0])
    every = rw.with_approximate_diffuse(states="all")
    np.testing.assert_array_equal(every.a0, [0.0, 0.0])


def test_constructor_error_mentions_helper():
    with pytest.raises(ValueError, match="approximate_diffuse_initial_cov"):
        StateSpace([[1.0]], [[1.0]], [[1.0]], [1.0])

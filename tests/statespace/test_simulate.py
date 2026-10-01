"""Tests for nowcastbox.statespace.simulate."""

from __future__ import annotations

import numpy as np
import pytest

from nowcastbox.statespace import (
    StateSpace,
    random_missing,
    random_state_space,
    simulate_state_space,
)


def test_reproducible():
    ssm = random_state_space(3, 2, random_state=0)
    y1, s1 = simulate_state_space(ssm, 20, random_state=5)
    y2, s2 = simulate_state_space(ssm, 20, random_state=np.random.default_rng(5))
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(s1, s2)


def test_moments_match_stationary_distribution():
    ssm = StateSpace(
        [[0.5]],
        [[1.0], [2.0]],
        [[0.75]],
        [0.5, 0.1],
        state_intercept=[1.0],
        obs_intercept=[0.0, 1.0],
    )
    y, alpha = simulate_state_space(ssm, 40000, random_state=1)
    assert alpha.mean() == pytest.approx(2.0, abs=0.05)
    assert alpha.var() == pytest.approx(1.0, abs=0.05)
    resid = y - alpha @ ssm.Z.T - ssm.d
    np.testing.assert_allclose(resid.var(axis=0), [0.5, 0.1], rtol=0.05)
    # lag-1 autocorrelation of the state
    assert np.corrcoef(alpha[1:, 0], alpha[:-1, 0])[0, 1] == pytest.approx(0.5, abs=0.03)


def test_full_obs_cov_and_initial_state():
    h = np.array([[1.0, 0.8], [0.8, 1.0]])
    ssm = StateSpace([[0.0]], [[0.0], [0.0]], [[1.0]], h)
    y, alpha = simulate_state_space(ssm, 20000, initial_state=[3.0], random_state=2)
    assert alpha[0, 0] == 3.0
    np.testing.assert_allclose(np.cov(y.T), h, atol=0.05)


def test_singular_state_cov():
    ssm = StateSpace(
        [[0.5, 0.0], [1.0, 0.0]], [[1.0, 0.0]], [[1.0]], [1.0], selection=[[1.0], [0.0]]
    )
    _, alpha = simulate_state_space(ssm, 50, random_state=3)
    np.testing.assert_allclose(alpha[1:, 1], alpha[:-1, 0])


def test_simulate_errors():
    ssm = random_state_space(2, 2, random_state=0)
    with pytest.raises(ValueError, match="positive integer"):
        simulate_state_space(ssm, 0)
    with pytest.raises(ValueError, match="positive integer"):
        simulate_state_space(ssm, 2.5)
    with pytest.raises(ValueError, match="initial_state"):
        simulate_state_space(ssm, 5, initial_state=[1.0])


def test_random_state_space_options():
    ssm = random_state_space(
        4,
        3,
        n_disturbances=1,
        diagonal_obs_cov=False,
        intercepts=True,
        spectral_radius=0.5,
        random_state=1,
    )
    assert ssm.n_disturbances == 1
    assert not ssm.obs_cov_is_diagonal
    assert np.max(np.abs(np.linalg.eigvals(ssm.T))) == pytest.approx(0.5)
    assert np.any(ssm.c != 0)
    assert np.any(ssm.d != 0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_obs": 0, "n_states": 1}, "n_obs and n_states"),
        ({"n_obs": 1, "n_states": 2, "n_disturbances": 3}, "n_disturbances"),
        ({"n_obs": 1, "n_states": 2, "spectral_radius": 1.0}, "spectral_radius"),
    ],
)
def test_random_state_space_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        random_state_space(**kwargs)


def test_random_missing():
    y = np.ones((100, 5))
    out = random_missing(y, 0.3, random_state=0)
    assert 0.2 < np.isnan(out).mean() < 0.4
    assert not np.isnan(y).any()
    assert not np.isnan(random_missing(y, 0.0)).any()
    assert np.isnan(random_missing(y, 1.0)).all()
    with pytest.raises(ValueError, match="fraction"):
        random_missing(y, 1.5)

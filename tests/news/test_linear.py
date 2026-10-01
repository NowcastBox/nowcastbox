"""Tests of the batched linear smoother :func:`nowcastbox.news._linear.batched_functional`."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nowcastbox.news._linear import batched_functional
from nowcastbox.statespace import (
    StateSpace,
    kalman_smoother,
    random_missing,
    random_state_space,
    simulate_state_space,
)


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    n_obs=st.integers(1, 4),
    n_states=st.integers(1, 4),
    diagonal=st.booleans(),
    intercepts=st.booleans(),
    frac=st.floats(0.0, 0.7),
)
def test_matches_kalman_smoother(seed, n_obs, n_states, diagonal, intercepts, frac):
    ssm = random_state_space(
        n_obs, n_states, diagonal_obs_cov=diagonal, intercepts=intercepts, random_state=seed
    )
    rng = np.random.default_rng(seed)
    n_periods = 25
    y = random_missing(rng.standard_normal((n_periods, n_obs)), frac, random_state=seed)
    pattern = ~np.isnan(y)
    gain = rng.standard_normal(n_states)
    pos = int(rng.integers(0, n_periods))
    data = np.stack([np.nan_to_num(y), np.nan_to_num(2.0 * y - 1.0)], axis=2)
    out = batched_functional(ssm, gain, 0.3, pattern, data, pos)
    for k, values in enumerate([y, np.where(pattern, 2.0 * y - 1.0, np.nan)]):
        ref = gain @ kalman_smoother(ssm, values).smoothed_state[pos] + 0.3
        assert out[k] == pytest.approx(ref, abs=1e-8 * (1 + abs(ref)))


def test_singular_innovation_covariance():
    """Two identical noiseless series: F is singular, the pseudo-inverse is used."""
    ssm = StateSpace([[0.6]], [[1.0], [1.0]], [[1.0]], [0.0, 0.0])
    y = np.array([[1.0, 1.0], [np.nan, np.nan], [0.5, 0.5]])
    pattern = ~np.isnan(y)
    out = batched_functional(ssm, np.ones(1), 0.0, pattern, np.nan_to_num(y)[:, :, None], 1)
    single = StateSpace([[0.6]], [[1.0]], [[1.0]], [0.0])
    ref = kalman_smoother(single, y[:, :1]).smoothed_state[1, 0]
    assert out[0] == pytest.approx(ref, abs=1e-10)


def test_errors():
    ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
    data = np.zeros((3, 1, 1))
    pattern = np.ones((3, 1), bool)
    with pytest.raises(ValueError, match="shapes"):
        batched_functional(ssm, np.ones(1), 0.0, pattern[:2], data, 0)
    with pytest.raises(ValueError, match="position"):
        batched_functional(ssm, np.ones(1), 0.0, pattern, data, 3)


@pytest.mark.parametrize("diagonal", [True, False])
def test_time_varying_observation_equation(diagonal):
    base = random_state_space(3, 2, diagonal_obs_cov=diagonal, intercepts=True, random_state=4)
    other = random_state_space(3, 2, diagonal_obs_cov=diagonal, intercepts=True, random_state=5)
    rng = np.random.default_rng(6)
    n_periods = 30
    index = rng.integers(0, 2, n_periods)
    tv = StateSpace(
        base.T,
        np.stack([base.Z, other.Z]),
        base.Q,
        np.stack([base.H, other.H]),
        selection=base.R,
        obs_intercept=np.stack([base.d, other.d]),
        initial_state_cov=base.P0,
        obs_index=index,
    )
    y = random_missing(rng.standard_normal((n_periods, 3)), 0.3, random_state=7)
    pattern = ~np.isnan(y)
    gain = np.array([0.7, -0.4])
    out = batched_functional(tv, gain, 0.5, pattern, np.nan_to_num(y)[:, :, None], 12)
    ref = gain @ kalman_smoother(tv, y).smoothed_state[12] + 0.5
    assert out[0] == pytest.approx(ref, abs=1e-10)
    with pytest.raises(ValueError, match="30 periods"):
        batched_functional(tv, gain, 0.0, pattern[:5], np.zeros((5, 3, 1)), 0)


def test_simulated_data(rng):
    ssm = random_state_space(3, 2, random_state=1)
    y, _ = simulate_state_space(ssm, 40, random_state=2)
    y = random_missing(y, 0.3, random_state=3)
    pattern = ~np.isnan(y)
    out = batched_functional(
        ssm, np.array([1.0, -1.0]), 0.0, pattern, np.nan_to_num(y)[:, :, None], 39
    )
    ref = np.array([1.0, -1.0]) @ kalman_smoother(ssm, y).smoothed_state[39]
    assert out[0] == pytest.approx(ref, abs=1e-9)

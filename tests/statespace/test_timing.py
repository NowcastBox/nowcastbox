"""Simple timing test of the univariate and collapsed filters on a large panel (I2)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from nowcastbox.statespace import (
    StateSpace,
    companion_matrix,
    kalman_smoother,
    random_missing,
    simulate_state_space,
)


@pytest.mark.slow
def test_large_panel_timing(timing_reliable):
    rng = np.random.default_rng(0)
    r, n, n_periods = 3, 200, 300
    t_mat = companion_matrix([np.diag([0.7, 0.5, 0.3])], n_lags=5)
    sel = np.zeros((5 * r, r))
    sel[:r] = np.eye(r)
    z = np.zeros((n, 5 * r))
    z[:, :r] = rng.normal(size=(n, r))
    ssm = StateSpace(t_mat, z, np.eye(r), rng.uniform(0.2, 1.0, n), selection=sel)
    y, _ = simulate_state_space(ssm, n_periods, random_state=1)
    y[:24, :20] = np.nan  # late starters
    y[-6:, : n // 2] = np.nan  # ragged edge
    y_scattered = random_missing(y, 0.05, random_state=2)

    kalman_smoother(ssm, y[:5])  # compile
    kalman_smoother(ssm, y[:5], collapse=True)

    start = time.perf_counter()
    uni = kalman_smoother(ssm, y)
    t_uni = time.perf_counter() - start
    start = time.perf_counter()
    col = kalman_smoother(ssm, y, collapse=True)
    t_col = time.perf_counter() - start

    np.testing.assert_allclose(col.smoothed_state, uni.smoothed_state, atol=1e-8)
    assert col.loglikelihood == pytest.approx(uni.loglikelihood, abs=1e-6)
    if timing_reliable:
        assert t_uni < 10.0
        assert t_col < 10.0
    # arbitrary (scattered) missing patterns: one collapse per distinct pattern
    start = time.perf_counter()
    scattered = kalman_smoother(ssm, y_scattered, collapse=True)
    t_scattered = time.perf_counter() - start
    if timing_reliable:
        assert t_scattered < 30.0
    reference = kalman_smoother(ssm, y_scattered)
    np.testing.assert_allclose(scattered.smoothed_state, reference.smoothed_state, atol=1e-8)

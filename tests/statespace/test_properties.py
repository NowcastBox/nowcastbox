"""Property-based tests (hypothesis) for the statespace module."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nowcastbox.statespace import (
    kalman_filter,
    kalman_smoother,
    random_state_space,
    simulate_state_space,
)

pytestmark = pytest.mark.property


@st.composite
def models_with_missing(draw, diagonal=True):
    seed = draw(st.integers(0, 2**31 - 1))
    n_obs = draw(st.integers(1, 6))
    n_states = draw(st.integers(1, 4))
    n_periods = draw(st.integers(1, 25))
    frac = draw(st.floats(0.0, 0.9))
    rng = np.random.default_rng(seed)
    ssm = random_state_space(
        n_obs, n_states, diagonal_obs_cov=diagonal, intercepts=bool(seed % 2), random_state=rng
    )
    y, _ = simulate_state_space(ssm, n_periods, random_state=rng)
    y[rng.random(y.shape) < frac] = np.nan
    return ssm, y


@given(models_with_missing())
def test_univariate_equals_multivariate(case):
    ssm, y = case
    uni = kalman_smoother(ssm, y, method="univariate")
    multi = kalman_smoother(ssm, y, method="multivariate")
    assert uni.loglikelihood == pytest.approx(multi.loglikelihood, rel=1e-9, abs=1e-8)
    np.testing.assert_allclose(
        uni.filter_result.filtered_state, multi.filter_result.filtered_state, atol=1e-8
    )
    np.testing.assert_allclose(uni.smoothed_state, multi.smoothed_state, atol=1e-8)
    np.testing.assert_allclose(uni.smoothed_state_cov, multi.smoothed_state_cov, atol=1e-8)


@given(models_with_missing(), st.randoms(use_true_random=False))
def test_invariant_to_series_order(case, rnd):
    ssm, y = case
    perm = list(range(ssm.n_obs))
    rnd.shuffle(perm)
    permuted = ssm.replace(design=ssm.Z[perm], obs_cov=ssm.H[perm], obs_intercept=ssm.d[perm])
    a = kalman_filter(ssm, y)
    b = kalman_filter(permuted, y[:, perm])
    assert a.loglikelihood == pytest.approx(b.loglikelihood, rel=1e-9, abs=1e-8)
    np.testing.assert_allclose(a.filtered_state, b.filtered_state, atol=1e-8)


@given(models_with_missing())
def test_smoother_equals_filter_at_last_period_and_psd(case):
    ssm, y = case
    res = kalman_smoother(ssm, y)
    np.testing.assert_allclose(
        res.smoothed_state[-1], res.filter_result.filtered_state[-1], atol=1e-9
    )
    for mat in res.smoothed_state_cov:
        assert np.linalg.eigvalsh(mat).min() > -1e-9
    assert np.isfinite(res.loglikelihood)


@given(models_with_missing(diagonal=False))
def test_collapse_equals_multivariate_full_h(case):
    ssm, y = case
    full = kalman_smoother(ssm, y)
    col = kalman_smoother(ssm, y, collapse=True)
    assert col.loglikelihood == pytest.approx(full.loglikelihood, rel=1e-9, abs=1e-7)
    np.testing.assert_allclose(col.smoothed_state, full.smoothed_state, atol=1e-7)

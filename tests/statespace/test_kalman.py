"""Tests for nowcastbox.statespace.kalman (filter, log-likelihood)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.statespace import (
    StateSpace,
    kalman_filter,
    loglikelihood,
    prepare_observations,
    random_missing,
    random_state_space,
    simulate_state_space,
)
from tests.statespace._reference import naive_filter, statsmodels_smoother

TOL = 1e-8


def _model_and_data(seed, n_obs=6, n_states=3, n_periods=60, frac=0.35, **kw):
    ssm = random_state_space(n_obs, n_states, random_state=seed, **kw)
    y, _ = simulate_state_space(ssm, n_periods, random_state=seed + 100)
    y = random_missing(y, frac, random_state=seed + 200)
    y[5] = np.nan  # a fully missing period
    return ssm, y


@pytest.mark.reference_validation
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize(
    ("method", "collapse"), [("univariate", False), ("multivariate", False), ("univariate", True)]
)
def test_matches_statsmodels_diagonal(seed, method, collapse):
    ssm, y = _model_and_data(seed, n_disturbances=2 if seed % 2 else None, intercepts=seed % 3 == 0)
    ref = statsmodels_smoother(ssm, y)
    res = kalman_filter(ssm, y, method=method, collapse=collapse)
    np.testing.assert_allclose(res.loglikelihood_obs, ref.llf_obs, atol=TOL, rtol=0)
    np.testing.assert_allclose(res.filtered_state, ref.filtered_state.T, atol=TOL, rtol=0)
    np.testing.assert_allclose(
        res.filtered_state_cov, ref.filtered_state_cov.transpose(2, 0, 1), atol=TOL, rtol=0
    )
    np.testing.assert_allclose(res.predicted_state, ref.predicted_state.T, atol=TOL, rtol=0)
    np.testing.assert_allclose(
        res.predicted_state_cov, ref.predicted_state_cov.transpose(2, 0, 1), atol=TOL, rtol=0
    )
    np.testing.assert_allclose(res.forecasts(), ref.forecasts.T, atol=TOL, rtol=0)
    obs = ~np.isnan(y)
    np.testing.assert_allclose(res.forecast_errors()[obs], ref.forecasts_error.T[obs], atol=TOL)


@pytest.mark.reference_validation
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(("method", "collapse"), [("multivariate", False), ("auto", True)])
def test_matches_statsmodels_full_h(seed, method, collapse):
    ssm, y = _model_and_data(seed, diagonal_obs_cov=False, intercepts=True)
    ref = statsmodels_smoother(ssm, y)
    res = kalman_filter(ssm, y, method=method, collapse=collapse)
    np.testing.assert_allclose(res.loglikelihood, ref.llf_obs.sum(), atol=TOL, rtol=0)
    np.testing.assert_allclose(res.filtered_state, ref.filtered_state.T, atol=TOL, rtol=0)
    np.testing.assert_allclose(
        res.filtered_state_cov, ref.filtered_state_cov.transpose(2, 0, 1), atol=TOL, rtol=0
    )


def test_matches_naive_filter(rng):
    ssm, y = _model_and_data(11, n_obs=4, n_states=2)
    a_pred, p_pred, a_filt, p_filt, ll = naive_filter(ssm, y)
    res = kalman_filter(ssm, y)
    np.testing.assert_allclose(res.predicted_state, a_pred, atol=1e-10)
    np.testing.assert_allclose(res.predicted_state_cov, p_pred, atol=1e-10)
    np.testing.assert_allclose(res.filtered_state, a_filt, atol=1e-10)
    np.testing.assert_allclose(res.filtered_state_cov, p_filt, atol=1e-10)
    assert res.loglikelihood == pytest.approx(ll, abs=1e-9)


def test_local_level_closed_form():
    """Local level model: a_{t|t} = a_t + P_t/(P_t+h)(y_t-a_t), P_{t+1} = P_t h/(P_t+h)+q."""
    h, q, p1 = 2.0, 0.5, 3.0
    ssm = StateSpace([[1.0]], [[1.0]], [[q]], [h], initial_state=[1.0], initial_state_cov=[[p1]])
    y = np.array([2.0, np.nan, -1.0, 0.5])
    a, p, ll = 1.0, p1, 0.0
    exp_filt = []
    for yt in y:
        if not np.isnan(yt):
            f = p + h
            ll += -0.5 * (np.log(2 * np.pi * f) + (yt - a) ** 2 / f)
            a = a + p / f * (yt - a)
            p = p * h / f
        exp_filt.append((a, p))
        p = p + q
    res = kalman_filter(ssm, y)
    np.testing.assert_allclose(res.filtered_state[:, 0], [e[0] for e in exp_filt])
    np.testing.assert_allclose(res.filtered_state_cov[:, 0, 0], [e[1] for e in exp_filt])
    assert res.loglikelihood == pytest.approx(ll)
    assert res.predicted_state_cov[-1, 0, 0] == pytest.approx(p)


def test_all_missing_is_pure_prediction():
    ssm = random_state_space(3, 2, random_state=3)
    y = np.full((5, 3), np.nan)
    res = kalman_filter(ssm, y)
    assert res.loglikelihood == 0.0
    np.testing.assert_array_equal(res.n_observed, np.zeros(5))
    p = ssm.P0.copy()
    for t in range(5):
        np.testing.assert_allclose(res.filtered_state_cov[t], p, atol=1e-12)
        p = ssm.T @ p @ ssm.T.T + ssm.state_disturbance_cov
    for method in ("univariate", "multivariate"):
        assert kalman_filter(ssm, y, method=method).loglikelihood == 0.0
    assert kalman_filter(ssm, y, collapse=True).loglikelihood == 0.0


def test_zero_variance_perfectly_predicted_observation_is_skipped():
    # second series is an exact copy of the first: after observing y1 its F is 0
    ssm = StateSpace([[0.5]], [[1.0], [1.0]], [[1.0]], [0.0, 0.0])
    y = np.array([[0.7, 0.7], [0.2, 0.2]])
    res = kalman_filter(ssm, y, method="univariate")
    single = kalman_filter(
        ssm.replace(design=[[1.0], [1.0]]), np.column_stack([y[:, 0], [np.nan] * 2])
    )
    np.testing.assert_allclose(res.filtered_state, single.filtered_state)
    assert res.loglikelihood == pytest.approx(single.loglikelihood)


def test_multivariate_not_positive_definite_raises():
    ssm = StateSpace([[0.5]], [[1.0], [1.0]], [[1.0]], [0.0, 0.0])
    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        kalman_filter(ssm, np.array([[0.7, 0.7]]), method="multivariate")


def test_n_observed_and_properties():
    ssm, y = _model_and_data(2)
    res = kalman_filter(ssm, y)
    np.testing.assert_array_equal(res.n_observed, (~np.isnan(y)).sum(axis=1))
    assert res.n_periods == y.shape[0]
    assert res.n_states == ssm.n_states
    assert res.has_smoother_quantities
    assert res.method == "univariate"
    assert not res.collapsed
    assert "method='univariate'" in repr(res)


def test_forecast_cov():
    ssm, y = _model_and_data(4, diagonal_obs_cov=False)
    res = kalman_filter(ssm, y)
    expected = ssm.Z @ res.predicted_state_cov[3] @ ssm.Z.T + ssm.H
    np.testing.assert_allclose(res.forecast_cov(3), expected)
    np.testing.assert_allclose(res.forecast_cov(-1), res.forecast_cov(y.shape[0] - 1))
    with pytest.raises(IndexError):
        res.forecast_cov(y.shape[0])


def test_store_smoother_false():
    ssm, y = _model_and_data(5)
    res = kalman_filter(ssm, y, store_smoother=False)
    assert res.smoother_score is None
    assert not res.has_smoother_quantities
    assert res.loglikelihood == pytest.approx(kalman_filter(ssm, y).loglikelihood)


def test_loglikelihood_function():
    ssm, y = _model_and_data(6)
    ll = loglikelihood(ssm, y)
    assert ll == pytest.approx(kalman_filter(ssm, y).loglikelihood, abs=1e-12)
    assert loglikelihood(ssm, y, method="multivariate") == pytest.approx(ll, abs=1e-9)
    assert loglikelihood(ssm, y, collapse=True) == pytest.approx(ll, abs=1e-9)


def test_auto_method_selection():
    diag, y = _model_and_data(7)
    full, y2 = _model_and_data(7, diagonal_obs_cov=False)
    assert kalman_filter(diag, y).method == "univariate"
    assert kalman_filter(full, y2).method == "multivariate"
    assert kalman_filter(full, y2, collapse=True).method == "univariate"


def test_method_errors():
    full, y = _model_and_data(8, diagonal_obs_cov=False)
    with pytest.raises(ValueError, match="diagonal obs_cov"):
        kalman_filter(full, y, method="univariate")
    with pytest.raises(ValueError, match="method must be one of"):
        kalman_filter(full, y, method="fast")
    with pytest.raises(ValueError, match="collapse=True"):
        kalman_filter(full, y, method="multivariate", collapse=True)


class TestPrepareObservations:
    def test_dataframe_and_1d(self):
        ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        out = prepare_observations(ssm, df)
        assert out.shape == (3, 1)
        assert np.isnan(out[1, 0])
        assert prepare_observations(ssm, pd.Series([1.0, 2.0])).shape == (2, 1)
        assert prepare_observations(ssm, [1.0, 2.0]).shape == (2, 1)

    def test_nullable_dataframe(self):
        ssm = StateSpace([[0.5]], [[1.0], [1.0]], [[1.0]], [1.0, 1.0])
        df = pd.DataFrame({"a": pd.array([1, None], dtype="Int64"), "b": [1.0, 2.0]})
        out = prepare_observations(ssm, df)
        assert np.isnan(out[1, 0])

    def test_copy(self):
        ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
        y = np.ones((3, 1))
        out = prepare_observations(ssm, y)
        out[0, 0] = 5.0
        assert y[0, 0] == 1.0

    @pytest.mark.parametrize(
        ("y", "match"),
        [
            (np.ones((3, 2)), "shape"),
            (np.ones((0, 1)), "at least one period"),
            (np.array([[np.inf]]), "infinite"),
            (np.array([["a"]]), "numeric"),
            (np.ones((2, 1, 1)), "shape"),
        ],
    )
    def test_errors(self, y, match):
        ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
        with pytest.raises(NowcastDataError, match=match):
            prepare_observations(ssm, y)

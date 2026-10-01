"""Time-varying observation equations (``Z_t``, ``H_t``, ``d_t`` through a store + index)."""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.statespace import (
    StateSpace,
    collapse_observations,
    kalman_filter,
    kalman_smoother,
    loglikelihood,
    simulate_state_space,
)

pytestmark = pytest.mark.reference_validation


def _tv_model(rng, n_periods=30, full_h=False, n_obs=4, m=3, k=3):
    t_mat = np.diag(rng.uniform(0.2, 0.8, m))
    t_mat[0, 1] = 0.2
    zs = rng.normal(size=(k, n_obs, m))
    if full_h:
        hs = np.stack(
            [a @ a.T / n_obs + 0.3 * np.eye(n_obs) for a in rng.normal(size=(k, n_obs, n_obs))]
        )
    else:
        hs = rng.uniform(0.2, 1.0, size=(k, n_obs))
    ds = rng.normal(size=(k, n_obs))
    index = rng.integers(0, k, n_periods)
    return StateSpace(
        t_mat, zs, np.eye(m), hs, obs_intercept=ds, state_intercept=rng.normal(size=m) * 0.1,
        obs_index=index,
    )  # fmt: skip


def _statsmodels(model: StateSpace, y: np.ndarray):
    n = y.shape[0]
    idx = model.obs_index
    ks = KalmanSmoother(model.n_obs, model.n_states, model.n_disturbances)
    ks.bind(np.asarray(y, dtype=float).copy())
    ks["design"] = np.moveaxis(model.design_store[idx], 0, -1).copy()
    ks["obs_intercept"] = model.obs_intercept_store[idx].T.copy()
    ks["obs_cov"] = np.moveaxis(model.obs_cov_matrix_store[idx], 0, -1).copy()
    ks["transition"] = model.T
    ks["state_intercept"] = model.c
    ks["selection"] = model.R
    ks["state_cov"] = model.Q
    ks.initialize_known(np.array(model.a0), np.array(model.P0))
    assert n == idx.size
    return ks.smooth()


class TestRepresentation:
    def test_properties(self, rng):
        ssm = _tv_model(rng)
        assert ssm.is_time_varying and ssm.n_periods == 30
        assert ssm.design_store.shape == (3, 4, 3)
        assert ssm.obs_cov_store.shape == (3, 4)
        assert ssm.obs_cov_matrix_store.shape == (3, 4, 4)
        assert ssm.obs_cov_diagonal_store.shape == (3, 4)
        assert ssm.obs_intercept_store.shape == (3, 4)
        t = 7
        k = int(ssm.obs_index[t])
        np.testing.assert_array_equal(ssm.design_at(t), ssm.design_store[k])
        np.testing.assert_array_equal(ssm.obs_cov_at(t), np.diag(ssm.obs_cov_store[k]))
        np.testing.assert_array_equal(ssm.obs_intercept_at(t), ssm.obs_intercept_store[k])
        assert ssm.designs(30).shape == (30, 4, 3)
        assert "time_varying=3" in repr(ssm)
        assert ssm.obs_cov_is_diagonal

    def test_invariant_accessors_raise(self, rng):
        ssm = _tv_model(rng)
        for name in ("Z", "H", "d", "obs_cov_diagonal", "obs_cov_matrix"):
            with pytest.raises(ValueError, match="time varying"):
                getattr(ssm, name)

    def test_invariant_model_stores(self):
        ssm = StateSpace([[0.5]], [[1.0], [2.0]], [[1.0]], [[1.0, 0.2], [0.2, 1.0]])
        assert not ssm.is_time_varying and ssm.obs_index is None and ssm.n_periods is None
        assert ssm.design_store.shape == (1, 2, 1)
        np.testing.assert_array_equal(ssm.obs_cov_diagonal_store, [[1.0, 1.0]])
        np.testing.assert_array_equal(ssm.period_index(3), [0, 0, 0])
        assert ssm.designs(4).shape == (4, 2, 1)
        assert not ssm.obs_cov_is_diagonal

    def test_period_errors(self, rng):
        ssm = _tv_model(rng)
        with pytest.raises(ValueError, match="30 periods"):
            ssm.period_index(10)
        with pytest.raises(IndexError):
            ssm.design_at(30)
        assert ssm.design_at(-1).shape == (4, 3)

    def test_roundtrip_and_replace(self, rng):
        ssm = _tv_model(rng)
        params = ssm.to_dict()
        assert params["design"].ndim == 3 and "obs_index" in params
        clone = ssm.replace()
        np.testing.assert_array_equal(clone.obs_index, ssm.obs_index)
        inv = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
        tv = inv.replace(
            design=[[[1.0]], [[2.0]]],
            obs_cov=[[1.0], [1.0]],
            obs_intercept=[[0.0], [1.0]],
            obs_index=[1, 0],
        )
        np.testing.assert_array_equal(tv.design_at(0), [[2.0]])

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"design": [[1.0]]}, "dimension"),
            ({"obs_index": [[0]]}, "1-D"),
            ({"obs_index": []}, "1-D"),
            ({"obs_index": [0.5]}, "1-D"),
            ({"obs_index": [2]}, r"\[0, 2\)"),
            ({"obs_index": [-1]}, r"\[0, 2\)"),
            ({"obs_cov": [[1.0], [1.0], [1.0]]}, "obs_cov"),
            ({"obs_cov": [[-1.0], [1.0]]}, "non-negative"),
            ({"obs_cov": [[[1.0]], [[1.0]], [[1.0]]]}, "obs_cov"),
            ({"obs_intercept": [0.0, 1.0]}, "dimension"),
            ({"design": np.zeros((0, 1, 1))}, "at least one design"),
        ],
    )
    def test_validation(self, kwargs, match):
        base = {
            "transition": [[0.5]],
            "design": [[[1.0]], [[2.0]]],
            "state_cov": [[1.0]],
            "obs_cov": [[1.0], [1.0]],
            "obs_index": [0, 1, 1],
        }
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            StateSpace(
                base.pop("transition"), base.pop("design"), base.pop("state_cov"),
                base.pop("obs_cov"), **base,
            )  # fmt: skip

    def test_full_covariance_store(self, rng):
        ssm = _tv_model(rng, full_h=True)
        assert ssm.obs_cov_store.ndim == 3
        assert not ssm.obs_cov_is_diagonal
        np.testing.assert_allclose(
            ssm.obs_cov_diagonal_store, np.diagonal(ssm.obs_cov_store, axis1=1, axis2=2)
        )

    def test_no_observed_series(self):
        with pytest.raises(ValueError, match="at least one observed"):
            StateSpace([[0.5]], np.zeros((1, 0, 1)), [[1.0]], np.zeros((1, 0)), obs_index=[0])


class TestFiltering:
    @pytest.mark.parametrize("full_h", [False, True])
    def test_matches_statsmodels(self, rng, full_h):
        ssm = _tv_model(rng, full_h=full_h)
        y, _ = simulate_state_space(ssm, 30, random_state=rng)
        y[rng.random(y.shape) < 0.2] = np.nan
        y[4] = np.nan
        ours = kalman_smoother(ssm, y)
        ref = _statsmodels(ssm, y)
        assert ours.loglikelihood == pytest.approx(float(np.sum(ref.llf_obs)), rel=1e-9)
        np.testing.assert_allclose(ours.smoothed_state, ref.smoothed_state.T, atol=1e-8)
        np.testing.assert_allclose(
            ours.smoothed_state_cov, np.moveaxis(ref.smoothed_state_cov, -1, 0), atol=1e-8
        )
        np.testing.assert_allclose(ours.filter_result.forecasts(), ref.forecasts.T, atol=1e-8)
        t = 5
        np.testing.assert_allclose(
            ours.filter_result.forecast_cov(t), ref.forecasts_error_cov[:, :, t], atol=1e-8
        )

    def test_collapse_matches(self, rng):
        ssm = _tv_model(rng, n_obs=12, m=2, full_h=True)
        y, _ = simulate_state_space(ssm, 30, random_state=rng)
        y[rng.random(y.shape) < 0.1] = np.nan
        col = collapse_observations(ssm, y)
        assert col.n_patterns >= 3
        a = kalman_smoother(ssm, y, collapse=True)
        b = kalman_smoother(ssm, y)
        np.testing.assert_allclose(a.smoothed_state, b.smoothed_state, atol=1e-8)
        assert loglikelihood(ssm, y, collapse=True) == pytest.approx(b.loglikelihood, rel=1e-9)

    def test_smoothed_signal(self, rng):
        ssm = _tv_model(rng)
        y, _ = simulate_state_space(ssm, 30, random_state=rng)
        res = kalman_smoother(ssm, y)
        expected = np.einsum("tij,tj->ti", ssm.designs(30), res.smoothed_state)
        expected += ssm.obs_intercept_store[ssm.obs_index]
        np.testing.assert_allclose(res.smoothed_signal(), expected)
        z = ssm.design_at(3)
        np.testing.assert_allclose(res.smoothed_signal_cov(3), z @ res.smoothed_state_cov[3] @ z.T)
        np.testing.assert_allclose(res.smoothed_signal_cov(-1), res.smoothed_signal_cov(29))

    def test_wrong_number_of_periods(self, rng):
        ssm = _tv_model(rng)
        with pytest.raises(NowcastDataError, match="time-varying"):
            kalman_filter(ssm, np.zeros((10, 4)))
        with pytest.raises(ValueError, match="30 periods"):
            simulate_state_space(ssm, 10)

    def test_simulation_uses_period_systems(self):
        zs = np.array([[[1.0]], [[0.0]]])
        ssm = StateSpace(
            [[0.5]],
            zs,
            [[1.0]],
            [[0.0], [0.0]],
            obs_intercept=[[0.0], [5.0]],
            obs_index=[0, 1, 0, 1],
        )
        y, states = simulate_state_space(ssm, 4, random_state=0)
        np.testing.assert_allclose(y[[0, 2], 0], states[[0, 2], 0])
        np.testing.assert_allclose(y[[1, 3], 0], 5.0)
        full = StateSpace([[0.5]], [[[1.0]]], [[1.0]], [[[0.0]]], obs_index=[0, 0])
        y2, s2 = simulate_state_space(full, 2, random_state=1)
        np.testing.assert_allclose(y2, s2)

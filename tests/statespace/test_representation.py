"""Tests for nowcastbox.statespace.representation."""

from __future__ import annotations

import numpy as np
import pytest

from nowcastbox.statespace import StateSpace, companion_matrix, stationary_initial_cov


def _basic(**kw):
    args = {
        "transition": [[0.5, 0.1], [0.0, 0.3]],
        "design": [[1.0, 0.0], [0.5, 1.0], [0.0, 2.0]],
        "state_cov": np.eye(2),
        "obs_cov": [0.5, 0.2, 0.1],
    }
    args.update(kw)
    return StateSpace(
        args.pop("transition"),
        args.pop("design"),
        args.pop("state_cov"),
        args.pop("obs_cov"),
        **args,
    )


class TestConstruction:
    def test_dimensions_and_defaults(self):
        ssm = _basic()
        assert (ssm.n_states, ssm.n_obs, ssm.n_disturbances) == (2, 3, 2)
        np.testing.assert_array_equal(ssm.R, np.eye(2))
        np.testing.assert_array_equal(ssm.c, np.zeros(2))
        np.testing.assert_array_equal(ssm.d, np.zeros(3))
        np.testing.assert_array_equal(ssm.a0, np.zeros(2))
        assert ssm.is_stationary
        assert ssm.obs_cov_is_diagonal
        # P0 solves the Lyapunov equation
        np.testing.assert_allclose(ssm.P0, ssm.T @ ssm.P0 @ ssm.T.T + ssm.state_disturbance_cov)

    def test_stationary_mean_with_intercept(self):
        ssm = _basic(state_intercept=[1.0, 2.0])
        np.testing.assert_allclose(ssm.a0, np.linalg.solve(np.eye(2) - ssm.T, [1.0, 2.0]))

    def test_arrays_are_read_only_copies(self):
        t_mat = np.array([[0.5]])
        ssm = StateSpace(t_mat, [[1.0]], [[1.0]], [1.0])
        t_mat[0, 0] = 0.9
        assert ssm.T[0, 0] == 0.5
        with pytest.raises(ValueError, match="read-only"):
            ssm.T[0, 0] = 0.1

    def test_selection_and_state_disturbance_cov(self):
        r_mat = np.array([[1.0], [0.0]])
        ssm = _basic(selection=r_mat, state_cov=[[2.0]])
        assert ssm.n_disturbances == 1
        np.testing.assert_allclose(ssm.state_disturbance_cov, [[2.0, 0.0], [0.0, 0.0]])

    def test_full_obs_cov(self):
        h = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 1.0]])
        ssm = _basic(obs_cov=h)
        assert not ssm.obs_cov_is_diagonal
        np.testing.assert_array_equal(ssm.obs_cov_matrix, h)
        np.testing.assert_array_equal(ssm.obs_cov_diagonal, np.ones(3))

    def test_full_but_diagonal_obs_cov_detected(self):
        ssm = _basic(obs_cov=np.diag([1.0, 2.0, 3.0]))
        assert ssm.obs_cov_is_diagonal
        np.testing.assert_array_equal(ssm.obs_cov_diagonal, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(_basic().obs_cov_matrix, np.diag([0.5, 0.2, 0.1]))

    def test_nonstationary_requires_p0(self):
        with pytest.raises(ValueError, match="initial_state_cov must be given"):
            StateSpace([[1.0]], [[1.0]], [[1.0]], [1.0])
        ssm = StateSpace([[1.0]], [[1.0]], [[1.0]], [1.0], initial_state_cov=[[10.0]])
        assert not ssm.is_stationary
        np.testing.assert_array_equal(ssm.a0, [0.0])
        with pytest.raises(ValueError, match="no stationary initialization"):
            ssm.with_stationary_initialization()

    def test_repr(self):
        assert "obs_cov=diagonal" in repr(_basic())
        assert "obs_cov=full" in repr(_basic(obs_cov=np.full((3, 3), 0.1) + np.eye(3)))


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"transition": [[0.5, 0.1]]}, "transition must have shape"),
            ({"transition": [0.5]}, "must have 2 dimension"),
            ({"transition": np.zeros((0, 0))}, "at least one state"),
            ({"design": np.zeros((0, 2))}, "at least one observed"),
            ({"design": [[1.0, 2.0, 3.0]]}, "design must have shape"),
            ({"transition": [[np.nan, 0], [0, 0.1]]}, "NaN or infinite"),
            ({"state_cov": [[1.0, 0.5], [0.0, 1.0]]}, "symmetric"),
            ({"state_cov": [[1.0, 0.0], [0.0, -1.0]]}, "positive semi-definite"),
            ({"state_cov": np.eye(3)}, "state_cov must have shape"),
            ({"obs_cov": [0.5, -0.2, 0.1]}, "non-negative"),
            ({"obs_cov": [0.5, 0.2]}, "obs_cov must have shape"),
            ({"obs_cov": np.ones((2, 2, 2))}, "1 or 2 dimension"),
            ({"selection": np.zeros((3, 1))}, "selection must have shape"),
            ({"selection": np.zeros((2, 0))}, "selection must have shape"),
            ({"state_intercept": [1.0]}, "state_intercept must have shape"),
            ({"obs_intercept": [1.0]}, "obs_intercept must have shape"),
            ({"initial_state": [1.0]}, "initial_state must have shape"),
            ({"initial_state_cov": np.eye(3)}, "initial_state_cov must have shape"),
            ({"initial_state_cov": -np.eye(2)}, "positive semi-definite"),
        ],
    )
    def test_invalid(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _basic(**kwargs)

    def test_non_numeric(self):
        with pytest.raises(TypeError, match="numeric"):
            _basic(transition=[["a", "b"], ["c", "d"]])


class TestDerivation:
    def test_to_dict_roundtrip(self):
        ssm = _basic(state_intercept=[0.1, 0.2], obs_intercept=[1.0, 2.0, 3.0])
        params = ssm.to_dict()
        clone = StateSpace(
            params.pop("transition"),
            params.pop("design"),
            params.pop("state_cov"),
            params.pop("obs_cov"),
            **params,
        )
        for key, value in clone.to_dict().items():
            np.testing.assert_array_equal(value, ssm.to_dict()[key])

    def test_replace(self):
        ssm = _basic()
        new = ssm.replace(obs_cov=[1.0, 1.0, 1.0])
        np.testing.assert_array_equal(new.H, np.ones(3))
        np.testing.assert_array_equal(ssm.H, [0.5, 0.2, 0.1])
        with pytest.raises(TypeError, match="unknown"):
            ssm.replace(foo=1)

    def test_with_stationary_initialization(self):
        ssm = _basic(initial_state=[5.0, 5.0], initial_state_cov=np.eye(2) * 100)
        stat = ssm.with_stationary_initialization()
        np.testing.assert_allclose(stat.P0, _basic().P0)
        np.testing.assert_allclose(stat.a0, np.zeros(2))

    def test_filter_and_smooth_shortcuts(self):
        ssm = _basic()
        y = np.ones((4, 3))
        assert ssm.filter(y).n_periods == 4
        assert ssm.smooth(y).smoothed_state.shape == (4, 2)


class TestHelpers:
    def test_stationary_initial_cov_ar1(self):
        np.testing.assert_allclose(stationary_initial_cov([[0.5]], [[0.75]]), [[1.0]])

    def test_stationary_initial_cov_errors(self):
        with pytest.raises(ValueError, match="not stationary"):
            stationary_initial_cov([[1.2]], [[1.0]])
        with pytest.raises(ValueError, match="shape"):
            stationary_initial_cov([[0.5, 0.0]], [[1.0]])
        with pytest.raises(ValueError, match="shape"):
            stationary_initial_cov([[0.5]], [[1.0, 0.0]])

    def test_companion_matrix(self):
        a1 = np.array([[0.5, 0.1], [0.0, 0.4]])
        a2 = np.array([[0.2, 0.0], [0.1, 0.1]])
        comp = companion_matrix([a1, a2])
        assert comp.shape == (4, 4)
        np.testing.assert_array_equal(comp[:2, :2], a1)
        np.testing.assert_array_equal(comp[:2, 2:], a2)
        np.testing.assert_array_equal(comp[2:, :2], np.eye(2))
        np.testing.assert_array_equal(comp[2:, 2:], np.zeros((2, 2)))
        padded = companion_matrix([a1], n_lags=5)
        assert padded.shape == (10, 10)
        np.testing.assert_array_equal(padded[:2, 2:], 0.0)
        np.testing.assert_array_equal(padded[2:, :8], np.eye(8))
        np.testing.assert_array_equal(companion_matrix([[[0.3]]]), [[0.3]])

    def test_companion_matrix_errors(self):
        with pytest.raises(ValueError, match="at least one"):
            companion_matrix([])
        with pytest.raises(ValueError, match="n_lags"):
            companion_matrix([np.eye(2), np.eye(2)], n_lags=1)
        with pytest.raises(ValueError, match="shape"):
            companion_matrix([np.eye(2), np.eye(3)])

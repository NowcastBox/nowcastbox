"""Structured (precision-based) smoother: exactness against the Kalman smoother."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.statespace import (
    SmoothedMoments,
    StateSpace,
    StructuredSmootherResult,
    detect_structure,
    kalman_smoother,
    random_state_space,
    smoothed_moments,
    structured_smoother,
)
from nowcastbox.statespace import structured as structured_mod

from ._dfm import dfm_state_space, ragged_panel


def _assert_matches(
    model: StateSpace, y: np.ndarray, tol: float = 1e-7
) -> StructuredSmootherResult:
    fast = structured_smoother(model, y)
    ref = kalman_smoother(model, y)
    assert fast.loglikelihood == pytest.approx(ref.loglikelihood, rel=1e-10, abs=1e-7)
    np.testing.assert_allclose(fast.smoothed_state, ref.smoothed_state, atol=tol, rtol=tol)
    for view, dense in (
        (fast.smoothed_state_cov, ref.smoothed_state_cov),
        (fast.smoothed_state_autocov, ref.smoothed_state_autocov),
    ):
        full = view.to_array()
        known = np.isfinite(full)
        assert known.any()
        np.testing.assert_allclose(full[known], dense[known], atol=tol, rtol=tol)
    return fast


class TestExactness:
    def test_toy_model(self):
        t_mat = np.diag([0.8, 0.4, -0.2])
        z = np.array([[1.0, 1.0, 0.0], [0.7, 0.0, 1.0]])
        ssm = StateSpace(t_mat, z, np.diag([1.0, 0.3, 0.3]), [1e-2, 1e-2])
        y = np.random.default_rng(0).standard_normal((40, 2))
        y[-3:, 0] = np.nan
        _assert_matches(ssm, y)

    @pytest.mark.parametrize("idio", ["ar1", "iid"])
    def test_mixed_frequency_dfm(self, rng, idio):
        ssm = dfm_state_space(12, 4, rng=rng, idiosyncratic=idio)
        y = ragged_panel(ssm, 90, rng, n_quarterly=4)
        res = _assert_matches(ssm, y)
        if idio == "iid":
            assert res.structure.n_private_groups == 0

    def test_var2_two_factors_two_blocks_with_intercepts(self, rng):
        ssm = dfm_state_space(
            9, 3, rng=rng, n_factors=2, factor_lags=2, n_blocks=2,
            intercepts=True, initial_mean=True,
        )  # fmt: skip
        y = ragged_panel(ssm, 60, rng, n_quarterly=3)
        _assert_matches(ssm, y)

    def test_larger_measurement_noise(self, rng):
        ssm = dfm_state_space(8, 2, rng=rng, obs_var=0.3)
        y = ragged_panel(ssm, 50, rng, n_quarterly=2)
        _assert_matches(ssm, y)

    def test_time_varying_observation_equation(self, rng):
        ssm = dfm_state_space(6, 2, rng=rng, n_patterns=3, n_periods=45, intercepts=True)
        y = ragged_panel(ssm, 45, rng, n_quarterly=2)
        _assert_matches(ssm, y)

    def test_without_common_states(self, rng):
        # two independent AR(1) signals, each observed by one series
        ssm = StateSpace(np.diag([0.5, -0.3]), np.eye(2), np.eye(2), [0.1, 0.2])
        y = rng.standard_normal((20, 2))
        y[5, :] = np.nan
        res = _assert_matches(ssm, y)
        assert res.structure.n_common_states == 0
        assert res.common_cov.shape == (20, 0, 0)

    def test_unobserved_block_and_empty_series(self, rng):
        # state 2 is loaded by nobody, series 2 loads nothing, series 1 is never observed
        t_mat = np.diag([0.6, 0.4, 0.9])
        z = np.array([[1.0, 1.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        ssm = StateSpace(t_mat, z, np.eye(3), [0.2, 0.3, 0.4, 0.5])
        y = rng.standard_normal((25, 4))
        y[:, 1] = np.nan
        y[10] = np.nan
        res = _assert_matches(ssm, y)
        owners = [g.owner for g in res.structure.groups]
        assert -1 in owners[1:]

    def test_common_chains_of_different_lengths(self, rng):
        # common group: a 3-lag factor chain and an AR(1) loaded by two series (padding)
        t_mat = np.zeros((5, 5))
        t_mat[:3, :3] = [[0.6, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        t_mat[3, 3] = 0.4
        t_mat[4, 4] = 0.2
        sel = np.zeros((5, 3))
        sel[0, 0] = sel[3, 1] = sel[4, 2] = 1.0
        z = np.array(
            [[1.0, 0.5, 0.2, 1.0, 0.0], [0.5, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0]]
        )
        ssm = StateSpace(t_mat, z, np.eye(3), [0.2, 0.3, 0.05], selection=sel)
        y = rng.standard_normal((25, 3))
        y[rng.random(y.shape) < 0.2] = np.nan
        res = _assert_matches(ssm, y)
        assert res.structure.groups[0].width == 2
        assert res.structure.groups[0].n_lags == 3

    def test_private_factor_block(self, rng):
        # a VAR(2) block loaded by a single series becomes part of its private group
        t_mat = np.zeros((4, 4))
        t_mat[0, 0] = 0.7
        t_mat[1:3, 1:3] = [[0.5, 0.2], [1.0, 0.0]]
        t_mat[3, 3] = 0.3
        sel = np.zeros((4, 3))
        sel[0, 0] = sel[1, 1] = sel[3, 2] = 1.0
        z = np.array([[1.0, 0.0, 0.0, 0.0], [0.8, 0.0, 0.0, 0.0], [0.5, 1.0, 0.5, 1.0]])
        ssm = StateSpace(t_mat, z, np.eye(3), [0.3, 0.3, 0.01], selection=sel)
        y = rng.standard_normal((30, 3))
        y[rng.random(y.shape) < 0.2] = np.nan
        res = _assert_matches(ssm, y)
        assert res.structure.series_group.tolist() == [-1, -1, 1]
        states, cov, autocov = res.group_moments(1)
        assert states.tolist() == [1, 2, 3]
        assert cov.shape == autocov.shape == (30, 3, 3)

    def test_single_period(self, rng):
        ssm = dfm_state_space(4, 0, rng=rng)
        y = rng.standard_normal((1, 4))
        _assert_matches(ssm, y)

    @settings(max_examples=25, deadline=None)
    @given(
        seed=st.integers(0, 10_000),
        n_monthly=st.integers(1, 5),
        n_quarterly=st.integers(0, 2),
        n_factors=st.integers(1, 2),
        factor_lags=st.integers(1, 2),
        n_periods=st.integers(2, 25),
        missing=st.floats(0.0, 0.6),
    )
    def test_property_matches_kalman(
        self, seed, n_monthly, n_quarterly, n_factors, factor_lags, n_periods, missing
    ):
        rng = np.random.default_rng(seed)
        ssm = dfm_state_space(
            n_monthly, n_quarterly, rng=rng, n_factors=n_factors, factor_lags=factor_lags,
            obs_var=float(rng.uniform(1e-3, 0.5)),
        )  # fmt: skip
        y = ragged_panel(ssm, n_periods, rng, n_quarterly=n_quarterly)
        y[rng.random(y.shape) < missing] = np.nan
        _assert_matches(ssm, y, tol=1e-6)


class TestResultAccess:
    @pytest.fixture
    def result(self, rng):
        ssm = dfm_state_space(5, 2, rng=rng)
        y = ragged_panel(ssm, 30, rng, n_quarterly=2)
        return structured_smoother(ssm, y), kalman_smoother(ssm, y)

    def test_orthogonal_indexing(self, result):
        fast, ref = result
        view = fast.smoothed_state_cov
        assert view.shape == (30, fast.model.n_states, fast.model.n_states)
        assert view.ndim == 3 and len(view) == 30
        rows = np.array([0, 1, 5])
        out = view[np.ix_([2, 3], rows, [0, 5])]
        np.testing.assert_allclose(out, ref.smoothed_state_cov[np.ix_([2, 3], rows, [0, 5])])
        np.testing.assert_allclose(view[4, 0, 0], ref.smoothed_state_cov[4, 0, 0])
        np.testing.assert_allclose(view[-1, :5, 5], ref.smoothed_state_cov[-1, :5, 5])
        mask = np.zeros(fast.model.n_states, dtype=bool)
        mask[[0, 6]] = True
        np.testing.assert_allclose(
            view[3, mask, mask], ref.smoothed_state_cov[3][np.ix_(mask, mask)]
        )
        np.testing.assert_allclose(view[1:3, :5], ref.smoothed_state_cov[1:3, :5, :])
        auto = fast.smoothed_state_autocov
        np.testing.assert_allclose(
            auto[np.ix_(range(30), [5, 6], [0, 1])],
            ref.smoothed_state_autocov[np.ix_(range(30), [5, 6], [0, 1])],
        )
        np.testing.assert_allclose(auto[:, :5, 5], ref.smoothed_state_autocov[:, :5, 5])

    def test_covariance_and_moment_sums(self, result):
        fast, ref = result
        np.testing.assert_allclose(
            fast.covariance([0, 1], [0, 5], [0, 5], lag=1),
            ref.smoothed_state_autocov[np.ix_([0, 1], [0, 5], [0, 5])],
        )
        with pytest.raises(ValueError, match="lag"):
            fast.covariance([0], [0], [0], lag=2)
        s11, _, _ = fast.moment_sums()
        assert s11.shape == (fast.model.n_states,) * 2

    def test_moment_sums_without_common_states(self, rng):
        ssm = StateSpace(np.diag([0.5, -0.3]), np.eye(2), np.eye(2), [0.1, 0.2])
        y = rng.standard_normal((20, 2))
        s11, s00, s10 = structured_smoother(ssm, y).moment_sums()
        ref = smoothed_moments(ssm, y, method="univariate")
        np.testing.assert_allclose(np.diag(s11), np.diag(ref.s11))
        np.testing.assert_allclose(np.diag(s10), np.diag(ref.s10))
        assert np.isnan(s00[0, 1])

    def test_cross_private_access_raises(self, result):
        fast, _ = result
        with pytest.raises(ValueError, match="private groups"):
            fast.smoothed_state_cov[0, 5, 6]
        with pytest.raises(ValueError, match="private groups"):
            fast.smoothed_state_cov[0]

    def test_index_errors(self, result):
        fast, _ = result
        view = fast.smoothed_state_cov
        with pytest.raises(IndexError):
            view[100, 0, 0]
        with pytest.raises(IndexError):
            view[0, [0, 999], 0]
        with pytest.raises(IndexError, match="too many"):
            view[0, 0, 0, 0]
        with pytest.raises(IndexError):
            fast.group_moments(0)
        assert view[0, np.array([], dtype=int), 0].shape == (0,)

    def test_signal_and_variance(self, result):
        fast, ref = result
        np.testing.assert_allclose(fast.smoothed_signal(), ref.smoothed_signal(), atol=1e-9)
        expected = np.stack([np.diag(ref.smoothed_signal_cov(t)) for t in range(30)])
        np.testing.assert_allclose(fast.smoothed_signal_variance(), expected, atol=1e-9)
        np.testing.assert_allclose(ref.smoothed_signal_variance(), expected, atol=1e-12)
        assert fast.n_periods == 30

    def test_signal_variance_with_unloaded_series(self, rng):
        ssm = StateSpace(np.diag([0.5, 0.2]), [[1.0, 1.0], [0.0, 0.0]], np.eye(2), [1.0, 1.0])
        res = structured_smoother(ssm, rng.standard_normal((5, 2)))
        np.testing.assert_array_equal(res.smoothed_signal_variance()[:, 1], 0.0)

    def test_precomputed_structure(self, rng):
        ssm = dfm_state_space(4, 1, rng=rng)
        y = ragged_panel(ssm, 20, rng, n_quarterly=1)
        st_ = detect_structure(ssm)
        a = structured_smoother(ssm, y, structure=st_)
        b = structured_smoother(ssm, y)
        np.testing.assert_allclose(a.smoothed_state, b.smoothed_state)
        other = dfm_state_space(2, 0, rng=rng)
        with pytest.raises(ValueError, match="number of states"):
            structured_smoother(ssm, y, structure=detect_structure(other))


class TestErrors:
    def test_unstructured_model(self, rng):
        ssm = random_state_space(3, 2, diagonal_obs_cov=False, random_state=1)
        with pytest.raises(ValueError, match="diagonal"):
            structured_smoother(ssm, rng.standard_normal((5, 3)))

    def test_bad_observations(self, rng):
        ssm = dfm_state_space(3, 0, rng=rng)
        with pytest.raises(NowcastDataError):
            structured_smoother(ssm, np.ones((5, 4)))

    def test_group_not_positive_definite(self, rng, monkeypatch):
        ssm = dfm_state_space(3, 0, rng=rng)
        monkeypatch.setattr(structured_mod._bk, "group_eliminate", lambda *a: np.nan)
        with pytest.raises(np.linalg.LinAlgError, match="group"):
            structured_smoother(ssm, rng.standard_normal((5, 3)))

    def test_common_not_positive_definite(self, rng, monkeypatch):
        ssm = dfm_state_space(3, 0, rng=rng)

        def boom(*args, **kwargs):
            raise np.linalg.LinAlgError("not pd")

        monkeypatch.setattr(structured_mod.scipy.linalg, "cho_factor", boom)
        with pytest.raises(np.linalg.LinAlgError, match="common"):
            structured_smoother(ssm, rng.standard_normal((5, 3)))


class TestSmoothedMoments:
    @pytest.mark.parametrize("idio", ["ar1", "iid"])
    def test_structured_equals_dense(self, rng, idio):
        ssm = dfm_state_space(8, 2, rng=rng, n_factors=2, idiosyncratic=idio)
        y = ragged_panel(ssm, 40, rng, n_quarterly=2)
        fast = smoothed_moments(ssm, y, method="structured")
        ref = smoothed_moments(ssm, y, method="univariate")
        assert isinstance(fast, SmoothedMoments)
        assert fast.method == "structured" and ref.method == "univariate"
        assert fast.n_pairs == ref.n_pairs == 39
        assert fast.loglikelihood == pytest.approx(ref.loglikelihood, rel=1e-10)
        for a, b in ((fast.s11, ref.s11), (fast.s00, ref.s00), (fast.s10, ref.s10)):
            known = np.isfinite(a)
            np.testing.assert_allclose(a[known], b[known], rtol=1e-8, atol=1e-8)
        if idio == "ar1":
            assert not np.isfinite(fast.s11).all()  # cross-private entries are not formed

    def test_auto_prefers_structured_for_large_panels(self, rng):
        ssm = dfm_state_space(30, 5, rng=rng)
        y = ragged_panel(ssm, 40, rng, n_quarterly=5)
        assert smoothed_moments(ssm, y).method == "structured"

    def test_auto_falls_back(self, rng):
        ssm = random_state_space(3, 2, diagonal_obs_cov=False, random_state=2)
        y = rng.standard_normal((10, 3))
        assert smoothed_moments(ssm, y).method == "multivariate"
        diag = random_state_space(3, 2, random_state=2)
        assert smoothed_moments(diag, y).method == "univariate"
        mom = smoothed_moments(diag, y, method="multivariate")
        ref = kalman_smoother(diag, y, method="multivariate")
        np.testing.assert_allclose(mom.smoother.smoothed_state, ref.smoothed_state)

    def test_errors(self, rng):
        ssm = dfm_state_space(3, 0, rng=rng)
        with pytest.raises(ValueError, match="method"):
            smoothed_moments(ssm, np.ones((5, 3)), method="fast")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="two periods"):
            smoothed_moments(ssm, np.ones((1, 3)))
        bad = random_state_space(3, 2, diagonal_obs_cov=False, random_state=3)
        with pytest.raises(ValueError, match="diagonal"):
            smoothed_moments(bad, np.ones((5, 3)), method="structured")


class TestEMDropIn:
    """The moments plug into the (unchanged) M-step of MixedFreqDFM."""

    @pytest.mark.parametrize("idio", ["ar1", "iid"])
    def test_m_step_identical(self, rng, idio):
        from nowcastbox.models._em_steps import (
            EMParameters,
            StateLayout,
            build_state_space,
            e_step,
            m_step,
        )

        n_m, n_q = 6, 2
        series = [f"m{i}" for i in range(n_m)] + [f"q{i}" for i in range(n_q)]
        weights = [[1.0]] * n_m + [[1.0, 2.0, 3.0, 2.0, 1.0]] * n_q
        membership = np.ones((n_m + n_q, 2), dtype=bool)
        membership[: n_m // 2, 1] = False
        lay = StateLayout(series, ["g", "r"], [1, 1], 2, membership, weights, idio)
        n_series = n_m + n_q
        loadings = np.zeros((n_series, lay.n_factor_states))
        for i in range(n_series):
            idx = lay.loading_index(i)
            w = np.repeat(np.asarray(weights[i]), len(lay.series_blocks(i)))
            loadings[i, idx] = (
                w
                * rng.uniform(0.3, 1.0, len(lay.series_blocks(i)))
                .repeat(len(weights[i]))
                .reshape(len(lay.series_blocks(i)), -1)
                .T.ravel()
            )
        params = EMParameters(
            (np.array([[0.5, 0.1]]), np.array([[0.3, 0.0]])),
            (np.eye(1), np.eye(1) * 0.5),
            loadings,
            rng.uniform(-0.3, 0.6, n_series) if idio == "ar1" else np.zeros(n_series),
            rng.uniform(0.3, 1.0, n_series),
            np.full(n_series, 1e-4) if idio == "ar1" else rng.uniform(0.3, 1.0, n_series),
        )
        model = build_state_space(params, lay)
        y = ragged_panel(model, 48, rng, n_quarterly=n_q)
        ref = e_step(model, y)
        fast = smoothed_moments(model, y, method="structured")
        p_ref = m_step(ref, lay, y, params)
        p_fast = m_step(fast, lay, y, params)  # type: ignore[arg-type]
        assert fast.loglikelihood == pytest.approx(ref.loglikelihood, rel=1e-10)
        assert p_ref.max_abs_difference(p_fast) < 1e-8

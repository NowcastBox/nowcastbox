"""Unit, analytical and property tests of the EM building blocks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.models._em_steps import (
    VARIANCE_FLOOR,
    EMParameters,
    StateLayout,
    SufficientStatistics,
    build_state_space,
    e_step,
    em_converged,
    m_step,
    restricted_least_squares,
    trim_weights,
)
from nowcastbox.models._init_conditions import (
    _ar1_from_residuals,
    _initial_var,
    fill_for_initialization,
    pca_initial_parameters,
    principal_components,
)
from nowcastbox.preprocessing.aggregation import loading_constraints
from nowcastbox.statespace import StateSpace, kalman_smoother
from tests.models.test_em_simulation import MM_WEIGHTS, simulate_mixed_dfm

MM = MM_WEIGHTS.tolist()


def _layout(idio="ar1", weights=None, membership=None, n_factors=(1,), p=1, names=None):
    weights = weights or [[1.0], [1.0], MM]
    n = len(weights)
    membership = np.ones((n, len(n_factors)), bool) if membership is None else membership
    names = names or [f"b{k}" for k in range(len(n_factors))]
    return StateLayout([f"s{i}" for i in range(n)], names, n_factors, p, membership, weights, idio)


def _params(layout: StateLayout, rng: np.random.Generator) -> EMParameters:
    trans = tuple(
        0.5 * np.hstack([np.eye(r)] + [0.1 * np.eye(r)] * (layout.factor_lags - 1))
        for r in layout.n_factors
    )
    covs = tuple(np.eye(r) for r in layout.n_factors)
    loadings = np.zeros((layout.n_series, layout.n_factor_states))
    for i in range(layout.n_series):
        idx = layout.loading_index(i)
        r_i = idx.size // layout.weights[i].size
        lam0 = rng.uniform(0.5, 1.5, r_i)
        loadings[i, idx] = np.kron(layout.weights[i], lam0)
    n = layout.n_series
    ar = np.full(n, 0.3) if layout.idiosyncratic == "ar1" else np.zeros(n)
    return EMParameters(trans, covs, loadings, ar, np.full(n, 0.5), np.full(n, 1e-2))


# ---------------------------------------------------------------- weights / layout
def test_trim_weights():
    assert trim_weights([1, 0, 0]).tolist() == [1.0]
    assert trim_weights([1, 0, 2]).tolist() == [1.0, 0.0, 2.0]
    for bad in ([], [0.0, 1.0], [1.0, np.nan]):
        with pytest.raises(ValueError):
            trim_weights(bad)


def test_layout_dimensions_ar1():
    lay = _layout(n_factors=(2,), p=2)
    assert lay.block_lags == (5,)
    assert lay.n_factor_states == 10
    assert lay.n_states == 10 + 1 + 1 + 5
    assert lay.n_shocks == 2 + 3
    assert lay.loading_index(0).tolist() == [0, 1]
    assert lay.loading_index(2).tolist() == list(range(10))
    assert lay.idio_index(2).tolist() == [12, 13, 14, 15, 16]
    assert lay.factor_names == ["b0_f1", "b0_f2"]
    assert "n_states=17" in repr(lay)


def test_layout_iid_and_blocks():
    membership = np.array([[1, 0], [1, 1], [0, 1]], bool)
    lay = _layout(
        "iid", weights=[[1.0], [1.0], [1.0]], membership=membership, n_factors=(1, 2), p=3
    )
    assert lay.block_lags == (3, 3)
    assert lay.n_states == lay.n_factor_states == 3 + 6
    assert lay.idio_index(0).size == 0
    assert lay.loading_index(1).tolist() == [0, 3, 4]
    assert lay.current_factor_index.tolist() == [0, 3, 4]
    assert lay.series_blocks(2) == [1]
    lay2 = StateLayout(["a"], ["g"], [1], 1, [[True]], [[1.0]], "iid", block_prefix=False)
    assert lay2.factor_names == ["f1"]


def test_layout_lag_major_order_matches_constraints():
    membership = np.array([[1, 1]], bool)
    lay = StateLayout(["q"], ["g", "r"], [1, 1], 1, membership, [MM], "ar1")
    idx = lay.loading_index(0)
    # lag-major: (g_t, r_t, g_{t-1}, r_{t-1}, ...)
    assert idx[:4].tolist() == [0, 5, 1, 6]
    R, _ = lay.constraints(0)
    assert R.shape == (8, 10)
    assert np.allclose(R, loading_constraints(MM, 2)[0])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"series": ["a", "a"]}, "unique"),
        ({"n_factors": [1, 1]}, "per block"),
        ({"n_factors": [0]}, "positive"),
        ({"factor_lags": 0}, "positive"),
        ({"weights": [[1.0]]}, "one vector"),
        ({"idiosyncratic": "garch"}, "idiosyncratic"),
        ({"membership": [[True], [False]]}, "at least one block"),
    ],
)
def test_layout_validation(kwargs, match):
    base = {
        "series": ["a", "b"],
        "block_names": ["g"],
        "n_factors": [1],
        "factor_lags": 1,
        "membership": [[True], [True]],
        "weights": [[1.0], [1.0]],
        "idiosyncratic": "ar1",
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        StateLayout(**base)


def test_layout_empty_block_rejected():
    with pytest.raises(ValueError, match="at least one series"):
        StateLayout(["a"], ["g", "r"], [1, 1], 1, [[True, False]], [[1.0]], "ar1")


def test_layout_compatibility():
    a, b = _layout(), _layout()
    assert a.is_compatible(b)
    assert not a.is_compatible(_layout("iid"))
    assert not a.is_compatible(_layout(weights=[[1.0], [1.0], [1.0, 1.0, 1.0]]))


# ---------------------------------------------------------------- state space
def test_build_state_space_structure(rng):
    lay = _layout(n_factors=(2,), p=2)
    p = _params(lay, rng)
    ssm = build_state_space(p, lay)
    assert ssm.n_states == lay.n_states
    assert ssm.n_disturbances == lay.n_shocks
    # factor companion
    assert np.allclose(ssm.T[:2, :4], p.transition[0])
    assert np.allclose(ssm.T[2:4, :2], np.eye(2))
    # quarterly idio segment aggregates with the weights
    e_idx = lay.idio_index(2)
    assert np.allclose(ssm.Z[2, e_idx], MM)
    assert np.isclose(ssm.T[e_idx[0], e_idx[0]], 0.3)
    assert np.allclose(ssm.T[e_idx[1:], e_idx[:-1]], 1.0)
    # stationary initial covariance of the idio AR(1) segment
    P = ssm.P0[np.ix_(e_idx, e_idx)]
    assert np.isclose(P[0, 0], 0.5 / (1 - 0.09))
    assert np.isclose(P[0, 2], 0.5 / (1 - 0.09) * 0.09)
    # P0 solves the Lyapunov equation of the whole (block-diagonal) system
    rqr = ssm.R @ ssm.Q @ ssm.R.T
    assert np.allclose(ssm.T @ ssm.P0 @ ssm.T.T + rqr, ssm.P0, atol=1e-10)


def test_build_state_space_nonstationary_fallback(rng):
    lay = _layout("iid", weights=[[1.0], [1.0]])
    p = _params(lay, rng)
    p = EMParameters(
        (np.array([[1.2]]),), p.factor_cov, p.loadings, p.idio_ar, p.idio_var, p.obs_var
    )
    ssm = build_state_space(p, lay)
    assert np.allclose(ssm.P0, 10.0 * np.eye(1))


def test_parameters_copy_and_difference(rng):
    lay = _layout()
    p = _params(lay, rng)
    q = p.copy()
    assert p.max_abs_difference(q) == 0.0
    q.loadings[0, 0] += 0.25
    assert np.isclose(p.max_abs_difference(q), 0.25)


# ---------------------------------------------------------------- E-step
def test_e_step_moments_match_definition(rng):
    lay = _layout()
    p = _params(lay, rng)
    ssm = build_state_space(p, lay)
    from nowcastbox.statespace import simulate_state_space

    y, _ = simulate_state_space(ssm, 40, random_state=1)
    y[rng.random(y.shape) < 0.2] = np.nan
    stats = e_step(ssm, y)
    sm = kalman_smoother(ssm, y, method="univariate")
    a, P, C = sm.smoothed_state, sm.smoothed_state_cov, sm.smoothed_state_autocov
    s11 = sum(np.outer(a[t], a[t]) + P[t] for t in range(1, 40))
    s10 = sum(np.outer(a[t], a[t - 1]) + C[t - 1] for t in range(1, 40))
    s00 = sum(np.outer(a[t - 1], a[t - 1]) + P[t - 1] for t in range(1, 40))
    assert np.allclose(stats.s11, s11)
    assert np.allclose(stats.s10, s10)
    assert np.allclose(stats.s00, s00)
    assert stats.n_pairs == 39
    assert np.isclose(stats.loglikelihood, sm.loglikelihood)


def test_e_step_needs_two_periods():
    ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
    with pytest.raises(ValueError, match="two periods"):
        e_step(ssm, np.array([[1.0]]))


# ---------------------------------------------------------------- M-step
def _degenerate_stats(states: np.ndarray, model: StateSpace) -> SufficientStatistics:
    """Sufficient statistics of perfectly known states (zero smoothed covariances)."""
    n, m = states.shape
    from nowcastbox.statespace.kalman import FilterResult
    from nowcastbox.statespace.smoother import SmootherResult

    zeros = np.zeros((n, m, m))
    fr = FilterResult(
        model=model,
        observations=np.zeros((n, model.n_obs)),
        predicted_state=np.zeros((n + 1, m)),
        predicted_state_cov=np.zeros((n + 1, m, m)),
        filtered_state=states,
        filtered_state_cov=zeros,
        loglikelihood_obs=np.zeros(n),
        n_observed=np.zeros(n, dtype=np.int64),
        method="univariate",
    )
    sm = SmootherResult(fr, states, zeros, zeros)
    a = states
    return SufficientStatistics(sm, a[1:].T @ a[1:], a[:-1].T @ a[:-1], a[1:].T @ a[:-1], n - 1)


def test_m_step_with_known_states_is_ols(rng):
    """With known states the M-step reduces to OLS regressions (analytical check)."""
    lay = StateLayout(["a", "b", "c"], ["g"], [2], 2, np.ones((3, 1), bool), [[1.0]] * 3, "iid")
    n = 400
    f = np.zeros((n, 2))
    A1 = np.array([[0.5, 0.1], [0.0, 0.4]])
    A2 = np.array([[0.2, 0.0], [0.1, 0.1]])
    for t in range(2, n):
        f[t] = A1 @ f[t - 1] + A2 @ f[t - 2] + rng.standard_normal(2)
    states = np.hstack([f, np.vstack([np.zeros((1, 2)), f[:-1]])])
    lam = rng.standard_normal((3, 2))
    y = f @ lam.T + 0.3 * rng.standard_normal((n, 3))
    y[rng.random(y.shape) < 0.1] = np.nan
    p0 = EMParameters(
        (np.zeros((2, 4)),), (np.eye(2),), np.zeros((3, 4)), np.zeros(3), np.ones(3), np.ones(3)
    )
    model = build_state_space(p0, lay)
    new = m_step(_degenerate_stats(states, model), lay, y, p0)
    # VAR by OLS on pairs t = 1..n-1 with regressors (f_{t-1}, f_{t-2})
    X, Y = states[:-1], f[1:]
    A_ols = np.linalg.lstsq(X, Y, rcond=None)[0].T
    assert np.allclose(new.transition[0], A_ols)
    resid = Y - X @ A_ols.T
    assert np.allclose(new.factor_cov[0], resid.T @ resid / (n - 1))
    for i in range(3):
        ok = ~np.isnan(y[:, i])
        coef = np.linalg.lstsq(f[ok], y[ok, i], rcond=None)[0]
        assert np.allclose(new.loadings[i, :2], coef)
        assert np.allclose(new.loadings[i, 2:], 0.0)
        e = y[ok, i] - f[ok] @ coef
        assert np.isclose(new.idio_var[i], e @ e / ok.sum())
    assert np.allclose(new.obs_var, new.idio_var)


def test_m_step_ar1_idiosyncratic_known_states(rng):
    lay = StateLayout(["a"], ["g"], [1], 1, [[True]], [[1.0]], "ar1")
    n = 500
    f = np.zeros(n)
    e = np.zeros(n)
    for t in range(1, n):
        f[t] = 0.6 * f[t - 1] + rng.standard_normal()
        e[t] = 0.3 * e[t - 1] + 0.5 * rng.standard_normal()
    states = np.column_stack([f, e])
    y = (0.8 * f + e)[:, None]
    p0 = EMParameters(
        (np.array([[0.1]]),),
        (np.eye(1),),
        np.ones((1, 1)),
        np.zeros(1),
        np.ones(1),
        np.full(1, 1e-4),
    )
    new = m_step(_degenerate_stats(states, build_state_space(p0, lay)), lay, y, p0)
    rho = (e[1:] @ e[:-1]) / (e[:-1] @ e[:-1])
    assert np.isclose(new.idio_ar[0], rho)
    v = e[1:] - rho * e[:-1]
    assert np.isclose(new.idio_var[0], v @ v / (n - 1))
    assert np.isclose(new.loadings[0, 0], 0.8)
    assert new.obs_var[0] == 1e-4


def test_m_step_clips_ar_and_floors_variance(rng):
    lay = StateLayout(["a"], ["g"], [1], 1, [[True]], [[1.0]], "ar1")
    n = 50
    e = np.arange(n, dtype=float)  # explosive path -> rho > 1 before clipping
    f = rng.standard_normal(n)
    states = np.column_stack([f, e])
    y = (f + e)[:, None]
    p0 = EMParameters(
        (np.array([[0.1]]),),
        (np.eye(1),),
        np.ones((1, 1)),
        np.zeros(1),
        np.ones(1),
        np.full(1, 1e-4),
    )
    new = m_step(_degenerate_stats(states, build_state_space(p0, lay)), lay, y, p0)
    assert new.idio_ar[0] == pytest.approx(0.995)
    states2 = np.column_stack([f, np.zeros(n)])
    new2 = m_step(_degenerate_stats(states2, build_state_space(p0, lay)), lay, y, p0)
    assert new2.idio_var[0] == pytest.approx(VARIANCE_FLOOR)
    assert new2.idio_ar[0] == p0.idio_ar[0]


def test_m_step_quarterly_loadings_respect_aggregation(rng):
    lay = _layout()
    p = _params(lay, rng)
    ssm = build_state_space(p, lay)
    from nowcastbox.statespace import simulate_state_space

    y, _ = simulate_state_space(ssm, 120, random_state=3)
    y[np.arange(120) % 3 != 2, 2] = np.nan
    new = m_step(e_step(ssm, y), lay, y, p)
    lam = new.loadings[2, lay.loading_index(2)]
    assert np.allclose(lam, lam[0] * np.array(MM))
    R, q = lay.constraints(2)
    assert np.allclose(R @ lam, q)


def test_m_step_keeps_loadings_of_unobserved_series(rng):
    lay = _layout("iid", weights=[[1.0], [1.0]])
    p = _params(lay, rng)
    y = rng.standard_normal((30, 2))
    y[:, 1] = np.nan
    new = m_step(e_step(build_state_space(p, lay), y), lay, y, p)
    assert np.allclose(new.loadings[1], p.loadings[1])


# ---------------------------------------------------------------- restricted LS
def test_restricted_least_squares_matches_reparametrisation(rng):
    """Restricted LS equals LS on the aggregated regressor (lambda_l = w_l lambda_0)."""
    n, L = 200, 5
    G = rng.standard_normal((n, L))
    y = G @ np.array(MM) * 0.7 + 0.1 * rng.standard_normal(n)
    R, q = loading_constraints(MM, 1)
    beta = restricted_least_squares(G.T @ G, G.T @ y, R, q)
    z = G @ np.array(MM)
    lam0 = (z @ y) / (z @ z)
    assert np.allclose(beta, lam0 * np.array(MM))


def test_restricted_least_squares_unrestricted_and_singular():
    S = np.array([[2.0, 0.0], [0.0, 1.0]])
    assert np.allclose(restricted_least_squares(S, np.array([2.0, 1.0])), [1.0, 1.0])
    assert np.allclose(
        restricted_least_squares(S, np.array([2.0, 1.0]), np.zeros((0, 2))), [1.0, 1.0]
    )
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    beta = restricted_least_squares(singular, np.array([2.0, 2.0]))
    assert np.allclose(singular @ beta, [2.0, 2.0])


@pytest.mark.property
@given(
    st.integers(min_value=0, max_value=10_000),
    st.sampled_from([[1.0, 2.0, 3.0, 2.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.5], [1.0, -1.0, 2.0]]),
    st.integers(min_value=1, max_value=3),
)
def test_restricted_least_squares_property(seed, weights, r):
    """The solution is feasible and no feasible point has a lower objective."""
    rng = np.random.default_rng(seed)
    k = len(weights) * r
    X = rng.standard_normal((3 * k + 5, k))
    S = X.T @ X
    s = X.T @ rng.standard_normal(3 * k + 5)
    R, q = loading_constraints(weights, r)
    beta = restricted_least_squares(S, s, R, q)
    assert np.allclose(R @ beta, q, atol=1e-8)

    def objective(b):
        return b @ S @ b - 2 * b @ s

    lam0 = rng.standard_normal(r)
    other = np.kron(np.asarray(weights), lam0)
    assert objective(beta) <= objective(other) + 1e-8


def test_restricted_least_squares_with_rhs():
    beta = restricted_least_squares(np.eye(2), np.zeros(2), np.array([[1.0, 1.0]]), np.array([2.0]))
    assert np.allclose(beta, [1.0, 1.0])


# ---------------------------------------------------------------- convergence
def test_em_converged():
    assert em_converged(-100.0, -100.0, 1e-4) == (True, 0.0)
    ok, change = em_converged(-90.0, -100.0, 1e-4)
    assert not ok
    assert change > 0
    ok, change = em_converged(-110.0, -100.0, 1e-4)
    assert not ok
    assert change < 0


@pytest.mark.property
@given(st.floats(-1e6, 1e6), st.floats(-1e6, 1e6))
def test_em_converged_symmetry(a, b):
    _, c1 = em_converged(a, b, 1e-4)
    _, c2 = em_converged(b, a, 1e-4)
    assert np.isclose(c1, -c2)
    assert abs(c1) <= 2.0 + 1e-12


# ---------------------------------------------------------------- initial conditions
def test_principal_components_recover_rank_one():
    t = np.linspace(-1, 1, 30)
    x = np.outer(t, [1.0, -0.5, 2.0])
    pc = principal_components(x, 1)[:, 0]
    assert abs(np.corrcoef(pc, t)[0, 1]) > 0.999999
    with pytest.raises(ValueError, match="cannot extract"):
        principal_components(x, 4)


def test_principal_components_sign_convention(rng):
    x = rng.standard_normal((50, 4))
    pc = principal_components(x, 2)
    pc_neg = principal_components(x, 2)
    assert np.allclose(pc, pc_neg)


def test_fill_for_initialization_mixed():
    idx = pd.period_range("2020-01", periods=9, freq="M")
    df = pd.DataFrame(
        {
            "m": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, np.nan, np.nan],
            "q": [np.nan, np.nan, 1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan],
        },
        index=idx,
    )
    filled = fill_for_initialization(MixedFrequencyData(df, {"m": "M", "q": "Q"}))
    assert not np.isnan(filled).any()
    assert np.allclose(filled[:3, 1], 1.0)
    assert np.allclose(filled[3:6, 1], [2.0, 3.0, 4.0])
    assert np.allclose(filled[[2, 5], 1], [1.0, 4.0])  # observed slots are kept


def test_fill_for_initialization_rejects_empty_series():
    idx = pd.period_range("2020-01", periods=4, freq="M")
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": np.nan}, index=idx)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfd = MixedFrequencyData(df, "M")
    with pytest.raises(NowcastDataError, match="without observations"):
        fill_for_initialization(mfd)


def test_initial_var_errors_and_values(rng):
    f = np.zeros((300, 1))
    for t in range(1, 300):
        f[t] = 0.8 * f[t - 1] + rng.standard_normal()
    A, Q = _initial_var(f, 1)
    assert abs(A[0, 0] - 0.8) < 0.1
    assert abs(Q[0, 0] - 1.0) < 0.2
    with pytest.raises(NowcastDataError, match="Too few periods"):
        _initial_var(f[:3], 2)


def test_ar1_from_residuals():
    resid = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    rho, var = _ar1_from_residuals(resid, np.arange(5))
    assert 0.4 < rho < 0.6
    assert var > 0
    rho0, var0 = _ar1_from_residuals(resid[:2], np.array([0, 3]))
    assert rho0 == 0.0
    assert var0 == pytest.approx(np.mean(resid[:2] ** 2))
    rho_z, _ = _ar1_from_residuals(np.zeros(6), np.arange(6))
    assert rho_z == 0.0


def test_pca_initial_parameters_on_simulated_panel():
    sim = simulate_mixed_dfm(
        n_periods=120, n_monthly=10, n_quarterly=2, blocks={"global": 1, "real": 1}, seed=4
    )
    mfd, _ = MixedFrequencyData(sim.data, sim.frequencies).standardize()
    lay = StateLayout(
        mfd.columns,
        ["global", "real"],
        [1, 1],
        2,
        sim.blocks.to_numpy(),
        [[1.0]] * 10 + [MM] * 2,
        "ar1",
    )
    p = pca_initial_parameters(mfd, lay)
    assert p.loadings.shape == (12, lay.n_factor_states)
    for i in (10, 11):
        lam = p.loadings[i, lay.loading_index(i)]
        R, q = lay.constraints(i)
        assert np.allclose(R @ lam, q, atol=1e-8)
    assert np.all(p.idio_var > 0)
    assert np.all(np.abs(p.idio_ar) < 1)
    assert np.allclose(p.obs_var, 1e-4)
    # non-member loadings stay exactly zero
    real_idx = lay.factor_index(1, 0)
    non_members = ~sim.blocks["real"].to_numpy()
    assert np.allclose(p.loadings[np.ix_(non_members, real_idx)], 0.0)
    with pytest.raises(ValueError, match="columns and layout"):
        pca_initial_parameters(mfd.select(mfd.columns[::-1]), lay)


def test_pca_initialisation_block_with_quarterly_only_series():
    sim = simulate_mixed_dfm(n_periods=90, n_monthly=6, n_quarterly=2, seed=5)
    mfd, _ = MixedFrequencyData(sim.data, sim.frequencies).standardize()
    membership = np.zeros((8, 2), bool)
    membership[:, 0] = True
    membership[6:, 1] = True  # block made only of quarterly series
    lay = StateLayout(mfd.columns, ["g", "q"], [1, 1], 1, membership, [[1.0]] * 6 + [MM] * 2, "iid")
    p = pca_initial_parameters(mfd, lay)
    assert np.isfinite(p.loadings).all()
    lay_bad = StateLayout(
        mfd.columns, ["g", "q"], [1, 3], 1, membership, [[1.0]] * 6 + [MM] * 2, "iid"
    )
    with pytest.raises(ValueError, match="factors were requested"):
        pca_initial_parameters(mfd, lay_bad)


def test_pca_initialisation_series_without_usable_observations():
    idx = pd.period_range("2020-01", periods=40, freq="M")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((40, 2)), index=idx, columns=["a", "b"])
    df["q"] = np.nan
    df.iloc[2, 2] = 1.0  # single observation, before 5 factor lags are available
    mfd = MixedFrequencyData(df, {"a": "M", "b": "M", "q": "Q"})
    lay = StateLayout(mfd.columns, ["g"], [1], 1, np.ones((3, 1), bool), [[1.0], [1.0], MM], "ar1")
    with pytest.raises(NowcastDataError, match="complete factor lags"):
        pca_initial_parameters(mfd, lay)


def test_e_step_structured_method_falls_back_with_offset():
    """``method='auto'`` with a state offset uses the dense smoother (same moments)."""
    from nowcastbox.statespace import StateSpace

    ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [0.5])
    y = np.array([[1.0], [np.nan], [0.2], [0.4]])
    offset = np.full((4, 1), 0.1)
    auto = e_step(ssm, y, method="auto", state_offset=offset)
    dense = e_step(ssm, y, method="univariate", state_offset=offset)
    np.testing.assert_allclose(auto.s11, dense.s11)
    np.testing.assert_allclose(auto.smoother.smoothed_state, dense.smoother.smoothed_state)
    plain = e_step(ssm, y, method="auto")
    np.testing.assert_allclose(plain.loglikelihood, e_step(ssm, y).loglikelihood, rtol=1e-10)

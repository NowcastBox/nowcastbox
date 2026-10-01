"""Unit and integration tests of the robust EM (innovation I3, ``nowcastbox.models.robust``)."""

from __future__ import annotations

import doctest
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from hypothesis import given
from hypothesis import strategies as st

import nowcastbox.models.robust as robust
from nowcastbox.core.exceptions import ConvergenceWarning, DataQualityWarning, NowcastDataError
from nowcastbox.models import EMParameters, MixedFreqDFM, StateLayout
from nowcastbox.models._em_steps import build_state_space, e_step, smoother_with_obs_var
from nowcastbox.models.robust import (
    RobustSpec,
    RobustState,
    StudentTWeights,
    expected_squared_residuals,
    flag_outliers,
    intervention_offset,
    period_mask,
    robust_smoother,
    run_robust_em,
    standardized_innovations,
    student_t_bound,
    update_df,
    update_interventions,
)
from nowcastbox.statespace import StateSpace, kalman_smoother
from tests.models.test_robust_simulation import simulate_heavy_tails, simulate_pandemic


def _layout(kind: str = "iid", lags: int = 1, n: int = 2) -> StateLayout:
    return StateLayout(
        [f"s{i}" for i in range(n)], ["g"], [1], lags, np.ones((n, 1), bool), [[1.0]] * n, kind
    )


def _params(lags: int = 1, n: int = 2) -> EMParameters:
    A = np.zeros((1, lags))
    A[0, 0] = 0.5
    loadings = np.zeros((n, lags))
    loadings[:, 0] = 1.0
    return EMParameters((A,), (np.eye(1),), loadings, np.zeros(n), np.ones(n), np.ones(n))


@pytest.fixture(scope="module")
def heavy():
    return simulate_heavy_tails(7, n=150, n_series=8, hold=4)


@pytest.fixture(scope="module")
def fit_t(heavy):
    return MixedFreqDFM(idiosyncratic="student_t", outliers="auto", max_iter=60).fit(
        heavy.data, "gdp", frequency={"gdp": "Q"}
    )


@pytest.fixture(scope="module")
def pandemic():
    return simulate_pandemic(3, n_series=6)


@pytest.fixture(scope="module")
def fit_dummy(pandemic):
    return MixedFreqDFM(covid="dummy", covid_window=("2020-03", "2020-08"), max_iter=60).fit(
        pandemic.data, "gdp", frequency={"gdp": "Q"}
    )


# ====================================================================== periods
class TestPeriodMask:
    idx = pd.period_range("2019-11", "2021-02", freq="M")

    def test_none_and_single_periods(self):
        assert not period_mask(self.idx, None).any()
        assert period_mask(self.idx, "2020-04").sum() == 1
        assert period_mask(self.idx, pd.Period("2020Q2")).sum() == 3
        assert period_mask(self.idx, "2020").sum() == 12

    def test_ranges_and_lists(self):
        m = period_mask(self.idx, ("2020-03", "2020Q3"))
        assert m.sum() == 7 and self.idx[m][0] == pd.Period("2020-03", "M")
        both = period_mask(self.idx, (("2019-11", "2019-12"), ("2021-01", "2021-02")))
        assert both.sum() == 4
        assert period_mask(self.idx, ["2020-01", ("2020-02", "2020-02")]).sum() == 2

    def test_out_of_sample_period_is_empty(self):
        assert not period_mask(self.idx, "1999-01").any()

    def test_errors(self):
        with pytest.raises(ValueError, match="interpret"):
            period_mask(self.idx, ["not a period"])
        with pytest.raises(ValueError, match="ends before"):
            period_mask(self.idx, ("2020-05", "2020-01"))


def test_robust_spec_activity():
    assert not RobustSpec().is_active
    assert not RobustSpec(excluded=np.zeros((3, 2), bool)).is_active
    assert RobustSpec(excluded=np.ones((3, 2), bool)).is_active
    first_only = np.array([True, False, False])
    assert not RobustSpec(intervention=first_only).has_intervention
    assert not RobustSpec(intervention=first_only).is_active
    assert RobustSpec(intervention=np.array([False, True])).is_active
    assert RobustSpec(outliers=True).is_active


# ====================================================================== Student-t
def test_student_t_weights_formulas():
    q = StudentTWeights.posterior(np.array([[0.0, 4.0], [np.nan, 1.0]]), 4.0)
    assert q.shape == 2.5
    np.testing.assert_allclose(q.weights, [[5 / 4, 5 / 8], [1.0, 1.0]])
    expected = scipy.special.digamma(2.5) - np.log(q.rate)
    np.testing.assert_allclose(q.log_weights, expected)


@given(
    delta=st.lists(st.floats(0.0, 1e4), min_size=1, max_size=20),
    df=st.floats(0.5, 500.0),
)
def test_weights_are_bounded_and_decreasing(delta, df):
    d = np.sort(np.asarray(delta))[:, None]
    w = StudentTWeights.posterior(d, df).weights.ravel()
    assert np.all(w > 0) and np.all(w <= (df + 1) / df * (1 + 1e-12))
    assert np.all(np.diff(w) <= 1e-12)


@given(
    shape=st.floats(0.3, 100.0),
    rates=st.lists(st.floats(0.05, 100.0), min_size=1, max_size=10),
    df=st.floats(0.5, 300.0),
)
def test_bound_correction_is_non_positive(shape, rates, df):
    q = StudentTWeights(shape, np.asarray(rates)[:, None])
    assert student_t_bound(q, df, np.ones((len(rates), 1), bool)) <= 1e-9


def test_bound_correction_vanishes_at_the_prior_for_large_df():
    q = StudentTWeights.prior((5, 2), 1e9)
    assert abs(student_t_bound(q, 1e9, np.ones((5, 2), bool))) < 1e-6


def test_update_df_recovers_true_degrees_of_freedom():
    rng = np.random.default_rng(1)
    lam = rng.gamma(4.0, 1 / 4.0, size=(20000, 1))  # nu = 8
    q = StudentTWeights(1e7, 1e7 / lam)
    assert abs(update_df(q, np.ones_like(lam, bool)) - 8.0) < 0.5
    empty = update_df(q, np.zeros_like(lam, bool))
    assert empty == pytest.approx(np.sqrt(2.0 * 200.0))


def test_update_df_is_clipped_to_bounds():
    q = StudentTWeights.prior((50, 1), 1e6)
    assert update_df(q, np.ones((50, 1), bool)) == pytest.approx(200.0, rel=1e-4)


def test_expected_squared_residuals_matches_definition():
    ssm = StateSpace([[0.5]], [[1.0], [2.0]], [[1.0]], [0.5, 0.5])
    y = np.array([[1.0, np.nan], [0.3, -0.2], [np.nan, 1.0]])
    sm = kalman_smoother(ssm, y)
    out = expected_squared_residuals(sm, y)
    a, V = sm.smoothed_state[:, 0], sm.smoothed_state_cov[:, 0, 0]
    expected = (y - np.outer(a, [1.0, 2.0])) ** 2 + np.outer(V, [1.0, 4.0])
    np.testing.assert_allclose(out, expected)
    assert np.isnan(out[0, 1])


# ====================================================================== smoother with H_t
def test_time_varying_measurement_variance_is_exact():
    """With T = 0 every period is a static Gaussian update: closed-form posterior."""
    z = np.array([1.0, 0.5, 2.0])
    ssm = StateSpace([[0.0]], z[:, None], [[1.5]], np.ones(3))
    rng = np.random.default_rng(0)
    y = rng.standard_normal((4, 3))
    y[1, 2] = np.nan
    h = rng.uniform(0.2, 3.0, size=(4, 3))
    sm = smoother_with_obs_var(ssm, y, h)
    loglik = 0.0
    for t in range(4):
        obs = ~np.isnan(y[t])
        prec = 1 / 1.5 + np.sum(z[obs] ** 2 / h[t, obs])
        mean = np.sum(z[obs] * y[t, obs] / h[t, obs]) / prec
        assert sm.smoothed_state[t, 0] == pytest.approx(mean)
        assert sm.smoothed_state_cov[t, 0, 0] == pytest.approx(1 / prec)
        cov = 1.5 * np.outer(z[obs], z[obs]) + np.diag(h[t, obs])
        loglik += scipy.stats.multivariate_normal(np.zeros(obs.sum()), cov).logpdf(y[t, obs])
    assert sm.loglikelihood == pytest.approx(loglik)


def test_smoother_with_obs_var_rejects_bad_variances():
    ssm = StateSpace([[0.5]], [[1.0]], [[1.0]], [1.0])
    with pytest.raises(ValueError, match="obs_var"):
        smoother_with_obs_var(ssm, np.zeros((3, 1)), np.ones((2, 1)))
    with pytest.raises(ValueError, match="obs_var"):
        smoother_with_obs_var(ssm, np.zeros((2, 1)), -np.ones((2, 1)))


# ====================================================================== interventions
def test_intervention_offset_follows_the_companion_form():
    lay = _layout(lags=2)
    p = _params(lags=2)
    p = EMParameters(
        (np.array([[0.5, 0.2]]),), p.factor_cov, p.loadings, p.idio_ar, p.idio_var, p.obs_var
    )
    dummies = np.zeros((4, 1))
    dummies[1] = 1.0
    g = intervention_offset(dummies, p, lay)
    np.testing.assert_allclose(g[:, 0], [0.0, 1.0, 0.5, 0.45])
    np.testing.assert_allclose(g[:, 1], [0.0, 0.0, 1.0, 0.5])
    back = update_interventions(g, p, lay, np.array([True, True, False, False]))
    np.testing.assert_allclose(back[:, 0], [0.0, 1.0, 0.0, 0.0])


def test_e_step_offset_and_profiled_moments():
    lay, p = _layout(), _params()
    model = build_state_space(p, lay)
    y = np.random.default_rng(2).standard_normal((12, 2))
    g = np.zeros((12, 1))
    g[5:] = 0.8 ** np.arange(7)[:, None]
    shifted = e_step(model, y, state_offset=g)
    plain = e_step(model, y - g @ model.Z.T)
    np.testing.assert_allclose(shifted.smoother.smoothed_state, plain.smoother.smoothed_state + g)
    free = np.zeros(12, bool)
    free[5] = True
    prof = e_step(model, y, free_periods=free, free_states=np.arange(1))
    base = e_step(model, y)
    a = base.smoother.smoothed_state[:, 0]
    assert prof.s11[0, 0] == pytest.approx(base.s11[0, 0] - a[5] ** 2)
    assert prof.s00[0, 0] == pytest.approx(base.s00[0, 0] - a[4] ** 2)
    assert prof.s10[0, 0] == pytest.approx(base.s10[0, 0] - a[5] * a[4])
    first_only = np.zeros(12, bool)
    first_only[0] = True
    same = e_step(model, y, free_periods=first_only, free_states=np.arange(1))
    np.testing.assert_allclose(same.s11, base.s11)


# ====================================================================== outliers
def test_standardized_innovations_and_flags():
    ssm = StateSpace([[0.0]], [[1.0], [1.0]], [[1.0]], [1.0, 3.0])
    y = np.array([[2.0, np.nan], [0.0, 8.0]])
    sm = kalman_smoother(ssm, y)
    z = standardized_innovations(sm, y)
    np.testing.assert_allclose(z[0, 0], 2.0 / np.sqrt(2.0))
    assert z[1, 1] == pytest.approx(4.0)
    assert np.isnan(z[0, 1])
    shifted = standardized_innovations(sm, y, offset=np.ones((2, 1)))
    np.testing.assert_allclose(shifted[1, 1], 7.0 / 2.0)
    np.testing.assert_array_equal(flag_outliers(z, 3.5), [[False, False], [False, True]])


def test_robust_state_padding():
    q = StudentTWeights(2.0, np.full((3, 2), 4.0))
    state = RobustState(np.ones((3, 2), bool), q, 3.0, np.ones((3, 1)))
    padded = state.padded(5)
    assert padded.flags[3:].sum() == 0 and padded.flags[:3].all()
    assert padded.q is not None and np.all(padded.q.weights[3:] == 1.0)
    assert padded.dummies is not None and np.all(padded.dummies[3:] == 0.0)


# ====================================================================== robust EM
def test_run_robust_em_rejects_ar1_with_student_t():
    with pytest.raises(ValueError, match="iid"):
        run_robust_em(_params(), _layout("ar1"), np.zeros((5, 2)), RobustSpec(student_t=True))


def test_run_robust_em_non_finite_objective(monkeypatch):
    monkeypatch.setattr(robust, "student_t_bound", lambda *a, **k: float("nan"))
    y = np.random.default_rng(0).standard_normal((20, 2))
    with pytest.raises(NowcastDataError, match="Non-finite"):
        run_robust_em(_params(), _layout(), y, RobustSpec(student_t=True), max_iter=2)


def test_run_robust_em_counts_decreases(monkeypatch):
    values = iter([0.0, -5.0, -6.0])

    def fake_objective(*args, **kwargs):
        return next(values)

    monkeypatch.setattr(robust, "_objective", fake_objective)
    y = np.random.default_rng(0).standard_normal((20, 2))
    out = run_robust_em(_params(), _layout(), y, RobustSpec(student_t=True), max_iter=2, tol=0)
    assert out.n_decreases == 2 and out.n_iter == 2


def test_outlier_passes_are_capped():
    y = np.random.default_rng(3).standard_normal((40, 2))
    y[10, 0] = 50.0
    spec = RobustSpec(outliers=True, outlier_threshold=0.5, max_outlier_passes=2)
    out = run_robust_em(_params(), _layout(), y, spec, max_iter=5)
    assert out.n_passes == 2
    assert out.state.flags[10, 0]
    assert out.total_iter >= out.n_iter


def test_robust_smoother_with_dummies_uses_offset():
    lay, p = _layout(), _params()
    y = np.random.default_rng(4).standard_normal((10, 2))
    window = np.zeros(10, bool)
    window[4] = True
    dummies = np.zeros((10, 1))
    dummies[4] = 2.0
    stats = robust_smoother(
        p,
        lay,
        y,
        RobustSpec(intervention=window),
        RobustState(np.zeros((10, 2), bool), dummies=dummies),
    )
    g = intervention_offset(dummies, p, lay)
    plain = kalman_smoother(build_state_space(p, lay), y - g @ np.ones((1, 2)))
    np.testing.assert_allclose(stats.smoother.smoothed_state, plain.smoothed_state + g)


# ====================================================================== estimator
pytestmark = pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.ConvergenceWarning")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"df": 4.0}, "df is only used"),
        ({"idiosyncratic": "student_t", "df": 0.0}, "df"),
        ({"idiosyncratic": "student_t", "df": "five"}, "df"),
        ({"outliers": "always"}, "outliers"),
        ({"outlier_threshold": 0.0}, "outlier_threshold"),
        ({"covid": "drop"}, "covid"),
        ({"covid_window": "2020"}, "covid_window"),
        ({"covid_window": ("2020-03", "x")}, "interpret"),
        ({"exclude_periods": ["garbage"]}, "interpret"),
    ],
)
def test_robust_parameter_validation(heavy, kwargs, match):
    with pytest.raises(ValueError, match=match):
        MixedFreqDFM(**kwargs).fit(heavy.data, "gdp", frequency={"gdp": "Q"})


def test_new_parameters_have_neutral_defaults():
    params = MixedFreqDFM().get_params()
    assert params["df"] is None and params["outliers"] == "none"
    assert params["covid"] == "none" and params["covid_window"] == ("2020-03", "2021-12")
    assert params["exclude_periods"] is None and params["long_run_mean"] == "constant"


def test_default_fit_has_no_robust_fields(heavy):
    res = MixedFreqDFM(max_iter=5).fit(heavy.data, "gdp", frequency={"gdp": "Q"})
    for name in (
        "observation_weights",
        "student_t_df",
        "outlier_flags",
        "excluded_observations",
        "interventions",
        "state_offset",
        "long_run_mean",
    ):
        assert getattr(res, name) is None
    assert "objective" not in res.info


def test_student_t_results(fit_t, heavy):
    res = fit_t
    assert res.state_layout is not None and res.state_layout.idiosyncratic == "iid"
    weights = res.observation_weights
    assert weights is not None and weights.shape == heavy.data.shape
    expected = heavy.data.isna().to_numpy() | res.excluded_observations.to_numpy()
    np.testing.assert_array_equal(weights.isna().to_numpy(), expected)
    assert res.outlier_flags is not None and res.outlier_flags.to_numpy().sum() > 0
    assert res.n_iter >= res.info["n_iter_last_pass"]
    text = res.summary()
    for label in ("student_t", "Student-t df", "Outliers flagged", "Excluded obs."):
        assert label in text
    std = res.nowcast["std"].dropna()
    assert (std > 0).all()


def test_fixed_df_is_kept(heavy):
    res = MixedFreqDFM(idiosyncratic="student_t", df=5.0, max_iter=10).fit(
        heavy.data, "gdp", frequency={"gdp": "Q"}
    )
    assert res.student_t_df == 5.0


def test_small_df_uses_scale_for_the_noise_variance(heavy):
    res = MixedFreqDFM(idiosyncratic="student_t", df=1.5, max_iter=5).fit(
        heavy.data, "gdp", frequency={"gdp": "Q"}
    )
    assert res.student_t_df == 1.5 and np.isfinite(res.nowcast["std"].dropna()).all()


def test_exclude_periods_equals_setting_missing(heavy):
    periods = [("2005-01", "2005-06"), "2007Q3"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = MixedFreqDFM(idiosyncratic="iid", exclude_periods=periods, max_iter=8).fit(
            heavy.data, "gdp", frequency={"gdp": "Q"}
        )
        manual = heavy.data.copy()
        manual.loc[period_mask(manual.index, periods)] = np.nan
        ref = MixedFreqDFM(idiosyncratic="iid", max_iter=8).fit(
            manual, "gdp", frequency={"gdp": "Q"}
        )
    assert res.loglikelihood == pytest.approx(ref.loglikelihood)
    np.testing.assert_allclose(res.loadings, ref.loadings)
    excluded = res.excluded_observations
    assert excluded is not None and excluded.loc["2005-03"].all()
    _, values = res.observations()
    assert np.isnan(values[excluded.to_numpy()]).all()
    # the excluded target observation is still reported as observed
    assert res.nowcast.loc[pd.Period("2007Q3"), "observed"] == pytest.approx(
        heavy.data.loc["2007-09", "gdp"]
    )
    np.testing.assert_allclose(res.predict().to_numpy(), ref.predict().to_numpy(), atol=1e-8)


def test_covid_mask_equals_exclude_periods(pandemic):
    kw = {"max_iter": 6, "tol": 0.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        a = MixedFreqDFM(covid="mask", **kw).fit(pandemic.data, "gdp", frequency={"gdp": "Q"})
        b = MixedFreqDFM(exclude_periods=[("2020-03", "2021-12")], **kw).fit(
            pandemic.data, "gdp", frequency={"gdp": "Q"}
        )
    assert a.loglikelihood == pytest.approx(b.loglikelihood)


def test_covid_window_outside_the_sample_warns(heavy):
    with pytest.warns(DataQualityWarning, match="does not overlap"):
        MixedFreqDFM(covid="mask", max_iter=3).fit(heavy.data, "gdp", frequency={"gdp": "Q"})


def test_dummy_results_and_vintage_smoothing(fit_dummy, pandemic):
    res = fit_dummy
    assert res.state_offset is not None and res.grid is not None
    assert res.state_offset.shape == (len(res.grid), res.state_layout.n_states)
    assert "COVID dummies" in res.summary()
    sm = res.smooth()
    np.testing.assert_allclose(sm.smoothed_state, res.smoothed_state, atol=1e-8)
    later = pandemic.data.reindex(pd.period_range("2005-01", "2025-06", freq="M"))
    sm_long = res.smooth(later)
    assert sm_long.smoothed_state.shape[0] == len(later)
    offset = res._offset(len(later))
    assert offset is not None
    T = res.state_space.T
    np.testing.assert_allclose(offset[-1], T @ offset[-2])
    assert res.predict(later).shape == later.shape


def test_offset_is_none_without_dummies(fit_t):
    assert fit_t._offset(10) is None


def test_warm_start_from_robust_results(fit_t, heavy):
    res = MixedFreqDFM(idiosyncratic="student_t", init=fit_t, max_iter=3).fit(
        heavy.data, "gdp", frequency={"gdp": "Q"}
    )
    assert res.converged is not None


@pytest.mark.parametrize("module", [robust])
def test_doctests(module):
    failures, _ = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert failures == 0


def test_dummies_reduce_pandemic_outlier_flags(pandemic):
    """Innovations include the dummy offset, so the pandemic swing is not flagged."""
    kw = {"outliers": "auto", "covid_window": ("2020-03", "2020-08"), "max_iter": 60}
    plain = MixedFreqDFM(**kw).fit(pandemic.data, "gdp", frequency={"gdp": "Q"})
    dummy = MixedFreqDFM(covid="dummy", **kw).fit(pandemic.data, "gdp", frequency={"gdp": "Q"})
    window = slice("2020-03", "2020-08")
    assert plain.outlier_flags is not None and dummy.outlier_flags is not None
    n_plain = int(plain.outlier_flags.loc[window].to_numpy().sum())
    n_dummy = int(dummy.outlier_flags.loc[window].to_numpy().sum())
    assert n_dummy < n_plain, (n_dummy, n_plain)

"""Tests of the time-varying long-run mean (innovation I4, ``nowcastbox.models.long_run``)."""

from __future__ import annotations

import doctest
import warnings

import numpy as np
import pandas as pd
import pytest

import nowcastbox.models.long_run as long_run_module
from nowcastbox.core.exceptions import ConvergenceWarning
from nowcastbox.models import EMParameters, MixedFreqDFM, StateLayout
from nowcastbox.models._em_steps import build_state_space, m_step
from nowcastbox.models._init_conditions import initial_long_run
from nowcastbox.models.long_run import (
    LONG_RUN_INITIAL_VARIANCE,
    LONG_RUN_START_VARIANCE,
    long_run_frame,
    long_run_restrictions,
    random_walk_variance,
    resolve_long_run_series,
)
from tests.models.test_em_steps import _degenerate_stats
from tests.models.test_robust_simulation import MM

pytestmark = pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.ConvergenceWarning")


def simulate_declining_trend(seed: int, *, n: int = 300, n_series: int = 10, hold: int = 4):
    """Target growth = declining long-run mean (3.0 -> 0.5) + cycle; indicators trendless."""
    rng = np.random.default_rng(seed)
    f = np.zeros(n + 50)
    shocks = rng.standard_normal(n + 50)
    for t in range(1, n + 50):
        f[t] = 0.6 * f[t - 1] + shocks[t]
    f = f[50:]
    x = np.outer(f, rng.uniform(0.5, 1.2, n_series)) + 0.6 * rng.standard_normal((n, n_series))
    mu = np.linspace(3.0, 0.5, n)
    gdp_true = np.full(n, np.nan)
    for t in range(4, n):
        gdp_true[t] = mu[t] + 0.5 * (MM @ f[t - np.arange(5)])
    gdp = gdp_true + 0.3 * rng.standard_normal(n)
    gdp[np.arange(n) % 3 != 2] = np.nan
    idx = pd.period_range("1995-01", periods=n, freq="M")
    frame = pd.DataFrame(x, idx, [f"x{i}" for i in range(n_series)]).assign(gdp=gdp)
    # one indicator shares the trend (e.g. consumption growth)
    frame["cons"] = 0.8 * mu + 0.5 * f + 0.4 * rng.standard_normal(n)
    hidden = np.flatnonzero(np.arange(n) % 3 == 2)[-hold:]
    frame.iloc[hidden, frame.columns.get_loc("gdp")] = np.nan
    truth = pd.Series(gdp_true[hidden], index=idx[hidden].asfreq("Q"))
    return frame, truth, pd.Series(mu, idx)


FREQ = {"gdp": "Q"}


@pytest.fixture(scope="module")
def trend_panel():
    return simulate_declining_trend(0)


@pytest.fixture(scope="module")
def fit_tv(trend_panel):
    frame, _, _ = trend_panel
    return MixedFreqDFM(long_run_mean="time_varying", max_iter=300).fit(
        frame.drop(columns="cons"), "gdp", frequency=FREQ
    )


# ====================================================================== helpers
def test_random_walk_variance():
    assert random_walk_variance(5.0, 4.5, 4.5, 2) == 0.25
    assert random_walk_variance(1.0, 1.0, 1.0, 3, floor=1e-4) == 1e-4
    with pytest.raises(ValueError, match="n_pairs"):
        random_walk_variance(1.0, 1.0, 1.0, 0)


def test_long_run_restrictions():
    R0 = np.array([[1.0, -1.0]])
    R, q = long_run_restrictions(R0, np.zeros(1), fixed=False)
    np.testing.assert_array_equal(R, [[1.0, -1.0, 0.0]])
    np.testing.assert_array_equal(q, [0.0])
    R, q = long_run_restrictions(R0, np.zeros(1), fixed=True)
    np.testing.assert_array_equal(R, [[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(q, [0.0, 1.0])


def test_resolve_long_run_series():
    cols = ["a", "gdp", "c"]
    assert resolve_long_run_series("time_varying", None, cols, "gdp") == (1,)
    assert resolve_long_run_series("time_varying", "c", cols, "gdp") == (1, 2)
    assert resolve_long_run_series("time_varying", ["gdp", "a"], cols, "gdp") == (1, 0)
    with pytest.raises(ValueError, match="long_run_mean"):
        resolve_long_run_series("trend", None, cols, "gdp")
    with pytest.raises(ValueError, match="time_varying"):
        resolve_long_run_series("constant", ["a"], cols, "gdp")
    with pytest.raises(ValueError, match="unknown"):
        resolve_long_run_series("time_varying", ["zz"], cols, "gdp")
    with pytest.raises(ValueError, match="duplicated"):
        resolve_long_run_series("time_varying", ["a", "a"], cols, "gdp")


def test_long_run_frame_units():
    idx = pd.period_range("2020-01", periods=3, freq="M")
    frame = long_run_frame(
        np.array([0.0, 0.5, 1.0]),
        np.full(3, 0.04),
        np.array([1.0, -2.0]),
        np.array([1.0, 0.0]),
        np.array([2.0, 1.0]),
        idx,
        ["gdp", "cons"],
    )
    np.testing.assert_allclose(frame["gdp"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(frame["cons"], [0.0, -1.0, -2.0])
    np.testing.assert_allclose(frame["cons_std"], 0.4)
    assert list(frame.columns) == ["gdp", "gdp_std", "cons", "cons_std"]


# ====================================================================== layout & state space
def _layout(long_run=(1,), variance=None, kind="iid") -> StateLayout:
    return StateLayout(
        ["a", "gdp", "c"],
        ["g"],
        [1],
        1,
        np.ones((3, 1), bool),
        [[1.0], [1.0, 2.0, 3.0, 2.0, 1.0], [1.0]],
        kind,
        long_run=long_run,
        long_run_variance=variance,
    )


def test_layout_with_long_run():
    plain = StateLayout(
        ["a", "gdp", "c"], ["g"], [1], 1, np.ones((3, 1), bool), [[1.0], [1.0] * 5, [1.0]], "iid"
    )
    lay = _layout(long_run=(1, 2))
    assert not plain.has_long_run and plain.trend_index == -1
    assert lay.has_long_run and lay.trend_index == lay.n_factor_states
    assert lay.n_states == plain.n_states + 1 and lay.n_shocks == plain.n_shocks + 1
    np.testing.assert_array_equal(lay.regressor_index(1), [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(lay.regressor_index(0), [0])
    R, q = lay.constraints(1)
    assert R.shape == (5, 6) and q[-1] == 1.0
    R2, _ = lay.constraints(2)
    assert R2.shape == (0, 2)
    assert not lay.is_compatible(_layout(long_run=(1,)))
    ar1 = _layout(kind="ar1")
    assert ar1.trend_index == ar1.n_states - 1


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"long_run": (5,)}, "distinct"),
        ({"long_run": (1, 1)}, "distinct"),
        ({"variance": 0.0}, "positive"),
    ],
)
def test_layout_long_run_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _layout(**kwargs)


def _params(lay: StateLayout, trend=True) -> EMParameters:
    loadings = np.zeros((3, lay.n_factor_states))
    loadings[[0, 2], 0] = 1.0
    loadings[1, :5] = [1.0, 2.0, 3.0, 2.0, 1.0]
    trend_loadings = np.array([0.0, 1.0, 0.0]) if trend else None
    return EMParameters(
        (np.array([[0.5]]),),
        (np.eye(1),),
        loadings,
        np.zeros(3),
        np.ones(3),
        np.ones(3),
        trend_loadings,
        0.01,
    )


def test_state_space_with_long_run():
    lay = _layout()
    model = build_state_space(_params(lay), lay)
    k = lay.trend_index
    assert model.T[k, k] == 1.0 and model.Z[1, k] == 1.0 and model.Z[0, k] == 0.0
    assert model.P0[k, k] == LONG_RUN_INITIAL_VARIANCE
    assert model.state_disturbance_cov[k, k] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="trend_loadings"):
        build_state_space(_params(lay, trend=False), lay)


def test_parameters_copy_and_difference_with_trend():
    lay = _layout()
    p = _params(lay)
    c = p.copy()
    assert c.trend_loadings is not p.trend_loadings and c.trend_var == p.trend_var
    q = EMParameters(
        *[getattr(p, f) for f in ("transition", "factor_cov", "loadings")],
        p.idio_ar,
        p.idio_var,
        p.obs_var,
        p.trend_loadings + 0.3,
        0.01,
    )
    assert p.max_abs_difference(q) == pytest.approx(0.3)


def test_initial_long_run():
    assert initial_long_run(_layout(long_run=())) == (None, 0.0)
    loads, var = initial_long_run(_layout(long_run=(1, 0)))
    np.testing.assert_array_equal(loads, [0.0, 1.0, 0.0])
    assert var == LONG_RUN_START_VARIANCE
    assert initial_long_run(_layout(variance=0.02))[1] == 0.02


def test_m_step_long_run_with_known_states(rng):
    """Known states: target loading stays one, others are OLS, variance is RW formula."""
    lay = StateLayout(
        ["a", "b"], ["g"], [1], 1, np.ones((2, 1), bool), [[1.0]] * 2, "iid", long_run=(0, 1)
    )
    n = 500
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = 0.5 * f[t - 1] + rng.standard_normal()
    mu = np.cumsum(0.05 * rng.standard_normal(n))
    states = np.column_stack([f, mu])
    y = np.column_stack([f + mu, 0.7 * f + 2.0 * mu]) + 0.1 * rng.standard_normal((n, 2))
    stats = _known_state_stats(lay, states)
    p0 = EMParameters(
        (np.array([[0.1]]),),
        (np.eye(1),),
        np.ones((2, 1)),
        np.zeros(2),
        np.ones(2),
        np.ones(2),
        np.array([1.0, 0.0]),
        0.1,
    )
    new = m_step(stats, lay, y, p0)
    assert new.trend_loadings is not None
    assert new.trend_loadings[0] == pytest.approx(1.0)
    ols = np.linalg.lstsq(states, y[:, 1], rcond=None)[0]
    np.testing.assert_allclose([new.loadings[1, 0], new.trend_loadings[1]], ols, rtol=1e-8)
    assert new.trend_var == pytest.approx(np.mean(np.diff(mu) ** 2))
    fixed = StateLayout(
        ["a", "b"],
        ["g"],
        [1],
        1,
        np.ones((2, 1), bool),
        [[1.0]] * 2,
        "iid",
        long_run=(0,),
        long_run_variance=0.1,
    )
    assert m_step(_known_state_stats(fixed, states), fixed, y, p0).trend_var == 0.1


def _known_state_stats(layout: StateLayout, states: np.ndarray):
    p = EMParameters(
        (np.array([[0.5]]),),
        (np.eye(1),),
        np.ones((layout.n_series, 1)),
        np.zeros(layout.n_series),
        np.ones(layout.n_series),
        np.ones(layout.n_series),
        np.ones(layout.n_series),
        0.1,
    )
    return _degenerate_stats(states, build_state_space(p, layout))


# ====================================================================== estimator
def test_time_varying_mean_tracks_declining_trend(trend_panel, fit_tv):
    frame, truth, mu = trend_panel
    res = fit_tv
    path = res.long_run_mean
    assert path is not None and list(path.columns) == ["gdp", "gdp_std"]
    in_sample = path["gdp"].reindex(mu.index)
    assert float(np.sqrt(np.mean((in_sample - mu) ** 2))) < 0.4
    assert abs(in_sample.iloc[-1] - 0.5) < 0.5
    assert in_sample.iloc[:60].mean() > in_sample.iloc[-60:].mean() + 1.5
    constant = MixedFreqDFM(max_iter=300).fit(frame.drop(columns="cons"), "gdp", frequency=FREQ)

    def rmse(r):
        est = r.nowcast["out_of_sample"].reindex(truth.index)
        return float(np.sqrt(np.mean((est - truth) ** 2)))

    assert rmse(res) < 0.5 * rmse(constant)
    assert "long_run_mean" in res.nowcast.columns
    assert "long_run_mean" not in constant.nowcast.columns
    assert constant.long_run_mean is None


def test_long_run_results_contents(fit_tv):
    res = fit_tv
    assert res.params["long_run_loadings"]["gdp"] == pytest.approx(1.0)
    assert res.params["long_run_variance"] > 0
    assert "Long-run mean" in res.summary()
    assert res.state_layout is not None and res.state_layout.has_long_run
    # the common component (in-sample fit) includes the long-run mean
    obs = res.nowcast["observed"].dropna()
    fit = res.nowcast["in_sample"].reindex(obs.index)
    assert float(np.corrcoef(obs, fit)[0, 1]) > 0.9
    native = res.nowcast["long_run_mean"].dropna()
    assert native.index.freqstr.startswith("Q")


def test_fixed_long_run_variance_and_extra_series(trend_panel):
    frame, _, _ = trend_panel
    res = MixedFreqDFM(
        long_run_mean="time_varying",
        long_run_series=["cons"],
        long_run_variance=2e-3,
        max_iter=200,
    ).fit(frame, "gdp", frequency=FREQ)
    assert res.params["long_run_variance"] == 2e-3
    lr = res.long_run_mean
    assert lr is not None and {"gdp", "cons", "cons_std"} <= set(lr.columns)
    # cons trend = 0.8 mu in original units: its long-run mean declines by 0.8 * 2.5
    drop = lr["cons"].iloc[:24].mean() - lr["cons"].iloc[-24:].mean()
    assert 1.2 < drop < 2.8


def test_long_run_monotone_em(trend_panel):
    frame, _, _ = trend_panel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = MixedFreqDFM(long_run_mean="time_varying", max_iter=30, tol=0.0).fit(
            frame.drop(columns="cons"), "gdp", frequency=FREQ
        )
    path = res.loglikelihood_path
    assert path is not None
    assert np.all(np.diff(path) >= -1e-7 * np.abs(path[1:]))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"long_run_mean": "linear"}, "long_run_mean"),
        ({"long_run_series": ["x0"]}, "time_varying"),
        ({"long_run_variance": 0.1}, "time_varying"),
        ({"long_run_mean": "time_varying", "long_run_variance": -1.0}, "long_run_variance"),
        ({"long_run_mean": "time_varying", "long_run_series": ["nope"]}, "unknown"),
    ],
)
def test_long_run_parameter_validation(trend_panel, kwargs, match):
    frame, _, _ = trend_panel
    with pytest.raises(ValueError, match=match):
        MixedFreqDFM(**kwargs).fit(frame, "gdp", frequency=FREQ)


def test_warm_start_requires_trend_parameters(fit_tv, trend_panel):
    frame, _, _ = trend_panel
    data = frame.drop(columns="cons")
    res = MixedFreqDFM(long_run_mean="time_varying", init=fit_tv, max_iter=3).fit(
        data, "gdp", frequency=FREQ
    )
    assert res.long_run_mean is not None
    params = fit_tv.em_parameters
    assert params is not None
    stripped = EMParameters(
        params.transition,
        params.factor_cov,
        params.loadings,
        params.idio_ar,
        params.idio_var,
        params.obs_var,
    )
    with pytest.raises(ValueError, match="init parameters"):
        MixedFreqDFM(long_run_mean="time_varying", init=stripped).fit(data, "gdp", frequency=FREQ)


def test_doctests():
    failures, _ = doctest.testmod(long_run_module, optionflags=doctest.ELLIPSIS)
    assert failures == 0

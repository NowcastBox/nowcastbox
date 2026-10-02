"""Tests of the large mixed-frequency BVAR (``models/bvar.py``, ``models/_bvar_blocking.py``).

Validation (plan, item 11): blocking/unblocking, conditional forecasts equal to the
analytic conditional expectation (Kalman smoother, closed form and the fitted model),
posterior densities, news through the linear-Gaussian representation, the ``"bvar"``
extrapolator, pseudo real-time use without look-ahead.
"""

from __future__ import annotations

import doctest
import pickle
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

import nowcastbox as nb
import nowcastbox.models._bvar_blocking as bb
import nowcastbox.models.bvar as bv
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import ConvergenceWarning, NowcastDataError
from nowcastbox.density import NowcastDistribution
from nowcastbox.models import BridgeCombination, LargeBVAR, LargeBVARResults
from nowcastbox.models._bvar_blocking import (
    VARParameters,
    balanced_run,
    block_values,
    blocked_layout,
    conditional_forecast,
    conditional_moments,
    initial_state,
    native_periods,
    unblock_values,
    var_state_space,
    window_end,
)
from nowcastbox.models._bvar_prior import BVARHyperparameters
from nowcastbox.models.bvar_extrapolation import BVARExtrapolator
from nowcastbox.models.extrapolation import available_extrapolators, make_extrapolator
from tests.models.bvar_helpers import (
    analytic_conditional,
    blocked_panel,
    random_covariance,
    simulate_var,
    stable_coefficients,
)


def _dist(res: LargeBVARResults, **kwargs: Any) -> NowcastDistribution:
    out = res.distribution(**kwargs)
    assert isinstance(out, NowcastDistribution)
    return out


@pytest.fixture(scope="module")
def sim() -> Any:
    return nb.simulate.dfm(n_series=5, n_factors=1, n_periods=120, random_state=3)


@pytest.fixture(scope="module")
def res(sim: Any) -> LargeBVARResults:
    out = LargeBVAR(lags=2).fit(sim.data, "gdp")
    assert isinstance(out, LargeBVARResults)
    return out


# ====================================================================== blocking
def _small_panel() -> MixedFrequencyData:
    idx = pd.period_range("2020-02", periods=8, freq="M")
    df = pd.DataFrame(
        {
            "ip": np.arange(8.0),
            "gdp": [np.nan, 1.0, np.nan, np.nan, 2.0, np.nan, np.nan, np.nan],
        },
        index=idx,
    )
    return MixedFrequencyData(df, {"ip": "M", "gdp": "Q"})


def test_blocking_round_trip() -> None:
    panel = _small_panel()
    layout = blocked_layout(panel)
    assert layout.columns == ("ip[m1]", "ip[m2]", "ip[m3]", "gdp")
    assert layout.n == 4
    assert layout.source_name(3) == "gdp"
    assert layout.columns_of("gdp") == [3]
    with pytest.raises(KeyError):
        layout.columns_of("nope")
    quarters, values = block_values(panel.data, layout)
    assert [str(q) for q in quarters] == ["2020Q1", "2020Q2", "2020Q3"]
    np.testing.assert_array_equal(values[1], [2.0, 3.0, 4.0, 2.0])
    assert np.isnan(values[0, 0])
    back = unblock_values(values, quarters, layout)
    pd.testing.assert_frame_equal(
        back.loc[panel.index], panel.data, check_names=False, check_freq=False
    )
    months = native_periods(quarters, layout, 1)
    assert [str(m) for m in months] == ["2020-02", "2020-05", "2020-08"]
    assert native_periods(quarters, layout, 3).equals(quarters)


def test_blocking_errors() -> None:
    idx = pd.period_range("2020-01", periods=24, freq="M")
    frame = pd.DataFrame({"x": np.arange(24.0), "a": np.nan}, index=idx)
    frame.loc[idx[11], "a"] = 1.0
    frame.loc[idx[23], "a"] = 2.0
    annual = MixedFrequencyData(frame, {"x": "M", "a": "A"})
    with pytest.raises(NowcastDataError, match="monthly and quarterly"):
        blocked_layout(annual)
    widx = pd.period_range("2020-01-06", periods=10, freq="W")
    weekly = MixedFrequencyData(pd.DataFrame({"w": np.arange(10.0)}, index=widx), "W")
    with pytest.raises(NowcastDataError, match="monthly base"):
        blocked_layout(weekly)
    layout = blocked_layout(_small_panel())
    with pytest.raises(NowcastDataError, match="monthly PeriodIndex"):
        block_values(pd.DataFrame({"ip": [1.0]}), layout)
    with pytest.raises(NowcastDataError, match="do not contain"):
        block_values(_small_panel().data[["ip"]], layout)


def test_balanced_run_and_window_end() -> None:
    v = np.array([[1.0], [2.0], [np.nan], [3.0], [4.0], [np.nan]])
    assert balanced_run(v) == (3, 4)
    assert window_end(v, 2) == 4
    with pytest.raises(NowcastDataError, match="No quarter"):
        balanced_run(np.full((3, 2), np.nan))
    with pytest.raises(NowcastDataError, match="consecutive"):
        window_end(np.array([[1.0], [np.nan], [2.0]]), 2)
    with pytest.raises(NowcastDataError, match="consecutive"):
        window_end(np.array([[1.0]]), 2)


# ====================================================================== VAR algebra
def test_var_parameters_from_stacked_and_fitted(rng: np.random.Generator) -> None:
    n, p = 3, 2
    A = stable_coefficients(n, p, rng)
    c = rng.standard_normal(n)
    B = np.vstack([c[None, :], *[A[l].T for l in range(p)]])
    par = VARParameters.from_stacked(B, np.eye(n), p)
    np.testing.assert_allclose(par.coefficients, A)
    np.testing.assert_allclose(par.intercept, c)
    y = rng.standard_normal((6, n))
    fitted = par.fitted(y)
    assert np.isnan(fitted[:p]).all()
    np.testing.assert_allclose(fitted[4], c + A[0] @ y[3] + A[1] @ y[2])
    np.testing.assert_allclose(par.one_step(y[2:4]), fitted[4])
    nc = VARParameters.from_stacked(B[1:], np.eye(n), p, constant=False)
    np.testing.assert_array_equal(nc.intercept, np.zeros(n))
    assert par.impulse_responses(0).shape == (0, n, n)
    phi = par.impulse_responses(3)
    np.testing.assert_allclose(phi[2], A[0] @ A[0] + A[1])


def test_initial_state_with_missing_presample(rng: np.random.Generator) -> None:
    n, p = 2, 3
    A = stable_coefficients(n, p, rng)
    sigma = random_covariance(n, rng)
    par = VARParameters(np.ones(n), A, sigma)
    pre = rng.standard_normal((p, n))
    a0, P0 = initial_state(par, pre)
    np.testing.assert_allclose(a0[:n], par.one_step(pre))
    np.testing.assert_allclose(a0[n : 2 * n], pre[-1])
    np.testing.assert_allclose(P0[:n, :n], sigma)
    assert np.allclose(P0[n:], 0.0)
    pre[-2, 0] = np.nan  # lag 2 of variable 0 missing
    a0, P0 = initial_state(par, pre)
    big = bb._DIFFUSE_VARIANCE
    assert P0[2 * n, 2 * n] == big
    np.testing.assert_allclose(P0[:n, 2 * n], A[1][:, 0] * big)
    np.testing.assert_allclose(P0[:n, :n], sigma + big * np.outer(A[1][:, 0], A[1][:, 0]))
    assert np.all(np.linalg.eigvalsh(P0) > -1e-8 * big)
    with pytest.raises(ValueError, match="presample"):
        initial_state(par, pre[:2])
    assert var_state_space(par, np.zeros((p, n))).n_states == n * p


@pytest.mark.parametrize("lags", [1, 3])
def test_conditional_forecast_equals_analytic(rng: np.random.Generator, lags: int) -> None:
    n, h = 5, 4
    A = stable_coefficients(n, lags, rng)
    c = 0.3 * rng.standard_normal(n)
    sigma = random_covariance(n, rng)
    par = VARParameters(c, A, sigma)
    pre = rng.standard_normal((lags, n))
    future = rng.standard_normal((h, n))
    future[1, [0, 3]] = np.nan
    future[2:, 1:] = np.nan
    future[3, 0] = np.nan
    ref_mean, ref_var = analytic_conditional(A, c, sigma, pre, future)
    fc = conditional_forecast(par, pre, future)
    np.testing.assert_allclose(fc.mean, ref_mean, atol=1e-9)
    np.testing.assert_allclose(fc.variance, ref_var, atol=1e-9)
    cells = np.argwhere(np.isnan(future))
    mean, var = conditional_moments(par, pre, future, cells)
    np.testing.assert_allclose(mean, ref_mean[np.isnan(future)], atol=1e-9)
    np.testing.assert_allclose(var, ref_var[np.isnan(future)], atol=1e-9)
    empty = np.full((h, n), np.nan)
    m0, v0 = conditional_moments(par, pre, empty, cells)
    r0, rv = analytic_conditional(A, c, sigma, pre, np.where(np.isnan(empty), np.nan, 0))
    np.testing.assert_allclose(m0, r0[np.isnan(future)], atol=1e-9)
    np.testing.assert_allclose(v0, rv[np.isnan(future)], atol=1e-9)
    with pytest.raises(ValueError, match="complete pre-sample"):
        conditional_moments(par, np.full((lags, n), np.nan), future, cells)


def test_fitted_model_nowcast_is_analytic_conditional_expectation(
    rng: np.random.Generator,
) -> None:
    A = stable_coefficients(7, 2, rng)
    sigma = random_covariance(7, rng)
    values = simulate_var(A, 0.1 * np.ones(7), sigma, 60, rng)
    values[-1, 2:] = np.nan  # current quarter: only x1[m1], x1[m2] released
    values[-2, 6] = np.nan  # previous quarter's q1 not yet released
    panel = blocked_panel(values, n_monthly=2, n_quarterly=1)
    res = LargeBVAR(lags=2, prior={"lambda": 0.3}, standardize=False).fit(panel, "q1")
    eng = res.bvar
    assert eng is not None
    par = eng.parameters
    end = window_end(eng.values, 2)
    assert end == len(values) - 3
    mean, var = analytic_conditional(
        par.coefficients, par.intercept, par.sigma, values[end - 1 : end + 1], values[end + 1 :]
    )
    oos = res.out_of_sample.dropna()
    assert [str(p) for p in oos.index] == ["2004Q3", "2004Q4"]
    np.testing.assert_allclose(oos.to_numpy(), mean[:, 6], atol=1e-9)
    np.testing.assert_allclose(
        res.nowcast["std"].dropna().to_numpy(), np.sqrt(var[:, 6]), atol=1e-9
    )
    # predict() on the estimation data reproduces the edge
    pred = res.predict()
    np.testing.assert_allclose(pred["x2"].iloc[-3:].to_numpy(), mean[-1, 3:6], atol=1e-9)


# ====================================================================== estimator API
def test_fit_basic_results(res: LargeBVARResults, sim: Any) -> None:
    frame = res.nowcast
    assert {"observed", "in_sample", "out_of_sample", "std", "lower_90", "upper_90"} <= set(
        frame.columns
    )
    both = frame["in_sample"].notna() & frame["out_of_sample"].notna()
    assert not both.any()
    assert pd.PeriodIndex(frame.index).freqstr == "Q-DEC"
    oos = frame["out_of_sample"].dropna()
    assert len(oos) >= 1
    assert (frame.loc[oos.index, "std"] > 0).all()
    assert (frame.loc[oos.index, "lower_68"] < oos).all()
    assert res.window_end is not None and res.window_end < oos.index[0]
    assert res.n_variables == 16
    assert res.coefficients().shape == (1 + 2 * 16, 16)
    assert res.blocked_data().shape[1] == 16
    assert res.smoothed_data is not None
    assert pd.PeriodIndex(res.smoothed_data.index).freqstr == "M"
    text = res.summary()
    assert "Bayesian VAR" in text and "lambda" in text
    assert res.converged is True
    assert res.loglikelihood is not None and np.isfinite(res.loglikelihood)
    assert res.info["n_variables"] == 16
    assert res.model_params["lags"] == 2
    # in-sample values are one-step VAR predictions, close to the data on average
    ins = frame["in_sample"].dropna()
    err = (ins - frame.loc[ins.index, "observed"]).abs().mean()
    assert err < frame["observed"].std()


def test_formula_target_and_clone(sim: Any) -> None:
    model = LargeBVAR(lags=1, prior={"lambda": 0.2})
    res = model.fit(sim.data, "gdp ~ x01 + x02")
    assert res.n_variables == 7
    assert model.is_fitted
    clone = model.clone()
    assert clone.get_params() == model.get_params()
    assert not clone.is_fitted


def test_fixed_priors_and_options(sim: Any) -> None:
    data = sim.data.select(["x01", "x02", "gdp"])
    base = LargeBVAR(lags=1, prior={"lambda": 0.2}).fit(data, "gdp")
    assert base.bvar is not None and base.bvar.selection is None
    assert base.params["lambda"] == 0.2 and base.params["mu"] is None
    psi = base.params["psi"]
    hyper = BVARHyperparameters(0.2, psi)
    same = LargeBVAR(lags=1, prior=hyper).fit(data, "gdp")
    assert same.get_nowcast() == pytest.approx(base.get_nowcast())
    with pytest.raises(ValueError, match="psi"):
        LargeBVAR(lags=1, prior=BVARHyperparameters(0.2, np.ones(1))).fit(data, "gdp")
    soc = LargeBVAR(
        lags=1,
        prior={"lambda": 0.2, "mu": 0.5, "psi": 1.0},
        sum_of_coefficients=True,
        initial_observation=True,
    ).fit(data, "gdp")
    assert soc.params["mu"] == 0.5 and soc.params["delta"] == 1.0
    np.testing.assert_allclose(soc.params["psi"], 1.0)
    mapped = LargeBVAR(lags=1, prior={"lambda": 0.2}, prior_mean={"x01": "random_walk"}).fit(
        data, "gdp"
    )
    assert mapped.bvar is not None
    np.testing.assert_allclose(mapped.bvar.settings.own_lag_mean(7), [1, 1, 1, 0, 0, 0, 0])
    numeric = LargeBVAR(lags=1, prior={"lambda": 0.2}, prior_mean=0.5).fit(data, "gdp")
    assert numeric.bvar is not None
    np.testing.assert_allclose(numeric.bvar.settings.own_lag_mean(7), 0.5)


def test_glp_levels_with_dummy_priors(rng: np.random.Generator) -> None:
    A = stable_coefficients(4, 1, rng, radius=0.5)
    growth = simulate_var(A, np.zeros(4), random_covariance(4, rng), 70, rng)
    levels = 100.0 + np.cumsum(growth, axis=0)
    panel = blocked_panel(levels, n_monthly=1, n_quarterly=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = LargeBVAR(
            lags=2,
            prior_mean="random_walk",
            sum_of_coefficients=True,
            initial_observation=True,
            estimate_psi=True,
        ).fit(panel, "q1")
    assert res.bvar is not None and res.bvar.selection is not None
    assert res.bvar.selection.names[:3] == ("lambda", "mu", "delta")
    assert len(res.bvar.selection.names) == 3 + 4
    assert abs(res.get_nowcast() - levels[-1, 3]) < 10.0
    text = res.summary()
    assert "mu (SoC)" in text and "delta (DIO)" in text
    assert "Conditioning from" not in res.replace(window_end=None).summary()


def test_standardize_false_is_nearly_scale_invariant(sim: Any) -> None:
    data = sim.data.select(["x01", "x02", "x03", "gdp"])
    a = LargeBVAR(lags=1, prior={"lambda": 0.2}).fit(data, "gdp")
    b = LargeBVAR(lags=1, prior={"lambda": 0.2}, standardize=False).fit(data, "gdp")
    assert b.standardization is not None
    assert (b.standardization.std == 1.0).all()
    np.testing.assert_allclose(
        a.out_of_sample.dropna().to_numpy(), b.out_of_sample.dropna().to_numpy(), rtol=1e-3
    )


def test_monthly_target_and_horizon(sim: Any) -> None:
    data = sim.data.select(["x01", "x02", "gdp"])
    res = LargeBVAR(lags=1, prior={"lambda": 0.2}).fit(data, "x01")
    assert pd.PeriodIndex(res.nowcast.index).freqstr == "M"
    oos = res.out_of_sample.dropna()
    assert len(oos) >= 1 and oos.index[0] > data.to_native("x01", dropna=True).index[-1]
    with pytest.raises(NotImplementedError, match="quarterly target"):
        res.news(data, data)
    with pytest.raises(NotImplementedError, match="quarterly target"):
        res.level_contributions()
    far = LargeBVAR(lags=1, prior={"lambda": 0.2}).fit(data, "gdp", horizon=2)
    near = LargeBVAR(lags=1, prior={"lambda": 0.2}).fit(data, "gdp")
    assert far.nowcast.index[-1] == near.nowcast.index[-1] + 2


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"lags": 0}, "lags"),
        ({"lags": True}, "lags"),
        ({"max_iter": 0}, "max_iter"),
        ({"prior": "minnesota"}, "prior must be"),
        ({"prior": {"mu": 1.0}}, "lambda"),
        ({"prior": {"lambda": 0.2, "kappa": 1.0}}, "lambda"),
        ({"prior_mean": "levels"}, "prior_mean"),
        ({"prior_mean": {"x01": "levels"}}, "prior_mean"),
        ({"prior_mean": [1.0]}, "numeric"),
        ({"prior_mean": float("nan")}, "finite"),
        ({"sum_of_coefficients": 1}, "bool"),
        ({"lag_decay": 0.0}, "lag_decay"),
        ({"n_draws": -1}, "n_draws"),
    ],
)
def test_invalid_parameters(sim: Any, params: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        LargeBVAR(**params).fit(sim.data, "gdp")


def test_invalid_fit_options_and_short_data(sim: Any) -> None:
    with pytest.raises(TypeError, match="fit options"):
        LargeBVAR().fit(sim.data, "gdp", bogus=1)
    with pytest.raises(ValueError, match="horizon"):
        LargeBVAR().fit(sim.data, "gdp", horizon=-1)
    short = sim.data.truncate(end="2000-12")
    with pytest.raises(NowcastDataError, match="consecutive"):
        LargeBVAR(lags=2).fit(short, "gdp")
    with pytest.raises(ValueError, match="lags"):
        bv.fit_blocked_bvar(sim.data, lags=0)


def test_interior_gap_shortens_sample_with_warning(sim: Any) -> None:
    from nowcastbox.core.exceptions import DataQualityWarning

    frame = sim.data.data.copy()
    frame.loc[frame.index[30], "x01"] = np.nan  # 2002-07: 2002Q3 incomplete
    gappy = sim.data.with_data(frame)
    with pytest.warns(DataQualityWarning, match=r"2002Q4 onwards.*2002Q3 \(x01\)"):
        res = LargeBVAR(prior={"lambda": 0.2}).fit(gappy, "gdp")
    assert res.info["estimation_sample"][0] == pd.Period("2002Q4", "Q")
    many = frame.copy()
    many.loc[many.index[32], :] = np.nan  # 2002-09: every series, gdp included
    with pytest.warns(DataQualityWarning, match="and 1 more"):
        bv.fit_blocked_bvar(sim.data.with_data(many), prior={"lambda": 0.2})
    with warnings.catch_warnings():  # complete panel (or a partial first quarter): no warning
        warnings.simplefilter("error", DataQualityWarning)
        LargeBVAR(prior={"lambda": 0.2}).fit(sim.data, "gdp")
        LargeBVAR(prior={"lambda": 0.2}).fit(sim.data.truncate(start="2000-02"), "gdp")


def test_singular_posterior_raises_data_error(sim: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(bv, "posterior", broken)
    with pytest.raises(NowcastDataError, match="posterior"):
        LargeBVAR(prior={"lambda": 0.2}).fit(sim.data, "gdp")


def test_results_without_engine_and_data(res: LargeBVARResults) -> None:
    bare = res.replace(bvar=None)
    with pytest.raises(ValueError, match="fitted BVAR"):
        bare.coefficients()
    assert "Bayesian VAR" not in bare.summary()
    nodata = res.replace(data=None)
    with pytest.raises(ValueError, match="No data"):
        nodata.blocked_data()
    with pytest.raises(ValueError, match="No data"):
        nodata.predict()
    with pytest.raises(NowcastDataError, match="do not contain"):
        res.bvar.standardized_blocks(res.data.data[["x01"]])  # type: ignore[union-attr]


def test_predict_on_other_vintages(res: LargeBVARResults, sim: Any) -> None:
    old = sim.data.truncate(end=sim.data.index[-8])
    pred = res.predict(old)
    assert pred.index[-1] == res.predict().index[-1]
    assert pred.iloc[-1].notna().all()
    stamped = sim.data.data.copy()
    stamped.index = stamped.index.to_timestamp(how="end").normalize()
    by_date = res.predict(stamped)
    np.testing.assert_allclose(by_date.to_numpy(), res.predict().to_numpy(), equal_nan=True)
    eng = res.bvar
    assert eng is not None
    _, mean, _ = eng.edge(eng.values)
    end, again, var = eng.edge(mean)
    assert end == len(mean) - 1
    np.testing.assert_array_equal(again, mean)
    assert not var.any()


def test_target_released_after_the_window(rng: np.random.Generator) -> None:
    A = stable_coefficients(7, 1, rng)
    values = simulate_var(A, np.zeros(7), np.eye(7), 40, rng)
    values[-2, 3:6] = np.nan  # x2 late in the previous quarter, q1 released
    values[-1, 1:] = np.nan
    panel = blocked_panel(values, n_monthly=2, n_quarterly=1)
    res = LargeBVAR(prior={"lambda": 0.2}).fit(panel, "q1")
    assert res.window_end is not None
    assert str(res.window_end) == str(res.nowcast.index[-3])
    locs, _ = res.posterior_mixture(4, random_state=0)
    assert list(locs.index) == list(res.out_of_sample.dropna().index)
    assert res.in_sample.loc[res.nowcast.index[-2]] == res.in_sample.loc[res.nowcast.index[-2]]


def test_save_and_load(res: LargeBVARResults, tmp_path: Any) -> None:
    path = res.save(tmp_path / "bvar.pkl")
    loaded = LargeBVARResults.load(path)
    assert loaded.get_nowcast() == res.get_nowcast()
    pd.testing.assert_frame_equal(loaded.predict(), res.predict())
    assert pickle.loads(pickle.dumps(res.bvar)).layout == res.bvar.layout  # type: ignore[union-attr]


# ====================================================================== densities
def test_posterior_draws_density(sim: Any) -> None:
    data = sim.data.select(["x01", "x02", "x03", "gdp"])
    res = LargeBVAR(lags=1, n_draws=300, random_state=1).fit(data, "gdp")
    assert res.draw_locs is not None and res.draw_scales is not None
    assert res.draw_locs.shape[1] == 300
    dist = _dist(res)
    assert dist.n_components == 300 and dist.info["source"] == "posterior"
    period = dist.index[-1]
    point = res.get_nowcast(period)
    assert dist.point.iloc[-1] == pytest.approx(point)
    # the mixture mean is close to the nowcast at the posterior mean of the parameters
    assert abs(dist.mean.iloc[-1] - point) < 0.25 * dist.std.iloc[-1]
    # the std column is the mixture std; parameter uncertainty widens the Kalman std
    assert float(res.nowcast["std"].iloc[-1]) == pytest.approx(dist.std.iloc[-1])
    kalman = LargeBVAR(lags=1).fit(data, "gdp")
    assert dist.std.iloc[-1] > float(kalman.nowcast["std"].iloc[-1])
    analytic = _dist(res, method="analytic")
    assert analytic.is_gaussian
    fresh = _dist(kalman, n_draws=50, random_state=0)
    assert fresh.n_components == 50
    assert _dist(kalman).is_gaussian
    again = _dist(kalman, method="posterior", random_state=0, periods=[str(period)])
    assert again.n_components == bv._DEFAULT_DRAWS
    from nowcastbox.evaluation.scoring import crps, pit

    crps_value = crps(dist, [point])
    assert np.isfinite(crps_value).all()
    assert 0.0 < float(pit(dist, [point])[0]) < 1.0


def test_distribution_errors(res: LargeBVARResults) -> None:
    with pytest.raises(ValueError, match="method"):
        _dist(res, method="magic")
    with pytest.raises(TypeError, match="Unexpected"):
        _dist(res, levels=3)
    with pytest.raises(NowcastDataError, match="No out-of-sample"):
        _dist(res, n_draws=5, periods=["2001Q1"])
    with pytest.raises(ValueError, match="n_draws"):
        res.posterior_mixture(0)
    with pytest.raises(ValueError, match="backtest"):
        _dist(res, method="empirical")
    with pytest.raises(ValueError, match="n_draws"):
        res.bvar.mixture(res.bvar.values, np.zeros((0, 2)), 0)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="after the last"):
        res.bvar.mixture(res.bvar.values, [(0, 0)], 3)  # type: ignore[union-attr]


def test_no_out_of_sample_target_period(res: LargeBVARResults) -> None:
    eng = res.bvar
    assert eng is not None
    _, mean, var = eng.edge(eng.values)
    complete = bv.dataclasses.replace(eng, values=mean)
    with pytest.raises(NowcastDataError, match="No out-of-sample"):
        res.replace(bvar=complete).posterior_mixture(3)
    parts = bv._TargetParts.build(eng, "gdp", mean, var, window_end(eng.values, eng.lags))
    none = bv.dataclasses.replace(parts, out_of_sample=np.zeros_like(parts.out_of_sample))
    assert bv._stored_draws(eng, "gdp", none, 3, 0) is None


# ====================================================================== news
def test_news_identity_and_mapping(res: LargeBVARResults, sim: Any) -> None:
    data = sim.data
    old = data.truncate(end=data.index[-3])
    news = res.news(old, data)
    assert news.check_identity()
    assert news.old_nowcast == pytest.approx(res.predict(old)["gdp"].iloc[-1])
    assert news.new_nowcast == pytest.approx(res.get_nowcast())
    rel = news.releases
    assert set(rel["series"]) <= set(data.columns)
    assert rel["slot"].map(lambda p: p.freqstr).eq("M").all()
    assert (rel["reference_period"] == rel["slot"]).all()
    assert news.info["series"] == list(data.columns)
    assert len(news.info["blocked_series"]) == res.n_variables
    assert len(news.info["block_labels"]) == len(data.columns)
    same = res.news(data, data)
    assert same.releases.empty and same.news_effect == 0.0
    by_series = news.to_frame("series")
    assert by_series["news"].sum() == pytest.approx(news.news_effect)
    assert set(by_series.index) <= set(data.columns)


def test_news_with_revisions_reestimation_and_categories(res: LargeBVARResults, sim: Any) -> None:
    data = sim.data
    old = data.truncate(end=data.index[-2])
    frame = data.data.copy()
    frame.iloc[-6, 0] += 0.5  # revise x01 inside the conditioning window
    new = data.with_data(frame)
    new_res = LargeBVAR(lags=2).fit(new, "gdp")
    news = res.news(old, new, new_results=new_res, categories={"x01": "soft"})
    assert news.check_identity()
    assert "x01" in news.revisions.index
    assert news.revisions.loc["x01", "n_revised"] == 1
    assert news.revisions.loc["x01", "category"] == "soft"
    assert news.reestimation_effect != 0.0
    assert news.new_nowcast == pytest.approx(new_res.get_nowcast())
    with pytest.raises(TypeError, match="Unexpected"):
        res.news(old, new, bogus=1)


def test_level_contributions_and_tracker(res: LargeBVARResults, sim: Any) -> None:
    lc = res.level_contributions()
    assert lc.check_identity()
    assert list(lc.contributions.index) == list(sim.data.columns)
    assert lc.nowcast == pytest.approx(res.get_nowcast())
    lc2 = res.level_contributions(sim.data, categories={"gdp": "hard"})
    assert lc2.contributions.loc["gdp", "category"] == "hard"
    with pytest.raises(NotImplementedError, match="nowcast_tracker"):
        res.nowcast_tracker(sim.data)


def test_linear_model_hook_type_check(res: LargeBVARResults) -> None:
    from nowcastbox.news._model import linear_model

    class Fake:
        def linear_nowcast_model(self) -> str:
            return "not a model"

    with pytest.raises(TypeError, match="LinearNowcastModel"):
        linear_model(Fake())  # type: ignore[arg-type]
    lin = linear_model(res, {"x01[m1]": "soft"})
    assert lin.categories["x01[m1]"] == "soft"


def test_categories_from_metadata(sim: Any) -> None:
    data = sim.data.select(["x01", "x02", "gdp"])
    data = data.with_metadata("x01", category="soft", blocks=("real",))
    res = LargeBVAR(lags=1, prior={"lambda": 0.2}).fit(data, "gdp")
    lin = res.linear_nowcast_model()
    assert lin.categories["x01[m2]"] == "soft"
    assert lin.blocks["x01[m3]"] == "real"
    assert lin.categories["gdp"] == "uncategorized"


# ====================================================================== extrapolator
def test_bvar_extrapolator(sim: Any) -> None:
    assert "bvar" in available_extrapolators()
    ext = make_extrapolator("bvar", lags=1, prior={"lambda": 0.2})
    assert isinstance(ext, BVARExtrapolator)
    assert ext.mode(sim.data, ["x01", "x02", "gdp"]) == "blocked"
    end = pd.Period("2010Q1", "Q")
    out = ext(sim.data, ["x01", "x02", "gdp"], end)
    for col in ("x01", "x02"):
        series = out[col]
        assert str(series.index[-1]) == "2010-03"
        native = sim.data.to_native(col)
        observed = native.dropna()
        pd.testing.assert_series_equal(
            series.loc[observed.index], observed, check_names=False, check_freq=False
        )
        assert series.loc[observed.index[-1] :].notna().all()
    assert str(out["gdp"].index[-1]) == "2010Q1"
    with pytest.raises(ValueError, match="lags"):
        BVARExtrapolator(lags=0)
    empty = sim.data.data.copy()
    empty["x05"] = np.nan
    with pytest.raises(NowcastDataError, match="x05"):
        ext(sim.data.with_data(empty), ["x05"], end)
    with pytest.raises(NowcastDataError):
        ext(sim.data.truncate(end="2000-09"), ["x01", "gdp"], end)


def test_bvar_extrapolator_no_observation_column(sim: Any) -> None:
    frame = sim.data.data.copy()
    frame["x05"] = np.nan
    frame.loc[frame.index[:5], "x05"] = 1.0
    panel = sim.data.with_data(frame)
    ext = BVARExtrapolator(prior={"lambda": 0.2})
    out = ext(panel, ["x01", "x02"], pd.Period("2009Q4", "Q"))
    assert set(out) == {"x01", "x02"}


def test_bridge_combination_with_bvar_extrapolation(sim: Any) -> None:
    model = BridgeCombination(
        max_monthly=1,
        max_quarterly=0,
        extrapolation="bvar",
        extrapolation_options={"prior": {"lambda": 0.2}},
    )
    res = model.fit(sim.data.select(["x01", "x02", "x03", "gdp"]), "gdp")
    assert np.isfinite(res.get_nowcast())


# ====================================================================== real time
def test_no_look_ahead_in_pseudo_real_time(sim: Any) -> None:
    vintage = pd.Timestamp("2008-11-20")
    month = pd.Period(vintage, freq="M")
    full = sim.data.as_of(vintage)
    truncated = sim.data.truncate(end=month).as_of(vintage)
    a = LargeBVAR(lags=1).fit(full.truncate(end=month), "gdp")
    b = LargeBVAR(lags=1).fit(truncated, "gdp")
    assert a.get_nowcast() == pytest.approx(b.get_nowcast(), rel=1e-12)
    # future data never changes the estimate at the vintage
    later = sim.data.truncate(end=month + 6).as_of(vintage).truncate(end=month)
    c = LargeBVAR(lags=1).fit(later, "gdp")
    assert c.get_nowcast() == pytest.approx(a.get_nowcast(), rel=1e-12)


def test_backtest_runs_and_matches_manual_fits(sim: Any) -> None:
    from nowcastbox.evaluation import BacktestResults, PseudoRealTimeBacktest

    data = sim.data.select(["x01", "x02", "x03", "gdp"])
    model = LargeBVAR(lags=1, prior={"lambda": 0.2})
    bt = PseudoRealTimeBacktest(
        model,
        data,
        "gdp",
        start="2008-06-15",
        end="2008-11-15",
        step="M",
        refit_every=2,
        target_offsets=(0,),
    )
    out = bt.run()
    assert isinstance(out, BacktestResults)
    frame = out.to_frame()
    assert frame["forecast"].notna().all()
    first = frame.iloc[0]
    vintage = pd.Timestamp(first["vintage"])
    panel = data.as_of(vintage)
    month = pd.Period(vintage, freq="M")
    period = first["target_period"]
    window = panel.truncate(end=month)
    needed = period.asfreq("M", how="E")
    if needed > window.end:
        window = window.extend(int((needed - window.end).n))
    manual = LargeBVAR(lags=1, prior={"lambda": 0.2}).fit(window, "gdp")
    assert first["forecast"] == pytest.approx(manual.get_nowcast(period))
    assert out.metrics().shape[0] >= 1


def test_doctests() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for module in (bb, bv):
            result = doctest.testmod(
                module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
            )
            assert result.failed == 0

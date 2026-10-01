"""Tests of :class:`nowcastbox.models.MixedFreqDFM` (EM of Banbura & Modugno, 2014)."""

from __future__ import annotations

import doctest
import warnings

import numpy as np
import pandas as pd
import pytest

import nowcastbox.models._em_steps as em_steps_module
import nowcastbox.models._init_conditions as init_module
import nowcastbox.models.em as em_module
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import (
    ConvergenceWarning,
    DataQualityWarning,
    ModelNotFittedError,
    NowcastDataError,
)
from nowcastbox.core.results import FactorResults, build_nowcast_frame
from nowcastbox.models import EMParameters, MixedFreqDFM, MixedFreqDFMResults, StateLayout
from nowcastbox.models._init_conditions import BLOCK_ORDERS
from nowcastbox.models.em import (
    _extension_periods,
    build_layout,
    resolve_blocks,
    run_em,
    series_weights,
)
from tests.models.test_em_simulation import simulate_mixed_dfm

REL_NOISE = 1e-9


def _assert_monotone(path: np.ndarray, rel: float = REL_NOISE) -> None:
    diffs = np.diff(path)
    tol = rel * np.abs(path[1:]) + 1e-10
    assert np.all(diffs >= -tol), diffs.min()


def _corr(a: pd.Series, b: pd.Series) -> float:
    x, y = a.align(b, join="inner")
    ok = x.notna() & y.notna()
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


@pytest.fixture(scope="module")
def sim_one():
    return simulate_mixed_dfm(n_periods=180, n_monthly=12, n_quarterly=2, seed=11)


@pytest.fixture(scope="module")
def fit_one(sim_one):
    model = MixedFreqDFM(n_factors=1, factor_lags=1, max_iter=300, tol=1e-6)
    return model, model.fit(sim_one.data, target="q0", frequency=sim_one.frequencies)


@pytest.fixture(scope="module")
def sim_blocks():
    return simulate_mixed_dfm(
        n_periods=200,
        n_monthly=16,
        n_quarterly=2,
        blocks={"global": 1, "real": 1},
        block_share=0.5,
        seed=12,
    )


@pytest.fixture(scope="module")
def fit_blocks(sim_blocks):
    model = MixedFreqDFM(n_factors=1, factor_lags=1, blocks=sim_blocks.blocks, tol=1e-5)
    return model.fit(sim_blocks.data, target="q0", frequency=sim_blocks.frequencies)


# ====================================================================== estimation
class TestEstimationSingleBlock:
    def test_converges_and_loglik_monotone(self, fit_one):
        _, res = fit_one
        assert res.converged
        assert res.n_iter == res.loglikelihood_path.size - 1
        assert res.loglikelihood == pytest.approx(res.loglikelihood_path[-1])
        _assert_monotone(res.loglikelihood_path)
        assert res.info["n_loglikelihood_decreases"] == 0
        assert res.loglikelihood_path[-1] > res.loglikelihood_path[0]

    def test_common_component_recovery(self, fit_one, sim_one):
        _, res = fit_one
        for col in ["m0", "m5", "q0", "q1"]:
            assert _corr(res.common_component[col], sim_one.common[col]) > 0.97, col

    def test_factor_recovery_up_to_sign(self, fit_one, sim_one):
        _, res = fit_one
        true = pd.Series(sim_one.factors["global"][:, 0], index=sim_one.data.index)
        assert abs(_corr(res.factors["f1"], true)) > 0.97

    def test_transition_and_idiosyncratic_parameters(self, fit_one, sim_one):
        _, res = fit_one
        true_a = float(sim_one.transition["global"][0, 0])
        assert res.transition.shape == (1, 1)
        assert abs(res.transition[0, 0] - true_a) < 0.1
        rho = res.idiosyncratic_ar[[f"m{i}" for i in range(12)]]
        assert abs(rho.mean() - 0.4) < 0.1
        assert (res.idiosyncratic_variance > 0).all()

    def test_quarterly_loadings_satisfy_aggregation(self, fit_one):
        _, res = fit_one
        lay = res.state_layout
        for name in ("q0", "q1"):
            i = lay.series.index(name)
            lam = res.em_parameters.loadings[i, lay.loading_index(i)]
            assert np.allclose(lam / lam[0], [1, 2, 3, 2, 1])
            assert res.loadings.loc[name, "f1"] == pytest.approx(lam[0])

    def test_smoothed_data_reproduces_observations(self, fit_one, sim_one):
        _, res = fit_one
        obs = sim_one.data
        smoothed = res.smoothed_data.reindex(obs.index)
        diff = (smoothed - obs).abs().max().max()
        assert diff < 0.05 * obs.std().min()

    def test_nowcast_frame(self, fit_one, sim_one):
        _, res = fit_one
        frame = res.nowcast
        assert frame.index.freqstr.startswith("Q")
        assert {"common", "std", "lower_68", "upper_90"} <= set(frame.columns)
        observed = frame["observed"].notna()
        assert frame.loc[observed, "in_sample"].notna().all()
        assert frame.loc[observed, "std"].isna().all()
        first = frame["observed"].first_valid_index()
        presample = frame.loc[frame.index < first]
        assert presample["out_of_sample"].isna().all() and presample["std"].isna().all()
        assert presample["common"].notna().all()  # the model's estimate is kept there
        oos = frame.loc[~observed & (frame.index > first)]
        assert oos["out_of_sample"].notna().all()
        assert (oos["std"] > 0).all()
        assert (oos["lower_90"] < oos["lower_68"]).all()
        assert (oos["upper_68"] < oos["upper_90"]).all()
        # in-sample estimates are the common component of the target
        assert np.allclose(frame.loc[observed, "in_sample"], frame.loc[observed, "common"])
        # the current quarter (hidden in the simulation) is the default nowcast
        assert res.get_nowcast() == pytest.approx(oos["out_of_sample"].iloc[-1])
        truth = sim_one.truth["q0"].iloc[-1]
        assert abs(res.get_nowcast() - truth) < 3 * oos["std"].iloc[-1]

    def test_results_api(self, fit_one, sim_one):
        model, res = fit_one
        assert isinstance(res, FactorResults)
        assert model.results_ is res
        assert res.model_name == "MixedFreqDFM"
        assert res.model_params["n_factors"] == 1
        assert list(res.factors.columns) == ["f1"]
        assert res.loadings.shape == (14, 1)
        assert res.n_factors == 1
        assert res.factor_lags == 1
        assert res.n_shocks == 1
        assert np.allclose(res.shock_loadings @ res.shock_loadings.T, res.params["factor_cov"])
        assert res.transition_matrices()[0].shape == (1, 1)
        assert set(res.params) >= {
            "transition_blocks",
            "factor_cov_blocks",
            "loadings_all_lags",
            "design",
            "obs_cov",
            "idiosyncratic_ar",
            "aggregation_weights",
        }
        assert res.params["aggregation_weights"]["q0"].tolist() == [1, 2, 3, 2, 1]
        assert res.params["loadings_all_lags"].columns[0] == "f1_L0"
        assert res.state_space.n_states == res.state_layout.n_states
        assert res.grid[-1] >= sim_one.data.index[-1]
        assert res.smoothed_state.shape == (len(res.grid), res.state_layout.n_states)
        assert res.standardization is not None
        text = res.summary()
        assert "EM estimation" in text
        assert "global (1)" in text
        assert "MixedFreqDFMResults" in repr(res)

    def test_block_factors(self, fit_one):
        _, res = fit_one
        assert res.block_factors("global").equals(res.factors)
        with pytest.raises(KeyError, match="Unknown block"):
            res.block_factors("nominal")

    def test_predict_and_smooth_reproduce_fit(self, fit_one):
        _, res = fit_one
        pred = res.predict()
        assert np.allclose(pred.to_numpy(), res.smoothed_data.to_numpy(), equal_nan=True)
        sm = res.smooth()
        assert np.allclose(sm.smoothed_state, res.smoothed_state)

    def test_predict_on_new_vintage(self, fit_one, sim_one):
        _, res = fit_one
        newer = sim_one.data.copy()
        newer.loc[newer.index[-1], "m0"] = 0.0
        later = pd.concat(
            [
                newer,
                pd.DataFrame(
                    np.nan,
                    index=pd.period_range(newer.index[-1] + 1, periods=4, freq="M"),
                    columns=newer.columns,
                ),
            ]
        )
        grid, values = res.observations(later)
        assert grid[-1] == later.index[-1]
        assert values.shape == (len(grid), 14)
        pred = res.predict(MixedFrequencyData(later, sim_one.frequencies))
        assert pred.index[-1] == later.index[-1]
        # the new observation is reproduced
        assert abs(pred.loc[newer.index[-1], "m0"]) < 0.05
        with pytest.raises(NowcastDataError, match="model series"):
            res.observations(later.drop(columns="m0"))
        with pytest.raises(NowcastDataError, match="PeriodIndex"):
            res.observations(later.reset_index(drop=True))

    def test_save_load_roundtrip(self, fit_one, tmp_path):
        _, res = fit_one
        path = res.save(tmp_path / "em.pkl")
        loaded = MixedFreqDFMResults.load(path)
        assert np.allclose(loaded.loglikelihood_path, res.loglikelihood_path)
        assert loaded.state_layout.is_compatible(res.state_layout)
        assert loaded.get_nowcast() == pytest.approx(res.get_nowcast())

    def test_warm_start(self, fit_one, sim_one):
        _, res = fit_one
        warm = MixedFreqDFM(n_factors=1, init=res, max_iter=50, tol=1e-6)
        res2 = warm.fit(sim_one.data, target="q0", frequency=sim_one.frequencies)
        assert res2.n_iter <= 2
        assert res2.loglikelihood >= res.loglikelihood - 1e-6 * abs(res.loglikelihood)
        params = MixedFreqDFM(n_factors=1, init=res.em_parameters, max_iter=0).fit(
            sim_one.data, target="q0", frequency=sim_one.frequencies
        )
        assert params.loglikelihood == pytest.approx(res.loglikelihood)


class TestNowcastAccuracy:
    def test_quarterly_target_nowcast_accuracy(self):
        sim = simulate_mixed_dfm(
            n_periods=240, n_monthly=20, n_quarterly=1, seed=21, quarterly_idio_scale=0.1
        )
        data = sim.data.copy()
        slots = np.flatnonzero(data.index.month % 3 == 0)
        hidden = slots[-5:-1]  # 4 quarters with known truth (last one already hidden)
        data.iloc[hidden, data.columns.get_loc("q0")] = np.nan
        res = MixedFreqDFM(n_factors=1, tol=1e-5).fit(data, "q0", frequency=sim.frequencies)
        periods = data.index[hidden].asfreq("Q")
        est = res.nowcast.loc[periods, "out_of_sample"].to_numpy()
        truth = sim.truth["q0"].to_numpy()[hidden]
        rmse = float(np.sqrt(np.mean((est - truth) ** 2)))
        q = sim.data["q0"].dropna()
        rmse_mean = float(np.sqrt(np.mean((q.mean() - truth) ** 2)))
        assert rmse < 0.35 * q.std()
        assert rmse < 0.5 * rmse_mean

    def test_monthly_target(self, sim_one):
        res = MixedFreqDFM(n_factors=1, tol=1e-4).fit(
            sim_one.data, target="m0", frequency=sim_one.frequencies
        )
        assert res.nowcast.index.freqstr == "M"
        oos = res.nowcast["out_of_sample"].dropna()
        assert oos.index[-1] == sim_one.data.index[-1]
        assert (
            abs(res.get_nowcast() - sim_one.truth["m0"].iloc[-1]) < 3 * res.nowcast["std"].iloc[-1]
        )

    def test_horizon_extends_forecasts(self, sim_one):
        data = sim_one.data.iloc[:-1]  # ends in the 2nd month of a quarter
        base = MixedFreqDFM(max_iter=3, tol=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            r0 = base.fit(data, "q0", frequency=sim_one.frequencies)
            r2 = base.fit(data, "q0", frequency=sim_one.frequencies, horizon=2)
        assert len(r2.nowcast) == len(r0.nowcast) + 2
        assert r2.grid[-1] == r0.grid[-1] + 6
        assert r0.grid[-1] == data.index[-1] + 1
        with pytest.raises(TypeError, match="Unknown fit options"):
            base.fit(data, "q0", frequency=sim_one.frequencies, horizn=1)
        with pytest.raises(ValueError, match="horizon"):
            base.fit(data, "q0", frequency=sim_one.frequencies, horizon=-1)

    def test_formula_target(self, sim_one):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            res = MixedFreqDFM(max_iter=2).fit(
                sim_one.data, target="q0 ~ m0 + m1 + m2", frequency=sim_one.frequencies
            )
        assert list(res.loadings.index) == ["m0", "m1", "m2", "q0"]


class TestIdiosyncraticIid:
    def test_iid_monotone_and_obs_cov(self, sim_one):
        res = MixedFreqDFM(idiosyncratic="iid", tol=1e-7, max_iter=300).fit(
            sim_one.data, "q0", frequency=sim_one.frequencies
        )
        _assert_monotone(res.loglikelihood_path)
        assert res.state_layout.n_states == 5
        assert np.allclose(res.idiosyncratic_ar, 0.0)
        assert np.allclose(res.params["obs_cov"], res.idiosyncratic_variance)
        assert _corr(res.common_component["q0"], sim_one.common["q0"]) > 0.97
        frame = res.nowcast
        oos = frame["observed"].isna() & frame["out_of_sample"].notna()
        assert np.allclose(frame.loc[oos, "out_of_sample"], frame.loc[oos, "common"])


class TestBlocks:
    def test_block_factor_recovery_up_to_rotation(self, fit_blocks, sim_blocks):
        res = fit_blocks
        assert list(res.factors.columns) == ["global_f1", "real_f1"]
        est = res.factors.reindex(sim_blocks.data.index).to_numpy()
        X = np.column_stack([np.ones(len(est)), est])
        for block in ("global", "real"):
            y = sim_blocks.factors[block][:, 0]
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            r2 = 1 - np.var(y - X @ beta) / np.var(y)
            assert r2 > 0.85, (block, r2)
        _assert_monotone(res.loglikelihood_path)

    def test_block_structure_of_parameters(self, fit_blocks, sim_blocks):
        res = fit_blocks
        non_members = sim_blocks.blocks.index[~sim_blocks.blocks["real"]]
        assert np.allclose(res.loadings.loc[non_members, "real_f1"], 0.0)
        A = res.transition
        assert A.shape == (2, 2)
        assert A[0, 1] == 0.0
        assert A[1, 0] == 0.0
        Q = res.params["factor_cov"]
        assert Q[0, 1] == 0.0
        assert set(res.params["transition_blocks"]) == {"global", "real"}
        assert res.block_factors("real").columns.tolist() == ["real_f1"]
        for name in ("q0", "q1"):
            assert _corr(res.common_component[name], sim_blocks.common[name]) > 0.95

    def test_blocks_from_metadata_and_mapping(self, sim_blocks):
        mfd = MixedFrequencyData(sim_blocks.data, sim_blocks.frequencies, blocks=sim_blocks.blocks)
        mapping = {
            s: [b for b in sim_blocks.blocks.columns if sim_blocks.blocks.loc[s, b]]
            for s in sim_blocks.blocks.index
        }
        lay_meta = build_layout(mfd, {"global": 1, "real": 1}, 2, "data", "ar1")
        lay_map = build_layout(mfd, 1, 2, mapping, "ar1")
        lay_arr = build_layout(mfd, 1, 2, sim_blocks.blocks.to_numpy().astype(int), "ar1")
        assert lay_meta.is_compatible(lay_map)
        assert np.array_equal(lay_arr.membership, lay_map.membership)
        assert lay_arr.block_names == ("block1", "block2")
        assert lay_meta.block_lags == (5, 5)


class TestBlockResolution:
    @pytest.fixture
    def panel(self):
        idx = pd.period_range("2020-01", periods=6, freq="M")
        df = pd.DataFrame(np.arange(18.0).reshape(6, 3) % 5, index=idx, columns=["a", "b", "c"])
        return MixedFrequencyData(df, "M")

    def test_default(self, panel):
        m, prefix = resolve_blocks(None, panel)
        assert m.columns.tolist() == ["global"]
        assert m.all().all()
        assert prefix is False

    def test_one_dimensional_array(self, panel):
        m, _ = resolve_blocks([1, 1, 1], panel)
        assert m.columns.tolist() == ["block1"]

    def test_dataframe_extra_rows_ignored(self, panel):
        frame = pd.DataFrame({"g": [1, 1, 1, 1], "r": [0, 1, 0, 1]}, index=["a", "b", "c", "z"])
        m, _ = resolve_blocks(frame, panel)
        assert m.index.tolist() == ["a", "b", "c"]
        assert m["r"].tolist() == [False, True, False]

    @pytest.mark.parametrize(
        ("spec", "match"),
        [
            ("auto", "blocks must be None"),
            ("data", "no block metadata"),
            (pd.DataFrame({"g": [1, 1]}, index=["a", "b"]), "no row"),
            (pd.DataFrame({"g": [1, 2, 1]}, index=["a", "b", "c"]), "0/1"),
            (pd.DataFrame({"g": ["x", "y", "z"]}, index=["a", "b", "c"]), "0/1"),
            ({"a": "g", "b": "g"}, "no entry"),
            (np.ones((2, 1)), "shape"),
            (np.ones((3, 1, 1)), "shape"),
            (pd.DataFrame({"g": [1, 1, 0]}, index=["a", "b", "c"]), "any block"),
        ],
    )
    def test_errors(self, panel, spec, match):
        with pytest.raises(ValueError, match=match):
            resolve_blocks(spec, panel)

    def test_empty_block_dropped_with_warning(self, panel):
        frame = pd.DataFrame({"g": [1, 1, 1], "r": [0, 0, 0]}, index=["a", "b", "c"])
        with pytest.warns(DataQualityWarning, match="contain no series"):
            m, _ = resolve_blocks(frame, panel)
        assert m.columns.tolist() == ["g"]

    def test_n_factors_mapping_errors(self, panel):
        with pytest.raises(ValueError, match="no entry for the blocks"):
            build_layout(panel, {"other": 1}, 1, None, "ar1")
        with pytest.raises(ValueError, match="3 series but 4 factors"):
            build_layout(panel, 4, 1, None, "ar1")
        lay = build_layout(panel, {"global": 2}, 1, None, "iid")
        assert lay.n_factors == (2,)


# ====================================================================== I1: frequencies
class TestArbitraryFrequencies:
    def test_series_weights_from_metadata(self):
        idx = pd.period_range("2020-01", periods=12, freq="M")
        df = pd.DataFrame(
            {
                "m": np.arange(12.0),
                "q_flow": np.where(idx.month % 3 == 0, 1.0, np.nan),
                "q_stock": np.where(idx.month % 3 == 0, 2.0, np.nan),
                "a": np.where(idx.month == 12, 1.0, np.nan),
            },
            index=idx,
        )
        mfd = MixedFrequencyData(
            df,
            {"m": "M", "q_flow": "Q", "q_stock": "Q", "a": "A"},
            aggregations={"q_flow": "flow", "q_stock": "stock", "a": "average"},
        )
        w = series_weights(mfd)
        assert w[0].tolist() == [1.0]
        assert w[1].tolist() == [1.0, 1.0, 1.0]
        assert w[2].tolist() == [1.0, 0.0, 0.0]
        assert np.allclose(w[3], np.full(12, 1 / 12))
        lay = build_layout(mfd, 1, 1, None, "ar1")
        assert lay.weights[2].tolist() == [1.0]  # stock: trailing zeros trimmed
        assert lay.block_lags == (12,)

    def test_flow_and_stock_aggregation_fit(self):
        rng = np.random.default_rng(3)
        n = 150
        f = np.zeros(n)
        for t in range(1, n):
            f[t] = 0.8 * f[t - 1] + rng.standard_normal()
        idx = pd.period_range("2005-01", periods=n, freq="M")
        x = np.outer(f, rng.uniform(0.5, 1.5, 8)) + 0.5 * rng.standard_normal((n, 8))
        df = pd.DataFrame(x, index=idx, columns=[f"m{i}" for i in range(8)])
        flow = np.convolve(f, [1, 1, 1])[:n] + 0.1 * rng.standard_normal(n)
        stock = f + 0.1 * rng.standard_normal(n)
        slot = idx.month % 3 == 0
        df["flow"] = np.where(slot, flow, np.nan)
        df["stock"] = np.where(slot, stock, np.nan)
        freqs = dict.fromkeys(df.columns[:8], "M") | {"flow": "Q", "stock": "Q"}
        mfd = MixedFrequencyData(df, freqs, aggregations={"flow": "flow", "stock": "stock"})
        res = MixedFreqDFM(tol=1e-5, max_iter=300).fit(mfd, target="flow")
        _assert_monotone(res.loglikelihood_path)
        lay = res.state_layout
        i = lay.series.index("flow")
        lam = res.em_parameters.loadings[i, lay.loading_index(i)]
        assert np.allclose(lam, lam[0])
        latent = pd.Series(np.convolve(f, [1, 1, 1])[:n], index=idx)
        assert abs(_corr(res.common_component["flow"], latent)) > 0.97
        stock_latent = pd.Series(f, index=idx)
        assert abs(_corr(res.common_component["stock"], stock_latent)) > 0.97

    def test_annual_series_on_quarterly_grid(self):
        rng = np.random.default_rng(5)
        n = 120
        f = np.zeros(n)
        for t in range(1, n):
            f[t] = 0.6 * f[t - 1] + rng.standard_normal()
        idx = pd.period_range("1990Q1", periods=n, freq="Q")
        x = np.outer(f, rng.uniform(0.5, 1.5, 6)) + 0.4 * rng.standard_normal((n, 6))
        df = pd.DataFrame(x, index=idx, columns=[f"x{i}" for i in range(6)])
        annual = np.convolve(f, [1, 2, 3, 4, 3, 2, 1])[:n] / 4 + 0.1 * rng.standard_normal(n)
        df["gdp"] = np.where(idx.quarter == 4, annual, np.nan)
        df.iloc[-1, -1] = np.nan
        freqs = dict.fromkeys(df.columns[:6], "Q") | {"gdp": "A"}
        res = MixedFreqDFM(tol=1e-6, max_iter=300).fit(df, target="gdp", frequency=freqs)
        _assert_monotone(res.loglikelihood_path)
        assert res.state_layout.block_lags == (7,)
        assert res.nowcast.index.freqstr.startswith("Y") or res.nowcast.index.freqstr.startswith(
            "A"
        )
        latent = pd.Series(np.convolve(f, [1, 2, 3, 4, 3, 2, 1])[:n], index=idx)
        assert abs(_corr(res.common_component["gdp"], latent)) > 0.95


# ====================================================================== validation
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_factors": 0}, "n_factors"),
        ({"n_factors": 1.5}, "n_factors"),
        ({"n_factors": True}, "n_factors"),
        ({"n_factors": {}}, "must not be empty"),
        ({"n_factors": {"global": 0}}, "n_factors"),
        ({"factor_lags": 0}, "factor_lags"),
        ({"max_iter": -1}, "max_iter"),
        ({"tol": -1e-3}, "tol"),
        ({"tol": float("nan")}, "tol"),
        ({"tol": "small"}, "tol"),
        ({"obs_noise_var": -1.0}, "obs_noise_var"),
        ({"idiosyncratic": "cauchy"}, "idiosyncratic"),
        ({"filter_method": "fast"}, "filter_method"),
        ({"init": "random"}, "init"),
    ],
)
def test_parameter_validation(sim_one, kwargs, match):
    with pytest.raises(ValueError, match=match):
        MixedFreqDFM(**kwargs).fit(sim_one.data, "q0", frequency=sim_one.frequencies)


def test_get_params_and_clone():
    model = MixedFreqDFM(n_factors={"g": 2}, factor_lags=2, idiosyncratic="iid")
    params = model.get_params()
    assert params["n_factors"] == {"g": 2}
    assert params["filter_method"] == "auto"
    assert model.clone().get_params()["factor_lags"] == 2
    model._validate_params()  # a mapping of positive integers is valid
    with pytest.raises(ModelNotFittedError):
        _ = model.results_


def test_convergence_warning_and_zero_iterations(sim_one):
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        res = MixedFreqDFM(max_iter=2, tol=0.0).fit(
            sim_one.data, "q0", frequency=sim_one.frequencies
        )
    assert res.n_iter == 2
    assert res.converged is False
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        res0 = MixedFreqDFM(max_iter=0).fit(sim_one.data, "q0", frequency=sim_one.frequencies)
    assert res0.n_iter == 0
    assert "Initial log-lik." not in res0.summary()
    assert res0.loglikelihood_path.size == 1
    assert np.isnan(res0.info["last_relative_change"])


def test_univariate_and_multivariate_e_steps_agree(sim_one):
    kw = {"max_iter": 0}
    r_uni = MixedFreqDFM(**kw).fit(sim_one.data, "q0", frequency=sim_one.frequencies)
    r_multi = MixedFreqDFM(filter_method="multivariate", **kw).fit(
        sim_one.data, "q0", frequency=sim_one.frequencies
    )
    assert r_uni.loglikelihood == pytest.approx(r_multi.loglikelihood, rel=1e-8)
    assert np.allclose(r_uni.smoothed_state, r_multi.smoothed_state, atol=1e-6)


def test_warm_start_errors(fit_one, sim_one):
    _, res = fit_one
    with pytest.raises(ValueError, match="different model structure"):
        MixedFreqDFM(n_factors=1, idiosyncratic="iid", init=res).fit(
            sim_one.data, "q0", frequency=sim_one.frequencies
        )
    bare = res.replace(em_parameters=None)
    with pytest.raises(ValueError, match="do not contain EM parameters"):
        MixedFreqDFM(init=bare).fit(sim_one.data, "q0", frequency=sim_one.frequencies)
    p = res.em_parameters
    wrong = EMParameters(
        p.transition, p.factor_cov, p.loadings[:, :2], p.idio_ar, p.idio_var, p.obs_var
    )
    with pytest.raises(ValueError, match="do not match"):
        MixedFreqDFM(init=wrong).fit(sim_one.data, "q0", frequency=sim_one.frequencies)


def test_non_finite_loglikelihood_raises(monkeypatch, sim_one):
    """Dense E-step (its filter output is patched to produce a NaN log-likelihood)."""
    real = em_module.e_step

    def broken(*args, **kwargs):
        stats = real(*args, **kwargs)
        stats.smoother.filter_result.loglikelihood_obs[0] = np.nan
        return stats

    monkeypatch.setattr(em_module, "e_step", broken)
    with pytest.raises(NowcastDataError, match="Non-finite log-likelihood"):
        MixedFreqDFM(max_iter=1, filter_method="univariate").fit(
            sim_one.data, "q0", frequency=sim_one.frequencies
        )


def test_decrease_is_counted(monkeypatch):
    """A (synthetic) decrease of the log-likelihood is recorded, not hidden."""
    lay = StateLayout(["a", "b"], ["g"], [1], 1, np.ones((2, 1), bool), [[1.0], [1.0]], "iid")
    p = EMParameters(
        (np.array([[0.5]]),), (np.eye(1),), np.ones((2, 1)), np.zeros(2), np.ones(2), np.ones(2)
    )
    y = np.random.default_rng(1).standard_normal((40, 2))
    values = iter([-10.0, -12.0, -12.0])
    real = em_module.e_step

    def fake(*args, **kwargs):
        stats = real(*args, **kwargs)
        obs = stats.smoother.filter_result.loglikelihood_obs
        obs[:] = 0.0
        obs[0] = next(values)
        return stats

    monkeypatch.setattr(em_module, "e_step", fake)
    out = run_em(p, lay, y, max_iter=5, tol=1e-8)
    assert out.n_decreases == 1
    assert out.converged


def test_extension_periods():
    idx = pd.period_range("2020-01", periods=8, freq="M")  # ends in August (Q3 month 2)
    q = np.where(idx.month % 3 == 0, 1.0, np.nan)
    df = pd.DataFrame({"m": np.arange(8.0), "q": q}, index=idx)
    mfd = MixedFrequencyData(df, {"m": "M", "q": "Q"})
    assert _extension_periods(mfd, "q", 0) == 1
    assert _extension_periods(mfd, "q", 2) == 7
    assert _extension_periods(mfd, "m", 0) == 1  # monthly target observed up to the end
    full = mfd.extend(1).with_data(
        mfd.extend(1).data.assign(
            q=np.where(pd.period_range("2020-01", periods=9, freq="M").month % 3 == 0, 1.0, np.nan)
        )
    )
    assert _extension_periods(full, "q", 0) == 3


def test_results_without_model_raise():
    idx = pd.period_range("2020Q1", periods=2, freq="Q")
    frame = build_nowcast_frame(pd.Series([1.0, np.nan], idx), pd.Series([1.0, 2.0], idx))
    res = MixedFreqDFMResults(target="y", nowcast=frame)
    with pytest.raises(ValueError, match="state-space model"):
        res.block_factors("global")
    assert res._summary_sections()[-1][0] == "Factor dynamics"


def test_results_without_data_or_standardization(fit_one):
    _, res = fit_one
    with pytest.raises(ValueError, match="standardisation"):
        res.replace(standardization=None).smooth()
    with pytest.raises(ValueError, match="No data"):
        res.replace(data=None).observations()


def test_exports():
    import nowcastbox.models as models

    for name in ("MixedFreqDFM", "MixedFreqDFMResults", "EMParameters", "StateLayout"):
        assert name in models.__all__


@pytest.mark.parametrize("module", [em_module, em_steps_module, init_module])
def test_doctests(module):
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.failed == 0


@pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("method", ["structured", "univariate", "multivariate"])
def test_filter_methods_give_the_same_fit(sim_one, method):
    """The structured E-step and final pass (I2) reproduce the dense Kalman path."""
    kw = {"max_iter": 5, "tol": 0.0}
    ref = MixedFreqDFM(filter_method="auto", **kw).fit(
        sim_one.data, "q0", frequency=sim_one.frequencies
    )
    res = MixedFreqDFM(filter_method=method, **kw).fit(
        sim_one.data, "q0", frequency=sim_one.frequencies
    )
    np.testing.assert_allclose(res.loglikelihood_path, ref.loglikelihood_path, rtol=1e-9)
    num = ref.nowcast.select_dtypes("number").columns
    np.testing.assert_allclose(
        res.nowcast[num].to_numpy(dtype=float), ref.nowcast[num].to_numpy(dtype=float), atol=1e-8
    )
    assert res.filter_method == method


@pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.ConvergenceWarning")
def test_structured_final_pass_is_used_and_matches_smooth(sim_one):
    """``filter_method='structured'`` fits; ``smooth`` (dense) reproduces the states."""
    res = MixedFreqDFM(filter_method="structured", max_iter=3, tol=0.0).fit(
        sim_one.data, "q0", frequency=sim_one.frequencies
    )
    sm = res.smooth()
    assert res.smoothed_state is not None
    n = res.smoothed_state.shape[0]
    np.testing.assert_allclose(sm.smoothed_state[:n], res.smoothed_state, atol=1e-8)


@pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.ConvergenceWarning")
def test_structured_method_with_long_run_mean(sim_one):
    """The structured smoother handles the random-walk long-run state."""
    kw = {"max_iter": 3, "tol": 0.0, "long_run_mean": "time_varying", "long_run_variance": 1e-3}
    a = MixedFreqDFM(filter_method="structured", **kw).fit(
        sim_one.data, "q0", frequency=sim_one.frequencies
    )
    b = MixedFreqDFM(filter_method="univariate", **kw).fit(
        sim_one.data, "q0", frequency=sim_one.frequencies
    )
    assert a.long_run_mean is not None and b.long_run_mean is not None
    np.testing.assert_allclose(a.long_run_mean.to_numpy(), b.long_run_mean.to_numpy(), atol=1e-7)


# ====================================================================== start values
def test_single_block_start_is_sequential(fit_one):
    _, res = fit_one
    assert res.info["initialization"] == {"method": "pca", "block_order": "given"}


def test_multistart_picks_most_likely_block_order(fit_blocks):
    res = fit_blocks
    init = res.info["initialization"]
    scores = init["loglikelihood"]
    assert set(scores) == set(BLOCK_ORDERS)
    assert scores[init["block_order"]] == max(scores.values())


@pytest.mark.parametrize("order", BLOCK_ORDERS)
def test_forced_block_order(sim_blocks, order):
    model = MixedFreqDFM(n_factors=1, blocks=sim_blocks.blocks, init=f"pca_{order}", max_iter=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(sim_blocks.data, target="q0", frequency=sim_blocks.frequencies)
    assert res.info["initialization"]["block_order"] == order


def test_multistart_skips_failing_orders(sim_blocks, monkeypatch):
    real = em_module.pca_initial_parameters

    def flaky(panel, layout, *, obs_noise_var, block_order):
        if block_order != "independent":
            raise np.linalg.LinAlgError("degenerate")
        return real(panel, layout, obs_noise_var=obs_noise_var, block_order=block_order)

    monkeypatch.setattr(em_module, "pca_initial_parameters", flaky)
    model = MixedFreqDFM(n_factors=1, blocks=sim_blocks.blocks, max_iter=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(sim_blocks.data, target="q0", frequency=sim_blocks.frequencies)
    assert res.info["initialization"]["block_order"] == "independent"

    def broken(panel, layout, *, obs_noise_var, block_order):
        raise np.linalg.LinAlgError("degenerate")

    monkeypatch.setattr(em_module, "pca_initial_parameters", broken)
    with pytest.raises(np.linalg.LinAlgError):
        model.fit(sim_blocks.data, target="q0", frequency=sim_blocks.frequencies)


def test_multistart_without_finite_start(sim_blocks, monkeypatch):
    monkeypatch.setattr(em_module, "loglikelihood", lambda model, values: float("nan"))
    model = MixedFreqDFM(n_factors=1, blocks=sim_blocks.blocks, max_iter=0)
    with pytest.raises(NowcastDataError, match="finite starting"):
        model.fit(sim_blocks.data, target="q0", frequency=sim_blocks.frequencies)


def test_unknown_block_order():
    with pytest.raises(ValueError, match="block_order"):
        init_module._block_factors(np.zeros((3, 2)), None, np.ones(2, bool), "random")  # type: ignore[arg-type]

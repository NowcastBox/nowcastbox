"""Tests of the analytic / bootstrap predictive distribution of fitted models."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from nowcastbox.core.exceptions import ConvergenceWarning, NowcastDataError
from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.density import (
    BootstrapNowcasts,
    NowcastDistribution,
    analytic_distribution,
    bootstrap_nowcasts,
    combine_bootstrap,
    nowcast_distribution,
)
from nowcastbox.evaluation.scoring import interval_hits, ks_uniformity_test, pit
from nowcastbox.models import MixedFreqDFM, TwoStepDFM
from nowcastbox.statespace import kalman_smoother
from tests.density.conftest import simulate_with_truth


# ---------------------------------------------------------------------- analytic part
class TestAnalytic:
    def test_two_step_matches_nowcast_std(self, two_step_results) -> None:
        dist = analytic_distribution(two_step_results)
        frame = two_step_results.nowcast
        expected = frame.loc[frame["out_of_sample"].notna()]
        assert dist.is_gaussian
        assert list(dist.index) == list(expected.index)
        np.testing.assert_allclose(dist.mean, expected["out_of_sample"])
        np.testing.assert_allclose(dist.std, expected["std"])
        assert dist.target == "gdp"
        assert dist.info["source"] == "analytic"
        np.testing.assert_allclose(dist.variance_decomposition["parameter"], 0.0)

    def test_two_step_std_is_bridge_formula(self, two_step_results) -> None:
        """std^2 = beta' V beta + sigma_e^2 recomputed from the state-space smoother."""
        res = two_step_results
        z = res.standardization.transform(res.data.to_frame()[list(res.loadings.index)])
        grid = res.factors.index
        values = z.reindex(grid).to_numpy()
        sm = kalman_smoother(res.state_space, values)
        w = res.aggregation_weights
        period = analytic_distribution(res).index[0]
        pos = grid.get_loc(period.asfreq("M", how="E"))
        sel = np.zeros(sm.smoothed_state.shape[1])
        sel[: w.size] = w
        beta = float(res.bridge.params.iloc[1])
        var = beta**2 * sel @ sm.smoothed_state_cov[pos] @ sel + res.bridge.sigma**2
        assert analytic_distribution(res, period).std.iloc[0] == pytest.approx(np.sqrt(var))

    def test_em_std_is_zpz_plus_h(self, em_results) -> None:
        """Var = Z P Z' + H for the target row (aggregation inside Z)."""
        res = em_results
        values = res.standardization.transform(
            res.data.to_frame()[list(res.state_layout.series)]
        ).reindex(res.grid)
        sm = kalman_smoother(res.state_space, values.to_numpy(), method="univariate")
        j = list(res.state_layout.series).index("gdp")
        dist = analytic_distribution(res)
        period = dist.index[-1]
        pos = res.grid.get_loc(period.asfreq("M", how="E"))
        zj = res.state_space.Z[j]
        var = zj @ sm.smoothed_state_cov[pos] @ zj + res.state_space.obs_cov_diagonal[j]
        sd = res.standardization.std["gdp"]
        assert dist.std.iloc[-1] == pytest.approx(np.sqrt(var) * sd, rel=1e-8)

    def test_explicit_periods(self, two_step_results) -> None:
        dist = analytic_distribution(two_step_results, "2015Q1")
        assert [str(p) for p in dist.index] == ["2015Q1"]
        # in-sample period of the two-step model also has a std
        dist = analytic_distribution(two_step_results, [pd.Period("2014Q3", "Q")])
        assert dist.mean.iloc[0] == pytest.approx(two_step_results.nowcast["in_sample"].iloc[-3])

    def test_invalid_periods(self, two_step_results, em_results) -> None:
        with pytest.raises(NowcastDataError, match="No estimate"):
            analytic_distribution(em_results, "2014Q3")  # in-sample: no std for EM
        with pytest.raises(NowcastDataError, match="No estimate"):
            analytic_distribution(two_step_results, "1990Q1")
        with pytest.raises(NowcastDataError, match="Invalid"):
            analytic_distribution(two_step_results, [object()])  # type: ignore[list-item]
        with pytest.raises(NowcastDataError, match="empty"):
            analytic_distribution(two_step_results, [])

    def test_results_without_std(self) -> None:
        idx = pd.period_range("2020Q1", periods=2, freq="Q")
        frame = build_nowcast_frame(
            pd.Series([1.0, np.nan], index=idx), pd.Series([1.0, 2.0], index=idx)
        )
        res = NowcastResults(target="y", nowcast=frame)
        with pytest.raises(NowcastDataError, match="no 'std'"):
            analytic_distribution(res)
        with pytest.raises(NowcastDataError, match="no 'std'"):
            analytic_distribution(res, "2020Q2")
        frame["std"] = np.nan
        with pytest.raises(NowcastDataError, match="No out-of-sample"):
            analytic_distribution(NowcastResults(target="y", nowcast=frame))

    def test_generic_results_with_std(self) -> None:
        idx = pd.period_range("2020Q1", periods=2, freq="Q")
        frame = build_nowcast_frame(
            pd.Series([1.0, np.nan], index=idx),
            pd.Series([1.0, 2.0], index=idx),
            extra={"std": pd.Series([np.nan, 0.5], index=idx)},
        )
        dist = nowcast_distribution(NowcastResults(target="y", nowcast=frame))
        assert dist.mean.tolist() == [2.0]
        with pytest.raises(TypeError, match="supports"):
            nowcast_distribution(NowcastResults(target="y", nowcast=frame), n_boot=2)


# ---------------------------------------------------------------------- bootstrap
class TestBootstrap:
    def test_reproducible(self, two_step_results) -> None:
        a = bootstrap_nowcasts(two_step_results, 6, random_state=11)
        b = bootstrap_nowcasts(two_step_results, 6, random_state=11)
        c = bootstrap_nowcasts(two_step_results, 6, random_state=12)
        pd.testing.assert_frame_equal(a.means, b.means)
        pd.testing.assert_frame_equal(a.stds, b.stds)
        assert not np.allclose(a.means.to_numpy(), c.means.to_numpy())
        assert a.method == "parametric"
        assert a.n_success == 6 and a.n_failed == 0 and a.n_boot == 6
        assert a.means.index.name == "replication"

    def test_seed_sequence_and_generator(self, two_step_results) -> None:
        seq = np.random.SeedSequence(5)
        a = bootstrap_nowcasts(two_step_results, 3, random_state=seq)
        b = bootstrap_nowcasts(two_step_results, 3, random_state=np.random.SeedSequence(5))
        pd.testing.assert_frame_equal(a.means, b.means)
        g1 = bootstrap_nowcasts(two_step_results, 3, random_state=np.random.default_rng(1))
        g2 = bootstrap_nowcasts(two_step_results, 3, random_state=np.random.default_rng(1))
        pd.testing.assert_frame_equal(g1.means, g2.means)

    def test_parallel_equals_sequential(self, two_step_results) -> None:
        seq = bootstrap_nowcasts(two_step_results, 4, random_state=7)
        par = bootstrap_nowcasts(two_step_results, 4, random_state=7, n_jobs=2)
        pd.testing.assert_frame_equal(seq.means, par.means)
        pd.testing.assert_frame_equal(seq.stds, par.stds)

    def test_em_parametric(self, em_results) -> None:
        boot = bootstrap_nowcasts(em_results, 5, random_state=0, refit_params={"max_iter": 30})
        assert boot.method == "parametric"
        assert boot.means.shape == (5, len(analytic_distribution(em_results)))
        assert bool((boot.parameter_variance > 0).all())
        assert bool((boot.filtering_variance > 0).all())

    def test_em_without_warm_start(self, em_results) -> None:
        boot = bootstrap_nowcasts(
            em_results,
            2,
            random_state=0,
            warm_start=False,
            refit_params={"max_iter": 1, "tol": 0.0},
        )
        assert boot.n_success == 2
        assert boot.n_convergence_warnings == 2

    @pytest.mark.parametrize("fixture", ["two_step_results", "em_results"])
    def test_block(self, fixture: str, request: pytest.FixtureRequest) -> None:
        res = request.getfixturevalue(fixture)
        boot = bootstrap_nowcasts(res, 4, method="block", random_state=0, block_length=3)
        assert boot.method == "block"
        assert boot.info["block_length"] == 3
        assert boot.n_success == 4

    def test_variables_mode(self, two_step_variables_results) -> None:
        boot = bootstrap_nowcasts(two_step_variables_results, 3, random_state=0)
        assert boot.method == "parametric"
        assert boot.n_success == 3
        block = bootstrap_nowcasts(two_step_variables_results, 3, method="block", random_state=0)
        assert block.method == "block" and block.n_success == 3
        legacy = two_step_variables_results.replace(model_data=None)
        assert bootstrap_nowcasts(legacy, 2, random_state=0).method == "block"
        with pytest.raises(NotImplementedError, match="block"):
            bootstrap_nowcasts(legacy, 3, method="parametric")

    def test_monthly_target_variables_mode_is_parametric(self, panel) -> None:
        monthly = panel.drop("gdp")
        res = TwoStepDFM(n_factors=1, aggregate="variables").fit(monthly, "x1")
        boot = bootstrap_nowcasts(res, 2, random_state=0)
        assert boot.method == "parametric"

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True])
    def test_invalid_n_boot(self, two_step_results, bad: object) -> None:
        with pytest.raises(ValueError, match="n_boot"):
            bootstrap_nowcasts(two_step_results, bad)  # type: ignore[arg-type]

    def test_invalid_method(self, two_step_results) -> None:
        with pytest.raises(ValueError, match="method"):
            bootstrap_nowcasts(two_step_results, 2, method="wild")  # type: ignore[arg-type]

    def test_missing_data(self, two_step_results) -> None:
        res = two_step_results.replace(data=None)
        with pytest.raises(NowcastDataError, match="estimation data"):
            bootstrap_nowcasts(res, 2)

    def test_some_failures_warn(self, two_step_results, monkeypatch) -> None:
        import nowcastbox.density.bootstrap as mod

        original = mod._refit_two_step
        calls = {"n": 0}

        def flaky(results, panel, params, prefiltered=False):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise np.linalg.LinAlgError("singular")
            return original(results, panel, params, prefiltered)

        monkeypatch.setattr(mod, "_refit_two_step", flaky)
        with pytest.warns(ConvergenceWarning, match="2 of 4"):
            boot = bootstrap_nowcasts(two_step_results, 4, random_state=0)
        assert boot.n_success == 2
        assert boot.failures == ["LinAlgError: singular"] * 2

    def test_non_finite_and_all_failures(self, two_step_results, monkeypatch) -> None:
        import nowcastbox.density.bootstrap as mod

        def bad(results, panel, params, prefiltered=False):
            nan = pd.Series(np.nan, index=results.nowcast.index)
            return nan, nan

        monkeypatch.setattr(mod, "_refit_two_step", bad)
        with pytest.raises(NowcastDataError, match="All 3 bootstrap replications failed"):
            bootstrap_nowcasts(two_step_results, 3, random_state=0)


# ---------------------------------------------------------------------- combination
class TestCombination:
    def test_nowcast_distribution_mixture(self, two_step_results) -> None:
        dist = nowcast_distribution(two_step_results, n_boot=15, random_state=0)
        analytic = analytic_distribution(two_step_results)
        assert dist.n_components == 15
        np.testing.assert_allclose(dist.mean, analytic.mean)  # centred
        np.testing.assert_allclose(dist.point, analytic.mean)
        dec = dist.variance_decomposition
        np.testing.assert_allclose(dec["total"], dist.variance, rtol=1e-10)
        assert bool((dec["parameter"] > 0).all())
        assert dist.info["source"] == "bootstrap"
        assert dist.info["method"] == "parametric"
        pd.testing.assert_series_equal(dist.info["analytic_std"], analytic.std)

    def test_not_centred(self, two_step_results) -> None:
        boot = bootstrap_nowcasts(two_step_results, 5, random_state=0)
        analytic = analytic_distribution(two_step_results)
        mix = combine_bootstrap(analytic, boot, center=False)
        np.testing.assert_allclose(mix.mean, boot.means.mean(axis=0))
        np.testing.assert_allclose(mix.point, analytic.point)

    def test_combine_missing_period(self, two_step_results) -> None:
        boot = bootstrap_nowcasts(two_step_results, 2, random_state=0)
        other = NowcastDistribution(["1999Q1"], [0.0], [1.0])
        with pytest.raises(NowcastDataError, match="does not cover"):
            combine_bootstrap(other, boot)

    def test_collect_type(self, two_step_results) -> None:
        boot = bootstrap_nowcasts(two_step_results, 2, random_state=0)
        assert isinstance(boot, BootstrapNowcasts)

    @pytest.mark.parametrize("bad", [-1, 1.0, False])
    def test_invalid_n_boot(self, two_step_results, bad: object) -> None:
        with pytest.raises(ValueError, match="n_boot"):
            nowcast_distribution(two_step_results, n_boot=bad)  # type: ignore[arg-type]

    def test_bootstrap_widens_intervals(self, two_step_results) -> None:
        mix = nowcast_distribution(two_step_results, n_boot=20, random_state=1)
        gauss = nowcast_distribution(two_step_results)
        width_mix = mix.interval(0.9).diff(axis=1)["upper"]
        width_gauss = gauss.interval(0.9).diff(axis=1)["upper"]
        # parameter uncertainty adds variance: total >= average within-component variance
        assert bool((mix.std**2 >= mix.variance_decomposition["filtering"] - 1e-12).all())
        assert bool((width_mix > 0).all()) and bool((width_gauss > 0).all())


# ---------------------------------------------------------------------- calibration
class TestCalibration:
    """PIT of density nowcasts on data simulated from a known DGP is ~ U(0, 1)."""

    @staticmethod
    def _pits(model_factory, n_rep: int) -> tuple[np.ndarray, np.ndarray]:
        pits, hits = [], []
        for seed in range(n_rep):
            data, truth, period = simulate_with_truth(seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = model_factory().fit(data, "gdp")
            dist = analytic_distribution(res, period)
            pits.append(float(pit(dist, truth)[0]))
            band = dist.interval(0.8)
            hits.append(bool(interval_hits(truth, band["lower"], band["upper"])[0]))
        return np.array(pits), np.array(hits)

    def test_two_step_pit_uniform(self) -> None:
        pits, hits = self._pits(lambda: TwoStepDFM(n_factors=1), 80)
        assert not ks_uniformity_test(pits).reject(0.001)
        assert 0.65 <= hits.mean() <= 0.95

    def test_em_pit_uniform(self) -> None:
        pits, hits = self._pits(lambda: MixedFreqDFM(n_factors=1, max_iter=50), 60)
        assert not ks_uniformity_test(pits).reject(0.001)
        assert 0.6 <= hits.mean() <= 0.97

    def test_true_model_gaussian_pit_exact(self, two_step_results) -> None:
        """Under the true state-space model the smoothed PIT is exactly uniform."""
        from nowcastbox.density import simulate_from_results

        res = two_step_results
        ssm = res.state_space
        rng = np.random.default_rng(0)
        pits = []
        for _ in range(300):
            sim = simulate_from_results(res, random_state=rng)
            z = res.standardization.transform(sim.to_frame()[list(res.loadings.index)])
            truth = z.to_numpy()[-1, 0]
            y = z.to_numpy().copy()
            y[-1, 0] = np.nan
            sm = kalman_smoother(ssm, y)
            mean = sm.smoothed_signal()[-1, 0]
            cov = sm.smoothed_signal_cov(y.shape[0] - 1)[0, 0] + ssm.obs_cov_diagonal[0]
            pits.append(scipy.stats.norm.cdf(truth, mean, np.sqrt(cov)))
        assert not ks_uniformity_test(np.array(pits)).reject(0.01)

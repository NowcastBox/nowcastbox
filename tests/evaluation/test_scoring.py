"""Tests of the density-forecast scores and calibration tests (nowcastbox.evaluation.scoring)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.integrate
import scipy.stats
from hypothesis import given
from hypothesis import strategies as st

from nowcastbox.density import NowcastDistribution
from nowcastbox.evaluation.scoring import (
    BerkowitzTestResult,
    CoverageTestResult,
    berkowitz_test,
    christoffersen_test,
    crps,
    crps_gaussian,
    crps_mixture,
    crps_sample,
    interval_coverage,
    interval_hits,
    interval_score,
    ks_uniformity_test,
    log_score,
    log_score_gaussian,
    log_score_sample,
    pit,
    quantile_score,
    weighted_quantile_score,
)

finite = st.floats(-50, 50, allow_nan=False)
positive = st.floats(0.05, 20, allow_nan=False)


def crps_numeric(cdf, y: float, lo: float, hi: float) -> float:
    """CRPS by numerical integration of (F(x) - 1{x >= y})^2."""
    left = scipy.integrate.quad(lambda x: cdf(x) ** 2, lo, y, limit=200)[0]
    right = scipy.integrate.quad(lambda x: (1 - cdf(x)) ** 2, y, hi, limit=200)[0]
    return left + right


# ====================================================================== CRPS
class TestCRPS:
    @pytest.mark.parametrize(("y", "mu", "sd"), [(0.0, 0.0, 1.0), (1.3, -0.2, 0.7), (-4, 1, 2)])
    def test_gaussian_vs_integral(self, y: float, mu: float, sd: float) -> None:
        expected = crps_numeric(
            lambda x: scipy.stats.norm.cdf(x, mu, sd), y, mu - 12 * sd, mu + 12 * sd
        )
        assert float(crps_gaussian(y, mu, sd)) == pytest.approx(expected, rel=1e-6)

    def test_gaussian_vs_monte_carlo(self) -> None:
        draws = np.random.default_rng(0).normal(0.5, 1.5, size=200_000)
        assert float(crps_sample(1.2, draws)) == pytest.approx(
            float(crps_gaussian(1.2, 0.5, 1.5)), rel=5e-3
        )

    def test_gaussian_known_value(self) -> None:
        # CRPS(N(0,1), 0) = 2 phi(0) - 1/sqrt(pi) = (sqrt(2) - 1)/sqrt(pi)
        assert float(crps_gaussian(0, 0, 1)) == pytest.approx((np.sqrt(2) - 1) / np.sqrt(np.pi))

    def test_gaussian_vectorised(self) -> None:
        out = crps_gaussian([0.0, 1.0], [0.0, 0.0], 1.0)
        assert out.shape == (2,)

    def test_mixture_vs_integral(self) -> None:
        locs, scales, w = np.array([-1.0, 0.5, 2.0]), np.array([0.5, 1.0, 0.3]), [0.2, 0.5, 0.3]

        def cdf(x: float) -> float:
            return float(np.dot(w, scipy.stats.norm.cdf(x, locs, scales)))

        expected = crps_numeric(cdf, 0.7, -15, 15)
        assert float(crps_mixture(0.7, locs, scales, w)) == pytest.approx(expected, rel=1e-6)

    def test_mixture_vs_monte_carlo(self) -> None:
        rng = np.random.default_rng(1)
        locs, scales = rng.normal(size=4), rng.uniform(0.2, 1.0, size=4)
        comp = rng.integers(0, 4, size=300_000)
        draws = locs[comp] + scales[comp] * rng.standard_normal(comp.size)
        assert float(crps_mixture(0.3, locs, scales)) == pytest.approx(
            float(crps_sample(0.3, draws)), rel=5e-3
        )

    def test_mixture_batch_shapes(self) -> None:
        locs = np.zeros((3, 2))
        scales = np.ones((3, 2))
        out = crps_mixture([0.0, 1.0, 2.0], locs, scales)
        np.testing.assert_allclose(out, crps_gaussian([0.0, 1.0, 2.0], 0.0, 1.0))

    @pytest.mark.parametrize(
        ("locs", "scales", "weights", "match"),
        [
            ([0.0, 1.0], [1.0], None, "same shape"),
            ([0.0], [1.0], [1.0, 1.0], "weights"),
            ([0.0], [1.0], [-1.0], "weights"),
            ([0.0], [0.0], None, "positive"),
            ([np.nan], [1.0], None, "finite"),
        ],
    )
    def test_mixture_invalid(self, locs, scales, weights, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            crps_mixture(0.0, locs, scales, weights)

    def test_sample_brute_force(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(size=37)
        y = 0.4
        brute = np.abs(x - y).mean() - 0.5 * np.abs(x[:, None] - x[None, :]).mean()
        assert float(crps_sample(y, x)) == pytest.approx(brute)
        fair = np.abs(x - y).mean() - np.abs(x[:, None] - x[None, :]).sum() / (2 * 37 * 36)
        assert float(crps_sample(y, x, fair=True)) == pytest.approx(fair)

    def test_sample_batch_and_single_draw(self) -> None:
        samples = np.array([[0.0, 1.0], [2.0, 4.0]])
        out = crps_sample([0.5, 3.0], samples)
        assert out.shape == (2,)
        assert float(crps_sample(2.0, [5.0])) == 3.0  # point forecast: absolute error

    def test_sample_invalid(self) -> None:
        with pytest.raises(ValueError, match="draw"):
            crps_sample(0.0, np.empty((2, 0)))
        with pytest.raises(ValueError, match="draw"):
            crps_sample(0.0, 1.0)
        with pytest.raises(ValueError, match="two draws"):
            crps_sample(0.0, [1.0], fair=True)
        with pytest.raises(ValueError):
            crps_sample([0.0, 1.0, 2.0], np.zeros((2, 5)))  # incompatible shapes
        with pytest.raises(ValueError, match="finite"):
            crps_sample(np.nan, [1.0, 2.0])

    def test_gaussian_invalid(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            crps_gaussian(0.0, 0.0, -1.0)
        with pytest.raises(ValueError, match="finite"):
            crps_gaussian(0.0, np.inf, 1.0)

    def test_dispatch(self) -> None:
        gauss = NowcastDistribution(["2020Q1", "2020Q2"], [0.0, 1.0], [1.0, 2.0])
        np.testing.assert_allclose(
            crps(gauss, [0.5, 0.5]), crps_gaussian([0.5, 0.5], [0, 1], [1, 2])
        )
        mix = NowcastDistribution(["2020Q1"], [[0.0, 1.0]], [[1.0, 0.5]])
        np.testing.assert_allclose(crps(mix, 0.2), crps_mixture(0.2, [0.0, 1.0], [1.0, 0.5]))
        np.testing.assert_allclose(crps([1.0, 2.0, 3.0], 2.0), crps_sample(2.0, [1.0, 2.0, 3.0]))

    @given(y=finite, mu=finite, sd=positive)
    def test_property_gaussian_nonnegative_and_mixture_consistent(
        self, y: float, mu: float, sd: float
    ) -> None:
        g = float(crps_gaussian(y, mu, sd))
        assert g >= -1e-12
        assert float(crps_mixture(y, [mu, mu], [sd, sd])) == pytest.approx(g, abs=1e-8 * (1 + g))
        # CRPS is bounded by the absolute error plus the spread term
        assert g <= abs(y - mu) + sd

    @given(
        draws=st.lists(finite, min_size=1, max_size=30),
        y=finite,
    )
    def test_property_sample_nonnegative(self, draws: list[float], y: float) -> None:
        assert float(crps_sample(y, draws)) >= -1e-9


# ====================================================================== log score
class TestLogScore:
    def test_gaussian(self) -> None:
        np.testing.assert_allclose(
            log_score_gaussian([0.1, 2.0], 0.5, 1.5), scipy.stats.norm.logpdf([0.1, 2.0], 0.5, 1.5)
        )
        with pytest.raises(ValueError):
            log_score_gaussian(0.0, 0.0, 0.0)

    def test_dispatch_distribution(self) -> None:
        d = NowcastDistribution(["2020Q1"], [[0.0, 3.0]], [[1.0, 1.0]])
        expected = np.log(0.5 * scipy.stats.norm.pdf(1.0) + 0.5 * scipy.stats.norm.pdf(-2.0))
        np.testing.assert_allclose(log_score(d, 1.0), expected)

    def test_sample_kde(self) -> None:
        draws = np.random.default_rng(3).normal(1.0, 2.0, size=20_000)
        approx = float(log_score(draws, 1.5))
        assert approx == pytest.approx(scipy.stats.norm.logpdf(1.5, 1.0, 2.0), abs=0.03)
        out = log_score_sample([0.0, 1.0], np.vstack([draws, draws]), bandwidth=0.3)
        assert out.shape == (2,)

    def test_sample_degenerate(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            log_score_sample(0.0, [1.0, 1.0, 1.0])


# ====================================================================== PIT
class TestPIT:
    def test_distribution_and_samples(self) -> None:
        d = NowcastDistribution(["2020Q1", "2020Q2"], [0.0, 1.0], [1.0, 1.0])
        np.testing.assert_allclose(pit(d, [0.0, 2.0]), [0.5, scipy.stats.norm.cdf(1.0)])
        assert float(pit([1.0, 2.0, 3.0, 4.0], 2.0)) == 0.5
        assert pit(np.zeros((3, 10)), [0.0, -1.0, 1.0]).tolist() == [1.0, 0.0, 1.0]

    def test_randomized_pit_uniform_with_ties(self) -> None:
        rng = np.random.default_rng(4)
        draws = rng.integers(0, 3, size=(4000, 20)).astype(float)
        obs = rng.integers(0, 3, size=4000).astype(float)
        u = pit(draws, obs, randomize=True, random_state=5)
        assert ((u >= 0) & (u <= 1)).all()
        assert not ks_uniformity_test(u).reject(0.001)

    def test_calibrated_gaussian_pit_uniform(self) -> None:
        rng = np.random.default_rng(6)
        mu = rng.normal(size=1000)
        sd = rng.uniform(0.5, 2.0, size=1000)
        y = mu + sd * rng.standard_normal(1000)
        idx = pd.period_range("1800Q1", periods=1000, freq="Q")
        u = pit(NowcastDistribution(idx, mu, sd), y)
        assert not ks_uniformity_test(u).reject(0.01)
        assert not berkowitz_test(u).reject(0.01)

    def test_overdispersed_forecast_detected(self) -> None:
        rng = np.random.default_rng(7)
        y = rng.standard_normal(500)
        u = scipy.stats.norm.cdf(y, 0.0, 3.0)  # too wide
        assert ks_uniformity_test(u).reject(0.01)
        res = berkowitz_test(u)
        assert res.reject(0.01)
        assert res.variance < 0.5


# ====================================================================== uniformity tests
class TestUniformity:
    def test_ks(self) -> None:
        res = ks_uniformity_test([0.1, 0.2, 0.3, 0.9])
        expected = scipy.stats.kstest([0.1, 0.2, 0.3, 0.9], "uniform")
        assert res.statistic == pytest.approx(expected.statistic)
        assert res.pvalue == pytest.approx(expected.pvalue)
        assert res.nobs == 4

    @pytest.mark.parametrize("bad", [[0.5], [0.2, 1.2], [np.nan, 0.5], [-0.1, 0.4]])
    def test_invalid_pit(self, bad: list[float]) -> None:
        with pytest.raises(ValueError):
            ks_uniformity_test(bad)

    def test_berkowitz_detects_autocorrelation(self) -> None:
        rng = np.random.default_rng(8)
        z = np.zeros(600)
        for t in range(1, 600):
            z[t] = 0.6 * z[t - 1] + np.sqrt(1 - 0.36) * rng.standard_normal()
        res = berkowitz_test(scipy.stats.norm.cdf(z))
        assert isinstance(res, BerkowitzTestResult)
        assert res.rho == pytest.approx(0.6, abs=0.1)
        assert res.independence_pvalue < 0.001
        assert res.reject(0.01)
        assert res.nobs == 600

    def test_berkowitz_mle_matches_statsmodels(self) -> None:
        from statsmodels.tsa.arima.model import ARIMA

        rng = np.random.default_rng(9)
        z = np.zeros(300)
        for t in range(1, 300):
            z[t] = 0.3 + 0.4 * (z[t - 1] - 0.3) + 0.8 * rng.standard_normal()
        z[0] = 0.3
        res = berkowitz_test(scipy.stats.norm.cdf(z))
        ref = ARIMA(z, order=(1, 0, 0), trend="c").fit()
        const, ar, sigma2 = ref.params
        assert res.mean == pytest.approx(const, abs=2e-3)
        assert res.rho == pytest.approx(ar, abs=2e-3)
        assert res.variance == pytest.approx(sigma2, rel=5e-3)

    def test_berkowitz_constant_input(self) -> None:
        with pytest.raises(ValueError, match="constant"):
            berkowitz_test([0.5, 0.5, 0.5, 0.5])

    def test_berkowitz_invalid(self) -> None:
        with pytest.raises(ValueError, match="clip"):
            berkowitz_test([0.1, 0.5, 0.9], clip=0.0)
        with pytest.raises(ValueError, match="At least 3"):
            berkowitz_test([0.1, 0.5])


# ====================================================================== coverage
class TestCoverage:
    def test_hits_and_coverage(self) -> None:
        hits = interval_hits([0.0, 1.0, 2.0], [-1.0, 1.0, 2.5], [1.0, 2.0, 3.0])
        assert hits.tolist() == [True, True, False]
        assert interval_coverage([0.0, 1.0, 2.0], [-1.0, 1.0, 2.5], [1.0, 2.0, 3.0]) == 2 / 3
        with pytest.raises(ValueError, match="exceed"):
            interval_hits(0.0, 1.0, 0.0)
        with pytest.raises(ValueError, match="No observations"):
            interval_coverage([], [], [])

    def test_interval_score(self) -> None:
        out = interval_score([0.0, -3.0, 4.0], -1.0, 1.0, 0.8)
        np.testing.assert_allclose(out, [2.0, 2.0 + 10 * 2.0, 2.0 + 10 * 3.0])
        with pytest.raises(ValueError, match="level"):
            interval_score(0.0, -1.0, 1.0, 1.0)

    def test_interval_score_equals_quantile_scores(self) -> None:
        """S_alpha = (2/alpha) (QS_{alpha/2} + QS_{1-alpha/2}) (Gneiting & Raftery, 2007)."""
        y, lo, hi, alpha = np.array([0.3, -2.0, 5.0]), -1.0, 2.0, 0.2
        qs = quantile_score(y, lo, alpha / 2) + quantile_score(y, hi, 1 - alpha / 2)
        np.testing.assert_allclose(interval_score(y, lo, hi, 1 - alpha), 2 / alpha * qs)

    def test_christoffersen_hand_computed(self) -> None:
        hits = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
        res = christoffersen_test(hits, 0.9)
        n1, n0 = hits.sum(), (1 - hits).sum()
        pi = n1 / hits.size
        lr_uc = -2 * (n1 * np.log(0.9) + n0 * np.log(0.1) - n1 * np.log(pi) - n0 * np.log(1 - pi))
        prev, cur = hits[:-1], hits[1:]
        n00 = np.sum((prev == 0) & (cur == 0))
        n01 = np.sum((prev == 0) & (cur == 1))
        n10 = np.sum((prev == 1) & (cur == 0))
        n11 = np.sum((prev == 1) & (cur == 1))
        p01, p11 = n01 / (n00 + n01), n11 / (n10 + n11)
        p = (n01 + n11) / (hits.size - 1)
        l1 = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11) + n11 * np.log(p11)
        l0 = (n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p)
        assert isinstance(res, CoverageTestResult)
        assert res.lr_uc == pytest.approx(lr_uc)
        assert res.lr_ind == pytest.approx(-2 * (l0 - l1))
        assert res.lr_cc == pytest.approx(res.lr_uc + res.lr_ind)
        assert res.pvalue_cc == pytest.approx(scipy.stats.chi2.sf(res.lr_cc, 2))
        assert res.transitions.to_numpy().tolist() == [[n00, n01], [n10, n11]]
        assert res.hit_rate == pi and res.nobs == 20

    def test_christoffersen_edge_cases(self) -> None:
        all_hits = christoffersen_test([True] * 30, 0.9)
        assert all_hits.hit_rate == 1.0
        assert all_hits.lr_uc == pytest.approx(-2 * 30 * np.log(0.9))
        assert all_hits.lr_ind == 0.0
        assert all_hits.reject(0.05, test="uc")
        assert not all_hits.reject(0.05, test="ind")
        with pytest.raises(ValueError, match="test"):
            all_hits.reject(test="xx")
        no_hits = christoffersen_test([0, 0, 0], 0.5)
        assert no_hits.hit_rate == 0.0

    def test_christoffersen_detects_clustering(self) -> None:
        hits = np.array(([1] * 18 + [0] * 2) * 10)  # right rate, clustered misses
        res = christoffersen_test(hits, 0.9)
        assert not res.reject(0.05, test="uc")
        assert res.reject(0.05, test="ind")

    @pytest.mark.parametrize(
        ("hits", "coverage", "match"),
        [
            ([1], 0.9, "at least 2"),
            ([[1, 0]], 0.9, "1-D"),
            ([1, 2], 0.9, "0/1"),
            ([1, 0], 1.0, "level"),
        ],
    )
    def test_christoffersen_invalid(self, hits, coverage: float, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            christoffersen_test(hits, coverage)

    def test_calibrated_intervals_pass(self) -> None:
        rng = np.random.default_rng(10)
        y = rng.standard_normal(1000)
        lo, hi = scipy.stats.norm.ppf([0.05, 0.95])
        res = christoffersen_test(interval_hits(y, lo, hi), 0.9)
        assert not res.reject(0.01)


# ====================================================================== quantile scores
class TestQuantileScores:
    def test_pinball(self) -> None:
        np.testing.assert_allclose(quantile_score([0.0, 2.0], 1.0, 0.25), [0.75, 0.25])
        with pytest.raises(ValueError, match="levels"):
            quantile_score(0.0, 0.0, 1.0)

    def test_quantile_score_minimised_at_true_quantile(self) -> None:
        y = np.random.default_rng(11).standard_normal(50_000)
        grid = np.linspace(0.5, 2.0, 61)
        risk = [quantile_score(y, q, 0.9).mean() for q in grid]
        assert grid[int(np.argmin(risk))] == pytest.approx(scipy.stats.norm.ppf(0.9), abs=0.05)

    def test_uniform_weights_approximate_crps(self) -> None:
        d = NowcastDistribution(["2020Q1", "2020Q2"], [0.0, 1.0], [1.0, 0.5])
        taus = (np.arange(1, 400) - 0.5) / 399
        wqs = weighted_quantile_score([0.3, -0.4], d.quantiles(taus))
        np.testing.assert_allclose(wqs, crps(d, [0.3, -0.4]), rtol=1e-2)

    def test_weight_functions(self) -> None:
        taus = np.array([0.1, 0.5, 0.9])
        q = np.array([[-1.28, 0.0, 1.28]])
        y = [2.0]
        base = 2 * quantile_score(np.array(y)[:, None], q, taus[None, :])[0]
        for name, w in {
            "uniform": np.ones(3),
            "center": taus * (1 - taus),
            "left": (1 - taus) ** 2,
            "right": taus**2,
        }.items():
            got = weighted_quantile_score(y, q, taus, weights=name)  # type: ignore[arg-type]
            assert got[0] == pytest.approx(float(base @ w) / 3)
        explicit = weighted_quantile_score(y, q, taus, weights=[1.0, 0.0, 0.0])
        assert explicit[0] == pytest.approx(base[0] / 3)

    @pytest.mark.parametrize(
        ("quantiles", "levels", "kwargs", "match"),
        [
            (np.zeros((1, 3)), None, {}, "levels are required"),
            (np.zeros((1, 3)), [0.1, 0.5], {}, "one column"),
            (np.zeros((2, 3)), [0.1, 0.5, 0.9], {}, "one row"),
            (np.zeros((1, 3)), [0.1, 0.5, 0.9], {"weights": "tails"}, "weights must be one of"),
            (np.zeros((1, 3)), [0.1, 0.5, 0.9], {"weights": [1, -1, 1]}, "non-negative"),
        ],
    )
    def test_invalid(self, quantiles, levels, kwargs, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            weighted_quantile_score([0.0], quantiles, levels, **kwargs)

    @given(y=finite, q=finite, tau=st.floats(0.01, 0.99))
    def test_property_pinball_nonnegative(self, y: float, q: float, tau: float) -> None:
        assert float(quantile_score(y, q, tau)) >= 0.0

"""Tests for nowcastbox.selection.targeted (Bai & Ng, 2008 targeted predictors)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import ConvergenceWarning, DataQualityWarning, NowcastDataError
from nowcastbox.selection import (
    TargetedPredictorsResult,
    elastic_net,
    elastic_net_path,
    hard_threshold,
    newey_west_lags,
    select_targeted_predictors,
    soft_threshold,
)
from nowcastbox.selection.targeted import _align, _cd_loop, _cd_loop_fast, _ols_tstat


def _sparse_design(seed: int, n_obs: int = 300, n_pred: int = 40):
    relevant = (0, 5, 9) if n_pred >= 10 else (0, 1, 2)
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n_obs, n_pred)), columns=[f"x{i}" for i in range(n_pred)])
    beta = np.zeros(n_pred)
    beta[list(relevant)] = [1.0, -0.8, 0.6]
    y = pd.Series(x.to_numpy() @ beta + 0.5 * rng.normal(size=n_obs), index=x.index)
    return x, y, [f"x{i}" for i in relevant]


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
class TestAlign:
    def test_horizon_and_lags(self):
        y = pd.Series(np.arange(10, dtype=float))
        x = pd.DataFrame({"a": np.arange(10, dtype=float) * 10})
        preds, lhs, w = _align(x, y, horizon=2, y_lags=2)
        # controls y_t, y_{t-1} (h > 0): first usable t = 1, last = 7
        np.testing.assert_array_equal(preds["a"], np.arange(1, 8) * 10)
        np.testing.assert_array_equal(lhs, np.arange(3, 10))
        np.testing.assert_array_equal(w[:, 1], np.arange(1, 8))
        np.testing.assert_array_equal(w[:, 2], np.arange(0, 7))
        np.testing.assert_array_equal(w[:, 0], 1.0)

    def test_horizon_zero_starts_at_lag_one(self):
        y = pd.Series(np.arange(6, dtype=float))
        x = pd.DataFrame({"a": np.zeros(6) + np.arange(6)})
        _, lhs, w = _align(x, y, horizon=0, y_lags=1)
        np.testing.assert_array_equal(lhs, np.arange(1, 6))
        np.testing.assert_array_equal(w[:, 1], np.arange(0, 5))

    def test_quarterly_target_on_monthly_grid(self):
        idx = pd.period_range("2000-01", periods=24, freq="M")
        y = pd.Series(np.nan, index=idx)
        q_end = idx.month % 3 == 0
        y[q_end] = np.arange(8, dtype=float)
        x = pd.DataFrame({"a": np.arange(24, dtype=float)}, index=idx)
        preds, lhs, _ = _align(x, y, horizon=1, y_lags=0)
        assert list(preds.index.month) == [3, 6, 9, 12, 3, 6, 9]
        np.testing.assert_array_equal(lhs, np.arange(1, 8))

    def test_too_short(self):
        with pytest.raises(NowcastDataError, match="Too few"):
            _align(pd.DataFrame({"a": np.arange(4.0)}), np.arange(4.0), horizon=2, y_lags=0)

    def test_target_as_column_name_and_mfd(self):
        idx = pd.period_range("2000-01", periods=12, freq="M")
        df = pd.DataFrame({"a": np.arange(12.0), "y": np.arange(12.0) ** 2}, index=idx)
        preds, lhs, _ = _align(df, "y", 0, 0)
        assert list(preds.columns) == ["a"]
        mfd = MixedFrequencyData(df, "M")
        _, lhs2, _ = _align(mfd, "y", 0, 0)
        np.testing.assert_array_equal(lhs, lhs2)

    def test_series_reindexed_or_positional(self):
        x = np.arange(20.0).reshape(10, 2)
        y = pd.Series(np.arange(10.0), index=range(100, 110))
        preds, lhs, _ = _align(x, y, 0, 0)  # ndarray x: positional
        np.testing.assert_array_equal(lhs, np.arange(10.0))
        xf = pd.DataFrame(x, index=range(100, 110))
        y_part = pd.Series([1.0, 2.0, 3.0, 4.0], index=[101, 102, 103, 104])
        preds, lhs, _ = _align(xf, y_part, 0, 0)
        assert list(preds.index) == [101, 102, 103, 104]

    def test_input_errors(self):
        x = pd.DataFrame({"a": np.arange(5.0)})
        with pytest.raises(NowcastDataError, match="not a column"):
            _align(x, "zz", 0, 0)
        with pytest.raises(NowcastDataError, match="share no index"):
            _align(x, pd.Series([1.0], index=[99]), 0, 0)
        with pytest.raises(NowcastDataError, match="values but x"):
            _align(x, np.arange(3.0), 0, 0)
        with pytest.raises(TypeError, match="DataFrame"):
            _align([1, 2, 3], np.arange(3.0), 0, 0)
        with pytest.raises(NowcastDataError, match="Non-numeric"):
            _align(pd.DataFrame({"a": list("abcde")}), np.arange(5.0), 0, 0)
        with pytest.raises(NowcastDataError, match="infinite"):
            _align(x, np.array([1.0, np.inf, 2, 3, 4]), 0, 0)
        with pytest.raises(NowcastDataError, match="No candidate"):
            _align(pd.DataFrame({"y": np.arange(5.0)}), "y", 0, 0)


# ---------------------------------------------------------------------------
# Hard thresholding
# ---------------------------------------------------------------------------
class TestHardThreshold:
    @pytest.mark.parametrize("seed", range(5))
    def test_recovers_relevant_predictors(self, seed):
        x, y, relevant = _sparse_design(seed)
        res = hard_threshold(x, y, threshold=2.58, cov_type="nonrobust")
        assert set(relevant) <= set(res.selected)
        assert len(res.selected) <= 3 + 3  # few false positives at 1%
        assert res.selected[0] == "x0"  # largest |t|

    def test_tstats_match_statsmodels(self):
        x, y, _ = _sparse_design(1, n_obs=120, n_pred=5)
        res_h = hard_threshold(x, y, threshold=0.0, cov_type="hac", hac_lags=3)
        res_n = hard_threshold(x, y, threshold=0.0, cov_type="nonrobust")
        for col in x.columns:
            design = sm.add_constant(x[col].to_numpy())
            fit_hac = sm.OLS(y.to_numpy(), design).fit(
                cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": False}
            )
            fit_ols = sm.OLS(y.to_numpy(), design).fit()
            assert res_h.scores[col] == pytest.approx(abs(fit_hac.tvalues[1]), rel=1e-10)
            assert res_n.scores[col] == pytest.approx(abs(fit_ols.tvalues[1]), rel=1e-10)

    def test_with_lags_and_horizon_matches_statsmodels(self):
        rng = np.random.default_rng(4)
        n = 200
        x = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
        y = pd.Series(np.r_[0.0, x["a"].to_numpy()[:-1]] + rng.normal(size=n))
        res = hard_threshold(x, y, horizon=1, y_lags=2, cov_type="nonrobust")
        # manual regression y_{t+1} on (1, y_t, y_{t-1}, a_t)
        t = np.arange(1, n - 1)
        design = np.column_stack([np.ones(len(t)), y[t], y[t - 1], x["a"].to_numpy()[t]])
        fit = sm.OLS(y.to_numpy()[t + 1], design).fit()
        assert res.scores["a"] == pytest.approx(abs(fit.tvalues[-1]), rel=1e-10)
        assert res.selected[0] == "a"
        assert res.params["y_lags"] == 2 and res.horizon == 1

    def test_default_hac_lags(self):
        assert newey_west_lags(100) == 4
        assert newey_west_lags(500) == 5
        x, y, _ = _sparse_design(0, n_obs=100, n_pred=3)
        auto = hard_threshold(x, y, threshold=0.0)
        manual = hard_threshold(x, y, threshold=0.0, hac_lags=4)
        pd.testing.assert_series_equal(auto.scores, manual.scores)

    def test_max_predictors_and_ranking(self):
        x, y, _ = _sparse_design(2)
        res = hard_threshold(x, y, threshold=1.0, max_predictors=2)
        assert res.n_selected == 2
        assert res.ranking[res.selected[0]] == 1 and res.ranking[res.selected[1]] == 2

    def test_empty_selection_warns(self):
        x, y, _ = _sparse_design(0, n_obs=50, n_pred=3)
        with pytest.warns(DataQualityWarning, match="empty"):
            res = hard_threshold(x, y, threshold=1e6)
        assert res.selected == []

    def test_missing_and_constant_predictors(self):
        x, y, _ = _sparse_design(0, n_obs=60, n_pred=4)
        x.loc[:55, "x1"] = np.nan  # too few observations
        x["x2"] = 1.0  # constant
        x.loc[[3, 7], "x0"] = np.nan  # pairwise deletion
        with pytest.warns(DataQualityWarning, match="2 predictors skipped"):
            res = hard_threshold(x, y)
        assert np.isnan(res.scores["x1"]) and np.isnan(res.scores["x2"])
        assert "x0" in res.selected
        assert res.n_obs == 60

    def test_no_usable_predictor(self):
        x = pd.DataFrame({"a": np.ones(30)})
        with (
            pytest.warns(DataQualityWarning),
            pytest.raises(NowcastDataError, match="No predictor"),
        ):
            hard_threshold(x, np.arange(30.0))

    def test_degenerate_variance_gives_zero_t(self):
        design = np.column_stack([np.ones(5), np.arange(5.0)])
        assert _ols_tstat(design, np.zeros(5), "nonrobust", 0) == 0.0
        assert _ols_tstat(design, np.zeros(5), "hac", 2) == 0.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"threshold": -1.0}, "threshold"),
            ({"threshold": np.nan}, "threshold"),
            ({"cov_type": "hc3"}, "cov_type"),
            ({"hac_lags": -1}, "hac_lags"),
            ({"max_predictors": 0}, "max_predictors"),
            ({"horizon": -1}, "horizon"),
            ({"y_lags": 1.5}, "y_lags"),
        ],
    )
    def test_validation(self, kwargs, match):
        x, y, _ = _sparse_design(0, n_obs=40, n_pred=3)
        with pytest.raises(ValueError, match=match):
            hard_threshold(x, y, **kwargs)


# ---------------------------------------------------------------------------
# Elastic net
# ---------------------------------------------------------------------------
class TestElasticNet:
    @pytest.mark.parametrize(("alpha", "l1_ratio"), [(0.05, 0.5), (0.2, 1.0), (0.01, 0.1)])
    def test_matches_statsmodels(self, alpha, l1_ratio):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(150, 10))
        y = X[:, 0] - 0.5 * X[:, 3] + rng.normal(size=150)
        coef, intercept = elastic_net(X, y, alpha, l1_ratio, fit_intercept=False)
        ref = sm.OLS(y, X).fit_regularized(
            method="elastic_net", alpha=alpha, L1_wt=l1_ratio, cnvrg_tol=1e-14, maxiter=2000
        )
        np.testing.assert_allclose(coef, ref.params, atol=1e-7)
        assert intercept == 0.0

    def test_lasso_orthogonal_closed_form(self):
        n = 4
        X = np.sqrt(n) * np.eye(n)  # X'X / n = I
        y = np.array([3.0, -1.0, 0.2, 0.0]) * np.sqrt(n)
        alpha, l1_ratio = 0.5, 0.6
        coef, _ = elastic_net(X, y, alpha, l1_ratio, fit_intercept=False)
        z = X.T @ y / n
        expected = (
            np.sign(z) * np.maximum(np.abs(z) - alpha * l1_ratio, 0) / (1 + alpha * (1 - l1_ratio))
        )
        np.testing.assert_allclose(coef, expected, atol=1e-12)

    def test_alpha_zero_is_ols_with_intercept(self, rng):
        X = rng.normal(size=(80, 4))
        y = 2.0 + X @ np.array([1.0, 0.0, -1.0, 0.5]) + rng.normal(size=80)
        coef, intercept = elastic_net(X, y, 0.0, 1.0)
        ols = sm.OLS(y, sm.add_constant(X)).fit()
        np.testing.assert_allclose(np.r_[intercept, coef], ols.params, atol=1e-7)

    def test_kkt_conditions(self, rng):
        X = rng.normal(size=(100, 15))
        y = X[:, :3] @ np.array([2.0, -1.0, 1.0]) + rng.normal(size=100)
        alpha, rho = 0.1, 0.7
        coef, intercept = elastic_net(X, y, alpha, rho)
        resid = y - intercept - X @ coef
        grad = X.T @ resid / 100 - alpha * (1 - rho) * coef
        active = coef != 0
        np.testing.assert_allclose(grad[active], alpha * rho * np.sign(coef[active]), atol=1e-6)
        assert np.all(np.abs(grad[~active]) <= alpha * rho + 1e-8)

    def test_python_and_numba_loops_agree(self, rng):
        X = rng.normal(size=(60, 8))
        y = X[:, 1] + rng.normal(size=60)
        gram, xty = X.T @ X / 60, X.T @ y / 60
        b1, b2 = np.zeros(8), np.zeros(8)
        n1 = _cd_loop(gram, xty, b1, 0.05, 0.02, 1000, 1e-10)
        n2 = _cd_loop_fast(gram, xty, b2, 0.05, 0.02, 1000, 1e-10)
        np.testing.assert_allclose(b1, b2, atol=1e-12)
        assert n1 == n2

    def test_python_loop_constant_column_and_no_convergence(self):
        gram = np.array([[0.0, 0.0], [0.0, 1.0]])
        beta = np.array([5.0, 0.0])
        sweeps = _cd_loop(gram, np.array([0.0, 1.0]), beta, 0.1, 0.0, 1, 1e-300)
        assert beta[0] == 0.0 and sweeps == 2

    def test_convergence_warning(self, rng):
        X = rng.normal(size=(50, 5))
        X[:, 1] = X[:, 0] + 1e-3 * rng.normal(size=50)  # strongly correlated
        y = X[:, 0] + rng.normal(size=50)
        with pytest.warns(ConvergenceWarning, match="did not converge"):
            elastic_net(X, y, 1e-4, 0.5, max_iter=1)

    def test_warm_start_shape(self, rng):
        with pytest.raises(ValueError, match="coef_init"):
            elastic_net(rng.normal(size=(10, 3)), rng.normal(size=10), 0.1, coef_init=np.zeros(2))

    @pytest.mark.parametrize(
        ("alpha", "l1_ratio", "max_iter", "tol", "match"),
        [
            (-1.0, 0.5, 10, 1e-6, "alpha"),
            (0.1, 0.0, 10, 1e-6, "l1_ratio"),
            (0.1, 1.5, 10, 1e-6, "l1_ratio"),
            (0.1, 0.5, 0, 1e-6, "max_iter"),
            (0.1, 0.5, 10, 0.0, "tol"),
        ],
    )
    def test_validation(self, rng, alpha, l1_ratio, max_iter, tol, match):
        with pytest.raises(ValueError, match=match):
            elastic_net(
                rng.normal(size=(10, 2)),
                rng.normal(size=10),
                alpha,
                l1_ratio,
                max_iter=max_iter,
                tol=tol,
            )

    def test_data_errors(self, rng):
        with pytest.raises(NowcastDataError, match="rows"):
            elastic_net(rng.normal(size=(10, 2)), rng.normal(size=9), 0.1)
        X = rng.normal(size=(10, 2))
        X[0, 0] = np.nan
        with pytest.raises(NowcastDataError, match="finite"):
            elastic_net(X, rng.normal(size=10), 0.1)


class TestElasticNetPath:
    def test_path_properties(self, rng):
        X = rng.normal(size=(100, 6))
        y = X[:, 0] * 2 + rng.normal(size=100)
        alphas, coefs = elastic_net_path(X, y, 0.5, n_alphas=20, eps=1e-2)
        assert np.all(np.diff(alphas) < 0)
        assert alphas[-1] == pytest.approx(alphas[0] * 1e-2)
        assert np.all(coefs[0] == 0)  # alpha_max kills every coefficient
        assert coefs[1, 0] != 0  # the strongest predictor enters first
        assert coefs[-1, 0] > 1.5
        for a, b in zip(alphas[[3, 10]], coefs[[3, 10]], strict=True):
            np.testing.assert_allclose(b, elastic_net(X, y, a, 0.5)[0], atol=1e-6)

    def test_explicit_alphas_sorted(self, rng):
        X = rng.normal(size=(30, 3))
        y = rng.normal(size=30)
        alphas, coefs = elastic_net_path(X, y, alphas=[0.01, 1.0, 0.1])
        np.testing.assert_allclose(alphas, [1.0, 0.1, 0.01])
        assert coefs.shape == (3, 3)
        with pytest.raises(ValueError, match="empty"):
            elastic_net_path(X, y, alphas=[])

    def test_zero_response(self, rng):
        X = rng.normal(size=(30, 3))
        alphas, coefs = elastic_net_path(X, np.ones(30), n_alphas=3)
        assert alphas[0] == 1.0 and np.all(coefs == 0)

    @pytest.mark.parametrize(("n_alphas", "eps"), [(1, 1e-3), (10, 0.0), (10, 1.0), (2.5, 0.1)])
    def test_validation(self, rng, n_alphas, eps):
        with pytest.raises(ValueError, match="n_alphas"):
            elastic_net_path(
                rng.normal(size=(10, 2)), rng.normal(size=10), n_alphas=n_alphas, eps=eps
            )


# ---------------------------------------------------------------------------
# Soft thresholding
# ---------------------------------------------------------------------------
class TestSoftThreshold:
    @pytest.mark.parametrize("seed", range(5))
    def test_recovers_relevant_predictors(self, seed):
        x, y, relevant = _sparse_design(seed)
        res = soft_threshold(x, y, n_predictors=3)
        assert res.selected == relevant  # entry order = strength order here
        assert res.method == "soft" and res.n_obs == 300
        assert res.ranking[relevant[0]] == 1

    def test_default_n_predictors_is_30(self):
        x, y, _ = _sparse_design(0, n_pred=40)
        res = soft_threshold(x, y, eps=1e-4)
        assert res.n_selected == 30
        assert res.params["alpha"] > 0

    def test_fixed_alpha(self):
        x, y, relevant = _sparse_design(1)
        res = soft_threshold(x, y, alpha=0.3, l1_ratio=1.0)
        assert set(res.selected) == set(relevant)
        assert res.params == {"alpha": 0.3, "l1_ratio": 1.0}
        # scores are coefficients on standardised predictors
        z = (x - x.mean()) / x.std(ddof=0)
        coef, _ = elastic_net(z.to_numpy(), y.to_numpy(), 0.3, 1.0)
        np.testing.assert_allclose(res.scores.to_numpy(), coef)
        assert np.isnan(res.ranking[[c for c in x.columns if c not in relevant]]).all()

    def test_fixed_alpha_too_large_warns(self):
        x, y, _ = _sparse_design(1)
        with pytest.warns(DataQualityWarning, match="no predictor"):
            res = soft_threshold(x, y, alpha=100.0)
        assert res.selected == []

    def test_path_too_short_warns(self):
        x, y, _ = _sparse_design(1, n_pred=20)
        with pytest.warns(DataQualityWarning, match="became active"):
            res = soft_threshold(x, y, n_predictors=20, n_alphas=3, eps=0.5)
        assert res.n_selected < 20

    def test_missing_rows_dropped(self):
        x, y, relevant = _sparse_design(2)
        x.iloc[:10, 3] = np.nan
        with pytest.warns(DataQualityWarning, match="Dropped 10 of 300"):
            res = soft_threshold(x, y, n_predictors=3)
        assert res.n_obs == 290 and res.selected == relevant

    def test_quarterly_target_mfd(self):
        idx = pd.period_range("2000-01", periods=360, freq="M")
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(360, 5)), index=idx, columns=list("abcde"))
        gdp = pd.Series(np.nan, index=idx)
        q_end = idx.month % 3 == 0
        gdp[q_end] = 2 * df.loc[q_end, "c"].to_numpy() + 0.3 * rng.normal(size=q_end.sum())
        df["gdp"] = gdp
        mfd = MixedFrequencyData(df, {**dict.fromkeys("abcde", "M"), "gdp": "Q"})
        res = select_targeted_predictors(mfd, "gdp", method="soft", n_predictors=1)
        assert res.selected == ["c"] and res.n_obs == 120
        sub = res.transform(mfd)
        assert isinstance(sub, MixedFrequencyData) and list(sub.columns) == ["c"]

    def test_errors(self):
        x, y, _ = _sparse_design(0, n_pred=5)
        with pytest.raises(ValueError, match="either alpha or n_predictors"):
            soft_threshold(x, y, alpha=0.1, n_predictors=2)
        with pytest.raises(ValueError, match="n_predictors"):
            soft_threshold(x, y, n_predictors=6)
        with pytest.raises(ValueError, match="n_predictors"):
            soft_threshold(x, y, n_predictors=0)
        x["x1"] = 2.0
        with pytest.raises(NowcastDataError, match="Constant"):
            soft_threshold(x, y)
        x2 = pd.DataFrame({"a": [np.nan, np.nan, 1.0, 2.0, np.nan]})
        with (
            pytest.warns(DataQualityWarning),
            pytest.raises(NowcastDataError, match="Too few complete"),
        ):
            soft_threshold(x2, np.arange(5.0))


# ---------------------------------------------------------------------------
# Result object and dispatcher
# ---------------------------------------------------------------------------
class TestResultAndDispatch:
    def test_dispatch(self):
        x, y, _ = _sparse_design(0, n_pred=6)
        assert select_targeted_predictors(x, y).method == "hard"
        assert select_targeted_predictors(x, y, method="soft", n_predictors=2).method == "soft"
        with pytest.raises(ValueError, match="method"):
            select_targeted_predictors(x, y, method="lasso")

    def test_transform_and_summary(self):
        x, y, _ = _sparse_design(0, n_pred=6)
        res = hard_threshold(x, y, threshold=2.58)
        assert isinstance(res, TargetedPredictorsResult)
        assert list(res.transform(x).columns) == res.selected
        with pytest.raises(NowcastDataError, match="not found"):
            res.transform(x.drop(columns=res.selected[0]))
        text = res.summary()
        assert "hard thresholding" in text and res.selected[0] in text
        assert str(res) == text

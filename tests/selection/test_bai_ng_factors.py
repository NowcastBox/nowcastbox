"""Tests for nowcastbox.selection.bai_ng_factors (Bai & Ng, 2002)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.selection import (
    CRITERIA,
    FactorSelectionResult,
    bai_ng_penalty,
    factor_criteria,
    select_factors,
)
from nowcastbox.selection._panel import principal_components
from tests.selection.simulate import static_factor_panel

matplotlib.use("Agg")

# small panels are used on purpose; the small-sample warning is tested elsewhere
pytestmark = pytest.mark.filterwarnings("ignore:Bai-Ng criteria are unreliable")


# ---------------------------------------------------------------------------
# Penalties and criteria (analytical)
# ---------------------------------------------------------------------------
class TestPenalty:
    @pytest.mark.parametrize(("n", "t"), [(100, 100), (50, 200), (300, 40)])
    def test_formulas(self, n, t):
        c2 = min(n, t)
        assert bai_ng_penalty(n, t, 1) == pytest.approx((n + t) / (n * t) * np.log(n * t / (n + t)))
        assert bai_ng_penalty(n, t, 2) == pytest.approx((n + t) / (n * t) * np.log(c2))
        assert bai_ng_penalty(n, t, 3) == pytest.approx(np.log(c2) / c2)

    def test_g2_at_least_g1_when_square(self):
        # with N = T: g1 = (2/T) ln(T/2) < g2 = (2/T) ln T
        assert bai_ng_penalty(100, 100, 1) < bai_ng_penalty(100, 100, 2)

    @pytest.mark.parametrize("k", [0, 4, "1"])
    def test_invalid_k(self, k):
        with pytest.raises(ValueError, match="Penalty k"):
            bai_ng_penalty(10, 10, k)

    def test_invalid_dims(self):
        with pytest.raises(ValueError, match="positive"):
            bai_ng_penalty(0, 10, 1)


class TestFactorCriteria:
    def test_v_matches_explicit_pca_residuals(self, rng):
        x = rng.normal(size=(60, 15))
        x = (x - x.mean(0)) / x.std(0, ddof=1)
        s = np.linalg.svd(x, compute_uv=False)
        v, _ = factor_criteria(s, 15, 60, 5)
        for r in range(6):
            f, lam, _ = principal_components(x, r)
            resid = x - f @ lam.T
            assert v[r] == pytest.approx(np.mean(resid**2), rel=1e-10)
        assert v[0] == pytest.approx(np.mean(x**2))

    def test_criteria_definitions(self, rng):
        x = rng.normal(size=(40, 12))
        s = np.linalg.svd(x, compute_uv=False)
        v, crit = factor_criteria(s, 12, 40, 4)
        r = np.arange(5)
        for k in (1, 2, 3):
            g = bai_ng_penalty(12, 40, k)
            np.testing.assert_allclose(crit[f"IC{k}"], np.log(v) + r * g)
            np.testing.assert_allclose(crit[f"PC{k}"], v + r * v[4] * g)
        assert list(crit.columns) == list(CRITERIA)
        assert crit.index.name == "n_factors"


# ---------------------------------------------------------------------------
# select_factors: recovery on simulated data
# ---------------------------------------------------------------------------
class TestRecovery:
    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("true_r", [1, 3, 5])
    def test_ic_and_pc12_recover_r(self, seed, true_r):
        x = static_factor_panel(150, 100, true_r, seed=seed)
        res = select_factors(x, rmax=8)
        for crit in ("IC1", "IC2", "PC1", "PC2"):
            assert res.r_star_by_criterion[crit] == true_r, crit

    @pytest.mark.parametrize("seed", range(4))
    def test_all_criteria_large_panel(self, seed):
        x = static_factor_panel(300, 300, 3, seed=seed)
        res = select_factors(x, rmax=8)
        for crit in ("IC1", "IC2", "IC3", "PC1", "PC2"):
            assert res.r_star_by_criterion[crit] == 3
        # PC3 is known to over-fit in finite samples (Bai & Ng, 2002, Table 1)
        assert res.r_star_by_criterion["PC3"] >= 3

    def test_pure_noise_selects_zero_with_ic2(self):
        x = np.random.default_rng(5).normal(size=(200, 100))
        assert select_factors(x, rmax=5).r_star == 0

    @pytest.mark.parametrize("criterion", CRITERIA)
    def test_r_star_follows_criterion(self, criterion):
        x = static_factor_panel(100, 60, 2, seed=1)
        res = select_factors(x, rmax=6, criterion=criterion)
        assert res.criterion == criterion
        assert res.r_star == int(res.criteria[criterion].idxmin())

    @pytest.mark.parametrize(
        ("alias", "expected"), [("icp2", "IC2"), ("IC_p1", "IC1"), ("PCp3", "PC3"), ("pc_2", "PC2")]
    )
    def test_criterion_aliases(self, alias, expected):
        x = static_factor_panel(50, 20, 1, seed=0)
        assert select_factors(x, rmax=3, criterion=alias).criterion == expected


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------
class TestResult:
    @pytest.fixture
    def res(self) -> FactorSelectionResult:
        x = pd.DataFrame(
            static_factor_panel(120, 30, 2, seed=3), columns=[f"s{i}" for i in range(30)]
        )
        return select_factors(x, rmax=6)

    def test_fields(self, res):
        assert res.n_series == 30 and res.n_periods == 120 and res.rmax == 6
        assert res.columns[0] == "s0"
        assert len(res.eigenvalues) == 30
        assert res.explained_variance_ratio.sum() == pytest.approx(1.0)
        # standardized panel: eigenvalues of X'X/T sum to N (T-1)/T
        assert res.eigenvalues.sum() == pytest.approx(30 * 119 / 120)
        assert res.ssr.is_monotonic_decreasing
        assert res.dropped_rows.empty and res.dropped_columns == []

    def test_to_frame_and_summary(self, res):
        frame = res.to_frame()
        assert list(frame.columns) == ["V", *CRITERIA]
        text = res.summary()
        assert "r* = 2" in text and "IC2" in text
        assert str(res) == text

    def test_plot_matplotlib(self, res):
        import matplotlib.pyplot as plt

        ax = res.plot()
        assert ax.get_xlabel() == "number of factors r"
        ax2 = res.plot(criteria=["IC1", "PC1"])
        assert len(ax2.get_legend().get_texts()) == 2
        ax3 = res.plot("eigenvalues")
        assert ax3.get_title() == "Scree plot"
        plt.close("all")

    def test_plot_existing_axes(self, res):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        assert res.plot(ax=ax) is ax
        plt.close("all")

    def test_plot_plotly(self, res):
        fig = res.plot(backend="plotly")
        assert fig.layout.title.text.startswith("Bai & Ng")
        fig2 = res.plot("eigenvalues", backend="plotly")
        assert len(fig2.data) == 2  # line + selected marker

    def test_plot_errors(self, res):
        with pytest.raises(ValueError, match="Unknown criteria"):
            res.plot(criteria=["IC9"])
        with pytest.raises(ValueError, match="kind"):
            res.plot("loadings")
        with pytest.raises(ValueError, match="backend"):
            res.plot(backend="bokeh")
        with pytest.raises(ValueError, match="ax"):
            res.plot(backend="plotly", ax=object())


# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
class TestDataHandling:
    def test_standardization_invariance(self, rng):
        x = static_factor_panel(100, 40, 2, seed=7)
        scaled = x * rng.uniform(0.1, 50, size=40) + rng.normal(size=40) * 10
        a = select_factors(x, rmax=5).criteria
        b = select_factors(scaled, rmax=5).criteria
        pd.testing.assert_frame_equal(a, b)

    def test_no_standardize_uses_raw_scale(self):
        x = static_factor_panel(100, 40, 2, seed=7)
        res = select_factors(x * 10, rmax=5, standardize=False)
        assert res.ssr[0] == pytest.approx(np.mean((x * 10) ** 2))

    def test_drop_rows_with_nan_warns(self):
        x = pd.DataFrame(static_factor_panel(100, 20, 2, seed=2))
        x.iloc[[0, 5, 99], [1, 3, 4]] = np.nan
        with pytest.warns(DataQualityWarning, match="Dropped 3 of 100 periods"):
            res = select_factors(x, rmax=4)
        assert res.n_periods == 97
        assert list(res.dropped_rows) == [0, 5, 99]
        assert "Dropped periods (missing values): 3" in res.summary()

    def test_drop_columns_with_nan(self):
        x = pd.DataFrame(static_factor_panel(100, 20, 2, seed=2))
        x.iloc[[0, 5], [1, 3]] = np.nan
        with pytest.warns(DataQualityWarning, match="Dropped 2 series"):
            res = select_factors(x, rmax=4, missing="drop_columns")
        assert res.n_series == 18 and res.dropped_columns == ["1", "3"]
        assert "Dropped series: 2" in res.summary()

    def test_missing_raise(self):
        x = static_factor_panel(30, 10, 1, seed=0)
        x[0, 0] = np.nan
        with pytest.raises(NowcastDataError, match="missing values"):
            select_factors(x, rmax=2, missing="raise")

    def test_invalid_missing_policy(self):
        with pytest.raises(ValueError, match="missing"):
            select_factors(np.ones((5, 5)), rmax=2, missing="fill")

    def test_mixed_frequency_input_excludes_quarterly(self):
        idx = pd.period_range("2000-01", periods=120, freq="M")
        x = static_factor_panel(120, 10, 1, seed=4)
        df = pd.DataFrame(x, index=idx, columns=[f"m{i}" for i in range(10)])
        gdp = pd.Series(np.nan, index=idx)
        gdp[idx.month % 3 == 0] = np.arange(40, dtype=float)
        df["gdp"] = gdp
        freqs = {**{f"m{i}": "M" for i in range(10)}, "gdp": "Q"}
        mfd = MixedFrequencyData(df, freqs)
        with pytest.warns(DataQualityWarning, match="lower than the base grid"):
            res = select_factors(mfd, rmax=3)
        assert res.n_series == 10 and res.n_periods == 120
        assert res.dropped_columns == ["gdp"]
        assert "Dropped" in res.summary()

    def test_too_many_dropped_rows(self):
        x = np.random.default_rng(0).normal(size=(5, 4))
        x[:3, 0] = np.nan
        with (
            pytest.warns(DataQualityWarning),
            pytest.raises(NowcastDataError, match="periods"),
        ):
            select_factors(x, rmax=1)

    def test_too_few_series(self):
        with pytest.raises(NowcastDataError, match="series"):
            select_factors(np.random.default_rng(0).normal(size=(10, 1)), rmax=1)

    def test_inf_raises(self):
        x = np.random.default_rng(0).normal(size=(10, 4))
        x[2, 2] = np.inf
        with pytest.raises(NowcastDataError, match="infinite"):
            select_factors(x, rmax=1)

    def test_constant_series_raises(self):
        x = np.random.default_rng(0).normal(size=(10, 4))
        x[:, 1] = 3.0
        with pytest.raises(NowcastDataError, match="constant"):
            select_factors(x, rmax=1)

    def test_non_numeric_raises(self):
        df = pd.DataFrame({"a": ["x", "y", "z", "w"], "b": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(NowcastDataError, match="non-numeric"):
            select_factors(df, rmax=1)

    def test_bad_types(self):
        with pytest.raises(TypeError, match="DataFrame"):
            select_factors([[1.0, 2.0]], rmax=1)
        with pytest.raises(NowcastDataError, match="2-D"):
            select_factors(np.ones(5), rmax=1)


class TestValidation:
    @pytest.mark.parametrize("rmax", [0, -1, 2.5, True, "3"])
    def test_invalid_rmax(self, rmax):
        with pytest.raises(ValueError, match="rmax"):
            select_factors(np.random.default_rng(0).normal(size=(20, 8)), rmax=rmax)

    def test_rmax_too_large(self):
        with pytest.raises(ValueError, match="smaller than min"):
            select_factors(np.random.default_rng(0).normal(size=(20, 8)), rmax=8)

    @pytest.mark.parametrize("criterion", ["IC4", "BIC", "", 2])
    def test_invalid_criterion(self, criterion):
        with pytest.raises(ValueError, match="criterion"):
            select_factors(np.random.default_rng(0).normal(size=(20, 8)), criterion=criterion)

    def test_numpy_integer_rmax(self):
        res = select_factors(np.random.default_rng(0).normal(size=(20, 8)), rmax=np.int64(3))
        assert res.rmax == 3

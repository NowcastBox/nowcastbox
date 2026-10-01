"""Tests of Diebold-Mariano, Giacomini-White and the Model Confidence Set."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.evaluation import (
    diebold_mariano,
    giacomini_white,
    model_confidence_set,
)
from nowcastbox.evaluation.tests import _block_indices


def identity(e: np.ndarray) -> np.ndarray:
    return e


class TestDieboldMariano:
    def test_matches_textbook_formula(self, rng) -> None:
        e1, e2 = rng.normal(0, 1, 60), rng.normal(0, 1.3, 60)
        d = e1**2 - e2**2
        n = d.size
        c = d - d.mean()
        gamma = [c[k:] @ c[: n - k] / n for k in range(3)]
        dm = d.mean() / math.sqrt((gamma[0] + 2 * (gamma[1] + gamma[2])) / n)
        res = diebold_mariano(e1, e2, h=3, hln=False)
        assert res.statistic == pytest.approx(dm)
        assert res.pvalue == pytest.approx(2 * stats.norm.sf(abs(dm)))
        hln = diebold_mariano(e1, e2, h=3)
        factor = math.sqrt((n + 1 - 6 + 3 * 2 / n) / n)
        assert hln.statistic == pytest.approx(dm * factor)
        assert hln.pvalue == pytest.approx(2 * stats.t(n - 1).sf(abs(dm * factor)))
        assert hln.n_obs == n
        assert hln.h == 3
        assert hln.mean_loss_differential == pytest.approx(d.mean())

    def test_alternatives_and_absolute_loss(self, rng) -> None:
        e1, e2 = rng.normal(0, 1, 100), rng.normal(0, 2, 100)
        less = diebold_mariano(e1, e2, alternative="less", loss="absolute")
        greater = diebold_mariano(e1, e2, alternative="greater", loss="absolute")
        assert less.pvalue < 0.01
        assert greater.pvalue == pytest.approx(1 - less.pvalue)
        assert less.reject()

    def test_drops_missing_pairs(self, rng) -> None:
        e1, e2 = rng.normal(size=30), rng.normal(size=30)
        e1[:5] = np.nan
        assert diebold_mariano(e1, e2).n_obs == 25

    def test_bartlett_fallback(self) -> None:
        d = np.tile([1.0, -1.0, 0.2], 20)
        with pytest.warns(DataQualityWarning, match="Bartlett"):
            res = diebold_mariano(d, np.zeros_like(d), h=2, loss=identity)
        assert np.isfinite(res.statistic)

    def test_constant_differential(self) -> None:
        e = np.ones(10)
        with pytest.warns(DataQualityWarning, match="Constant"):
            res = diebold_mariano(e, e)
        assert np.isnan(res.statistic)
        assert not res.reject()

    def test_errors(self) -> None:
        with pytest.raises(NowcastDataError, match="at least 3"):
            diebold_mariano([1.0, 2.0], [0.0, 1.0])
        with pytest.raises(ValueError, match="h must be"):
            diebold_mariano([1.0] * 5, [0.0] * 5, h=0)
        with pytest.raises(ValueError, match="alternative"):
            diebold_mariano([1.0] * 5, [0.0] * 5, alternative="two")
        with pytest.raises(ValueError, match="different lengths"):
            diebold_mariano([1.0] * 5, [0.0] * 4)

    @settings(max_examples=30)
    @given(st.integers(0, 10_000), st.integers(1, 4))
    def test_antisymmetry(self, seed: int, h: int) -> None:
        rng = np.random.default_rng(seed)
        e1, e2 = rng.normal(size=40), rng.normal(size=40)
        a = diebold_mariano(e1, e2, h=h)
        b = diebold_mariano(e2, e1, h=h)
        assert a.statistic == pytest.approx(-b.statistic)
        assert a.pvalue == pytest.approx(b.pvalue)

    def test_size_under_the_null(self) -> None:
        rng = np.random.default_rng(42)
        rejections = [
            diebold_mariano(rng.normal(size=40), rng.normal(size=40)).reject(0.05)
            for _ in range(400)
        ]
        assert 0.02 <= np.mean(rejections) <= 0.09


class TestGiacominiWhite:
    def test_unconditional_closed_form(self, rng) -> None:
        e1, e2 = rng.normal(0, 1, 80), rng.normal(0, 1.2, 80)
        d = e1**2 - e2**2
        res = giacomini_white(e1, e2, instruments=np.ones(80))
        assert res.df == 1
        assert res.statistic == pytest.approx(80 * d.mean() ** 2 / np.mean(d**2))
        assert res.pvalue == pytest.approx(stats.chi2.sf(res.statistic, 1))

    def test_default_instruments(self, rng) -> None:
        e1, e2 = rng.normal(0, 1, 200), rng.normal(0, 1.5, 200)
        res = giacomini_white(e1, e2)
        assert res.df == 2
        assert res.n_obs == 199
        assert res.reject(0.05)
        assert res.mean_loss_differential < 0

    def test_multistep_hac(self, rng) -> None:
        e1, e2 = rng.normal(size=120), rng.normal(size=120)
        res = giacomini_white(e1, e2, h=3, instruments=pd.DataFrame({"c": np.ones(120)}))
        d = e1**2 - e2**2
        Z = d[:, None]
        omega = Z.T @ Z / 120
        for k in (1, 2):
            g = Z[k:].T @ Z[:-k] / 120
            omega = omega + (1 - k / 3) * (g + g.T)
        expected = 120 * d.mean() ** 2 / omega[0, 0]
        assert res.statistic == pytest.approx(expected)
        assert res.h == 3

    def test_errors(self, rng) -> None:
        e1, e2 = rng.normal(size=20), rng.normal(size=20)
        with pytest.raises(ValueError, match="one row per"):
            giacomini_white(e1, e2, instruments=np.ones((10, 1)))
        with pytest.raises(NowcastDataError, match="Too few"):
            giacomini_white(e1[:3], e2[:3])
        with pytest.raises(NowcastDataError, match="Singular"):
            giacomini_white(e1, e2, instruments=np.zeros((20, 2)))
        assert not giacomini_white(e1, e1 * 1.001, instruments=np.ones(20)).reject(1e-12)


class TestModelConfidenceSet:
    def test_eliminates_bad_models(self, rng) -> None:
        e = rng.standard_normal((300, 4)) * np.array([1.0, 1.01, 1.6, 2.5])
        res = model_confidence_set(
            pd.DataFrame(e**2, columns=list("ABCD")), alpha=0.1, n_bootstrap=400
        )
        assert sorted(res.included) == ["A", "B"]
        assert res.eliminated[:2] == ["D", "C"]
        assert res.pvalues.iloc[-1] == 1.0
        assert (np.diff(res.pvalues.to_numpy()) >= 0).all()
        assert res.n_obs == 300
        assert res.block_length == 7
        frame = res.to_frame()
        assert frame.index[-1] == "D"
        assert frame["included"].sum() == 2
        assert "Model Confidence Set" in res.summary()

    def test_range_statistic(self, rng) -> None:
        e = rng.standard_normal((200, 3)) * np.array([1.0, 1.0, 3.0])
        res = model_confidence_set(e**2, statistic="range", n_bootstrap=300, block_length=3)
        assert res.eliminated[0] == "model3"
        assert "model3" not in res.included
        assert res.statistic == "range"

    def test_identical_models_are_kept(self, rng) -> None:
        losses = np.repeat(rng.random((50, 1)), 3, axis=1)
        res = model_confidence_set(losses, names=["a", "b", "c"], n_bootstrap=50)
        assert sorted(res.included) == ["a", "b", "c"]
        assert (res.pvalues == 1.0).all()

    def test_reproducible_with_seed(self, rng) -> None:
        losses = rng.random((60, 3))
        a = model_confidence_set(losses, n_bootstrap=100, random_state=5)
        b = model_confidence_set(losses, n_bootstrap=100, random_state=5)
        pd.testing.assert_series_equal(a.pvalues, b.pvalues)

    def test_dataframe_names_override_and_nan_rows(self, rng) -> None:
        losses = pd.DataFrame(rng.random((30, 2)), columns=["x", "y"])
        losses.iloc[0, 0] = np.nan
        res = model_confidence_set(losses, names=["p", "q"], n_bootstrap=50)
        assert set(res.mean_loss.index) == {"p", "q"}
        assert res.n_obs == 29

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"alpha": 0.0}, "alpha"),
            ({"statistic": "sq"}, "statistic"),
            ({"n_bootstrap": 0}, "must be a positive"),
            ({"block_length": 0}, "must be a positive"),
        ],
    )
    def test_invalid_parameters(self, rng, kwargs, match) -> None:
        with pytest.raises(ValueError, match=match):
            model_confidence_set(rng.random((10, 2)), **kwargs)

    def test_invalid_inputs(self, rng) -> None:
        with pytest.raises(ValueError, match="2-D"):
            model_confidence_set(rng.random(10))
        with pytest.raises(ValueError, match="two models"):
            model_confidence_set(rng.random((10, 1)))
        with pytest.raises(ValueError, match="two models"):
            model_confidence_set(rng.random((10, 2)), names=["a", "a"])
        with pytest.raises(NowcastDataError, match="3 complete"):
            model_confidence_set(rng.random((2, 2)))

    @settings(max_examples=25)
    @given(st.integers(0, 10_000), st.integers(2, 5))
    def test_pvalue_properties(self, seed: int, m: int) -> None:
        rng = np.random.default_rng(seed)
        losses = rng.random((40, m)) * rng.uniform(0.5, 2.0, m)
        res = model_confidence_set(losses, n_bootstrap=60)
        p = res.pvalues.to_numpy()
        assert ((p >= 0) & (p <= 1)).all()
        assert (np.diff(p) >= 0).all()
        assert len(res.included) >= 1
        assert len(res.eliminated) == m - 1


def test_block_indices() -> None:
    idx = _block_indices(10, 3, 5, np.random.default_rng(0))
    assert idx.shape == (5, 10)
    assert idx.min() >= 0
    assert idx.max() <= 9
    assert ((np.diff(idx[:, :3], axis=1) % 10) == 1).all()

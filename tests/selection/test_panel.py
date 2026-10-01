"""Tests for the private helpers of nowcastbox.selection (panel preparation and PCA)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.selection import __all__ as public_names
from nowcastbox.selection._panel import prepare_panel, principal_components


def test_public_api_exported():
    import nowcastbox.selection as sel

    for name in public_names:
        assert hasattr(sel, name)
    assert "select_factors" in public_names and "select_shocks" in public_names


class TestPreparePanel:
    def test_standardized(self, rng):
        x = rng.normal(loc=5, scale=3, size=(50, 4))
        p = prepare_panel(x)
        np.testing.assert_allclose(p.values.mean(0), 0, atol=1e-12)
        np.testing.assert_allclose(p.values.std(0, ddof=1), 1)
        assert p.n_periods == 50 and p.n_series == 4
        assert list(p.columns) == ["x1", "x2", "x3", "x4"]

    def test_not_standardized_copy(self, rng):
        x = pd.DataFrame(rng.normal(size=(10, 3)))
        p = prepare_panel(x, standardize=False)
        np.testing.assert_array_equal(p.values, x.to_numpy())
        p.values[0, 0] = 99.0
        assert x.iloc[0, 0] != 99.0

    def test_mfd_monthly_only_no_warning(self, rng, recwarn):
        idx = pd.period_range("2010-01", periods=24, freq="M")
        frame = pd.DataFrame(rng.normal(size=(24, 3)), index=idx, columns=["a", "b", "c"])
        mfd = MixedFrequencyData(frame, "M")
        p = prepare_panel(mfd)
        assert p.n_series == 3 and len(recwarn) == 0
        assert isinstance(p.index, pd.PeriodIndex)


class TestPrincipalComponents:
    def test_normalization_and_sign(self, rng):
        x = rng.normal(size=(40, 10))
        f, lam, s = principal_components(x, 3)
        np.testing.assert_allclose(f.T @ f / 40, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(lam, x.T @ f / 40)
        idx = np.argmax(np.abs(lam), axis=0)
        assert np.all(lam[idx, range(3)] > 0)
        np.testing.assert_allclose(s, np.linalg.svd(x, compute_uv=False))

    def test_zero_factors(self, rng):
        f, lam, _ = principal_components(rng.normal(size=(10, 4)), 0)
        assert f.shape == (10, 0) and lam.shape == (4, 0)

    def test_rank_one_exact(self):
        f_true = np.linspace(-1, 1, 20)[:, None]
        lam_true = np.array([[1.0, 2.0, -1.0]])
        x = f_true @ lam_true
        f, lam, s = principal_components(x, 1)
        np.testing.assert_allclose(f @ lam.T, x, atol=1e-12)
        assert s[1] == pytest.approx(0.0, abs=1e-12)

"""Tests of the simulated mixed-frequency DFM generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.datasets import simulate_mixed_frequency_dfm


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_monthly": 0}, "n_monthly"),
        ({"n_factors": 1.5}, "n_factors"),
        ({"n_factors": True}, "n_factors"),
        ({"n_periods": 12}, "n_periods"),
        ({"r2_range": (0.0, 0.5)}, "r2_range"),
        ({"r2_range": (0.6, 0.5)}, "r2_range"),
        ({"r2_range": (0.2, 1.0)}, "r2_range"),
        ({"target_r2": 1.0}, "target_r2"),
        ({"target_r2": 0.0}, "target_r2"),
    ],
)
def test_invalid_arguments(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        simulate_mixed_frequency_dfm(**kwargs)  # type: ignore[arg-type]


def test_generator_argument_and_numpy_ints() -> None:
    gen = np.random.default_rng(3)
    data, truth = simulate_mixed_frequency_dfm(np.int64(3), 1, 30, random_state=gen)
    assert data.shape == (30, 4)
    assert truth["transition"].shape == (1, 1)


def test_factor_process_moments() -> None:
    """Long simulation: unit factor variance, AR coefficients and R^2 shares recovered."""
    data, truth = simulate_mixed_frequency_dfm(
        n_monthly=40, n_factors=2, n_periods=20_000, ragged_edge=False, random_state=7
    )
    f = truth["factors"].to_numpy()
    np.testing.assert_allclose(f.var(axis=0), 1.0, atol=0.08)
    phi = [np.corrcoef(f[1:, j], f[:-1, j])[0, 1] for j in range(2)]
    np.testing.assert_allclose(phi, np.diag(truth["transition"]), atol=0.03)
    lam = truth["loadings"].to_numpy()[:-1]
    common = (lam**2).sum(axis=1)
    r2 = common / (common + truth["idiosyncratic_var"].to_numpy()[:-1])
    assert ((r2 >= 0.3 - 1e-12) & (r2 <= 0.8 + 1e-12)).all()
    resid = data.drop(columns="gdp").to_numpy() - f @ lam.T
    np.testing.assert_allclose(
        resid.var(axis=0) / truth["idiosyncratic_var"].to_numpy()[:-1], 1.0, atol=0.08
    )


@settings(max_examples=25, deadline=None)
@given(
    n_monthly=st.integers(1, 8),
    n_factors=st.integers(1, 3),
    n_periods=st.integers(24, 90),
    start=st.sampled_from(["2000-01", "2001-02", "1999-12"]),
    seed=st.integers(0, 2**16),
    ragged=st.booleans(),
)
def test_structure_property(
    n_monthly: int, n_factors: int, n_periods: int, start: str, seed: int, ragged: bool
) -> None:
    data, truth = simulate_mixed_frequency_dfm(
        n_monthly, n_factors, n_periods, start=start, ragged_edge=ragged, random_state=seed
    )
    assert data.shape == (n_periods, n_monthly + 1)
    assert isinstance(data.index, pd.PeriodIndex) and str(data.index[0]) == start
    gdp = data["gdp"]
    assert (gdp.dropna().index.month % 3 == 0).all()
    mfd = MixedFrequencyData(data, frequencies={"gdp": "Q"})  # slot rule respected
    assert mfd.quarterly_columns == ["gdp"]
    lags = truth["publication_lags"]
    assert set(lags.unique()) <= {0, 1, 2}
    if not ragged:
        assert (lags == 0).all()
    for i, name in enumerate(data.columns[:-1]):
        lag = int(lags[name])
        assert data[name].iloc[: n_periods - lag].notna().all()
        assert data[name].iloc[n_periods - lag :].isna().all()
        assert truth["delay_days"][name] == {0: 5, 1: 35, 2: 65}[lag]
        assert i < n_monthly
    assert truth["delay_days"]["gdp"] == 45
    assert truth["loadings"].shape == (n_monthly + 1, n_factors)

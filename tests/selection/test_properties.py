"""Property-based tests (hypothesis) for the selection module."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nowcastbox.selection import elastic_net, select_factors, shock_statistics

pytestmark = [
    pytest.mark.property,
    pytest.mark.filterwarnings("ignore:Bai-Ng criteria are unreliable"),
]


@st.composite
def panels(draw):
    t = draw(st.integers(min_value=8, max_value=40))
    n = draw(st.integers(min_value=3, max_value=15))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    return np.random.default_rng(seed).normal(size=(t, n)), seed


@given(panels(), st.integers(min_value=1, max_value=6))
def test_factor_criteria_properties(panel, rmax):
    x, seed = panel
    rmax = min(rmax, min(x.shape) - 1)
    res = select_factors(x, rmax=rmax)
    # V(r) is non-increasing and V(0) = mean of squares of the standardized panel
    assert np.all(np.diff(res.ssr.to_numpy()) <= 1e-12)
    assert res.ssr[0] == pytest.approx((x.shape[0] - 1) / x.shape[0])
    assert 0 <= res.r_star <= rmax
    # invariance to column permutations and positive rescaling
    perm = np.random.default_rng(seed).permutation(x.shape[1])
    scale = np.random.default_rng(seed + 1).uniform(0.5, 4.0, size=x.shape[1])
    other = select_factors(x[:, perm] * scale[perm], rmax=rmax)
    np.testing.assert_allclose(other.criteria.to_numpy(), res.criteria.to_numpy(), atol=1e-8)


@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1e3, allow_nan=False), min_size=1, max_size=8
    ).filter(lambda v: sum(c * c for c in v) > 1e-6)
)
def test_shock_statistics_properties(eigs):
    stats = shock_statistics(np.array(eigs))
    d1, d2 = stats["D1"].to_numpy(), stats["D2"].to_numpy()
    assert np.all(d1 <= d2 + 1e-12)
    assert np.all(np.diff(d2) <= 1e-12)
    assert d1[-1] == 0.0 and d2[-1] == 0.0
    assert np.all((d2 >= 0) & (d2 <= 1 + 1e-12))


@given(
    st.integers(min_value=0, max_value=10_000),
    st.floats(min_value=1e-3, max_value=2.0),
    st.floats(min_value=0.05, max_value=1.0),
)
def test_elastic_net_kkt(seed, alpha, l1_ratio):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(40, 6))
    y = X @ rng.normal(size=6) + rng.normal(size=40)
    coef, intercept = elastic_net(X, y, alpha, l1_ratio, tol=1e-12)
    grad = X.T @ (y - intercept - X @ coef) / 40 - alpha * (1 - l1_ratio) * coef
    l1 = alpha * l1_ratio
    active = coef != 0
    np.testing.assert_allclose(grad[active], l1 * np.sign(coef[active]), atol=1e-6)
    assert np.all(np.abs(grad[~active]) <= l1 + 1e-8)

"""Property-based tests (hypothesis) for the preprocessing module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from nowcastbox.preprocessing.aggregation import (
    loading_constraints,
    month_to_quarter,
    quarter_to_month,
    rolling_aggregate,
)
from nowcastbox.preprocessing.missing import fill_missing, ragged_edge_mask
from nowcastbox.preprocessing.outliers import iqr_outlier_mask, replace_outliers
from nowcastbox.preprocessing.transforms import Diff, get_transform

pytestmark = pytest.mark.property

FREQ_PANDAS = {"M": "M", "Q": "Q"}

positive = st.floats(min_value=0.5, max_value=200.0, allow_nan=False, allow_infinity=False)
real = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)

elementary = st.sampled_from(
    ["log", "diff", "diff(2)", "pct_change", "pct_change(year)", "diff(year)", "scale(2.5)"]
)


@st.composite
def level_series(draw, min_size: int = 30, max_size: int = 60):
    freq = draw(st.sampled_from(["M", "Q"]))
    n = draw(st.integers(min_size, max_size))
    values = draw(arrays(float, n, elements=positive))
    idx = pd.period_range("2000-01" if freq == "M" else "2000Q1", periods=n, freq=freq)
    return pd.Series(values, index=idx)


@given(x=level_series(), code=st.integers(0, 7))
def test_codes_invert_exactly(x, code):
    freq = x.index.freqstr[0]
    t = get_transform(code)
    try:
        k = max(t.n_lags(freq), 1)
    except ValueError:
        return  # e.g. qoq is fine, but other spans may be undefined (not for M/Q)
    assume(len(x) > k + 1)
    y = t.apply(x)
    rebuilt = t.inverse(y.iloc[k:], x.iloc[:k])
    np.testing.assert_allclose(rebuilt.to_numpy(), x.to_numpy(), rtol=1e-8)


@pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.DataQualityWarning")
@given(x=level_series(), steps=st.lists(elementary, min_size=1, max_size=3))
def test_compositions_invert_and_round_trip_spec(x, steps):
    freq = x.index.freqstr[0]
    spec = "|".join(steps)
    t = get_transform(spec)
    assert get_transform(t.to_spec()) == t
    k = max(t.n_lags(freq), 1)
    assume(len(x) > k + 2)
    with np.errstate(all="ignore"):
        y = t.apply(x) if "log" not in steps[1:] else None
    assume(y is not None)
    assume(np.isfinite(y.iloc[k:]).all())
    rebuilt = t.inverse(y.iloc[k:], x.iloc[:k])
    np.testing.assert_allclose(rebuilt.to_numpy(), x.to_numpy(), rtol=1e-6)


@given(
    a=arrays(float, 25, elements=real),
    b=arrays(float, 25, elements=real),
    alpha=real,
    lag=st.integers(1, 6),
)
def test_diff_is_linear(a, b, alpha, lag):
    idx = pd.period_range("2000-01", periods=25, freq="M")
    x, y = pd.Series(a, index=idx), pd.Series(b, index=idx)
    lhs = Diff(lag).apply(alpha * x + y)
    rhs = alpha * Diff(lag).apply(x) + Diff(lag).apply(y)
    np.testing.assert_allclose(lhs, rhs, atol=1e-6 * (1 + abs(alpha)) * 1e3, equal_nan=True)


@given(x=arrays(float, 40, elements=real), c=real)
def test_rolling_aggregate_linear_and_constant(x, c):
    w = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 3
    np.testing.assert_allclose(
        rolling_aggregate(x + c, w), rolling_aggregate(x, w) + c * w.sum(), atol=1e-6
    )


@given(r=st.integers(1, 4), k=st.integers(1, 6), lam=st.data())
def test_mm_constraints_admit_structured_loadings(r, k, lam):
    w = np.concatenate([np.arange(1, k + 1), np.arange(k - 1, 0, -1)]).astype(float)
    R, q = loading_constraints(w, r)
    base = np.array(lam.draw(st.lists(real, min_size=r, max_size=r)))
    stacked = np.concatenate([wj / w[0] * base for wj in w])
    np.testing.assert_allclose(R @ stacked, q, atol=1e-6)


@given(
    values=arrays(
        float,
        st.integers(5, 40),
        elements=st.one_of(real, st.just(np.nan)),
    ),
    method=st.sampled_from(["spline", "linear", "moving_median"]),
)
def test_fill_keeps_observations_and_ragged_edge(values, method):
    assume(np.isfinite(values).sum() >= 2)
    s = pd.Series(values, index=pd.period_range("2000-01", periods=len(values), freq="M"))
    out = fill_missing(s, method, frequency="M")
    obs = s.notna()
    np.testing.assert_array_equal(out[obs].to_numpy(), s[obs].to_numpy())
    ragged = ragged_edge_mask(s, frequency="M")
    assert out[ragged].isna().all()
    first = int(np.flatnonzero(obs.to_numpy())[0])
    assert out.iloc[first:][~ragged.iloc[first:]].notna().all()


@given(
    base=arrays(float, st.integers(8, 60), elements=st.floats(-10.0, 10.0)),
    position=st.integers(0, 59),
    k=st.floats(0.5, 6.0),
)
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_outlier_replacements_within_clean_range(base, position, k):
    values = base.copy()
    values[position % len(values)] = 1e4  # guaranteed outlier: few inputs are discarded
    s = pd.Series(values, index=pd.period_range("2000-01", periods=len(values), freq="M"))
    mask = iqr_outlier_mask(values, k)
    assume(mask.any() and (~mask).any())
    out = replace_outliers(s, k, frequency="M")
    clean = values[~mask]
    replaced = out.to_numpy()[mask]
    assert (replaced >= clean.min() - 1e-9).all()
    assert (replaced <= clean.max() + 1e-9).all()


@given(values=arrays(float, st.integers(1, 20), elements=real))
def test_quarter_month_round_trips(values):
    q = pd.Series(values, index=pd.period_range("2000Q1", periods=len(values), freq="Q"))
    for up, down in [("repeat", "mean"), ("end", "last"), ("start", "first")]:
        back = month_to_quarter(quarter_to_month(q, up), down, complete=False)
        np.testing.assert_allclose(back.to_numpy(), q.to_numpy(), rtol=1e-12, atol=1e-9)

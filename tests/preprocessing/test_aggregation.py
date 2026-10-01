"""Tests for nowcastbox.preprocessing.aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import AggregationType, Frequency
from nowcastbox.preprocessing.aggregation import (
    TemporalAggregation,
    aggregate_panel,
    aggregation_weights,
    loading_constraints,
    mariano_murasawa_weights,
    month_to_quarter,
    quarter_to_month,
    rolling_aggregate,
    temporal_aggregation,
    to_higher_frequency,
    to_lower_frequency,
)
from nowcastbox.preprocessing.transforms import apply_transforms
from tests.preprocessing.conftest import FREQS

M_IDX = pd.period_range("2015-01", periods=24, freq="M")

# ---------------------------------------------------------------------- weights


def test_mm_weights():
    np.testing.assert_array_equal(mariano_murasawa_weights(), [1, 2, 3, 2, 1])
    np.testing.assert_allclose(
        mariano_murasawa_weights(normalize=True), np.array([1, 2, 3, 2, 1]) / 3
    )
    w = mariano_murasawa_weights(12)
    assert len(w) == 23
    assert w.max() == 12
    assert w.sum() == 144  # k^2


@pytest.mark.parametrize(
    ("high", "low", "agg", "expected"),
    [
        ("M", "Q", "flow", [1, 1, 1]),
        ("M", "Q", "average", [1 / 3] * 3),
        ("M", "Q", "stock", [1, 0, 0]),
        ("M", "Q", "mm", [1, 2, 3, 2, 1]),
        ("Q", "A", "mm", [1, 2, 3, 4, 3, 2, 1]),
        ("M", "A", "stock", [1] + [0] * 11),
    ],
)
def test_aggregation_weights(high, low, agg, expected):
    np.testing.assert_allclose(aggregation_weights(high, low, agg), expected)


def test_aggregation_weights_errors():
    with pytest.raises(ValueError, match="not fixed"):
        aggregation_weights("D", "M")
    with pytest.raises(ValueError, match="higher"):
        aggregation_weights("Q", "M")
    with pytest.raises(ValueError, match="Unknown"):
        aggregation_weights("M", "Q", "median")


def test_temporal_aggregation_object():
    agg = temporal_aggregation("monthly", 4, "flow")
    assert isinstance(agg, TemporalAggregation)
    assert agg.high is Frequency.MONTHLY
    assert agg.low is Frequency.QUARTERLY
    assert agg.aggregation is AggregationType.FLOW
    assert agg.ratio == 3
    assert agg.n_lags == 2
    direct = TemporalAggregation("M", "Q", "mm", normalize=True)  # type: ignore[arg-type]
    np.testing.assert_allclose(direct.weights.sum(), 3.0)
    with pytest.raises(ValueError, match="not fixed"):
        TemporalAggregation("W", "Q", "flow")  # type: ignore[arg-type]
    R, q = agg.loading_constraints(2)
    assert R.shape == (4, 6)
    assert q.shape == (4,)


def test_rolling_aggregate_shapes(rng):
    x = rng.normal(size=20)
    w = [1, 2, 3, 2, 1]
    out = rolling_aggregate(x, w)
    expected = np.array(
        [np.nan] * 4 + [sum(w[j] * x[t - j] for j in range(5)) for t in range(4, 20)]
    )
    np.testing.assert_allclose(out, expected)
    s = pd.Series(x, index=pd.period_range("2000-01", periods=20, freq="M"), name="s")
    out_s = rolling_aggregate(s, w)
    assert out_s.index.equals(s.index)
    assert out_s.name == "s"
    df = pd.DataFrame({"a": x, "b": 2 * x}, index=s.index)
    out_df = rolling_aggregate(df, w)
    np.testing.assert_allclose(out_df["b"], 2 * out, equal_nan=True)
    out_2d = rolling_aggregate(df.to_numpy(), w)
    np.testing.assert_allclose(out_2d, out_df.to_numpy(), equal_nan=True)
    assert np.isnan(rolling_aggregate(np.ones(3), w)).all()
    xm = x.copy()
    xm[10] = np.nan
    assert np.isnan(rolling_aggregate(xm, w)[10:15]).all()
    with pytest.raises(ValueError, match="non-empty"):
        rolling_aggregate(x, [])
    with pytest.raises(ValueError, match="non-empty"):
        rolling_aggregate(x, [1, np.nan])
    with pytest.raises(ValueError, match="non-zero"):
        rolling_aggregate(x, [0, 0])
    # zero weights do not need data
    xs = x.copy()
    xs[3] = np.nan
    np.testing.assert_allclose(rolling_aggregate(xs, [1, 0, 0]), xs)
    np.testing.assert_allclose(rolling_aggregate(x, [1, 0, 1])[2:], x[2:] + x[:-2])


def test_temporal_aggregation_apply_flow_sums():
    x = np.arange(1.0, 7.0)
    np.testing.assert_allclose(
        temporal_aggregation("M", "Q", "flow").apply(x), [np.nan, np.nan, 6, 9, 12, 15]
    )


# ---------------------------------------------------------------------- constraints


@pytest.mark.parametrize("n_factors", [1, 2, 3])
@pytest.mark.parametrize("agg", ["mm", "flow", "stock", "average"])
def test_loading_constraints_null_space(rng, n_factors, agg):
    w = aggregation_weights("M", "Q", agg)
    R, q = loading_constraints(w, n_factors)
    L = len(w)
    assert R.shape == ((L - 1) * n_factors, L * n_factors)
    assert np.allclose(q, 0)
    lam = rng.normal(size=n_factors)
    stacked = np.concatenate([w[j] / w[0] * lam for j in range(L)])
    np.testing.assert_allclose(R @ stacked, 0.0, atol=1e-12)
    # the restrictions leave exactly n_factors free parameters
    assert np.linalg.matrix_rank(R) == (L - 1) * n_factors
    # a vector violating the structure is detected
    bad = stacked.copy()
    bad[-1] += 1.0
    assert np.abs(R @ bad).max() > 0.1


def test_loading_constraints_bm_layout():
    R, _ = loading_constraints([1, 2, 3, 2, 1])
    expected = np.array(
        [[2, -1, 0, 0, 0], [3, 0, -1, 0, 0], [2, 0, 0, -1, 0], [1, 0, 0, 0, -1]], dtype=float
    )
    np.testing.assert_array_equal(R, expected)


@pytest.mark.parametrize(
    ("weights", "n_factors"),
    [([], 1), ([0, 1], 1), ([1, np.inf], 1), ([1, 2], 0), ([1, 2], 1.5), ([1, 2], True)],
)
def test_loading_constraints_errors(weights, n_factors):
    with pytest.raises(ValueError):
        loading_constraints(weights, n_factors)


# ---------------------------------------------------------------------- MM identity


def test_mariano_murasawa_exact_for_geometric_mean(rng):
    """Quarterly log-growth of a geometric-mean aggregate equals (1/3)(1,2,3,2,1) * dlog."""
    n = 60
    idx = pd.period_range("2010-01", periods=n, freq="M")
    log_x = np.cumsum(rng.normal(0.003, 0.01, n))
    x = pd.Series(np.exp(log_x), index=idx)
    dlog = np.log(x).diff()
    # quarterly geometric mean of levels -> quarterly log level = mean of monthly log levels
    q_log = month_to_quarter(np.log(x), "mean")
    q_growth = q_log.diff()
    mm = month_to_quarter(dlog, "mariano_murasawa")
    np.testing.assert_allclose(mm.dropna().to_numpy(), q_growth.dropna().to_numpy(), atol=1e-12)


def test_mariano_murasawa_approximates_arithmetic_flow(rng):
    n = 240
    idx = pd.period_range("2000-01", periods=n, freq="M")
    x = pd.Series(np.exp(np.cumsum(rng.normal(0.002, 0.005, n))), index=idx)
    exact = np.log(month_to_quarter(x, "sum")).diff().dropna()
    approx = month_to_quarter(np.log(x).diff(), "mariano_murasawa").dropna()
    assert np.abs(exact.to_numpy() - approx.to_numpy()).max() < 1e-4


# ---------------------------------------------------------------------- conversions


def test_month_to_quarter_methods():
    m = pd.Series(np.arange(1.0, 25.0), index=M_IDX, name="m")
    np.testing.assert_allclose(month_to_quarter(m, "mean")[:2], [2.0, 5.0])
    np.testing.assert_allclose(month_to_quarter(m, "sum")[:2], [6.0, 15.0])
    np.testing.assert_allclose(month_to_quarter(m, "first")[:2], [1.0, 4.0])
    np.testing.assert_allclose(month_to_quarter(m, "middle")[:2], [2.0, 5.0])
    np.testing.assert_allclose(month_to_quarter(m, "last")[:2], [3.0, 6.0])
    np.testing.assert_allclose(month_to_quarter(m, 3)[:2], [3.0, 6.0])
    out = month_to_quarter(m)
    assert out.index.freqstr.startswith("Q")
    assert out.name == "m"
    assert len(out) == 8


def test_month_to_quarter_incomplete():
    m = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0], index=M_IDX[:7])
    strict = month_to_quarter(m)
    assert np.isnan(strict.iloc[0])
    assert strict.iloc[1] == 5.0
    assert np.isnan(strict.iloc[2])  # only July observed
    loose = month_to_quarter(m, complete=False)
    assert loose.tolist() == [1.5, 5.0, 7.0]
    loose_sum = month_to_quarter(m, "sum", complete=False)
    assert loose_sum.tolist() == [3.0, 15.0, 7.0]
    # sample starting mid-quarter
    m2 = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.period_range("2015-02", periods=4, freq="M"))
    out = month_to_quarter(m2)
    assert np.isnan(out.iloc[0])
    assert out.index[0] == pd.Period("2015Q1", freq="Q")


def test_month_to_quarter_frame():
    df = pd.DataFrame({"a": np.arange(6.0), "b": np.ones(6)}, index=M_IDX[:6])
    out = month_to_quarter(df, "sum")
    assert isinstance(out, pd.DataFrame)
    assert out["a"].tolist() == [3.0, 12.0]
    assert out["b"].tolist() == [3.0, 3.0]


def test_to_lower_frequency_generic():
    m = pd.Series(np.arange(1.0, 25.0), index=M_IDX)
    annual = to_lower_frequency(m, "A", "sum")
    assert annual.tolist() == [sum(range(1, 13)), sum(range(13, 25))]
    q = month_to_quarter(m, "mean")
    assert to_lower_frequency(q, "A", "last").tolist() == [11.0, 23.0]
    assert to_lower_frequency(m, "A", 12).tolist() == [12.0, 24.0]
    days = pd.Series(1.0, index=pd.period_range("2020-01-01", "2020-03-31", freq="D"))
    np.testing.assert_allclose(to_lower_frequency(days, "M", "sum"), [31.0, 29.0, 31.0])
    np.testing.assert_allclose(to_lower_frequency(days, "M", "last"), [1.0, 1.0, 1.0])


def test_to_lower_frequency_errors():
    m = pd.Series(np.arange(6.0), index=M_IDX[:6])
    with pytest.raises(ValueError, match="not lower"):
        to_lower_frequency(m, "M")
    with pytest.raises(ValueError, match="how"):
        to_lower_frequency(m, "Q", "median")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="how"):
        to_lower_frequency(m, "Q", 0)
    with pytest.raises(ValueError, match="exceeds"):
        to_lower_frequency(m, "Q", 4)
    with pytest.raises(NowcastDataError, match="monthly"):
        month_to_quarter(month_to_quarter(m))
    with pytest.raises(NowcastDataError, match="PeriodIndex"):
        month_to_quarter(pd.Series([1.0, 2.0]))
    with pytest.raises(NowcastDataError, match="Series or DataFrame"):
        month_to_quarter([1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(NowcastDataError, match="empty"):
        month_to_quarter(m.iloc[:0])
    dup = pd.Series([1.0, 2.0], index=pd.PeriodIndex(["2020-01", "2020-01"], freq="M"))
    with pytest.raises(NowcastDataError, match="duplicated"):
        month_to_quarter(dup)


def test_quarter_to_month_methods():
    q = pd.Series([3.0, 6.0, 12.0], index=pd.period_range("2020Q1", periods=3, freq="Q"), name="q")
    end = quarter_to_month(q)
    assert end.index.freqstr == "M"
    assert end.index[0] == pd.Period("2020-01", freq="M")
    assert end.dropna().tolist() == [3.0, 6.0, 12.0]
    assert end.dropna().index.month.tolist() == [3, 6, 9]
    assert quarter_to_month(q, "start").dropna().index.month.tolist() == [1, 4, 7]
    assert quarter_to_month(q, "middle").dropna().index.month.tolist() == [2, 5, 8]
    assert quarter_to_month(q, "repeat").tolist() == [3.0] * 3 + [6.0] * 3 + [12.0] * 3
    assert quarter_to_month(q, "divide").tolist() == [1.0] * 3 + [2.0] * 3 + [4.0] * 3
    lin = quarter_to_month(q, "linear")
    np.testing.assert_allclose(lin.iloc[2:], [3, 4, 5, 6, 8, 10, 12])
    assert lin.iloc[:2].isna().all()
    spl = quarter_to_month(q, "spline")
    np.testing.assert_allclose(spl.iloc[[2, 5, 8]], [3.0, 6.0, 12.0])
    assert spl.iloc[3:5].notna().all()
    assert end.name == "q"


def test_quarter_to_month_round_trip(rng):
    q = pd.Series(rng.normal(size=12), index=pd.period_range("2010Q1", periods=12, freq="Q"))
    for how_up, how_down in [("end", "last"), ("repeat", "mean"), ("divide", "sum")]:
        back = month_to_quarter(quarter_to_month(q, how_up), how_down, complete=how_up != "end")
        pd.testing.assert_series_equal(back, q, check_names=False, check_freq=False)


def test_quarter_to_month_matches_storage_convention(quarterly_native, levels):
    out = quarter_to_month(quarterly_native)
    np.testing.assert_allclose(out.to_numpy(), levels["gdp"].to_numpy(), equal_nan=True)


def test_to_higher_frequency_generic_and_errors():
    a = pd.Series([12.0, 24.0], index=pd.period_range("2020", periods=2, freq="Y"))
    m = to_higher_frequency(a, "M", "divide")
    assert len(m) == 24
    assert m.iloc[0] == 1.0
    q = to_higher_frequency(pd.DataFrame({"a": a}), "Q", "end")
    assert isinstance(q, pd.DataFrame)
    assert q["a"].dropna().tolist() == [12.0, 24.0]
    one = to_higher_frequency(a.iloc[:1], "M", "linear")
    assert one.notna().sum() == 1
    with pytest.raises(ValueError, match="not higher"):
        to_higher_frequency(a, "A")
    with pytest.raises(ValueError, match="how"):
        to_higher_frequency(a, "M", "cubic")  # type: ignore[arg-type]
    with pytest.raises(NowcastDataError, match="quarterly"):
        quarter_to_month(a)


# ---------------------------------------------------------------------- panels


def test_aggregate_panel(panel):
    growth = apply_transforms(panel, "dlog")
    out = aggregate_panel(growth)
    assert isinstance(out, MixedFrequencyData)
    expected = rolling_aggregate(growth["ip"], mariano_murasawa_weights(normalize=True))
    np.testing.assert_allclose(out["ip"], expected, equal_nan=True)
    np.testing.assert_allclose(out["gdp"], growth["gdp"], equal_nan=True)
    raw = aggregate_panel(growth, normalize=False, columns=["ip"])
    np.testing.assert_allclose(raw["ip"], 3 * expected, equal_nan=True)
    np.testing.assert_allclose(raw["sales"], growth["sales"], equal_nan=True)
    avg = aggregate_panel(growth.to_frame(), "average", frequency=FREQS)
    assert isinstance(avg, pd.DataFrame)
    np.testing.assert_allclose(avg["ip"], growth["ip"].rolling(3).mean(), equal_nan=True)


def test_aggregate_panel_uses_metadata_and_low_frequency(panel):
    mfd = panel.with_metadata("sales", aggregation="stock")
    out = aggregate_panel(mfd)
    np.testing.assert_allclose(out["sales"], panel["sales"], equal_nan=True)
    annual = aggregate_panel(panel, "flow", low_frequency="A", columns=["ip"])
    np.testing.assert_allclose(annual["ip"], panel["ip"].rolling(12).sum(), equal_nan=True)


def test_aggregate_panel_errors(panel):
    with pytest.raises(NowcastDataError, match="Only monthly"):
        aggregate_panel(panel, columns=["gdp"])
    monthly = panel.select(["ip", "sales"])
    with pytest.raises(ValueError, match="single frequency"):
        aggregate_panel(monthly)

"""Tests for nowcastbox.preprocessing.transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.frequency import Frequency
from nowcastbox.preprocessing.transforms import (
    NAMED_TRANSFORMS,
    TRANSFORM_CODES,
    Compose,
    Diff,
    Identity,
    Log,
    PctChange,
    Scale,
    Transform,
    apply_transforms,
    get_transform,
    invert_transforms,
    resolve_lag,
    resolve_transforms,
    transform_from_code,
)
from tests.preprocessing.conftest import FREQS

# ---------------------------------------------------------------------- resolve_lag


@pytest.mark.parametrize(
    ("lag", "freq", "expected"),
    [
        ("year", "M", 12),
        ("year", "Q", 4),
        ("year", "A", 1),
        ("year", "W", 52),
        ("quarter", "M", 3),
        ("quarter", "Q", 1),
        ("month", "M", 1),
        ("annual", Frequency.QUARTERLY, 4),
        (3, None, 3),
        ("2", None, 2),
    ],
)
def test_resolve_lag(lag, freq, expected):
    assert resolve_lag(lag, freq) == expected


@pytest.mark.parametrize(
    ("lag", "freq"),
    [
        (0, "M"),
        (-1, "M"),
        (True, "M"),
        (1.5, "M"),
        ("fortnight", "M"),
        ("year", None),
        ("month", "Q"),  # shorter than one period
        ("month", "W"),  # not an integer number of weeks
    ],
)
def test_resolve_lag_errors(lag, freq):
    with pytest.raises(ValueError):
        resolve_lag(lag, freq)


@pytest.mark.parametrize("bad", [0, -2, True, 2.5, "nope", None])
def test_lagged_constructor_validates(bad):
    with pytest.raises(ValueError):
        Diff(bad)


def test_lag_string_normalised():
    assert Diff(" Year ").to_spec() == "diff(year)"
    assert Diff("12").to_spec() == "diff(12)"
    assert Diff("12") == Diff(12)


# ---------------------------------------------------------------------- elementary


def test_forward_matches_pandas(monthly_series):
    x = monthly_series
    pd.testing.assert_series_equal(Diff(1).apply(x), x.diff(), check_names=False)
    pd.testing.assert_series_equal(Diff(12).apply(x), x.diff(12), check_names=False)
    pd.testing.assert_series_equal(PctChange(1).apply(x), x.pct_change(), check_names=False)
    pd.testing.assert_series_equal(PctChange("year").apply(x), x.pct_change(12), check_names=False)
    np.testing.assert_allclose(Log().apply(x), np.log(x))
    np.testing.assert_allclose(Scale(100).apply(x), 100 * x)
    pd.testing.assert_series_equal(Identity().apply(x), x, check_names=False)


def test_apply_preserves_index_and_name(monthly_series):
    out = (Log() | Diff(1)).apply(monthly_series)
    assert out.index.equals(monthly_series.index)
    assert out.name == monthly_series.name


def test_call_is_apply(monthly_series):
    pd.testing.assert_series_equal(Diff(1)(monthly_series), Diff(1).apply(monthly_series))


@pytest.mark.parametrize("code", range(8))
def test_codes_match_plan_formulas_monthly(monthly_series, code):
    x = monthly_series
    out = transform_from_code(code).apply(x)
    expected = {
        0: x,
        1: (x - x.shift(1)) / x.shift(1),
        2: x - x.shift(1),
        3: ((x - x.shift(12)) / x.shift(12)).diff(1),
        4: (x - x.shift(12)).diff(1),
        5: x - x.shift(12),
        6: (x - x.shift(12)) / x.shift(12),
        7: (x - x.shift(3)) / x.shift(3),
    }[code]
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), equal_nan=True)


def test_quarterly_series_on_monthly_grid_uses_native_lags(levels, quarterly_native):
    gdp = levels["gdp"]
    out = Diff(1).apply(gdp, "Q")
    native_diff = quarterly_native.diff()
    stored = out.dropna()
    np.testing.assert_allclose(stored.to_numpy(), native_diff.dropna().to_numpy())
    assert stored.index.month.isin([3, 6, 9, 12]).all()
    # yoy = 4 quarters for a quarterly series
    yoy = PctChange("year").apply(gdp, "Q").dropna()
    np.testing.assert_allclose(yoy.to_numpy(), quarterly_native.pct_change(4).dropna().to_numpy())
    # code 7 (qoq) on a quarterly series is the 1-quarter change
    np.testing.assert_allclose(
        transform_from_code(7).apply(quarterly_native).dropna().to_numpy(),
        quarterly_native.pct_change(1).dropna().to_numpy(),
    )


def test_quarterly_frequency_inferred_on_monthly_grid(levels):
    np.testing.assert_allclose(
        Diff(1).apply(levels["gdp"]).to_numpy(),
        Diff(1).apply(levels["gdp"], "Q").to_numpy(),
        equal_nan=True,
    )


def test_off_slot_values_rejected(levels):
    gdp = levels["gdp"].copy()
    gdp.iloc[0] = 1.0
    with pytest.raises(NowcastDataError, match="outside"):
        Diff(1).apply(gdp, "Q")


def test_frequency_higher_than_index_rejected(quarterly_native):
    with pytest.raises(NowcastDataError, match="higher"):
        Diff(1).apply(quarterly_native, "M")


def test_monthly_lag_undefined_for_quarterly(quarterly_native):
    with pytest.raises(ValueError, match="undefined"):
        get_transform("mom").apply(quarterly_native)


def test_log_rejects_non_positive():
    x = pd.Series([1.0, 0.0, 2.0], index=pd.period_range("2020-01", periods=3, freq="M"))
    with pytest.raises(NowcastDataError, match="positive"):
        Log().apply(x)


def test_pct_change_zero_denominator_warns():
    x = pd.Series([0.0, 1.0, 2.0], index=pd.period_range("2020-01", periods=3, freq="M"))
    with pytest.warns(DataQualityWarning, match="division"):
        out = PctChange(1).apply(x)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(1.0)


def test_non_contiguous_index_is_regularised():
    idx = pd.PeriodIndex(["2020-01", "2020-02", "2020-04"], freq="M")
    x = pd.Series([1.0, 2.0, 4.0], index=idx)
    out = Diff(1).apply(x)
    assert out.index.equals(idx)
    assert out.tolist()[1] == 1.0
    assert np.isnan(out.iloc[2])  # previous month (March) unknown


@pytest.mark.parametrize(
    "bad",
    [
        [1.0, 2.0],
        pd.Series([1.0, 2.0]),
        pd.Series([1.0, np.inf], index=pd.period_range("2020-01", periods=2, freq="M")),
        pd.Series(["a", "b"], index=pd.period_range("2020-01", periods=2, freq="M")),
        pd.Series([1.0, 2.0], index=pd.PeriodIndex(["2020-01", "2020-01"], freq="M")),
    ],
)
def test_invalid_series_input(bad):
    with pytest.raises(NowcastDataError):
        Diff(1).apply(bad)


def test_scale_validation():
    for bad in [0, np.inf, "a", True]:
        with pytest.raises(ValueError):
            Scale(bad)


# ---------------------------------------------------------------------- composition


def test_compose_pipe_and_flatten():
    t = Log() | Diff(1) | Diff(12)
    assert isinstance(t, Compose)
    assert len(t.steps) == 3
    assert t.to_spec() == "log|diff(1)|diff(12)"
    assert Compose(Log() | Diff(1), Diff(12)) == t
    assert t.n_lags() == 13
    assert (Log() | "diff").to_spec() == "log|diff(1)"
    assert ("log" | Diff(1)).to_spec() == "log|diff(1)"
    assert (Log() | 2).to_spec() == "log|diff(1)"


def test_compose_validation():
    with pytest.raises(ValueError, match="at least one"):
        Compose()
    with pytest.raises(ValueError, match="Transform"):
        Compose("log")  # type: ignore[arg-type]


def test_equality_hash_repr():
    assert Diff(1) == get_transform("diff")
    assert Diff(1) != Diff(2)
    assert Diff(1).__eq__("diff") is NotImplemented
    assert len({Diff(1), get_transform(2), Log()}) == 2
    assert repr(Diff(3)) == "Transform('diff(3)')"
    assert str(Log() | Diff(1)) == "log|diff(1)"
    assert Identity().n_lags() == 0
    assert Diff("year").n_lags("M") == 12


# ---------------------------------------------------------------------- registry


def test_codes_table():
    assert set(TRANSFORM_CODES) == set(range(8))
    for name in TRANSFORM_CODES.values():
        assert name in NAMED_TRANSFORMS


@pytest.mark.parametrize("name", sorted(NAMED_TRANSFORMS))
def test_named_transforms_parse_and_round_trip(name):
    t = get_transform(name)
    assert get_transform(t.to_spec()) == t


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (None, "level"),
        (float("nan"), "level"),
        (0, "level"),
        (2.0, "diff(1)"),
        (np.int64(6), "pct_change(year)"),
        ("5", "diff(year)"),
        ("4.0", "diff(year)|diff(1)"),
        ("DLOG", "log|diff(1)"),
        ("log | diff( 12 )", "log|diff(12)"),
        ("pct_change(quarter)|scale(100)", "pct_change(quarter)|scale(100)"),
        ("identity", "level"),
        ("diff_of_yoy", "pct_change(year)|diff(1)"),
    ],
)
def test_get_transform_specs(spec, expected):
    assert get_transform(spec).to_spec() == expected


def test_get_transform_passthrough():
    t = Diff(2)
    assert get_transform(t) is t


@pytest.mark.parametrize(
    "bad",
    ["", "  ", "foo", "diff(", "log||diff", "log(2)", "scale", "scale(x)", "8", [1], True],
)
def test_get_transform_errors(bad):
    with pytest.raises(ValueError):
        get_transform(bad)


@pytest.mark.parametrize("bad", [8, -1, 1.5, "a", True])
def test_transform_from_code_errors(bad):
    with pytest.raises(ValueError):
        transform_from_code(bad)


# ---------------------------------------------------------------------- inverse


@pytest.mark.parametrize("code", range(8))
def test_inverse_round_trip_codes(monthly_series, code):
    t = transform_from_code(code)
    k = t.n_lags("M")
    y = t.apply(monthly_series)
    hist = monthly_series.iloc[: max(k, 1)]
    rebuilt = t.inverse(y.iloc[max(k, 1) :], hist)
    np.testing.assert_allclose(rebuilt.to_numpy(), monthly_series.to_numpy(), rtol=1e-10)
    assert rebuilt.index.equals(monthly_series.index)


@pytest.mark.parametrize(
    "spec", ["dlog", "dlog_yoy", "log", "scale(100)", "pct_change|scale(100)", "mom", "qoq"]
)
def test_inverse_round_trip_named(monthly_series, spec):
    t = get_transform(spec)
    k = max(t.n_lags("M"), 1)
    y = t.apply(monthly_series)
    rebuilt = t.inverse(y.iloc[k:], monthly_series.iloc[:k])
    np.testing.assert_allclose(rebuilt.to_numpy(), monthly_series.to_numpy(), rtol=1e-10)


def test_inverse_without_history_for_lagless_transforms(monthly_series):
    y = Log().apply(monthly_series)
    np.testing.assert_allclose(Log().inverse(y), monthly_series)
    np.testing.assert_allclose(Identity().inverse(y), y)
    np.testing.assert_allclose(Scale(2).inverse(Scale(2).apply(monthly_series)), monthly_series)


def test_inverse_requires_history_with_lags(monthly_series):
    with pytest.raises(ValueError, match="history"):
        Diff(1).inverse(Diff(1).apply(monthly_series))


def test_inverse_nowcast_extends_beyond_history(monthly_series):
    # rebuild the level of a future period from a growth nowcast
    hist = monthly_series
    nxt = hist.index[-1] + 1
    nowcast = pd.Series([0.01], index=pd.PeriodIndex([nxt]))
    out = get_transform("mom").inverse(nowcast, hist)
    assert out.index[-1] == nxt
    assert out.iloc[-1] == pytest.approx(hist.iloc[-1] * 1.01)
    pd.testing.assert_series_equal(out.iloc[:-1], hist, check_names=False)


def test_inverse_history_wins_and_gaps_propagate():
    idx = pd.period_range("2020-01", periods=5, freq="M")
    hist = pd.Series([1.0, np.nan, np.nan, 10.0, np.nan], index=idx)
    y = pd.Series([np.nan, 1.0, np.nan, 99.0, 1.0], index=idx)
    out = Diff(1).inverse(y, hist)
    assert out.tolist()[:2] == [1.0, 2.0]
    assert np.isnan(out.iloc[2])  # neither history nor transformed value
    assert out.iloc[3] == 10.0  # history wins over the transformed value
    assert out.iloc[4] == 11.0


def test_inverse_quarterly_on_monthly_grid(levels):
    gdp = levels["gdp"]
    t = get_transform("dlog")
    y = t.apply(gdp, "Q")
    rebuilt = t.inverse(y.iloc[3:], gdp.iloc[:3], "Q")
    np.testing.assert_allclose(rebuilt.to_numpy(), gdp.to_numpy(), equal_nan=True, rtol=1e-12)
    # without explicit frequency it is inferred from the combined pattern
    rebuilt2 = t.inverse(y.iloc[3:], gdp.iloc[:3])
    np.testing.assert_allclose(rebuilt2.to_numpy(), gdp.to_numpy(), equal_nan=True, rtol=1e-12)


def test_inverse_input_validation(monthly_series, quarterly_native):
    with pytest.raises(NowcastDataError, match="Series"):
        Diff(1).inverse([1.0], monthly_series)  # type: ignore[arg-type]
    with pytest.raises(NowcastDataError, match="PeriodIndex"):
        Diff(1).inverse(pd.Series([1.0]), monthly_series)
    with pytest.raises(NowcastDataError, match="same index frequency"):
        Diff(1).inverse(monthly_series, quarterly_native)
    dup = pd.Series([1.0, 2.0], index=pd.PeriodIndex(["2020-01", "2020-01"], freq="M"))
    with pytest.raises(NowcastDataError, match="duplicated"):
        Diff(1).inverse(dup, monthly_series)
    empty = monthly_series.iloc[:0]
    with pytest.raises(NowcastDataError, match="empty"):
        Log().inverse(empty, empty)


# ---------------------------------------------------------------------- panels


def test_apply_transforms_dataframe(levels):
    out = apply_transforms(levels, {"ip": "dlog", "sales": 2, "gdp": 7}, frequency=FREQS)
    assert isinstance(out, pd.DataFrame)
    np.testing.assert_allclose(out["ip"], np.log(levels["ip"]).diff(), equal_nan=True)
    np.testing.assert_allclose(out["sales"], levels["sales"].diff(), equal_nan=True)
    q = levels["gdp"].dropna()
    np.testing.assert_allclose(out["gdp"].dropna(), q.pct_change().dropna())


def test_apply_transforms_scalar_and_sequence(levels):
    a = apply_transforms(levels, "diff", frequency=FREQS)
    b = apply_transforms(levels, [2, 2, 2], frequency=FREQS)
    c = apply_transforms(levels, pd.Series({"ip": 2, "sales": 2, "gdp": 2}), frequency=FREQS)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)
    d = apply_transforms(levels, Diff(1), frequency=FREQS)
    pd.testing.assert_frame_equal(a, d)


def test_apply_transforms_errors(levels):
    with pytest.raises(NowcastDataError, match="unknown"):
        apply_transforms(levels, {"zzz": 1}, frequency=FREQS)
    with pytest.raises(NowcastDataError, match="elements"):
        apply_transforms(levels, [1, 2], frequency=FREQS)


def test_apply_transforms_panel_uses_metadata(panel):
    out = apply_transforms(panel)
    assert isinstance(out, MixedFrequencyData)
    ref = np.log(panel["ip"]).diff()
    np.testing.assert_allclose(out["ip"], ref, equal_nan=True)
    assert out.metadata["ip"].transform == "dlog"
    # mapping overrides only some series; the others keep their metadata transform
    out2 = apply_transforms(panel, {"ip": "level"})
    np.testing.assert_allclose(out2["ip"], panel["ip"], equal_nan=True)
    assert out2.metadata["ip"].transform == "level"
    np.testing.assert_allclose(out2["sales"], panel["sales"].diff(), equal_nan=True)


def test_apply_on_panel_via_transform_object(panel):
    out = Diff(1).apply(panel)
    assert isinstance(out, MixedFrequencyData)
    out_df = Diff(1).apply(panel.to_frame(), FREQS)
    assert isinstance(out_df, pd.DataFrame)
    np.testing.assert_allclose(out.to_frame(), out_df, equal_nan=True)
    with pytest.raises(NowcastDataError, match="frequency cannot"):
        Diff(1).apply(panel, FREQS)


def test_resolve_transforms(panel):
    assert resolve_transforms(panel, None) == {"ip": "dlog", "sales": 2, "gdp": 7}
    assert resolve_transforms(panel, 0) == {"ip": 0, "sales": 0, "gdp": 0}
    assert resolve_transforms(panel, {"gdp": 1})["gdp"] == 1
    assert resolve_transforms(panel, ["a", "b", "c"]) == {"ip": "a", "sales": "b", "gdp": "c"}


def test_invert_transforms_panel_round_trip(panel):
    tr = apply_transforms(panel)
    hist = panel.truncate(end="2015-12")
    lev = invert_transforms(tr, hist)
    assert isinstance(lev, MixedFrequencyData)
    np.testing.assert_allclose(lev.to_frame(), panel.to_frame(), equal_nan=True, rtol=1e-10)
    assert all(lev.metadata[c].transform is None for c in lev.columns)


def test_invert_transforms_uses_history_metadata(panel):
    tr = apply_transforms(panel).to_frame()
    tr_panel = MixedFrequencyData(tr, FREQS)
    lev = invert_transforms(tr_panel, panel.truncate(end="2015-12"))
    np.testing.assert_allclose(lev.to_frame(), panel.to_frame(), equal_nan=True, rtol=1e-10)


def test_invert_transforms_dataframe(levels):
    specs = {"ip": "dlog", "sales": 4, "gdp": "qoq"}
    tr = apply_transforms(levels, specs, frequency=FREQS)
    lev = invert_transforms(tr, levels.iloc[:13], specs, frequency=FREQS)
    assert isinstance(lev, pd.DataFrame)
    np.testing.assert_allclose(lev, levels, equal_nan=True, rtol=1e-10)


def test_invert_transforms_errors(panel, levels):
    tr = apply_transforms(panel)
    with pytest.raises(NowcastDataError, match="same series"):
        invert_transforms(tr, panel.select(["ip"]))
    q_levels = levels[["gdp"]].dropna()
    q_levels.index = q_levels.index.asfreq("Q")
    with pytest.raises(NowcastDataError, match="base frequency"):
        invert_transforms(tr.select(["gdp"]), MixedFrequencyData(q_levels, "Q"))


def test_transform_is_abstract():
    with pytest.raises(TypeError):
        Transform()  # type: ignore[abstract]


# ---------------------------------------------------------------- applied flag (integration fix)


def _levels_panel() -> MixedFrequencyData:
    idx = pd.period_range("2020-01", periods=12, freq="M")
    df = pd.DataFrame({"a": np.arange(1.0, 13.0), "b": np.arange(1.0, 13.0) ** 2}, index=idx)
    return MixedFrequencyData(df, "M", transforms={"a": "diff", "b": 0})


def test_metadata_transform_is_not_applied_twice():
    panel = _levels_panel()
    once = apply_transforms(panel)
    twice = apply_transforms(once)
    assert once.metadata["a"].transform_applied
    assert once.metadata["b"].transform_applied
    pd.testing.assert_frame_equal(once.data, twice.data)


def test_resolve_transforms_skips_applied_unless_requested():
    once = apply_transforms(_levels_panel())
    assert resolve_transforms(once, None) == {"a": None, "b": None}
    assert resolve_transforms(once, {"b": "log"}) == {"a": None, "b": "log"}
    assert resolve_transforms(once, None, include_applied=True) == {"a": "diff", "b": 0}


def test_explicit_transform_on_applied_series_warns():
    once = apply_transforms(_levels_panel())
    with pytest.warns(DataQualityWarning, match="already transformed"):
        apply_transforms(once, {"a": "diff"})


def test_invert_resets_applied_flag():
    panel = _levels_panel()
    once = apply_transforms(panel)
    back = invert_transforms(once, panel.truncate(end="2020-01"))
    np.testing.assert_allclose(back["a"].to_numpy(), panel["a"].to_numpy())
    assert back.metadata["a"].transform_applied is False
    assert back.metadata["a"].transform is None

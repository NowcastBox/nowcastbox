"""Tests of :func:`nowcastbox.diagnostics.indicator_zscores` (ECB parity item 4)."""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.diagnostics import (
    MM_SMOOTHING_WEIGHTS,
    IndicatorZScores,
    indicator_zscores,
    smooth_series,
)
from nowcastbox.models import TwoStepDFM

matplotlib.use("Agg")


def make_panel(n: int = 60, seed: int = 1) -> MixedFrequencyData:
    """Monthly ip/pmi/spread + quarterly gdp, ragged edge, metadata for groups."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2015-01", periods=n, freq="M")
    frame = pd.DataFrame(
        rng.normal(size=(n, 3)) + np.array([1.0, 50.0, -2.0]),
        index=idx,
        columns=["ip", "pmi", "spread"],
    )
    gdp = np.where(np.asarray(idx.month % 3 == 0), rng.normal(size=n), np.nan)
    frame["gdp"] = gdp
    frame.iloc[-2:, 0] = np.nan  # ip: two months pending
    frame.iloc[-1, 2] = np.nan  # spread: one month pending
    frame.iloc[-1, 3] = np.nan  # last gdp pending
    return MixedFrequencyData(
        frame,
        {"ip": "M", "pmi": "M", "spread": "M", "gdp": "Q"},
        categories={"ip": "hard", "pmi": "soft", "spread": "financial", "gdp": "hard"},
        blocks={"ip": ["global", "real"], "pmi": ["global"], "spread": ["global"], "gdp": ["real"]},
        release_delays={"ip": 40, "pmi": 1, "spread": 0, "gdp": 60},
    )


@pytest.fixture
def panel() -> MixedFrequencyData:
    return make_panel()


def test_mm_weights_sum_to_nine():
    raw = np.asarray(MM_SMOOTHING_WEIGHTS) * 9
    np.testing.assert_allclose(raw, [1, 2, 3, 2, 1])
    assert sum(MM_SMOOTHING_WEIGHTS) == pytest.approx(1.0)


def test_smooth_series_manual():
    x = np.arange(1.0, 9.0)
    out = smooth_series(x)
    assert np.isnan(out[:4]).all()
    manual = (x[4] + 2 * x[3] + 3 * x[2] + 2 * x[1] + x[0]) / 9
    assert out[4] == pytest.approx(manual)
    asym = smooth_series([1.0, 2.0, 4.0], [0.5, 0.5, 0.0])  # most recent first
    assert asym[2] == pytest.approx(3.0)
    assert np.isnan(smooth_series([1.0, 2.0])).all()  # shorter than the window
    with_gap = smooth_series([1.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.isnan(with_gap[4:7]).all()
    assert with_gap[7] == pytest.approx(1.0)


def test_known_zscores_without_smoothing():
    idx = pd.period_range("2020-01", periods=8, freq="M")
    x = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
    z = indicator_zscores(MixedFrequencyData(pd.DataFrame({"x": x}, index=idx), "M"), smooth=None)
    expected = (x - x.mean()) / x.std(ddof=1)
    np.testing.assert_allclose(z.zscores["x"], expected)
    assert isinstance(z, IndicatorZScores)
    assert z.smooth_weights is None
    assert z.series == ["x"]
    z0 = indicator_zscores(
        MixedFrequencyData(pd.DataFrame({"x": x}, index=idx), "M"), smooth="none", ddof=0
    )
    np.testing.assert_allclose(z0.zscores["x"], (x - x.mean()) / x.std(ddof=0))


def test_smoothing_then_standardisation(panel):
    z = indicator_zscores(panel)
    s = pd.Series(smooth_series(panel["pmi"].to_numpy()), index=panel.index)
    np.testing.assert_allclose(z.smoothed["pmi"], s, equal_nan=True)
    expected = (s - s.mean()) / s.std(ddof=1)
    np.testing.assert_allclose(z.zscores["pmi"], expected, equal_nan=True)
    # quarterly series are not smoothed
    np.testing.assert_allclose(z.smoothed["gdp"], panel["gdp"], equal_nan=True)
    assert z.zscores["gdp"].notna().sum() == panel["gdp"].notna().sum()


def test_nan_at_the_edges(panel):
    z = indicator_zscores(panel)
    assert z.zscores["pmi"].iloc[:4].isna().all()
    assert z.zscores["pmi"].iloc[4:].notna().all()
    assert z.zscores["ip"].iloc[-2:].isna().all()
    assert z.zscores["spread"].iloc[-1:].isna().all()
    assert z.zscores.index[-1] == panel.index[-1]  # pmi complete: no trimming


def test_trailing_rows_trimmed_after_as_of(panel):
    z = indicator_zscores(panel, as_of="2019-06-15")
    assert z.zscores.index[-1] == pd.Period("2019-05", freq="M")  # pmi of May: June 1
    assert z.as_of == pd.Timestamp("2019-06-15")


def test_custom_weights_are_normalised(panel):
    z = indicator_zscores(panel, smooth=[2.0, 2.0])
    assert z.smooth_weights == (0.5, 0.5)
    with pytest.raises(ValueError, match="positive sum"):
        indicator_zscores(panel, smooth=[1.0, -1.0])
    with pytest.raises(ValueError, match="positive sum"):
        indicator_zscores(panel, smooth=[])
    with pytest.raises(ValueError, match="smooth must be"):
        indicator_zscores(panel, smooth="hp")
    assert indicator_zscores(panel, smooth="mariano_murasawa").smooth_weights == (
        MM_SMOOTHING_WEIGHTS
    )


def test_rolling_window(panel):
    z = indicator_zscores(panel, window=24, min_periods=12, series=["pmi"])
    s = z.smoothed["pmi"]
    mean = s.rolling(24, min_periods=12).mean()
    std = s.rolling(24, min_periods=12).std(ddof=1)
    np.testing.assert_allclose(z.zscores["pmi"], (s - mean) / std, equal_nan=True)
    assert z.window == 24
    assert z.zscores["pmi"].iloc[:15].isna().all()  # 4 smoothing + 11 short windows


def test_no_look_ahead(panel):
    vintage = "2018-03-20"
    base = indicator_zscores(panel, as_of=vintage, window=None)
    frame = panel.to_frame()
    cut = panel.as_of(vintage).to_frame()
    future = cut.isna() & frame.notna()
    altered = frame.mask(future, frame + 100.0)
    changed = panel.with_data(altered)
    again = indicator_zscores(changed, as_of=vintage, window=None)
    pd.testing.assert_frame_equal(base.zscores, again.zscores)
    pd.testing.assert_frame_equal(base.mean, again.mean)
    # the same computation without as_of does see the future
    full = indicator_zscores(changed)
    assert not np.allclose(full.mean["pmi"].iloc[0], base.mean["pmi"].iloc[0])


def test_rolling_window_is_trailing(panel):
    z = indicator_zscores(panel, window=12)
    frame = panel.to_frame()
    frame.iloc[40:, :] = frame.iloc[40:, :] * 3.0 + 7.0
    z2 = indicator_zscores(panel.with_data(frame), window=12)
    pd.testing.assert_frame_equal(z.zscores.iloc[:40], z2.zscores.iloc[:40])


def test_groups_category_block_mapping(panel):
    z = indicator_zscores(panel, by="category")
    assert list(z.membership) == ["hard", "soft", "financial"]
    hard = z.zscores[["ip", "gdp"]].mean(axis=1)
    pd.testing.assert_series_equal(z.groups["hard"], hard, check_names=False)
    blocks = indicator_zscores(panel, by="block")
    assert blocks.membership["global"] == ("ip", "pmi", "spread")
    assert blocks.membership["real"] == ("ip", "gdp")
    mapped = indicator_zscores(panel, by={"ip": "activity", "pmi": ["activity", "surveys"]})
    assert set(mapped.membership) == {"activity", "surveys", "unassigned"}
    with pytest.raises(ValueError, match="by must be"):
        indicator_zscores(panel, by="nonsense")


def test_table_latest_summary(panel):
    z = indicator_zscores(panel, by="category")
    table = z.table(last=6)
    assert table.shape == (4, 6)
    assert z.table("group").shape[0] == 3
    quarterly = z.table(frequency="Q", last=4)
    assert quarterly.columns.astype(str).tolist() == ["2019Q1", "2019Q2", "2019Q3", "2019Q4"]
    latest = z.latest()
    assert str(latest.loc["ip", "period"]) == "2019-10"
    assert latest.loc["pmi", "zscore"] == pytest.approx(z.zscores["pmi"].iloc[-1])
    assert str(z.latest("group").loc["hard", "period"]) == "2019-10"
    text = z.summary()
    assert "group" in text and "series" in text and "whole sample" in text
    assert "rolling 12" in indicator_zscores(panel, window=12, smooth=None).summary()
    with pytest.raises(ValueError, match="last must be"):
        z.table(last=0)
    with pytest.raises(ValueError, match="level must be"):
        z.table("block")
    with pytest.raises(ValueError, match="No grouping"):
        indicator_zscores(panel).table("group")


def test_latest_all_missing_series():
    idx = pd.period_range("2020-01", periods=12, freq="M")
    df = pd.DataFrame({"x": np.arange(12.0), "y": [np.nan] * 12}, index=idx)
    with pytest.warns(DataQualityWarning, match="'y'"):
        z = indicator_zscores(MixedFrequencyData(df, "M"), smooth=None)
    latest = z.latest()
    assert pd.isna(latest.loc["y", "period"])
    assert np.isnan(latest.loc["y", "zscore"])


def test_constant_series_warns():
    idx = pd.period_range("2020-01", periods=12, freq="M")
    df = pd.DataFrame({"c": np.ones(12), "x": np.arange(12.0)}, index=idx)
    with pytest.warns(DataQualityWarning, match="zero variance"):
        z = indicator_zscores(MixedFrequencyData(df, "M"))
    assert z.zscores["c"].isna().all()
    assert z.zscores["x"].notna().any()


def test_all_missing_panel_is_empty():
    idx = pd.period_range("2020-01", periods=4, freq="M")
    df = pd.DataFrame({"x": [np.nan] * 4}, index=idx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        z = indicator_zscores(MixedFrequencyData(df, "M"))
    assert z.zscores.shape == (0, 1)


def test_option_errors(panel):
    with pytest.raises(ValueError, match="window"):
        indicator_zscores(panel, window=1)
    with pytest.raises(ValueError, match="ddof"):
        indicator_zscores(panel, ddof=-1)
    with pytest.raises(ValueError, match="min_periods"):
        indicator_zscores(panel, min_periods=1)
    with pytest.raises(ValueError, match="min_periods"):
        indicator_zscores(panel, ddof=3, min_periods=3)
    with pytest.raises(ValueError, match="cannot exceed"):
        indicator_zscores(panel, window=5, min_periods=6)
    with pytest.raises(ValueError, match="Unknown series"):
        indicator_zscores(panel, series=["nope"])
    no_delay = MixedFrequencyData(panel.to_frame()[["pmi"]], "M")
    with pytest.raises(NowcastDataError):
        indicator_zscores(no_delay, as_of="2018-01-01")


def test_series_subset_and_dataframe_input(panel):
    z = indicator_zscores(panel, series=["pmi", "ip"])
    assert z.series == ["pmi", "ip"]
    frame = panel.to_frame()[["pmi"]].copy()
    frame.index = frame.index.to_timestamp()
    zf = indicator_zscores(frame)
    np.testing.assert_allclose(zf.zscores["pmi"], z.zscores["pmi"], equal_nan=True)


def test_results_input_drops_target(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = TwoStepDFM(n_factors=1).fit(panel, "gdp")
    z = indicator_zscores(res)
    assert "gdp" not in z.series
    assert indicator_zscores(res, series=["gdp", "ip"]).series == ["gdp", "ip"]
    with pytest.raises(ValueError, match="estimation data"):
        indicator_zscores(res.replace(data=None))


def test_plot_method(panel):
    z = indicator_zscores(panel, by="category")
    fig = z.plot(last=12)
    assert fig.data[0].type == "heatmap"
    assert len(fig.data[0].y) == 3  # groups by default
    with pytest.raises(ValueError, match="Unknown plot kind"):
        z.plot("bars")

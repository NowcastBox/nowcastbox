"""Tests of the indicator z-score heatmap and the released-share chart."""

from __future__ import annotations

import doctest

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

import nowcastbox.visualization as viz
import nowcastbox.visualization.data_flow as data_flow_mod
import nowcastbox.visualization.heatmap as heatmap_mod
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.results import available_plots
from nowcastbox.diagnostics import indicator_zscores


@pytest.fixture
def grouped(panel):
    groups = {c: ("hard" if i % 2 else "soft") for i, c in enumerate(panel.columns)}
    return panel.with_data(panel.to_frame()), groups


def test_registered(two_step, em):
    for res in (two_step, em):
        assert "indicator_heatmap" in available_plots(res)
        assert "released_share" in available_plots(res)


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_results_plot(two_step, backend):
    fig = two_step.plot("indicator_heatmap", backend=backend)
    assert fig is not None
    fig2 = two_step.plot("released_share", backend=backend)
    assert fig2 is not None


def test_results_heatmap_excludes_target(two_step):
    fig = two_step.plot("indicator_heatmap", last=6)
    rows = list(fig.data[0].y)
    assert "gdp" not in rows
    assert len(fig.data[0].x) == 6


def test_heatmap_table_levels(panel):
    groups = {c: "even" if i % 2 == 0 else "odd" for i, c in enumerate(panel.columns)}
    by_group = viz.heatmap_table(panel, by=groups, last=5)
    assert list(by_group.index) == ["even", "odd"]
    by_series = viz.heatmap_table(panel, by=groups, level="series", last=5)
    assert list(by_series.index) == panel.columns
    z = indicator_zscores(panel)
    pd.testing.assert_frame_equal(viz.heatmap_table(z, last=None), z.table())
    quarterly = viz.heatmap_table(z, frequency="Q", last=4)
    assert isinstance(quarterly.columns, pd.PeriodIndex)
    with pytest.raises(ValueError, match="level must be"):
        viz.heatmap_table(z, level="block")
    with pytest.raises(TypeError, match="Unexpected arguments"):
        viz.heatmap_table(z, by="category")


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_plot_options(panel, backend):
    z = indicator_zscores(panel)
    fig = viz.plot_indicator_heatmap(
        z, backend=backend, annotate=True, zmax=3.0, title="Heat", frequency="Q", last=8
    )
    if backend == "plotly":
        trace = fig.data[0]
        assert trace.zmin == -3.0 and trace.zmax == 3.0 and trace.zmid == 0.0
        assert trace.texttemplate == "%{z:.1f}"
        assert fig.layout.title.text == "Heat"
    else:
        assert isinstance(fig, matplotlib.figure.Figure)
        assert any(t.get_text() for t in fig.axes[0].texts)


def test_plot_on_existing_axes(panel):
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    fig = viz.plot_indicator_heatmap(panel, backend="matplotlib", ax=ax)
    assert fig is ax.get_figure()


def test_plot_errors(panel):
    with pytest.raises(ValueError, match="zmax"):
        viz.plot_indicator_heatmap(panel, zmax=0.0)
    with pytest.raises(ValueError, match="backend"):
        viz.plot_indicator_heatmap(panel, backend="bokeh")


# ---------------------------------------------------------------------- released share
@pytest.fixture
def small() -> MixedFrequencyData:
    idx = pd.period_range("2020-01", periods=6, freq="M")
    df = pd.DataFrame(
        {
            "ip": [1.0, 1.0, 1.0, 1.0, np.nan, np.nan],
            "pmi": [1.0] * 6,
            "gdp": [np.nan, np.nan, 1.0, np.nan, np.nan, np.nan],
        },
        index=idx,
    )
    return MixedFrequencyData(
        df, {"ip": "M", "pmi": "M", "gdp": "Q"}, categories={"ip": "hard", "pmi": "soft"}
    )


def test_released_share_table_panel(small):
    table = viz.released_share_table(small, "2020Q2")
    assert list(table.index) == ["hard", "soft", "uncategorized", "total"]
    assert table.loc["total", "share"] == pytest.approx(4 / 7)
    with pytest.raises(ValueError, match="period is required"):
        viz.released_share_table(small)


def test_released_share_table_results(two_step):
    table = viz.released_share_table(two_step, by="series")
    assert "gdp" not in table.index
    period = two_step.nowcast["out_of_sample"].dropna().index
    manual = two_step.data.released_share(
        period[period > two_step.nowcast["observed"].last_valid_index()][0],
        series=[c for c in two_step.data.columns if c != "gdp"],
    )
    pd.testing.assert_frame_equal(table, manual)


def test_released_share_results_with_series(two_step):
    table = viz.released_share_table(two_step, "2014Q4", by=None, series=["gdp", "x1"])
    assert list(table.index) == ["gdp", "x1", "total"]


def test_released_share_results_without_future_period(two_step):
    observed = two_step.nowcast.copy()
    observed["observed"] = observed["observed"].fillna(0.0)
    res = two_step.replace(nowcast=observed)
    with pytest.raises(ValueError, match="No target period"):
        viz.released_share_table(res)


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_plot_released_share(small, backend):
    fig = viz.plot_released_share(small, "2020Q2", backend=backend)
    if backend == "plotly":
        np.testing.assert_allclose(fig.data[0].x, [100 / 3, 100.0, 0.0])
        assert fig.layout.title.text == "Data released for 2020Q2: 57%"
        assert fig.layout.shapes  # total line
    else:
        assert isinstance(fig, matplotlib.figure.Figure)


@pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.DataQualityWarning")
@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_plot_released_share_nan_total(backend):
    idx = pd.period_range("2020-01", periods=3, freq="M")
    mfd = MixedFrequencyData(pd.DataFrame({"a": [np.nan] * 3}, index=idx), "A")
    fig = viz.plot_released_share(mfd, "2020Q1", by=None, backend=backend, title="t")
    assert fig is not None


@pytest.mark.parametrize("module", [heatmap_mod, data_flow_mod], ids=lambda m: m.__name__)
def test_doctests(module):
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.failed == 0
    assert result.attempted > 0

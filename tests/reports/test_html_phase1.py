"""Report sections of the ECB-parity features (empirical bands, alternative models,
released-data share, indicator heatmap, backtest accuracy table)."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from nowcastbox.density import empirical_bands
from nowcastbox.diagnostics import indicator_zscores
from nowcastbox.evaluation.backtest import BacktestResults
from nowcastbox.experiment import alternative_models
from nowcastbox.models import TwoStepDFM
from nowcastbox.reports import NowcastReport
from nowcastbox.reports.html import _flat_index
from tests.density.test_empirical import make_table
from tests.visualization import _fixtures

GROUPS = {"real": ["x1", "x2"], "soft": ["x3", "x4"], "fin": ["x5", "x6"]}


@pytest.fixture(scope="module")
def bands():
    results = _fixtures.two_step_results()
    backtest = BacktestResults(make_table("1990-01", "2007-03"), ["DFM"], "gdp")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return empirical_bands(results, backtest, vintage="2007-04-20")


@pytest.fixture(scope="module")
def alternatives():
    return alternative_models(TwoStepDFM(n_factors=2), _fixtures.panel(), "gdp", by=GROUPS, drop=1)


def test_bands_and_alternatives_sections(two_step, bands, alternatives):
    html = NowcastReport(two_step, bands=bands, alternatives=alternatives, plotlyjs="cdn").render()
    assert "57.5% empirical band" in html and "90% empirical band" in html
    assert "Empirical error bands" in html
    assert "Nowcasts of alternative models" in html
    assert "-real" in html and "Chart not available" not in html


def test_bands_tiles_skip_periods_without_band(two_step, bands):
    report = NowcastReport(two_step, bands=bands, plotlyjs="cdn")
    assert report._band_tiles(pd.Period("2030Q1", "Q")) == []
    assert len(report._band_tiles(bands.index[0])) == len(bands.levels)


def test_released_share_and_heatmap(two_step):
    html = NowcastReport(two_step, heatmap=True, heatmap_last=12, plotlyjs="cdn").render()
    assert "Share of the 2007Q2 data already released" in html
    assert "Indicator z-scores" in html
    assert "Chart not available" not in html


def test_heatmap_object_and_disabled_share(two_step):
    z = indicator_zscores(two_step.data, by={"x1": "real", "x2": "real", "x3": "soft"})
    html = NowcastReport(two_step, heatmap=z, released_share=False, plotlyjs="cdn").render()
    assert "data already released" not in html
    assert html.count("plotly-graph-div") >= 2


def test_released_share_failure_is_a_placeholder(two_step, monkeypatch):
    import nowcastbox.reports.html as html_module

    def boom(*args, **kwargs):
        raise ValueError("no share")

    monkeypatch.setattr(html_module, "released_share_table", boom)
    html = NowcastReport(two_step, plotlyjs="cdn").render()
    assert "Released-data share not available: no share" in html


def test_backtest_metrics_table(two_step):
    index = pd.MultiIndex.from_tuples([("Covid", 0), ("Covid", 1)], names=["period", "h"])
    columns = pd.MultiIndex.from_tuples([("rmsfe", "DFM"), ("fda", "DFM")])
    metrics = pd.DataFrame([[1.0, 0.5], [2.0, 0.75]], index=index, columns=columns)
    html = NowcastReport(two_step, backtest_metrics=metrics, plotlyjs="cdn").render()
    assert "Backtest accuracy" in html and "Covid | 1" in html and "fda | DFM" in html


def test_flat_index_keeps_flat_frames():
    frame = pd.DataFrame({"a": [1.0]}, index=pd.Index(["x"], name="k"))
    assert _flat_index(frame).equals(frame)


def test_invalid_heatmap_last(two_step):
    with pytest.raises(ValueError, match="heatmap_last"):
        NowcastReport(two_step, heatmap_last=0)

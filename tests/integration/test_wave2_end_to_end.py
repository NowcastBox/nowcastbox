"""End-to-end integration test of the wave-2 modules on the Brazilian dataset.

``nb.load_brazil_nowcast()`` (15 series, 2010-2023) -> ``prepare_panel`` -> robust EM
(Student-t errors and pandemic dummies, I3) -> news between two pseudo real-time
vintages and the nowcast tracker (I6) -> density nowcast with bootstrap (I5) and its
scores -> short pseudo real-time backtest against an AR benchmark -> diagnostics (I9)
-> HTML report written to ``tmp_path``. Everything goes through the public namespace
(``nb.*`` and the results' methods), as a user would.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nowcastbox as nb

pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.ConvergenceWarning"),
    pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.DataQualityWarning"),
]

TARGET = "pib"
PERIOD = "2023Q4"
OLD_VINTAGE = "2023-11-01"
NEW_VINTAGE = "2023-12-15"


@dataclass(frozen=True)
class Pipeline:
    """Objects produced by the wave-2 workflow (computed once per module)."""

    dataset: nb.Dataset
    panel: nb.MixedFrequencyData
    old: nb.MixedFrequencyData
    new: nb.MixedFrequencyData
    results: nb.MixedFreqDFMResults


def _columns(legend: pd.DataFrame) -> list[str]:
    monthly = legend[legend["frequency"] == "M"]
    hard = list(monthly[monthly["category"] == "hard"].index[:10])
    other = list(monthly[monthly["category"] != "hard"].index[:4])
    return [TARGET, *hard, *other]


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    full = nb.load_brazil_nowcast()
    ds = full.select(_columns(full.legend)).truncate("2010-01", "2023-12")
    panel = nb.prepare_panel(ds.data, ds.transform)
    old = nb.pseudo_real_time(panel, delay=ds.delay, vintage=OLD_VINTAGE)
    new = nb.pseudo_real_time(panel, delay=ds.delay, vintage=NEW_VINTAGE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = nb.MixedFreqDFM(n_factors=1, idiosyncratic="student_t", covid="dummy", max_iter=50)
        results = model.fit(new, target=TARGET)
    return Pipeline(ds, panel, old, new, results)


def test_dataset_and_robust_fit(pipeline: Pipeline) -> None:
    ds, res = pipeline.dataset, pipeline.results
    assert ds.target == TARGET
    assert pipeline.panel.metadata[TARGET].frequency == nb.Frequency.QUARTERLY
    assert isinstance(res, nb.MixedFreqDFMResults)
    assert res.student_t_df is not None and res.student_t_df > 2.0
    assert res.interventions is not None and len(res.interventions) > 0
    assert res.observation_weights is not None
    assert np.isfinite(res.get_nowcast(PERIOD))
    # the ragged edge of the new vintage: the target is not yet observed in 2023Q4
    assert np.isnan(res.nowcast.loc[pd.Period(PERIOD), "observed"])


def test_news_and_tracker(pipeline: Pipeline) -> None:
    res = pipeline.results
    news = res.news(old=pipeline.old, new=pipeline.new, target_period=PERIOD)
    assert isinstance(news, nb.NewsResults)
    assert news.n_releases > 0
    assert news.check_identity()
    by_category = news.to_frame(by="category")
    assert {"hard", "financial"} & set(by_category.index)
    assert news.plot("waterfall", backend="plotly").data
    assert news.plot("waterfall").axes  # default Matplotlib chart of nowcastbox.news

    calendar = pipeline.dataset.calendar()
    tracker = res.nowcast_tracker(pipeline.panel, calendar, PERIOD, "2023-10-01", "2023-12-31")
    assert isinstance(tracker, nb.NowcastTracker)
    assert tracker.check_identity()
    assert tracker.plot("path", backend="plotly").data

    level = res.level_contributions(pipeline.new, PERIOD)
    assert level.check_identity()


def test_density_and_scores(pipeline: Pipeline) -> None:
    res = pipeline.results
    dist = res.distribution(n_boot=5, random_state=0)
    assert isinstance(dist, nb.NowcastDistribution)
    assert pd.Period(PERIOD) in dist.index
    low, high = dist.interval(0.9).loc[pd.Period(PERIOD)]
    assert low < res.get_nowcast(PERIOD) < high
    # scores of the distribution at a hypothetical outcome
    outcome = float(dist.select([PERIOD]).mean.iloc[0])
    crps = nb.scoring.crps(dist.select([PERIOD]), [outcome])
    assert np.all(np.asarray(crps) >= 0)
    pit = nb.evaluation.pit(dist.select([PERIOD]), [outcome])
    assert 0.0 < float(np.asarray(pit)[0]) < 1.0
    assert res.plot("density", n_boot=0).data


def test_backtest_against_ar(pipeline: Pipeline) -> None:
    ds = pipeline.dataset
    backtest = nb.PseudoRealTimeBacktest(
        model=nb.MixedFreqDFM(n_factors=1, max_iter=20),
        data=pipeline.panel,
        target=TARGET,
        delay=ds.delay,
        start="2023-01-01",
        end="2023-04-01",
        step="M",
        benchmarks=[nb.benchmarks.AR(p=1)],
    )
    out = backtest.run()
    assert isinstance(out, nb.BacktestResults)
    table = out.rmsfe_by_horizon()
    assert {"MixedFreqDFM", "AR"} <= set(table.columns)
    assert np.isfinite(table.to_numpy()).any()
    assert "Pseudo real-time backtest" in out.summary()


def test_diagnostics(pipeline: Pipeline) -> None:
    report = pipeline.results.diagnostics()
    assert isinstance(report, nb.DiagnosticsReport)
    assert report.convergence is not None and report.convergence.converged
    overview = report.series_overview()
    assert TARGET in overview.index
    assert "Loading stability" in report.summary()


def test_html_report(pipeline: Pipeline, tmp_path: Path) -> None:
    res = pipeline.results
    news = res.news(pipeline.old, pipeline.new, PERIOD)
    tracker = res.nowcast_tracker(
        pipeline.panel, pipeline.dataset.calendar(), PERIOD, "2023-10-01", "2023-12-31"
    )
    dist = res.distribution()
    report = nb.NowcastReport(
        res,
        news=news,
        tracker=tracker,
        quantiles=dist,
        diagnostics=True,
        author="nowcastbox integration test",
        plotlyjs="cdn",
    )
    path = tmp_path / "report.html"
    html = report.to_html(path)
    assert path.exists() and path.stat().st_size > 10_000
    assert html == path.read_text(encoding="utf-8")
    for section in ("headline", "path", "news", "data_flow", "diagnostics"):
        assert f'id="{section}"' in html
    assert "Chart not available" not in html
    assert "DFM diagnostics by series" in html

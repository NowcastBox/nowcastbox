import re

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.results import FactorResults, NowcastResults, build_nowcast_frame
from nowcastbox.reports import REPORT_SECTIONS, NowcastReport, nowcast_period
from nowcastbox.reports.html import _fmt


def _sections(html):
    return re.findall(r'data-section="(\w+)"', html)


@pytest.fixture
def news():
    return pd.DataFrame(
        {
            "series": ["x1", "x2", "x3"],
            "impact": [0.1, -0.04, 0.02],
            "category": ["hard", "soft", "hard"],
        }
    )


def _plain(observed_until_end=False):
    idx = pd.period_range("2020Q1", periods=4, freq="Q")
    obs = [1.0, 2.0, 1.5, 3.0] if observed_until_end else [1.0, 2.0, 1.5, np.nan]
    f = build_nowcast_frame(pd.Series(obs, index=idx), pd.Series([1.1, 1.8, 1.6, 1.2], index=idx))
    return NowcastResults(target="gdp", nowcast=f, model_name="Demo")


def test_full_report_sections(em, news, tmp_path):
    bt = pd.DataFrame({"DFM": [0.5, 0.6], "AR": [1.0, 1.0]}, index=pd.Index([0, 1], name="h"))
    idx = pd.period_range("2007-04", periods=3, freq="M")
    tracker = pd.DataFrame(
        {"hard": [0.1, 0.0, -0.1], "soft": [0.0, 0.1, 0.0], "nowcast": [1.0, 1.1, 1.0]}, index=idx
    )
    quantiles = pd.DataFrame(
        {"q05": [1.0], "q50": [2.0], "q95": [3.0]},
        index=pd.period_range("2007Q2", periods=1, freq="Q"),
    )
    report = NowcastReport(
        em,
        news=news,
        old_nowcast=1.8,
        news_group_by="category",
        tracker=tracker,
        quantiles=quantiles,
        backtest=bt,
        backtest_reference="AR",
        author="Team",
        notes="Note <b>x</b>",
    )
    path = tmp_path / "sub" / "report.html"
    html = report.to_html(path)
    assert path.read_text(encoding="utf-8") == html
    assert html.startswith("<!doctype html>")
    assert _sections(html) == [sid for sid, _ in REPORT_SECTIONS]
    for title in ("Headline nowcast", "Nowcast path", "News", "Data flow", "Diagnostics"):
        assert title in html
    period = str(nowcast_period(em))
    assert f"Nowcast {period}" in html
    assert "90% interval" in html and "Change vs previous" in html
    assert "Impacts on the nowcast" in html and "Release status by series" in html
    assert "RMSFE by horizon" in html and "Team" in html
    assert "Note &lt;b&gt;x&lt;/b&gt;" in html  # escaped
    assert html.count("plotly-graph-div") >= 9
    assert "Chart not available" not in html
    assert "No news decomposition supplied" not in html
    # self-contained: plotly.js inline exactly once, no CDN
    assert html.count("plotly.js v") <= 2
    assert 'src="https://cdn.plot.ly' not in html
    assert len(html) > 3_000_000


def test_cdn_report_is_small(two_step):
    html = NowcastReport(two_step, plotlyjs="cdn").render()
    assert 'src="https://cdn.plot.ly' in html
    assert len(html) < 1_000_000
    assert "No news decomposition supplied" in html
    assert "Pass diagnostics=True" in html
    assert "Scree plot" in html


def test_plain_results_placeholders():
    html = NowcastReport(_plain(), plotlyjs="cdn", title="My <report>").render()
    assert "<title>My &lt;report&gt;</title>" in html
    assert "The results do not carry their estimation data." in html
    assert _sections(html) == [sid for sid, _ in REPORT_SECTIONS]
    assert "Nowcast 2020Q4" in html


def test_target_observed_until_end():
    html = NowcastReport(_plain(observed_until_end=True), plotlyjs="cdn").render()
    assert "no nowcast period" in html


def test_figure_failures_become_placeholders(two_step, news):
    report = NowcastReport(
        two_step, news=news, news_group_by="nope", backtest="bad", plotlyjs="cdn"
    )
    html = report.render()
    assert "Chart not available" in html
    assert "News table not available" in html
    assert "Backtest table not available" in html


def test_news_object_change_tile(em, news):
    class NewsObj:
        releases = news
        old_nowcast = 0.5
        new_nowcast = 0.7
        target_period = nowcast_period(em)

    html = NowcastReport(em, news=NewsObj(), plotlyjs="cdn").render()
    assert "Change vs previous" in html
    assert "Previous nowcast" in html


def test_factor_results_without_loadings():
    base = _plain()
    res = FactorResults(
        target="gdp",
        nowcast=base.nowcast,
        factors=pd.DataFrame(
            {"f1": [0.0, 1.0]}, index=pd.period_range("2020-01", periods=2, freq="M")
        ),
    )
    html = NowcastReport(res, plotlyjs="cdn").render()
    assert "Chart not available" in html  # eigenvalues need data
    assert "Estimated factors" in html


def test_context_and_repr(two_step):
    report = NowcastReport(two_step, plotlyjs="cdn")
    ctx = report.context()
    assert ctx["meta"]["model_name"] == "TwoStepDFM"
    assert [s["id"] for s in ctx["sections"]] == [sid for sid, _ in REPORT_SECTIONS]
    assert "NowcastReport" in repr(report)


def test_custom_template(tmp_path, two_step):
    (tmp_path / "nowcast_report.html").write_text(
        "{{ title }}|{% for s in sections %}{{ s.id }},{% endfor %}", encoding="utf-8"
    )
    out = NowcastReport(two_step, template_dir=tmp_path, title="X", plotlyjs="cdn").render()
    assert out == "X|headline,path,news,data_flow,diagnostics,"


def test_invalid_arguments(two_step):
    with pytest.raises(TypeError):
        NowcastReport("x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="plotlyjs"):
        NowcastReport(two_step, plotlyjs="local")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_periods"):
        NowcastReport(two_step, n_periods=0)


def test_nowcast_period_none_observed():
    idx = pd.period_range("2020Q1", periods=2, freq="Q")
    f = build_nowcast_frame(
        pd.Series([np.nan, np.nan], index=idx), pd.Series([1.0, 2.0], index=idx)
    )
    assert str(nowcast_period(NowcastResults(target="y", nowcast=f))) == "2020Q1"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "-"),
        (True, "yes"),
        (np.bool_(False), "no"),
        (3, "3"),
        (np.int64(4), "4"),
        (1234.5678, "1,234.568"),
        (float("nan"), "-"),
        (pd.NA, "-"),
        (pd.NaT, "-"),
        ("a", "a"),
    ],
)
def test_fmt(value, expected):
    assert _fmt(value) == expected


# ---------------------------------------------------------------------- wave-2 inputs
def test_dfm_diagnostics_section(two_step):
    """``diagnostics=True`` runs :func:`run_diagnostics` (I9) and tabulates it."""
    html = NowcastReport(two_step, plotlyjs="cdn", diagnostics=True).render()
    assert "DFM diagnostics by series" in html
    assert "Loading stability" in html
    assert "Pass diagnostics=True" not in html


def test_dfm_diagnostics_object_and_empty_overview(two_step):
    class _Empty:
        def series_overview(self):
            return pd.DataFrame()

        def summary(self):
            return "custom diagnostics summary"

    html = NowcastReport(two_step, plotlyjs="cdn", diagnostics=_Empty()).render()
    assert "custom diagnostics summary" in html
    assert "DFM diagnostics by series" not in html


def test_dfm_diagnostics_unavailable():
    html = NowcastReport(_plain(), plotlyjs="cdn", diagnostics=True).render()
    assert "Diagnostics not available" in html


def test_distribution_fan_chart(two_step):
    """A :class:`NowcastDistribution` is accepted as ``quantiles`` (I5)."""
    from nowcastbox.density import nowcast_distribution

    dist = nowcast_distribution(two_step)
    html = NowcastReport(two_step, plotlyjs="cdn", quantiles=dist).render()
    assert "Chart not available" not in html

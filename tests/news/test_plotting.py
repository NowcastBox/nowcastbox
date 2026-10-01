"""Tests of the news plot registry and matplotlib defaults."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import nowcastbox.news.plotting as plotting
from nowcastbox.news import NewsResults, available_news_plots, register_news_plot
from nowcastbox.news.plotting import bar_chart, tracker_chart, waterfall_chart


def test_available():
    kinds = available_news_plots(NewsResults)
    assert "waterfall" in kinds


def test_register_and_dispatch(em_results, panel):
    from nowcastbox.news import news_decomposition

    @register_news_plot("custom_kind", NewsResults)
    def _custom(news, **kwargs):
        return ("figure", kwargs)

    news = news_decomposition(em_results, panel, panel, "2014Q4")
    assert news.plot("custom_kind", a=1) == ("figure", {"a": 1})
    assert "custom_kind" in available_news_plots(news)
    del plotting._REGISTRY[(NewsResults, "custom_kind")]


def test_visualization_import_failure_is_ignored(monkeypatch, em_results, panel):
    from nowcastbox.news import news_decomposition

    news = news_decomposition(em_results, panel, panel, "2014Q4")

    def boom(name):
        raise ImportError(name)

    monkeypatch.setattr(plotting.importlib, "import_module", boom)
    with pytest.raises(NotImplementedError):
        news.plot("does_not_exist")


def test_charts_on_existing_axes():
    fig, ax = plt.subplots()
    out = waterfall_chart(pd.Series([1.0, 0.5, -0.2, 1.3], index=list("abcd")), ax=ax)
    assert out is fig
    idx = pd.period_range("2020-01", periods=2, freq="M")
    out2 = tracker_chart(
        pd.Series([1.0, 2.0], index=idx),
        pd.DataFrame({"g": [0.0, 1.0]}, index=idx),
        ax=ax,
        title="t",
    )
    assert out2 is fig
    out3 = bar_chart(pd.Series({"a": 1.0}), ax=ax, title="b")
    assert out3 is fig
    plt.close("all")

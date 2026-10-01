import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nowcastbox.visualization import (
    news_waterfall_table,
    plot_news_waterfall,
    plot_nowcast_tracker,
    tracker_table,
)


@pytest.fixture
def news():
    return pd.DataFrame(
        {
            "series": ["ip", "pmi", "retail", "spread"],
            "impact": [0.10, -0.04, 0.02, 0.05],
            "category": ["hard", "soft", "hard", "financial"],
            "actual": [1.0, 50.0, 0.3, 2.0],
            "expected": [0.8, 51.0, 0.2, 1.9],
        }
    )


def test_table_levels(news):
    t = news_waterfall_table(news, old_nowcast=1.0)
    assert t["label"].tolist() == [
        "Previous nowcast",
        "ip",
        "spread",
        "pmi",
        "retail",
        "New nowcast",
    ]
    assert t["measure"].tolist()[0] == "absolute" and t["measure"].tolist()[-1] == "total"
    assert t["end"].iloc[-1] == pytest.approx(1.13)
    np.testing.assert_allclose(t["end"].iloc[1:5] - t["start"].iloc[1:5], t["value"].iloc[1:5])
    assert "actual" in t.columns


def test_table_residual_and_no_residual(news):
    t = news_waterfall_table(news, old_nowcast=1.0, new_nowcast=1.2)
    assert t.loc[t["label"] == "Other effects", "value"].iloc[0] == pytest.approx(0.07)
    t2 = news_waterfall_table(news, old_nowcast=1.0, new_nowcast=1.13)
    assert "Other effects" not in t2["label"].tolist()


def test_table_without_old(news):
    t = news_waterfall_table(news, new_nowcast=5.0)
    assert t["label"].iloc[-1] == "Total revision"
    assert t["value"].iloc[-1] == pytest.approx(0.13)


def test_table_group_and_top(news):
    t = news_waterfall_table(news, group_by="category")
    assert t["label"].tolist()[:3] == ["hard", "financial", "soft"]
    assert t["value"].iloc[0] == pytest.approx(0.12)
    t = news_waterfall_table(news, top_n=2)
    assert t["label"].tolist() == ["ip", "spread", "Other", "Total revision"]
    assert t["value"].iloc[2] == pytest.approx(-0.02)
    assert len(news_waterfall_table(news, top_n=10)) == 5


def test_table_series_input_and_index_labels():
    s = pd.Series({"a": 0.3, "b": -0.1})
    t = news_waterfall_table(s)
    assert t["label"].tolist() == ["a", "b", "Total revision"]
    df = pd.DataFrame({"impact": [0.1]}, index=["z"])
    assert news_waterfall_table(df)["label"].iloc[0] == "z"


class _NewsObject:
    """Mimics nowcastbox.news.NewsResults."""

    def __init__(self, releases):
        self.releases = releases
        self.old_nowcast = 1.0
        self.new_nowcast = 1.25
        self.target_period = pd.Period("2024Q2", freq="Q")
        self.revisions_effect = 0.1
        self.reestimation_effect = 0.0
        self.removal_effect = None


def test_news_object(news):
    obj = _NewsObject(news)
    t = news_waterfall_table(obj)
    labels = t["label"].tolist()
    assert labels[0] == "Previous nowcast" and "Data revisions" in labels
    assert "Re-estimation" not in labels
    assert t.loc[t["label"] == "Other effects", "value"].iloc[0] == pytest.approx(0.02)
    fig = plot_news_waterfall(obj)
    assert "2024Q2" in fig.layout.title.text


def test_news_object_impacts_and_to_frame(news):
    class Impacts:
        impacts = news

    class ToFrame:
        def to_frame(self):
            return news

    class Bad:
        def to_frame(self):
            return "nope"

    assert len(news_waterfall_table(Impacts())) == 5
    assert len(news_waterfall_table(ToFrame())) == 5
    with pytest.raises(TypeError):
        news_waterfall_table(Bad())
    with pytest.raises(TypeError):
        news_waterfall_table(3)


def test_table_errors(news):
    with pytest.raises(ValueError, match="'impact'"):
        news_waterfall_table(pd.DataFrame({"x": [1.0]}))
    with pytest.raises(ValueError, match="empty"):
        news_waterfall_table(pd.DataFrame({"impact": []}))
    with pytest.raises(ValueError, match="group_by"):
        news_waterfall_table(news, group_by="nope")
    with pytest.raises(ValueError, match="top_n"):
        news_waterfall_table(news, top_n=0)


@given(
    st.lists(st.floats(-5, 5, allow_nan=False), min_size=1, max_size=12),
    st.floats(-10, 10, allow_nan=False),
    st.one_of(st.none(), st.integers(1, 5)),
)
def test_table_identity(impacts, old, top_n):
    t = news_waterfall_table(pd.Series(impacts), old_nowcast=old, top_n=top_n)
    assert t["end"].iloc[-1] == pytest.approx(old + sum(impacts), abs=1e-9)
    rel = t[t["measure"] == "relative"]
    np.testing.assert_allclose(rel["end"] - rel["start"], rel["value"], atol=1e-12)


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_waterfall_both_backends(news, backend):
    assert plot_news_waterfall(news, old_nowcast=1.0, backend=backend) is not None
    assert plot_news_waterfall(news, group_by="category", backend=backend) is not None


def test_waterfall_plotly(news):
    fig = plot_news_waterfall(news, old_nowcast=1.0, ylabel="pp")
    wf = fig.data[0]
    assert wf.type == "waterfall"
    assert next(iter(wf.measure)) == "absolute"
    assert fig.layout.yaxis.range[0] > 0.5
    assert fig.layout.title.text == "News decomposition"
    fig2 = plot_news_waterfall(news)
    assert fig2.layout.yaxis.range is None


def test_waterfall_range_flat():
    fig = plot_news_waterfall(pd.Series({"a": 0.0}), old_nowcast=2.0, backend="matplotlib")
    assert fig.axes[0].get_ylim()[0] < 2.0


# ---------------------------------------------------------------------- tracker
@pytest.fixture
def wide():
    idx = pd.period_range("2024-04", periods=4, freq="M")
    return pd.DataFrame(
        {
            "hard": [0.1, 0.0, -0.1, 0.05],
            "soft": [0.05, 0.2, 0.0, np.nan],
            "nowcast": [1.0, 1.2, 1.1, 1.15],
        },
        index=idx,
    )


def test_tracker_table_wide(wide):
    contrib, level = tracker_table(wide)
    assert list(contrib.columns) == ["hard", "soft"]
    assert contrib.loc[contrib.index[-1], "soft"] == 0.0
    assert level is not None and level.iloc[-1] == 1.15
    contrib, level = tracker_table(wide.drop(columns="nowcast"))
    assert level is None


def test_tracker_table_long():
    long = pd.DataFrame(
        {
            "vintage": ["v1", "v1", "v2"],
            "category": ["hard", "soft", "hard"],
            "impact": [0.1, -0.2, 0.05],
        }
    )
    contrib, level = tracker_table(long)
    assert level is None
    assert contrib.loc["v1"].sum() == pytest.approx(-0.1)


def test_tracker_object(wide):
    class Tracker:
        contributions = wide[["hard", "soft"]]
        path = wide[["nowcast"]].assign(change=0.0)

    contrib, level = tracker_table(Tracker())
    assert list(contrib.columns) == ["hard", "soft"]
    assert level is not None


def test_tracker_errors(wide):
    with pytest.raises(TypeError):
        tracker_table([1, 2])
    with pytest.raises(ValueError, match="empty"):
        tracker_table(pd.DataFrame())
    with pytest.raises(ValueError, match="'category'"):
        tracker_table(pd.DataFrame({"vintage": [1], "impact": [0.1]}))
    with pytest.raises(ValueError, match="numeric"):
        tracker_table(pd.DataFrame({"nowcast": [1.0], "x": ["a"]}))


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_tracker_plot_both_backends(wide, backend):
    assert plot_nowcast_tracker(wide, actual=1.1, backend=backend) is not None
    assert plot_nowcast_tracker(wide.drop(columns="nowcast"), backend=backend) is not None


def test_tracker_plot_plotly(wide):
    fig = plot_nowcast_tracker(wide, actual=1.1, title="T")
    assert [t.name for t in fig.data] == ["hard", "soft", "Nowcast"]
    assert fig.layout.barmode == "relative"
    long = pd.DataFrame(
        {"vintage": ["a", "b"], "category": ["x", "x"], "impact": [0.1, 0.2], "nowcast": [1.0, 1.2]}
    )
    fig = plot_nowcast_tracker(long)
    assert list(fig.data[0].x) == ["a", "b"]

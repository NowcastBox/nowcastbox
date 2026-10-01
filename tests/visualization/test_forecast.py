import matplotlib.figure
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats

from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.visualization import (
    interval_levels,
    plot_fan_chart,
    plot_forecast,
    quantiles_from_nowcast,
)
from nowcastbox.visualization.forecast import _parse_level, _quantile_pairs, _runs


def _frame(with_intervals=True):
    idx = pd.period_range("2020Q1", periods=6, freq="Q")
    obs = pd.Series([1.0, 2.0, 1.5, 1.0, np.nan, np.nan], index=idx)
    est = pd.Series([1.1, 1.8, 1.6, 0.9, 1.2, 1.3], index=idx)
    extra = None
    if with_intervals:
        sd = pd.Series([np.nan] * 4 + [0.2, 0.4], index=idx)
        extra = {
            "std": sd,
            "lower_90": est - 1.645 * sd,
            "upper_90": est + 1.645 * sd,
            "lower_68": est - sd,
            "upper_68": est + sd,
        }
    return build_nowcast_frame(obs, est, extra=extra)


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_forecast_both_backends_frame(backend):
    fig = plot_forecast(_frame(), backend=backend, title="t", ylabel="y")
    expected = go.Figure if backend == "plotly" else matplotlib.figure.Figure
    assert isinstance(fig, expected)


def test_forecast_plotly_traces_and_intervals():
    fig = plot_forecast(_frame())
    names = [t.name for t in fig.data]
    assert names[-3:] == ["Observed", "In-sample", "Out-of-sample"]
    assert "90% interval" in names and "68% interval" in names
    assert fig.layout.title.text == "Nowcast"
    fig2 = plot_forecast(_frame(), show_intervals=False)
    assert [t.name for t in fig2.data] == ["Observed", "In-sample", "Out-of-sample"]


def test_forecast_isolated_interval_uses_error_bars():
    frame = _frame()
    frame.loc[frame.index[-1], ["lower_90", "upper_90", "lower_68", "upper_68"]] = np.nan
    fig = plot_forecast(frame)
    bars = [t for t in fig.data if t.error_y is not None and t.error_y.array is not None]
    assert bars
    half = bars[0].error_y.array[0]
    assert half == pytest.approx(1.645 * 0.2)


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_forecast_results(em, two_step, backend):
    for res in (em, two_step):
        fig = plot_forecast(res, backend=backend, start="2003Q1")
        assert fig is not None
    fig = plot_forecast(em)
    assert fig.layout.title.text == "Nowcast of gdp"


def test_forecast_matplotlib_ax():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    out = plot_forecast(_frame(), backend="matplotlib", ax=ax)
    assert out is fig
    assert ax.get_legend() is not None


def test_forecast_errors():
    with pytest.raises(ValueError, match="No periods"):
        plot_forecast(_frame(), start="2030Q1")
    with pytest.raises(ValueError, match="missing the columns"):
        plot_forecast(pd.DataFrame({"observed": [1.0]}))
    with pytest.raises(TypeError):
        plot_forecast([1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="backend"):
        plot_forecast(_frame(), backend="bokeh")


def test_interval_levels():
    cols = ["lower_68", "upper_68", "lower_90", "upper_90", "lower_50", "upper_x"]
    assert interval_levels(pd.DataFrame(columns=cols)) == ["90", "68"]


# ---------------------------------------------------------------------- quantiles
@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (0.05, 0.05),
        ("0.05", 0.05),
        ("q05", 0.05),
        ("q5", 0.05),
        ("q_95", 0.95),
        ("q0.25", 0.25),
        ("q2.5", 0.025),
        ("5%", 0.05),
        ("97.5%", 0.975),
        (np.float64(0.5), 0.5),
    ],
)
def test_parse_level(label, expected):
    assert _parse_level(label) == pytest.approx(expected)


@pytest.mark.parametrize("label", ["median", 1.5, "0", True])
def test_parse_level_errors(label):
    with pytest.raises(ValueError):
        _parse_level(label)


@given(st.floats(min_value=0.001, max_value=0.999))
def test_parse_level_roundtrip(q):
    assert _parse_level(q) == q
    assert _parse_level(f"{100 * q:.6f}%") == pytest.approx(q, abs=1e-8)


def test_quantile_pairs():
    assert _quantile_pairs([0.05, 0.25, 0.5, 0.75, 0.95]) == [(0.05, 0.95), (0.25, 0.75)]
    with pytest.raises(ValueError, match="symmetric counterpart"):
        _quantile_pairs([0.1, 0.5, 0.8])
    with pytest.raises(ValueError, match="at least one"):
        _quantile_pairs([0.5, 0.7])


def test_runs():
    assert _runs(np.array([True, True, False, True, False])) == [(0, 2), (3, 4)]
    assert _runs(np.array([False, True])) == [(1, 2)]
    assert _runs(np.array([], dtype=bool)) == []


def test_quantiles_from_nowcast_values():
    q = quantiles_from_nowcast(_frame(), [0.05, 0.5, 0.95])
    assert list(q.columns) == [0.05, 0.5, 0.95]
    assert len(q) == 2
    np.testing.assert_allclose(q[0.5], [1.2, 1.3])
    np.testing.assert_allclose(q[0.95], [1.2, 1.3] + np.array([0.2, 0.4]) * stats.norm.ppf(0.95))


def test_quantiles_from_nowcast_errors():
    with pytest.raises(ValueError, match="no 'std'"):
        quantiles_from_nowcast(_frame(with_intervals=False))
    frame = _frame()
    frame["std"] = np.nan
    with pytest.raises(ValueError, match="No out-of-sample"):
        quantiles_from_nowcast(frame)


def _quantiles():
    idx = pd.period_range("2024Q3", periods=3, freq="Q")
    return pd.DataFrame(
        {
            "q05": [0.0, -0.5, -1.0],
            "q25": [0.5, 0.3, 0.2],
            "q50": [1.0, 1.0, 1.0],
            "q75": [1.5, 1.7, 1.8],
            "q95": [2.0, 2.5, 3.0],
        },
        index=idx,
    )


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_fan_chart_both_backends(backend):
    obs = pd.Series([0.5, 0.8], index=pd.period_range("2024Q1", periods=2, freq="Q"))
    fig = plot_fan_chart(_quantiles(), observed=obs, backend=backend, title="F")
    assert fig is not None


def test_fan_chart_plotly_content():
    fig = plot_fan_chart(_quantiles())
    names = [t.name for t in fig.data]
    assert "90% band" in names and "50% band" in names and names[-1] == "Median"


def test_fan_chart_without_median_and_history():
    q = _quantiles().drop(columns="q50")
    obs = pd.Series(np.arange(20.0), index=pd.period_range("2019Q1", periods=20, freq="Q"))
    fig = plot_fan_chart(q, observed=obs, history=4)
    observed = next(t for t in fig.data if t.name == "Observed")
    assert len(observed.y) == 4
    fig0 = plot_fan_chart(q, observed=obs, history=0)
    assert all(t.name != "Observed" for t in fig0.data)
    fig_all = plot_fan_chart(q, observed=obs, history=None, backend="matplotlib")
    assert fig_all is not None


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_fan_chart_from_results(em, backend):
    fig = plot_fan_chart(em, backend=backend)
    assert fig is not None


def test_fan_chart_from_results_keeps_post_sample_periods(em):
    fig = plot_fan_chart(em)
    last = em.observed.last_valid_index()
    median = next(t for t in fig.data if t.name == "Median")
    first_x = pd.Timestamp(median.x[0])
    assert first_x > last.to_timestamp(how="end")


def test_fan_chart_errors():
    with pytest.raises(TypeError):
        plot_fan_chart([1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        plot_fan_chart(pd.DataFrame())
    dup = pd.DataFrame([[0.0, 0.0, 1.0]], columns=["q05", "0.05", "q95"])
    with pytest.raises(ValueError, match="Duplicate"):
        plot_fan_chart(dup)


def test_fan_from_results_without_std():
    idx = pd.period_range("2020Q1", periods=2, freq="Q")
    f = build_nowcast_frame(pd.Series([1.0, np.nan], index=idx), pd.Series([1.0, 2.0], index=idx))
    with pytest.raises(ValueError, match="std"):
        plot_fan_chart(NowcastResults(target="y", nowcast=f))


# ---------------------------------------------------------------------- distributions
def test_fan_chart_from_distribution():
    """``NowcastDistribution`` input (I5): exact quantiles, trimmed after ``observed``."""
    from nowcastbox.density import NowcastDistribution

    idx = pd.period_range("2020Q1", periods=3, freq="Q")
    dist = NowcastDistribution(idx, [1.0, 1.5, 2.0], [0.5, 0.6, 0.7])
    fig = plot_fan_chart(dist)  # no history: every period
    median = next(t for t in fig.data if t.name == "Median")
    assert len(median.x) == 3
    observed = pd.Series([0.9], index=idx[:1])
    fig = plot_fan_chart(dist, observed=observed)
    median = next(t for t in fig.data if t.name == "Median")
    assert len(median.x) == 2
    late = pd.Series([1.0, 1.0, 1.0], index=idx)  # observed everywhere: nothing trimmed
    fig = plot_fan_chart(dist, observed=late, backend="matplotlib")
    assert isinstance(fig, matplotlib.figure.Figure)

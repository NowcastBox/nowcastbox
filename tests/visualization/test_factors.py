import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.selection import select_factors
from nowcastbox.visualization import (
    panel_eigenvalues,
    plot_eigenvalues,
    plot_factors,
    plot_loadings,
)


def _plain_results(**kwargs):
    idx = pd.period_range("2020Q1", periods=2, freq="Q")
    f = build_nowcast_frame(pd.Series([1.0, np.nan], index=idx), pd.Series([1.0, 2.0], index=idx))
    return NowcastResults(target="y", nowcast=f, **kwargs)


# ---------------------------------------------------------------------- factors
@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_factors_results(two_step, em, backend):
    for res in (two_step, em):
        fig = plot_factors(res, backend=backend)
        expected = go.Figure if backend == "plotly" else matplotlib.figure.Figure
        assert isinstance(fig, expected)


def test_factors_traces_and_subset(two_step):
    fig = plot_factors(two_step)
    assert [t.name for t in fig.data] == ["f1", "f2"]
    fig = plot_factors(two_step, factors=["f2"], title="only f2")
    assert [t.name for t in fig.data] == ["f2"]
    assert fig.layout.title.text == "only f2"


def test_factors_forecast_shading(two_step):
    assert two_step.factors.index[-1] > two_step.data.end
    fig = plot_factors(two_step)
    assert any(s.type == "rect" for s in fig.layout.shapes)
    fig_no = plot_factors(two_step, shade_forecast=False)
    assert all(s.type != "rect" for s in fig_no.layout.shapes)
    assert plot_factors(two_step, backend="matplotlib") is not None


def test_factors_frame_and_ax():
    idx = pd.period_range("2020-01", periods=12, freq="M")
    frame = pd.DataFrame({"f1": np.arange(12.0)}, index=idx)
    fig, ax = plt.subplots()
    assert plot_factors(frame, backend="matplotlib", ax=ax) is fig


def test_factors_errors():
    with pytest.raises(ValueError, match="no factors"):
        plot_factors(_plain_results())
    with pytest.raises(TypeError):
        plot_factors("x")  # type: ignore[arg-type]
    idx = pd.period_range("2020-01", periods=3, freq="M")
    with pytest.raises(ValueError, match="Unknown factors"):
        plot_factors(pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=idx), factors=["f9"])
    with pytest.raises(ValueError, match="No factors"):
        plot_factors(pd.DataFrame(index=idx))


# ---------------------------------------------------------------------- eigenvalues
def test_panel_eigenvalues_analytical(rng):
    x = rng.normal(size=(200, 5))
    ev = panel_eigenvalues(pd.DataFrame(x))
    expected = np.sort(np.linalg.eigvalsh(np.corrcoef(x, rowvar=False)))[::-1]
    np.testing.assert_allclose(ev.to_numpy(), expected, atol=1e-12)
    assert ev.sum() == pytest.approx(5.0)
    assert list(ev.index) == [1, 2, 3, 4, 5]


def test_panel_eigenvalues_mixed_frequency(panel):
    ev = panel_eigenvalues(panel)
    assert len(ev) == 6  # monthly series only
    assert (np.diff(ev.to_numpy()) <= 1e-12).all()


def test_panel_eigenvalues_errors():
    with pytest.raises(ValueError, match="two base-frequency"):
        panel_eigenvalues(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    with pytest.raises(ValueError, match="constant"):
        panel_eigenvalues(pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 4.0]}))


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_eigenvalues_from_results(two_step, em, backend):
    assert plot_eigenvalues(two_step, backend=backend) is not None
    assert plot_eigenvalues(em, backend=backend) is not None


def test_eigenvalues_highlight_and_values(two_step):
    fig = plot_eigenvalues(two_step)
    bar = fig.data[0]
    np.testing.assert_allclose(bar.y, two_step.eigenvalues.to_numpy()[: len(bar.y)])
    colors = list(bar.marker.color)
    assert colors[0] == colors[1] != colors[2]


def test_eigenvalues_other_inputs(rng):
    sel = select_factors(rng.normal(size=(60, 8)), rmax=3)
    fig = plot_eigenvalues(sel)
    assert len(fig.data[0].x) == 8
    fig = plot_eigenvalues(pd.Series([3.0, 1.0, 0.5]), n_factors=2, max_components=2)
    assert list(fig.data[0].x) == [1, 2]
    fig = plot_eigenvalues(np.array([2.0, 1.0]), max_components=None, title="S")
    assert fig.layout.title.text == "S"
    fig = plot_eigenvalues([0.0, 0.0])
    assert list(fig.data[0].customdata[:, 0]) == [0.0, 0.0]


def test_eigenvalues_errors():
    with pytest.raises(ValueError, match="neither"):
        plot_eigenvalues(_plain_results())
    with pytest.raises(TypeError):
        plot_eigenvalues("x")
    with pytest.raises(ValueError, match="No eigenvalues"):
        plot_eigenvalues([])


# ---------------------------------------------------------------------- loadings
@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
@pytest.mark.parametrize("style", ["heatmap", "bar"])
def test_loadings_styles(two_step, em, backend, style):
    for res in (two_step, em):
        assert plot_loadings(res, style=style, backend=backend) is not None


def test_loadings_heatmap_symmetric_scale(two_step):
    fig = plot_loadings(two_step)
    hm = fig.data[0]
    assert hm.zmin == -hm.zmax
    assert hm.zmax == pytest.approx(np.abs(two_step.loadings.to_numpy()).max())


def test_loadings_bar_sorted_and_subset(two_step):
    fig = plot_loadings(two_step, style="bar", factors=["f1"], sort_by="f1")
    values = np.asarray(fig.data[0].x)
    assert (np.diff(values) >= 0).all()
    assert len(fig.data) == 1


def test_loadings_matplotlib_ax():
    lam = pd.DataFrame({"f1": [0.5, -0.2]}, index=["a", "b"])
    fig, ax = plt.subplots()
    assert plot_loadings(lam, style="bar", backend="matplotlib", ax=ax) is fig
    fig2, ax2 = plt.subplots()
    assert plot_loadings(lam, backend="matplotlib", ax=ax2) is fig2
    zero = pd.DataFrame({"f1": [0.0, 0.0]}, index=["a", "b"])
    assert plot_loadings(zero).data[0].zmax == 1.0


def test_loadings_errors(two_step):
    with pytest.raises(ValueError, match="style"):
        plot_loadings(two_step, style="pie")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no loadings"):
        plot_loadings(_plain_results())
    with pytest.raises(TypeError):
        plot_loadings(3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sort_by"):
        plot_loadings(two_step, sort_by="f9")
    with pytest.raises(ValueError, match="Unknown factors"):
        plot_loadings(two_step, factors=["f9"])
    with pytest.raises(ValueError, match="No loadings"):
        plot_loadings(pd.DataFrame())
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="single factor"):
        plot_loadings(two_step, style="bar", backend="matplotlib", ax=ax)

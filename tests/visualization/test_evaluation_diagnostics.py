import numpy as np
import pandas as pd
import pytest

from nowcastbox.visualization import plot_loglikelihood, plot_rmsfe_by_horizon, rmsfe_frame


@pytest.fixture
def table():
    return pd.DataFrame(
        {"DFM": [0.5, 0.8, 1.0], "AR": [1.0, 1.0, 1.25]}, index=pd.Index([-1, 0, 1], name="horizon")
    )


def test_rmsfe_frame(table):
    rel = rmsfe_frame(table, relative_to="AR")
    np.testing.assert_allclose(rel["DFM"], [0.5, 0.8, 0.8])
    np.testing.assert_allclose(rel["AR"], 1.0)
    assert list(rmsfe_frame(table["DFM"]).columns) == ["DFM"]
    assert list(rmsfe_frame(pd.Series([1.0, 2.0])).columns) == ["RMSFE"]

    class Backtest:
        def rmsfe_by_horizon(self):
            return table

    assert rmsfe_frame(Backtest()).equals(table)


def test_rmsfe_frame_errors(table):
    with pytest.raises(TypeError):
        rmsfe_frame([1, 2])
    with pytest.raises(ValueError, match="empty"):
        rmsfe_frame(pd.DataFrame())
    with pytest.raises(ValueError, match="relative_to"):
        rmsfe_frame(table, relative_to="X")
    with pytest.raises(ValueError, match="positive"):
        rmsfe_frame(table.assign(AR=0.0), relative_to="AR")


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
@pytest.mark.parametrize("style", ["line", "bar"])
def test_rmsfe_plot(table, backend, style):
    assert plot_rmsfe_by_horizon(table, style=style, backend=backend) is not None
    assert plot_rmsfe_by_horizon(table, style=style, backend=backend, relative_to="AR") is not None


def test_rmsfe_plot_content(table):
    fig = plot_rmsfe_by_horizon(table, relative_to="AR")
    assert fig.layout.title.text == "Relative RMSFE (vs AR)"
    assert [t.name for t in fig.data] == ["DFM", "AR"]
    assert plot_rmsfe_by_horizon(table, style="bar").data[0].type == "bar"
    with pytest.raises(ValueError, match="style"):
        plot_rmsfe_by_horizon(table, style="pie")  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_loglikelihood(em, backend):
    assert plot_loglikelihood(em, backend=backend) is not None
    fig = plot_loglikelihood(em)
    np.testing.assert_allclose(fig.data[0].y, em.loglikelihood_path)


def test_loglikelihood_errors(two_step):
    class NoPath:
        loglikelihood_path = None

    with pytest.raises(ValueError, match="no log-likelihood"):
        plot_loglikelihood(NoPath())
    with pytest.raises(ValueError, match="empty"):
        plot_loglikelihood([])

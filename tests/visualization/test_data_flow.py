import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning
from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.visualization import (
    AVAILABILITY_STATES,
    data_availability,
    plot_data_availability,
    release_table,
)


@pytest.fixture
def small():
    idx = pd.period_range("2020-01", periods=6, freq="M")
    df = pd.DataFrame(
        {
            "x": [1.0, np.nan, 1.0, 1.0, np.nan, np.nan],
            "q": [np.nan, np.nan, 2.0, np.nan, np.nan, np.nan],
            "e": [np.nan] * 6,
        },
        index=idx,
    )
    with pytest.warns(DataQualityWarning, match="without any observation"):
        return MixedFrequencyData(
            df,
            {"x": "M", "q": "Q", "e": "M"},
            release_delays={"x": 30, "q": 60, "e": 5},
            categories={"x": "hard"},
        )


def test_codes(small):
    codes = data_availability(small)
    assert codes.loc["x"].tolist() == [1, 2, 1, 1, 3, 3]
    assert codes.loc["q"].tolist() == [0, 0, 1, 0, 0, 3]
    assert codes.loc["e"].tolist() == [3] * 6
    assert AVAILABILITY_STATES[2] == "missing"


def test_codes_options(small):
    codes = data_availability(small, n_periods=2, series=["q"])
    assert codes.shape == (1, 2)
    assert data_availability(small, n_periods=None).shape == (3, 6)


def test_codes_errors(small):
    with pytest.raises(ValueError, match="Unknown series"):
        data_availability(small, series=["zz"])
    with pytest.raises(ValueError, match="n_periods"):
        data_availability(small, n_periods=0)
    with pytest.raises(TypeError):
        data_availability(pd.DataFrame())  # type: ignore[arg-type]
    idx = pd.period_range("2020Q1", periods=1, freq="Q")
    f = build_nowcast_frame(pd.Series([1.0], index=idx), pd.Series([1.0], index=idx))
    with pytest.raises(ValueError, match="estimation data"):
        data_availability(NowcastResults(target="y", nowcast=f))


def test_release_table(small, em):
    table = release_table(small)
    assert table.loc["x", "last_observed"] == "2020-04"
    assert table.loc["q", "last_observed"] == "2020Q1"
    assert table.loc["e", "last_observed"] == "-"
    assert table.loc["x", "pending"] == 2
    assert table.loc["x", "category"] == "hard"
    assert table.loc["q", "category"] == "-"
    assert table.loc["q", "release_delay"] == 60
    assert list(release_table(em).index) == em.data.columns


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_plot_both_backends(small, em, backend):
    assert plot_data_availability(small, backend=backend) is not None
    assert plot_data_availability(em, backend=backend, n_periods=12, title="T") is not None


def test_plot_plotly_content(small):
    fig = plot_data_availability(small)
    hm = fig.data[0]
    assert hm.text[0][4] == "Not yet released"
    assert list(fig.layout.coloraxis.colorbar.ticktext or hm.colorbar.ticktext) == [
        "No release slot",
        "Observed",
        "Missing",
        "Not yet released",
    ]

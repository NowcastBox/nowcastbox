import pandas as pd
import pytest

from nowcastbox.selection import select_factors, select_shocks
from nowcastbox.visualization import plot_factor_selection


@pytest.fixture
def x(rng):
    f = rng.normal(size=(120, 2))
    lam = rng.normal(size=(2, 12))
    return f @ lam + 0.5 * rng.normal(size=(120, 12))


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_factor_selection_result(x, backend):
    sel = select_factors(x, rmax=6)
    assert plot_factor_selection(sel, backend=backend) is not None


def test_markers_at_minimum(x):
    sel = select_factors(x, rmax=6)
    fig = plot_factor_selection(sel)
    names = [t.name for t in fig.data]
    assert names[:3] == ["IC1", "IC2", "IC3"]
    star = next(t for t in fig.data if t.name == "IC2 min")
    assert star.x[0] == sel.criteria["IC2"].idxmin()


def test_dataframe_and_normalize():
    ic = pd.DataFrame({"PC1": [10.0, 5.0, 6.0], "IC1": [0.0, -0.4, -0.3]})
    fig = plot_factor_selection(ic, criteria=["PC1", "IC1"], normalize=True)
    pc = fig.data[0]
    assert min(pc.y) == 0.0 and max(pc.y) == 1.0
    flat = pd.DataFrame({"PC1": [1.0, 1.0]})
    assert plot_factor_selection(flat, normalize=True) is not None
    assert [t.name for t in plot_factor_selection(flat).data] == ["PC1", "PC1 min"]


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_shock_selection(x, backend):
    sel = select_shocks(x, n_factors=2, factor_lags=1)
    fig = plot_factor_selection(sel, backend=backend)
    assert fig is not None


def test_shock_bound_selection():
    class Shock:
        statistics = pd.DataFrame({"D1": [0.9, 0.4, 0.1]}, index=[1, 2, 3])
        bound = 0.5
        statistic = "D1"

    fig = plot_factor_selection(Shock())
    star = next(t for t in fig.data if t.name == "D1 min")
    assert star.x[0] == 2

    class Never(Shock):
        bound = 0.0

    assert next(t for t in plot_factor_selection(Never()).data if t.name == "D1 min").x[0] == 3


def test_errors():
    with pytest.raises(TypeError):
        plot_factor_selection([1, 2])
    with pytest.raises(ValueError, match="empty"):
        plot_factor_selection(pd.DataFrame())
    with pytest.raises(ValueError, match="Unknown criteria"):
        plot_factor_selection(pd.DataFrame({"IC1": [1.0]}), criteria=["IC9"])

import pytest

from nowcastbox.visualization import themes
from nowcastbox.visualization._common import check_backend, rgba, to_plot_index
from nowcastbox.visualization.themes import (
    Theme,
    get_theme,
    list_themes,
    register_theme,
    set_default_theme,
)


def test_builtin_themes():
    assert {"default", "academic", "presentation"} <= set(list_themes())
    assert get_theme().name == "default"
    assert get_theme(get_theme("academic")).name == "academic"


def test_get_theme_errors():
    with pytest.raises(ValueError, match="Unknown theme"):
        get_theme("nope")
    with pytest.raises(TypeError):
        get_theme(3)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": ""},
        {"name": "x", "colors": ()},
        {"name": "x", "font_size": 0},
        {"name": "x", "line_width": 0},
    ],
)
def test_theme_validation(kwargs):
    with pytest.raises(ValueError):
        Theme(**kwargs)


def test_color_wraps_and_replace():
    th = Theme(name="t", colors=("#000000", "#ffffff"))
    assert th.color(2) == "#000000"
    assert th.replace(font_size=20).font_size == 20


def test_register_theme():
    th = Theme(name="test_register_theme_x")
    register_theme(th)
    assert get_theme("test_register_theme_x") is th
    with pytest.raises(ValueError, match="already exists"):
        register_theme(th)
    register_theme(th.replace(font_size=9), overwrite=True)
    assert get_theme("test_register_theme_x").font_size == 9
    with pytest.raises(TypeError):
        register_theme("x")  # type: ignore[arg-type]


def test_set_default_theme_registers_new():
    th = Theme(name="test_default_new")
    previous = set_default_theme(th)
    try:
        assert get_theme().name == "test_default_new"
        assert "test_default_new" in list_themes()
    finally:
        set_default_theme(previous)
    assert get_theme().name == previous.name


def test_matplotlib_rc_fonts():
    rc = get_theme("default").matplotlib_rc()
    assert rc["font.family"] == "sans-serif"
    assert rc["font.sans-serif"][0] == "Inter"
    assert get_theme("academic").matplotlib_rc()["font.family"] == "serif"


def test_plotly_layout_width():
    assert "width" not in get_theme().plotly_layout()
    assert get_theme().replace(width=800).plotly_layout()["width"] == 800
    assert get_theme().plotly_axis()["gridcolor"] == get_theme().grid_color


def test_default_palette_is_fixed_order():
    assert themes._CATEGORICAL[0] == "#2a78d6"
    assert len(set(themes._CATEGORICAL)) == len(themes._CATEGORICAL)


def test_common_helpers():
    with pytest.raises(ValueError, match="backend"):
        check_backend("bokeh")
    with pytest.raises(ValueError, match="'ax'"):
        check_backend("plotly", ax=object())
    assert check_backend("matplotlib", ax=object()) == "matplotlib"
    assert rgba("#ff0080", 0.5) == "rgba(255,0,128,0.5)"
    with pytest.raises(ValueError):
        rgba("red", 0.5)
    import pandas as pd

    idx = pd.Index([1, 2])
    assert to_plot_index(idx) is idx
    q = to_plot_index(pd.period_range("2020Q1", periods=1, freq="Q"))
    assert str(q[0].date()) == "2020-03-31"

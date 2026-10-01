"""Tests of the example helpers (``examples/utils``) and of the notebook builder.

Run with ``python3 -m pytest examples/tests`` (the notebooks themselves are checked with
``pytest --nbmake examples/notebooks``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

EXAMPLES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXAMPLES))

import utils
from utils import build_notebooks


def test_pct_formats_and_handles_nan() -> None:
    assert utils.pct(0.0123) == "1.23%"
    assert utils.pct(-0.004, 1) == "-0.4%"
    assert utils.pct(float("nan")) == "n/a"


def test_network_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOWCASTBOX_EXAMPLES_NETWORK", raising=False)
    assert not utils.network_enabled()
    monkeypatch.setenv("NOWCASTBOX_EXAMPLES_NETWORK", "1")
    assert utils.network_enabled()


def test_display_and_md_print_outside_notebooks(capsys: pytest.CaptureFixture[str]) -> None:
    assert not utils.in_notebook()
    utils.display("a", 1)
    utils.md("**b**")
    assert capsys.readouterr().out.split() == ["a", "1", "**b**"]


def test_show_saves_figures_and_accepts_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(utils, "OUTPUTS_DIR", tmp_path)
    fig, ax = plt.subplots()
    utils.show(ax, "axes_case")  # an Axes is resolved to its figure
    utils.show(fig)  # default name from the counter
    files = sorted(p.name for p in (tmp_path / "figures").iterdir())
    assert "axes_case.png" in files
    assert len(files) == 2


def test_setup_is_deterministic() -> None:
    utils.setup(blas_threads=None)
    a = np.random.rand(3)  # noqa: NPY002 - checks the legacy global seed
    utils.setup(blas_threads=None)
    assert np.allclose(a, np.random.rand(3))  # noqa: NPY002


def test_brazil_panel_and_core_columns() -> None:
    cols = utils.brazil_core_columns()
    assert cols[0] == "pib"
    assert len(cols) == len(set(cols)) == 22
    ds, panel = utils.brazil_panel(["pib", "ibc_br", "pim_geral"], start="2015-01")
    assert panel.columns == ["pib", "ibc_br", "pim_geral"]
    assert panel.metadata["ibc_br"].transform_applied
    _, levels = utils.brazil_panel(["pib", "ibc_br"], start="2015-01", prepare=False)
    assert not levels.metadata["ibc_br"].transform_applied
    assert ds.target == "pib"


def test_gdp_growth_vintages_match_the_index() -> None:
    rec = utils.gdp_growth_vintages()
    assert list(rec.columns) == ["series", "reference_period", "vintage_date", "value"]
    assert rec["value"].notna().all()
    # growth of the last vintage equals the pct change of that vintage's index
    import nowcastbox as nb

    matrix = nb.load_brazil_vintages().vintage_matrix("pib")
    last = matrix.columns[-1]
    expected = matrix[last].pct_change(fill_method=None).dropna()
    got = rec[rec["vintage_date"] == last].set_index("reference_period")["value"]
    pd.testing.assert_series_equal(got.loc[expected.index], expected, check_names=False)


def test_realtime_store_keeps_metadata() -> None:
    _, panel = utils.brazil_panel(["pib", "ibc_br", "selic"], start="2010-01")
    store = utils.brazil_realtime_store(panel)
    assert sorted(store.series) == ["ibc_br", "pib", "selic"]
    meta = utils.store_metadata(panel)
    data = store.as_of("2024-08-15", as_mixed=True, **meta)
    assert data.block_names == panel.block_names
    assert data.release_delays["ibc_br"] == panel.release_delays["ibc_br"]
    # on 15 Aug 2024 the last published GDP quarter was 2024Q1
    assert data.to_native("pib").dropna().index[-1] == pd.Period("2024Q1", "Q")


def test_parse_sidra_code() -> None:
    parsed = utils._parse_sidra_code("t8882/v7170/c11046=56734/c85=90672")
    assert parsed == {
        "table": 8882,
        "variable": 7170,
        "classifications": {"c11046": 56734, "c85": 90672},
    }


def test_fetch_live_panel_falls_back_without_connector(monkeypatch: pytest.MonkeyPatch) -> None:
    import nowcastbox as nb

    def boom(*args: object, **kwargs: object) -> None:
        raise ConnectionError("offline")

    monkeypatch.setattr(utils, "_fetch_one", boom)
    ds = nb.load_brazil_nowcast().select(["pib", "ibc_br", "focus_pib"])
    panel, status = utils.fetch_live_panel(ds)
    assert (status["origin"] == "shipped").all()
    assert status.loc["ibc_br", "fallback reason"].startswith("ConnectionError")
    assert panel.equals(ds.data)


def test_fetch_one_rejects_unknown_sources() -> None:
    with pytest.raises(NotImplementedError, match="no connector"):
        utils._fetch_one("x", "BCB/Focus (Olinda)", "code", "2020-01-01", 1.0)


def test_script_to_cells_splits_markdown_and_code() -> None:
    text = (
        '"""docstring dropped"""\n'
        "# %% [markdown]\n# # Title\n#\n# text\n\n"
        "# %%\nx = 1\n\n\n"
        "# %%\n\n"  # empty cell dropped
        "# %% [markdown]\n# more\n"
    )
    cells = build_notebooks.script_to_cells(text)
    assert [c.cell_type for c in cells] == ["markdown", "code", "markdown"]
    assert cells[0].source == "# Title\n\ntext"
    assert cells[1].source == "x = 1"


def test_build_without_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_notebooks, "NOTEBOOKS", tmp_path)
    script = tmp_path / "99_demo.py"
    script.write_text("# %% [markdown]\n# # Demo\n# %%\nprint(1)\n", encoding="utf-8")
    path, seconds = build_notebooks.build(script, execute=False)
    nb = nbformat.read(path, as_version=4)
    assert seconds == 0.0
    assert len(nb.cells) == 2
    assert nb.metadata["kernelspec"]["name"] == "python3"


def test_every_script_has_a_notebook() -> None:
    scripts = sorted(p.stem for p in (EXAMPLES / "scripts").glob("[0-9][0-9]_*.py"))
    notebooks = sorted(p.stem for p in (EXAMPLES / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    assert len(scripts) == 15
    assert scripts == notebooks


@pytest.mark.parametrize(
    "path", sorted((EXAMPLES / "notebooks").glob("[0-9][0-9]_*.ipynb")), ids=lambda p: p.stem
)
def test_notebooks_are_saved_with_outputs(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert code
    assert all(c.get("execution_count") for c in code)
    errors = [o for c in code for o in c.outputs if o.output_type == "error"]
    assert not errors
    assert any(
        "image/png" in o.get("data", {}) for c in code for o in c.outputs
    ) or path.stem.startswith(("11", "15"))


def test_notebook_branches(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(utils, "in_notebook", lambda: True)
    shown: list[object] = []

    class FakePlotly:
        __module__ = "plotly.graph_objs._figure"

        def show(self) -> None:
            shown.append(self)

    utils.show(FakePlotly())
    fig, _ = plt.subplots()
    utils.show(fig)  # rendered with IPython.display (text repr outside a kernel)
    utils.display(pd.DataFrame({"a": [1]}))
    utils.md("*x*")
    assert len(shown) == 1
    assert not plt.fignum_exists(fig.number)
    assert capsys.readouterr().out  # IPython's display falls back to printing


def test_show_writes_plotly_html_in_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(utils, "OUTPUTS_DIR", tmp_path)

    class FakePlotly:
        __module__ = "plotly.graph_objs._figure"

        def write_html(self, path: Path, include_plotlyjs: str) -> None:
            Path(path).write_text(include_plotlyjs, encoding="utf-8")

    utils.show(FakePlotly(), "p")
    assert (tmp_path / "figures" / "p.html").read_text(encoding="utf-8") == "cdn"


def test_fetch_one_dispatches_to_the_connectors(monkeypatch: pytest.MonkeyPatch) -> None:
    from nowcastbox import data_sources

    idx = pd.period_range("2020-01", periods=2, freq="M")
    calls: dict[str, object] = {}

    def sgs(codes: dict, **kw: object) -> pd.DataFrame:
        calls["sgs"] = codes
        return pd.DataFrame({k: [1.0, 2.0] for k in codes}, index=idx)

    def sidra(**kw: object) -> pd.DataFrame:
        calls["sidra"] = kw["table"]
        return pd.DataFrame({"whatever": [3.0, 4.0]}, index=idx)

    def ipea(codes: dict, **kw: object) -> pd.DataFrame:
        calls["ipea"] = codes
        return pd.DataFrame({k: [5.0, 6.0] for k in codes}, index=idx)

    monkeypatch.setattr(data_sources, "fetch_sgs", sgs)
    monkeypatch.setattr(data_sources, "fetch_sidra", sidra)
    monkeypatch.setattr(data_sources, "fetch_ipeadata", ipea)
    assert utils._fetch_one("a", "BCB/SGS", "123", "2020", 1.0).tolist() == [1.0, 2.0]
    s = utils._fetch_one("b", "IBGE/SIDRA", "t1/v2/c3=4", "2020", 1.0)
    assert s.name == "b"
    assert s.tolist() == [3.0, 4.0]
    assert utils._fetch_one("c", "IPEADATA", "X", "2020", 1.0).tolist() == [5.0, 6.0]
    assert calls == {"sgs": {"a": 123}, "sidra": 1, "ipea": {"c": "X"}}


def test_fetch_live_panel_extends_the_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    import nowcastbox as nb

    ds = nb.load_brazil_nowcast().select(["pib", "selic"])
    shipped = ds.data.data

    def fake(name: str, source: str, code: str, start: str, timeout: float) -> pd.Series:
        s = shipped[name].dropna()
        if name == "selic":  # one more month than the shipped data
            s = pd.concat([s, pd.Series([0.0], index=[shipped.index[-1] + 1])])
        return s

    monkeypatch.setattr(utils, "_fetch_one", fake)
    panel, status = utils.fetch_live_panel(ds)
    assert (status["origin"] == "live").all()
    assert panel.end == shipped.index[-1] + 1


def test_build_executes_and_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_notebooks, "NOTEBOOKS", tmp_path / "nb")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "98_exec.py").write_text("# %%\nx = 6 * 7\nx\n", encoding="utf-8")
    (scripts / "97_other.py").write_text("# %%\nx = 1\n", encoding="utf-8")
    monkeypatch.setattr(build_notebooks, "SCRIPTS", scripts)
    monkeypatch.setattr(build_notebooks, "EXAMPLES", tmp_path)
    path, seconds = build_notebooks.build(scripts / "98_exec.py", execute=True, timeout=120)
    nb = nbformat.read(path, as_version=4)
    assert seconds > 0
    assert nb.cells[0].outputs[0]["data"]["text/plain"] == "42"
    assert build_notebooks.main(["97", "--no-execute"]) == 0
    assert (tmp_path / "nb" / "97_other.ipynb").exists()

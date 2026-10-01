"""Offline tests of the pure helpers of ``scripts/build_datasets`` (no network)."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "build_datasets"


@pytest.fixture(scope="module")
def scripts() -> Iterator[dict[str, ModuleType]]:
    sys.path.insert(0, str(SCRIPTS))
    try:
        import _common
        import brazil_spec
        import build_brazil_nowcast
        import build_brazil_vintages
        import build_nyfed
        import build_us_fred_md

        yield {
            "common": _common,
            "spec": brazil_spec,
            "brazil": build_brazil_nowcast,
            "vintages": build_brazil_vintages,
            "nyfed": build_nyfed,
            "fred": build_us_fred_md,
        }
    finally:
        sys.path.remove(str(SCRIPTS))


def test_legacy_code(scripts: dict[str, ModuleType]) -> None:
    code = scripts["common"].legacy_code
    assert code("level") == "0"
    assert code(None) == "0"
    assert code("diff") == "2"
    assert code("yoy") == "6"
    assert code("qoq") == "7"
    assert code("dlog") == ""
    assert code("pct_change(1)|scale(100)") == ""


def test_wide_table_fills_gaps(scripts: dict[str, ModuleType]) -> None:
    idx = pd.PeriodIndex(["2020-01", "2020-03"], freq="M")
    table = scripts["common"].wide_table(pd.DataFrame({"x": [1.0, 2.0]}, index=idx))
    assert table["period"].tolist() == ["2020-01", "2020-02", "2020-03"]
    assert np.isnan(table["x"].iloc[1])
    with pytest.raises(ValueError, match="monthly"):
        scripts["common"].wide_table(pd.DataFrame({"x": [1.0]}))


def test_brazil_spec_is_consistent(scripts: dict[str, ModuleType]) -> None:
    from nowcastbox.preprocessing.transforms import get_transform

    series = scripts["spec"].SERIES
    names = [s["name"] for s in series]
    assert len(names) == len(set(names))
    for s in series:
        get_transform(s["transform"])
        assert s["frequency"] in {"M", "Q"}
        assert s["blocks"].startswith("global")
        assert s["delay_days"] >= 0


def test_to_monthly(scripts: dict[str, ModuleType]) -> None:
    to_monthly = scripts["brazil"].to_monthly
    q = pd.Series([1.0, 2.0], index=pd.PeriodIndex(["2002Q4", "2003Q1"], freq="Q"))
    out = to_monthly(q, "Q")
    assert [str(p) for p in out.index] == ["2003-03"]
    m = pd.Series([1.0], index=pd.PeriodIndex(["2003-02"], freq="M"))
    assert to_monthly(m, "M").tolist() == [1.0]
    with pytest.raises(ValueError, match="quarterly"):
        to_monthly(m, "Q")
    with pytest.raises(ValueError, match="monthly"):
        to_monthly(q, "M")


TABLE6 = """
     Tabela 6: Série Encadeada do Índice Trimestral com Ajuste Sazonal (média de 1995=100)
                Tabela 6 - Série Encadeada do Índice de Volume Trimestral com Ajuste Sazonal
 Período    Agropecuária     Indústria   Serviços    VApb       PIB pm
2019.IV         232,4    132,3   179,2   169,3    171,9   185,4   147,6   147,8   311,2   251,3
2020.I          236,6    129,9   178,4   165,6    168,2   181,8   148,2   155,4   302,7   251,9
                Contas Nacionais Trimestrais (page header)
2020.I          999,9    129,9   178,4   165,6    999,9   181,8   148,2   155,4   302,7   251,9
2020.II       1.238,1    114,5   161,0   151,4    153,1   162,9   136,3   131,1   301,0   218,3
                Tabela 7 - Taxa Trimestre contra Trimestre Imediatamente Anterior (%)
2020.III          1,0      1,0     1,0     1,0      1,0     1,0     1,0     1,0     1,0     1,0
"""


def test_parse_table6(scripts: dict[str, ModuleType]) -> None:
    vint = scripts["vintages"]
    table = vint.parse_table6(TABLE6)
    assert [str(p) for p in table.index] == ["2019Q4", "2020Q1", "2020Q2"]
    assert table.loc[pd.Period("2020Q1", "Q"), "pib"] == 168.2  # first occurrence kept
    assert table.loc[pd.Period("2020Q2", "Q"), "pib_agropecuaria"] == 1238.1
    assert list(table.columns) == vint.COLUMNS


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("nothing here", "not found"),
        ("Tabela 6 - Serie\nno rows\nTabela 7 - x", "no rows"),
        ("Tabela 6 - Serie\n2020.I 1,0 2,0\n", "layout"),
        (
            "Tabela 6 - Serie\n2020.I " + " 1,0" * 10 + "\n2020.III" + " 1,0" * 10 + "\n",
            "gaps",
        ),
    ],
)
def test_parse_table6_errors(scripts: dict[str, ModuleType], text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        scripts["vintages"].parse_table6(text)


def test_release_date_rules(scripts: dict[str, ModuleType]) -> None:
    rd = scripts["vintages"].release_date
    q = pd.Period("2019Q2", "Q")
    # 'Publicado em' with a wrong year (as in the real 2019Q2 publication) is fixed
    date, how = rd(q, "Publicado em 29/08/2018 às 9 horas", {}, None, None)
    assert date == pd.Timestamp("2019-08-29") and "Publicado" in how
    # implausible publication date ignored -> FTP stamp at 09:00
    entry = {"pdf": ("u", pd.Timestamp("2019-08-29 09:00"))}
    date, how = rd(q, "Publicado em 01/01/2019", entry, None, None)
    assert date == pd.Timestamp("2019-08-29") and "FTP" in how
    # batch upload time stamps are not release times -> ZIP member time (after 9h: next day)
    entry = {"pdf": ("u", pd.Timestamp("2016-08-17 11:10"))}
    date, how = rd(q, "", entry, pd.Timestamp("2019-08-28 15:41"), None)
    assert date == pd.Timestamp("2019-08-29") and "ZIP" in how
    # early-morning stamp on a Saturday rolls to Monday
    date, _ = rd(q, "", {}, pd.Timestamp("2019-08-31 08:00"), None)
    assert date == pd.Timestamp("2019-09-02")
    # PDF creation date, then the fallback estimate
    date, how = rd(q, "", {}, pd.Timestamp("2019-12-30"), pd.Timestamp("2019-08-28 07:00"))
    assert date == pd.Timestamp("2019-08-28") and "PDF" in how
    date, how = rd(q, "", {}, None, None)
    assert date == pd.Timestamp("2019-06-30") + pd.Timedelta(days=65) and "estimated" in how


def test_changes_only(scripts: dict[str, ModuleType]) -> None:
    rec = pd.DataFrame(
        {
            "series": ["a"] * 4,
            "reference_period": ["2020Q1"] * 3 + ["2020Q2"],
            "vintage_date": ["2020-06-01", "2020-09-01", "2020-12-01", "2020-09-01"],
            "value": [1.0, 1.0, 1.1, 2.0],
        }
    )
    out = scripts["vintages"].changes_only(rec)
    assert out["vintage_date"].tolist() == ["2020-06-01", "2020-12-01", "2020-09-01"]


def test_fred_tcode_map_matches_mcCracken_ng(scripts: dict[str, ModuleType]) -> None:
    """Each mapped transform reproduces the FRED-MD definition on a positive series."""
    from nowcastbox.preprocessing.transforms import get_transform

    x = np.exp(np.cumsum(np.random.default_rng(0).normal(0.01, 0.02, 40))) * 100
    s = pd.Series(x, index=pd.period_range("2000-01", periods=40, freq="M"))
    expected = {
        1: s,
        2: s.diff(),
        3: s.diff().diff(),
        4: np.log(s),
        5: np.log(s).diff(),
        6: np.log(s).diff().diff(),
        7: (s / s.shift(1) - 1).diff(),
    }
    for code, spec in scripts["fred"].TCODE_TRANSFORM.items():
        got = get_transform(spec).apply(s.to_frame("x"))["x"]
        np.testing.assert_allclose(got.dropna(), expected[code].dropna(), rtol=1e-12)


def test_fred_legend_row(scripts: dict[str, ModuleType]) -> None:
    row = scripts["fred"].legend_row("UNRATE", 2, {"group": 2, "description": "Unemployment"})
    assert row["blocks"] == "global;labor"
    assert row["category"] == "hard"
    assert row["delay_days"] == 5
    assert row["legacy_code"] == "2"
    fallback = scripts["fred"].legend_row("NEWSERIES", 5, None)
    assert fallback["description"] == "NEWSERIES"


def test_nyfed_legend_rows(scripts: dict[str, ModuleType]) -> None:
    spec = pd.DataFrame(
        {
            "Model": [1, 0],
            "SeriesID": ["GACDISA066MSFRBNY", "GDPC1"],
            "SeriesName": ["Empire State Mfg Index", "Real GDP"],
            "Frequency": ["m", "q"],
            "Block1-Global": [1, 1],
            "Block2-Soft": [1, 0],
            "Block3-Real": [0, 1],
            "Block4-Labor": [0, 0],
            "Transformation": ["lin", "pca"],
            "Units": ["Index", "Chained $"],
            "Category": ["Surveys", "National Accounts"],
        }
    )
    rows = scripts["nyfed"].legend_rows(spec)
    assert rows[0]["blocks"] == "global;soft" and rows[0]["category"] == "soft"
    assert rows[1]["frequency"] == "Q" and rows[1]["transform"] == "log|diff(1)|scale(400)"
    assert rows[1]["nyfed_model"] == 0

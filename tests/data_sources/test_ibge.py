from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.core.frequency import Frequency
from nowcastbox.data_sources import (
    SIDRA_URL,
    DataSourceError,
    HTTPStatusError,
    fetch_sidra,
    fetch_sidra_raw,
    parse_sidra_period,
    sidra_path,
)
from tests.data_sources.fakes import FakeResponse, FakeSession


def header(period_label="Mês", extra=("Seções e atividades industriais (CNAE 2.0)",)):
    h = {
        "NC": "Nível Territorial (Código)",
        "NN": "Nível Territorial",
        "MC": "Unidade de Medida (Código)",
        "MN": "Unidade de Medida",
        "V": "Valor",
        "D1C": "Brasil (Código)",
        "D1N": "Brasil",
        "D2C": "Variável (Código)",
        "D2N": "Variável",
        "D3C": f"{period_label} (Código)",
        "D3N": period_label,
    }
    for k, label in enumerate(extra, start=4):
        h[f"D{k}C"] = f"{label} (Código)"
        h[f"D{k}N"] = label
    return h


def row(period, value, var=("12606", "PIMPF - Número-índice"), cat=("129314", "1 Indústria geral")):
    return {
        "NC": "1",
        "NN": "Brasil",
        "MC": "30",
        "MN": "Número-índice",
        "V": value,
        "D1C": "1",
        "D1N": "Brasil",
        "D2C": var[0],
        "D2N": var[1],
        "D3C": period,
        "D3N": period,
        "D4C": cat[0],
        "D4N": cat[1],
    }


def session_for(payload, status=200):
    return FakeSession(lambda url, params: FakeResponse.json_body(payload, status_code=status))


class TestPath:
    def test_full(self):
        path = sidra_path(8888, 12606, {"c544": [129314, 129315]}, periods="last 12")
        assert path == "/t/8888/n1/all/v/12606/p/last 12/c544/129314,129315"

    def test_defaults_and_levels(self):
        assert sidra_path("1620") == "/t/1620/n1/all/v/all/p/all"
        assert sidra_path(
            1620, [583, 584], {315: None}, territorial_level="N3", territories=[35, 33]
        ) == ("/t/1620/n3/35,33/v/583,584/p/all/c315/all")

    @pytest.mark.parametrize(
        ("kwargs", "exc"),
        [
            ({"table": "abc"}, ValueError),
            ({"table": 1, "territorial_level": "x1"}, ValueError),
            ({"table": 1, "variable": []}, ValueError),
            ({"table": 1, "variable": "1/2"}, ValueError),
            ({"table": 1, "variable": 1.5}, TypeError),
            ({"table": 1, "classifications": {"cx": 1}}, ValueError),
            ({"table": 1, "periods": ""}, ValueError),
        ],
    )
    def test_invalid(self, kwargs, exc):
        with pytest.raises(exc):
            sidra_path(**kwargs)


class TestPeriods:
    @pytest.mark.parametrize(
        ("code", "freq", "expected"),
        [
            ("202301", "M", pd.Period("2023-01", "M")),
            ("202312", "M", pd.Period("2023-12", "M")),
            ("202304", "Q", pd.Period("2023Q4", "Q")),
            ("2023", "A", pd.Period("2023", "Y")),
        ],
    )
    def test_valid(self, code, freq, expected):
        assert parse_sidra_period(code, freq) == expected

    @pytest.mark.parametrize(
        ("code", "freq"), [("202313", "M"), ("202305", "Q"), ("2023", "M"), ("20231", "Q")]
    )
    def test_invalid(self, code, freq):
        with pytest.raises(NowcastDataError):
            parse_sidra_period(code, freq)


class TestRaw:
    def test_raw_long_frame(self):
        payload = [header(), row("202301", "100.5"), row("202302", "-"), row("202303", "..")]
        session = session_for(payload)
        raw = fetch_sidra_raw(8888, 12606, {"c544": 129314}, session=session, cache=False)
        assert session.calls[0].url == SIDRA_URL + "/t/8888/n1/all/v/12606/p/all/c544/129314"
        assert raw["value"].iloc[:2].tolist() == [100.5, 0.0]
        assert np.isnan(raw["value"].iloc[2])
        assert raw.attrs["frequency"] == Frequency.MONTHLY
        assert "Variável (Código)" in raw.columns
        assert ("Variável (Código)", "Variável") in raw.attrs["dimensions"]

    def test_dash_as_nan(self):
        """``dash_as=nan`` keeps unavailable index values missing (wave-2 integration)."""
        payload = [header(), row("202301", "100.5"), row("202302", "-")]
        raw = fetch_sidra_raw(8888, session=session_for(payload), cache=False, dash_as=np.nan)
        assert raw["value"].iloc[0] == 100.5
        assert np.isnan(raw["value"].iloc[1])
        wide = fetch_sidra(
            8888, name="pim", session=session_for(payload), cache=False, dash_as=np.nan
        )
        assert wide["pim"].isna().sum() == 1

    def test_unknown_value_warns(self):
        payload = [header(), row("202301", "1,5")]
        with pytest.warns(DataQualityWarning, match="Unrecognized"):
            raw = fetch_sidra_raw(8888, session=session_for(payload), cache=False)
        assert np.isnan(raw["value"].iloc[0])

    def test_none_value(self):
        payload = [header(), row("202301", None)]
        raw = fetch_sidra_raw(8888, session=session_for(payload), cache=False)
        assert np.isnan(raw["value"].iloc[0])

    @pytest.mark.parametrize(
        ("label", "freq"),
        [
            ("Trimestre", Frequency.QUARTERLY),
            ("Trimestre Móvel", Frequency.MONTHLY),
            ("Ano", Frequency.ANNUAL),
        ],
    )
    def test_period_labels(self, label, freq):
        code = "2023" if freq == Frequency.ANNUAL else "202301"
        raw = fetch_sidra_raw(1, session=session_for([header(label), row(code, "1")]), cache=False)
        assert raw.attrs["frequency"] == freq

    def test_unsupported_period(self):
        with pytest.raises(NowcastDataError, match="not supported"):
            fetch_sidra_raw(
                1, session=session_for([header("Semestre"), row("202301", "1")]), cache=False
            )

    def test_override_frequency(self):
        raw = fetch_sidra_raw(
            1,
            session=session_for([header("Trimestre"), row("202311", "1")]),
            cache=False,
            native_frequency="M",
        )
        assert raw["period"].iloc[0] == pd.Period("2023-11", "M")

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ({"a": 1}, "header row"),
            ([], "header row"),
            ([[1]], "header row"),
            ([{"NC": "x"}], "'V'"),
            ([{"V": "Valor", "D1C": "Brasil (Código)"}], "No period dimension"),
        ],
    )
    def test_bad_payload(self, payload, match):
        with pytest.raises(DataSourceError, match=match):
            fetch_sidra_raw(1, session=session_for(payload), cache=False)

    def test_http_400_message(self):
        session = FakeSession(
            lambda url, params: FakeResponse(
                400, "Parâmetro V (Variável) com código 9999 inexistente"
            )
        )
        with pytest.raises(HTTPStatusError, match="inexistente"):
            fetch_sidra_raw(1620, 9999, session=session, cache=False)


class TestFetchSidra:
    def test_single_series_default_name(self):
        payload = [header(), row("202301", "1"), row("202303", "3")]
        df = fetch_sidra(8888, 12606, session=session_for(payload), cache=False)
        assert df.columns.tolist() == ["sidra_8888_v12606"]
        assert df.index.astype(str).tolist() == ["2023-01", "2023-02", "2023-03"]
        assert np.isnan(df.iloc[1, 0])

    def test_name_and_table_default(self):
        payload = [header(), row("202301", "1")]
        assert fetch_sidra(
            8888, session=session_for(payload), cache=False, name="pim"
        ).columns.tolist() == ["pim"]
        assert fetch_sidra(8888, session=session_for(payload), cache=False).columns.tolist() == [
            "sidra_8888"
        ]

    def test_multiple_categories_become_columns(self):
        payload = [
            header(),
            row("202301", "1", cat=("1", "Geral")),
            row("202301", "2", cat=("2", "Extrativa")),
            row("202302", "3", cat=("1", "Geral")),
            row("202302", "4", cat=("2", "Extrativa")),
        ]
        df = fetch_sidra(8888, 12606, {"c544": "all"}, session=session_for(payload), cache=False)
        assert df.columns.tolist() == ["Geral", "Extrativa"]
        assert df["Extrativa"].tolist() == [2.0, 4.0]
        prefixed = fetch_sidra(8888, 12606, session=session_for(payload), cache=False, name="pim")
        assert prefixed.columns.tolist() == ["pim: Geral", "pim: Extrativa"]

    def test_duplicate_labels_get_codes(self):
        payload = [
            header(),
            row("202301", "1", cat=("1", "Same")),
            row("202301", "2", cat=("2", "Same")),
        ]
        df = fetch_sidra(8888, session=session_for(payload), cache=False)
        assert df.columns.tolist() == ["Same [1]", "Same [2]"]

    def test_two_varying_dimensions(self):
        payload = [
            header(),
            row("202301", "1", var=("1", "A"), cat=("1", "X")),
            row("202301", "2", var=("2", "B"), cat=("2", "Y")),
        ]
        df = fetch_sidra(8888, session=session_for(payload), cache=False)
        assert df.columns.tolist() == ["A | X", "B | Y"]

    def test_quarterly_on_monthly_grid_and_range(self):
        payload = [header("Trimestre")] + [row(f"2022{q:02d}", str(q)) for q in range(1, 5)]
        df = fetch_sidra(
            1620,
            583,
            session=session_for(payload),
            cache=False,
            base_frequency="M",
            start="2022Q2",
            end="2022Q3",
        )
        assert df.index.astype(str).tolist() == ["2022-06", "2022-07", "2022-08", "2022-09"]
        assert df.iloc[:, 0].dropna().tolist() == [2.0, 3.0]

    def test_empty_answer(self):
        with pytest.raises(NowcastDataError, match="no observations"):
            fetch_sidra(8888, session=session_for([header()]), cache=False)

    def test_duplicated_observations(self):
        payload = [header(), row("202301", "1"), row("202301", "1")]
        with pytest.raises(NowcastDataError, match="duplicated"):
            fetch_sidra(8888, session=session_for(payload), cache=False)

    def test_reversed_range(self):
        payload = [header(), row("202301", "1")]
        with pytest.raises(ValueError, match="after"):
            fetch_sidra(
                8888, session=session_for(payload), cache=False, start="2024-01", end="2023-01"
            )

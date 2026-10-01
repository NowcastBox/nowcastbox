from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.data_sources import (
    IPEADATA_URL,
    DataSourceError,
    fetch_ipeadata,
    fetch_ipeadata_metadata,
    fetch_ipeadata_series,
)
from tests.data_sources.fakes import FakeResponse, FakeSession


def odata(code, dates, values=None, level="", territory=""):
    values = values if values is not None else list(range(len(dates)))
    return [
        {
            "SERCODIGO": code,
            "VALDATA": f"{d:%Y-%m-%d}T00:00:00-03:00",
            "VALVALOR": v,
            "NIVNOME": level,
            "TERCODIGO": territory,
        }
        for d, v in zip(dates, values, strict=True)
    ]


MONTHS = pd.date_range("2019-01-01", periods=24, freq="MS")
QUARTERS = pd.date_range("2015-01-01", periods=12, freq="QS")


def server(url, params):
    if "Metadados" in url:
        if "NOPE" in url:
            return FakeResponse.json_body({"value": []})
        return FakeResponse.json_body({"value": [{"SERCODIGO": "PIB", "PERNOME": "Trimestral"}]})
    code = url.split("SERCODIGO='")[1].rstrip("')")
    if code == "MENSAL":
        return FakeResponse.json_body({"value": odata(code, MONTHS)})
    if code == "TRIM":
        return FakeResponse.json_body({"value": odata(code, QUARTERS)})
    if code == "REGIONAL":
        rows = odata(code, MONTHS[:3], level="Brasil", territory="0") + odata(
            code, MONTHS[:3], values=[10, 11, 12], level="Estados", territory="35"
        )
        return FakeResponse.json_body({"value": rows})
    return FakeResponse.json_body({"value": []})


class TestSeries:
    def test_url_and_parsing(self):
        session = FakeSession(server)
        s = fetch_ipeadata_series("MENSAL", session=session, cache=False)
        assert session.calls[0].url == f"{IPEADATA_URL}/ValoresSerie(SERCODIGO='MENSAL')"
        assert s.name == "MENSAL"
        assert s.index.freqstr == "M"
        assert s.index[0] == pd.Period("2019-01", "M")  # UTC offset never shifts the day
        assert s.iloc[:3].tolist() == [0.0, 1.0, 2.0]

    def test_range_filter(self):
        s = fetch_ipeadata_series(
            "MENSAL", "2019-06", "2019-08", session=FakeSession(server), cache=False
        )
        assert s.index.astype(str).tolist() == ["2019-06", "2019-07", "2019-08"]

    def test_null_values(self):
        rows = odata("X", MONTHS[:2], values=[None, 1.5])
        session = FakeSession(lambda url, params: FakeResponse.json_body({"value": rows}))
        s = fetch_ipeadata_series("X", session=session, cache=False)
        assert np.isnan(s.iloc[0]) and s.iloc[1] == 1.5

    def test_territories(self):
        with pytest.raises(NowcastDataError, match="2 territories"):
            fetch_ipeadata_series("REGIONAL", session=FakeSession(server), cache=False)
        sp = fetch_ipeadata_series(
            "REGIONAL", session=FakeSession(server), cache=False, territory_code="35"
        )
        assert sp.tolist() == [10.0, 11.0, 12.0]
        br = fetch_ipeadata_series(
            "REGIONAL", session=FakeSession(server), cache=False, territorial_level="Brasil"
        )
        assert br.tolist() == [0.0, 1.0, 2.0]

    def test_unknown_code_empty(self):
        with pytest.raises(NowcastDataError, match="no observations"):
            fetch_ipeadata_series("UNKNOWN", session=FakeSession(server), cache=False)

    @pytest.mark.parametrize("code", ["A B", "X')/Y", ""])
    def test_invalid_code(self, code):
        with pytest.raises(ValueError, match="Invalid IPEADATA"):
            fetch_ipeadata_series(code, cache=False)

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ([1], "unexpected IPEADATA answer"),
            ({"error": {"message": "x"}}, "IPEADATA error"),
            ({"foo": 1}, "no 'value' list"),
            ({"value": [{"VALVALOR": 1}]}, "malformed"),
            ({"value": [{"VALDATA": "not a date", "VALVALOR": 1}]}, "malformed"),
        ],
    )
    def test_bad_payload(self, payload, match):
        session = FakeSession(lambda url, params: FakeResponse.json_body(payload))
        with pytest.raises(DataSourceError, match=match):
            fetch_ipeadata_series("X", session=session, cache=False)

    def test_custom_base_url(self):
        session = FakeSession(server)
        fetch_ipeadata_series(
            "MENSAL", session=session, cache=False, base_url="https://mirror/odata4"
        )
        assert session.calls[0].url.startswith("https://mirror/odata4/ValoresSerie")


class TestFetch:
    def test_codes_and_names(self):
        df = fetch_ipeadata(["MENSAL"], session=FakeSession(server), cache=False)
        assert df.columns.tolist() == ["MENSAL"]
        df = fetch_ipeadata({"m": "MENSAL"}, session=FakeSession(server), cache=False)
        assert df.columns.tolist() == ["m"]

    def test_mixed(self):
        with pytest.raises(NowcastDataError, match="base_frequency"):
            fetch_ipeadata(["MENSAL", "TRIM"], session=FakeSession(server), cache=False)
        df = fetch_ipeadata(
            {"m": "MENSAL", "q": "TRIM"},
            session=FakeSession(server),
            cache=False,
            base_frequency="M",
        )
        assert df.index[0] == pd.Period("2015-03", "M")
        assert df.index[-1] == pd.Period("2020-12", "M")
        assert df["q"].notna().sum() == 12


class TestMetadata:
    def test_metadata(self):
        session = FakeSession(server)
        meta = fetch_ipeadata_metadata("PIB", session=session, cache=False)
        assert meta["PERNOME"] == "Trimestral"
        assert session.calls[0].url == f"{IPEADATA_URL}/Metadados('PIB')"

    def test_metadata_not_found(self):
        with pytest.raises(DataSourceError, match="not found"):
            fetch_ipeadata_metadata("NOPE", session=FakeSession(server), cache=False)

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.data_sources import (
    FRED_URL,
    DataSourceError,
    DiskCache,
    HTTPStatusError,
    MissingAPIKeyError,
    fetch_fred,
    fetch_fred_series,
    get_fred_api_key,
)
from tests.data_sources.fakes import FakeResponse, FakeSession

KEY = "0123456789abcdef0123456789abcdef"


def observations(dates, values):
    return {
        "observations": [
            {
                "realtime_start": "2024-01-01",
                "realtime_end": "2024-01-01",
                "date": f"{d:%Y-%m-%d}",
                "value": v,
            }
            for d, v in zip(dates, values, strict=True)
        ]
    }


def server(url, params):
    sid = params["series_id"]
    if sid == "GDPC1":
        return FakeResponse.json_body(
            observations(pd.date_range("2020-01-01", periods=4, freq="QS"), ["1", "2", ".", "4"])
        )
    if sid == "INDPRO":
        return FakeResponse.json_body(
            observations(
                pd.date_range("2020-01-01", periods=12, freq="MS"), [str(i) for i in range(12)]
            )
        )
    return FakeResponse.json_body(
        {"error_code": 400, "error_message": "Bad Request. The series does not exist."},
        status_code=400,
    )


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)


class TestKey:
    def test_argument_and_env(self, monkeypatch):
        assert get_fred_api_key(" k ") == "k"
        monkeypatch.setenv("FRED_API_KEY", "envkey")
        assert get_fred_api_key() == "envkey"

    @pytest.mark.parametrize("key", [None, "", "   "])
    def test_missing(self, key):
        with pytest.raises(MissingAPIKeyError, match="FRED_API_KEY"):
            get_fred_api_key(key)

    def test_fetch_without_key(self):
        with pytest.raises(MissingAPIKeyError):
            fetch_fred("GDPC1", cache=False)


class TestSeries:
    def test_request_and_parsing(self):
        session = FakeSession(server)
        s = fetch_fred_series(
            "gdpc1", "2020-01-01", "2020Q4", api_key=KEY, session=session, cache=False
        )
        call = session.calls[0]
        assert call.url == f"{FRED_URL}/series/observations"
        assert call.params == {
            "series_id": "GDPC1",
            "api_key": KEY,
            "file_type": "json",
            "observation_start": "2020-01-01",
            "observation_end": "2020-12-31",
        }
        assert s.name == "GDPC1"
        assert s.index.freqstr.startswith("Q")
        assert np.isnan(s.iloc[2]) and s.iloc[3] == 4.0

    def test_vintage(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", KEY)
        session = FakeSession(server)
        fetch_fred_series("GDPC1", vintage_date="2021-07-29", session=session, cache=False)
        assert session.calls[0].params["realtime_start"] == "2021-07-29"
        assert session.calls[0].params["realtime_end"] == "2021-07-29"

    def test_api_error_redacts_key(self):
        with pytest.raises(HTTPStatusError) as info:
            fetch_fred_series("NOPE", api_key=KEY, session=FakeSession(server), cache=False)
        assert KEY not in str(info.value)
        assert "does not exist" in str(info.value)

    def test_cache_never_contains_key(self, tmp_path):
        cache = DiskCache(tmp_path)
        session = FakeSession(server)
        fetch_fred_series("GDPC1", api_key=KEY, session=session, cache=cache)
        fetch_fred_series("GDPC1", api_key="another-key", session=session, cache=cache)
        assert len(session.calls) == 1  # same cache entry regardless of the key
        for path in tmp_path.iterdir():
            assert KEY not in path.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        ("payload", "exc", "match"),
        [
            ({"error_message": "bad"}, DataSourceError, "API error: bad"),
            ([1, 2], DataSourceError, "API error"),
            ({"foo": 1}, DataSourceError, "observations"),
            ({"observations": []}, NowcastDataError, "no observations"),
            ({"observations": [{"value": "1"}]}, DataSourceError, "malformed"),
        ],
    )
    def test_bad_payload(self, payload, exc, match):
        session = FakeSession(lambda url, params: FakeResponse.json_body(payload))
        with pytest.raises(exc, match=match):
            fetch_fred_series("X", api_key=KEY, session=session, cache=False)

    @pytest.mark.parametrize("sid", ["A B", "x/y", ""])
    def test_invalid_id(self, sid):
        with pytest.raises(ValueError, match="Invalid FRED"):
            fetch_fred_series(sid, api_key=KEY, cache=False)


class TestFetch:
    def test_mixed_on_monthly_grid(self):
        df = fetch_fred(
            {"gdp": "GDPC1", "ip": "INDPRO"},
            api_key=KEY,
            base_frequency="M",
            session=FakeSession(server),
            cache=False,
        )
        assert df.columns.tolist() == ["gdp", "ip"]
        assert df.index.freqstr == "M" and len(df) == 12
        assert df["gdp"].loc[pd.Period("2020-03", "M")] == 1.0

    def test_list(self):
        df = fetch_fred(["indpro"], api_key=KEY, session=FakeSession(server), cache=False)
        assert df.columns.tolist() == ["INDPRO"]

    def test_mixed_without_base(self):
        with pytest.raises(NowcastDataError):
            fetch_fred(["GDPC1", "INDPRO"], api_key=KEY, session=FakeSession(server), cache=False)

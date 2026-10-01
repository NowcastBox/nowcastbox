from __future__ import annotations

import datetime as dt
import itertools

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.data_sources import (
    SGS_URL,
    DataSourceError,
    DiskCache,
    HTTPStatusError,
    fetch_sgs,
    fetch_sgs_series,
    sgs_windows,
)
from tests.data_sources.fakes import FakeResponse, FakeSession

WINDOW_ERROR = {
    "error": "O sistema aceita uma janela de consulta de, no máximo, 10 anos em séries de "
    "periodicidade diária",
    "message": "Para acessar uma série de periodicidade diária, é necessário informar a dataInicial",
}
NOT_FOUND = {"erro": {"statusCode": 404, "detail": "SGSNegocioException: Value(s) not found"}}


def code_of(url: str) -> str:
    return url.split("bcdata.sgs.")[1].split("/")[0]


def records(dates, values=None):
    values = values if values is not None else [f"{i + 0.5:.2f}" for i in range(len(dates))]
    return [
        {"data": d.strftime("%d/%m/%Y"), "valor": v} for d, v in zip(dates, values, strict=True)
    ]


class SGSServer:
    """In-memory SGS: monthly/quarterly answer undated queries; daily enforce 10-year windows."""

    def __init__(self, series: dict[str, tuple[pd.DatetimeIndex, bool]]):
        self.series = series

    def __call__(self, url, params):
        dates, daily = self.series[code_of(url)]
        start = params.get("dataInicial")
        end = params.get("dataFinal")
        if daily:
            if start is None:
                return FakeResponse.json_body(WINDOW_ERROR, status_code=406)
            lo = pd.to_datetime(start, format="%d/%m/%Y")
            hi = pd.to_datetime(end, format="%d/%m/%Y") if end else pd.Timestamp.today()
            if hi > lo + pd.DateOffset(years=10) - pd.Timedelta(days=1):
                return FakeResponse.json_body(WINDOW_ERROR, status_code=406)
        lo = pd.to_datetime(start, format="%d/%m/%Y") if start else pd.Timestamp.min
        hi = pd.to_datetime(end, format="%d/%m/%Y") if end else pd.Timestamp.max
        sel = dates[(dates >= lo) & (dates <= hi)]
        if len(sel) == 0:
            return FakeResponse.json_body(NOT_FOUND, status_code=404)
        return FakeResponse.json_body(records(sel))


MONTHLY = pd.date_range("1995-01-01", "2020-12-01", freq="MS")
QUARTERLY = pd.date_range("1996-01-01", "2020-10-01", freq="QS")
DAILY = pd.bdate_range("1990-01-02", "2021-06-30")


@pytest.fixture
def server():
    return SGSServer({"433": (MONTHLY, False), "22099": (QUARTERLY, False), "1": (DAILY, True)})


class TestWindows:
    def test_example(self):
        w = sgs_windows("2000-01-01", "2021-06-30")
        assert [(a.date().isoformat(), b.date().isoformat()) for a, b in w] == [
            ("2000-01-01", "2009-12-31"),
            ("2010-01-01", "2019-12-31"),
            ("2020-01-01", "2021-06-30"),
        ]

    def test_single_day_and_period_bounds(self):
        assert sgs_windows("2020-01-01", "2020-01-01") == [
            (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01"))
        ]
        w = sgs_windows("2020Q1", "2020Q2")
        assert w == [(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30"))]

    @pytest.mark.parametrize("years", [0, -1, 1.5, True])
    def test_invalid_years(self, years):
        with pytest.raises(ValueError):
            sgs_windows("2000-01-01", "2001-01-01", years=years)

    def test_missing_bound(self):
        with pytest.raises(ValueError, match="both"):
            sgs_windows(None, "2001-01-01")

    def test_reversed(self):
        with pytest.raises(ValueError):
            sgs_windows("2001-01-01", "2000-01-01")

    @given(
        start=st.dates(min_value=dt.date(1900, 1, 1), max_value=dt.date(2030, 1, 1)),
        length=st.integers(min_value=0, max_value=60_000),
        years=st.integers(min_value=1, max_value=12),
    )
    def test_property_partition(self, start, length, years):
        first = pd.Timestamp(start)
        last = first + pd.Timedelta(days=length)
        w = sgs_windows(first, last, years=years)
        assert w[0][0] == first and w[-1][1] == last
        for lo, hi in w:
            assert lo <= hi
            assert hi <= lo + pd.DateOffset(years=years) - pd.Timedelta(days=1)
        for (_, hi), (lo, _) in itertools.pairwise(w):
            assert lo == hi + pd.Timedelta(days=1)


class TestFetchSeries:
    def test_monthly_undated_single_request(self, server):
        session = FakeSession(server)
        s = fetch_sgs_series(433, session=session, cache=False)
        assert len(session.calls) == 1
        assert session.calls[0].url == SGS_URL.format(code=433)
        assert session.calls[0].params == {"formato": "json"}
        assert s.index.freqstr == "M"
        assert s.name == "sgs_433"
        assert s.index[0] == pd.Period("1995-01", "M") and len(s) == len(MONTHLY)

    def test_quarterly_dates_are_period_starts(self, server):
        s = fetch_sgs_series(
            "22099", start="2019-01", end="2019-12", session=FakeSession(server), cache=False
        )
        assert s.index.astype(str).tolist() == ["2019Q1", "2019Q2", "2019Q3", "2019Q4"]

    def test_dated_request_params_and_chunks(self, server):
        session = FakeSession(server)
        s = fetch_sgs_series(433, start="1990-01-01", end="2015-06", session=session, cache=False)
        params = [c.params for c in session.calls]
        assert params[0] == {
            "formato": "json",
            "dataInicial": "01/01/1990",
            "dataFinal": "31/12/1999",
        }
        assert params[-1]["dataFinal"] == "30/06/2015"
        assert len(params) == 3
        assert s.index[0] == pd.Period("1995-01", "M") and s.index[-1] == pd.Period("2015-06", "M")

    def test_daily_with_start_forward_windows(self, server):
        session = FakeSession(server)
        s = fetch_sgs_series(1, start="2000-01-01", end="2021-06-30", session=session, cache=False)
        assert len(session.calls) == 3
        assert s.index.freqstr == "D"
        expected = DAILY[DAILY >= "2000-01-01"]
        assert len(s) == len(expected)
        assert not s.index.has_duplicates

    def test_daily_without_start_walks_back(self, server):
        session = FakeSession(server)
        s = fetch_sgs_series(1, end="2021-06-30", session=session, cache=False)
        assert len(s) == len(DAILY)
        # 1 window per 10 years back to 1990 plus one empty window that stops the walk
        assert len(session.calls) == 5
        assert s.index.is_monotonic_increasing

    def test_daily_undated_falls_back(self, server, monkeypatch):
        monkeypatch.setattr(
            pd.Timestamp, "today", classmethod(lambda cls: pd.Timestamp("2021-07-15"))
        )
        session = FakeSession(server)
        s = fetch_sgs_series(1, session=session, cache=False)
        assert session.calls[0].params == {"formato": "json"}
        assert len(s) == len(DAILY)

    def test_start_without_end_uses_today(self, server, monkeypatch):
        monkeypatch.setattr(
            pd.Timestamp, "today", classmethod(lambda cls: pd.Timestamp("2021-01-15"))
        )
        session = FakeSession(server)
        s = fetch_sgs_series(433, start="2020-01", session=session, cache=False)
        assert session.calls[-1].params["dataFinal"] == "15/01/2021"
        assert s.index[-1] == pd.Period("2020-12", "M")

    def test_empty_range_raises(self, server):
        with pytest.raises(NowcastDataError, match="no observations"):
            fetch_sgs_series(
                433, start="1950-01", end="1960-01", session=FakeSession(server), cache=False
            )

    def test_undated_404_raises(self):
        session = FakeSession(
            lambda url, params: FakeResponse.json_body(NOT_FOUND, status_code=404)
        )
        with pytest.raises(HTTPStatusError):
            fetch_sgs_series(433, session=session, cache=False)

    def test_other_http_error_in_undated(self):
        session = FakeSession(lambda url, params: FakeResponse(400, "bad"))
        with pytest.raises(HTTPStatusError):
            fetch_sgs_series(433, session=session, cache=False)

    def test_invalid_code_html(self):
        session = FakeSession(lambda url, params: FakeResponse(200, "<html>SGS</html>"))
        with pytest.raises(DataSourceError, match="did not return JSON"):
            fetch_sgs_series(999999, session=session, cache=False, max_retries=0)

    @pytest.mark.parametrize("code", ["abc", "-1", "1.5", ""])
    def test_invalid_code(self, code):
        with pytest.raises(ValueError, match="SGS codes"):
            fetch_sgs_series(code, cache=False)

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ({"error": "boom"}, "API error: boom"),
            ("text", "unexpected response"),
            ([{"data": "01/01/2020"}], "malformed record"),
            ([1], "malformed record"),
        ],
    )
    def test_bad_payloads(self, payload, match):
        session = FakeSession(lambda url, params: FakeResponse.json_body(payload))
        with pytest.raises(DataSourceError, match=match):
            fetch_sgs_series(433, session=session, cache=False)

    def test_values_parsing(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="MS")
        session = FakeSession(
            lambda url, params: FakeResponse.json_body(records(dates, ["1.5", "", "-2"]))
        )
        s = fetch_sgs_series(433, session=session, cache=False)
        assert s.iloc[0] == 1.5 and np.isnan(s.iloc[1]) and s.iloc[2] == -2.0

    def test_native_frequency_override(self):
        dates = pd.to_datetime(["2020-01-01"])
        session = FakeSession(lambda url, params: FakeResponse.json_body(records(dates)))
        s = fetch_sgs_series(433, session=session, cache=False, native_frequency="Q", name="x")
        assert s.index[0] == pd.Period("2020Q1", "Q") and s.name == "x"

    def test_reversed_range(self):
        with pytest.raises(ValueError, match="after"):
            fetch_sgs_series(433, start="2021-01", end="2020-01", cache=False)

    def test_cache_avoids_second_download(self, server, tmp_path):
        session = FakeSession(server)
        cache = DiskCache(tmp_path)
        a = fetch_sgs_series(433, start="2010-01", end="2012-12", session=session, cache=cache)
        b = fetch_sgs_series(433, start="2010-01", end="2012-12", session=session, cache=cache)
        assert len(session.calls) == 1
        pd.testing.assert_series_equal(a, b)

    def test_default_cache_is_used(self, server, tmp_path, monkeypatch):
        monkeypatch.setenv("NOWCASTBOX_CACHE_DIR", str(tmp_path / "dflt"))
        session = FakeSession(server)
        fetch_sgs_series(433, session=session)
        fetch_sgs_series(433, session=session)
        assert len(session.calls) == 1
        assert len(list((tmp_path / "dflt").glob("*.json"))) == 1


class TestFetchSGS:
    def test_named_columns_same_frequency(self, server):
        df = fetch_sgs(
            {"ipca": 433}, start="2020-01", end="2020-06", session=FakeSession(server), cache=False
        )
        assert df.columns.tolist() == ["ipca"]
        assert df.index.freqstr == "M" and len(df) == 6
        assert df.dtypes.iloc[0] == np.float64

    def test_list_of_codes(self, server):
        df = fetch_sgs(
            [433], start="2020-01", end="2020-03", session=FakeSession(server), cache=False
        )
        assert df.columns.tolist() == ["sgs_433"]

    def test_mixed_requires_base_frequency(self, server):
        with pytest.raises(NowcastDataError, match="base_frequency"):
            fetch_sgs(
                [433, 22099],
                start="2020-01",
                end="2020-12",
                session=FakeSession(server),
                cache=False,
            )

    def test_mixed_on_monthly_grid(self, server):
        df = fetch_sgs(
            {"ipca": 433, "gdp": 22099},
            start="2020-01",
            end="2020-12",
            base_frequency="M",
            session=FakeSession(server),
            cache=False,
        )
        assert df.index.freqstr == "M" and len(df) == 12
        assert df["gdp"].notna().sum() == 4
        assert df["gdp"].dropna().index.month.tolist() == [3, 6, 9, 12]

    def test_native_frequency_mapping(self, server):
        df = fetch_sgs(
            {"gdp": 22099},
            start="2020-01",
            end="2020-12",
            native_frequency={"gdp": "Q"},
            session=FakeSession(server),
            cache=False,
        )
        assert df.index.freqstr.startswith("Q")

    def test_invalid_codes(self):
        with pytest.raises(ValueError):
            fetch_sgs(["x"], cache=False)

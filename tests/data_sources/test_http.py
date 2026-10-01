from __future__ import annotations

import pytest
import requests

from nowcastbox.data_sources import DataSourceError, DiskCache, HTTPStatusError, _http
from nowcastbox.data_sources._http import RequestOptions, describe_url, get_json, get_text
from tests.data_sources.fakes import FakeResponse, FakeSession

URL = "https://example.org/api"


def options(session, **kw):
    kw.setdefault("cache", False)
    return RequestOptions(session=session, **kw)


class TestRequestOptions:
    @pytest.mark.parametrize(
        ("kwargs", "exc"),
        [
            ({"max_retries": -1}, ValueError),
            ({"max_retries": 1.5}, TypeError),
            ({"max_retries": True}, TypeError),
            ({"timeout": 0}, ValueError),
            ({"backoff_factor": -1}, ValueError),
        ],
    )
    def test_validation(self, kwargs, exc):
        with pytest.raises(exc):
            RequestOptions(**kwargs)


class TestDescribeUrl:
    def test_sorted_and_redacted(self):
        text = describe_url(URL, {"b": 2, "api_key": "SECRET", "a": "x y"}, ("api_key",))
        assert text.startswith(URL + "?a=x+y&b=2")
        assert "SECRET" not in text
        assert "api_key=***" in text

    def test_no_params(self):
        assert describe_url(URL, None) == URL


class TestGetText:
    def test_success_and_headers(self):
        session = FakeSession(lambda url, params: FakeResponse(200, "hello"))
        assert get_text(URL, {"q": 1}, options=options(session, timeout=7)) == "hello"
        call = session.calls[0]
        assert call.params == {"q": 1}
        assert call.timeout == 7
        assert call.headers["User-Agent"].startswith("nowcastbox/")

    def test_retry_on_5xx_then_success(self, sleeps):
        session = FakeSession(
            [FakeResponse(503, "busy"), FakeResponse(500), FakeResponse(200, "ok")]
        )
        assert get_text(URL, options=options(session, backoff_factor=0.5)) == "ok"
        assert sleeps == [0.5, 1.0]
        assert len(session.calls) == 3

    def test_retry_after_header(self, sleeps):
        session = FakeSession(
            [FakeResponse(429, headers={"Retry-After": "3"}), FakeResponse(200, "ok")]
        )
        assert get_text(URL, options=options(session)) == "ok"
        assert sleeps == [3.0]

    def test_retry_after_capped_and_invalid(self, sleeps):
        session = FakeSession(
            [
                FakeResponse(429, headers={"Retry-After": "1000"}),
                FakeResponse(429, headers={"Retry-After": "Wed, 21 Oct 2015"}),
                FakeResponse(200, "ok"),
            ]
        )
        assert get_text(URL, options=options(session, backoff_factor=1.0)) == "ok"
        assert sleeps == [60.0, 2.0]

    def test_retry_on_connection_errors(self, sleeps):
        session = FakeSession(
            [requests.ConnectionError("down"), requests.Timeout("slow"), FakeResponse(200, "ok")]
        )
        assert get_text(URL, options=options(session)) == "ok"
        assert sleeps == [1.0, 2.0]

    def test_retries_exhausted(self, sleeps):
        session = FakeSession(lambda url, params: FakeResponse(502, "bad gateway"))
        with pytest.raises(DataSourceError, match="after 3 attempt"):
            get_text(URL, options=options(session, max_retries=2))
        assert len(session.calls) == 3

    def test_exhausted_connection_error_message(self):
        session = FakeSession(
            lambda url, params: (_ for _ in ()).throw(requests.ConnectionError("x"))
        )
        with pytest.raises(DataSourceError, match="ConnectionError"):
            get_text(URL, options=options(session, max_retries=0))

    def test_client_error_not_retried(self):
        session = FakeSession(lambda url, params: FakeResponse(400, "bad request: details"))
        with pytest.raises(HTTPStatusError) as info:
            get_text(URL, options=options(session))
        assert info.value.status_code == 400
        assert "details" in info.value.body
        assert len(session.calls) == 1

    def test_other_request_exception(self):
        session = FakeSession([requests.TooManyRedirects("loop")])
        with pytest.raises(DataSourceError, match="TooManyRedirects"):
            get_text(URL, options=options(session))

    def test_secret_never_in_errors(self):
        session = FakeSession(lambda url, params: FakeResponse(400, "nope"))
        opts = options(session, secret_params=("api_key",))
        with pytest.raises(HTTPStatusError) as info:
            get_text(URL, {"api_key": "TOPSECRET", "x": 1}, options=opts)
        assert "TOPSECRET" not in str(info.value)
        assert info.value.__cause__ is None
        assert session.calls[0].params["api_key"] == "TOPSECRET"

    def test_cache_used_and_secret_not_stored(self, tmp_path):
        cache = DiskCache(tmp_path)
        session = FakeSession(lambda url, params: FakeResponse(200, "payload"))
        opts = RequestOptions(session=session, cache=cache, secret_params=("api_key",))
        assert get_text(URL, {"api_key": "TOPSECRET"}, options=opts) == "payload"
        assert get_text(URL, {"api_key": "TOPSECRET"}, options=opts) == "payload"
        assert len(session.calls) == 1
        for path in tmp_path.iterdir():
            assert "TOPSECRET" not in path.read_text(encoding="utf-8")

    def test_errors_not_cached(self, tmp_path):
        cache = DiskCache(tmp_path)
        session = FakeSession([FakeResponse(404, "nf"), FakeResponse(200, "ok")])
        opts = RequestOptions(session=session, cache=cache)
        with pytest.raises(HTTPStatusError):
            get_text(URL, options=opts)
        assert len(cache) == 0
        assert get_text(URL, options=opts) == "ok"

    def test_default_session_created_and_closed(self, monkeypatch):
        created = []

        def factory():
            s = FakeSession(lambda url, params: FakeResponse(200, "ok"))
            created.append(s)
            return s

        monkeypatch.setattr(_http.requests, "Session", factory)
        assert get_text(URL, options=RequestOptions(cache=False)) == "ok"
        assert created[0].closed

    def test_user_session_not_closed(self):
        session = FakeSession(lambda url, params: FakeResponse(200, "ok"))
        get_text(URL, options=options(session))
        assert not session.closed


class TestGetJson:
    def test_decodes(self):
        session = FakeSession(lambda url, params: FakeResponse.json_body({"a": [1, 2]}))
        assert get_json(URL, options=options(session)) == {"a": [1, 2]}

    def test_html_answer_retried_then_ok(self, sleeps):
        session = FakeSession([FakeResponse(200, "<html>busy</html>"), FakeResponse.json_body([1])])
        assert get_json(URL, options=options(session)) == [1]
        assert sleeps == [1.0]

    def test_html_answer_exhausted(self):
        session = FakeSession(lambda url, params: FakeResponse(200, "<html>\n  error page</html>"))
        with pytest.raises(DataSourceError, match=r"did not return JSON.*error page"):
            get_json(URL, options=options(session, max_retries=1))
        assert len(session.calls) == 2

    def test_cache_roundtrip_and_corrupted_cache(self, tmp_path):
        cache = DiskCache(tmp_path)
        session = FakeSession(lambda url, params: FakeResponse.json_body({"v": 1}))
        opts = RequestOptions(session=session, cache=cache)
        assert get_json(URL, options=opts) == {"v": 1}
        assert get_json(URL, options=opts) == {"v": 1}
        assert len(session.calls) == 1
        cache.set(f"GET {URL}", "<html>")
        assert get_json(URL, options=opts) == {"v": 1}
        assert len(session.calls) == 2

    def test_invalid_json_not_cached(self, tmp_path):
        cache = DiskCache(tmp_path)
        session = FakeSession(lambda url, params: FakeResponse(200, "<html>"))
        with pytest.raises(DataSourceError):
            get_json(URL, options=RequestOptions(session=session, cache=cache, max_retries=0))
        assert len(cache) == 0

"""Native analysis methods of the results and :func:`nowcastbox.news.attach` / :func:`detach`."""

from __future__ import annotations

import pytest

from nowcastbox.core.results import NowcastResults
from nowcastbox.news import (
    LevelContributions,
    NewsResults,
    NowcastTracker,
    attach,
    detach,
)
from tests.news.conftest import vintage_pair


class _Duck:
    """A results class without the native methods (attach target)."""


@pytest.fixture
def attached():
    names = attach(_Duck)
    yield names
    detach(_Duck)


def test_native_methods(em_results, panel):
    """``NowcastResults`` delegates natively to the news module (integration of wave 2)."""
    old, new = vintage_pair(panel)
    news = em_results.news(old, new, "2014Q4")
    assert isinstance(news, NewsResults)
    assert news.check_identity()
    keyword = em_results.news(old=old, new=new, target_period="2014Q4")
    assert keyword.new_nowcast == pytest.approx(news.new_nowcast)
    lc = em_results.level_contributions(target_period="2014Q4")
    assert isinstance(lc, LevelContributions)
    tr = em_results.nowcast_tracker(panel, None, "2014Q4", "2014-10-01", "2014-12-31")
    assert isinstance(tr, NowcastTracker)


def test_attach_keeps_native_methods():
    assert attach() == []
    assert getattr(NowcastResults.__dict__["news"], "_nowcastbox_news_method", False) is False
    detach()  # never removes the native methods
    assert "news" in NowcastResults.__dict__


def test_attach_duck_class(attached):
    assert sorted(attached) == ["level_contributions", "news", "nowcast_tracker"]
    assert callable(_Duck.news)  # type: ignore[attr-defined]


def test_idempotent_and_detach(attached):
    assert sorted(attach(_Duck)) == sorted(attached)
    detach(_Duck)
    assert not hasattr(_Duck, "news")
    detach(_Duck)  # nothing to remove


def test_does_not_overwrite_native_methods():
    class Custom(NowcastResults):
        def news(self):  # type: ignore[override]
            return "native"

    names = attach(Custom)
    assert "news" not in names
    assert Custom.__dict__["news"].__name__ == "news"
    detach(Custom)
    assert "news" in Custom.__dict__


def test_duck_methods_delegate(attached):
    """The attached methods delegate to the news functions (which reject non-models)."""
    duck = _Duck()
    with pytest.raises(TypeError):
        duck.news(None, None)  # type: ignore[attr-defined]
    with pytest.raises((TypeError, ValueError)):
        duck.nowcast_tracker(None)  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        duck.level_contributions()  # type: ignore[attr-defined]

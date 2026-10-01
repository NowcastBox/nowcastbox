from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nowcastbox.data_sources import (
    DEFAULT_TTL,
    DiskCache,
    default_cache_dir,
    get_default_cache,
    resolve_cache,
    set_default_cache,
)
from nowcastbox.data_sources import cache as cache_mod


@pytest.fixture
def clock(monkeypatch):
    state = {"now": 1_000_000.0}
    monkeypatch.setattr(cache_mod, "_now", lambda: state["now"])
    return state


class TestDiskCache:
    def test_roundtrip(self, tmp_path):
        cache = DiskCache(tmp_path / "c", ttl=None)
        assert cache.get("k") is None
        path = cache.set("k", "value é ✓")
        assert path.exists()
        assert path.parent == tmp_path / "c"
        assert cache.get("k") == "value é ✓"
        assert "k" in cache
        assert 1 not in cache
        assert len(cache) == 1

    def test_directory_created_lazily(self, tmp_path):
        cache = DiskCache(tmp_path / "lazy")
        assert not (tmp_path / "lazy").exists()
        assert len(cache) == 0
        assert cache.clear() == 0
        cache.set("a", "b")
        assert (tmp_path / "lazy").is_dir()

    def test_overwrite(self, tmp_path):
        cache = DiskCache(tmp_path)
        cache.set("k", "1")
        cache.set("k", "2")
        assert cache.get("k") == "2"
        assert len(cache) == 1

    def test_ttl_expiry(self, tmp_path, clock):
        cache = DiskCache(tmp_path, ttl=10)
        cache.set("k", "v")
        clock["now"] += 9.9
        assert cache.get("k") == "v"
        clock["now"] += 0.1
        assert cache.get("k") is None
        assert not cache.path_for("k").exists()  # expired entries are deleted

    def test_ttl_override_and_none(self, tmp_path, clock):
        cache = DiskCache(tmp_path, ttl=10)
        cache.set("k", "v")
        clock["now"] += 100
        assert cache.get("k", ttl=None) == "v"
        assert cache.get("k", ttl=timedelta(seconds=1000)) == "v"
        assert cache.get("k", ttl=50) is None

    def test_ttl_zero_always_expired(self, tmp_path, clock):
        cache = DiskCache(tmp_path, ttl=0)
        cache.set("k", "v")
        assert cache.get("k") is None

    def test_timedelta_ttl(self, tmp_path):
        assert DiskCache(tmp_path, ttl=timedelta(hours=1)).ttl == 3600.0

    @pytest.mark.parametrize("ttl", [-1, float("nan"), timedelta(seconds=-5)])
    def test_invalid_ttl_value(self, tmp_path, ttl):
        with pytest.raises(ValueError):
            DiskCache(tmp_path, ttl=ttl)

    @pytest.mark.parametrize("ttl", ["10", True, [1]])
    def test_invalid_ttl_type(self, tmp_path, ttl):
        with pytest.raises(TypeError):
            DiskCache(tmp_path, ttl=ttl)

    def test_invalid_key_and_value(self, tmp_path):
        cache = DiskCache(tmp_path)
        with pytest.raises(TypeError):
            cache.set("k", b"bytes")
        with pytest.raises(TypeError):
            cache.get(123)

    def test_default_on_miss(self, tmp_path):
        assert DiskCache(tmp_path).get("x", default="d") == "d"

    def test_delete(self, tmp_path):
        cache = DiskCache(tmp_path)
        cache.set("k", "v")
        assert cache.delete("k") is True
        assert cache.delete("k") is False

    def test_clear_and_prune(self, tmp_path, clock):
        cache = DiskCache(tmp_path, ttl=10)
        cache.set("a", "1")
        clock["now"] += 20
        cache.set("b", "2")
        cache.path_for("c").write_text("not json", encoding="utf-8")
        assert len(cache) == 3
        assert cache.prune() == 2  # expired "a" and corrupted "c"
        assert cache.get("b") == "2"
        assert cache.clear() == 1
        assert len(cache) == 0

    def test_corrupted_entry_is_a_miss(self, tmp_path):
        cache = DiskCache(tmp_path)
        cache.set("k", "v")
        cache.path_for("k").write_text("{broken", encoding="utf-8")
        assert cache.get("k") is None
        assert not cache.path_for("k").exists()

    @pytest.mark.parametrize(
        "content",
        [[1, 2], {"key": "k", "created": 1.0}, {"key": "k", "created": "x", "value": "v"}],
    )
    def test_malformed_entry_is_a_miss(self, tmp_path, content):
        cache = DiskCache(tmp_path)
        tmp_path.mkdir(exist_ok=True)
        cache.path_for("k").write_text(json.dumps(content), encoding="utf-8")
        assert cache.get("k") is None

    def test_foreign_key_in_file_is_a_miss(self, tmp_path):
        cache = DiskCache(tmp_path)
        cache.set("other", "v")
        cache.path_for("other").rename(cache.path_for("k"))
        assert cache.get("k") is None

    def test_atomic_write_cleans_temp_on_failure(self, tmp_path, monkeypatch):
        cache = DiskCache(tmp_path)

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(cache_mod.os, "replace", boom)
        with pytest.raises(OSError, match="disk full"):
            cache.set("k", "v")
        assert list(tmp_path.iterdir()) == []

    def test_hash_key_is_stable(self):
        assert DiskCache.hash_key("abc") == DiskCache.hash_key("abc")
        assert DiskCache.hash_key("abc") != DiskCache.hash_key("abd")

    def test_repr(self, tmp_path):
        assert "DiskCache(directory=" in repr(DiskCache(tmp_path, ttl=5))

    def test_default_directory(self, monkeypatch, tmp_path):
        monkeypatch.setenv("NOWCASTBOX_CACHE_DIR", str(tmp_path / "x"))
        assert DiskCache().directory == tmp_path / "x"
        assert DiskCache().ttl == DEFAULT_TTL

    @given(key=st.text(max_size=200), value=st.text(max_size=500))
    def test_roundtrip_property(self, tmp_path_factory, key, value):
        cache = DiskCache(tmp_path_factory.mktemp("prop"), ttl=None)
        cache.set(key, value)
        assert cache.get(key) == value


class TestDefaultCache:
    def test_default_cache_dir_precedence(self, monkeypatch, tmp_path):
        monkeypatch.setenv("NOWCASTBOX_CACHE_DIR", str(tmp_path / "explicit"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        assert default_cache_dir() == tmp_path / "explicit"
        monkeypatch.delenv("NOWCASTBOX_CACHE_DIR")
        assert default_cache_dir() == tmp_path / "xdg" / "nowcastbox"
        monkeypatch.delenv("XDG_CACHE_HOME")
        assert default_cache_dir() == Path.home() / ".cache" / "nowcastbox"

    def test_env_ttl(self, monkeypatch):
        monkeypatch.setenv("NOWCASTBOX_CACHE_TTL", "120")
        assert get_default_cache().ttl == 120.0
        monkeypatch.setenv("NOWCASTBOX_CACHE_TTL", "none")
        assert get_default_cache().ttl is None
        monkeypatch.setenv("NOWCASTBOX_CACHE_TTL", " ")
        assert get_default_cache().ttl == DEFAULT_TTL
        monkeypatch.setenv("NOWCASTBOX_CACHE_TTL", "soon")
        with pytest.raises(ValueError, match="NOWCASTBOX_CACHE_TTL"):
            get_default_cache()

    def test_env_disable(self, monkeypatch):
        monkeypatch.setenv("NOWCASTBOX_DISABLE_CACHE", "1")
        assert get_default_cache() is None

    def test_set_default_cache(self, tmp_path):
        custom = DiskCache(tmp_path, ttl=5)
        set_default_cache(custom)
        assert get_default_cache() is custom
        assert resolve_cache(None) is custom
        assert resolve_cache(True) is custom
        set_default_cache(None)
        assert get_default_cache() is None
        set_default_cache(False)
        assert resolve_cache(None) is None
        set_default_cache(True)
        assert isinstance(get_default_cache(), DiskCache)
        with pytest.raises(TypeError):
            set_default_cache("yes")

    def test_resolve_cache(self, tmp_path):
        custom = DiskCache(tmp_path)
        assert resolve_cache(custom) is custom
        assert resolve_cache(False) is None
        assert isinstance(resolve_cache(None), DiskCache)
        with pytest.raises(TypeError):
            resolve_cache("cache")

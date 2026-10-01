"""Simple on-disk cache for downloaded API responses.

Every connector in :mod:`nowcastbox.data_sources` stores the raw text of successful
HTTP responses in a :class:`DiskCache`, keyed by the request URL and its (non-secret)
query parameters. Entries expire after a time-to-live (TTL), so repeated calls inside
a session (or across sessions) do not hit the public APIs again.

Location of the default cache, in order of precedence:

1. the ``NOWCASTBOX_CACHE_DIR`` environment variable;
2. ``$XDG_CACHE_HOME/nowcastbox`` when ``XDG_CACHE_HOME`` is set;
3. ``~/.cache/nowcastbox``.

The default TTL is one day; it can be changed with ``NOWCASTBOX_CACHE_TTL`` (seconds,
``"none"`` for no expiry) or programmatically with :func:`set_default_cache`. Setting
``NOWCASTBOX_DISABLE_CACHE=1`` turns the default cache off.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import timedelta
from pathlib import Path

from nowcastbox._logging import get_logger

__all__ = [
    "DEFAULT_TTL",
    "DiskCache",
    "default_cache_dir",
    "get_default_cache",
    "resolve_cache",
    "set_default_cache",
]

logger = get_logger(__name__)

DEFAULT_TTL: float = 86_400.0
"""Default time-to-live of cache entries, in seconds (one day)."""

_FILE_SUFFIX = ".json"
_FORMAT_VERSION = 1

TTLLike = float | int | timedelta | None


def _now() -> float:
    """Current wall-clock time in seconds since the epoch (patchable in tests)."""
    return time.time()


def _ttl_seconds(ttl: TTLLike) -> float | None:
    """Validate a TTL and convert it to seconds (``None`` = never expires)."""
    if ttl is None:
        return None
    if isinstance(ttl, timedelta):
        seconds = ttl.total_seconds()
    elif isinstance(ttl, bool) or not isinstance(ttl, int | float):
        raise TypeError(f"ttl must be a number of seconds, a timedelta or None, got {ttl!r}.")
    else:
        seconds = float(ttl)
    if not seconds >= 0.0:  # also rejects NaN
        raise ValueError(f"ttl must be non-negative, got {ttl!r}.")
    return seconds


def default_cache_dir() -> Path:
    """Directory used by the default cache.

    Returns
    -------
    pathlib.Path
        ``$NOWCASTBOX_CACHE_DIR``, else ``$XDG_CACHE_HOME/nowcastbox``, else
        ``~/.cache/nowcastbox``. The directory is not created.

    Examples
    --------
    >>> from nowcastbox.data_sources import default_cache_dir
    >>> default_cache_dir().name  # doctest: +SKIP
    'nowcastbox'
    """
    explicit = os.environ.get("NOWCASTBOX_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "nowcastbox"
    return Path.home() / ".cache" / "nowcastbox"


class DiskCache:
    """Key-value cache of text payloads stored as JSON files on disk.

    Each entry lives in its own file named after the SHA-256 hash of the key, written
    atomically (temporary file + rename), so concurrent processes never observe a
    partially written entry. Corrupted files are treated as cache misses and removed.

    Parameters
    ----------
    directory : str or pathlib.Path, optional
        Cache directory (created lazily on the first write). Defaults to
        :func:`default_cache_dir`.
    ttl : float, int, datetime.timedelta or None, default 86400
        Time-to-live of entries, in seconds. ``None`` means entries never expire;
        ``0`` means every entry is already expired (useful to force a refresh while
        still recording the latest download).

    Attributes
    ----------
    directory : pathlib.Path
        Cache directory.
    ttl : float or None
        Time-to-live in seconds.

    Examples
    --------
    >>> import tempfile
    >>> from nowcastbox.data_sources import DiskCache
    >>> cache = DiskCache(tempfile.mkdtemp(), ttl=3600)
    >>> _ = cache.set("https://example.org/data?x=1", '{"a": 1}')
    >>> cache.get("https://example.org/data?x=1")
    '{"a": 1}'
    >>> "https://example.org/other" in cache
    False
    >>> cache.clear()
    1
    """

    def __init__(
        self, directory: str | os.PathLike[str] | None = None, ttl: TTLLike = DEFAULT_TTL
    ) -> None:
        self.directory = (
            Path(directory).expanduser() if directory is not None else default_cache_dir()
        )
        self.ttl = _ttl_seconds(ttl)

    def __repr__(self) -> str:
        return f"DiskCache(directory={str(self.directory)!r}, ttl={self.ttl!r})"

    # ------------------------------------------------------------------ paths
    @staticmethod
    def hash_key(key: str) -> str:
        """SHA-256 hex digest identifying ``key`` on disk.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        str
            64-character hexadecimal digest.

        Examples
        --------
        >>> from nowcastbox.data_sources import DiskCache
        >>> len(DiskCache.hash_key("abc"))
        64
        """
        if not isinstance(key, str):
            raise TypeError(f"Cache keys must be strings, got {type(key).__name__}.")
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def path_for(self, key: str) -> Path:
        """File that stores (or would store) the entry for ``key``.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        pathlib.Path
            Path inside :attr:`directory`.

        Examples
        --------
        >>> from nowcastbox.data_sources import DiskCache
        >>> DiskCache("/tmp/nc").path_for("k").suffix
        '.json'
        """
        return self.directory / f"{self.hash_key(key)}{_FILE_SUFFIX}"

    # --------------------------------------------------------------- reading
    def _read_entry(self, path: Path) -> dict | None:
        """Load an entry file; corrupted or unreadable files count as misses."""
        try:
            with path.open("r", encoding="utf-8") as fh:
                entry = json.load(fh)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            logger.warning("Discarding unreadable cache entry %s (%s).", path, exc)
            self._unlink(path)
            return None
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("value"), str)
            or not isinstance(entry.get("created"), int | float)
        ):
            logger.warning("Discarding malformed cache entry %s.", path)
            self._unlink(path)
            return None
        return entry

    def _is_expired(self, entry: dict, ttl: float | None) -> bool:
        if ttl is None:
            return False
        return (_now() - float(entry["created"])) >= ttl

    def get(
        self, key: str, default: str | None = None, *, ttl: TTLLike | str = "default"
    ) -> str | None:
        """Return the cached payload for ``key`` if present and not expired.

        Parameters
        ----------
        key : str
            Cache key.
        default : str, optional
            Value returned on a miss.
        ttl : float, timedelta, None or "default", default "default"
            TTL override for this lookup (``"default"`` uses :attr:`ttl`).

        Returns
        -------
        str or None
            Cached payload, or ``default`` on a miss. Expired entries are deleted.

        Examples
        --------
        >>> import tempfile
        >>> from nowcastbox.data_sources import DiskCache
        >>> cache = DiskCache(tempfile.mkdtemp())
        >>> cache.get("missing", default="fallback")
        'fallback'
        """
        effective = self.ttl if ttl == "default" else _ttl_seconds(ttl)  # type: ignore[arg-type]
        path = self.path_for(key)
        entry = self._read_entry(path)
        if entry is None:
            return default
        if entry.get("key") != key:  # hash collision or foreign file: treat as a miss
            return default
        if self._is_expired(entry, effective):
            logger.debug("Cache entry expired: %s", key)
            self._unlink(path)
            return default
        logger.debug("Cache hit: %s", key)
        return entry["value"]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.get(key) is not None

    # --------------------------------------------------------------- writing
    def set(self, key: str, value: str) -> Path:
        """Store ``value`` under ``key`` (atomic write).

        Parameters
        ----------
        key : str
            Cache key.
        value : str
            Payload (typically the raw text of an HTTP response).

        Returns
        -------
        pathlib.Path
            File written.

        Raises
        ------
        TypeError
            If ``value`` is not a string.

        Examples
        --------
        >>> import tempfile
        >>> from nowcastbox.data_sources import DiskCache
        >>> cache = DiskCache(tempfile.mkdtemp())
        >>> cache.set("k", "v").exists()
        True
        """
        if not isinstance(value, str):
            raise TypeError(f"Cache values must be strings, got {type(value).__name__}.")
        path = self.path_for(key)
        self.directory.mkdir(parents=True, exist_ok=True)
        entry = {"version": _FORMAT_VERSION, "key": key, "created": _now(), "value": value}
        fd, tmp_name = tempfile.mkstemp(dir=self.directory, prefix=".tmp-", suffix=_FILE_SUFFIX)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(entry, fh, ensure_ascii=False)
            os.replace(tmp_name, path)
        except BaseException:
            self._unlink(Path(tmp_name))
            raise
        logger.debug("Cache store: %s -> %s", key, path.name)
        return path

    @staticmethod
    def _unlink(path: Path) -> bool:
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except OSError as exc:  # pragma: no cover - permission problems are platform specific
            logger.warning("Could not delete cache file %s (%s).", path, exc)
            return False
        return True

    def delete(self, key: str) -> bool:
        """Remove the entry for ``key``.

        Parameters
        ----------
        key : str
            Cache key.

        Returns
        -------
        bool
            ``True`` if an entry was removed.

        Examples
        --------
        >>> import tempfile
        >>> from nowcastbox.data_sources import DiskCache
        >>> cache = DiskCache(tempfile.mkdtemp())
        >>> cache.delete("absent")
        False
        """
        return self._unlink(self.path_for(key))

    def _entry_files(self) -> list[Path]:
        if not self.directory.is_dir():
            return []
        return sorted(
            p for p in self.directory.glob(f"*{_FILE_SUFFIX}") if not p.name.startswith(".")
        )

    def clear(self) -> int:
        """Remove every entry of this cache.

        Returns
        -------
        int
            Number of entries removed.

        Examples
        --------
        >>> import tempfile
        >>> from nowcastbox.data_sources import DiskCache
        >>> DiskCache(tempfile.mkdtemp()).clear()
        0
        """
        return sum(self._unlink(p) for p in self._entry_files())

    def prune(self) -> int:
        """Remove expired and corrupted entries.

        Returns
        -------
        int
            Number of entries removed.

        Examples
        --------
        >>> import tempfile
        >>> from nowcastbox.data_sources import DiskCache
        >>> DiskCache(tempfile.mkdtemp(), ttl=0).prune()
        0
        """
        removed = 0
        for path in self._entry_files():
            entry = self._read_entry(path)
            if entry is None:
                removed += 1
            elif self._is_expired(entry, self.ttl):
                removed += self._unlink(path)
        return removed

    def __len__(self) -> int:
        """Number of entry files (expired entries included until pruned)."""
        return len(self._entry_files())


_default_cache: DiskCache | None = None
_default_cache_set: bool = False


def _env_ttl() -> TTLLike:
    raw = os.environ.get("NOWCASTBOX_CACHE_TTL")
    if raw is None or raw.strip() == "":
        return DEFAULT_TTL
    if raw.strip().lower() in {"none", "inf", "never"}:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"NOWCASTBOX_CACHE_TTL must be a number of seconds or 'none', got {raw!r}."
        ) from None


def _env_disabled() -> bool:
    return os.environ.get("NOWCASTBOX_DISABLE_CACHE", "").strip().lower() in {"1", "true", "yes"}


def get_default_cache() -> DiskCache | None:
    """Cache used by the connectors when no ``cache`` argument is given.

    Returns
    -------
    DiskCache or None
        The cache registered with :func:`set_default_cache` if any; otherwise a cache
        in :func:`default_cache_dir` with the TTL from ``NOWCASTBOX_CACHE_TTL``
        (default one day), or ``None`` when ``NOWCASTBOX_DISABLE_CACHE`` is set.

    Examples
    --------
    >>> from nowcastbox.data_sources import get_default_cache
    >>> cache = get_default_cache()  # doctest: +SKIP
    """
    if _default_cache_set:
        return _default_cache
    if _env_disabled():
        return None
    return DiskCache(default_cache_dir(), ttl=_env_ttl())


def set_default_cache(cache: DiskCache | bool | None) -> None:
    """Register the cache used by default (process-wide).

    Parameters
    ----------
    cache : DiskCache, None or bool
        A cache instance; ``None``/``False`` disables the default cache; ``True``
        restores the environment-based behaviour of :func:`get_default_cache`.

    Examples
    --------
    >>> import tempfile
    >>> from nowcastbox.data_sources import DiskCache, set_default_cache, get_default_cache
    >>> set_default_cache(DiskCache(tempfile.mkdtemp(), ttl=60))
    >>> get_default_cache().ttl
    60.0
    >>> set_default_cache(True)  # back to the environment defaults
    """
    global _default_cache, _default_cache_set
    if cache is True:
        _default_cache, _default_cache_set = None, False
    elif cache is None or cache is False:
        _default_cache, _default_cache_set = None, True
    elif isinstance(cache, DiskCache):
        _default_cache, _default_cache_set = cache, True
    else:
        raise TypeError(f"cache must be a DiskCache, a bool or None, got {type(cache).__name__}.")


def resolve_cache(cache: DiskCache | bool | None) -> DiskCache | None:
    """Turn a connector's ``cache`` argument into a cache instance (or ``None``).

    Parameters
    ----------
    cache : DiskCache, bool or None
        ``None``/``True``: the default cache; ``False``: no caching; a
        :class:`DiskCache`: used as is.

    Returns
    -------
    DiskCache or None
        Cache to use.

    Examples
    --------
    >>> from nowcastbox.data_sources import resolve_cache
    >>> resolve_cache(False) is None
    True
    """
    if cache is None or cache is True:
        return get_default_cache()
    if cache is False:
        return None
    if isinstance(cache, DiskCache):
        return cache
    raise TypeError(f"cache must be a DiskCache, a bool or None, got {type(cache).__name__}.")

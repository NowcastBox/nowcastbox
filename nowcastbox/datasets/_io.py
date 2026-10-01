"""File-level helpers of the built-in datasets: paths, metadata, checksums and caching.

Every dataset ``<name>`` has a metadata file ``metadata/<name>.yaml`` and zero or more
data files in ``data/`` (gzip-compressed CSV). The YAML records, for each data file,
its SHA-256 digest; :func:`read_table` verifies it the first time the file is read in a
process and caches the parsed table, so later loads are free.

The writers (:func:`write_csv_gz`, :func:`write_metadata`) are used by the build
scripts in ``scripts/build_datasets/``. They are deterministic: the gzip header carries
no timestamp and floats are written with a fixed format, so rebuilding from unchanged
sources reproduces byte-identical files (and digests).
"""

from __future__ import annotations

import copy
import functools
import gzip
import hashlib
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from nowcastbox._logging import get_logger
from nowcastbox.core.exceptions import NowcastDataError

__all__ = [
    "DATA_DIR",
    "FLOAT_FORMAT",
    "METADATA_DIR",
    "available_names",
    "clear_cache",
    "data_path",
    "read_metadata",
    "read_table",
    "sha256_file",
    "verify_file",
    "write_csv_gz",
    "write_metadata",
]

logger = get_logger(__name__)

DATA_DIR: Path = Path(__file__).resolve().parent / "data"
"""Directory of the shipped data files."""

METADATA_DIR: Path = Path(__file__).resolve().parent / "metadata"
"""Directory of the per-dataset metadata YAML files."""

FLOAT_FORMAT = "%.12g"
"""Float format of the shipped CSV files (12 significant digits)."""

_CHUNK = 1 << 16


def sha256_file(path: str | Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file.

    Parameters
    ----------
    path : str or pathlib.Path
        File to hash.

    Returns
    -------
    str
        64-character lowercase hexadecimal digest.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> from nowcastbox.datasets._io import sha256_file
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     p = pathlib.Path(tmp) / "x.txt"
    ...     _ = p.write_bytes(b"abc")
    ...     sha256_file(p)[:16]
    'ba7816bf8f01cfea'
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def available_names() -> list[str]:
    """Return the names of the datasets that have a metadata file.

    Returns
    -------
    list of str
        Sorted dataset names (YAML file stems).

    Examples
    --------
    >>> from nowcastbox.datasets._io import available_names
    >>> "brazil_nowcast" in available_names()
    True
    """
    return sorted(p.stem for p in METADATA_DIR.glob("*.yaml"))


@functools.cache
def _metadata_cached(name: str) -> dict[str, Any]:
    path = METADATA_DIR / f"{name}.yaml"
    if not path.is_file():
        raise KeyError(name)
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise NowcastDataError(f"Metadata file {path.name} does not hold a mapping.")
    return loaded


def read_metadata(name: str) -> dict[str, Any]:
    """Read the metadata YAML of a dataset.

    Parameters
    ----------
    name : str
        Dataset name (e.g. ``"brazil_nowcast"``).

    Returns
    -------
    dict
        Parsed metadata (a deep copy: changing it does not affect later reads).

    Raises
    ------
    ValueError
        If no dataset of that name exists.
    NowcastDataError
        If the YAML file is malformed.

    Examples
    --------
    >>> from nowcastbox.datasets._io import read_metadata
    >>> read_metadata("us_fred_md")["name"]
    'us_fred_md'
    """
    try:
        return copy.deepcopy(_metadata_cached(str(name)))
    except KeyError:
        raise ValueError(
            f"Unknown dataset {name!r}; available datasets: {available_names()}."
        ) from None


def _file_entry(name: str, key: str) -> Mapping[str, Any]:
    files = _metadata_cached(name).get("files") or {}
    entry = files.get(key)
    if not isinstance(entry, Mapping) or "path" not in entry or "sha256" not in entry:
        raise NowcastDataError(
            f"Dataset {name!r} has no data file {key!r} with 'path' and 'sha256' in its metadata."
        )
    return entry


def data_path(name: str, key: str = "data") -> Path:
    """Return the path of a data file of a dataset.

    Parameters
    ----------
    name : str
        Dataset name.
    key : str, default "data"
        Key of the file in the ``files`` section of the metadata.

    Returns
    -------
    pathlib.Path
        Absolute path inside the package ``data/`` directory.

    Raises
    ------
    ValueError
        Unknown dataset.
    NowcastDataError
        If the metadata has no such file.

    Examples
    --------
    >>> from nowcastbox.datasets._io import data_path
    >>> data_path("us_fred_md").name
    'us_fred_md.csv.gz'
    """
    if name not in available_names():
        raise ValueError(f"Unknown dataset {name!r}; available datasets: {available_names()}.")
    return DATA_DIR / str(_file_entry(name, key)["path"])


def verify_file(name: str, key: str = "data") -> Path:
    """Check that a shipped data file exists and matches its recorded SHA-256 digest.

    Parameters
    ----------
    name : str
        Dataset name.
    key : str, default "data"
        File key in the metadata ``files`` section.

    Returns
    -------
    pathlib.Path
        The verified file.

    Raises
    ------
    NowcastDataError
        If the file is missing or its digest differs from the metadata (corrupted or
        modified file: reinstall the package or rebuild the dataset with the scripts
        in ``scripts/build_datasets/``).

    Examples
    --------
    >>> from nowcastbox.datasets._io import verify_file
    >>> verify_file("nyfed").name
    'nyfed.csv.gz'
    """
    path = data_path(name, key)
    if not path.is_file():
        raise NowcastDataError(f"Data file {path.name} of dataset {name!r} is missing ({path}).")
    expected = str(_file_entry(name, key)["sha256"]).lower()
    actual = sha256_file(path)
    if actual != expected:
        raise NowcastDataError(
            f"Checksum mismatch for {path.name} of dataset {name!r}: expected sha256 "
            f"{expected}, got {actual}. The file is corrupted or was modified; reinstall "
            "nowcastbox or rebuild it with scripts/build_datasets/."
        )
    logger.debug("Verified %s (sha256 %s).", path.name, actual)
    return path


@functools.cache
def _table_cached(name: str, key: str, verify: bool) -> pd.DataFrame:
    path = verify_file(name, key) if verify else data_path(name, key)
    return pd.read_csv(path, compression="gzip", dtype={"period": str}, keep_default_na=True)


def read_table(name: str, key: str = "data", *, verify: bool = True) -> pd.DataFrame:
    """Read a shipped CSV.gz table (cached per process).

    Parameters
    ----------
    name : str
        Dataset name.
    key : str, default "data"
        File key in the metadata ``files`` section.
    verify : bool, default True
        Verify the SHA-256 digest before the first read.

    Returns
    -------
    pandas.DataFrame
        A fresh copy of the parsed table (callers may modify it).

    Raises
    ------
    NowcastDataError
        Missing file or checksum mismatch.

    Examples
    --------
    >>> from nowcastbox.datasets._io import read_table
    >>> read_table("nyfed").columns[0]
    'period'
    """
    return _table_cached(str(name), str(key), bool(verify)).copy()


def clear_cache() -> None:
    """Forget cached metadata and tables (next loads re-read and re-verify the files).

    Examples
    --------
    >>> from nowcastbox.datasets._io import clear_cache
    >>> clear_cache()
    """
    _metadata_cached.cache_clear()
    _table_cached.cache_clear()


def write_csv_gz(frame: pd.DataFrame, path: str | Path) -> str:
    """Write a table as a deterministic gzip-compressed CSV and return its SHA-256.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table to write (the index is not written: put periods in a column).
    path : str or pathlib.Path
        Destination file (``.csv.gz``).

    Returns
    -------
    str
        SHA-256 digest of the written file.

    Examples
    --------
    >>> import tempfile, pathlib, pandas as pd
    >>> from nowcastbox.datasets._io import write_csv_gz
    >>> df = pd.DataFrame({"period": ["2020-01"], "x": [1.5]})
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     a = write_csv_gz(df, pathlib.Path(tmp) / "a.csv.gz")
    ...     b = write_csv_gz(df, pathlib.Path(tmp) / "b.csv.gz")
    >>> a == b
    True
    """
    text = frame.to_csv(index=False, float_format=FLOAT_FORMAT, lineterminator="\n")
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as gz:
        gz.write(text.encode("utf-8"))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(buffer.getvalue())
    return sha256_file(target)


def write_metadata(metadata: Mapping[str, Any], path: str | Path) -> None:
    """Write a dataset metadata mapping as UTF-8 YAML (key order preserved).

    Parameters
    ----------
    metadata : mapping
        Metadata (plain Python types only).
    path : str or pathlib.Path
        Destination ``.yaml`` file.

    Examples
    --------
    >>> import tempfile, pathlib, yaml
    >>> from nowcastbox.datasets._io import write_metadata
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     p = pathlib.Path(tmp) / "m.yaml"
    ...     write_metadata({"name": "demo", "license": "MIT"}, p)
    ...     yaml.safe_load(p.read_text())["name"]
    'demo'
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(
        dict(metadata), sort_keys=False, allow_unicode=True, width=100, default_flow_style=False
    )
    target.write_text(text, encoding="utf-8")

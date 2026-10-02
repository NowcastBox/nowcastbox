"""On-disk checkpoint of a specification search (Parquet, CSV when no Parquet engine).

Every evaluated specification is one row: its draw number, its key
(:func:`~nowcastbox.selection._search_space.spec_key`), the fingerprint of the search
settings, the run status and the accuracy criteria. The file is rewritten atomically
after each evaluation, so an interrupted search resumes from the last completed one.
A checkpoint written by a search with different settings (model, data and series
metadata such as release delays, target, backtest design, ranking, metrics) is refused.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nowcastbox._logging import get_logger
from nowcastbox.core.exceptions import NowcastBoxWarning

__all__ = ["SearchCheckpoint", "describe", "fingerprint"]

logger = get_logger(__name__)


def describe(value: Any) -> Any:
    """Session-independent description of a setting (for fingerprints).

    Parameters
    ----------
    value : object
        Setting: scalar, sequence, mapping, estimator, data panel or callable.

    Returns
    -------
    object
        Nested tuples of primitives and strings; estimators by their ``repr``
        (class and parameters), functions by their qualified name, other objects by
        their class name and a ``repr`` free of memory addresses.

    Examples
    --------
    >>> describe({"b": [1, 2], "a": len})
    (('a', 'builtins.len'), ('b', (1, 2)))
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping | list | tuple):
        return _describe_container(value)
    return _describe_object(value)


def _describe_container(value: Mapping[Any, Any] | list[Any] | tuple[Any, ...]) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), describe(v)) for k, v in value.items()))
    return tuple(describe(v) for v in value)


def _describe_object(value: Any) -> str:
    if isinstance(value, type) or (callable(value) and hasattr(value, "__qualname__")):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', '')}"
    text = repr(value)
    return type(value).__qualname__ if " at 0x" in text else text


def _frame_digest(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def fingerprint(parts: Mapping[str, Any], frame: pd.DataFrame | None = None) -> str:
    """Short hash of the settings of a search (and of its data).

    Parameters
    ----------
    parts : mapping
        Settings, described with :func:`describe`.
    frame : pandas.DataFrame, optional
        Data whose content enters the hash.

    Returns
    -------
    str
        16 hexadecimal characters.

    Examples
    --------
    >>> fingerprint({"a": 1}) == fingerprint({"a": 1})
    True
    >>> fingerprint({"a": 1}) == fingerprint({"a": 2})
    False
    """
    text = repr(describe(parts))
    if frame is not None:
        text += _frame_digest(frame) + repr(list(frame.columns))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class SearchCheckpoint:
    """Read and write the evaluated specifications of a search.

    Parameters
    ----------
    path : str or pathlib.Path
        Checkpoint file. A ``.csv`` suffix writes CSV; anything else Parquet, falling
        back to CSV next to it (same name, ``.csv`` suffix) with a warning when no
        Parquet engine (``pyarrow``) is installed.
    fingerprint : str
        Fingerprint of the search settings.

    Examples
    --------
    >>> import tempfile, pathlib
    >>> folder = pathlib.Path(tempfile.mkdtemp())
    >>> store = SearchCheckpoint(folder / "search.csv", "abc")
    >>> store.save([{"draw": 0, "spec_key": "k0", "status": "ok", "rmsfe|nowcast": 1.5}])
    >>> store.load()["k0"]["rmsfe|nowcast"]
    1.5
    """

    def __init__(self, path: str | Path, fingerprint: str) -> None:
        self.path = Path(path)
        self.fingerprint = str(fingerprint)
        self._csv = self.path.suffix.lower() == ".csv"

    @property
    def csv_path(self) -> Path:
        """CSV file used when no Parquet engine is available.

        Examples
        --------
        >>> SearchCheckpoint("out/search.parquet", "x").csv_path.name
        'search.csv'
        """
        return self.path if self._csv else self.path.with_suffix(".csv")

    def _readers(self) -> list[tuple[Path, Callable[[Path], pd.DataFrame]]]:
        out: list[tuple[Path, Callable[[Path], pd.DataFrame]]] = []
        if not self._csv:
            out.append((self.path, pd.read_parquet))
        out.append((self.csv_path, lambda p: pd.read_csv(p, keep_default_na=True)))
        return out

    def _read(self) -> pd.DataFrame | None:
        for path, reader in self._readers():
            if not path.exists():
                continue
            try:
                return reader(path)
            except ImportError:
                logger.info("No Parquet engine to read %s; trying CSV.", path)
        return None

    def load(self) -> dict[str, dict[str, Any]]:
        """Evaluated specifications by key (empty when there is no checkpoint yet).

        Returns
        -------
        dict
            ``spec_key`` to the stored record.

        Raises
        ------
        ValueError
            If the checkpoint was written by a search with other settings.

        Examples
        --------
        >>> SearchCheckpoint("missing.parquet", "x").load()
        {}
        """
        frame = self._read()
        if frame is None or frame.empty:
            return {}
        stored = set(frame["fingerprint"].astype(str)) if "fingerprint" in frame else set()
        if stored != {self.fingerprint}:
            raise ValueError(
                f"The checkpoint {self.path} was written by a search with different settings "
                "(model, data, target, backtest, ranking or metrics); use another file or "
                "delete it."
            )
        records: dict[str, dict[str, Any]] = {}
        for row in frame.to_dict("records"):
            record = {str(k): _clean(v) for k, v in row.items() if k != "fingerprint"}
            records[str(record["spec_key"])] = record
        return records

    def save(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Write every record (atomically replacing the file).

        Parameters
        ----------
        records : sequence of mapping
            Evaluated specifications (``draw``, ``spec_key``, status and criteria).

        Examples
        --------
        >>> store.save([])  # doctest: +SKIP
        """
        frame = pd.DataFrame.from_records([dict(r) for r in records])
        frame["fingerprint"] = self.fingerprint
        if not self._csv:
            try:
                self._write(frame, self.path, parquet=True)
                return
            except ImportError:
                warnings.warn(
                    "No Parquet engine (pyarrow) is installed; the search checkpoint is "
                    f"written as CSV to {self.csv_path}.",
                    NowcastBoxWarning,
                    stacklevel=2,
                )
                self.path = self.csv_path
                self._csv = True
        self._write(frame, self.csv_path, parquet=False)

    @staticmethod
    def _write(frame: pd.DataFrame, path: Path, *, parquet: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        if parquet:
            frame.to_parquet(tmp, index=False)
        else:
            frame.to_csv(tmp, index=False)
        os.replace(tmp, path)


def _clean(value: Any) -> Any:
    """Python scalar of a stored cell (NumPy scalars unwrapped)."""
    return value.item() if isinstance(value, np.generic) else value

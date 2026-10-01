"""Versioned nowcast snapshots (innovation I10).

Every production run can be frozen in a **snapshot**: a timestamped folder holding the
spec, a fingerprint (SHA-256) of the data vintage, the nowcast table, the model
parameters, the news against the previous snapshot and the HTML report. A
:class:`SnapshotStore` writes snapshots atomically and lists, loads and compares them —
the snapshot history *is* the nowcast history of a production model.

Layout of a snapshot folder (``<root>/<YYYYmmdd-HHMMSS-ffffff>_<name>/``)::

    manifest.json   id, name, target, creation time, vintage, data hash, model,
                    headline nowcast, previous snapshot, file list, extra metadata
    spec.yaml       normalised spec of the run
    nowcast.csv     nowcast table (observed / in_sample / out_of_sample / bands)
    panel.csv       model-ready data vintage (used for news against later runs)
    params.json     model hyper-parameters and estimates (JSON-safe)
    news.csv        news decomposition against the previous snapshot (optional)
    report.html     HTML report (optional)
    *.csv / *.txt   further tables and texts (density, diagnostics, backtest ...)
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import json
import math
import os
import re
import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nowcastbox._logging import get_logger

__all__ = ["MANIFEST", "Snapshot", "SnapshotDiff", "SnapshotStore", "headline_period", "jsonable"]

logger = get_logger(__name__)

MANIFEST = "manifest.json"
"""Name of the manifest file that marks a folder as a snapshot."""

_FORMAT_VERSION = 1
_MAX_ARRAY = 10_000
_ID_RE = re.compile(r"^\d{8}-\d{6}-\d{6}_[\w.\-]+$")


def jsonable(value: Any, *, max_array: int = _MAX_ARRAY) -> Any:
    """JSON-safe copy of ``value`` (numpy, pandas, dates, enums, NaN -> ``None``).

    Parameters
    ----------
    value : object
        Arbitrary (nested) value.
    max_array : int, default 10000
        Arrays with more elements are replaced by a ``"<array shape=...>"`` note.

    Returns
    -------
    object
        Value built from ``dict``, ``list``, ``str``, ``int``, ``float``, ``bool`` and
        ``None`` only.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> jsonable({"a": np.float64(1.5), "b": (1, 2), "p": pd.Period("2020Q1"), "n": np.nan})
    {'a': 1.5, 'b': [1, 2], 'p': '2020Q1', 'n': None}
    """
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, enum.Enum):
        return jsonable(value.value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(k): jsonable(v, max_array=max_array) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [jsonable(v, max_array=max_array) for v in value]
    return _jsonable_object(value, max_array)


def _jsonable_object(value: Any, max_array: int) -> Any:
    if isinstance(value, np.ndarray):
        if value.size > max_array:
            return f"<array shape={value.shape}>"
        return jsonable(value.tolist(), max_array=max_array)
    if isinstance(value, pd.Series):
        return {str(k): jsonable(v, max_array=max_array) for k, v in value.items()}
    if isinstance(value, pd.DataFrame):
        if value.size > max_array:
            return f"<DataFrame shape={value.shape}>"
        return {str(c): jsonable(value[c], max_array=max_array) for c in value.columns}
    if isinstance(value, pd.Period | pd.Timestamp | _dt.date | Path):
        return str(value)
    return str(value)


# ---------------------------------------------------------------------------- snapshot
def _read_indexed_csv(path: Path, freq: str | None) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, float_precision="round_trip").astype(float)
    if freq:
        frame.index = pd.PeriodIndex([str(v) for v in frame.index], freq=freq)
        frame.index.name = "period"
    return frame


def _estimate(frame: pd.DataFrame) -> pd.Series:
    """In-sample estimate where the target is observed, out-of-sample elsewhere."""
    return frame["in_sample"].combine_first(frame["out_of_sample"])


@dataclasses.dataclass(frozen=True)
class Snapshot:
    """One stored nowcast run (a folder with a manifest).

    Parameters
    ----------
    path : pathlib.Path
        Snapshot folder.
    manifest : dict
        Parsed ``manifest.json``.

    Examples
    --------
    >>> import tempfile, pandas as pd
    >>> store = SnapshotStore(tempfile.mkdtemp())
    >>> table = pd.DataFrame(
    ...     {"observed": [1.0, None], "in_sample": [0.9, None], "out_of_sample": [None, 1.2]},
    ...     index=pd.period_range("2020Q1", periods=2, freq="Q"),
    ... )
    >>> snap = store.write(name="demo", target="gdp", nowcast=table, vintage="2020-05-01")
    >>> snap.headline_period, snap.headline
    ('2020Q2', 1.2)
    >>> snap.nowcast().index.freqstr
    'Q-DEC'
    """

    path: Path
    manifest: dict[str, Any]

    @property
    def id(self) -> str:
        """Snapshot identifier (folder name)."""
        return str(self.manifest["id"])

    @property
    def name(self) -> str:
        """Name of the spec that produced the snapshot."""
        return str(self.manifest.get("name", ""))

    @property
    def target(self) -> str:
        """Target series."""
        return str(self.manifest.get("target", ""))

    @property
    def created(self) -> pd.Timestamp:
        """Creation time (UTC)."""
        return pd.Timestamp(self.manifest["created"])

    @property
    def vintage(self) -> str | None:
        """Information-set date of the run."""
        return self.manifest.get("vintage")

    @property
    def data_hash(self) -> str | None:
        """SHA-256 of the data vintage."""
        return self.manifest.get("data_hash")

    @property
    def model(self) -> str | None:
        """Estimator name."""
        return self.manifest.get("model")

    @property
    def headline_period(self) -> str | None:
        """Target period of the headline nowcast."""
        return self.manifest.get("headline_period")

    @property
    def headline(self) -> float:
        """Headline nowcast (``NaN`` when unavailable)."""
        value = self.manifest.get("headline")
        return float("nan") if value is None else float(value)

    @property
    def previous(self) -> str | None:
        """Identifier of the snapshot that was the latest when this one was written."""
        return self.manifest.get("previous")

    @property
    def files(self) -> list[str]:
        """Files stored in the snapshot."""
        return sorted(p.name for p in self.path.iterdir() if p.is_file())

    def file(self, name: str) -> Path | None:
        """Path of a stored file (``None`` when absent).

        Parameters
        ----------
        name : str
            File name (e.g. ``"report.html"``).

        Returns
        -------
        pathlib.Path or None
            Existing file.

        Examples
        --------
        >>> import tempfile, pandas as pd
        >>> store = SnapshotStore(tempfile.mkdtemp())
        >>> table = pd.DataFrame(
        ...     {"observed": [None], "in_sample": [None], "out_of_sample": [0.5]},
        ...     index=pd.period_range("2020Q1", periods=1, freq="Q"),
        ... )
        >>> snap = store.write(name="demo", target="gdp", nowcast=table)
        >>> snap.file("nowcast.csv").name, snap.file("report.html")
        ('nowcast.csv', None)
        """
        path = self.path / name
        return path if path.is_file() else None

    @property
    def report_path(self) -> Path | None:
        """The HTML report, if stored."""
        return self.file("report.html")

    def nowcast(self) -> pd.DataFrame:
        """Nowcast table on the target's PeriodIndex.

        Returns
        -------
        pandas.DataFrame
            Columns ``observed``, ``in_sample``, ``out_of_sample`` and extras.

        Examples
        --------
        >>> import tempfile, pandas as pd
        >>> store = SnapshotStore(tempfile.mkdtemp())
        >>> table = pd.DataFrame(
        ...     {"observed": [None], "in_sample": [None], "out_of_sample": [0.5]},
        ...     index=pd.period_range("2020Q1", periods=1, freq="Q"),
        ... )
        >>> store.write(name="demo", target="gdp", nowcast=table).nowcast()["out_of_sample"].iloc[0]
        np.float64(0.5)
        """
        return _read_indexed_csv(self.path / "nowcast.csv", self.manifest.get("target_frequency"))

    def estimate(self, period: object = None) -> float:
        """Estimate of the target for ``period`` (default: the headline period).

        Parameters
        ----------
        period : period-like, optional
            Target period.

        Returns
        -------
        float
            In-sample estimate where the target is observed, nowcast otherwise;
            ``NaN`` if the period is not in the table.

        Examples
        --------
        >>> import tempfile, pandas as pd
        >>> store = SnapshotStore(tempfile.mkdtemp())
        >>> table = pd.DataFrame(
        ...     {"observed": [1.0, None], "in_sample": [0.9, None], "out_of_sample": [None, 1.2]},
        ...     index=pd.period_range("2020Q1", periods=2, freq="Q"),
        ... )
        >>> snap = store.write(name="demo", target="gdp", nowcast=table)
        >>> snap.estimate("2020Q1"), snap.estimate("2021Q1")
        (0.9, nan)
        """
        period = self.headline_period if period is None else period
        if period is None:
            return float("nan")
        table = self.nowcast()
        key = _key(table, period)
        if key not in table.index:
            return float("nan")
        return float(_estimate(table).loc[key])

    def panel(self) -> pd.DataFrame | None:
        """Model-ready data vintage (``None`` if not stored).

        Returns
        -------
        pandas.DataFrame or None
            Values on the base PeriodIndex.

        Examples
        --------
        >>> import tempfile, pandas as pd
        >>> store = SnapshotStore(tempfile.mkdtemp())
        >>> table = pd.DataFrame(
        ...     {"observed": [None], "in_sample": [None], "out_of_sample": [0.5]},
        ...     index=pd.period_range("2020Q1", periods=1, freq="Q"),
        ... )
        >>> store.write(name="demo", target="gdp", nowcast=table).panel() is None
        True
        """
        path = self.path / "panel.csv"
        if not path.is_file():
            return None
        return _read_indexed_csv(path, self.manifest.get("base_frequency"))

    def table(self, name: str) -> pd.DataFrame | None:
        """A stored CSV table by stem (``"news"``, ``"density"`` ...).

        Parameters
        ----------
        name : str
            File stem.

        Returns
        -------
        pandas.DataFrame or None
            The table (first column as index), ``None`` if absent.

        Examples
        --------
        >>> import tempfile, pandas as pd
        >>> store = SnapshotStore(tempfile.mkdtemp())
        >>> table = pd.DataFrame(
        ...     {"observed": [None], "in_sample": [None], "out_of_sample": [0.5]},
        ...     index=pd.period_range("2020Q1", periods=1, freq="Q"),
        ... )
        >>> snap = store.write(
        ...     name="demo", target="gdp", nowcast=table, tables={"extra": pd.DataFrame({"a": [1]})}
        ... )
        >>> snap.table("extra")["a"].tolist()
        [1]
        """
        path = self.path / f"{name}.csv"
        return (
            pd.read_csv(path, index_col=0, float_precision="round_trip") if path.is_file() else None
        )

    def news(self) -> pd.DataFrame | None:
        """News table against the previous snapshot (``None`` if not computed)."""
        return self.table("news")

    def params(self) -> dict[str, Any]:
        """Stored model parameters (``params.json``; empty when absent)."""
        path = self.path / "params.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def spec(self) -> dict[str, Any]:
        """Stored spec (``spec.yaml``; empty when absent)."""
        path = self.path / "spec.yaml"
        if not path.is_file():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def summary(self) -> str:
        """Human-readable description.

        Returns
        -------
        str
            Multi-line text.

        Examples
        --------
        >>> import tempfile, pandas as pd
        >>> store = SnapshotStore(tempfile.mkdtemp())
        >>> table = pd.DataFrame(
        ...     {"observed": [None], "in_sample": [None], "out_of_sample": [0.5]},
        ...     index=pd.period_range("2020Q1", periods=1, freq="Q"),
        ... )
        >>> print(
        ...     store.write(name="demo", target="gdp", nowcast=table).summary()
        ... )  # doctest: +ELLIPSIS
        Snapshot ..._demo
        ...
          headline   : 2020Q1 = 0.5
        ...
        """
        m = self.manifest
        lines = [
            f"Snapshot {self.id}",
            f"  name       : {self.name}",
            f"  target     : {self.target}",
            f"  created    : {m.get('created')}",
            f"  vintage    : {self.vintage}",
            f"  model      : {self.model}",
            f"  data hash  : {(self.data_hash or '')[:16]}",
            f"  headline   : {self.headline_period} = {_fmt(self.headline)}",
            f"  previous   : {self.previous}",
        ]
        news = m.get("news")
        if isinstance(news, Mapping):
            lines.append(
                f"  news       : {_fmt(news.get('old_nowcast'))} -> {_fmt(news.get('new_nowcast'))}"
                f" (against {news.get('against')})"
            )
        for note in m.get("warnings") or []:
            lines.append(f"  warning    : {note}")
        lines.append(f"  files      : {', '.join(self.files)}")
        return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if math.isnan(number) else f"{number:.6g}"


# ---------------------------------------------------------------------------- diff
def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping) and value:
        out: dict[str, Any] = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: value}


def _changed(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    a, b = _flatten(old), _flatten(new)
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


@dataclasses.dataclass(frozen=True)
class SnapshotDiff:
    """Comparison of two snapshots (``old`` -> ``new``).

    Parameters
    ----------
    old, new : Snapshot
        Compared snapshots.
    nowcast : pandas.DataFrame
        Estimates of both snapshots per target period (``old``, ``new``, ``change``).
    data_revisions : dict
        Counts of ``new``, ``revised`` and ``removed`` data cells between the stored
        vintages (empty when a panel is missing).
    spec_changes : dict
        ``{dotted key: (old, new)}`` of the spec.
    param_changes : dict
        ``{dotted key: (old, new)}`` of the model hyper-parameters.

    Examples
    --------
    >>> import tempfile, pandas as pd
    >>> store = SnapshotStore(tempfile.mkdtemp())
    >>> idx = pd.period_range("2020Q1", periods=1, freq="Q")
    >>> mk = lambda v: pd.DataFrame(
    ...     {"observed": [None], "in_sample": [None], "out_of_sample": [v]}, index=idx
    ... )
    >>> a = store.write(name="demo", target="gdp", nowcast=mk(0.5))
    >>> b = store.write(name="demo", target="gdp", nowcast=mk(0.7))
    >>> round(store.diff(a, b).headline_change, 6)
    0.2
    """

    old: Snapshot
    new: Snapshot
    nowcast: pd.DataFrame
    data_revisions: dict[str, int]
    spec_changes: dict[str, tuple[Any, Any]]
    param_changes: dict[str, tuple[Any, Any]]

    @property
    def data_changed(self) -> bool:
        """Whether the data vintages differ (by hash)."""
        return self.old.data_hash != self.new.data_hash

    @property
    def headline_change(self) -> float:
        """Change of the estimate for the new headline period."""
        period = self.new.headline_period
        if period is None:
            return float("nan")
        key = _key(self.nowcast, period)
        if key not in self.nowcast.index:
            return float("nan")
        return float(self.nowcast["change"].loc[key])

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary.

        Returns
        -------
        dict
            Identifiers, headline change, data change flag and counts, spec and
            parameter changes, and the nowcast comparison.

        Examples
        --------
        >>> import tempfile, pandas as pd
        >>> store = SnapshotStore(tempfile.mkdtemp())
        >>> idx = pd.period_range("2020Q1", periods=1, freq="Q")
        >>> t = pd.DataFrame({"observed": [None], "in_sample": [None], "out_of_sample": [1.0]}, idx)
        >>> d = store.diff(
        ...     store.write(name="a", target="y", nowcast=t),
        ...     store.write(name="a", target="y", nowcast=t),
        ... ).to_dict()
        >>> d["headline_change"], d["data_changed"]
        (0.0, False)
        """
        table = self.nowcast.copy()
        table.index = table.index.astype(str)
        return jsonable(
            {
                "old": self.old.id,
                "new": self.new.id,
                "headline_period": self.new.headline_period,
                "headline_change": self.headline_change,
                "data_changed": self.data_changed,
                "data_revisions": self.data_revisions,
                "spec_changes": self.spec_changes,
                "param_changes": self.param_changes,
                "nowcast": table.to_dict(orient="index"),
            }
        )

    def summary(self, n_periods: int = 6) -> str:
        """Human-readable comparison.

        Parameters
        ----------
        n_periods : int, default 6
            Last target periods shown.

        Returns
        -------
        str
            Multi-line text.

        Examples
        --------
        >>> import tempfile, pandas as pd
        >>> store = SnapshotStore(tempfile.mkdtemp())
        >>> idx = pd.period_range("2020Q1", periods=1, freq="Q")
        >>> t = pd.DataFrame({"observed": [None], "in_sample": [None], "out_of_sample": [1.0]}, idx)
        >>> s = store.diff(
        ...     store.write(name="a", target="y", nowcast=t),
        ...     store.write(name="a", target="y", nowcast=t),
        ... ).summary()
        >>> "headline change" in s
        True
        """
        lines = [
            f"Snapshot diff {self.old.id} -> {self.new.id}",
            f"  vintage          : {self.old.vintage} -> {self.new.vintage}",
            f"  data changed     : {'yes' if self.data_changed else 'no'}",
        ]
        if self.data_revisions:
            counts = ", ".join(f"{k} {v}" for k, v in self.data_revisions.items())
            lines.append(f"  data cells       : {counts}")
        lines.append(
            f"  headline change  : {self.new.headline_period} "
            f"{_fmt(self.old.estimate(self.new.headline_period))} -> "
            f"{_fmt(self.new.headline)} ({_fmt(self.headline_change)})"
        )
        for title, changes in (("spec", self.spec_changes), ("params", self.param_changes)):
            for key, (a, b) in changes.items():
                lines.append(f"  {title} {key}: {a!r} -> {b!r}")
        lines.append("")
        lines.append(self.nowcast.tail(n_periods).to_string(float_format=lambda v: f"{v:.6g}"))
        return "\n".join(lines)


def _panel_revisions(old: pd.DataFrame | None, new: pd.DataFrame | None) -> dict[str, int]:
    if old is None or new is None:
        return {}
    if _freqstr(old) != _freqstr(new):
        return {}
    index = old.index.union(new.index)
    columns = old.columns.union(new.columns)
    a = old.reindex(index=index, columns=columns)
    b = new.reindex(index=index, columns=columns)
    was, now = a.notna().to_numpy(), b.notna().to_numpy()
    both = was & now
    revised = both & ~np.isclose(a.to_numpy(), b.to_numpy(), rtol=0.0, atol=1e-12, equal_nan=True)
    return {
        "new": int((now & ~was).sum()),
        "revised": int(revised.sum()),
        "removed": int((was & ~now).sum()),
    }


# ---------------------------------------------------------------------------- store
class SnapshotStore:
    """Directory of versioned nowcast snapshots.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory (created on the first write).

    Examples
    --------
    >>> import tempfile, pandas as pd
    >>> store = SnapshotStore(tempfile.mkdtemp())
    >>> idx = pd.period_range("2020Q1", periods=2, freq="Q")
    >>> for value in (0.5, 0.6):
    ...     table = pd.DataFrame(
    ...         {"observed": [1.0, None], "in_sample": [1.1, None], "out_of_sample": [None, value]},
    ...         index=idx,
    ...     )
    ...     _ = store.write(name="gdp", target="gdp", nowcast=table)
    >>> len(store), store.history()["value"].tolist()
    (2, [0.5, 0.6])
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    def __repr__(self) -> str:
        return f"SnapshotStore({str(self.root)!r}, n={len(self)})"

    def __len__(self) -> int:
        return len(self.snapshots())

    def __iter__(self) -> Iterator[Snapshot]:
        return iter(self.snapshots())

    def __contains__(self, snapshot_id: object) -> bool:
        return isinstance(snapshot_id, str) and (self.root / snapshot_id / MANIFEST).is_file()

    # ------------------------------------------------------------------ write
    def write(
        self,
        *,
        name: str,
        target: str,
        nowcast: pd.DataFrame,
        vintage: str | None = None,
        data_hash: str | None = None,
        model: str | None = None,
        headline_period: object = None,
        spec: Mapping[str, Any] | None = None,
        panel: pd.DataFrame | None = None,
        params: Mapping[str, Any] | None = None,
        news: pd.DataFrame | None = None,
        news_info: Mapping[str, Any] | None = None,
        tables: Mapping[str, pd.DataFrame] | None = None,
        texts: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        warnings: list[str] | None = None,
        created: pd.Timestamp | None = None,
    ) -> Snapshot:
        """Write a new snapshot atomically (temporary folder renamed at the end).

        Parameters
        ----------
        name : str
            Spec name (snapshots are grouped by it).
        target : str
            Target series.
        nowcast : pandas.DataFrame
            Nowcast table with ``observed``, ``in_sample`` and ``out_of_sample``
            columns on a PeriodIndex.
        vintage : str, optional
            Information-set date.
        data_hash : str, optional
            Fingerprint of the data vintage.
        model : str, optional
            Estimator name.
        headline_period : period-like, optional
            Period of the headline nowcast (default: the first period with an
            out-of-sample value after the last observation).
        spec : mapping, optional
            Normalised spec (``spec.yaml``).
        panel : pandas.DataFrame, optional
            Model-ready data on the base PeriodIndex (``panel.csv``).
        params : mapping, optional
            Model parameters (``params.json``, JSON-safe conversion).
        news : pandas.DataFrame, optional
            News table (``news.csv``).
        news_info : mapping, optional
            News summary stored in the manifest (old/new nowcast, reference).
        tables : mapping of str to DataFrame, optional
            Extra tables stored as ``<key>.csv``.
        texts : mapping of str to str, optional
            Extra text files, by file name (e.g. ``{"report.html": html}``).
        metadata : mapping, optional
            Extra manifest entries.
        warnings : list of str, optional
            Warnings raised during the run.
        created : pandas.Timestamp, optional
            Creation time (default: now, UTC).

        Returns
        -------
        Snapshot
            The stored snapshot.

        Raises
        ------
        ValueError
            If ``nowcast`` lacks the required columns or a PeriodIndex, or a file
            name is invalid.

        Examples
        --------
        >>> import tempfile, pandas as pd
        >>> store = SnapshotStore(tempfile.mkdtemp())
        >>> table = pd.DataFrame(
        ...     {"observed": [None], "in_sample": [None], "out_of_sample": [0.5]},
        ...     index=pd.period_range("2020Q1", periods=1, freq="Q"),
        ... )
        >>> snap = store.write(name="demo", target="gdp", nowcast=table, texts={"note.txt": "hi"})
        >>> snap.files
        ['manifest.json', 'note.txt', 'nowcast.csv']
        """
        _check_table(nowcast)
        stamp = _utc(created)
        snapshot_id = self._new_id(stamp, name)
        previous = self.latest(name)
        tmp = self.root / f".tmp-{snapshot_id}"
        tmp.mkdir(parents=True)
        try:
            files = _write_files(tmp, nowcast, spec, panel, params, news, tables, texts)
            period = headline_period if headline_period is not None else _headline_of(nowcast)
            manifest = {
                "format_version": _FORMAT_VERSION,
                "id": snapshot_id,
                "name": name,
                "target": target,
                "created": stamp.isoformat(),
                "vintage": vintage,
                "data_hash": data_hash,
                "model": model,
                "target_frequency": _freqstr(nowcast),
                "base_frequency": None if panel is None else _freqstr(panel),
                "headline_period": None if period is None else str(period),
                "headline": _value_at(nowcast, period),
                "previous": None if previous is None else previous.id,
                "news": None if news_info is None else dict(news_info),
                "warnings": list(warnings or []),
                "files": sorted([*files, MANIFEST]),
                **dict(metadata or {}),
            }
            (tmp / MANIFEST).write_text(json.dumps(jsonable(manifest), indent=2), encoding="utf-8")
            final = self.root / snapshot_id
            os.replace(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        logger.info("snapshot %s written to %s", snapshot_id, final)
        return self.load(snapshot_id)

    def _new_id(self, stamp: pd.Timestamp, name: str) -> str:
        if not re.fullmatch(r"[\w.\-]+", name):
            raise ValueError(f"Snapshot name must use letters, digits, '_', '-', '.': {name!r}.")
        while True:
            candidate = f"{stamp:%Y%m%d-%H%M%S-%f}_{name}"
            if (
                not (self.root / candidate).exists()
                and not (self.root / f".tmp-{candidate}").exists()
            ):
                return candidate
            stamp = stamp + pd.Timedelta(microseconds=1)

    # ------------------------------------------------------------------ read
    def snapshots(self, name: str | None = None) -> list[Snapshot]:
        """Stored snapshots, oldest first.

        Parameters
        ----------
        name : str, optional
            Keep only snapshots of this spec name.

        Returns
        -------
        list of Snapshot
            Sorted by creation time.

        Examples
        --------
        >>> import tempfile
        >>> SnapshotStore(tempfile.mkdtemp()).snapshots()
        []
        """
        if not self.root.is_dir():
            return []
        out = []
        for folder in self.root.iterdir():
            manifest = folder / MANIFEST
            if folder.is_dir() and not folder.name.startswith(".") and manifest.is_file():
                try:
                    snap = Snapshot(folder, json.loads(manifest.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    logger.warning("skipping unreadable snapshot %s", folder)
                    continue
                if name is None or snap.name == name:
                    out.append(snap)
        return sorted(out, key=lambda s: (s.created, s.id))

    def ids(self, name: str | None = None) -> list[str]:
        """Identifiers of the stored snapshots, oldest first.

        Parameters
        ----------
        name : str, optional
            Spec name filter.

        Returns
        -------
        list of str
            Snapshot ids.

        Examples
        --------
        >>> import tempfile
        >>> SnapshotStore(tempfile.mkdtemp()).ids()
        []
        """
        return [s.id for s in self.snapshots(name)]

    def latest(self, name: str | None = None) -> Snapshot | None:
        """Most recent snapshot (``None`` when the store is empty).

        Parameters
        ----------
        name : str, optional
            Spec name filter.

        Returns
        -------
        Snapshot or None
            Latest snapshot.

        Examples
        --------
        >>> import tempfile
        >>> SnapshotStore(tempfile.mkdtemp()).latest() is None
        True
        """
        snaps = self.snapshots(name)
        return snaps[-1] if snaps else None

    def load(self, ref: str, name: str | None = None) -> Snapshot:
        """Load a snapshot by id, unique id prefix, ``"latest"`` or ``"latest~k"``.

        Parameters
        ----------
        ref : str
            Snapshot reference: an id, a unique prefix of one, ``"latest"`` (or
            ``"latest~1"`` for the one before it, ...).
        name : str, optional
            Spec name filter used to resolve ``latest`` and prefixes.

        Returns
        -------
        Snapshot
            The snapshot.

        Raises
        ------
        KeyError
            If no (or more than one) snapshot matches.

        Examples
        --------
        >>> import tempfile
        >>> SnapshotStore(tempfile.mkdtemp()).load("latest")
        Traceback (most recent call last):
        ...
        KeyError: "No snapshot matches 'latest' ..."
        """
        folder = self.root / ref
        if _ID_RE.match(ref) and (folder / MANIFEST).is_file():
            return Snapshot(folder, json.loads((folder / MANIFEST).read_text(encoding="utf-8")))
        snaps = self.snapshots(name)
        match = re.fullmatch(r"latest(?:~(\d+))?", ref)
        if match:
            back = int(match.group(1) or 0)
            if back < len(snaps):
                return snaps[-1 - back]
            raise KeyError(f"No snapshot matches {ref!r} in {self.root} ({len(snaps)} stored).")
        found = [s for s in snaps if s.id.startswith(ref)]
        if len(found) == 1:
            return found[0]
        if not found:
            raise KeyError(f"No snapshot matches {ref!r} in {self.root}.")
        raise KeyError(f"Ambiguous snapshot reference {ref!r}: {[s.id for s in found]}.")

    def to_frame(self, name: str | None = None) -> pd.DataFrame:
        """Table of the stored snapshots.

        Parameters
        ----------
        name : str, optional
            Spec name filter.

        Returns
        -------
        pandas.DataFrame
            Index ``id``; columns ``name``, ``target``, ``created``, ``vintage``,
            ``model``, ``headline_period``, ``headline``, ``data_hash`` (first 12
            characters) and ``report``.

        Examples
        --------
        >>> import tempfile
        >>> list(SnapshotStore(tempfile.mkdtemp()).to_frame().columns)[:3]
        ['name', 'target', 'created']
        """
        rows = [
            {
                "id": s.id,
                "name": s.name,
                "target": s.target,
                "created": s.created,
                "vintage": s.vintage,
                "model": s.model,
                "headline_period": s.headline_period,
                "headline": s.headline,
                "data_hash": (s.data_hash or "")[:12],
                "report": s.report_path is not None,
            }
            for s in self.snapshots(name)
        ]
        columns = ["name", "target", "created", "vintage", "model", "headline_period"]
        columns += ["headline", "data_hash", "report"]
        return pd.DataFrame(rows, columns=["id", *columns]).set_index("id")

    def history(self, name: str | None = None, period: object = None) -> pd.DataFrame:
        """Nowcast history: the estimate of a target period across snapshots.

        Parameters
        ----------
        name : str, optional
            Spec name filter.
        period : period-like, optional
            Target period (default: each snapshot's headline period).

        Returns
        -------
        pandas.DataFrame
            Index ``id``; columns ``created``, ``vintage``, ``period``, ``value``
            (in-sample estimate once the target is observed), ``observed`` and
            ``data_hash``.

        Examples
        --------
        >>> import tempfile
        >>> SnapshotStore(tempfile.mkdtemp()).history().empty
        True
        """
        rows = []
        for snap in self.snapshots(name):
            target = snap.headline_period if period is None else str(period)
            observed = float("nan")
            if target is not None:
                table = snap.nowcast()
                key = _key(table, target)
                if key in table.index:
                    observed = float(table["observed"].loc[key])
            rows.append(
                {
                    "id": snap.id,
                    "created": snap.created,
                    "vintage": snap.vintage,
                    "period": target,
                    "value": snap.estimate(target),
                    "observed": observed,
                    "data_hash": (snap.data_hash or "")[:12],
                }
            )
        columns = ["id", "created", "vintage", "period", "value", "observed", "data_hash"]
        return pd.DataFrame(rows, columns=columns).set_index("id")

    def diff(self, old: Snapshot | str, new: Snapshot | str) -> SnapshotDiff:
        """Compare two snapshots.

        Parameters
        ----------
        old, new : Snapshot or str
            Snapshots or references (see :meth:`load`).

        Returns
        -------
        SnapshotDiff
            Nowcast changes per period, data revisions, spec and parameter changes.

        Raises
        ------
        KeyError
            If a reference does not resolve.

        Examples
        --------
        >>> import tempfile, pandas as pd
        >>> store = SnapshotStore(tempfile.mkdtemp())
        >>> idx = pd.period_range("2020Q1", periods=1, freq="Q")
        >>> t = pd.DataFrame({"observed": [None], "in_sample": [None], "out_of_sample": [1.0]}, idx)
        >>> _ = store.write(name="a", target="y", nowcast=t)
        >>> _ = store.write(name="a", target="y", nowcast=t + 1)
        >>> store.diff("latest~1", "latest").nowcast["change"].tolist()
        [1.0]
        """
        a = old if isinstance(old, Snapshot) else self.load(old)
        b = new if isinstance(new, Snapshot) else self.load(new)
        ta, tb = a.nowcast(), b.nowcast()
        if _freqstr(ta) != _freqstr(tb):
            raise ValueError("The snapshots have targets of different frequencies.")
        ea, eb = _estimate(ta), _estimate(tb)
        table = pd.DataFrame({"old": ea, "new": eb})
        table["change"] = table["new"] - table["old"]
        return SnapshotDiff(
            old=a,
            new=b,
            nowcast=table,
            data_revisions=_panel_revisions(a.panel(), b.panel()),
            spec_changes=_changed(a.spec(), b.spec()),
            param_changes=_changed(
                a.params().get("model_params", {}), b.params().get("model_params", {})
            ),
        )

    def delete(self, ref: str) -> None:
        """Remove a snapshot folder.

        Parameters
        ----------
        ref : str
            Snapshot reference (see :meth:`load`).

        Raises
        ------
        KeyError
            If the reference does not resolve.

        Examples
        --------
        >>> import tempfile, pandas as pd
        >>> store = SnapshotStore(tempfile.mkdtemp())
        >>> idx = pd.period_range("2020Q1", periods=1, freq="Q")
        >>> t = pd.DataFrame({"observed": [None], "in_sample": [None], "out_of_sample": [1.0]}, idx)
        >>> store.delete(store.write(name="a", target="y", nowcast=t).id)
        >>> len(store)
        0
        """
        shutil.rmtree(self.load(ref).path)


# ---------------------------------------------------------------------------- helpers
def _check_table(nowcast: pd.DataFrame) -> None:
    if not isinstance(nowcast.index, pd.PeriodIndex):
        raise ValueError("The nowcast table needs a PeriodIndex.")
    missing = {"observed", "in_sample", "out_of_sample"} - set(nowcast.columns)
    if missing:
        raise ValueError(f"The nowcast table lacks the columns {sorted(missing)}.")


def _utc(created: pd.Timestamp | None) -> pd.Timestamp:
    stamp = pd.Timestamp.now(tz="UTC") if created is None else pd.Timestamp(created)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _freqstr(frame: pd.DataFrame) -> str | None:
    index = frame.index
    return index.freqstr if isinstance(index, pd.PeriodIndex) else None


def _key(frame: pd.DataFrame, period: object) -> Any:
    """``period`` as a Period at the frequency of ``frame``'s PeriodIndex."""
    return pd.Period(str(period), freq=_freqstr(frame))


def headline_period(nowcast: pd.DataFrame) -> pd.Period | None:
    """Period of the headline nowcast of a nowcast table.

    Parameters
    ----------
    nowcast : pandas.DataFrame
        Table with ``observed`` and ``out_of_sample`` columns on a PeriodIndex.

    Returns
    -------
    pandas.Period or None
        First period with an out-of-sample estimate after the last observation of the
        target (``None`` if there is none).

    Examples
    --------
    >>> import pandas as pd
    >>> table = pd.DataFrame(
    ...     {"observed": [1.0, None, None], "out_of_sample": [None, 0.4, 0.5]},
    ...     index=pd.period_range("2020Q1", periods=3, freq="Q"),
    ... )
    >>> str(headline_period(table))
    '2020Q2'
    """
    observed = nowcast["observed"].dropna()
    out = nowcast["out_of_sample"].dropna()
    if not observed.empty:
        out = out[out.index > observed.index[-1]]
    return None if out.empty else out.index[0]


_headline_of = headline_period


def _value_at(nowcast: pd.DataFrame, period: object) -> float | None:
    if period is None:
        return None
    key = _key(nowcast, period)
    if key not in nowcast.index:
        return None
    value = _estimate(nowcast).loc[key]
    return None if pd.isna(value) else float(value)


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    out = frame.copy()
    if isinstance(out.index, pd.PeriodIndex):
        out.index = out.index.astype(str)
    out.to_csv(path, float_format="%.17g")


def _write_files(
    folder: Path,
    nowcast: pd.DataFrame,
    spec: Mapping[str, Any] | None,
    panel: pd.DataFrame | None,
    params: Mapping[str, Any] | None,
    news: pd.DataFrame | None,
    tables: Mapping[str, pd.DataFrame] | None,
    texts: Mapping[str, str] | None,
) -> list[str]:
    frames: dict[str, pd.DataFrame] = {"nowcast": nowcast}
    if panel is not None:
        frames["panel"] = panel
    if news is not None:
        frames["news"] = news
    frames.update(tables or {})
    files = []
    for stem, frame in frames.items():
        _write_table(folder / _safe_name(f"{stem}.csv"), frame)
        files.append(f"{stem}.csv")
    if spec is not None:
        text = yaml.safe_dump(jsonable(spec), sort_keys=False, allow_unicode=True)
        (folder / "spec.yaml").write_text(text, encoding="utf-8")
        files.append("spec.yaml")
    if params is not None:
        (folder / "params.json").write_text(
            json.dumps(jsonable(params), indent=2), encoding="utf-8"
        )
        files.append("params.json")
    for file_name, text in (texts or {}).items():
        (folder / _safe_name(file_name)).write_text(text, encoding="utf-8")
        files.append(file_name)
    return files


def _safe_name(file_name: str) -> str:
    if not re.fullmatch(r"[\w.\-]+", file_name) or file_name in (MANIFEST, ".", ".."):
        raise ValueError(f"Invalid snapshot file name {file_name!r}.")
    return file_name

"""Shared helpers of the dataset build scripts.

The build scripts download each dataset from its **primary source** and write

* ``nowcastbox/datasets/data/<file>.csv.gz`` - deterministic gzip CSV (see
  :func:`nowcastbox.datasets._io.write_csv_gz`), and
* ``nowcastbox/datasets/metadata/<name>.yaml`` - provenance, license, citation, the
  SHA-256 digest of every data file and the series legend.

They are *not* part of the installed package and need network access. Run them from the
repository root, e.g. ``python3 scripts/build_datasets/build_all.py``.
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # allow running without an editable install
    sys.path.insert(0, str(REPO_ROOT))

from nowcastbox.datasets._io import (  # noqa: E402
    DATA_DIR,
    METADATA_DIR,
    write_csv_gz,
    write_metadata,
)
from nowcastbox.datasets.dataset import LEGEND_COLUMNS  # noqa: E402
from nowcastbox.preprocessing.transforms import TRANSFORM_CODES, get_transform  # noqa: E402

LOG = logging.getLogger("build_datasets")

_CODE_BY_SPEC = {get_transform(name).to_spec(): code for code, name in TRANSFORM_CODES.items()}


def setup_logging() -> None:
    """Configure console logging for the build scripts."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


def legacy_code(transform: str | None) -> str:
    """Return the legacy numeric code (0-7) equivalent to a named transform, or ``""``."""
    if transform is None:
        return "0"
    spec = get_transform(transform).to_spec()
    code = _CODE_BY_SPEC.get(spec)
    return "" if code is None else str(code)


def wide_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn a monthly PeriodIndex frame into the shipped table (``period`` column first)."""
    if not isinstance(frame.index, pd.PeriodIndex) or frame.index.freqstr != "M":
        raise ValueError("Expected a monthly PeriodIndex.")
    full = pd.period_range(frame.index.min(), frame.index.max(), freq="M")
    out = frame.reindex(full)
    table = out.reset_index(drop=True)
    table.insert(0, "period", [str(p) for p in full])
    return table


def legend_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalise legend rows to plain YAML-friendly dicts in :data:`LEGEND_COLUMNS` order."""
    out: list[dict[str, Any]] = []
    for row in rows:
        rec: dict[str, Any] = {}
        for col in LEGEND_COLUMNS:
            value = row.get(col, "")
            if col == "delay_days":
                rec[col] = None if value is None or value == "" else int(value)
            elif col == "transform":
                rec[col] = None if value is None else str(value)
            else:
                rec[col] = "" if value is None else str(value)
        for key, value in row.items():
            if key not in rec:
                rec[key] = value
        out.append(rec)
    return out


def summarise(frame: pd.DataFrame, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summary block of the metadata (sample and counts)."""
    freqs: dict[str, int] = {}
    for row in rows:
        freqs[str(row["frequency"])] = freqs.get(str(row["frequency"]), 0) + 1
    return {
        "n_series": len(rows),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "frequencies": freqs,
    }


def write_dataset(
    name: str,
    frame: pd.DataFrame,
    rows: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
    *,
    extra_files: Mapping[str, tuple[str, pd.DataFrame]] | None = None,
) -> Path:
    """Write the data file(s) and the metadata YAML of one dataset.

    Parameters
    ----------
    name : str
        Dataset name (YAML stem and default data file stem).
    frame : pandas.DataFrame
        Wide monthly panel (PeriodIndex) in levels, columns in legend order.
    rows : sequence of mapping
        Legend rows.
    meta : mapping
        Provenance fields (title, description, loader, target, url, sources, license,
        citation, notes, ...). ``files``, ``summary``, ``series`` and ``built`` are
        filled in here.
    extra_files : mapping, optional
        ``{key: (file_name, long_table)}`` written next to the main file.

    Returns
    -------
    pathlib.Path
        Path of the metadata YAML.
    """
    records = legend_records(rows)
    names = [r["name"] for r in records]
    if list(frame.columns) != names:
        raise ValueError(f"{name}: frame columns and legend rows differ.")
    files: dict[str, Any] = {}
    data_file = f"{name}.csv.gz"
    table = wide_table(frame)
    files["data"] = {
        "path": data_file,
        "sha256": write_csv_gz(table, DATA_DIR / data_file),
        "format": "csv.gz (wide; period column YYYY-MM; quarterly values in 3rd month)",
        "n_rows": len(table),
        "n_columns": len(table.columns) - 1,
    }
    for key, (file_name, long_table) in (extra_files or {}).items():
        files[key] = {
            "path": file_name,
            "sha256": write_csv_gz(long_table, DATA_DIR / file_name),
            "format": "csv.gz (long)",
            "n_rows": len(long_table),
        }
    out: dict[str, Any] = dict(meta)
    out["name"] = name
    out["built"] = {
        "date": dt.date.today().isoformat(),
        "script": f"scripts/build_datasets/{meta.get('build_script', f'build_{name}.py')}",
        **dict(meta.get("built", {})),
    }
    out.pop("build_script", None)
    out["files"] = files
    out["summary"] = summarise(frame, records)
    out["series"] = records
    path = METADATA_DIR / f"{name}.yaml"
    write_metadata(out, path)
    LOG.info("Wrote %s (%d series) and %s", data_file, len(records), path.name)
    return path


def write_metadata_only(name: str, meta: Mapping[str, Any]) -> Path:
    """Write a metadata YAML for a dataset without its own data file."""
    out = {"name": name, **dict(meta)}
    out.setdefault("built", {"date": dt.date.today().isoformat()})
    path = METADATA_DIR / f"{name}.yaml"
    write_metadata(out, path)
    LOG.info("Wrote %s", path.name)
    return path

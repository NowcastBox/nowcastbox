"""Tests of ``nowcastbox.datasets._io`` (paths, checksums, caching, writers)."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.datasets import _io

SHIPPED = ["brazil_nowcast", "brazil_vintages", "nyfed", "us_fred_md"]


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    payload = b"nowcastbox" * 10_000
    path = tmp_path / "x.bin"
    path.write_bytes(payload)
    assert _io.sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_available_names_lists_every_yaml() -> None:
    names = _io.available_names()
    assert names == sorted(names)
    expected = {
        "brazil_calendar",
        "brazil_nowcast",
        "brazil_vintages",
        "nyfed",
        "simulated_dfm",
        "us_fred_md",
        "us_grs_like",
    }
    assert expected <= set(names)


def test_read_metadata_returns_independent_copies() -> None:
    first = _io.read_metadata("nyfed")
    first["title"] = "changed"
    first["series"].clear()
    second = _io.read_metadata("nyfed")
    assert second["title"] != "changed"
    assert len(second["series"]) == 29


def test_read_metadata_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        _io.read_metadata("does_not_exist")


def test_data_path_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        _io.data_path("nope")


def test_data_path_unknown_file_key() -> None:
    with pytest.raises(NowcastDataError, match="no data file 'other'"):
        _io.data_path("nyfed", "other")


def test_dataset_without_files_section() -> None:
    with pytest.raises(NowcastDataError, match="no data file"):
        _io.data_path("simulated_dfm")


@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_files_match_their_checksums(name: str) -> None:
    path = _io.verify_file(name)
    assert path.is_file()
    assert path.parent == _io.DATA_DIR


def test_releases_file_shared_by_calendar_and_vintages() -> None:
    cal = _io.read_metadata("brazil_calendar")["files"]["releases"]
    vin = _io.read_metadata("brazil_vintages")["files"]["releases"]
    assert cal == vin
    assert _io.verify_file("brazil_calendar", "releases") == _io.verify_file(
        "brazil_vintages", "releases"
    )


def test_grs_like_shares_fred_md_data_file() -> None:
    grs = _io.read_metadata("us_grs_like")["files"]["data"]
    fred = _io.read_metadata("us_fred_md")["files"]["data"]
    assert grs["path"] == fred["path"] and grs["sha256"] == fred["sha256"]


def test_shipped_data_stay_small() -> None:
    total = sum(p.stat().st_size for p in _io.DATA_DIR.glob("*.csv.gz"))
    assert total < 5 * 1024**2


def test_checksum_mismatch_is_detected(isolated_store: Path) -> None:
    path = _io.data_path("nyfed")
    raw = bytearray(path.read_bytes())
    raw[-10] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(NowcastDataError, match="Checksum mismatch"):
        _io.verify_file("nyfed")
    with pytest.raises(NowcastDataError, match="Checksum mismatch"):
        _io.read_table("nyfed")


def test_verify_false_skips_the_checksum(isolated_store: Path) -> None:
    path = _io.data_path("nyfed")
    text = gzip.decompress(path.read_bytes()).decode()
    path.write_bytes(gzip.compress(text.replace("period", "period", 1).encode()))
    with pytest.raises(NowcastDataError):
        _io.read_table("nyfed")
    table = _io.read_table("nyfed", verify=False)
    assert table.columns[0] == "period"


def test_missing_data_file(isolated_store: Path) -> None:
    _io.data_path("nyfed").unlink()
    with pytest.raises(NowcastDataError, match="missing"):
        _io.verify_file("nyfed")


def test_metadata_must_be_a_mapping(isolated_store: Path) -> None:
    (isolated_store / "metadata" / "broken.yaml").write_text("- just\n- a list\n")
    assert "broken" in _io.available_names()
    with pytest.raises(NowcastDataError, match="mapping"):
        _io.read_metadata("broken")


def test_read_table_is_cached_and_returns_copies() -> None:
    first = _io.read_table("nyfed")
    first.iloc[0, 1] = -1.0
    second = _io.read_table("nyfed")
    assert second.iloc[0, 1] != -1.0
    assert _io._table_cached.cache_info().hits >= 1


def test_clear_cache_forces_reverification(isolated_store: Path) -> None:
    _io.read_table("nyfed")
    path = _io.data_path("nyfed")
    path.write_bytes(path.read_bytes() + b"\x00")
    _io.read_table("nyfed")  # cached: no re-read
    _io.clear_cache()
    with pytest.raises(NowcastDataError):
        _io.read_table("nyfed")


def test_write_csv_gz_is_deterministic(tmp_path: Path) -> None:
    frame = pd.DataFrame({"period": ["2020-01", "2020-02"], "x": [1.0 / 3.0, float("nan")]})
    a = _io.write_csv_gz(frame, tmp_path / "a.csv.gz")
    b = _io.write_csv_gz(frame, tmp_path / "sub" / "b.csv.gz")
    assert a == b
    back = pd.read_csv(tmp_path / "a.csv.gz")
    assert back["x"].iloc[0] == pytest.approx(1.0 / 3.0, rel=1e-11)
    assert back["x"].isna().iloc[1]


def test_write_metadata_round_trip(tmp_path: Path) -> None:
    meta = {"name": "demo", "license": "MIT", "series": [{"name": "ação", "delay_days": 3}]}
    path = tmp_path / "deep" / "demo.yaml"
    _io.write_metadata(meta, path)
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == meta
    assert list(yaml.safe_load(path.read_text(encoding="utf-8"))) == ["name", "license", "series"]

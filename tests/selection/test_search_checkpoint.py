"""Tests of the on-disk checkpoint of the specification search."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.exceptions import NowcastBoxWarning
from nowcastbox.models import TwoStepDFM
from nowcastbox.selection._search_checkpoint import SearchCheckpoint, describe, fingerprint

RECORDS = [
    {"draw": 0, "spec_key": "k0", "status": "ok", "error": "", "rmsfe|nowcast": 1.5},
    {"draw": 3, "spec_key": "k3", "status": "failed", "error": "boom", "rmsfe|nowcast": np.nan},
]


class _Opaque:
    pass


class TestDescribe:
    def test_primitives_and_containers(self):
        assert describe(None) is None
        assert describe(np.int64(3)) == 3
        assert describe([1, (2, "a")]) == (1, (2, "a"))
        assert describe({"b": 1, "a": 2}) == (("a", 2), ("b", 1))

    def test_estimators_functions_and_objects(self):
        assert describe(TwoStepDFM(n_factors=2)).startswith("TwoStepDFM(n_factors=2")
        assert describe(TwoStepDFM) == "nowcastbox.models.two_step.TwoStepDFM"
        assert describe(np.mean).endswith("mean")
        assert describe(_Opaque()) == "_Opaque"
        assert describe(pd.Timestamp("2020-01-01")) == "Timestamp('2020-01-01 00:00:00')"

    def test_fingerprint_depends_on_data(self):
        frame = pd.DataFrame({"a": [1.0, 2.0]})
        changed = pd.DataFrame({"a": [1.0, 2.5]})
        assert fingerprint({"x": 1}, frame) == fingerprint({"x": 1}, frame.copy())
        assert fingerprint({"x": 1}, frame) != fingerprint({"x": 1}, changed)
        assert len(fingerprint({"x": 1})) == 16


class TestRoundTrip:
    @pytest.mark.parametrize("name", ["search.parquet", "search.csv"])
    def test_save_and_load(self, tmp_path, name):
        pytest.importorskip("pyarrow")
        store = SearchCheckpoint(tmp_path / "sub" / name, "fp")
        assert store.load() == {}
        store.save(RECORDS)
        loaded = SearchCheckpoint(tmp_path / "sub" / name, "fp").load()
        assert set(loaded) == {"k0", "k3"}
        assert loaded["k0"]["rmsfe|nowcast"] == 1.5
        assert loaded["k3"]["draw"] == 3
        assert isinstance(loaded["k3"]["draw"], int)
        assert np.isnan(loaded["k3"]["rmsfe|nowcast"])
        assert "fingerprint" not in loaded["k0"]
        assert not list((tmp_path / "sub").glob("*.tmp"))

    def test_other_settings_are_refused(self, tmp_path):
        SearchCheckpoint(tmp_path / "s.csv", "fp").save(RECORDS)
        with pytest.raises(ValueError, match="different settings"):
            SearchCheckpoint(tmp_path / "s.csv", "other").load()

    def test_file_without_fingerprint_is_refused(self, tmp_path):
        pd.DataFrame(RECORDS).to_csv(tmp_path / "s.csv", index=False)
        with pytest.raises(ValueError, match="different settings"):
            SearchCheckpoint(tmp_path / "s.csv", "fp").load()

    def test_empty_file(self, tmp_path):
        pd.DataFrame({"spec_key": []}).to_csv(tmp_path / "s.csv", index=False)
        assert SearchCheckpoint(tmp_path / "s.csv", "fp").load() == {}

    def test_csv_path(self):
        assert SearchCheckpoint("a/b.parquet", "x").csv_path.name == "b.csv"
        assert SearchCheckpoint("a/b.CSV", "x").csv_path.name == "b.CSV"


class TestWithoutParquetEngine:
    def test_falls_back_to_csv(self, tmp_path, monkeypatch):
        def no_engine(*args, **kwargs):
            raise ImportError("no pyarrow")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", no_engine)
        monkeypatch.setattr(pd, "read_parquet", no_engine)
        store = SearchCheckpoint(tmp_path / "s.parquet", "fp")
        with pytest.warns(NowcastBoxWarning, match="written as CSV"):
            store.save(RECORDS)
        assert store.path == tmp_path / "s.csv"
        assert (tmp_path / "s.csv").exists()
        assert not (tmp_path / "s.parquet").exists()
        store.save(RECORDS[:1])  # already in CSV mode: no new warning
        fresh = SearchCheckpoint(tmp_path / "s.parquet", "fp")
        assert set(fresh.load()) == {"k0"}

    def test_parquet_file_unreadable_without_engine(self, tmp_path, monkeypatch):
        pytest.importorskip("pyarrow")
        SearchCheckpoint(tmp_path / "s.parquet", "fp").save(RECORDS)

        def no_engine(*args, **kwargs):
            raise ImportError("no pyarrow")

        monkeypatch.setattr(pd, "read_parquet", no_engine)
        assert SearchCheckpoint(tmp_path / "s.parquet", "fp").load() == {}

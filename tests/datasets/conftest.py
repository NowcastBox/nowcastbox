"""Fixtures of the dataset tests."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nowcastbox.datasets import _io


@pytest.fixture
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Copy the shipped data/metadata to a temporary directory and point ``_io`` at it."""
    data = tmp_path / "data"
    meta = tmp_path / "metadata"
    shutil.copytree(_io.DATA_DIR, data)
    shutil.copytree(_io.METADATA_DIR, meta)
    monkeypatch.setattr(_io, "DATA_DIR", data)
    monkeypatch.setattr(_io, "METADATA_DIR", meta)
    _io.clear_cache()
    yield tmp_path
    _io.clear_cache()


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    """Every test starts and ends with an empty dataset cache."""
    _io.clear_cache()
    yield
    _io.clear_cache()


def make_legend(names: list[str], frequencies: list[str]) -> pd.DataFrame:
    """Minimal valid legend for ``names``."""
    n = len(names)
    return pd.DataFrame(
        {
            "name": names,
            "description": [f"series {c}" for c in names],
            "source": ["test"] * n,
            "source_code": [str(i) for i in range(n)],
            "frequency": frequencies,
            "transform": ["dlog"] * n,
            "legacy_code": [""] * n,
            "delay_days": [30] * n,
            "blocks": ["global;real"] * n,
            "category": ["hard"] * n,
            "units": ["index"] * n,
        }
    )


@pytest.fixture
def small_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two monthly series and one quarterly series in levels, with a legend."""
    idx = pd.period_range("2020-01", periods=12, freq="M")
    rng = np.random.default_rng(0)
    q = np.full(12, np.nan)
    q[[2, 5, 8, 11]] = [100.0, 101.0, 102.5, 103.0]
    data = pd.DataFrame(
        {
            "a": 100 + rng.random(12).cumsum(),
            "b": 50 + rng.random(12).cumsum(),
            "gdp": q,
        },
        index=idx,
    )
    legend = make_legend(["a", "b", "gdp"], ["M", "M", "Q"])
    legend.loc[1, "blocks"] = "global;nominal"
    legend.loc[1, "category"] = "financial"
    return data, legend

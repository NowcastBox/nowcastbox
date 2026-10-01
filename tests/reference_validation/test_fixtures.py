"""Housekeeping of the reference fixtures (size budget, provenance, clean skipping)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.reference_validation._helpers import (
    FIXTURES,
    abs_correlations,
    column_signs,
    fixture_path,
    max_abs,
    max_rel,
    read_json,
)

pytestmark = pytest.mark.reference_validation

MAX_TOTAL_BYTES = 3 * 1024 * 1024


def test_fixture_budget() -> None:
    if not FIXTURES.exists():  # pragma: no cover - fixtures not checked out
        pytest.skip("no fixtures")
    total = sum(p.stat().st_size for p in FIXTURES.rglob("*") if p.is_file())
    assert total < MAX_TOTAL_BYTES


def test_provenance() -> None:
    session = read_json("r/session.json")
    assert session["nowcasting_version"] == "1.1.2"
    timings = read_json("r/timings.json")
    assert {"usgdp_2s", "usgdp_2s_agg", "nyfed_em"} <= set(timings)
    sm = read_json("statsmodels/nyfed_dfmq.json")
    assert sm["loglike"] < 0


def test_missing_fixture_skips() -> None:
    with pytest.raises(pytest.skip.Exception):
        fixture_path("r/does_not_exist.json")


def test_helpers_edge_cases() -> None:
    nan = np.full(3, np.nan)
    assert max_abs(nan, nan) == 0.0
    assert max_rel(nan, np.zeros(3)) == 0.0
    assert max_rel([2.0, np.nan], [1.0, 5.0]) == 1.0
    assert column_signs(np.zeros((2, 1)), np.ones((2, 1))).tolist() == [1.0]
    x = np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.5]])
    assert abs_correlations(x, -x) == pytest.approx([1.0, 1.0])
    frame = pd.DataFrame({"a": [1.0, np.nan]})
    assert max_abs(frame, frame) == 0.0

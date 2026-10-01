"""Tests of :mod:`nowcastbox.news.revisions`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from nowcastbox.news import compare_vintages, data_revisions


def test_compare_shapes():
    with pytest.raises(ValueError, match="shapes"):
        compare_vintages(np.zeros(2), np.zeros(3))


@given(
    old=arrays(np.float64, (6, 3), elements=st.one_of(st.just(np.nan), st.floats(-3, 3))),
    new=arrays(np.float64, (6, 3), elements=st.one_of(st.just(np.nan), st.floats(-3, 3))),
)
def test_masks_partition_cells(old, new):
    d = compare_vintages(old, new)
    obs_old, obs_new = ~np.isnan(old), ~np.isnan(new)
    assert not (d.released & d.revised).any()
    assert not (d.released & d.removed).any()
    assert not (d.revised & d.removed).any()
    assert (d.released == (obs_new & ~obs_old)).all()
    assert (d.removed == (obs_old & ~obs_new)).all()
    assert d.n_released + d.n_removed + d.n_revised <= old.size


def test_data_revisions_table():
    idx = pd.period_range("2020-01", periods=3, freq="M")
    old = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [1.0, np.nan, np.nan]}, index=idx)
    new = pd.DataFrame(
        {"a": [1.0, 2.5, 3.0], "b": [np.nan, 4.0, np.nan], "c": [1.0, 1.0, 1.0]},
        index=idx.append(pd.PeriodIndex([], freq="M")),
    )
    table = data_revisions(old, new)
    assert table[["series", "kind"]].values.tolist() == [
        ["a", "revised"],
        ["a", "released"],
        ["b", "removed"],
        ["b", "released"],
    ]
    assert table.loc[0, "revision"] == pytest.approx(0.5)


def test_data_revisions_empty():
    idx = pd.period_range("2020-01", periods=2, freq="M")
    frame = pd.DataFrame({"a": [1.0, 2.0]}, index=idx)
    out = data_revisions(frame, frame)
    assert out.empty
    assert list(out.columns) == ["series", "period", "kind", "old", "new", "revision"]

"""Tests of the accuracy metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nowcastbox.evaluation import (
    METRICS,
    accuracy_by_horizon,
    bias,
    forecast_errors,
    loss_values,
    mae,
    metric_by_horizon,
    mse,
    relative_rmsfe,
    rmsfe,
)

finite = st.floats(-1e3, 1e3)


def test_basic_metrics() -> None:
    errors = np.array([1.0, -2.0, np.nan, 3.0])
    assert mse(errors) == pytest.approx(14 / 3)
    assert rmsfe(errors) == pytest.approx(np.sqrt(14 / 3))
    assert mae(errors) == pytest.approx(2.0)
    assert bias(pd.Series(errors)) == pytest.approx(2 / 3)
    assert METRICS["n"](errors) == 3.0


def test_empty_inputs_give_nan() -> None:
    for func in (mse, rmsfe, mae, bias):
        assert np.isnan(func([np.nan]))
        assert np.isnan(func([]))


def test_forecast_errors() -> None:
    np.testing.assert_allclose(forecast_errors([1.0, np.nan], [0.0, 1.0]), [1.0, np.nan])
    with pytest.raises(ValueError, match="different lengths"):
        forecast_errors([1.0], [1.0, 2.0])


def test_relative_rmsfe() -> None:
    assert relative_rmsfe([1.0, np.nan, 3.0], [2.0, 5.0, np.nan]) == pytest.approx(0.5)
    assert np.isnan(relative_rmsfe([1.0], [np.nan]))
    assert np.isnan(relative_rmsfe([1.0], [0.0]))
    with pytest.raises(ValueError, match="different lengths"):
        relative_rmsfe([1.0], [1.0, 2.0])


def test_loss_values() -> None:
    np.testing.assert_allclose(loss_values([-2.0, 1.0]), [4.0, 1.0])
    np.testing.assert_allclose(loss_values([-2.0, 1.0], "absolute"), [2.0, 1.0])
    np.testing.assert_allclose(loss_values([-2.0, 1.0], lambda e: e**4), [16.0, 1.0])
    with pytest.raises(ValueError, match="one loss per error"):
        loss_values([1.0, 2.0], lambda e: e.sum())
    with pytest.raises(ValueError, match="loss must be"):
        loss_values([1.0], "pinball")


@given(st.lists(finite, min_size=1, max_size=50))
def test_metric_inequalities(values: list[float]) -> None:
    assert abs(bias(values)) <= mae(values) + 1e-9
    assert mae(values) <= rmsfe(values) + 1e-9
    assert rmsfe(values) ** 2 == pytest.approx(mse(values), rel=1e-9, abs=1e-9)


@given(st.lists(finite, min_size=1, max_size=30), st.floats(0.1, 10))
def test_relative_rmsfe_scale(values: list[float], scale: float) -> None:
    ref = np.asarray(values) * scale
    if rmsfe(values) > 1e-100:  # squares of tinier values underflow to subnormals
        assert relative_rmsfe(values, ref) == pytest.approx(1 / scale)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["A", "A", "A", "B", "B", "B"],
            "h": [0, 0, 1, 0, 0, 1],
            "error": [1.0, -1.0, 2.0, 0.5, np.nan, 3.0],
        }
    )


def test_metric_by_horizon(frame) -> None:
    table = metric_by_horizon(frame, "rmsfe", "h")
    assert table.columns.tolist() == ["A", "B"]
    assert table.index.name == "h"
    np.testing.assert_allclose(table.to_numpy(), [[1.0, 0.5], [2.0, 3.0]])
    pooled = metric_by_horizon(frame, "n", None)
    assert pooled.loc["all"].tolist() == [3.0, 2.0]


def test_metric_by_horizon_errors(frame) -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        metric_by_horizon(frame, "mape", "h")
    with pytest.raises(KeyError, match="not in the frame"):
        metric_by_horizon(frame, "rmsfe", "days")


def test_accuracy_by_horizon(frame) -> None:
    table = accuracy_by_horizon(frame, "h", ("rmsfe", "bias"))
    assert table.columns.names == ["metric", "model"]
    assert table[("bias", "A")].tolist() == [0.0, 2.0]
    with pytest.raises(ValueError, match="At least one"):
        accuracy_by_horizon(frame, "h", ())

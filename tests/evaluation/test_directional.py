"""Tests of the forecast directional accuracy (FDA) and the Pesaran-Timmermann test."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.evaluation import (
    DIRECTIONAL_METRICS,
    PesaranTimmermannResult,
    accuracy_by_horizon,
    directional_accuracy,
    directional_changes,
    metric_by_horizon,
    pesaran_timmermann,
)

NAN = float("nan")


# ---------------------------------------------------------------------- FDA
class TestDirectionalAccuracy:
    def test_hand_computed(self) -> None:
        actual = [1.4, 0.5, 2.0, 1.2]
        forecast = [1.3, 0.8, 1.5, 0.9]
        # changes: (+0.4, +0.3) hit, (-0.5, -0.2) hit, (+1.0, +0.5) hit, (+0.2, -0.1) miss
        assert directional_accuracy(actual, forecast, [1.0] * 4) == 0.75

    def test_perfect_forecast(self, rng) -> None:
        actual = rng.standard_normal(50)
        previous = rng.standard_normal(50)
        assert directional_accuracy(actual, actual, previous) == 1.0

    def test_antisymmetry(self, rng) -> None:
        actual, forecast, previous = rng.standard_normal((3, 60))
        mirrored = 2 * previous - forecast  # same distance, opposite direction
        fda = directional_accuracy(actual, forecast, previous)
        assert directional_accuracy(actual, mirrored, previous) == pytest.approx(1 - fda)

    def test_zero_change_is_a_miss(self) -> None:
        assert directional_accuracy([1.0, 2.0], [2.0, 1.0], [1.0, 1.0]) == 0.0

    def test_nans_are_dropped(self) -> None:
        assert directional_accuracy([2.0, NAN, 0.0], [3.0, 1.0, 1.0], [1.0, 1.0, NAN]) == 1.0
        assert math.isnan(directional_accuracy([NAN], [1.0], [0.0]))
        assert math.isnan(directional_accuracy([], [], []))

    def test_changes(self) -> None:
        dy, dyhat = directional_changes([2.0, 1.0], [0.0, 3.0], [1.0, 2.0])
        assert dy.tolist() == [1.0, -1.0]
        assert dyhat.tolist() == [-1.0, 1.0]
        with pytest.raises(ValueError, match="different lengths"):
            directional_changes([1.0], [1.0, 2.0], [1.0])

    def test_registry(self) -> None:
        assert DIRECTIONAL_METRICS["fda"] is directional_accuracy


class TestDirectionalByHorizon:
    @pytest.fixture
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "model": ["A", "A", "A", "B", "B", "B"],
                "h": [0, 0, 1, 0, 0, 1],
                "actual": [2.0, 0.0, 2.0, 2.0, 0.0, 2.0],
                "forecast": [1.5, 0.5, 0.5, 0.5, 1.5, 1.5],
                "previous_actual": 1.0,
            }
        ).assign(error=lambda f: f["actual"] - f["forecast"])

    def test_metric_by_horizon(self, frame) -> None:
        table = metric_by_horizon(frame, "fda", "h")
        assert table.to_dict() == {"A": {0: 1.0, 1: 0.0}, "B": {0: 0.0, 1: 1.0}}
        renamed = frame.rename(columns={"previous_actual": "prev"})
        pooled = metric_by_horizon(renamed, "fda", None, previous_col="prev")
        assert pooled.loc["all", "A"] == pytest.approx(2 / 3)

    def test_accuracy_by_horizon_mixes_metrics(self, frame) -> None:
        table = accuracy_by_horizon(frame, "h", ("rmsfe", "fda", "n"))
        assert table.columns.get_level_values("metric").unique().tolist() == [
            "rmsfe",
            "fda",
            "n",
        ]
        assert table.loc[0, ("fda", "A")] == 1.0

    def test_missing_previous_column(self, frame) -> None:
        with pytest.raises(KeyError, match="previous_actual"):
            metric_by_horizon(frame.drop(columns="previous_actual"), "fda", "h")

    def test_unknown_metric_lists_directional(self, frame) -> None:
        with pytest.raises(ValueError, match="fda"):
            metric_by_horizon(frame, "hit", "h")


# ---------------------------------------------------------------------- Pesaran-Timmermann
def _pt_manual(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Statistic of Pesaran & Timmermann (1992, eq. 3-6) written out directly."""
    n = actual.size
    y, x = actual > 0, forecast > 0
    p_hat = np.mean(y == x)
    py, px = y.mean(), x.mean()
    p_star = py * px + (1 - py) * (1 - px)
    v_p = p_star * (1 - p_star) / n
    v_star = (
        (2 * py - 1) ** 2 * px * (1 - px) / n
        + (2 * px - 1) ** 2 * py * (1 - py) / n
        + 4 * py * px * (1 - py) * (1 - px) / n**2
    )
    return float((p_hat - p_star) / np.sqrt(v_p - v_star))


class TestPesaranTimmermann:
    def test_matches_formula(self, rng) -> None:
        change = rng.standard_normal(80)
        forecast = 0.4 * change + rng.standard_normal(80)
        res = pesaran_timmermann(change + 3.0, forecast + 3.0, np.full(80, 3.0))
        assert res.statistic == pytest.approx(_pt_manual(change, forecast))
        assert res.pvalue == pytest.approx(0.5 * math.erfc(res.statistic / math.sqrt(2)))
        assert res.n_obs == 80
        assert res.alternative == "greater"

    def test_hand_example(self) -> None:
        actual = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=float)
        forecast = np.array([1, 1, -1, -1, -1, 1, 1, -1], dtype=float)
        res = pesaran_timmermann(actual, forecast, np.zeros(8))
        assert res.hit_rate == 0.75
        assert res.expected_hit_rate == 0.5
        # V(P) = 0.25/8, V(P*) = 4 * 1/16 / 64 -> S = 0.25 / sqrt(1/32 - 1/256)
        assert res.statistic == pytest.approx(0.25 / math.sqrt(1 / 32 - 1 / 256))

    def test_hit_rate_equals_fda(self, rng) -> None:
        actual, forecast, previous = rng.standard_normal((3, 40))
        res = pesaran_timmermann(actual, forecast, previous)
        assert res.hit_rate == pytest.approx(directional_accuracy(actual, forecast, previous))

    def test_power(self, rng) -> None:
        change = rng.standard_normal(200)
        res = pesaran_timmermann(change, change + 0.5 * rng.standard_normal(200), np.zeros(200))
        assert res.reject(0.01)

    def test_size_under_the_null(self) -> None:
        rng = np.random.default_rng(123)
        n, reps = 100, 2000
        rejections = 0
        for _ in range(reps):
            actual, forecast = rng.standard_normal((2, n))
            rejections += pesaran_timmermann(actual, forecast, np.zeros(n)).reject(0.05)
        assert 0.03 < rejections / reps < 0.075

    def test_alternatives(self, rng) -> None:
        change = rng.standard_normal(60)
        forecast = change + rng.standard_normal(60)
        greater = pesaran_timmermann(change, forecast, np.zeros(60))
        two = pesaran_timmermann(change, forecast, np.zeros(60), alternative="two-sided")
        less = pesaran_timmermann(change, forecast, np.zeros(60), alternative="less")
        assert two.pvalue == pytest.approx(2 * greater.pvalue)
        assert less.pvalue == pytest.approx(1 - greater.pvalue)
        with pytest.raises(ValueError, match="alternative"):
            pesaran_timmermann(change, forecast, np.zeros(60), alternative="up")

    def test_degenerate_directions(self) -> None:
        with pytest.warns(DataQualityWarning, match="Degenerate"):
            res = pesaran_timmermann([1.0, -1.0, 2.0, -3.0], [1.0, 2.0, 3.0, 4.0], np.zeros(4))
        assert math.isnan(res.statistic) and math.isnan(res.pvalue)
        assert res.hit_rate == 0.5
        assert not res.reject()

    def test_too_few_and_nans(self) -> None:
        with pytest.raises(NowcastDataError, match="at least 3"):
            pesaran_timmermann([1.0, NAN, 2.0], [1.0, 1.0, NAN], [0.0, 0.0, 0.0])

    def test_result(self) -> None:
        res = PesaranTimmermannResult(1.0, 0.16, 0.6, 0.5, 30, "greater")
        assert not res.reject(0.1)
        assert res.reject(0.2)

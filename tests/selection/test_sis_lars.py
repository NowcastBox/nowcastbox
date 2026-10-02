"""Tests for the SIS and LARS rankings of nowcastbox.selection.targeted."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.selection import (
    TargetedPredictorsResult,
    lars_path,
    lars_select,
    select_targeted_predictors,
    sis,
)
from nowcastbox.selection.targeted import _default_sis_size


def _design(seed: int = 0, n_obs: int = 120, n_pred: int = 8):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n_obs, n_pred)), columns=[f"x{i}" for i in range(n_pred)])
    y = 2.0 * x["x3"] - 1.0 * x["x5"] + 0.5 * x["x0"] + rng.normal(size=n_obs)
    return x, y


# ---------------------------------------------------------------------------
# SIS
# ---------------------------------------------------------------------------
class TestSis:
    def test_matches_manual_correlation(self):
        x, y = _design()
        res = sis(x, y, n_predictors=3)
        manual = x.apply(lambda col: abs(np.corrcoef(col, y)[0, 1]))
        np.testing.assert_allclose(res.scores.to_numpy(), manual.to_numpy())
        assert res.selected == list(manual.sort_values(ascending=False).index[:3])
        assert res.method == "sis" and isinstance(res, TargetedPredictorsResult)
        assert res.ranking[res.selected[0]] == 1

    def test_default_size_fan_lv(self):
        x, y = _design(n_obs=120, n_pred=40)
        res = sis(x, y)
        assert res.n_selected == int(np.floor(120 / np.log(120)))
        assert _default_sis_size(5, 2) == 2
        assert _default_sis_size(2, 10) == 1

    def test_pairwise_missing_and_horizon(self):
        x, y = _design()
        x.iloc[:10, 0] = np.nan
        res = sis(x, y, horizon=1, n_predictors=2)
        ok = x["x0"].notna().to_numpy()[:-1]
        expected = abs(np.corrcoef(x["x0"].to_numpy()[:-1][ok], y.to_numpy()[1:][ok])[0, 1])
        assert res.scores["x0"] == pytest.approx(expected)
        assert res.horizon == 1

    def test_skipped_predictors_warn(self):
        x, y = _design()
        x["const"] = 1.0
        x["sparse"] = np.nan
        x.loc[:4, "sparse"] = 1.0 + np.arange(5)
        with pytest.warns(DataQualityWarning, match="2 predictors skipped"):
            res = sis(x, y, n_predictors=2)
        assert np.isnan(res.scores[["const", "sparse"]]).all()
        assert np.isnan(res.ranking["const"])

    def test_nothing_usable_raises(self):
        x = pd.DataFrame({"a": np.ones(20)})
        with pytest.warns(DataQualityWarning), pytest.raises(NowcastDataError, match="enough"):
            sis(x, np.arange(20.0))

    @pytest.mark.parametrize("bad", [0, 9, -1, 1.5, True])
    def test_invalid_size(self, bad):
        x, y = _design()
        with pytest.raises(ValueError, match="n_predictors"):
            sis(x, y, n_predictors=bad)

    def test_summary_label(self):
        x, y = _design()
        assert "sure independence screening" in sis(x, y, n_predictors=1).summary()


# ---------------------------------------------------------------------------
# LARS selection
# ---------------------------------------------------------------------------
class TestLarsSelect:
    def test_matches_path_on_standardized_data(self):
        x, y = _design()
        res = lars_select(x, y, n_predictors=3)
        z = (x - x.mean()) / x.std(ddof=0)
        path = lars_path(z.to_numpy(), (y - y.mean()).to_numpy())
        names = list(x.columns)
        assert res.selected == [names[j] for j in path.entry_order[:3]]
        assert res.selected[:2] == ["x3", "x5"]
        np.testing.assert_allclose(res.scores.to_numpy(), path.entry_step)
        assert res.ranking["x3"] == 1 and res.method == "lars"
        assert res.params["n_steps"] == path.n_steps

    def test_default_size_and_lasso(self):
        x, y = _design()
        res = lars_select(x, y, method="lasso")
        assert res.n_selected == x.shape[1]
        assert res.params["method"] == "lasso"
        assert "least angle regression" in res.summary()

    def test_missing_drop_vs_mean(self):
        x, y = _design()
        x.iloc[:5, 1] = np.nan
        with pytest.warns(DataQualityWarning, match="Dropped 5"):
            dropped = lars_select(x, y, n_predictors=2)
        assert dropped.n_obs == len(y) - 5
        with pytest.warns(DataQualityWarning, match="replaced by the mean"):
            imputed = lars_select(x, y, n_predictors=2, missing="mean")
        assert imputed.n_obs == len(y)
        assert imputed.selected == ["x3", "x5"]

    def test_constant_and_empty_predictors_excluded(self):
        x, y = _design()
        x["const"] = 3.0
        x["empty"] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataQualityWarning)
            res = lars_select(x, y, n_predictors=2, missing="mean")
        assert np.isnan(res.scores[["const", "empty"]]).all()
        with pytest.warns(DataQualityWarning, match="excluded from LARS"):
            lars_select(x.drop(columns="empty"), y, n_predictors=2)

    def test_all_constant_raises(self):
        x = pd.DataFrame({"a": np.ones(20), "b": np.ones(20)})
        with pytest.warns(DataQualityWarning), pytest.raises(NowcastDataError, match="variation"):
            lars_select(x, np.arange(20.0))

    def test_fewer_entries_than_requested_warns(self):
        rng = np.random.default_rng(3)
        x = pd.DataFrame(rng.normal(size=(6, 10)), columns=[f"v{i}" for i in range(10)])
        y = pd.Series(rng.normal(size=6))
        with pytest.warns(DataQualityWarning, match="entered the LARS path"):
            res = lars_select(x, y, n_predictors=10)
        assert res.n_selected == 5  # rank of the centred design is n - 1

    def test_invalid_arguments(self):
        x, y = _design()
        with pytest.raises(ValueError, match="missing"):
            lars_select(x, y, missing="zero")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="n_predictors"):
            lars_select(x, y, n_predictors=0)
        with pytest.raises(ValueError, match="horizon"):
            lars_select(x, y, horizon=-1)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
class TestDispatch:
    def test_new_methods(self):
        x, y = _design()
        assert select_targeted_predictors(x, y, "sis", n_predictors=2).method == "sis"
        assert select_targeted_predictors(x, y, "lars", n_predictors=2).method == "lars"

    def test_unknown(self):
        x, y = _design()
        with pytest.raises(ValueError, match="'sis' or 'lars'"):
            select_targeted_predictors(x, y, "ridge")  # type: ignore[arg-type]

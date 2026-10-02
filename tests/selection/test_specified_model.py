"""Tests of SpecifiedModel (sample start + funnel selection around an estimator)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import nowcastbox as nb
from nowcastbox.core.base import BaseNowcaster
from nowcastbox.core.exceptions import ModelNotFittedError
from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.models import MixedFreqDFM, TwoStepDFM
from nowcastbox.selection import SpecifiedModel, preselect
from nowcastbox.selection import _specified as specified_module
from nowcastbox.selection._specified import configured_model, preselect_ranking, ranking_names


@pytest.fixture(scope="module")
def data():
    return nb.load_simulated_dfm().data


class MeanNowcaster(BaseNowcaster):
    """Results without predict and an estimator without update."""

    def __init__(self, scale=1.0):
        self.scale = scale

    def _fit(self, data, target, **kwargs):
        y = data.to_native(target)
        estimate = pd.Series(self.scale * y.mean(), index=y.index)
        return NowcastResults(target=target, nowcast=build_nowcast_frame(y, estimate))


class _OddResults(NowcastResults):
    def predict(self, data=None):
        return "not a frame"


class OddPredictNowcaster(MeanNowcaster):
    def _fit(self, data, target, **kwargs):
        res = super()._fit(data, target)
        return _OddResults(target=target, nowcast=res.nowcast)


class TestFit:
    def test_static_ranking_and_start(self, data):
        model = SpecifiedModel(
            TwoStepDFM(n_factors=1),
            start="2004-01",
            n_series=3,
            ranking=["unknown", "x05", "gdp", "x01", "x02", "x03"],
        )
        res = model.fit(data, "gdp")
        assert model.columns_ == ["x05", "x01", "x02"]
        assert str(res.nowcast.index[0]) == "2004Q1"
        assert isinstance(model.estimator_, TwoStepDFM)
        assert model.estimator_.n_factors == 1

    def test_params_are_set_on_a_copy(self, data):
        template = TwoStepDFM(n_factors=1)
        model = SpecifiedModel(template, params={"n_factors": 2})
        model.fit(data, "gdp")
        assert model.estimator_.n_factors == 2
        assert template.n_factors == 1
        assert model.columns_ == [c for c in data.columns if c != "gdp"]

    def test_start_before_the_sample_is_ignored(self, data):
        res = SpecifiedModel(TwoStepDFM, start="1990-01").fit(data, "gdp")
        assert str(res.nowcast.index[0]) == "2000Q1"

    @pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.DataQualityWarning")
    def test_preselection_result(self, data):
        pre = preselect(data, "gdp", methods=("sis",))
        model = SpecifiedModel(TwoStepDFM, n_series=2, ranking=pre)
        model.fit(data, "gdp")
        assert model.columns_ == list(pre.ranking().index[:2])

    def test_preselect_options_recomputed_on_the_data(self, data):
        model = SpecifiedModel(TwoStepDFM, n_series=3, ranking={"methods": ("sis",)})
        model.fit(data, "gdp")
        expected = list(preselect(data, "gdp", methods=("sis",)).ranking().index[:3])
        assert model.columns_ == expected
        vintage = data.as_of("2010-01-31")
        model.fit(vintage, "gdp")
        early = preselect(data, "gdp", methods=("sis",), as_of="2010-01-31")
        assert model.columns_ == list(early.ranking().index[:3])

    def test_preselect_string(self, data):
        model = SpecifiedModel(TwoStepDFM(n_factors=1), n_series=2, ranking="preselect")
        model.fit(data, "gdp")
        assert model.columns_ == list(preselect(data, "gdp").ranking().index[:2])

    def test_ranking_ignored_without_n_series(self, data):
        model = SpecifiedModel(TwoStepDFM, ranking=["x01"])
        model.fit(data, "gdp")
        assert len(model.columns_) == data.n_series - 1

    def test_n_series_capped(self, data):
        model = SpecifiedModel(TwoStepDFM(n_factors=1), n_series=10, ranking=["x01", "x02"])
        model.fit(data, "gdp")
        assert model.columns_ == ["x01", "x02"]

    def test_preprocess(self, data):
        seen = []

        def spy(panel):
            seen.append(panel.n_periods)
            return panel

        SpecifiedModel(TwoStepDFM, preprocess=spy).fit(data, "gdp")
        assert seen == [data.n_periods]

    def test_fit_kwargs_and_formula(self, data):
        model = SpecifiedModel(MixedFreqDFM(max_iter=5))
        res = model.fit(data.select(["gdp", "x01", "x02"]), "gdp ~ x01", horizon=1)
        assert model.columns_ == ["x01"]
        assert res.target == "gdp"

    def test_clone_and_reset(self, data):
        model = SpecifiedModel(TwoStepDFM(n_factors=1), params={"factor_lags": 1})
        model.fit(data, "gdp")
        copy = model.clone()
        assert copy.estimator_ is None
        assert copy.params == {"factor_lags": 1}
        model.set_params(start="2005-01")
        assert model.estimator_ is None
        assert model.columns_ is None


class TestValidation:
    def test_n_series_needs_ranking(self, data):
        with pytest.raises(ValueError, match="needs a ranking"):
            SpecifiedModel(TwoStepDFM, n_series=2).fit(data, "gdp")

    @pytest.mark.parametrize("n", [0, 1.5, True])
    def test_bad_n_series(self, data, n):
        with pytest.raises(ValueError, match="positive integer"):
            SpecifiedModel(TwoStepDFM, n_series=n, ranking=["x01"]).fit(data, "gdp")

    def test_bad_model(self, data):
        with pytest.raises(ValueError, match="must be an estimator"):
            SpecifiedModel(42).fit(data, "gdp")

    def test_no_indicator_left(self, data):
        with pytest.raises(ValueError, match="keeps no indicator"):
            SpecifiedModel(TwoStepDFM, n_series=2, ranking=["nope"]).fit(data, "gdp")

    def test_bad_static_ranking(self, data):
        with pytest.raises(ValueError, match="must be 'preselect'"):
            SpecifiedModel(TwoStepDFM, n_series=2, ranking="x01").fit(data, "gdp")
        with pytest.raises(TypeError, match="sequence of series names"):
            ranking_names([1, 2])


class TestUpdate:
    def test_not_fitted(self, data):
        with pytest.raises(ModelNotFittedError):
            SpecifiedModel(TwoStepDFM).update(data)

    def test_update_of_the_inner_model(self, data):
        model = SpecifiedModel(TwoStepDFM(n_factors=1), n_series=4, ranking={"methods": ("sis",)})
        res = model.fit(data.as_of("2015-06-30"), "gdp")
        same = model.update(data.as_of("2015-06-30"))
        pd.testing.assert_series_equal(same.estimate, res.estimate)
        later = model.update(data.as_of("2015-09-30"))
        assert isinstance(later, NowcastResults)
        assert model.columns_ == list(
            preselect(data.as_of("2015-06-30"), "gdp", methods=("sis",)).ranking().index[:4]
        )
        assert later.estimate.index[-1] >= res.estimate.index[-1]

    def test_predict_path(self, data):
        panel = data.select(["gdp", "x01", "x02", "x03"])
        model = SpecifiedModel(MixedFreqDFM(max_iter=10), start="2003-01")
        res = model.fit(panel, "gdp")
        again = model.update(panel)
        assert isinstance(again, NowcastResults)
        # the information update (smoothed signal) reproduces the out-of-sample nowcasts
        future = res.nowcast["out_of_sample"].dropna()
        np.testing.assert_allclose(
            again.nowcast["out_of_sample"].reindex(future.index), future, atol=1e-8
        )
        assert again.nowcast["observed"].notna().sum() == res.nowcast["observed"].notna().sum()

    def test_no_update_nor_predict(self, data):
        model = SpecifiedModel(MeanNowcaster())
        model.fit(data, "gdp")
        assert model.update(data) is None

    def test_predict_not_a_frame(self, data):
        model = SpecifiedModel(OddPredictNowcaster())
        model.fit(data, "gdp")
        assert model.update(data) is None


class TestHelpers:
    def test_configured_model(self):
        assert isinstance(configured_model(TwoStepDFM), TwoStepDFM)

        class NoClone:
            def __init__(self):
                self.value = [1]

            def fit(self, data, target):
                return self

        original = NoClone()
        copy = configured_model(original)
        assert copy is not original
        assert copy.value == [1]
        with pytest.raises(ValueError, match="has no set_params"):
            configured_model(original, {"value": 2})
        with pytest.raises(ValueError, match="Invalid parameter"):
            configured_model(TwoStepDFM(), {"nope": 2})

    def test_preselect_cache(self, data, monkeypatch):
        specified_module._preselect_cache.clear()
        calls = []
        real = specified_module.preselect

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(specified_module, "preselect", counting)
        monkeypatch.setattr(specified_module, "_CACHE_SIZE", 1)
        first = preselect_ranking(data, "gdp", {"methods": ("sis",)})
        assert preselect_ranking(data, "gdp", {"methods": ("sis",)}) == first
        assert len(calls) == 1
        preselect_ranking(data, "gdp", {"methods": ("tstat",)})
        assert len(calls) == 2
        assert len(specified_module._preselect_cache) == 1
        preselect_ranking(data, "gdp", {"methods": ("sis",)})
        assert len(calls) == 3

    def test_preselect_cache_depends_on_metadata(self, data, monkeypatch):
        specified_module._preselect_cache.clear()
        calls = []
        real = specified_module.preselect

        def counting(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        monkeypatch.setattr(specified_module, "preselect", counting)
        preselect_ranking(data, "gdp", {"methods": ("sis",)})
        preselect_ranking(data.with_metadata("x01", release_delay=90), "gdp", {"methods": ("sis",)})
        assert len(calls) == 2

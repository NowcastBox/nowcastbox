"""Tests of the Covid treatments of the robustness step."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.models import MixedFreqDFM, TwoStepDFM
from nowcastbox.selection import COVID_TREATMENTS, OutlierCorrection, TreatmentPlan
from nowcastbox.selection._search_treatments import resolve_treatments


class _Plain:
    """A model without get_params."""

    def fit(self, data, target):
        return self


class _OddParams:
    def get_params(self, deep=False):
        return ["covid"]


def test_builtin_names():
    assert list(COVID_TREATMENTS) == ["none", "dummy", "mask", "outliers"]


class TestMixedFreqDFM:
    @pytest.mark.parametrize(
        ("name", "params"),
        [
            ("none", {"covid": "none", "outliers": "none"}),
            ("dummy", {"covid": "dummy", "outliers": "none"}),
            ("mask", {"covid": "mask", "outliers": "none"}),
            ("outliers", {"covid": "none", "outliers": "auto"}),
        ],
    )
    def test_params(self, name, params):
        plan = COVID_TREATMENTS[name](MixedFreqDFM(covid="dummy", outliers="auto"), "gdp")
        assert plan.supported
        assert dict(plan.params) == params
        assert plan.preprocess is None
        assert plan.note


class TestModelsWithoutOptions:
    @pytest.mark.parametrize("name", ["dummy", "mask"])
    def test_covid_options_unsupported(self, name):
        plan = COVID_TREATMENTS[name](TwoStepDFM(), "gdp")
        assert not plan.supported
        assert "TwoStepDFM has no 'covid' option" in plan.note

    def test_outliers_fall_back_to_iqr_correction(self):
        plan = COVID_TREATMENTS["outliers"](TwoStepDFM(), "gdp")
        assert plan.supported
        assert dict(plan.params) == {}
        assert isinstance(plan.preprocess, OutlierCorrection)
        assert plan.preprocess.exclude == ("gdp",)

    def test_none_without_params(self):
        assert dict(COVID_TREATMENTS["none"](_Plain(), "y").params) == {}

    def test_non_mapping_params(self):
        assert not COVID_TREATMENTS["dummy"](_OddParams(), "y").supported


class TestOutlierCorrection:
    @pytest.fixture
    def panel(self):
        idx = pd.period_range("2020-01", periods=12, freq="M")
        x = np.array([0.0, 1.0, -1.0, 0.5, 30.0, -0.5, 0.2, 0.1, 0.3, -0.2, 0.4, 0.0])
        return MixedFrequencyData(pd.DataFrame({"x": x, "y": x, "z": x}, index=idx))

    def test_target_excluded(self, panel):
        out = OutlierCorrection(exclude=("y",))(panel)
        assert out["x"].iloc[4] != 30.0
        assert out["z"].iloc[4] != 30.0
        assert out["y"].iloc[4] == 30.0

    def test_nothing_to_correct(self, panel):
        only = panel.select(["y"])
        assert OutlierCorrection(exclude=("y",))(only) is only

    def test_threshold(self, panel):
        assert OutlierCorrection(threshold=100.0)(panel)["x"].iloc[4] == 30.0

    def test_repr_is_stable(self):
        assert repr(OutlierCorrection(("y",), 3, 5)) == (
            "OutlierCorrection(exclude=('y',), threshold=3.0, window=5)"
        )


class TestResolve:
    def test_names(self):
        assert resolve_treatments(("none", "mask")) == {
            "none": COVID_TREATMENTS["none"],
            "mask": COVID_TREATMENTS["mask"],
        }

    def test_single_name(self):
        assert list(resolve_treatments("dummy")) == ["dummy"]

    def test_custom_mapping(self):
        def custom(model, target):
            return TreatmentPlan({"covid": "mask"}, note="custom")

        out = resolve_treatments({"base": "none", "mine": custom})
        assert out["mine"] is custom
        assert out["base"] is COVID_TREATMENTS["none"]

    def test_unknown(self):
        with pytest.raises(ValueError, match="Unknown treatment"):
            resolve_treatments(("none", "drop"))

    def test_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            resolve_treatments(())

    def test_not_callable(self):
        with pytest.raises(TypeError, match="name or a callable"):
            resolve_treatments({"x": 3})


def test_unsupported_plan():
    plan = TreatmentPlan.unsupported("no")
    assert not plan.supported
    assert plan.note == "no"
    assert dict(plan.params) == {}

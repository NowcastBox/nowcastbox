from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.base import BaseBenchmark, BaseNowcaster, BenchmarkForecaster, ParamsMixin
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import FormulaError, ModelNotFittedError, NowcastDataError
from nowcastbox.core.frequency import Frequency
from nowcastbox.core.results import NowcastResults, build_nowcast_frame


class MeanNowcaster(BaseNowcaster):
    """Toy model: nowcast = scale * mean of the target."""

    def __init__(self, scale: float = 1.0, n_factors: int = 1):
        self.scale = scale
        self.n_factors = n_factors

    def _validate_params(self) -> None:
        if self.n_factors < 1:
            raise ValueError("n_factors must be positive")

    def _fit(self, data, target, **kwargs):
        self.seen_columns_ = data.columns
        self.seen_kwargs_ = kwargs
        y = data.to_native(target)
        est = pd.Series(self.scale * y.mean(), index=y.index)
        return NowcastResults(
            target=target,
            nowcast=build_nowcast_frame(y, est),
            model_name=type(self).__name__,
            model_params=self.get_params(),
            data=data,
        )


class BadNowcaster(BaseNowcaster):
    def __init__(self):
        pass

    def _fit(self, data, target, **kwargs):
        return "not results"


class Wrapper(ParamsMixin):
    def __init__(self, inner=None, alpha=0.1):
        self.inner = inner
        self.alpha = alpha


class Last(BaseBenchmark):
    """Benchmark returning the last observed value."""

    def __init__(self, shift: float = 0.0):
        self.shift = shift

    def _fit(self, data, target):
        self.last_ = float(data.to_native(target, dropna=True).iloc[-1])

    def _predict(self, periods):
        return pd.Series(self.last_ + self.shift, index=periods)


class Misaligned(Last):
    def _predict(self, periods):
        return pd.Series([0.0])


class TestParamsMixin:
    def test_get_set_params(self):
        m = MeanNowcaster(scale=2.0)
        assert m.get_params() == {"scale": 2.0, "n_factors": 1}
        assert m.set_params(n_factors=3) is m
        assert m.n_factors == 3

    def test_invalid_param(self):
        with pytest.raises(ValueError, match="Invalid parameter 'foo'"):
            MeanNowcaster().set_params(foo=1)

    def test_nested(self):
        w = Wrapper(inner=MeanNowcaster(scale=3.0))
        params = w.get_params()
        assert params["inner__scale"] == 3.0
        assert w.get_params(deep=False).keys() == {"inner", "alpha"}
        w.set_params(inner__scale=5.0, alpha=0.2)
        assert w.inner.scale == 5.0 and w.alpha == 0.2
        with pytest.raises(ValueError, match="not an estimator"):
            w.set_params(alpha__x=1)

    def test_clone_and_repr(self):
        m = MeanNowcaster(scale=2.0)
        c = m.clone()
        assert c is not m and c.get_params() == m.get_params()
        assert repr(m) == "MeanNowcaster(scale=2.0, n_factors=1)"

    def test_missing_attribute(self):
        class Forgetful(ParamsMixin):
            def __init__(self, a=1):
                pass

        with pytest.raises(AttributeError, match="must store"):
            Forgetful().get_params()

    def test_varargs_rejected(self):
        class Varargs(ParamsMixin):
            def __init__(self, *args):
                pass

        with pytest.raises(TypeError, match="args"):
            Varargs().get_params()

    def test_no_init(self):
        class Plain(ParamsMixin):
            pass

        assert Plain().get_params() == {}


class TestBaseNowcaster:
    def test_fit_dataframe(self, panel_frame):
        model = MeanNowcaster()
        assert not model.is_fitted
        res = model.fit(panel_frame, target="gdp", frequency={"gdp": "Q"}, extra=1)
        assert model.is_fitted
        assert model.results_ is res
        assert model.seen_kwargs_ == {"extra": 1}
        assert model.seen_columns_ == ["ip", "pmi", "gdp"]
        assert res.target_frequency is Frequency.QUARTERLY
        assert res.model_params == {"scale": 1.0, "n_factors": 1}
        observed_mean = panel_frame["gdp"].mean()
        assert res.get_nowcast() == pytest.approx(observed_mean)

    def test_fit_formula_selects_columns(self, panel):
        model = MeanNowcaster()
        model.fit(panel, target="gdp ~ pmi")
        assert model.seen_columns_ == ["pmi", "gdp"]
        model.fit(panel, target="gdp ~ . - ip")
        assert model.seen_columns_ == ["pmi", "gdp"]
        model.fit(panel, target="gdp ~ .")
        assert model.seen_columns_ == ["ip", "pmi", "gdp"]

    def test_fit_errors(self, panel):
        with pytest.raises(FormulaError):
            MeanNowcaster().fit(panel, target="zzz")
        with pytest.raises(NowcastDataError, match="already"):
            MeanNowcaster().fit(panel, target="gdp", frequency={"gdp": "Q"})
        with pytest.raises(ValueError, match="n_factors"):
            MeanNowcaster(n_factors=0).fit(panel, target="gdp")
        with pytest.raises(TypeError, match="must return a NowcastResults"):
            BadNowcaster().fit(panel, target="gdp")

    def test_not_fitted(self):
        with pytest.raises(ModelNotFittedError, match="not fitted"):
            _ = MeanNowcaster().results_

    def test_set_params_resets_fit(self, panel):
        model = MeanNowcaster()
        model.fit(panel, target="gdp")
        model.set_params(scale=2.0)
        assert not model.is_fitted

    def test_failed_fit_leaves_unfitted(self, panel):
        model = MeanNowcaster()
        model.fit(panel, target="gdp")
        model.n_factors = 0
        with pytest.raises(ValueError):
            model.fit(panel, target="gdp")
        assert not model.is_fitted


class TestBenchmarks:
    def test_protocol(self):
        assert isinstance(Last(), BenchmarkForecaster)
        assert not isinstance(object(), BenchmarkForecaster)

    def test_fit_predict(self, panel):
        bench = Last(shift=1.0)
        assert bench.name == "Last"
        assert not bench.is_fitted
        assert bench.fit(panel, "gdp") is bench
        assert bench.is_fitted
        assert bench.target_ == "gdp"
        assert bench.target_frequency_ is Frequency.QUARTERLY
        last = panel.to_native("gdp", dropna=True).iloc[-1]
        pred = bench.predict(["2019Q4", pd.Period("2020Q1", "Q")])
        assert pred.name == "gdp"
        np.testing.assert_allclose(pred.to_numpy(), last + 1.0)
        pred2 = bench.predict(pd.period_range("2019-12", periods=1, freq="M"))
        assert str(pred2.index[0]) == "2019Q4"

    def test_fit_dataframe(self, panel_frame):
        bench = Last().fit(panel_frame, "ip", frequency={"gdp": "Q"})
        assert bench.target_frequency_ is Frequency.MONTHLY

    def test_errors(self, panel):
        with pytest.raises(ModelNotFittedError):
            Last().predict(["2020Q1"])
        with pytest.raises(ModelNotFittedError):
            _ = Last().target_
        with pytest.raises(NowcastDataError, match="not a column"):
            Last().fit(panel, "zzz")
        with pytest.raises(ValueError, match="misaligned"):
            Misaligned().fit(panel, "gdp").predict(["2020Q1"])

    def test_set_params_resets(self, panel):
        bench = Last().fit(panel, "gdp")
        bench.set_params(shift=2.0)
        assert not bench.is_fitted


def test_toy_model_on_mixed_frequency_data_object():
    idx = pd.period_range("2020-01", periods=6, freq="M")
    df = pd.DataFrame(
        {"x": np.arange(6.0), "y": [np.nan, np.nan, 1.0, np.nan, np.nan, 3.0]}, index=idx
    )
    mfd = MixedFrequencyData(df, {"x": "M", "y": "Q"})
    res = MeanNowcaster(scale=2.0).fit(mfd, "y")
    assert res.estimate.tolist() == [4.0, 4.0]

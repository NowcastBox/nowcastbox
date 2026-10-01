import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.exceptions import ModelNotFittedError, NowcastBoxWarning
from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.experiment import NowcastExperiment, in_sample_metrics
from nowcastbox.models import MixedFreqDFM, TwoStepDFM


class _Failing:
    def fit(self, data, target, **kwargs):
        raise RuntimeError("boom")


class _Recorder:
    """Returns fixed results and records the fit arguments."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def fit(self, data, target, **kwargs):
        self.calls.append(kwargs)
        return self.results


class _NotResults:
    def fit(self, data, target, **kwargs):
        return 3


@pytest.fixture
def exp(panel):
    e = NowcastExperiment(panel, "gdp")
    e.add_model("2s", TwoStepDFM(n_factors=2)).add_model("em", MixedFreqDFM(max_iter=10))
    e.fit_all()
    return e


def test_compare_table(exp):
    table = exp.compare()
    assert table.index.tolist() == ["2s", "em"]
    for col in (
        "nowcast",
        "loglikelihood",
        "rmse_in_sample",
        "r2_in_sample",
        "fit_seconds",
        "lower_90",
        "converged",
        "n_factors",
    ):
        assert col in table.columns
    for name in ("2s", "em"):
        res = exp.get_results(name)
        assert table.loc[name, "nowcast"] == pytest.approx(res.get_nowcast())
    assert np.isnan(table.loc["2s", "lower_90"])
    assert not np.isnan(table.loc["em", "lower_90"])
    assert table.loc["em", "loglikelihood"] == pytest.approx(exp.get_results("em").loglikelihood)


def test_compare_period(exp):
    table = exp.compare(period="2006Q4")
    assert table.loc["2s", "nowcast_period"] == "2006Q4"
    assert table.loc["2s", "nowcast"] == pytest.approx(exp.get_results("2s").get_nowcast("2006Q4"))
    far = exp.compare(period="2030Q1")
    assert np.isnan(far.loc["2s", "nowcast"])


def test_nowcast_table(exp):
    table = exp.nowcast_table()
    assert table.columns.tolist() == ["observed", "2s", "em"]
    oos = exp.nowcast_table(kind="out_of_sample")
    assert oos["2s"].notna().sum() < table["2s"].notna().sum()
    with pytest.raises(ValueError):
        exp.nowcast_table(kind="x")  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_plot_nowcasts(exp, backend):
    assert exp.plot(backend=backend) is not None
    fig = exp.plot(n_periods=None)
    assert [t.name for t in fig.data] == ["Observed", "2s", "em"]
    with pytest.raises(ValueError, match="kind"):
        exp.plot("pie")  # type: ignore[arg-type]


def test_backtest_hook(exp):
    calls = []

    def hook(estimator, data, target, scale=1.0):
        calls.append((type(estimator).__name__, target))
        return pd.Series([0.5, 0.7], index=[0, 1]) * scale

    out = exp.run_backtest(hook, scale=2.0)
    assert set(out) == {"2s", "em"}
    assert calls == [("TwoStepDFM", "gdp"), ("MixedFreqDFM", "gdp")]
    table = exp.rmsfe_table()
    assert table["2s"].tolist() == [1.0, 1.4]
    assert exp.plot("rmsfe", relative_to="2s") is not None
    assert set(exp.backtests) == {"2s", "em"}
    exp.run_backtest(lambda e, d, t: pd.DataFrame({"a": [1.0], "b": [2.0]}), models=["em"])
    with pytest.raises(ValueError, match="columns"):
        exp.rmsfe_table()
    exp.run_backtest(lambda e, d, t: "x", models=["em"])
    with pytest.raises(ValueError, match="not RMSFE"):
        exp.rmsfe_table()
    with pytest.raises(TypeError):
        exp.run_backtest("hook")  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        exp.run_backtest(lambda e, d, t: None, models=["nope"])


def test_summary_report_repr(exp, tmp_path):
    text = exp.summary()
    assert "2 model(s), 2 fitted" in text and "rmse_in_sample" in text
    exp.run_backtest(lambda e, d, t: pd.Series([0.5], name="rmsfe"))
    html = exp.report("em", tmp_path / "em.html", plotlyjs="cdn")
    assert (tmp_path / "em.html").exists()
    assert "Nowcast report: gdp (em)" in html and "RMSFE by horizon" in html
    exp.run_backtest(lambda e, d, t: "bad", models=["em"])
    html = exp.report("2s", plotlyjs="cdn")
    assert 'data-section="diagnostics"' in html
    assert repr(exp) == "NowcastExperiment(target='gdp', models=['2s', 'em'], fitted=['2s', 'em'])"


def test_model_management(panel):
    exp = NowcastExperiment(panel, "gdp", models={"a": TwoStepDFM()})
    assert exp.list_models() == ["a"]
    with pytest.raises(ValueError, match="already exists"):
        exp.add_model("a", TwoStepDFM())
    with pytest.raises(ValueError, match="non-empty"):
        exp.add_model("", TwoStepDFM())
    with pytest.raises(TypeError, match="fit"):
        exp.add_model("b", object())
    with pytest.raises(KeyError):
        exp.get_model("zz")
    with pytest.raises(ModelNotFittedError):
        exp.get_results("a")
    with pytest.raises(ModelNotFittedError):
        exp.compare()
    with pytest.raises(ValueError, match="target"):
        NowcastExperiment(panel, " ")
    res = exp.fit_model("a")
    assert exp.results == {"a": res}
    exp.add_model("a", TwoStepDFM(n_factors=2), overwrite=True)
    assert exp.results == {}


def test_fit_all_errors_and_refit(panel, two_step):
    rec = _Recorder(two_step)
    exp = NowcastExperiment(panel, "gdp", frequency={"gdp": "Q"})
    exp.add_model("rec", rec).add_model("bad", _Failing())
    with pytest.raises(RuntimeError):
        exp.fit_all()
    with pytest.warns(NowcastBoxWarning, match="boom"):
        exp.fit_all(errors="warn")
    assert list(exp.failures) == ["bad"]
    assert "FAILED bad" in exp.summary()
    assert rec.calls[-1]["frequency"] == {"gdp": "Q"}
    n_calls = len(rec.calls)
    with pytest.warns(NowcastBoxWarning):
        exp.fit_all(errors="warn")
    assert len(rec.calls) == n_calls  # already fitted, not refitted
    with pytest.warns(NowcastBoxWarning):
        exp.fit_all(errors="warn", refit=True)
    assert len(rec.calls) == n_calls + 1
    with pytest.raises(ValueError, match="errors"):
        exp.fit_all(errors="ignore")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no models"):
        NowcastExperiment(panel, "gdp").fit_all()
    with pytest.raises(TypeError, match="NowcastResults"):
        exp.fit_model("x", _NotResults())


def test_in_sample_metrics():
    idx = pd.period_range("2020Q1", periods=3, freq="Q")
    f = build_nowcast_frame(
        pd.Series([1.0, 2.0, 3.0], index=idx), pd.Series([1.5, 2.0, 2.5], index=idx)
    )
    m = in_sample_metrics(NowcastResults(target="y", nowcast=f))
    assert m["rmse_in_sample"] == pytest.approx(np.sqrt(0.5 / 3))
    assert m["mae_in_sample"] == pytest.approx(1 / 3)
    assert m["r2_in_sample"] == pytest.approx(1 - 0.5 / 2.0)
    f1 = build_nowcast_frame(
        pd.Series([1.0, np.nan], index=idx[:2]), pd.Series([1.0, 2.0], index=idx[:2])
    )
    assert np.isnan(in_sample_metrics(NowcastResults(target="y", nowcast=f1))["rmse_in_sample"])
    flat = build_nowcast_frame(
        pd.Series([1.0, 1.0], index=idx[:2]), pd.Series([1.0, 1.2], index=idx[:2])
    )
    assert np.isnan(in_sample_metrics(NowcastResults(target="y", nowcast=flat))["r2_in_sample"])


def test_compare_fully_observed_target(panel):
    idx = pd.period_range("2020Q1", periods=2, freq="Q")
    f = build_nowcast_frame(pd.Series([1.0, 2.0], index=idx), pd.Series([1.0, 2.0], index=idx))
    exp = NowcastExperiment(
        panel, "gdp", models={"r": _Recorder(NowcastResults(target="y", nowcast=f))}
    )
    exp.fit_all()
    row = exp.compare().loc["r"]
    assert row["nowcast_period"] is None and np.isnan(row["nowcast"])


def test_empty_experiment_outputs(panel, two_step):
    exp = NowcastExperiment(panel, "gdp")
    with pytest.raises(ValueError, match="No backtest"):
        exp.rmsfe_table()
    assert exp.summary() == "NowcastExperiment: target gdp, 0 model(s), 0 fitted"
    exp.fit_model("r", _Recorder(two_step))
    exp.run_backtest(lambda e, d, t: pd.Series([0.9]))
    html = exp.report("r", plotlyjs="cdn", backtest=pd.Series([0.1], name="given"))
    assert "given" in html


@pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.ConvergenceWarning")
def test_default_backtest_is_pseudo_real_time(exp):
    """Without a hook, :class:`PseudoRealTimeBacktest` runs for every model (wave 2)."""
    from nowcastbox.evaluation import BacktestResults

    out = exp.run_backtest(delay=30, start="2007-01-01", end="2007-03-01", n_jobs=1)
    assert all(isinstance(r, BacktestResults) for r in out.values())
    table = exp.rmsfe_table()
    assert list(table.columns) == ["2s", "em"]
    assert np.isfinite(table.to_numpy()).any()

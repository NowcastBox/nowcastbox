from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core import results as results_module
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.frequency import Frequency
from nowcastbox.core.results import (
    NOWCAST_COLUMNS,
    FactorResults,
    NowcastResults,
    _companion,
    available_plots,
    build_nowcast_frame,
    register_plot,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    idx = pd.period_range("2019Q1", periods=6, freq="Q")
    observed = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, np.nan], index=idx)
    estimate = pd.Series([1.1, 1.9, 3.2, 3.9, 4.5, 5.0], index=idx)
    return build_nowcast_frame(observed, estimate)


@pytest.fixture
def results(frame, panel) -> NowcastResults:
    return NowcastResults(
        target="gdp",
        nowcast=frame,
        model_name="Demo",
        model_params={"n_factors": 2, "tol": 1e-4},
        factors=pd.DataFrame(
            np.zeros((panel.n_periods, 2)), index=panel.index, columns=["f1", "f2"]
        ),
        loadings=pd.DataFrame(np.ones((3, 2)), index=panel.columns, columns=["f1", "f2"]),
        params={"A": np.eye(2)},
        loglikelihood=-12.5,
        n_iter=7,
        converged=True,
        data=panel,
        info={"fit_time": 0.1},
    )


@pytest.fixture
def _clean_registry():
    saved = dict(results_module._PLOT_REGISTRY)
    yield
    results_module._PLOT_REGISTRY.clear()
    results_module._PLOT_REGISTRY.update(saved)


class TestBuildNowcastFrame:
    def test_split(self, frame):
        assert tuple(frame.columns) == NOWCAST_COLUMNS
        assert frame["in_sample"].notna().sum() == 4
        assert frame["out_of_sample"].notna().sum() == 2
        assert not (frame["in_sample"].notna() & frame["out_of_sample"].notna()).any()
        assert frame.index.name == "period"

    def test_union_index_and_extra(self):
        obs = pd.Series([1.0], index=pd.period_range("2020Q1", periods=1, freq="Q"))
        est = pd.Series([1.0, 2.0], index=pd.period_range("2020Q1", periods=2, freq="Q"))
        out = build_nowcast_frame(obs, est, extra={"std": est * 0.1})
        assert len(out) == 2
        assert out["std"].tolist() == pytest.approx([0.1, 0.2])
        assert out["out_of_sample"].tolist()[1] == 2.0

    def test_errors(self):
        q = pd.Series([1.0], index=pd.period_range("2020Q1", periods=1, freq="Q"))
        m = pd.Series([1.0], index=pd.period_range("2020-01", periods=1, freq="M"))
        with pytest.raises(NowcastDataError, match="PeriodIndex"):
            build_nowcast_frame(pd.Series([1.0]), q)
        with pytest.raises(NowcastDataError, match="same frequency"):
            build_nowcast_frame(q, m)


class TestNowcastResults:
    def test_validation(self, frame):
        with pytest.raises(NowcastDataError, match="DataFrame"):
            NowcastResults(target="y", nowcast=frame["observed"])  # type: ignore[arg-type]
        with pytest.raises(NowcastDataError, match="PeriodIndex"):
            NowcastResults(target="y", nowcast=frame.reset_index(drop=True))
        with pytest.raises(NowcastDataError, match="missing the columns"):
            NowcastResults(target="y", nowcast=frame.drop(columns="in_sample"))
        with pytest.raises(NowcastDataError, match="factors"):
            NowcastResults(target="y", nowcast=frame, factors=np.zeros((2, 2)))  # type: ignore[arg-type]

    def test_immutable_and_copies(self, frame, results):
        with pytest.raises(AttributeError):
            results.target = "x"  # type: ignore[misc]
        frame.iloc[0, 0] = 999.0
        assert results.nowcast.iloc[0, 0] == 1.0
        assert isinstance(results.loglikelihood, float)

    def test_accessors(self, results):
        assert results.target_frequency is Frequency.QUARTERLY
        assert results.observed.notna().sum() == 4
        assert results.in_sample.notna().sum() == 4
        assert results.out_of_sample.notna().sum() == 2
        assert results.estimate.notna().all()
        assert results.estimate.name == "estimate"
        assert results.n_factors == 2

    def test_n_factors_variants(self, frame):
        assert NowcastResults(target="y", nowcast=frame).n_factors is None
        loadings = pd.DataFrame(np.ones((2, 3)))
        assert NowcastResults(target="y", nowcast=frame, loadings=loadings).n_factors == 3

    def test_get_nowcast(self, results):
        assert results.get_nowcast() == pytest.approx(4.5)
        assert results.get_nowcast("2019Q2") == pytest.approx(1.9)
        assert results.get_nowcast(pd.Period("2020Q2", "Q")) == pytest.approx(5.0)
        with pytest.raises(KeyError, match="not in the nowcast"):
            results.get_nowcast("2030Q1")

    def test_get_nowcast_edge_cases(self):
        idx = pd.period_range("2020Q1", periods=2, freq="Q")
        full = build_nowcast_frame(
            pd.Series([1.0, 2.0], index=idx), pd.Series([1.0, 2.0], index=idx)
        )
        with pytest.raises(KeyError, match="No period after"):
            NowcastResults(target="y", nowcast=full).get_nowcast()
        none = build_nowcast_frame(
            pd.Series([np.nan, np.nan], index=idx), pd.Series([3.0, 4.0], index=idx)
        )
        assert NowcastResults(target="y", nowcast=none).get_nowcast() == 3.0

    def test_replace(self, results):
        new = results.replace(model_name="Other")
        assert new.model_name == "Other" and results.model_name == "Demo"
        assert type(new) is NowcastResults

    def test_to_frame(self, results):
        out = results.to_frame()
        out.iloc[0, 0] = -1.0
        assert results.nowcast.iloc[0, 0] == 1.0

    def test_summary(self, results):
        text = results.summary(n_periods=3)
        assert "Nowcast Results: Demo" in text
        assert "gdp (quarterly)" in text
        assert "-12.5000" in text
        assert "n_factors" in text
        assert "2020Q2" in text and "2019Q1" not in text
        assert str(results) == results.summary()

    def test_summary_minimal(self, frame):
        text = NowcastResults(target="y", nowcast=frame).summary()
        assert text.startswith("=")
        assert "Nowcast Results\n" in text

    def test_repr(self, results, frame):
        assert repr(results) == "NowcastResults(model='Demo', target='gdp', periods=2019Q1..2020Q2)"
        empty = frame.iloc[:0]
        assert "empty" in repr(NowcastResults(target="y", nowcast=empty))

    def test_save_load(self, results, tmp_path):
        path = results.save(tmp_path / "res.pkl")
        loaded = NowcastResults.load(path)
        pd.testing.assert_frame_equal(loaded.nowcast, results.nowcast)
        assert loaded.data is not None and loaded.data.equals(results.data)
        np.testing.assert_array_equal(loaded.params["A"], np.eye(2))

    def test_load_wrong_type(self, results, tmp_path):
        path = results.save(tmp_path / "res.pkl")
        with pytest.raises(TypeError, match="FactorResults"):
            FactorResults.load(path)


@pytest.mark.usefixtures("_clean_registry")
class TestPlotRegistry:
    def test_not_registered(self, results):
        with pytest.raises(NotImplementedError, match="No plot 'nope'"):
            results.plot("nope")

    def test_register_and_dispatch(self, results):
        @register_plot("demo")
        def _demo(res, **kwargs):
            return ("fig", res.target, kwargs)

        assert results.plot("demo", color="red") == ("fig", "gdp", {"color": "red"})
        assert "demo" in available_plots(results)
        assert "demo" in available_plots(FactorResults)

    def test_subclass_override(self, frame):
        @register_plot("demo")
        def _base(res, **kwargs):
            return "base"

        @register_plot("demo", FactorResults)
        def _factor(res, **kwargs):
            return "factor"

        assert NowcastResults(target="y", nowcast=frame).plot("demo") == "base"
        assert FactorResults(target="y", nowcast=frame).plot("demo") == "factor"

    def test_lazy_import_of_visualization(self, frame, monkeypatch):
        calls = []

        def fake_import(name):
            calls.append(name)
            register_plot("lazy")(lambda res, **kw: "lazy-fig")

        monkeypatch.setattr(results_module.importlib, "import_module", fake_import)
        assert NowcastResults(target="y", nowcast=frame).plot("lazy") == "lazy-fig"
        assert calls == ["nowcastbox.visualization"]


class TestFactorResults:
    def test_transition_matrices(self, frame):
        A1 = np.array([[0.5, 0.1], [0.0, 0.3]])
        A2 = np.array([[0.1, 0.0], [0.0, 0.1]])
        res = FactorResults(
            target="y", nowcast=frame, transition=np.hstack([A1, A2]), factor_lags=2, n_shocks=1
        )
        mats = res.transition_matrices()
        np.testing.assert_array_equal(mats[0], A1)
        np.testing.assert_array_equal(mats[1], A2)
        text = res.summary()
        assert "Factor dynamics" in text and "Max |eigenvalue|" in text

    def test_transition_errors(self, frame):
        with pytest.raises(ValueError, match="required"):
            FactorResults(target="y", nowcast=frame).transition_matrices()
        bad = FactorResults(target="y", nowcast=frame, transition=np.eye(2), factor_lags=2)
        with pytest.raises(ValueError, match="expected"):
            bad.transition_matrices()

    def test_summary_without_transition(self, frame):
        text = FactorResults(target="y", nowcast=frame, factor_lags=1).summary()
        assert "Factor dynamics" in text and "Max |eigenvalue|" not in text

    def test_companion_eigenvalues_match_var_stability(self):
        # AR(2) scalar: x_t = 0.5 x_{t-1} + 0.3 x_{t-2}; roots of z^2 - 0.5 z - 0.3
        comp = _companion(np.array([[0.5, 0.3]]))
        np.testing.assert_allclose(
            np.sort(np.linalg.eigvals(comp)), np.sort(np.roots([1, -0.5, -0.3]))
        )
        with pytest.raises(ValueError, match="not"):
            _companion(np.ones((2, 3)))

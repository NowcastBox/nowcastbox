"""MixedFreqDFM with calendar-aware aggregation: weekly + monthly + quarterly (innovation I1)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import ConvergenceWarning
from nowcastbox.core.frequency import is_period_end
from nowcastbox.models import MixedFreqDFM
from nowcastbox.models._em_steps import (
    EMParameters,
    StateLayout,
    build_state_space,
    signal_mean,
    signal_variance,
)
from nowcastbox.models.em import _extension_periods, build_layout, series_calendar
from nowcastbox.preprocessing.aggregation import calendar_aggregation


@dataclass
class WeeklySimulation:
    data: pd.DataFrame
    common: pd.DataFrame
    frequencies: dict[str, str]
    aggregations: dict[str, str]

    def panel(self, data: pd.DataFrame | None = None) -> MixedFrequencyData:
        frame = self.data if data is None else data
        return MixedFrequencyData(frame, self.frequencies, aggregations=self.aggregations)


def simulate_weekly_dfm(
    n_weeks: int = 520,
    n_weekly: int = 4,
    n_monthly: int = 5,
    *,
    seed: int = 0,
    start: str = "2010-01-03",
    phi: float = 0.9,
    noise: float = 0.6,
) -> WeeklySimulation:
    """One weekly AR(1) factor; weekly series, monthly averages and quarterly GDP growth.

    Monthly series are calendar averages of a latent weekly series (4 or 5 weeks), the
    quarterly target is the growth rate of the quarterly average of a latent weekly log
    level (calendar Mariano-Murasawa weights over 12 to 14 + 12 to 14 weeks); white
    measurement noise is added at the observation dates.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range(start, periods=n_weeks, freq="W")
    f = np.zeros(n_weeks)
    for t in range(1, n_weeks):
        f[t] = phi * f[t - 1] + np.sqrt(1.0 - phi**2) * rng.standard_normal()
    factor = pd.Series(f, index=idx)
    data: dict[str, np.ndarray] = {}
    common: dict[str, np.ndarray] = {}
    for j in range(n_weekly):
        lam = rng.uniform(0.6, 1.2)
        common[f"w{j}"] = lam * f
        data[f"w{j}"] = lam * f + noise * rng.standard_normal(n_weeks)
    month_end = is_period_end(idx, "M")
    monthly = calendar_aggregation("W", "M", "average").apply(factor).to_numpy()
    for j in range(n_monthly):
        lam = rng.uniform(0.6, 1.2)
        c = np.where(month_end, lam * monthly, np.nan)
        common[f"m{j}"] = c
        data[f"m{j}"] = c + 0.3 * noise * rng.standard_normal(n_weeks)
    quarter_end = is_period_end(idx, "Q")
    quarterly = calendar_aggregation("W", "Q").apply(factor).to_numpy()
    common["gdp"] = np.where(quarter_end, 0.8 * quarterly, np.nan)
    data["gdp"] = common["gdp"] + 0.2 * rng.standard_normal(n_weeks)
    frequencies = {**{f"w{j}": "W" for j in range(n_weekly)}}
    frequencies |= {f"m{j}": "M" for j in range(n_monthly)} | {"gdp": "Q"}
    return WeeklySimulation(
        pd.DataFrame(data, index=idx),
        pd.DataFrame(common, index=idx),
        frequencies,
        {f"m{j}": "average" for j in range(n_monthly)},
    )


def _fit(model: MixedFreqDFM, panel: MixedFrequencyData, target: str = "gdp", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return model.fit(panel, target, **kw)


@pytest.fixture(scope="module")
def weekly_sim() -> WeeklySimulation:
    return simulate_weekly_dfm()


@pytest.fixture(scope="module")
def hidden_quarters(weekly_sim):
    return weekly_sim.data.index[is_period_end(weekly_sim.data.index, "Q")][-3:]


@pytest.fixture(scope="module")
def weekly_fit(weekly_sim, hidden_quarters):
    data = weekly_sim.data.copy()
    data.loc[hidden_quarters, "gdp"] = np.nan  # the last three quarters are not released
    data.iloc[-2:, [c.startswith("m") for c in data.columns]] = np.nan
    return _fit(MixedFreqDFM(idiosyncratic="iid", max_iter=100), weekly_sim.panel(data))


class TestWeeklyMonthlyQuarterly:
    def test_time_varying_layout(self, weekly_fit):
        layout = weekly_fit.state_layout
        assert layout.is_time_varying
        assert layout.block_lags == (27,)  # quarterly growth: 14 + 14 - 1 weekly lags
        kinds = [None if c is None else c.low.value for c in layout.calendar]
        assert kinds == [None] * 4 + ["M"] * 5 + ["Q"]
        model, _, grid = weekly_fit.state_space_model()
        assert model.is_time_varying and model.n_periods == len(grid)
        assert model.design_store.shape[0] < len(grid)  # one design per calendar pattern

    def test_converges_monotonically(self, weekly_fit):
        path = weekly_fit.loglikelihood_path
        assert weekly_fit.converged
        assert np.all(np.diff(path) >= -1e-8 * np.abs(path[1:]))
        assert weekly_fit.info["n_loglikelihood_decreases"] == 0

    def test_recovers_common_components(self, weekly_fit, weekly_sim):
        n = len(weekly_sim.data)
        # weekly series are noisier (signal-to-noise about 2): weaker threshold
        for name, bound in (("w0", 0.95), ("m0", 0.98), ("m3", 0.98), ("gdp", 0.98)):
            truth = weekly_sim.common[name].to_numpy()
            fitted = weekly_fit.common_component[name].to_numpy()[:n]
            ok = np.isfinite(truth)
            assert np.corrcoef(truth[ok], fitted[ok])[0, 1] > bound, name

    def test_nowcasts_hidden_quarters_accurately(self, weekly_fit, weekly_sim, hidden_quarters):
        truth = weekly_sim.data.loc[hidden_quarters, "gdp"].to_numpy()
        quarters = hidden_quarters.asfreq("Q")
        estimate = weekly_fit.nowcast["out_of_sample"].reindex(quarters).to_numpy()
        spread = float(np.nanstd(weekly_sim.data["gdp"]))
        assert np.all(np.isfinite(estimate))
        assert np.max(np.abs(estimate - truth)) < 0.25 * spread
        std = weekly_fit.nowcast["std"].reindex(quarters).to_numpy()
        assert np.all(std > 0) and np.all(std < 0.5 * spread)

    def test_loadings_have_the_simulated_sign(self, weekly_fit):
        loadings = weekly_fit.loadings.iloc[:, 0]
        assert bool((np.sign(loadings) == np.sign(loadings.iloc[0])).all())

    def test_params_report_calendar_weights(self, weekly_fit):
        params = weekly_fit.params
        paths = params["aggregation_weight_paths"]
        assert set(paths) == {"m0", "m1", "m2", "m3", "m4", "gdp"}
        gdp = paths["gdp"]
        grid = weekly_fit.grid
        slots = is_period_end(grid, "Q")
        sums = 2.0 * gdp.to_numpy()[slots].sum(axis=1)[1:]  # n_T + n_{T-1}: 24 to 28 weeks
        assert np.allclose(sums, np.round(sums)) and sums.min() >= 24 and sums.max() <= 28
        assert params["aggregation_weights"]["gdp"].size == 27
        assert params["design"].shape == (len(grid), 10, weekly_fit.state_layout.n_states)

    def test_predict_on_a_longer_vintage(self, weekly_sim):
        short = weekly_sim.panel().truncate(end=weekly_sim.data.index[-30])
        res = _fit(MixedFreqDFM(idiosyncratic="iid", max_iter=5), short)
        longer = weekly_sim.panel()
        pred = res.predict(longer)
        assert pred.shape == (len(longer.index), 10)
        model, _, _ = res.state_space_model(len(longer.index))
        assert model.n_periods == len(longer.index)
        sm = res.smooth(longer)
        assert sm.smoothed_state.shape == (len(longer.index), res.state_layout.n_states)
        assert "E-step filter" in res.summary()

    def test_structured_equals_dense(self, weekly_sim):
        panel = weekly_sim.panel(weekly_sim.data.iloc[:260])
        fast = _fit(MixedFreqDFM(max_iter=3, tol=0.0, filter_method="structured"), panel)
        dense = _fit(MixedFreqDFM(max_iter=3, tol=0.0, filter_method="univariate"), panel)
        assert np.allclose(fast.loglikelihood_path, dense.loglikelihood_path, rtol=0, atol=1e-6)
        assert fast.get_nowcast() == pytest.approx(dense.get_nowcast(), abs=1e-6)
        assert np.all(
            np.diff(fast.loglikelihood_path) >= -1e-8 * np.abs(fast.loglikelihood_path[1:])
        )

    def test_ar1_idiosyncratic_components(self, weekly_sim):
        panel = weekly_sim.panel(weekly_sim.data.iloc[:260])
        res = _fit(MixedFreqDFM(max_iter=15), panel, horizon=1)
        layout = res.state_layout
        # factor lags + weekly chains + monthly averages (<= 5 weeks) + quarterly (27 weeks)
        assert layout.n_states == 27 + 4 * 1 + 5 * 5 + 27
        path = res.loglikelihood_path
        assert np.all(np.diff(path) >= -1e-8 * np.abs(path[1:]))
        assert np.isfinite(res.get_nowcast())
        assert res.nowcast.index[-1] > res.data.to_native("gdp", dropna=True).index[-1]

    def test_warm_start(self, weekly_fit, weekly_sim):
        res = _fit(MixedFreqDFM(idiosyncratic="iid", max_iter=2, init=weekly_fit), weekly_fit.data)
        assert res.loglikelihood_path[0] == pytest.approx(weekly_fit.loglikelihood, rel=1e-8)

    @pytest.mark.parametrize(
        "options",
        [
            {"idiosyncratic": "student_t", "df": 5.0},
            {"outliers": "auto"},
            {"covid": "dummy", "covid_window": ("2012-03", "2012-05")},
            {"exclude_periods": ["2011-02"]},
            {"long_run_mean": "time_varying", "long_run_variance": 1e-3},
        ],
    )
    def test_robust_and_long_run_options(self, weekly_sim, options):
        panel = weekly_sim.panel(weekly_sim.data.iloc[:220])
        res = _fit(MixedFreqDFM(max_iter=4, **options), panel)
        assert np.isfinite(res.get_nowcast())
        assert np.all(np.isfinite(res.loglikelihood_path))


class TestCalendarLayout:
    def test_series_calendar(self, weekly_sim):
        specs = series_calendar(weekly_sim.panel())
        assert specs[0] is None
        assert specs[4] is not None and specs[4].low.value == "M"
        assert specs[-1] is not None and specs[-1].n_weights == 27

    def test_weight_paths(self, weekly_sim):
        panel, _ = weekly_sim.panel(weekly_sim.data.iloc[:60]).standardize()
        layout = build_layout(panel, 1, 1, None, "iid")
        fixed = layout.weight_path(0, 60)
        assert fixed.shape == (60, 1) and np.all(fixed == 1.0)
        monthly = layout.weight_path(4, 60)
        slots = is_period_end(panel.index, "M")
        assert np.allclose(monthly[slots].sum(axis=1), 1.0)  # averages
        assert layout.weight_path(4, 60) is monthly  # cached

    def test_build_state_space_needs_n_periods(self, weekly_sim):
        panel, _ = weekly_sim.panel(weekly_sim.data.iloc[:60]).standardize()
        layout = build_layout(panel, 1, 1, None, "iid")
        params = _ones_params(layout)
        with pytest.raises(ValueError, match="n_periods"):
            build_state_space(params, layout)
        model = build_state_space(params, layout, 60)
        states = np.ones((60, layout.n_states))
        signal = signal_mean(model, states)
        assert np.allclose(signal[:, :4], 1.0)  # weekly series: loading one
        assert np.all(np.isfinite(signal_variance(model, np.ones((60, 27, 27)))))

    def test_layout_validation(self):
        spec = calendar_aggregation("W", "M")
        args = (["a", "b"], ["g"], [1], 1, np.ones((2, 1), bool), [[1.0], [1.0]], "iid")
        with pytest.raises(ValueError, match="start"):
            StateLayout(*args, calendar=[None, spec])
        with pytest.raises(ValueError, match="base frequency"):
            StateLayout(*args, calendar=[None, spec], start=pd.Period("2020-01", "M"))
        with pytest.raises(ValueError, match="common base"):
            StateLayout(
                *args,
                calendar=[calendar_aggregation("D", "M"), spec],
                start=pd.Period("2020-01-05", "W"),
            )
        with pytest.raises(ValueError, match="one entry per series"):
            StateLayout(*args, calendar=[spec])
        fixed = StateLayout(*args, calendar=[None, calendar_aggregation("M", "Q")])
        assert not fixed.is_time_varying  # fixed-ratio pairs keep fixed weights

    def test_compatibility_depends_on_start(self):
        spec = calendar_aggregation("W", "M")
        args = (["a", "b"], ["g"], [1], 1, np.ones((2, 1), bool), [[1.0], [1.0]], "iid")
        one = StateLayout(*args, calendar=[None, spec], start=pd.Period("2020-01-05", "W"))
        same = StateLayout(*args, calendar=[None, spec], start=pd.Period("2020-01-05", "W"))
        later = StateLayout(*args, calendar=[None, spec], start=pd.Period("2020-01-12", "W"))
        assert one.is_compatible(same)
        assert not one.is_compatible(later)

    def test_extension_periods(self, weekly_sim):
        panel = weekly_sim.panel(weekly_sim.data.iloc[:100])
        end = panel.index[-1]
        n = _extension_periods(panel, "gdp", 0)
        slot = pd.period_range(end, periods=n + 1, freq="W")[-1]
        assert is_period_end(pd.PeriodIndex([slot]), "Q")[0] or n == 0
        assert _extension_periods(panel, "gdp", 1) - n in (12, 13, 14)


class TestDailyGrid:
    def test_weekly_flows_on_a_daily_grid(self):
        """D -> W is a fixed ratio (7): the fixed-weight model, no time variation."""
        rng = np.random.default_rng(3)
        idx = pd.period_range("2021-01-04", periods=280, freq="D")
        f = np.zeros(280)
        for t in range(1, 280):
            f[t] = 0.95 * f[t - 1] + rng.standard_normal() * 0.3
        weekly = pd.Series(f, index=idx).rolling(7).sum().to_numpy()
        frame = pd.DataFrame(
            {
                "d1": f + 0.3 * rng.standard_normal(280),
                "d2": 0.8 * f + 0.3 * rng.standard_normal(280),
                "w": np.where(is_period_end(idx, "W"), weekly, np.nan),
            },
            index=idx,
        )
        panel = MixedFrequencyData(
            frame, {"d1": "D", "d2": "D", "w": "W"}, aggregations={"w": "flow"}
        )
        res = _fit(MixedFreqDFM(idiosyncratic="iid", max_iter=20), panel, target="w")
        assert not res.state_layout.is_time_varying
        assert res.state_layout.weights[2].tolist() == [1.0] * 7
        assert np.isfinite(res.get_nowcast())


def _ones_params(layout: StateLayout) -> EMParameters:
    loadings = np.zeros((layout.n_series, layout.n_factor_states))
    loadings[:, 0] = 1.0
    n = layout.n_series
    return EMParameters(
        (np.array([[0.5]]),), (np.eye(1),), loadings, np.zeros(n), np.ones(n), np.ones(n)
    )

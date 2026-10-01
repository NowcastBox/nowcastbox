"""End-to-end integration tests of the wave-1 modules.

Simulated mixed-frequency dataset in levels -> ``prepare_panel`` -> ``select_factors``
-> ``TwoStepDFM`` / ``MixedFreqDFM`` -> nowcast; pseudo real-time vintages -> refit ->
nowcast changes. Everything goes through the top-level ``nowcastbox`` namespace, as a
user would.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import nowcastbox as nb

pytestmark = pytest.mark.integration

N_MONTHLY = 30
N_FACTORS = 2
START = "2005-01"
N_MONTHS = 180  # 2005-01 .. 2019-12
TARGET = "gdp"
MM = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 3.0


@dataclass(frozen=True)
class Simulated:
    """Levels panel plus the true quantities of the DGP."""

    levels: nb.MixedFrequencyData
    factors: pd.DataFrame
    gdp_growth: pd.Series  # true quarterly growth (native quarterly index)


def simulate_levels(seed: int = 7) -> Simulated:
    """Monthly predictors and quarterly GDP **in levels** driven by two factors.

    Monthly growth: ``dx_t = Lambda f_t + e_t``; levels are cumulative sums (so the
    ``"diff"`` transformation recovers the growth). Quarterly GDP growth is the
    Mariano-Murasawa aggregate of a latent monthly growth loading on both factors;
    the level is its cumulative sum over quarters, stored in the third month.
    Monthly series have release delays of 15/30/45 days and GDP of 60 days.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range(START, periods=N_MONTHS, freq="M")
    a = np.array([[0.7, 0.1], [0.0, 0.5]])
    f = np.zeros((N_MONTHS, N_FACTORS))
    for t in range(1, N_MONTHS):
        f[t] = a @ f[t - 1] + rng.normal(size=N_FACTORS)
    # loadings bounded away from zero: every series is informative (with N(0, 1)
    # loadings, standardisation makes the noise strongly heteroskedastic and the
    # Bai-Ng criteria over-select at this sample size)
    lam = rng.uniform(0.5, 1.5, size=(N_MONTHLY, N_FACTORS)) * rng.choice(
        [-1.0, 1.0], size=(N_MONTHLY, N_FACTORS)
    )
    growth = f @ lam.T + 0.5 * rng.normal(size=(N_MONTHS, N_MONTHLY))
    levels = 100.0 + np.cumsum(growth, axis=0)
    frame = pd.DataFrame(levels, index=idx, columns=[f"x{i:02d}" for i in range(N_MONTHLY)])

    latent = f @ np.array([0.8, 0.5]) + 0.2 * rng.normal(size=N_MONTHS)
    quarterly_growth = np.full(N_MONTHS, np.nan)
    for t in range(4, N_MONTHS):
        if t % 3 == 2:
            quarterly_growth[t] = MM @ latent[t - 4 : t + 1][::-1] + 0.5
    q_slots = np.flatnonzero(~np.isnan(quarterly_growth))
    gdp_level = np.full(N_MONTHS, np.nan)
    gdp_level[q_slots] = 1000.0 + np.cumsum(quarterly_growth[q_slots])
    # first quarter level before the first growth observation
    gdp_level[q_slots[0] - 3] = 1000.0
    frame[TARGET] = gdp_level

    freqs = dict.fromkeys(frame.columns[:-1], "M") | {TARGET: "Q"}
    delays = {c: (15, 30, 45)[i % 3] for i, c in enumerate(frame.columns[:-1])} | {TARGET: 60}
    transforms = dict.fromkeys(frame.columns, "diff")
    levels_panel = nb.MixedFrequencyData(frame, freqs, transforms=transforms, release_delays=delays)
    q_index = pd.PeriodIndex(idx[q_slots], freq="M").asfreq("Q")
    return Simulated(
        levels=levels_panel,
        factors=pd.DataFrame(f, index=idx, columns=["f1", "f2"]),
        gdp_growth=pd.Series(quarterly_growth[q_slots], index=q_index, name=TARGET),
    )


@pytest.fixture(scope="module")
def sim() -> Simulated:
    return simulate_levels()


@pytest.fixture(scope="module")
def panel(sim: Simulated) -> nb.MixedFrequencyData:
    """Final-vintage panel prepared from the levels (transform from metadata)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nb.DataQualityWarning)
        out = nb.prepare_panel(sim.levels, keep=TARGET)
    assert isinstance(out, nb.MixedFrequencyData)
    return out


def _corr(a: pd.Series, b: pd.Series) -> float:
    x, y = a.align(b, join="inner")
    ok = x.notna() & y.notna()
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _canonical_r2(estimated: pd.DataFrame, true: pd.DataFrame) -> float:
    """R^2 of the regression of the true factors on the estimated ones (span check)."""
    x, y = estimated.align(true, join="inner", axis=0)
    ok = x.notna().all(axis=1) & y.notna().all(axis=1)
    xm = np.column_stack([np.ones(int(ok.sum())), x[ok].to_numpy()])
    ym = y[ok].to_numpy()
    beta, *_ = np.linalg.lstsq(xm, ym, rcond=None)
    resid = ym - xm @ beta
    return float(1.0 - resid.var(axis=0).sum() / ym.var(axis=0).sum())


# ---------------------------------------------------------------------- preprocessing
class TestPreparePanel:
    def test_transforms_from_metadata_and_flag(self, sim: Simulated, panel) -> None:
        assert set(panel.columns) == set(sim.levels.columns)
        assert all(panel.metadata[c].transform_applied for c in panel.columns)
        native = panel.to_native(TARGET).dropna()
        truth = sim.gdp_growth.reindex(native.index)
        np.testing.assert_allclose(native.to_numpy(), truth.to_numpy(), atol=1e-8)

    def test_preparing_twice_is_idempotent(self, panel) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", nb.DataQualityWarning)
            again = nb.prepare_panel(panel, keep=TARGET)
        pd.testing.assert_frame_equal(again.data, panel.data)


# ---------------------------------------------------------------------- selection
def test_select_factors_finds_true_number(panel) -> None:
    predictors = panel.drop([TARGET])
    sel = nb.select_factors(predictors, rmax=8, criterion="IC2")
    assert sel.r_star == N_FACTORS
    shocks = nb.select_shocks(predictors, n_factors=sel.r_star, factor_lags=1)
    assert 1 <= shocks.q_star <= sel.r_star


# ---------------------------------------------------------------------- models
@pytest.fixture(scope="module")
def two_step(panel) -> nb.TwoStepResults:
    return nb.TwoStepDFM(n_factors=N_FACTORS, factor_lags=1).fit(panel, target=TARGET)


@pytest.fixture(scope="module")
def em(panel) -> nb.MixedFreqDFMResults:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nb.ConvergenceWarning)
        return nb.MixedFreqDFM(n_factors=N_FACTORS, factor_lags=1, max_iter=100, tol=1e-5).fit(
            panel, target=TARGET, horizon=1
        )


@pytest.mark.parametrize("fixture", ["two_step", "em"])
def test_models_recover_factors_and_target(fixture: str, request, sim: Simulated) -> None:
    res = request.getfixturevalue(fixture)
    assert isinstance(res, nb.NowcastResults)
    assert _canonical_r2(res.factors, sim.factors) > 0.85
    fitted = res.nowcast["in_sample"]
    assert _corr(fitted, sim.gdp_growth) > 0.8
    # the last quarter of the panel is not observed and is nowcast
    last = res.nowcast.index[res.nowcast["observed"].notna()][-1]
    assert np.isfinite(res.nowcast.loc[last + 1, "out_of_sample"])
    assert res.summary().count("\n") > 10


def test_two_step_and_em_nowcasts_agree(two_step, em) -> None:
    common = two_step.nowcast.index.intersection(em.nowcast.index)
    a = two_step.nowcast.loc[common, "in_sample"]
    b = em.nowcast.loc[common, "in_sample"]
    assert _corr(a, b) > 0.85


# ---------------------------------------------------------------------- high-level API
def test_nowcast_api_with_bai_ng(panel) -> None:
    res = nb.nowcast(panel, TARGET, method="two_step")
    assert res.info["method"] == "two_step"
    assert res.info["selection"].r_star == N_FACTORS
    assert res.n_factors == N_FACTORS


def test_nowcast_api_em_matches_direct_fit(panel, em) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nb.ConvergenceWarning)
        res = nb.nowcast(panel, TARGET, method="em", max_iter=100, tol=1e-5)
    pd.testing.assert_frame_equal(res.nowcast, em.nowcast)


def test_nowcast_api_from_levels_with_preprocess(sim: Simulated) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nb.DataQualityWarning)
        res = nb.nowcast(sim.levels, TARGET, method="two_step", preprocess=True)
    assert res.n_factors == N_FACTORS
    assert _corr(res.nowcast["in_sample"], sim.gdp_growth) > 0.8


# ---------------------------------------------------------------------- vintages
VINTAGES = ("2019-10-20", "2019-11-20", "2019-12-20", "2020-01-20", "2020-03-05")


@pytest.fixture(scope="module")
def vintage_panels(sim: Simulated) -> dict[str, nb.MixedFrequencyData]:
    out: dict[str, nb.MixedFrequencyData] = {}
    for v in VINTAGES:
        levels = nb.pseudo_real_time(sim.levels, vintage=v)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", nb.DataQualityWarning)
            prepared = nb.prepare_panel(levels, keep=TARGET)
        out[v] = prepared
    return out


def test_vintages_are_nested(vintage_panels) -> None:
    counts = [int(vintage_panels[v].observation_mask().to_numpy().sum()) for v in VINTAGES]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]
    # GDP 2019Q4 (released 60 days after 2019-12-31, i.e. 2020-02-29) is only in the last one
    for v in VINTAGES[:-1]:
        assert np.isnan(vintage_panels[v].data.loc[pd.Period("2019-12", "M"), TARGET])
    assert np.isfinite(vintage_panels[VINTAGES[-1]].data.loc[pd.Period("2019-12", "M"), TARGET])


@pytest.mark.parametrize("method", ["two_step", "em"])
def test_refit_on_new_vintages_changes_nowcast(method: str, vintage_panels, sim) -> None:
    target_q = pd.Period("2019Q4", "Q")
    nowcasts: list[float] = []
    for v in VINTAGES[:-1]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", nb.ConvergenceWarning)
            extra = {"max_iter": 50} if method == "em" else {}
            res = nb.nowcast(vintage_panels[v], TARGET, method=method, n_factors=N_FACTORS, **extra)
        value = res.nowcast.loc[target_q, "out_of_sample"]
        assert np.isfinite(value)
        nowcasts.append(float(value))
    # new information moves the nowcast ...
    assert len(set(np.round(nowcasts, 10))) == len(nowcasts)
    # ... and the last nowcast (all monthly data of the quarter) is closer to the truth
    truth = float(sim.gdp_growth.loc[target_q])
    errors = np.abs(np.array(nowcasts) - truth)
    assert errors[-1] < errors[0] + 0.5


def test_calendar_and_store_reproduce_pseudo_real_time(sim: Simulated) -> None:
    vintage = "2019-11-20"
    cal = nb.ReleaseCalendar.from_data(sim.levels)
    by_calendar = nb.pseudo_real_time(sim.levels, cal, vintage=vintage)
    by_metadata = nb.pseudo_real_time(sim.levels, vintage=vintage)
    pd.testing.assert_frame_equal(by_calendar.data, by_metadata.data)
    store = nb.VintageStore.from_calendar(sim.levels)
    known = store.as_of(vintage, start=sim.levels.start, end=sim.levels.end)
    pd.testing.assert_frame_equal(
        known.reindex(columns=by_metadata.columns),
        by_metadata.data,
        check_names=False,
        check_freq=False,
    )

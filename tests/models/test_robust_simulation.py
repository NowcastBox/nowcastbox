"""Simulation tests of the robust EM (innovation I3) and simulators used by the robust tests.

The simulators are imported by ``tests/models/test_robust.py`` as
``tests.models.test_robust_simulation``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.exceptions import ConvergenceWarning
from nowcastbox.models import MixedFreqDFM

MM = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 3.0


@dataclass
class RobustPanel:
    data: pd.DataFrame  # monthly grid, quarterly "gdp" in the 3rd month
    truth: pd.Series  # noise-free target on the held-out quarters (quarterly index)
    factor: np.ndarray
    outliers: np.ndarray  # (n, N + 1) injected outlier positions


def _ar1_factor(rng: np.random.Generator, n: int, rho: float, impulses: np.ndarray) -> np.ndarray:
    shocks = rng.standard_normal(n + 50)
    f = np.zeros(n + 50)
    for t in range(1, n + 50):
        f[t] = rho * f[t - 1] + shocks[t] + impulses[t]
    return f[50:]


def _aggregate(f: np.ndarray) -> np.ndarray:
    out = np.full(f.size, np.nan)
    for t in range(MM.size - 1, f.size):
        out[t] = MM @ f[t - np.arange(MM.size)]
    return out


def _finish(
    x: np.ndarray, gdp_true: np.ndarray, gdp_noise: np.ndarray, idx: pd.PeriodIndex, hold: int
) -> tuple[pd.DataFrame, pd.Series]:
    n = idx.size
    gdp = gdp_true + gdp_noise
    gdp[np.arange(n) % 3 != 2] = np.nan
    frame = pd.DataFrame(x, index=idx, columns=[f"x{i}" for i in range(x.shape[1])])
    frame["gdp"] = gdp
    slots = np.flatnonzero(np.arange(n) % 3 == 2)
    hidden = slots[-hold:]
    frame.iloc[hidden, -1] = np.nan
    truth = pd.Series(gdp_true[hidden], index=idx[hidden].asfreq("Q"))
    return frame, truth


def simulate_heavy_tails(
    seed: int,
    *,
    n: int = 240,
    n_series: int = 15,
    df: float = 3.0,
    outlier_share: float = 0.02,
    outlier_size: float = 10.0,
    hold: int = 20,
) -> RobustPanel:
    """One-factor monthly/quarterly panel with Student-t noise and injected outliers.

    The last ``hold`` quarters of the target are hidden; ``truth`` holds the noise-free
    target there (common component), the nowcast benchmark.
    """
    rng = np.random.default_rng(seed)
    f = _ar1_factor(rng, n, 0.7, np.zeros(n + 50))
    lam = rng.uniform(0.5, 1.5, n_series)
    x = np.outer(f, lam) + 0.7 * rng.standard_t(df, size=(n, n_series))
    flags = rng.random((n, n_series)) < outlier_share
    x[flags] += outlier_size * rng.choice([-1.0, 1.0], int(flags.sum()))
    idx = pd.period_range("2000-01", periods=n, freq="M")
    frame, truth = _finish(x, _aggregate(f), 0.3 * rng.standard_t(df, n), idx, hold)
    outliers = np.hstack([flags, np.zeros((n, 1), bool)])
    return RobustPanel(frame, truth, f, outliers)


def simulate_pandemic(seed: int, *, n_series: int = 12, hold: int = 8) -> RobustPanel:
    """Panel on 2005-2024 whose factor gets huge impulses in 2020-03..2020-06."""
    idx = pd.period_range("2005-01", "2024-12", freq="M")
    n = idx.size
    rng = np.random.default_rng(seed)
    impulses = np.zeros(n + 50)
    start = 50 + idx.get_loc(pd.Period("2020-03", "M"))
    impulses[start : start + 4] = [-12.0, -10.0, 15.0, 8.0]
    f = _ar1_factor(rng, n, 0.7, impulses)
    lam = rng.uniform(0.5, 1.5, n_series)
    x = np.outer(f, lam) + 0.6 * rng.standard_normal((n, n_series))
    frame, truth = _finish(x, _aggregate(f), 0.3 * rng.standard_normal(n), idx, hold)
    return RobustPanel(frame, truth, f, np.zeros((n, n_series + 1), bool))


def nowcast_rmse(model: MixedFreqDFM, panel: RobustPanel) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = model.fit(panel.data, "gdp", frequency={"gdp": "Q"})
    est = res.nowcast["out_of_sample"].reindex(panel.truth.index)
    return float(np.sqrt(np.mean((est - panel.truth) ** 2)))


# ====================================================================== tests
def test_simulators_shapes():
    panel = simulate_heavy_tails(0, n=60, n_series=4, hold=2)
    assert panel.data.shape == (60, 5)
    assert panel.truth.index.freqstr.startswith("Q")
    assert panel.data["gdp"].notna().sum() == 17  # first quarter lacks 5 months
    pandemic = simulate_pandemic(0, n_series=3)
    assert pandemic.data.index[0] == pd.Period("2005-01", "M")
    assert np.abs(pandemic.factor).max() > 8.0


def test_student_t_nowcasts_beat_gaussian_with_heavy_tails():
    """Heavy tails + outliers: Student-t errors give smaller nowcast errors (I3)."""
    gaussian, robust = [], []
    for seed in (10, 11, 12):
        panel = simulate_heavy_tails(seed)
        gaussian.append(nowcast_rmse(MixedFreqDFM(idiosyncratic="iid", max_iter=200), panel))
        robust.append(nowcast_rmse(MixedFreqDFM(idiosyncratic="student_t", max_iter=200), panel))
    assert np.mean(robust) < 0.85 * np.mean(gaussian), (robust, gaussian)
    assert sum(r < g for r, g in zip(robust, gaussian, strict=True)) >= 2


def test_outlier_masking_beats_gaussian():
    """Automatic outlier handling also reduces the nowcast errors."""
    plain, masked = [], []
    for seed in (10, 11, 12):
        panel = simulate_heavy_tails(seed)
        plain.append(nowcast_rmse(MixedFreqDFM(idiosyncratic="iid", max_iter=200), panel))
        masked.append(
            nowcast_rmse(MixedFreqDFM(idiosyncratic="iid", outliers="auto", max_iter=200), panel)
        )
    assert np.mean(masked) < 0.95 * np.mean(plain), (masked, plain)


def test_outlier_flags_find_injected_outliers():
    panel = simulate_heavy_tails(3, df=30.0, outlier_share=0.01, outlier_size=12.0)
    res = MixedFreqDFM(idiosyncratic="iid", outliers="auto", max_iter=100).fit(
        panel.data, "gdp", frequency={"gdp": "Q"}
    )
    assert res.outlier_flags is not None
    flags = res.outlier_flags.to_numpy()
    injected = panel.outliers
    recall = (flags & injected).sum() / injected.sum()
    precision = (flags & injected).sum() / max(flags.sum(), 1)
    assert recall > 0.9 and precision > 0.8, (recall, precision)
    assert res.info["n_outliers"] == int(flags.sum())
    assert 1 <= res.info["n_outlier_passes"] <= 3


def test_student_t_weights_downweight_outliers_and_estimate_df():
    panel = simulate_heavy_tails(4, df=3.0)
    res = MixedFreqDFM(idiosyncratic="student_t", max_iter=200).fit(
        panel.data, "gdp", frequency={"gdp": "Q"}
    )
    assert res.observation_weights is not None and res.student_t_df is not None
    weights = res.observation_weights.to_numpy()
    observed = ~np.isnan(weights)
    assert np.nanmean(weights[panel.outliers & observed]) < 0.1
    assert np.nanmedian(weights[~panel.outliers & observed]) > 0.8
    assert 2.0 <= res.student_t_df < 10.0
    assert res.info["objective"] == "evidence_lower_bound"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"idiosyncratic": "student_t"},
        {"idiosyncratic": "student_t", "df": 4.0},
        {"idiosyncratic": "iid", "covid": "dummy"},
        {"covid": "dummy"},
        {"idiosyncratic": "student_t", "covid": "dummy", "long_run_mean": "time_varying"},
    ],
)
def test_ecm_objective_is_monotone(kwargs):
    """Every ECM step increases the objective (log-likelihood or evidence lower bound)."""
    panel = simulate_pandemic(1) if "covid" in kwargs else simulate_heavy_tails(5, n=150)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = MixedFreqDFM(max_iter=40, tol=0.0, covid_window=("2020-03", "2020-08"), **kwargs).fit(
            panel.data, "gdp", frequency={"gdp": "Q"}
        )
    path = res.loglikelihood_path
    assert path is not None and path.size == 41
    diffs = np.diff(path)
    assert np.all(diffs >= -1e-7 * np.abs(path[1:])), diffs.min()
    assert res.info["n_loglikelihood_decreases"] == 0


def test_covid_treatments_protect_the_factor_var():
    """Masking or dummying the pandemic keeps the VAR coefficient near its true value."""
    panel = simulate_pandemic(0)
    estimates = {}
    for covid in ("none", "mask", "dummy"):
        res = MixedFreqDFM(covid=covid, covid_window=("2020-03", "2020-08"), max_iter=100).fit(
            panel.data, "gdp", frequency={"gdp": "Q"}
        )
        estimates[covid] = float(res.transition[0, 0])
    assert abs(estimates["mask"] - 0.7) < abs(estimates["none"] - 0.7)
    assert abs(estimates["dummy"] - 0.7) < abs(estimates["none"] - 0.7)


def test_covid_dummy_keeps_the_pandemic_in_the_factors():
    """With dummies the factors follow the data in the window (unlike masking)."""
    panel = simulate_pandemic(2)
    window = pd.period_range("2020-03", "2020-06", freq="M")
    fits = {
        covid: MixedFreqDFM(covid=covid, covid_window=("2020-03", "2020-08"), max_iter=100).fit(
            panel.data, "gdp", frequency={"gdp": "Q"}
        )
        for covid in ("mask", "dummy")
    }
    truth = pd.Series(panel.factor, index=panel.data.index).loc[window]
    errors = {}
    for covid, res in fits.items():
        f = res.factors.iloc[:, 0].loc[window]
        scale = np.sign(np.corrcoef(f, truth)[0, 1]) * np.std(truth) / np.std(f)
        errors[covid] = float(np.abs(f * scale - truth).mean())
    assert errors["dummy"] < 0.5 * errors["mask"], errors
    dummies = fits["dummy"].interventions
    assert dummies is not None
    assert dummies.index[0] == pd.Period("2020-03", "M") and len(dummies) == 6
    assert np.abs(dummies.to_numpy()).max() > 1.0

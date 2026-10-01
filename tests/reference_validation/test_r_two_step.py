"""Two-step DFM vs ``nowcast(method = "2s" / "2s_agg")`` of the R package ``nowcasting`` 1.1.2.

Fixtures: ``scripts/reference_fixtures/r_two_step.R`` - USGDP (the package example, the
Giannone, Reichlin & Small 2008 panel, r = 2, p = 2, q = 2) and the shared simulated panel
(r = 2, p = 1 or 2). The R-processed USGDP panels are rebuilt with the validated emulation
of ``Bpanel``.

The comparison is split in three layers (``docs/validation/two_step.md``):

1. **first step** (principal components, VAR, idiosyncratic variances) on the balanced
   rows, where both libraries standardise on the same sample: loadings and VAR matrices
   agree to ~1e-14; R's eigenvalues and ``Psi`` use ``1/(T-1)`` and its ``BB`` is the
   demeaned sample covariance (``ddof = 1``) of the VAR residuals, nowcastbox uses
   ``1/T`` and ``e'e/(T-p)``;
2. **second step + bridge** with R's own parameters (``Lambda``, ``A``, ``BB``, ``Psi``,
   ``initx``, ``initV``, ``mean``, ``std``) run through nowcastbox's Kalman smoother,
   factor aggregation and bridge OLS: factors, coefficients and nowcasts agree to
   ~1e-13 (plan tolerance 1e-6);
3. **full default pipelines** (``reference_divergence``): R standardises over the
   balanced rows only, nowcastbox over every observation; moment conventions and the
   initial state differ (R: first factor values and their covariance; nowcastbox: the
   stationary distribution). For ``q < r`` R's ``BB`` is not symmetric, so R's
   state-space model is invalid there.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nowcastbox.models import TwoStepDFM, fit_bridge_regression
from nowcastbox.preprocessing import rolling_aggregate
from nowcastbox.statespace import kalman_smoother
from tests.reference_validation._bpanel import emulate_bpanel
from tests.reference_validation._helpers import (
    TOL_TWO_STEP,
    abs_correlations,
    codes,
    column_signs,
    max_abs,
    nowcast_estimate,
    pad,
    quarter_end_only,
    r_estimate,
    r_standardize,
    r_two_step_state_space,
    read_json,
    read_periods,
)

pytestmark = pytest.mark.reference_validation

MM = np.array([1.0, 2.0, 3.0, 2.0, 1.0])


@dataclass(frozen=True)
class Case:
    """One R run: its JSON outputs, the predictor panel, the target and the variant."""

    spec: dict[str, Any]
    panel: pd.DataFrame
    target: pd.Series
    aggregate: str

    @property
    def weights(self) -> np.ndarray:
        return MM if self.aggregate == "factors" else np.ones(1)


@cache
def _usgdp() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw = read_periods("inputs/usgdp_base.csv.gz")
    spec = codes("usgdp")
    x_codes = {k: v for k, v in spec.items() if k != "RGDPGR"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filtered = emulate_bpanel(raw.drop(columns="RGDPGR"), x_codes, aggregate=True)
        plain = emulate_bpanel(raw, spec)
    return filtered, plain, raw["RGDPGR"]


@cache
def _simulated(balanced: bool) -> pd.DataFrame:
    sim = read_periods("inputs/simulated.csv")
    if balanced:
        sim = sim.loc[:"2019-09"]
    return pad(sim, 12)


@cache
def case(prefix: str) -> Case:
    spec = read_json(f"r/{prefix}.json")
    cols = list(spec["columns"])
    if prefix == "usgdp_2s":
        filtered, _, gdp = _usgdp()
        return Case(spec, filtered[cols], gdp.reindex(filtered.index), "variables")
    if prefix == "usgdp_2s_agg":
        _, plain, _ = _usgdp()
        return Case(spec, plain[cols], plain["RGDPGR"], "factors")
    if prefix == "sim_2s":
        panel = read_periods("r/sim_2s_panel.csv.gz")
        gdp = _simulated(False)["gdp"].reindex(panel.index)
        return Case(spec, panel[cols], gdp, "variables")
    sim = _simulated(prefix.startswith("simbal"))
    return Case(spec, sim[cols], sim["gdp"], "factors")


SECOND_STEP_CASES = [
    "usgdp_2s",
    "usgdp_2s_agg",
    "sim_2s",
    "sim_2s_agg",
    "simbal_2s_agg",
    "simbal_2s_agg_p2",
]


def _ours(c: Case, data: pd.DataFrame | None = None) -> Any:
    spec = c.spec
    frame = (c.panel if data is None else data).copy()
    frame["__y"] = quarter_end_only(c.target.reindex(frame.index))
    model = TwoStepDFM(
        n_factors=spec["r"],
        factor_lags=spec["p"],
        n_shocks=spec["q"],
        aggregate=c.aggregate,  # type: ignore[arg-type]
        aggregation=[1.0] if c.aggregate == "variables" else None,
        horizon=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(frame, "__y", frequency={"__y": "Q"})


# ---------------------------------------------------------------------- 1. first step
@pytest.mark.parametrize("prefix", SECOND_STEP_CASES)
def test_first_step_on_balanced_rows(prefix: str) -> None:
    c = case(prefix)
    spec, r, p = c.spec, int(c.spec["r"]), int(c.spec["p"])
    balanced = c.panel.dropna()
    n = len(balanced)
    # R standardises over the balanced rows: same statistics as ours on that sample
    assert max_abs(balanced.mean().to_numpy(), spec["mean"]) < 1e-12
    assert max_abs(balanced.std().to_numpy(), spec["std"]) < 1e-10
    res = _ours(c, balanced)
    lam_r = np.asarray(spec["Lambda"])
    lam = res.loadings.loc[list(spec["columns"])].to_numpy()
    s = column_signs(lam, lam_r)
    assert max_abs(lam * s, lam_r) < TOL_TWO_STEP
    a_r = np.asarray(spec["A"])[:r, : r * p]
    assert max_abs(res.A * np.outer(s, np.tile(s, p)), a_r) < TOL_TWO_STEP
    eig_r = np.asarray(spec["eigenvalues"])
    assert max_abs(res.eigenvalues.to_numpy() * n / (n - 1), eig_r) < TOL_TWO_STEP * eig_r[0]
    psi = res.Psi.loc[list(spec["columns"])].to_numpy()
    assert max_abs(psi * n / (n - 1), spec["Psi"]) < TOL_TWO_STEP
    # BB: demeaned sample covariance (ddof=1) of the VAR residuals of our own factors
    z = ((balanced - balanced.mean()) / balanced.std()).to_numpy()
    f = z @ lam
    lags = np.hstack([f[p - k : n - k] for k in range(1, p + 1)])
    resid = f[p:] - lags @ res.A.T
    bb_r = np.asarray(spec["BB"])[:r, :r]
    assert max_abs(np.cov(resid.T) * np.outer(s, s), bb_r) < TOL_TWO_STEP


# ---------------------------------------------------------------------- 2. second step
def _second_step(c: Case) -> tuple[pd.DataFrame, Any, pd.Series]:
    spec = c.spec
    r = int(spec["r"])
    z = r_standardize(c.panel, spec, "mean", "std")
    smoothed = kalman_smoother(r_two_step_state_space(spec), z)
    names = [f"f{k + 1}" for k in range(r)]
    factors = pd.DataFrame(smoothed.smoothed_state[:, :r], index=c.panel.index, columns=names)
    agg = factors.apply(
        lambda col: pd.Series(rolling_aggregate(col.to_numpy(), c.weights), index=factors.index)
    )
    quarterly = agg[agg.index.month % 3 == 0]  # type: ignore[attr-defined]
    quarterly.index = quarterly.index.asfreq("Q")  # type: ignore[attr-defined]
    y = quarter_end_only(c.target.reindex(c.panel.index)).dropna()
    y.index = y.index.asfreq("Q")  # type: ignore[attr-defined]
    bridge = fit_bridge_regression(y.reindex(quarterly.index).rename("y"), quarterly)
    return factors, bridge, bridge.predict(quarterly)


@pytest.mark.parametrize("prefix", SECOND_STEP_CASES)
def test_second_step_and_bridge_with_r_parameters(prefix: str) -> None:
    c = case(prefix)
    factors, bridge, prediction = _second_step(c)
    r_factors = read_periods(f"r/{prefix}_factors.csv")
    assert max_abs(factors.to_numpy(), r_factors.reindex(factors.index).to_numpy()) < TOL_TWO_STEP
    coef = np.asarray(list(c.spec["reg_coefficients"].values()))
    assert bridge.n_obs == c.spec["reg_nobs"]
    assert max_abs(bridge.params.to_numpy(), coef) < TOL_TWO_STEP
    assert bridge.sigma == pytest.approx(c.spec["reg_sigma"], abs=TOL_TWO_STEP)
    yfcst = read_periods(f"r/{prefix}_yfcst.csv", "Q")
    reference = r_estimate(yfcst)
    ours = prediction.reindex(reference.index)
    assert max_abs(ours, reference) < TOL_TWO_STEP
    # R leaves quarters before the first target observation empty; we never miss more
    assert not bool((ours.isna() & reference.notna()).any())


@pytest.mark.parametrize("prefix", ["usgdp_2s_agg", "sim_2s_agg"])
def test_predictor_fit_with_r_parameters(prefix: str) -> None:
    """R's ``xfcst`` from the same smoother output.

    R estimates on data standardised with the **balanced-row** statistics but maps the
    smoothed common component back with the mean / standard deviation of **all**
    observations (identified exactly here); nowcastbox uses one set of statistics for
    both (documented divergence: 0.012 at the end of the simulated horizon).
    """
    c = case(prefix)
    spec = c.spec
    cols = list(spec["columns"])
    z = r_standardize(c.panel, spec, "mean", "std")
    signal = kalman_smoother(r_two_step_state_space(spec), z).smoothed_signal()
    xfcst = read_periods(f"r/{prefix}_xfcst.csv.gz")
    full = c.panel[cols]
    fitted = pd.DataFrame(
        signal * full.std().to_numpy() + full.mean().to_numpy(), index=c.panel.index, columns=cols
    )
    ours = full[xfcst.columns].where(full[xfcst.columns].notna(), fitted[xfcst.columns])
    assert max_abs(ours.reindex(xfcst.index).to_numpy(), xfcst.to_numpy()) < TOL_TWO_STEP
    consistent = pd.DataFrame(
        signal * np.asarray(spec["std"]) + np.asarray(spec["mean"]),
        index=c.panel.index,
        columns=cols,
    )
    gap = max_abs(
        consistent[xfcst.columns].reindex(xfcst.index), fitted[xfcst.columns].reindex(xfcst.index)
    )
    assert gap > 0.0


# ---------------------------------------------------------------------- 3. divergences
DIVERGENCE_CASES = {
    # prefix: (min |corr| of the factors, max |nowcast diff| / sd(y))
    "usgdp_2s": (0.9999, 0.02),
    "usgdp_2s_agg": (0.99, 0.10),
    "sim_2s": (0.9999, 0.01),
    "sim_2s_agg": (0.9999, 0.01),
    "simbal_2s_agg": (0.99999, 1e-4),
    "simbal_2s_agg_p2": (0.99999, 1e-3),
}


@pytest.mark.reference_divergence
@pytest.mark.parametrize("prefix", sorted(DIVERGENCE_CASES))
def test_default_pipeline_divergence(prefix: str) -> None:
    """Full nowcastbox defaults vs R (standardisation sample, ddof, initial state)."""
    c = case(prefix)
    min_corr, max_scaled = DIVERGENCE_CASES[prefix]
    res = _ours(c)
    r_factors = read_periods(f"r/{prefix}_factors.csv")
    ours_f = res.factors.reindex(r_factors.index).to_numpy()
    assert abs_correlations(ours_f, r_factors.to_numpy()).min() > min_corr
    yfcst = read_periods(f"r/{prefix}_yfcst.csv", "Q")
    reference = r_estimate(yfcst)
    ours = nowcast_estimate(res.nowcast).reindex(reference.index)
    assert max_abs(ours, reference) < max_scaled * float(reference.std())


@pytest.mark.reference_divergence
def test_r_shock_covariance_not_symmetric_when_q_below_r() -> None:
    """R's ``BB`` for q = 1 < r = 2 is not a covariance matrix; nowcastbox's ``B B'`` is."""
    spec = read_json("r/sim_2s_agg_q1.json")
    bb_r = np.asarray(spec["BB"])[:2, :2]
    assert abs(bb_r[0, 1] - bb_r[1, 0]) > 1.0
    c = case("sim_2s_agg")
    frame = c.panel.copy()
    frame["__y"] = quarter_end_only(c.target)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = TwoStepDFM(n_factors=2, factor_lags=1, n_shocks=1, horizon=0).fit(
            frame, "__y", frequency={"__y": "Q"}
        )
    bb = res.B @ res.B.T
    assert np.allclose(bb, bb.T)
    assert np.linalg.matrix_rank(bb) == 1
    eig = np.linalg.eigvalsh(res.params["residual_cov"])
    assert np.trace(bb) == pytest.approx(eig.max())  # leading principal component of Sigma_u

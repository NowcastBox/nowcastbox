"""Shared helpers of the reference-validation tests (plan section 10.4).

The fixtures in ``fixtures/`` are produced by the scripts in
``scripts/reference_fixtures/``: the GPL-3 R package ``nowcasting`` 1.1.2 is **called as
a black box** (its outputs are stored; its source code is never read) and statsmodels
is used as a numerical reference. Every test skips cleanly when a fixture is missing,
so neither R nor the fixture scripts are needed at test time.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nowcastbox.statespace import StateSpace

FIXTURES = Path(__file__).resolve().parent / "fixtures"
R_DIR = FIXTURES / "r"
INPUT_DIR = FIXTURES / "inputs"
SM_DIR = FIXTURES / "statsmodels"

# Tolerances of plan section 10.4 (maximum absolute error).
TOL_TRANSFORM = 1e-10
TOL_IC = 1e-8
TOL_TWO_STEP = 1e-6
TOL_EM = 1e-4
TOL_NEWS = 1e-5


def fixture_path(relative: str) -> Path:
    """Path of a fixture; skip the calling test when it does not exist."""
    path = FIXTURES / relative
    if not path.exists():
        pytest.skip(f"reference fixture {relative} not found (run scripts/reference_fixtures)")
    return path


def read_periods(relative: str, freq: str = "M") -> pd.DataFrame:
    """Read a ``period``-indexed CSV fixture (``freq`` = "M" or "Q")."""
    frame = pd.read_csv(fixture_path(relative))
    frame.index = pd.PeriodIndex(frame.pop("period"), freq=freq)
    frame.index.name = "period"
    return frame.astype(float)


def read_json(relative: str) -> dict[str, Any]:
    """Read a JSON fixture."""
    with fixture_path(relative).open(encoding="utf-8") as fh:
        return json.load(fh)


@cache
def _legend(name: str) -> pd.DataFrame:
    return pd.read_csv(fixture_path(f"inputs/{name}_legend.csv"))


def legend(name: str) -> pd.DataFrame:
    """Legend (name, transformation, ...) of a bundled R dataset."""
    return _legend(name).copy()


def codes(name: str) -> dict[str, int]:
    """Transformation codes of a bundled R dataset by series name."""
    leg = legend(name)
    return dict(zip(leg["name"], leg["transformation"].astype(int), strict=True))


def pad(frame: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Append ``n`` empty months (the forecast rows R's ``Bpanel`` adds)."""
    idx = pd.period_range(frame.index[0], periods=len(frame) + n, freq="M")
    return frame.reindex(idx)


def quarter_end_only(series: pd.Series) -> pd.Series:
    """Keep the values of the 3rd month of each quarter (nowcastbox storage convention)."""
    return series.where(series.index.month % 3 == 0)  # type: ignore[attr-defined]


def max_abs(a: Any, b: Any) -> float:
    """Maximum absolute difference over the cells observed in both inputs."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    both = np.isfinite(x) & np.isfinite(y)
    if not both.any():
        return 0.0
    return float(np.max(np.abs(x[both] - y[both])))


def max_rel(a: Any, b: Any) -> float:
    """``max |a - b| / max(1, |b|)`` over the cells observed in both inputs."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    both = np.isfinite(x) & np.isfinite(y)
    if not both.any():
        return 0.0
    return float(np.max(np.abs(x[both] - y[both]) / np.maximum(1.0, np.abs(y[both]))))


def same_missing(a: pd.DataFrame | pd.Series, b: pd.DataFrame | pd.Series) -> bool:
    """Whether two aligned frames have the same missing-value pattern."""
    return bool((a.isna().to_numpy() == b.isna().to_numpy()).all())


def column_signs(ours: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Signs aligning the columns of ``ours`` with ``reference`` (factor sign indeterminacy)."""
    s = np.sign(np.nansum(np.asarray(ours) * np.asarray(reference), axis=0))
    s[s == 0] = 1.0
    return s


def abs_correlations(ours: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Absolute correlation of matching columns over rows finite in both."""
    out = []
    for k in range(reference.shape[1]):
        a, b = ours[:, k], reference[:, k]
        ok = np.isfinite(a) & np.isfinite(b)
        out.append(abs(float(np.corrcoef(a[ok], b[ok])[0, 1])))
    return np.array(out)


def symmetric(m: np.ndarray) -> np.ndarray:
    """Symmetric part of a square matrix."""
    return 0.5 * (m + m.T)


def r_two_step_state_space(spec: dict[str, Any]) -> StateSpace:
    """State-space model of R's ``nowcast(method = "2s"/"2s_agg")`` from its outputs.

    State ``(f_t', ..., f_{t-p+1}')'``: transition ``A`` (companion), ``Z = [Lambda, 0]``,
    state disturbance covariance ``BB`` (upper-left ``r x r`` block), ``H = diag(Psi)``
    and the initial moments ``initx`` / ``initV`` as the distribution of the first
    state.
    """
    r = int(spec["r"])
    a = np.atleast_2d(np.asarray(spec["A"], dtype=float))
    k = a.shape[0]
    lam = np.atleast_2d(np.asarray(spec["Lambda"], dtype=float))
    design = np.zeros((lam.shape[0], k))
    design[:, :r] = lam
    bb = symmetric(np.atleast_2d(np.asarray(spec["BB"], dtype=float))[:r, :r])
    w, v = np.linalg.eigh(bb)
    selection = np.zeros((k, r))
    selection[:r] = v * np.sqrt(np.clip(w, 0.0, None))
    return StateSpace(
        a,
        design,
        np.eye(r),
        np.asarray(spec["Psi"], dtype=float),
        selection=selection,
        initial_state=np.asarray(spec["initx"], dtype=float).ravel(),
        initial_state_cov=symmetric(np.atleast_2d(np.asarray(spec["initV"], dtype=float))),
    )


def r_em_state_space(spec: dict[str, Any]) -> StateSpace:
    """State-space model of R's ``nowcast(method = "EM")`` from ``Res`` (A, C, Q, R, Z_0, V_0)."""
    return StateSpace(
        np.asarray(spec["A"], dtype=float),
        np.asarray(spec["C"], dtype=float),
        symmetric(np.asarray(spec["Q"], dtype=float)),
        np.asarray(spec["R"], dtype=float),
        initial_state=np.asarray(spec["Z_0"], dtype=float),
        initial_state_cov=symmetric(np.asarray(spec["V_0"], dtype=float)),
    )


def r_standardize(panel: pd.DataFrame, spec: dict[str, Any], mean: str, std: str) -> np.ndarray:
    """Standardise ``panel[spec['columns']]`` with R's stored means / standard deviations."""
    x = panel[list(spec["columns"])].to_numpy(dtype=float)
    return (x - np.asarray(spec[mean], dtype=float)) / np.asarray(spec[std], dtype=float)


def nowcast_estimate(nowcast: pd.DataFrame) -> pd.Series:
    """Single column with ``in_sample`` where available, else ``out_of_sample``."""
    return nowcast["in_sample"].fillna(nowcast["out_of_sample"])


def r_estimate(yfcst: pd.DataFrame) -> pd.Series:
    """Same combination for R's ``yfcst`` (columns ``y``, ``in``, ``out``)."""
    return yfcst["in"].fillna(yfcst["out"])

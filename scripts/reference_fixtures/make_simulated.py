"""Generate the shared simulated panel used by the R and Python reference comparisons.

The panel is written to ``tests/reference_validation/fixtures/inputs/simulated.csv``
(monthly ``period`` column; the quarterly target ``gdp`` is stored in the third month of
each quarter, NaN elsewhere) and is read by the R scripts in this folder as well as by
the tests. It is a stationary two-factor model (VAR(1) factors, iid idiosyncratic noise)
with 24 monthly series and a quarterly target that aggregates a latent monthly variable
with the Mariano-Murasawa weights (1, 2, 3, 2, 1)/3, plus a ragged edge.

Run from the repository root::

    python3 scripts/reference_fixtures/make_simulated.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("tests/reference_validation/fixtures/inputs/simulated.csv")
SEED = 20261001
N_MONTHLY = 24
N_PERIODS = 240  # 2000-01 .. 2019-12


def simulate(seed: int = SEED) -> pd.DataFrame:
    """Simulate the panel (deterministic for a given seed)."""
    rng = np.random.default_rng(seed)
    a = np.array([[0.7, 0.1], [0.0, 0.5]])
    burn = 50
    f = np.zeros((N_PERIODS + burn, 2))
    for t in range(1, N_PERIODS + burn):
        f[t] = a @ f[t - 1] + rng.standard_normal(2)
    f = f[burn:]
    lam = rng.normal(0.0, 1.0, size=(N_MONTHLY, 2))
    sig = rng.uniform(0.5, 1.5, size=N_MONTHLY)
    x = f @ lam.T + rng.standard_normal((N_PERIODS, N_MONTHLY)) * sig
    latent = f @ np.array([0.8, 0.4]) + 0.4 * rng.standard_normal(N_PERIODS)
    w = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 3.0
    gdp = np.full(N_PERIODS, np.nan)
    for t in range(4, N_PERIODS):
        if t % 3 == 2:
            gdp[t] = float(w @ latent[t - np.arange(5)]) + 1.0
    index = pd.period_range("2000-01", periods=N_PERIODS, freq="M")
    cols = [f"x{i + 1:02d}" for i in range(N_MONTHLY)]
    frame = pd.DataFrame(np.round(x, 10), index=index, columns=cols)
    # ragged edge: publication delays of 0-2 months, target missing in the last quarter
    for j in range(len(cols)):
        lag = j % 3
        if lag:
            frame.iloc[-lag:, j] = np.nan
    gdp[-3:] = np.nan
    frame["gdp"] = np.round(gdp, 10)
    frame.index.name = "period"
    return frame


def main() -> None:
    """Write the CSV."""
    frame = simulate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, float_format="%.10f", index_label="period")
    print(f"wrote {OUT} {frame.shape}")


if __name__ == "__main__":
    main()

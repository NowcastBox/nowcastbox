"""Diagnostics on a weekly base grid (calendar aggregation weights, innovation I1)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nowcastbox.diagnostics import run_diagnostics
from nowcastbox.diagnostics._common import aggregate_factors, series_weights
from nowcastbox.models import MixedFreqDFM
from tests.models.test_frequencies_em import simulate_weekly_dfm


@pytest.fixture(scope="module")
def weekly():
    sim = simulate_weekly_dfm(n_weeks=208, seed=5)
    panel = sim.panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MixedFreqDFM(idiosyncratic="iid", max_iter=20).fit(panel, "gdp")
    return panel, res


def test_calendar_series_weights(weekly):
    panel, _ = weekly
    w = series_weights(panel, ["w0", "m0", "gdp"])
    assert w["w0"].tolist() == [1.0]
    assert w["m0"].shape[0] == panel.n_periods and w["m0"].ndim == 2
    assert w["gdp"].shape[0] == panel.n_periods
    with pytest.raises(ValueError, match="finite"):
        series_weights(panel, ["m0"], {"m0": np.full((panel.n_periods, 2), np.nan)})
    frame = pd.DataFrame(w["m0"], index=panel.index)
    assert np.array_equal(series_weights(panel, ["m0"], {"m0": frame})["m0"], w["m0"])


def test_aggregate_factors_per_period():
    factors = np.arange(10.0)[:, None]
    weights = np.zeros((10, 3))
    weights[:, 0] = 1.0
    weights[5:, 1] = 0.5
    out = aggregate_factors(factors, weights)
    assert out[0, 0] == 0.0  # zero weights on pre-sample lags are skipped
    assert out[6, 0] == pytest.approx(6.0 + 0.5 * 5.0)


def test_run_diagnostics_weekly(weekly):
    _, res = weekly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = run_diagnostics(res, components=["stability", "contribution", "residuals"])
    table = report.stability.table
    assert {"m0", "gdp"} <= set(table.index)
    assert table.loc["m0", "n_obs"] > 30
    assert report.contribution is not None and report.residuals is not None

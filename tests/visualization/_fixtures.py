"""Shared fitted results for the visualization, reports and experiment tests."""

from __future__ import annotations

import functools

import matplotlib

matplotlib.use("Agg")

from nowcastbox.models import MixedFreqDFM, MixedFreqDFMResults, TwoStepDFM, TwoStepResults
from nowcastbox.models.two_step import simulate_two_step_example


@functools.cache
def panel():
    """Small simulated panel: 6 monthly predictors + quarterly gdp with ragged edge."""
    return simulate_two_step_example(n_periods=90, n_series=6, n_factors=2, random_state=0)


@functools.cache
def two_step_results() -> TwoStepResults:
    return TwoStepDFM(n_factors=2).fit(panel(), "gdp")


@functools.cache
def em_results() -> MixedFreqDFMResults:
    return MixedFreqDFM(n_factors=1, max_iter=15).fit(panel(), "gdp")

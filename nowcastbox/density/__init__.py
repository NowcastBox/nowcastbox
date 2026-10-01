"""Density nowcasts (innovation I5).

Predictive distribution of the target for fitted :class:`~nowcastbox.models.TwoStepDFM`
and :class:`~nowcastbox.models.MixedFreqDFM` models: filtering/smoothing uncertainty
from the Kalman smoother (Gaussian, analytic) and parameter uncertainty from a
parametric or block bootstrap with re-estimation, combined in a
:class:`NowcastDistribution` (Gaussian mixture) with moments, quantiles, intervals,
sampling, pdf/cdf, ``to_frame()`` and a fan-chart plot. Scores (CRPS, log score, PIT,
coverage tests) live in :mod:`nowcastbox.evaluation.scoring`.

Examples
--------
>>> from nowcastbox.models import TwoStepDFM
>>> from nowcastbox.models.two_step import simulate_two_step_example
>>> from nowcastbox.density import nowcast_distribution
>>> res = TwoStepDFM(n_factors=1).fit(simulate_two_step_example(random_state=0), "gdp")
>>> dist = nowcast_distribution(res)  # Gaussian: smoothing uncertainty only
>>> dist.interval(0.9).shape
(2, 2)
"""

from nowcastbox.density.bootstrap import (
    BOOTSTRAP_METHODS,
    BootstrapNowcasts,
    block_bootstrap_panel,
    bootstrap_nowcasts,
    simulate_from_results,
)
from nowcastbox.density.distribution import DEFAULT_LEVELS, NowcastDistribution
from nowcastbox.density.predictive import (
    analytic_distribution,
    combine_bootstrap,
    nowcast_distribution,
)

__all__ = [
    "BOOTSTRAP_METHODS",
    "DEFAULT_LEVELS",
    "BootstrapNowcasts",
    "NowcastDistribution",
    "analytic_distribution",
    "block_bootstrap_panel",
    "bootstrap_nowcasts",
    "combine_bootstrap",
    "nowcast_distribution",
    "simulate_from_results",
]

"""Density nowcasts (innovation I5).

Predictive distribution of the target for fitted :class:`~nowcastbox.models.TwoStepDFM`
and :class:`~nowcastbox.models.MixedFreqDFM` models: filtering/smoothing uncertainty
from the Kalman smoother (Gaussian, analytic) and parameter uncertainty from a
parametric or block bootstrap with re-estimation, combined in a
:class:`NowcastDistribution` (Gaussian mixture) with moments, quantiles, intervals,
sampling, pdf/cdf, ``to_frame()`` and a fan-chart plot. Empirical error bands
(:mod:`nowcastbox.density.empirical`, Reifschneider-Tulip / ECB style) build the
distribution from past backtest errors at the same horizon instead. Scores (CRPS, log score, PIT,
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
from nowcastbox.density.empirical import (
    EMPIRICAL_LEVELS,
    EMPIRICAL_METHODS,
    MAE_TO_SIGMA,
    EmpiricalGaussianDistribution,
    EmpiricalQuantileDistribution,
    available_errors,
    backtest_empirical_bands,
    empirical_bands,
    empirical_error_scales,
)
from nowcastbox.density.predictive import (
    analytic_distribution,
    combine_bootstrap,
    nowcast_distribution,
)

__all__ = [
    "BOOTSTRAP_METHODS",
    "DEFAULT_LEVELS",
    "EMPIRICAL_LEVELS",
    "EMPIRICAL_METHODS",
    "MAE_TO_SIGMA",
    "BootstrapNowcasts",
    "EmpiricalGaussianDistribution",
    "EmpiricalQuantileDistribution",
    "NowcastDistribution",
    "analytic_distribution",
    "available_errors",
    "backtest_empirical_bands",
    "block_bootstrap_panel",
    "bootstrap_nowcasts",
    "combine_bootstrap",
    "empirical_bands",
    "empirical_error_scales",
    "nowcast_distribution",
    "simulate_from_results",
]

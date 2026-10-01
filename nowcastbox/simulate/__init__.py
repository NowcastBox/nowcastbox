"""Simulation of dynamic factor models with known parameters.

:func:`dfm` (monthly indicators + quarterly target; the generator of
:func:`~nowcastbox.datasets.load_simulated_dfm`) and :func:`weekly_dfm` (weekly,
monthly and quarterly series on a weekly grid with calendar aggregation, innovation I1)
return a :class:`SimulatedDFM` with the panel and the true parameters. Used by tests,
tutorials and Monte Carlo studies.

Examples
--------
>>> import nowcastbox as nb
>>> data, truth = nb.simulate.dfm(n_series=6, n_factors=1, n_periods=60)
>>> data.columns[-1], truth["loadings"].shape
('gdp', (7, 1))
"""

from nowcastbox.simulate.dfm import SimulatedDFM, dfm, weekly_dfm

__all__ = ["SimulatedDFM", "dfm", "weekly_dfm"]

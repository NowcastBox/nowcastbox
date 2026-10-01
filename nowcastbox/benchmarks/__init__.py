"""Benchmark forecasters.

AR(p), random walk, historical mean, bridge equations, U-MIDAS and MIDAS
(exponential Almon / Beta lag polynomials) and an adapter for any scikit-learn
compatible regressor (plan innovation I11). All implement
:class:`nowcastbox.core.base.BaseBenchmark`: ``fit(data, target)`` on one vintage and
``predict(periods)`` for periods of the target's native frequency.

Examples
--------
>>> import numpy as np, pandas as pd
>>> from nowcastbox.benchmarks import AR, RandomWalk
>>> idx = pd.period_range("2010Q1", periods=40, freq="Q")
>>> frame = pd.DataFrame({"gdp": np.sin(np.arange(40.0))}, index=idx)
>>> RandomWalk().fit(frame, "gdp", frequency="Q").predict(["2020Q1"]).round(4).tolist()
[0.9638]
>>> AR(p="bic", max_p=3).fit(frame, "gdp", frequency="Q").order_ >= 1
True
"""

from nowcastbox.benchmarks.ar import AR
from nowcastbox.benchmarks.bridge import BridgeBenchmark
from nowcastbox.benchmarks.mean import HistoricalMean
from nowcastbox.benchmarks.midas import MIDAS, UMIDAS, beta_weights, exp_almon_weights
from nowcastbox.benchmarks.random_walk import RandomWalk
from nowcastbox.benchmarks.sklearn_adapter import SklearnBenchmark, clone_estimator

__all__ = [
    "AR",
    "MIDAS",
    "UMIDAS",
    "BridgeBenchmark",
    "HistoricalMean",
    "RandomWalk",
    "SklearnBenchmark",
    "beta_weights",
    "clone_estimator",
    "exp_almon_weights",
]

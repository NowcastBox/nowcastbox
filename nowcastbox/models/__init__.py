"""Nowcasting estimators.

``TwoStepDFM`` (Giannone, Reichlin & Small, 2008; Doz, Giannone & Reichlin, 2011),
``MixedFreqDFM`` (EM of Banbura & Modugno, 2014, with blocks and AR(1)
idiosyncratic components), ``BridgeEquation`` and ``BridgeCombination`` (all small
bridge equations combined; Bańbura, Belousova, Bodnár & Tóth, 2023) with pluggable
indicator extrapolators (:mod:`nowcastbox.models.extrapolation`).
"""

__all__: list[str] = []

from nowcastbox.models.bridge import (
    BridgeEquation,
    BridgeRegression,
    BridgeResults,
    fit_bridge_regression,
)
from nowcastbox.models.two_step import TwoStepDFM, TwoStepResults

__all__ += [
    "BridgeEquation",
    "BridgeRegression",
    "BridgeResults",
    "TwoStepDFM",
    "TwoStepResults",
    "fit_bridge_regression",
]

from nowcastbox.models.em import EMParameters, MixedFreqDFM, MixedFreqDFMResults, StateLayout

__all__ += ["EMParameters", "MixedFreqDFM", "MixedFreqDFMResults", "StateLayout"]

from nowcastbox.models.bridge_combination import (
    BridgeCombination,
    BridgeCombinationResults,
    bridge_equation_count,
)
from nowcastbox.models.extrapolation import (
    ARExtrapolator,
    Extrapolator,
    available_extrapolators,
    make_extrapolator,
    register_extrapolator,
)

__all__ += [
    "ARExtrapolator",
    "BridgeCombination",
    "BridgeCombinationResults",
    "Extrapolator",
    "available_extrapolators",
    "bridge_equation_count",
    "make_extrapolator",
    "register_extrapolator",
]

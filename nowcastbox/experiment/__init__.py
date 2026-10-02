"""Model comparison experiments and alternative-model nowcasts.

``NowcastExperiment`` follows the ``panelbox.experiment`` pattern::

    exp = NowcastExperiment(data, "gdp")
    exp.add_model("2s", TwoStepDFM(n_factors=2)).add_model("em", MixedFreqDFM())
    exp.fit_all()
    exp.compare()  # nowcasts, log-likelihoods, in-sample fit, timings

``alternative_models`` measures the dependence of the nowcast on groups of indicators
(Linzenich & Meunier, 2024, §3.5)::

    alt = alternative_models(TwoStepDFM(n_factors=2), data, "gdp", by="category")
    alt.table()
    alt.range()
    alt.plot()
"""

from nowcastbox.experiment.alternatives import (
    GROUPINGS,
    AlternativeNowcasts,
    alternative_models,
    combination_label,
    resolve_groups,
)
from nowcastbox.experiment.experiment import (
    BacktestHook,
    NowcastExperiment,
    in_sample_metrics,
    pseudo_real_time_hook,
)

__all__ = [
    "GROUPINGS",
    "AlternativeNowcasts",
    "BacktestHook",
    "NowcastExperiment",
    "alternative_models",
    "combination_label",
    "in_sample_metrics",
    "pseudo_real_time_hook",
    "resolve_groups",
]

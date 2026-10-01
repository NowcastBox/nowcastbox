"""``NowcastExperiment``: compare several model specifications on the same data.

Follows the ``panelbox.experiment`` pattern::

    exp = NowcastExperiment(data, "gdp")
    exp.add_model("2s", TwoStepDFM(n_factors=2)).add_model("em", MixedFreqDFM())
    exp.fit_all()
    exp.compare()  # nowcasts, log-likelihoods, in-sample fit, timings
"""

from nowcastbox.experiment.experiment import (
    BacktestHook,
    NowcastExperiment,
    in_sample_metrics,
    pseudo_real_time_hook,
)

__all__ = ["BacktestHook", "NowcastExperiment", "in_sample_metrics", "pseudo_real_time_hook"]

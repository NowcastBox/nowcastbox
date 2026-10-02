"""Pipeline wiring of the ECB-parity large BVAR (0.2.0): ``model: large_bvar`` and the
``"bvar"`` extrapolation of ``model: bridge_combination``."""

from __future__ import annotations

import pytest

from nowcastbox.density import NowcastDistribution
from nowcastbox.models import LargeBVAR
from nowcastbox.pipeline import NowcastSpec, SpecError, run_pipeline
from nowcastbox.pipeline.spec import MODEL_TYPES
from tests.pipeline.test_runner import base_spec

pytestmark = [
    pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.ConvergenceWarning"),
    pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.DataQualityWarning"),
]

BACKTEST = {"start": "2019-01-15", "end": "2019-06-15", "target_offsets": [0]}


def parse(**model):
    return NowcastSpec.from_dict(base_spec(model={"type": "large_bvar", **model}))


def problems(outputs=None, **model):
    spec = base_spec(model={"type": "large_bvar", **model})
    if outputs is not None:
        spec["outputs"] = outputs
    with pytest.raises(SpecError) as err:
        NowcastSpec.from_dict(spec)
    return [path for path, _ in err.value.issues]


class TestLargeBVARSpec:
    @pytest.mark.parametrize("name", ["large_bvar", "LargeBVAR", "bvar", "large-bvar"])
    def test_aliases(self, name):
        spec = NowcastSpec.from_dict(base_spec(model={"type": name, "lags": 2}))
        assert spec.model.type == "LargeBVAR"
        assert spec.model.method == "large_bvar"
        assert spec.model.estimator_class is LargeBVAR
        assert spec.model.options == {"lags": 2}
        assert "LargeBVAR" in MODEL_TYPES

    def test_round_trip(self):
        spec = parse(horizon=0, n_draws=10, prior_mean="random_walk")
        assert spec.model.to_dict() == {
            "type": "LargeBVAR",
            "horizon": 0,
            "n_draws": 10,
            "prior_mean": "random_walk",
        }
        assert NowcastSpec.from_dict(spec.to_dict()).model == spec.model

    def test_factor_settings_and_unknown_options_rejected(self):
        paths = problems(factors=2, criterion="IC1", combine="mean")
        assert {"model.factors", "model.criterion", "model.combine"} <= set(paths)

    def test_bootstrap_density_rejected(self):
        assert "outputs.density.n_boot" in problems(outputs={"density": {"n_boot": 20}})

    def test_news_and_density_accepted(self):
        spec = NowcastSpec.from_dict(
            {**base_spec(model={"type": "bvar"}), "outputs": ["news", "density"]}
        )
        assert spec.outputs.news is not None and spec.outputs.density is not None


class TestLargeBVARRun:
    def test_fit_outputs_and_backtest(self, tmp_path):
        spec = base_spec(
            model={"type": "large_bvar", "n_draws": 30, "horizon": 0},
            snapshot_dir=tmp_path,
        )
        spec["outputs"] = {
            "nowcast": True,
            "news": {"against": "2019-08-15"},
            "density": True,
            "report_html": True,
            "backtest": BACKTEST,
        }
        run = run_pipeline(spec)
        assert not [w for w in run.warnings if "skipped" in w]
        assert run.results.model_name == "LargeBVAR"
        assert run.results.model_params["n_draws"] == 30
        summary = run.summary()
        assert "lags=1" in summary and "lambda=" in summary
        dist = run.results.info["distribution"]
        assert isinstance(dist, NowcastDistribution) and dist.n_components == 30
        assert run.news is not None
        assert run.news.check_identity()
        assert run.report_path is not None
        assert run.backtest is not None and "LargeBVAR" in run.backtest.models

    def test_bridge_combination_with_bvar_extrapolation(self):
        spec = base_spec(
            model={
                "type": "bridge_combination",
                "max_monthly": 1,
                "extrapolation": "bvar",
                "extrapolation_options": {"lags": 1},
            }
        )
        spec["outputs"] = ["nowcast"]
        run = run_pipeline(spec)
        assert run.results.model_params["extrapolation"] == "bvar"
        assert run.results.extrapolated

"""Tests of the bootstrap data-generating processes (parametric simulation, block bootstrap)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.density import BOOTSTRAP_METHODS, block_bootstrap_panel, simulate_from_results
from nowcastbox.density.bootstrap import _parametric_supported


class TestSimulateFromResults:
    @pytest.mark.parametrize("fixture", ["two_step_results", "em_results"])
    def test_mask_and_metadata_preserved(self, fixture: str, request: pytest.FixtureRequest):
        res = request.getfixturevalue(fixture)
        sim = simulate_from_results(res, random_state=0)
        assert sim.shape == res.data.shape
        pd.testing.assert_frame_equal(sim.observation_mask(), res.data.observation_mask())
        assert sim.frequencies.equals(res.data.frequencies)
        assert not np.allclose(
            np.nan_to_num(sim.values), np.nan_to_num(res.data.values)
        )  # new values

    def test_reproducible(self, em_results) -> None:
        a = simulate_from_results(em_results, random_state=3)
        b = simulate_from_results(em_results, random_state=3)
        assert a.equals(b)

    def test_two_step_moments_match_model(self, two_step_results) -> None:
        """Simulated predictors have (approximately) the scale of the data."""
        res = two_step_results
        sims = [simulate_from_results(res, random_state=s).to_frame() for s in range(30)]
        stacked = pd.concat(sims)
        x_std = stacked[list(res.loadings.index)].std()
        ratio = x_std / res.standardization.std
        assert bool(((ratio > 0.7) & (ratio < 1.3)).all())
        gdp = stacked["gdp"].dropna()
        assert abs(gdp.mean() - res.data.to_frame()["gdp"].mean()) < 0.3

    def test_excluded_low_frequency_predictor_kept(self, panel: MixedFrequencyData) -> None:
        import warnings

        from nowcastbox.models import TwoStepDFM

        frame = panel.to_frame()
        idx = frame.index
        late = (idx.month % 3 == 0) & (idx.year >= 2014) & (idx.month >= 9)
        frame["q2"] = np.where(late, np.arange(len(idx), dtype=float), np.nan)
        data = MixedFrequencyData(frame, panel.frequencies.to_dict() | {"q2": "Q"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = TwoStepDFM(n_factors=1).fit(data, "gdp")
        assert res.info["excluded_predictors"] == ["q2"]
        sim = simulate_from_results(res, random_state=0)
        pd.testing.assert_series_equal(sim.to_frame()["q2"], data.to_frame()["q2"])

    def test_variables_simulates_filtered_panel(self, two_step_variables_results) -> None:
        res = two_step_variables_results
        sim = simulate_from_results(res, random_state=0)
        assert sim.index.equals(res.model_data.index)
        assert sim.observation_mask().equals(res.model_data.observation_mask())

    def test_unsupported(self, two_step_variables_results) -> None:
        with pytest.raises(NotImplementedError, match="block"):
            simulate_from_results(two_step_variables_results.replace(model_data=None))
        idx = pd.period_range("2020Q1", periods=2, freq="Q")
        frame = build_nowcast_frame(
            pd.Series([1.0, np.nan], index=idx), pd.Series([1.0, 2.0], index=idx)
        )
        generic = NowcastResults(target="y", nowcast=frame)
        assert not _parametric_supported(generic)
        with pytest.raises(NotImplementedError):
            simulate_from_results(generic)

    def test_methods_constant(self) -> None:
        assert BOOTSTRAP_METHODS == ("auto", "parametric", "block")


class TestBlockBootstrap:
    def test_units_are_whole_quarters(self, panel: MixedFrequencyData) -> None:
        boot = block_bootstrap_panel(panel, "gdp", block_length=2, random_state=1)
        original = panel.to_frame()
        resampled = boot.to_frame()
        assert boot.shape == panel.shape
        pd.testing.assert_frame_equal(boot.slot_mask(), panel.slot_mask())
        # every resampled quarter equals some original quarter (rows j..j+2)
        orig_blocks = {
            tuple(np.nan_to_num(original.iloc[i : i + 3].to_numpy()).ravel())
            for i in range(0, len(original) - 2, 3)
        }
        for i in range(0, len(resampled) - 2, 3):
            assert tuple(np.nan_to_num(resampled.iloc[i : i + 3].to_numpy()).ravel()) in (
                orig_blocks
            )

    def test_trailing_partial_period_kept(self, panel: MixedFrequencyData) -> None:
        short = panel.truncate(end=panel.index[-2])  # ends in the 2nd month of a quarter
        boot = block_bootstrap_panel(short, "gdp", random_state=0)
        pd.testing.assert_frame_equal(boot.to_frame().iloc[-2:], short.to_frame().iloc[-2:])

    def test_default_block_length_and_reproducible(self, panel: MixedFrequencyData) -> None:
        a = block_bootstrap_panel(panel, "gdp", random_state=4)
        b = block_bootstrap_panel(panel, "gdp", random_state=4)
        assert a.equals(b)
        long = block_bootstrap_panel(panel, "gdp", block_length=10_000, random_state=0)
        assert long.equals(panel)  # a single block covering the sample

    @pytest.mark.parametrize("bad", [0, -2, 1.5, True])
    def test_invalid_block_length(self, panel: MixedFrequencyData, bad: object) -> None:
        with pytest.raises(ValueError, match="block_length"):
            block_bootstrap_panel(panel, "gdp", block_length=bad)  # type: ignore[arg-type]

    def test_too_short(self, panel: MixedFrequencyData) -> None:
        tiny = panel.truncate(end=panel.index[4])
        with pytest.raises(NowcastDataError, match="two complete"):
            block_bootstrap_panel(tiny, "gdp")

    def test_lower_frequency_series_rejected(self, panel: MixedFrequencyData) -> None:
        frame = panel.to_frame()
        frame["annual"] = np.where(frame.index.month == 12, 1.0, np.nan)
        data = MixedFrequencyData(frame, panel.frequencies.to_dict() | {"annual": "A"})
        with pytest.raises(NowcastDataError, match="annual"):
            block_bootstrap_panel(data, "gdp")

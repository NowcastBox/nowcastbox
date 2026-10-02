"""Tests of the top-level namespace and the high-level :func:`nowcastbox.nowcast`."""

from __future__ import annotations

import dataclasses
import doctest
import warnings

import numpy as np
import pandas as pd
import pytest

import nowcastbox as nb
import nowcastbox.api as api_module
from nowcastbox.models.two_step import simulate_two_step_example


@pytest.fixture(scope="module")
def small() -> nb.MixedFrequencyData:
    return simulate_two_step_example(random_state=0)


@pytest.fixture(scope="module")
def wide() -> nb.MixedFrequencyData:
    data = simulate_two_step_example(n_series=30, n_factors=2, random_state=3)
    return data


# ---------------------------------------------------------------------- namespace
def test_all_names_resolve() -> None:
    for name in nb.__all__:
        assert hasattr(nb, name), name


@pytest.mark.parametrize(
    "name",
    [
        "datasets",
        "data_sources",
        "preprocessing",
        "statespace",
        "selection",
        "vintages",
        "models",
        "news",
        "density",
        "evaluation",
        "benchmarks",
        "diagnostics",
        "visualization",
        "reports",
        "experiment",
        "pipeline",
        "cli",
        "simulate",
    ],
)
def test_subpackages_are_attributes(name: str) -> None:
    assert getattr(nb, name).__name__ == f"nowcastbox.{name}"


def test_plan_entry_points_exist() -> None:
    for name in [
        "prepare_panel",
        "select_factors",
        "select_shocks",
        "pseudo_real_time",
        "ReleaseCalendar",
        "VintageStore",
        "TwoStepDFM",
        "MixedFreqDFM",
        "MixedFrequencyData",
        "nowcast",
        "run_pipeline",
        "NowcastSpec",
        "SnapshotStore",
        "select_blocks",
        "select_variables",
        "calendar_aggregation",
    ]:
        assert name in nb.__all__
    assert nb.utils.month_to_quarter is nb.preprocessing.month_to_quarter


@pytest.mark.parametrize("module", [nb, api_module, nb.utils])
def test_doctests(module: object) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)  # type: ignore[arg-type]
    assert result.failed == 0


# ---------------------------------------------------------------------- nowcast
@pytest.mark.parametrize("alias", ["two_step", "TwoStep", "2s", "two-step", "TwoStepDFM"])
def test_method_aliases_two_step(small, alias: str) -> None:
    res = nb.nowcast(small, "gdp", method=alias, n_factors=1)
    assert isinstance(res, nb.TwoStepResults)
    assert res.info["method"] == "two_step"
    assert "selection" not in res.info


@pytest.mark.parametrize("alias", ["em", "EM", "MixedFreqDFM", "bm"])
def test_method_aliases_em(small, alias: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nb.ConvergenceWarning)
        res = nb.nowcast(small, "gdp", method=alias, n_factors=1, max_iter=5)
    assert isinstance(res, nb.MixedFreqDFMResults)
    assert res.info["method"] == "em"


def test_unknown_method(small) -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        nb.nowcast(small, "gdp", method="bayes")


def test_bai_ng_selection_on_wide_panel(wide) -> None:
    res = nb.nowcast(wide, "gdp", method="two_step")
    assert res.info["selection"].r_star == 2
    assert res.n_factors == 2


def test_small_panel_warns_about_bai_ng(small) -> None:
    with pytest.warns(nb.DataQualityWarning, match="unreliable"):
        nb.nowcast(small, "gdp", method="two_step", rmax=3)


@pytest.mark.filterwarnings("ignore:Bai-Ng criteria are unreliable")
def test_zero_factors_floored_to_one(small, monkeypatch) -> None:
    real = api_module.select_factors

    def zero(*args, **kwargs):
        return dataclasses.replace(real(*args, **kwargs), r_star=0)

    monkeypatch.setattr(api_module, "select_factors", zero)
    with pytest.warns(nb.DataQualityWarning, match="selected 0 factors"):
        res = nb.nowcast(small, "gdp", method="two_step")
    assert res.n_factors == 1


def test_formula_target_restricts_predictors(small) -> None:
    res = nb.nowcast(small, "gdp ~ x1 + x2 + x3", method="two_step", n_factors=1)
    assert list(res.loadings.index) == ["x1", "x2", "x3"]


def test_dataframe_input_with_frequency(small) -> None:
    frame = small.data
    freqs = small.frequencies.to_dict()
    res = nb.nowcast(frame, "gdp", method="two_step", n_factors=1, frequency=freqs)
    expected = nb.nowcast(small, "gdp", method="two_step", n_factors=1)
    pd.testing.assert_frame_equal(res.nowcast, expected.nowcast)


def test_transform_without_preprocess(small) -> None:
    res = nb.nowcast(small, "gdp", method="two_step", n_factors=1, transform={"x1": "diff"})
    assert res.data is not None
    assert res.data.metadata["x1"].transform_applied
    assert not res.data.metadata["x2"].transform_applied


def test_preprocess_mapping_keeps_target(small) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nb.DataQualityWarning)
        res = nb.nowcast(
            small,
            "gdp",
            method="two_step",
            n_factors=1,
            preprocess={"max_na_prop": 0.0, "keep": "x1"},
        )
    assert res.data is not None
    assert {"gdp", "x1"} <= set(res.data.columns)


@pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.DataQualityWarning")
def test_no_predictor_left_after_preprocessing(small) -> None:
    sub = small.select(["x1", "gdp"])
    sub = sub.with_data(sub.data.assign(x1=np.where(np.arange(sub.n_periods) < 170, np.nan, 1.0)))
    with pytest.raises(nb.NowcastDataError, match="No predictors"):
        nb.nowcast(sub, "gdp", preprocess=True, n_factors=1)


def test_selection_needs_two_base_predictors(small) -> None:
    with pytest.raises(nb.NowcastDataError, match="at least two"):
        nb.nowcast(small.select(["x1", "gdp"]), "gdp", method="two_step")


def test_selection_needs_complete_rows(small) -> None:
    sub = small.select(["x1", "x2", "gdp"])
    frame = sub.data
    frame.loc[frame.index[1::2], "x1"] = np.nan
    frame.loc[frame.index[::2], "x2"] = np.nan
    with pytest.raises(nb.NowcastDataError, match="complete rows"):
        nb.nowcast(sub.with_data(frame), "gdp", method="two_step")


def test_auto_shocks(wide) -> None:
    res = nb.nowcast(wide, "gdp", method="two_step", n_factors=2, n_shocks="auto")
    sel = res.info["shock_selection"]
    assert isinstance(sel, nb.ShockSelectionResult)
    assert res.model_params["n_shocks"] == max(1, sel.q_star)


def test_auto_shocks_single_factor(small) -> None:
    res = nb.nowcast(small, "gdp", method="two_step", n_factors=1, n_shocks="auto")
    assert res.info["shock_selection"] is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "em", "n_shocks": "auto", "n_factors": 1}, "n_shocks='auto'"),
        ({"method": "two_step", "n_shocks": "many", "n_factors": 1}, "n_shocks must be"),
        ({"method": "em", "n_shocks": 1, "n_factors": 1}, "only used by"),
        ({"method": "two_step", "blocks": {"x1": "a"}, "n_factors": 1}, "blocks are only"),
    ],
)
def test_invalid_option_combinations(small, kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        nb.nowcast(small, "gdp", **kwargs)


def test_blocks_default_one_factor_per_block(small) -> None:
    blocks = {
        c: ["global", "real"] if c in {"x1", "x2", "x3", "x4"} else ["global"]
        for c in small.columns
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nb.ConvergenceWarning)
        res = nb.nowcast(small, "gdp", method="em", blocks=blocks, max_iter=5)
    assert "selection" not in res.info
    assert res.factors.shape[1] == 2


# ---------------------------------------------------------------------- wave 2
def test_wave2_entry_points_exist() -> None:
    for name in [
        "news_decomposition",
        "nowcast_tracker",
        "level_contributions",
        "PseudoRealTimeBacktest",
        "NowcastDistribution",
        "nowcast_distribution",
        "scoring",
        "run_diagnostics",
        "NowcastExperiment",
        "NowcastReport",
        "load_brazil_nowcast",
        "load_simulated_dfm",
    ]:
        assert name in nb.__all__
    assert nb.scoring is nb.evaluation.scoring
    assert nb.evaluation.crps is nb.evaluation.scoring.crps
    assert nb.benchmarks.AR is not None


def test_density_columns(small) -> None:
    res = nb.nowcast(small, "gdp", method="two_step", n_factors=1, density=True)
    dist = res.info["distribution"]
    assert isinstance(dist, nb.NowcastDistribution)
    frame = res.nowcast
    for column in api_module.DENSITY_COLUMNS:
        assert column in frame.columns
    period = dist.index[-1]
    assert (
        frame.loc[period, "lower_90"] < frame.loc[period, "median"] < frame.loc[period, "upper_90"]
    )
    assert frame.loc[period, "median"] == pytest.approx(frame.loc[period, "out_of_sample"])
    assert res.info["density_n_boot"] == 0


@pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.ConvergenceWarning")
def test_density_bootstrap_widens_or_keeps_intervals(small) -> None:
    gauss = nb.nowcast(small, "gdp", method="two_step", n_factors=1, density=True)
    boot = nb.nowcast(
        small, "gdp", method="two_step", n_factors=1, density=True, n_boot=20, random_state=0
    )
    period = gauss.info["distribution"].index[-1]
    assert boot.nowcast.loc[period, "std"] >= gauss.nowcast.loc[period, "std"] - 1e-12
    assert boot.info["distribution"].n_components > 1


def test_density_adds_missing_columns(small) -> None:
    res = nb.nowcast(small, "gdp", method="two_step", n_factors=1)
    bare = res.replace(
        nowcast=res.nowcast.drop(columns=["lower_90", "upper_90", "median"], errors="ignore")
    )
    out = api_module.add_density(bare)
    assert {"lower_90", "upper_90", "median"} <= set(out.nowcast.columns)


def test_n_boot_requires_density(small) -> None:
    with pytest.raises(ValueError, match="density=True"):
        nb.nowcast(small, "gdp", method="two_step", n_factors=1, n_boot=10)


@pytest.mark.parametrize(
    "option",
    [{"idiosyncratic": "student_t"}, {"long_run_mean": "time_varying"}, {"outliers": "auto"}],
)
def test_em_options_rejected_for_two_step(small, option) -> None:
    with pytest.raises(ValueError, match="only apply to method='em'"):
        nb.nowcast(small, "gdp", method="two_step", n_factors=1, **option)


@pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.ConvergenceWarning")
def test_em_options_forwarded(small) -> None:
    res = nb.nowcast(
        small,
        "gdp",
        n_factors=1,
        idiosyncratic="student_t",
        outliers="auto",
        long_run_mean="time_varying",
        long_run_variance=1e-3,
        max_iter=10,
    )
    assert res.model_params["idiosyncratic"] == "student_t"
    assert res.model_params["outliers"] == "auto"
    assert res.model_params["long_run_mean"] == "time_varying"
    assert res.student_t_df is not None
    assert "long_run_mean" in res.nowcast.columns


@pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.ConvergenceWarning")
def test_blocks_auto(small) -> None:
    no_blocks = nb.nowcast(small, "gdp", n_factors=1, blocks="auto", max_iter=5)
    assert no_blocks.state_layout.block_names == ("global",)
    data = nb.load_simulated_dfm(n_monthly=8, n_factors=1, n_periods=120)
    res = nb.nowcast(data.data, data.target, blocks="auto", max_iter=5)
    assert res.info["method"] == "em"
    assert res.n_factors == len(data.block_names)


def _with_target_outlier(small: nb.MixedFrequencyData) -> tuple[nb.MixedFrequencyData, object]:
    frame = small.data
    gdp = frame["gdp"].dropna()
    period = gdp.index[len(gdp) // 2]
    frame.loc[period, "gdp"] = float(gdp.mean() + 50 * gdp.std())
    return small.with_data(frame), period


def test_preprocess_leaves_target_uncleaned(small) -> None:
    data, period = _with_target_outlier(small)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nb.DataQualityWarning)
        res = nb.nowcast(data, "gdp", method="two_step", n_factors=1, preprocess=True)
        cleaned = nb.nowcast(
            data,
            "gdp",
            method="two_step",
            n_factors=1,
            preprocess={"clean_target": True},
        )
    assert res.data is not None and cleaned.data is not None
    raw_value = data.data.loc[period, "gdp"]
    assert res.data.data.loc[period, "gdp"] == pytest.approx(raw_value)
    assert cleaned.data.data.loc[period, "gdp"] != pytest.approx(raw_value)


def test_weekly_panel_em_and_two_step_rejected() -> None:
    from tests.models.test_frequencies_em import simulate_weekly_dfm

    panel = simulate_weekly_dfm(n_weeks=156, seed=1).panel()
    with pytest.raises(ValueError, match="method='em'"):
        nb.nowcast(panel, "gdp", method="two_step", n_factors=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = nb.nowcast(panel, "gdp", n_factors=1, max_iter=5)
    assert res.nowcast["out_of_sample"].notna().any()


def test_ecb_parity_entry_points_exist() -> None:
    for name in [
        "empirical_bands",
        "indicator_zscores",
        "alternative_models",
        "AlternativeNowcasts",
        "pesaran_timmermann",
    ]:
        assert name in nb.__all__
    assert nb.empirical_bands is nb.density.empirical_bands
    assert nb.alternative_models is nb.experiment.alternative_models
    assert hasattr(nb.MixedFrequencyData, "released_share")

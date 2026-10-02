"""Alternative-model nowcasts without 1-2 groups of variables (plan item 6)."""

from __future__ import annotations

import warnings

import matplotlib.figure
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.results import NowcastResults, build_nowcast_frame
from nowcastbox.experiment import (
    AlternativeNowcasts,
    alternative_models,
    combination_label,
    resolve_groups,
)
from nowcastbox.models import MixedFreqDFM, TwoStepDFM
from nowcastbox.models.two_step import simulate_two_step_example

CATEGORIES = {"x1": "hard", "x2": "hard", "x3": "soft", "x4": "soft", "x5": "financial"}
BLOCKS = {
    "x1": ("global", "real"),
    "x2": ("global", "real"),
    "x3": ("global", "soft"),
    "x4": ("global", "soft"),
    "x5": ("global",),
    "x6": ("global",),
    "gdp": ("global", "real"),
}
MAPPING = {"real": ["x1", "x2"], "soft": ["x3", "x4"], "fin": ["x5", "x6"]}


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def data() -> MixedFrequencyData:
    panel = simulate_two_step_example(n_periods=120, n_series=6, n_factors=1, random_state=3)
    for column, category in CATEGORIES.items():
        panel = panel.with_metadata(column, category=category)
    for column, blocks in BLOCKS.items():
        panel = panel.with_metadata(column, blocks=blocks)
    return panel


@pytest.fixture(scope="module")
def two_step_alt(data):
    return alternative_models(TwoStepDFM(n_factors=1), data, "gdp", by=MAPPING)


# ---------------------------------------------------------------------------- groups
def test_resolve_groups_by_category(data):
    groups = resolve_groups(data, "gdp", "category")
    assert groups == {"hard": ("x1", "x2"), "soft": ("x3", "x4"), "financial": ("x5",)}


def test_resolve_groups_by_block_allows_several_memberships(data):
    groups = resolve_groups(data, "gdp", "block")
    assert groups["global"] == ("x1", "x2", "x3", "x4", "x5", "x6")
    assert groups["real"] == ("x1", "x2") and groups["soft"] == ("x3", "x4")


def test_resolve_groups_mappings(data):
    assert resolve_groups(data, "gdp", MAPPING)["fin"] == ("x5", "x6")
    inverted = {"x1": "a", "x2": "a", "x3": "b"}
    assert resolve_groups(data, "gdp", inverted) == {"a": ("x1", "x2"), "b": ("x3",)}
    singles = {"one": "x4", "two": "x5"}
    assert resolve_groups(data, "gdp", singles) == {"one": ("x4",), "two": ("x5",)}
    mixed = {"one": "x4", "two": ["x5"]}
    assert resolve_groups(data, "gdp", mixed) == {"one": ("x4",), "two": ("x5",)}
    assert resolve_groups(data, "gdp", {"a": ["x1"], "empty": []}) == {"a": ("x1",)}


def test_resolve_groups_subset_keeps_order(data):
    groups = resolve_groups(data, "gdp", MAPPING, groups=["fin", "real", "fin"])
    assert list(groups) == ["fin", "real"]


@pytest.mark.parametrize(
    ("by", "groups", "match"),
    [
        ("sector", None, "by must be"),
        ({"a": ["x1", "nope"]}, None, "not predictors"),
        ({"a": ["gdp"]}, None, "not predictors"),
        (MAPPING, ["real", "nominal"], "Unknown groups"),
    ],
)
def test_resolve_groups_errors(data, by, groups, match):
    with pytest.raises(ValueError, match=match):
        resolve_groups(data, "gdp", by, groups)


def test_resolve_groups_without_categories():
    panel = simulate_two_step_example(n_periods=60, n_series=3, random_state=0)
    with pytest.raises(ValueError, match="No predictor of the panel has a group"):
        resolve_groups(panel, "gdp", "category")


def test_combination_label():
    assert combination_label(()) == "base"
    assert combination_label(("hard",)) == "-hard"
    assert combination_label(("hard", "soft")) == "-hard -soft"


# ---------------------------------------------------------------------------- combinations
def test_number_of_combinations(data):
    model = TwoStepDFM(n_factors=1)
    one = alternative_models(model, data, "gdp", by=MAPPING, drop=1, refit=False)
    assert one.n_models == 3
    assert one.combinations == (("real",), ("soft",), ("fin",))
    both = alternative_models(model, data, "gdp", by=MAPPING, drop=(2, 1, 2), refit=False)
    assert both.n_models == 3 + 3
    assert both.nowcasts.index.tolist() == [
        "base",
        "-real",
        "-soft",
        "-fin",
        "-real -soft",
        "-real -fin",
        "-soft -fin",
    ]
    assert both.skipped == ()


@pytest.mark.parametrize(
    ("drop", "match"),
    [(0, ">= 1"), (True, ">= 1"), ((1, 2.5), ">= 1"), ((), "at least one"), (4, "only 3")],
)
def test_invalid_drop(data, drop, match):
    with pytest.raises(ValueError, match=match):
        alternative_models(TwoStepDFM(n_factors=1), data, "gdp", by=MAPPING, drop=drop)


def test_combinations_removing_every_predictor_are_skipped(data):
    alt = alternative_models(TwoStepDFM(n_factors=1), data, "gdp", by="block", refit=False)
    assert ("global",) in alt.skipped
    assert all("global" not in combo for combo in alt.combinations)
    assert alt.combinations == (("real",), ("soft",), ("real", "soft"))
    assert "skipped" in alt.summary()


def test_drop_sizes_above_the_number_of_groups_are_ignored(data):
    alt = alternative_models(
        TwoStepDFM(n_factors=1), data, "gdp", by=MAPPING, groups=["real"], refit=False
    )
    assert alt.combinations == (("real",),)


def test_every_combination_removing_all_predictors_is_an_error(data):
    with pytest.raises(ValueError, match="removes all the predictors"):
        alternative_models(
            TwoStepDFM(n_factors=1), data, "gdp", by="block", groups=["global"], refit=False
        )


# ---------------------------------------------------------------------------- refit=False
def _masked(data: MixedFrequencyData, columns: list[str]) -> MixedFrequencyData:
    frame = data.to_frame()
    frame[columns] = np.nan
    return data.with_data(frame)


def test_refit_false_equals_update_with_missing_series(data):
    model = TwoStepDFM(n_factors=1)
    alt = alternative_models(model, data, "gdp", by=MAPPING, refit=False)
    fitted = TwoStepDFM(n_factors=1)
    base = fitted.fit(data, "gdp")
    np.testing.assert_allclose(
        alt.nowcasts.loc["base"].to_numpy(), base.out_of_sample.reindex(alt.periods), atol=1e-10
    )
    for combo in alt.combinations:
        dropped = list(alt.dropped_series(combo))
        updated = fitted.update(_masked(data, dropped))
        expected = updated.out_of_sample.reindex(alt.periods).to_numpy()
        got = alt.nowcasts.loc[combination_label(combo)].to_numpy()
        np.testing.assert_allclose(got, expected, atol=1e-9)


def test_refit_false_em_equals_smoother_with_missing_series(data):
    model = MixedFreqDFM(n_factors=1, max_iter=10)
    alt = alternative_models(model, data, "gdp", by="category", drop=1, refit=False)
    base = alt.base
    slots = alt.periods.asfreq("M", how="end")
    for combo in alt.combinations:
        predicted = base.predict(_masked(data, list(alt.dropped_series(combo))))["gdp"]
        got = alt.nowcasts.loc[combination_label(combo)].to_numpy()
        np.testing.assert_allclose(got, predicted.reindex(slots).to_numpy(), atol=1e-8)


def test_refit_false_from_results(data):
    model = TwoStepDFM(n_factors=1)
    results = model.fit(data, "gdp")
    from_results = alternative_models(results, None, "gdp", by=MAPPING, refit=False)
    from_model = alternative_models(TwoStepDFM(n_factors=1), data, "gdp", by=MAPPING, refit=False)
    pd.testing.assert_frame_equal(from_results.nowcasts, from_model.nowcasts)
    assert from_results.base is results
    assert from_results.refit is False and from_results.by == "mapping"


def test_refit_false_from_results_on_a_new_vintage(data):
    """Base and alternatives share the information set of ``data``, not of the fit."""
    fitted = TwoStepDFM(n_factors=1)
    results = fitted.fit(data.truncate(end="2009-06"), "gdp")
    newer = data.truncate(end="2009-08")  # two more months of indicators, same quarter
    alt = alternative_models(results, newer, "gdp", by=MAPPING, refit=False)
    assert alt.periods.tolist() == [pd.Period("2009Q3", "Q")]
    stale = results.out_of_sample.reindex(alt.periods).to_numpy()
    updated = fitted.update(newer).out_of_sample.reindex(alt.periods).to_numpy()
    np.testing.assert_allclose(alt.nowcasts.loc["base"].to_numpy(), updated, atol=1e-9)
    assert not np.allclose(updated, stale)
    expected = fitted.update(_masked(newer, ["x1", "x2"])).out_of_sample.reindex(alt.periods)
    np.testing.assert_allclose(alt.nowcasts.loc["-real"].to_numpy(), expected, atol=1e-9)


def test_refit_false_from_results_ignores_series_outside_the_model(data):
    results = TwoStepDFM(n_factors=1).fit(data.drop(["x6"]), "gdp")
    alt = alternative_models(results, data, "gdp", by="category", refit=False)
    assert "x6" not in {s for members in alt.groups.values() for s in members}
    bare = alternative_models(results.replace(data=None), data, "gdp", by="category", refit=False)
    pd.testing.assert_frame_equal(bare.nowcasts, alt.nowcasts)
    with pytest.raises(ValueError, match="not predictors"):
        alternative_models(results, data, "gdp", by={"other": ["x6"]}, refit=False)


def test_results_input_errors(data):
    results = TwoStepDFM(n_factors=1).fit(data, "gdp")
    with pytest.raises(ValueError, match="refit=True needs an estimator"):
        alternative_models(results, data, "gdp", by=MAPPING, refit=True)
    with pytest.raises(ValueError, match="not for 'x1'"):
        alternative_models(results, data, "x1", by=MAPPING, refit=False)
    with pytest.raises(ValueError, match="No data given"):
        alternative_models(results.replace(data=None), None, "gdp", by=MAPPING, refit=False)


def _plain_results(data: MixedFrequencyData, out_of_sample: bool = True) -> NowcastResults:
    index = pd.period_range("2010Q1", periods=4, freq="Q")
    observed = pd.Series([1.0, 2.0, np.nan, np.nan], index=index)
    estimate = pd.Series([1.1, 1.9, 0.5, 0.4], index=index)
    if not out_of_sample:
        observed = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    return NowcastResults(
        target="gdp",
        nowcast=build_nowcast_frame(observed, estimate),
        model_name="Plain",
        data=data,
    )


def test_refit_false_needs_a_state_space_model(data):
    with pytest.raises(TypeError, match="Use refit=True"):
        alternative_models(_plain_results(data), data, "gdp", by=MAPPING, refit=False)


def test_no_out_of_sample_period(data):
    with pytest.raises(ValueError, match="no out-of-sample nowcast"):
        alternative_models(_plain_results(data, False), data, "gdp", by=MAPPING, refit=False)


# ---------------------------------------------------------------------------- refit=True
def test_refit_true_equals_manual_fits(data, two_step_alt):
    alt = two_step_alt
    assert alt.refit is True and alt.info["model"] == "TwoStepDFM"
    for combo in alt.combinations:
        panel = data.drop(list(alt.dropped_series(combo)))
        expected = TwoStepDFM(n_factors=1).fit(panel, "gdp").out_of_sample.reindex(alt.periods)
        np.testing.assert_allclose(
            alt.nowcasts.loc[combination_label(combo)].to_numpy(), expected.to_numpy()
        )


def test_refit_true_parallel_equals_sequential(data, two_step_alt):
    parallel = alternative_models(TwoStepDFM(n_factors=1), data, "gdp", by=MAPPING, n_jobs=2)
    pd.testing.assert_frame_equal(parallel.nowcasts, two_step_alt.nowcasts)


def test_formula_target_and_fit_kwargs(data):
    calls = []

    class Recorder(TwoStepDFM):
        def fit(self, data, target, **kwargs):
            calls.append((tuple(data.columns), target, kwargs))
            return super().fit(data, target)

        def clone(self):
            return Recorder(n_factors=self.n_factors)

    alt = alternative_models(
        Recorder(n_factors=1),
        data,
        "gdp ~ . - x6",
        by={"a": ["x1", "x2"], "b": ["x3", "x4"]},
        drop=1,
        fit_kwargs={"marker": 1},
    )
    assert alt.n_models == 2
    assert calls[0] == (("x1", "x2", "x3", "x4", "x5", "gdp"), "gdp", {"marker": 1})
    assert calls[1][0] == ("x3", "x4", "x5", "gdp")


def test_model_and_data_validation(data):
    with pytest.raises(TypeError, match="estimator or results"):
        alternative_models(object(), data, "gdp", by=MAPPING)
    with pytest.raises(ValueError, match="data is required"):
        alternative_models(TwoStepDFM(n_factors=1), None, "gdp", by=MAPPING)


def test_no_look_ahead(data):
    """Values after the information set cannot move the alternatives."""
    vintage = data.truncate(end="2009-06")
    later = data.to_frame()
    after = later.index > pd.Period("2009-06", "M")
    later.loc[after] = later.loc[after].where(later.loc[after].isna(), 99.0)
    model = TwoStepDFM(n_factors=1)
    first = alternative_models(model, vintage, "gdp", by=MAPPING, refit=False)
    edited = data.with_data(later).truncate(end="2009-06")
    second = alternative_models(model, edited, "gdp", by=MAPPING, refit=False)
    pd.testing.assert_frame_equal(first.nowcasts, second.nowcasts)


# ---------------------------------------------------------------------------- periods
def test_explicit_periods(data, two_step_alt):
    period = str(two_step_alt.periods[-1])
    alt = alternative_models(
        TwoStepDFM(n_factors=1), data, "gdp", by=MAPPING, periods=period, refit=False
    )
    assert alt.periods.tolist() == [pd.Period(period, "Q")]
    both = alternative_models(
        TwoStepDFM(n_factors=1),
        data,
        "gdp",
        by=MAPPING,
        periods=list(two_step_alt.periods),
        refit=False,
    )
    assert both.periods.equals(two_step_alt.periods)
    with pytest.raises(ValueError, match="No out-of-sample nowcast for the periods"):
        alternative_models(
            TwoStepDFM(n_factors=1), data, "gdp", by=MAPPING, periods="2001Q1", refit=False
        )


# ---------------------------------------------------------------------------- outputs
def test_table_and_deviation(two_step_alt):
    table = two_step_alt.table()
    assert table.index.name == "model"
    assert table.columns[:3].tolist() == ["dropped", "n_dropped", "n_series_dropped"]
    assert table.loc["base", "dropped"] == ""
    assert table.loc["-real -fin", "n_series_dropped"] == 4
    deviation = two_step_alt.table(deviation=True)
    period = str(two_step_alt.periods[0])
    assert deviation.loc["base", period] == 0.0
    expected = table.loc["-soft", period] - table.loc["base", period]
    assert deviation.loc["-soft", period] == pytest.approx(expected)


def test_range(two_step_alt):
    rng = two_step_alt.range()
    alt = two_step_alt.alternatives
    assert rng.index.equals(two_step_alt.periods)
    for period in two_step_alt.periods:
        assert rng.loc[period, "min"] == pytest.approx(alt[period].min())
        assert rng.loc[period, "max"] == pytest.approx(alt[period].max())
        assert rng.loc[period, "median"] == pytest.approx(alt[period].median())
        assert rng.loc[period, "max_model"] == alt[period].idxmax()
        assert rng.loc[period, "min_model"] == alt[period].idxmin()
        assert rng.loc[period, "n_models"] == 6
    assert (rng["spread"] >= 0).all()


def test_range_with_missing_alternatives(two_step_alt):
    nowcasts = two_step_alt.nowcasts.copy()
    nowcasts.iloc[1:, 0] = np.nan
    alt = AlternativeNowcasts(
        target="gdp",
        by="mapping",
        refit=True,
        groups=two_step_alt.groups,
        combinations=two_step_alt.combinations,
        nowcasts=nowcasts,
        base=two_step_alt.base,
    )
    rng = alt.range()
    assert pd.isna(rng.iloc[0]["min_model"]) and rng.iloc[0]["n_models"] == 0


def test_dropped_series(two_step_alt):
    assert two_step_alt.dropped_series(("fin", "real")) == ("x5", "x6", "x1", "x2")
    with pytest.raises(KeyError, match="Unknown group"):
        two_step_alt.dropped_series(("nominal",))


def test_summary(two_step_alt):
    text = two_step_alt.summary()
    assert "Alternative models for 'gdp' (TwoStepDFM)" in text
    assert "6 alternatives, re-estimated" in text
    assert "spread" in text


def test_plot_backends(two_step_alt):
    fig = two_step_alt.plot()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1 + 6 + 1
    assert fig.layout.title.text.startswith("Alternative nowcasts of gdp")
    mpl = two_step_alt.plot(backend="matplotlib", title="Range")
    assert isinstance(mpl, matplotlib.figure.Figure)
    with pytest.raises(ValueError, match="backend"):
        two_step_alt.plot(backend="bokeh")


def test_doctests():
    import doctest

    import nowcastbox.experiment.alternatives as module

    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.failed == 0 and result.attempted > 0

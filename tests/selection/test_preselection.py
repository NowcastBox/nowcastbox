"""Tests for nowcastbox.selection.preselection (ECB-toolbox style pre-selection)."""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning, FormulaError, NowcastDataError
from nowcastbox.selection import (
    PRESELECTION_METHODS,
    PreselectionResult,
    align_to_target,
    hard_threshold,
    lars_select,
    preselect,
    rank_scores,
    sis,
)
from nowcastbox.selection.preselection import _aggregate_score, _check_lags

matplotlib.use("Agg")


def _panel(seed: int = 0, n_months: int = 180, n_noise: int = 6) -> MixedFrequencyData:
    """Monthly indicators and a quarterly target (3rd month of the quarter).

    ``gdp_q = mean_q(sig) + 0.8 * mean_{q-1}(lagged) + noise``; ``sig`` is the strongest
    contemporaneous signal and ``lagged`` only matters with a one-quarter lag.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2005-01", periods=n_months, freq="M")
    frame = pd.DataFrame(index=idx)
    frame["sig"] = rng.normal(size=n_months)
    frame["lagged"] = rng.normal(size=n_months)
    for i in range(n_noise):
        frame[f"n{i}"] = rng.normal(size=n_months)
    q_sig = frame["sig"].rolling(3).mean()
    q_lag = frame["lagged"].rolling(3).mean().shift(3)
    gdp = 2.0 * q_sig + 0.8 * q_lag + 0.2 * rng.normal(size=n_months)
    third = np.asarray(idx.month % 3 == 0)
    frame["gdp"] = np.where(third, gdp, np.nan)
    freq = dict.fromkeys(frame.columns, "M") | {"gdp": "Q"}
    delays = dict.fromkeys(frame.columns, 30) | {"gdp": 45}
    categories = dict.fromkeys(frame.columns, "soft") | {"sig": "hard"}
    return MixedFrequencyData(
        frame,
        freq,
        release_delays=delays,
        categories=categories,
        blocks={c: ["global"] for c in frame.columns},
    )


@pytest.fixture(scope="module")
def panel() -> MixedFrequencyData:
    return _panel()


@pytest.fixture(scope="module")
def result(panel) -> PreselectionResult:
    return preselect(panel, "gdp", x_lags=(0, 1), top=3)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
class TestAlign:
    def test_average_aggregation_and_lags(self, panel):
        frame, y = align_to_target(panel, "gdp", x_lags=(0, 1, -1))
        assert frame.index.freqstr.startswith("Q")
        monthly = panel.to_native("sig")
        q = monthly.groupby(monthly.index.asfreq("Q")).mean()
        np.testing.assert_allclose(frame["sig"].to_numpy(), q.to_numpy())
        np.testing.assert_allclose(frame["sig_lag1"].to_numpy()[1:], q.to_numpy()[:-1])
        np.testing.assert_allclose(frame["sig_lead1"].to_numpy()[:-1], q.to_numpy()[1:])
        assert y.notna().sum() == len(frame) - 1  # first quarter lacks the lagged term

    def test_metadata_and_mapping_aggregation(self, panel):
        mm = panel.with_metadata("sig", aggregation="mariano_murasawa")
        frame, _ = align_to_target(mm, "gdp", x_lags=0, aggregation={"n0": "stock"})
        m = mm.to_native("sig").to_numpy()
        w = np.array([1, 2, 3, 2, 1]) / 3
        np.testing.assert_allclose(frame["sig"].iloc[1], w @ m[5::-1][:5])
        n0 = panel.to_native("n0").to_numpy()
        np.testing.assert_allclose(frame["n0"].to_numpy(), n0[2::3])

    def test_same_frequency_candidate_and_formula(self, panel):
        df = panel.to_frame()
        df["q2"] = df["gdp"] * 2
        freq = dict.fromkeys(df.columns, "M") | {"gdp": "Q", "q2": "Q"}
        mfd = MixedFrequencyData(df, freq)
        frame, y = align_to_target(mfd, "gdp ~ q2 + sig")
        assert list(frame.columns) == ["q2", "sig"]
        np.testing.assert_allclose(frame["q2"].to_numpy(), 2 * y.to_numpy())

    def test_lower_frequency_candidate_skipped(self, panel):
        df = panel.to_frame()
        df["annual"] = np.where(np.asarray(df.index.month == 12), 1.0, np.nan)
        freq = dict.fromkeys(df.columns, "M") | {"gdp": "Q", "annual": "A"}
        mfd = MixedFrequencyData(df, freq)
        with pytest.warns(DataQualityWarning, match="annual"):
            frame, _ = align_to_target(mfd, "gdp")
        assert "annual" not in frame.columns

    def test_no_candidate_raises(self, panel):
        df = panel.to_frame()[["gdp"]].assign(a=np.nan)
        df.loc[df.index.month == 12, "a"] = 1.0
        mfd = MixedFrequencyData(df, {"gdp": "Q", "a": "A"})
        with pytest.warns(DataQualityWarning), pytest.raises(NowcastDataError, match="aligned"):
            align_to_target(mfd, "gdp")

    def test_non_fixed_ratio_skipped(self):
        idx = pd.period_range("2020-01-06", periods=60, freq="W")
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"w": rng.normal(size=60), "m": np.nan}, index=idx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataQualityWarning)
            slots = MixedFrequencyData(df, {"w": "W", "m": "M"}).slot_mask()["m"]
        df.loc[slots.to_numpy(), "m"] = rng.normal(size=int(slots.sum()))
        mfd = MixedFrequencyData(df, {"w": "W", "m": "M"})
        with pytest.warns(DataQualityWarning, match="non-fixed"), pytest.raises(NowcastDataError):
            align_to_target(mfd, "m")

    def test_dataframe_input_and_candidates(self, panel):
        frame, _ = align_to_target(panel, "gdp", candidates=["n1", "sig"])
        assert list(frame.columns) == ["n1", "sig"]
        with pytest.raises(NowcastDataError, match="candidate"):
            align_to_target(panel, "gdp", candidates=["nope"])
        with pytest.raises(NowcastDataError, match="candidate"):
            align_to_target(panel, "gdp", candidates=["gdp"])
        with pytest.raises(NowcastDataError, match="candidate"):
            align_to_target(panel, "gdp", candidates=[])
        with pytest.raises(FormulaError):
            align_to_target(panel, "unknown")

    @pytest.mark.parametrize(
        ("value", "expected"), [(0, (0,)), (2, (0, 1, 2)), ((0, -1), (0, -1)), ([3], (3,))]
    )
    def test_check_lags(self, value, expected):
        assert _check_lags(value) == expected

    @pytest.mark.parametrize("bad", [-1, True, (), (0, 0), (0, 1.5), "1", 1.5, (True,)])
    def test_invalid_lags(self, bad):
        with pytest.raises(ValueError, match="x_lags"):
            _check_lags(bad)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
class TestScores:
    def test_rank_scores_hand_example(self):
        ranks = pd.DataFrame({"m": [1.0, 3.0, 2.0, np.nan, np.nan]})
        assert rank_scores(ranks)["m"].tolist() == [1.0, 0.5, 0.75, 0.125, 0.125]
        assert rank_scores(pd.DataFrame({"m": [1.0]}))["m"].tolist() == [1.0]

    def test_aggregated_score_hand_example(self):
        ranks = pd.DataFrame(
            {"tstat": [1.0, 2.0, 3.0], "sis": [2.0, 1.0, 3.0], "lars": [3.0, 1.0, 2.0]},
            index=["a", "b", "c"],
        )
        equal = _aggregate_score(ranks, {"tstat": 1.0, "sis": 1.0, "lars": 1.0})
        np.testing.assert_allclose(equal.to_numpy(), [0.5, 5 / 6, 1 / 6])
        weighted = _aggregate_score(ranks, {"tstat": 2.0, "sis": 1.0, "lars": 1.0})
        np.testing.assert_allclose(weighted.to_numpy(), [0.625, 0.75, 0.125])

    def test_table_score_consistent_with_ranks(self, result):
        table = result.table()
        ranks = table[[f"rank_{m}" for m in result.methods]]
        expected = rank_scores(ranks).mean(axis=1)
        np.testing.assert_allclose(table["score"].to_numpy(), expected.to_numpy())
        assert table["score"].is_monotonic_decreasing
        assert table["rank"].tolist() == list(range(1, len(table) + 1))


# ---------------------------------------------------------------------------
# preselect
# ---------------------------------------------------------------------------
class TestPreselect:
    def test_planted_signals(self, result):
        assert result.selected[:2] == ["sig", "lagged"]
        table = result.table()
        assert table.loc["sig", "best_lag"] == 0
        assert table.loc["lagged", "best_lag"] == 1
        assert table["selected"].sum() == 3 and result.n_selected == 3
        assert result.methods == PRESELECTION_METHODS
        assert result.x_lags == (0, 1) and result.as_of is None
        assert result.n_obs == 59

    def test_metadata_columns(self, result):
        table = result.table()
        assert table.columns[:4].tolist() == ["rank", "score", "selected", "best_lag"]
        assert table.loc["sig", "frequency"] == "M"
        assert table.loc["sig", "release_delay"] == 30
        assert table.loc["sig", "category"] == "hard"
        assert table.loc["n0", "category"] == "soft"
        assert table.loc["sig", "blocks"] == "global"
        for col in ("abs_t", "abs_corr", "lars_entry"):
            assert col in table.columns

    def test_candidate_level_matches_building_blocks(self, panel, result):
        frame, y = align_to_target(panel, "gdp", x_lags=(0, 1))
        cand = result.table("candidate")
        assert len(cand) == 2 * (panel.n_series - 1)
        tstat = hard_threshold(frame, y, threshold=0.0)
        np.testing.assert_allclose(cand["abs_t"], tstat.scores.reindex(cand.index))
        corr = sis(frame, y, n_predictors=1)
        np.testing.assert_allclose(cand["abs_corr"], corr.scores.reindex(cand.index))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataQualityWarning)
            lars = lars_select(frame, y, n_predictors=1, missing="mean")
        np.testing.assert_allclose(cand["lars_entry"], lars.scores.reindex(cand.index))
        assert set(result.results) == set(PRESELECTION_METHODS)

    def test_series_rank_is_best_lag(self, result):
        cand = result.table("candidate")
        table = result.table()
        best = cand.groupby("series")["rank_sis"].min().rank(method="first")
        np.testing.assert_allclose(table["rank_sis"], best.reindex(table.index))

    def test_single_method_and_weights(self, panel):
        only_sis = preselect(panel, "gdp", methods="sis", top=4)
        assert only_sis.methods == ("sis",)
        assert only_sis.selected == only_sis.ranking("sis").index[:4].tolist()
        zero = preselect(panel, "gdp", weights={"tstat": 0.0, "lars": 0.0}, top=4)
        assert zero.selected == only_sis.selected
        assert zero.weights == {"tstat": 0.0, "sis": 1.0, "lars": 0.0}

    def test_default_and_capped_top(self, panel):
        assert preselect(panel, "gdp", methods=("sis",)).n_selected == panel.n_series - 1
        assert preselect(panel, "gdp", methods=("sis",), top=500).n_selected == 8

    def test_options_passed_through(self, panel):
        res = preselect(
            panel,
            "gdp",
            horizon=1,
            y_lags=1,
            cov_type="nonrobust",
            hac_lags=2,
            lars_method="lasso",
            missing="drop",
            top=2,
        )
        assert res.horizon == 1
        assert res.results["tstat"].params["y_lags"] == 1
        assert res.results["lars"].params["method"] == "lasso"

    def test_min_obs_applies_to_lars(self, panel):
        df = panel.to_frame()
        df["short"] = np.nan
        df.loc[df.index[-12:], "short"] = df["gdp"].ffill().iloc[-12:] * 3  # 4 quarters
        freq = dict.fromkeys(df.columns, "M") | {"gdp": "Q"}
        mfd = MixedFrequencyData(df, freq)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataQualityWarning)
            res = preselect(mfd, "gdp", top=3)
        row = res.table().loc["short"]
        assert np.isnan(row["rank_tstat"]) and np.isnan(row["rank_sis"])
        assert np.isnan(row["rank_lars"]) and np.isnan(row["lars_entry"])
        assert res.table("candidate")["rank_lars"].notna().sum() == len(res.table()) - 1
        with pytest.raises(NowcastDataError, match="enough observations for LARS"):
            preselect(mfd, "gdp", methods=("lars",), min_obs=500)

    def test_dataframe_input(self, panel):
        df = panel.to_frame()
        res = preselect(df, "gdp", methods=("tstat",), top=1, x_lags=0)
        assert res.selected == ["sig"]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"methods": ("tstat", "ridge")}, "methods"),
            ({"methods": ()}, "methods"),
            ({"methods": ("sis", "sis")}, "methods"),
            ({"weights": {"ridge": 1.0}}, "weights"),
            ({"weights": {"sis": -1.0}}, "weights"),
            ({"weights": {"sis": 0.0, "tstat": 0.0, "lars": 0.0}}, "weights"),
            ({"top": 0}, "top"),
            ({"top": True}, "top"),
        ],
    )
    def test_invalid_arguments(self, panel, kwargs, match):
        with pytest.raises(ValueError, match=match):
            preselect(panel, "gdp", **kwargs)


# ---------------------------------------------------------------------------
# No look-ahead
# ---------------------------------------------------------------------------
class TestAsOf:
    def test_only_released_data_used(self, panel):
        vintage = pd.Timestamp("2012-05-20")
        base = preselect(panel, "gdp", x_lags=(0, 1), as_of=vintage, top=3)
        # Scramble everything released after the vintage: the result must not change.
        released = panel.as_of(vintage).to_frame().notna()
        rng = np.random.default_rng(99)
        frame = panel.to_frame()
        noise = pd.DataFrame(rng.normal(size=frame.shape), index=frame.index, columns=frame.columns)
        scrambled = panel.with_data(frame.where(released | frame.isna(), noise * 50))
        again = preselect(scrambled, "gdp", x_lags=(0, 1), as_of=vintage, top=3)
        pd.testing.assert_frame_equal(base.table(), again.table())
        assert base.as_of == vintage
        assert base.n_obs < 59

    def test_equivalent_to_masked_panel(self, panel):
        vintage = "2010-11-30"
        a = preselect(panel, "gdp", as_of=vintage, top=3)
        b = preselect(panel.as_of(vintage), "gdp", top=3)
        pd.testing.assert_frame_equal(a.table(), b.table())
        assert "as_of: 2010-11-30" in a.summary()

    def test_release_delay_override_and_missing(self, panel):
        res = preselect(panel, "gdp", as_of="2010-11-30", release_delays={"gdp": 400}, top=2)
        assert res.n_obs < preselect(panel, "gdp", as_of="2010-11-30", top=2).n_obs
        df = panel.to_frame()
        no_delay = MixedFrequencyData(df, dict.fromkeys(df.columns, "M") | {"gdp": "Q"})
        with pytest.raises(NowcastDataError, match="release delay"):
            preselect(no_delay, "gdp", as_of="2010-11-30")


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------
class TestResult:
    def test_ranking(self, result):
        final = result.ranking()
        assert final.index[:3].tolist() == result.selected
        assert final.name == "rank"
        assert result.ranking("lars").is_monotonic_increasing
        with pytest.raises(ValueError, match="method"):
            result.ranking("ridge")

    def test_table_levels(self, result):
        assert result.table("candidate").index.name == "candidate"
        with pytest.raises(ValueError, match="level"):
            result.table("block")  # type: ignore[arg-type]

    def test_transform(self, panel, result):
        assert result.transform(panel).columns == [*result.selected, "gdp"]
        assert result.transform(panel, keep_target=False).columns == result.selected
        with pytest.raises(NowcastDataError, match="not found"):
            result.transform(panel.select(["gdp", "n0"]))

    def test_summary(self, result):
        text = str(result)
        assert text.startswith("Pre-selection of indicators for gdp")
        assert "as_of: -" in text and "sig" in text

    def test_plot_matplotlib(self, result):
        import matplotlib.pyplot as plt

        ax = result.plot()
        assert len(ax.patches) == len(result.table())
        _, ax2 = plt.subplots()
        assert result.plot(top=2, ax=ax2) is ax2
        assert len(ax2.patches) == 2
        plt.close("all")

    def test_plot_plotly(self, result):
        fig = result.plot(backend="plotly", top=4)
        assert len(fig.data[0].y) == 4
        with pytest.raises(ValueError, match="ax"):
            result.plot(backend="plotly", ax=object())
        with pytest.raises(ValueError, match="backend"):
            result.plot(backend="bokeh")  # type: ignore[arg-type]

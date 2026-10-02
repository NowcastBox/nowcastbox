"""Hand-computed checks of the weighted score of the specification search."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.exceptions import DataQualityWarning
from nowcastbox.selection import weighted_score


@pytest.fixture
def criteria():
    return pd.DataFrame(
        {
            "rmsfe|backcast": [1.0, 2.0, 3.0, 4.0],
            "rmsfe|nowcast": [2.0, 1.0, 4.0, 3.0],
            "rmsfe|forecast": [3.0, 3.5, 1.0, 2.0],
            "fda|nowcast": [0.5, 0.75, 1.0, 0.25],
            "fda|backcast": [0.6, 0.6, 0.6, 0.6],
            "bias|nowcast": [-0.3, 0.1, 0.2, -0.05],
            "ref|rmsfe|nowcast": [2.0, 2.0, 2.0, 2.0],
            "ref|fda|nowcast": [0.5, 0.5, 0.5, 0.5],
            "status": ["ok"] * 4,
        },
        index=pd.Index([10, 11, 12, 13], name="draw"),
    )


class TestRankNormalisation:
    def test_single_criterion_is_the_normalised_rank(self, criteria):
        out = weighted_score(criteria, {"rmsfe": 1}, {"nowcast": 1})
        # ranks 2, 1, 4, 3 -> (r - 1) / 3
        np.testing.assert_allclose(out, [1 / 3, 0.0, 1.0, 2 / 3])
        assert out.name == "score"
        assert list(out.index) == [10, 11, 12, 13]

    def test_metric_and_horizon_weights_by_hand(self, criteria):
        score = {"rmsfe": 0.7, "fda": 0.3}
        horizons = {"nowcast": 0.5, "backcast": 0.25}
        out = weighted_score(criteria, score, horizons)
        r_now = np.array([1, 0, 3, 2]) / 3
        r_back = np.array([0, 1, 2, 3]) / 3
        # fda nowcast: losses 0.5, 0.25, 0, 0.75 -> ranks 3, 2, 1, 4
        f_now = np.array([2, 1, 0, 3]) / 3
        # fda backcast: four ties -> average rank 2.5 -> 0.5
        f_back = np.full(4, 0.5)
        expected = (
            0.7 * 0.5 * r_now + 0.7 * 0.25 * r_back + 0.3 * 0.5 * f_now + 0.3 * 0.25 * f_back
        ) / (1.0 * 0.75)
        np.testing.assert_allclose(out, expected)

    def test_default_horizon_weights_are_equal(self, criteria):
        out = weighted_score(criteria, {"rmsfe": 1})
        expected = (
            np.array([0, 1, 2, 3]) / 3 + np.array([1, 0, 3, 2]) / 3 + np.array([2, 3, 0, 1]) / 3
        ) / 3
        np.testing.assert_allclose(out, expected)

    def test_default_score_is_rmsfe(self, criteria):
        pd.testing.assert_series_equal(
            weighted_score(criteria, None, {"nowcast": 1}),
            weighted_score(criteria, {"rmsfe": 1}, {"nowcast": 1}),
        )

    def test_metric_name_as_string(self, criteria):
        pd.testing.assert_series_equal(
            weighted_score(criteria, "fda", {"nowcast": 1}),
            weighted_score(criteria, {"fda": 2.0}, {"nowcast": 1}),
        )

    def test_bias_uses_absolute_value(self, criteria):
        out = weighted_score(criteria, "bias", {"nowcast": 1})
        # |bias| 0.3, 0.1, 0.2, 0.05 -> ranks 4, 2, 3, 1
        np.testing.assert_allclose(out, np.array([3, 1, 2, 0]) / 3)

    def test_zero_weights_are_ignored(self, criteria):
        out = weighted_score(criteria, {"rmsfe": 1, "mae": 0}, {"nowcast": 1, "x": 0})
        np.testing.assert_allclose(out, np.array([1, 0, 3, 2]) / 3)

    def test_missing_value_gives_inf_and_is_excluded_from_ranks(self, criteria):
        criteria.loc[12, "rmsfe|nowcast"] = np.nan
        out = weighted_score(criteria, "rmsfe", {"nowcast": 1})
        assert np.isinf(out.loc[12])
        np.testing.assert_allclose(out.drop(12), [0.5, 0.0, 1.0])

    def test_single_row(self, criteria):
        out = weighted_score(criteria.head(1), "rmsfe", {"nowcast": 1})
        assert out.tolist() == [0.0]


class TestOtherNormalisations:
    def test_none_uses_raw_losses(self, criteria):
        out = weighted_score(criteria, {"rmsfe": 1, "fda": 1}, {"nowcast": 1}, normalize="none")
        np.testing.assert_allclose(out, ([2, 1, 4, 3] + (1 - criteria["fda|nowcast"])) / 2)

    def test_relative_to_the_reference(self, criteria):
        out = weighted_score(criteria, {"rmsfe": 1, "fda": 1}, {"nowcast": 1}, normalize="relative")
        rel_rmsfe = np.array([2, 1, 4, 3]) / 2.0
        rel_fda = (1 - criteria["fda|nowcast"].to_numpy()) / 0.5
        np.testing.assert_allclose(out, (rel_rmsfe + rel_fda) / 2)

    def test_relative_needs_reference(self, criteria):
        with pytest.raises(ValueError, match="reference benchmark"):
            weighted_score(criteria, "rmsfe", {"backcast": 1}, normalize="relative")

    def test_relative_with_zero_reference_is_inf(self, criteria):
        criteria["ref|fda|nowcast"] = 1.0
        out = weighted_score(criteria, "fda", {"nowcast": 1}, normalize="relative")
        assert np.isinf(out).all()


class TestValidation:
    @pytest.mark.parametrize(
        ("score", "message"),
        [
            ({}, "non-empty mapping"),
            (["rmsfe"], "non-empty mapping"),
            ({"rmse": 1}, "Unknown score metric"),
            ({"n": 1}, "Unknown score metric"),
            ({"rmsfe": -1}, "non-negative"),
            ({"rmsfe": np.inf}, "non-negative"),
            ({"rmsfe": 0}, "positive sum"),
        ],
    )
    def test_bad_score(self, criteria, score, message):
        with pytest.raises(ValueError, match=message):
            weighted_score(criteria, score)

    @pytest.mark.parametrize(
        ("weights", "message"),
        [
            ({}, "non-empty mapping"),
            ([1.0], "non-empty mapping"),
            ({"nowcast": -1}, "non-negative"),
            ({"nowcast": 0}, "positive sum"),
        ],
    )
    def test_bad_horizon_weights(self, criteria, weights, message):
        with pytest.raises(ValueError, match=message):
            weighted_score(criteria, "rmsfe", weights)

    def test_unknown_horizon(self, criteria):
        with pytest.raises(ValueError, match="None of the weighted horizons"):
            weighted_score(criteria, "rmsfe", {"month2": 1})

    def test_horizon_not_evaluated_is_dropped(self, criteria):
        with pytest.warns(DataQualityWarning, match=r"Horizons \['month2'\]"):
            out = weighted_score(criteria, "rmsfe", {"month2": 3, "nowcast": 1})
        np.testing.assert_allclose(out, np.array([1, 0, 3, 2]) / 3)

    def test_metric_missing_for_an_evaluated_horizon(self, criteria):
        with pytest.raises(ValueError, match="'fda\\|forecast' was not computed"):
            weighted_score(criteria, {"rmsfe": 1, "fda": 1}, {"forecast": 1})

    def test_metric_not_computed(self, criteria):
        with pytest.raises(ValueError, match="No 'mae' criterion"):
            weighted_score(criteria, "mae")

    def test_unknown_normalisation(self, criteria):
        with pytest.raises(ValueError, match="normalize must be one of"):
            weighted_score(criteria, normalize="zscore")

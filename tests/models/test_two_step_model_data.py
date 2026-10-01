"""TwoStepResults.model_data / filter_panel / prefiltered fits (lacuna 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nowcastbox.core.exceptions import NowcastDataError
from nowcastbox.models import TwoStepDFM
from nowcastbox.models.two_step import TwoStepResults, simulate_two_step_example
from nowcastbox.preprocessing.aggregation import rolling_aggregate


@pytest.fixture(scope="module")
def data():
    return simulate_two_step_example(n_periods=150, n_series=8, random_state=3)


@pytest.fixture(scope="module")
def fitted_variables(data):
    model = TwoStepDFM(n_factors=1, aggregate="variables")
    return model, model.fit(data, "gdp")


def test_model_data_is_filtered_panel(data, fitted_variables):
    _, res = fitted_variables
    md = res.model_data
    assert md is not None
    assert res.is_filtered
    assert md.index.equals(data.index)
    assert sorted(md.columns) == sorted(data.columns)
    w = res.variable_weights["x1"]
    np.testing.assert_allclose(w, np.array([1, 2, 3, 2, 1]) / 3.0)
    expected = rolling_aggregate(data.to_frame()["x1"].to_numpy(), w)
    np.testing.assert_allclose(md.to_frame()["x1"].to_numpy(), expected, equal_nan=True)
    # target unchanged
    pd.testing.assert_series_equal(md.to_frame()["gdp"], data.to_frame()["gdp"])


def test_model_data_matches_state_space_observations(fitted_variables):
    _, res = fitted_variables
    z = res.standardization.transform(res.model_data.to_frame()[list(res.loadings.index)])
    xf = res.x_filled.reindex(res.model_data.index)[list(res.loadings.index)]
    observed = z.notna()
    # x_filled keeps the observed (filtered) values of the model panel
    np.testing.assert_allclose(
        xf.where(observed).to_numpy(),
        res.model_data.to_frame()[list(res.loadings.index)].where(observed).to_numpy(),
        equal_nan=True,
    )


def test_prefiltered_fit_reproduces_estimates(data, fitted_variables):
    _, res = fitted_variables
    again = TwoStepDFM(n_factors=1, aggregate="variables").fit(
        res.model_data, "gdp", prefiltered=True
    )
    pd.testing.assert_frame_equal(again.nowcast, res.nowcast)
    assert again.info["prefiltered"] is True
    assert res.info["prefiltered"] is False
    assert again.model_data.equals(res.model_data)


def test_filter_panel_on_vintage_and_update(data, fitted_variables):
    model, res = fitted_variables
    frame = data.to_frame()
    frame.iloc[-6:, :3] = np.nan
    vintage = data.with_data(frame)
    filtered = res.filter_panel(vintage)
    raw_update = model.update(vintage)
    pre_update = model.update(filtered, prefiltered=True)
    pd.testing.assert_frame_equal(raw_update.nowcast, pre_update.nowcast)
    assert raw_update.model_data.equals(filtered)
    # DataFrame input with frequencies
    again = res.filter_panel(frame, frequency=data.frequencies.to_dict())
    assert again.equals(filtered)


def test_filter_panel_factors_mode_is_identity(data):
    res = TwoStepDFM(n_factors=1).fit(data, "gdp")
    assert not res.is_filtered
    assert res.variable_weights == {}
    assert res.filter_panel(data).equals(data.select(list(res.model_data.columns)))
    # prefiltered has no effect without filters
    same = TwoStepDFM(n_factors=1).fit(data, "gdp", prefiltered=True)
    pd.testing.assert_frame_equal(same.nowcast, res.nowcast)


def test_filter_panel_errors(data, fitted_variables):
    _, res = fitted_variables
    with pytest.raises(NowcastDataError, match="lacks the series"):
        res.filter_panel(data.drop(["x2"]))
    bare = TwoStepResults(
        target="gdp",
        nowcast=res.nowcast,
        model_name="TwoStepDFM",
    )
    with pytest.raises(ValueError, match="do not store the model panel"):
        bare.filter_panel(data)
    assert not bare.is_filtered


def test_prefiltered_validation(data):
    model = TwoStepDFM(n_factors=1, aggregate="variables")
    with pytest.raises(ValueError, match="prefiltered must be a bool"):
        model.fit(data, "gdp", prefiltered="yes")
    with pytest.raises(TypeError, match="Unexpected fit options"):
        model.fit(data, "gdp", other=1)
    model.fit(data, "gdp")
    with pytest.raises(ValueError, match="prefiltered must be a bool"):
        model.update(data, prefiltered=1)


def test_parametric_bootstrap_contract(data, fitted_variables):
    """A panel simulated from the state space can be refitted with prefiltered=True."""
    from nowcastbox.statespace import simulate_state_space

    model, res = fitted_variables
    md = res.model_data
    cols = list(res.loadings.index)
    obs, _ = simulate_state_space(res.state_space, md.n_periods, random_state=1)
    sim = res.standardization.inverse_transform(pd.DataFrame(obs, index=md.index, columns=cols))
    frame = md.to_frame()
    mask = md.observation_mask()
    for c in cols:
        frame[c] = sim[c].where(mask[c])
    panel = md.with_data(frame)
    boot = TwoStepDFM(**model.get_params()).fit(panel, "gdp", prefiltered=True)
    est = TwoStepDFM(**model.get_params())
    est.fit(panel, "gdp", prefiltered=True)
    out = est.update(data)  # raw data, filtered with the stored weights
    assert np.isfinite(out.get_nowcast())
    assert boot.loadings.shape == res.loadings.shape

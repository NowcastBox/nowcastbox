"""Tests of the pluggable indicator extrapolators."""

from __future__ import annotations

import doctest
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

import nowcastbox.models.extrapolation as ext_module
from nowcastbox.core.data import MixedFrequencyData
from nowcastbox.core.exceptions import DataQualityWarning, NowcastDataError
from nowcastbox.models import BridgeCombination
from nowcastbox.models.bridge import ar_extend
from nowcastbox.models.extrapolation import (
    ARExtrapolator,
    Extrapolator,
    available_extrapolators,
    make_extrapolator,
    native_until,
    register_extrapolator,
)


@pytest.fixture(autouse=True)
def clean_registry() -> Iterator[None]:
    saved = dict(ext_module._REGISTRY)
    yield
    ext_module._REGISTRY.clear()
    ext_module._REGISTRY.update(saved)


@pytest.fixture
def mfd(rng) -> MixedFrequencyData:
    idx = pd.period_range("2000-01", periods=60, freq="M")
    x = rng.standard_normal(60)
    x[-2:] = np.nan
    q = np.where(idx.month % 3 == 0, rng.standard_normal(60), np.nan)
    q[-3:] = np.nan
    frame = pd.DataFrame({"x": x, "q": q}, index=idx)
    return MixedFrequencyData(frame, {"x": "M", "q": "Q"})


def test_ar_extrapolator_matches_ar_extend(mfd) -> None:
    out = ARExtrapolator(ar_lags=2)(mfd, ["x", "q"], pd.Period("2005Q2", "Q"))
    assert out["x"].index[-1] == pd.Period("2005-06", "M")
    assert out["q"].index[-1] == pd.Period("2005Q2", "Q")
    raw = mfd.to_native("x").to_numpy()
    expected = ar_extend(raw, 66 - 58, 2)  # 2000-01..2005-06, last observation 2004-10
    np.testing.assert_allclose(out["x"].to_numpy(), expected)
    assert out["q"].notna().all()


def test_native_until(mfd) -> None:
    series = native_until(mfd, "q", pd.Period("2005-12", "M"))
    assert series.index[-1] == pd.Period("2005Q4", "Q")
    assert series.iloc[-4:].isna().all()


def test_no_observations() -> None:
    idx = pd.period_range("2000-01", periods=12, freq="M")
    frame = pd.DataFrame({"x": np.nan, "y": np.arange(12.0)}, index=idx)
    with pytest.warns(DataQualityWarning):
        data = MixedFrequencyData(frame, "M")
    with pytest.raises(NowcastDataError, match="no observations"):
        ARExtrapolator()(data, ["x"], pd.Period("2001-03", "M"))


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_ar_lags(value) -> None:
    with pytest.raises(ValueError, match="ar_lags"):
        ARExtrapolator(ar_lags=value)


def test_registry() -> None:
    assert "ar" in available_extrapolators()
    assert isinstance(make_extrapolator("ar"), Extrapolator)
    assert make_extrapolator("ar", ar_lags=4) == ARExtrapolator(ar_lags=4)
    with pytest.raises(ValueError, match="already registered"):
        register_extrapolator("ar", ARExtrapolator)
    with pytest.raises(ValueError, match="non-empty"):
        register_extrapolator("", ARExtrapolator)
    with pytest.raises(TypeError, match="callable"):
        register_extrapolator("x", 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown extrapolation"):
        make_extrapolator("bvar")
    with pytest.raises(TypeError, match="name or a callable"):
        make_extrapolator(3)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        make_extrapolator("ar", lags=2)


def test_callable_passthrough() -> None:
    def fill(data, columns, end):
        return {}

    assert make_extrapolator(fill) is fill
    with pytest.raises(ValueError, match="only used with a registered name"):
        make_extrapolator(fill, ar_lags=2)


def test_registered_extrapolator_used_by_bridge_combination(mfd) -> None:
    """A Phase-3 style plug-in ("bvar") only has to register a factory."""
    seen: dict[str, object] = {}

    class LastValue:
        def __init__(self, scale: float = 1.0) -> None:
            self.scale = scale

        def __call__(self, data, columns, end):
            seen["columns"] = list(columns)
            return {c: native_until(data, c, end).ffill() * 1.0 for c in columns}

    register_extrapolator("last", LastValue)
    idx = mfd.index
    frame = mfd.to_frame()
    frame["y"] = (
        pd.Series(np.nan_to_num(frame["x"].to_numpy()), index=idx)
        .rolling(3)
        .mean()
        .where(idx.month % 3 == 0)
    )
    frame.iloc[-3:, 2] = np.nan
    data = MixedFrequencyData(frame, {"x": "M", "q": "Q", "y": "Q"})
    res = BridgeCombination(extrapolation="last", extrapolation_options={"scale": 2.0}).fit(
        data, "y"
    )
    assert seen["columns"] == ["x", "q"]
    last = frame["x"].dropna().iloc[-1]
    assert res.extrapolated["x"].iloc[-1] == pytest.approx(last)
    assert res.model_params["extrapolation"] == "last"


def test_doctests() -> None:
    result = doctest.testmod(ext_module, optionflags=doctest.ELLIPSIS)
    assert result.attempted > 0
    assert result.failed == 0

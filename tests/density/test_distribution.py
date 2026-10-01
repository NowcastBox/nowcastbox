"""Tests of NowcastDistribution (Gaussian / Gaussian-mixture predictive distribution)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.stats
from hypothesis import given
from hypothesis import strategies as st

from nowcastbox.density import DEFAULT_LEVELS, NowcastDistribution

PERIODS = ["2020Q1", "2020Q2", "2020Q3"]


@pytest.fixture
def gaussian() -> NowcastDistribution:
    return NowcastDistribution(PERIODS, [0.5, 1.0, -0.2], [0.2, 0.4, 1.0], target="gdp")


@pytest.fixture
def mixture() -> NowcastDistribution:
    rng = np.random.default_rng(0)
    locs = rng.normal(size=(2, 7)) + np.array([[0.0], [3.0]])
    scales = rng.uniform(0.3, 1.2, size=(2, 7))
    return NowcastDistribution(["2021Q1", "2021Q2"], locs, scales, target="gdp")


# ---------------------------------------------------------------------- construction
class TestConstruction:
    def test_attributes(self, gaussian: NowcastDistribution) -> None:
        assert gaussian.is_gaussian
        assert gaussian.n_periods == len(gaussian) == 3
        assert gaussian.n_components == 1
        assert gaussian.target == "gdp"
        assert isinstance(gaussian.index, pd.PeriodIndex)
        assert gaussian.index.name == "period"
        assert gaussian.weights.tolist() == [1.0]
        assert gaussian.locs.shape == gaussian.scales.shape == (3, 1)
        assert "Gaussian" in repr(gaussian) and "'gdp'" in repr(gaussian)
        assert gaussian.info == {}

    def test_arrays_read_only(self, gaussian: NowcastDistribution) -> None:
        with pytest.raises(ValueError):
            gaussian.locs[0, 0] = 1.0

    def test_period_index_input_and_single_period_repr(self) -> None:
        idx = pd.period_range("2020Q1", periods=1, freq="Q")
        d = NowcastDistribution(idx, 1.0, 2.0)
        assert d.locs.shape == (1, 1)
        assert repr(d) == "NowcastDistribution(2020Q1, Gaussian)"

    def test_mixture_repr_and_row_vector(self) -> None:
        d = NowcastDistribution(["2020Q1"], [0.0, 1.0, 2.0], [1.0, 1.0, 1.0])
        assert d.n_components == 3
        assert "mixture of 3" in repr(d)

    def test_weights_normalised(self) -> None:
        d = NowcastDistribution(["2020Q1"], [[0.0, 2.0]], [[1.0, 1.0]], weights=[1, 3])
        np.testing.assert_allclose(d.weights, [0.25, 0.75])
        assert d.mean.iloc[0] == pytest.approx(1.5)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"index": [], "locs": [], "scales": []}, "at least one"),
            ({"index": ["2020Q1", "2020Q1"], "locs": [0, 0], "scales": [1, 1]}, "duplicated"),
            ({"index": [object()], "locs": [0], "scales": [1]}, "periods"),
            ({"index": ["2020Q1"], "locs": [[0, 1]], "scales": [[1]]}, "same shape"),
            ({"index": ["2020Q1"], "locs": [0], "scales": [0]}, "positive"),
            ({"index": ["2020Q1"], "locs": [np.nan], "scales": [1]}, "finite"),
            ({"index": ["2020Q1", "2020Q2"], "locs": [0, 1, 2], "scales": [1, 1, 1]}, "shape"),
            ({"index": ["2020Q1"], "locs": np.zeros((1, 1, 1)), "scales": [1]}, "shape"),
            ({"index": ["2020Q1"], "locs": [0], "scales": [1], "weights": [1, 1]}, "weights"),
            ({"index": ["2020Q1"], "locs": [0], "scales": [1], "weights": [-1]}, "weights"),
            ({"index": ["2020Q1"], "locs": [0], "scales": [1], "weights": [0.0]}, "zero"),
            ({"index": ["2020Q1"], "locs": [0], "scales": [1], "point": [1, 2]}, "point"),
            ({"index": ["2020Q1"], "locs": [0], "scales": [1], "point": [np.inf]}, "point"),
        ],
    )
    def test_invalid(self, kwargs: dict, match: str) -> None:
        index = kwargs.pop("index")
        locs = kwargs.pop("locs")
        scales = kwargs.pop("scales")
        with pytest.raises(ValueError, match=match):
            NowcastDistribution(index, locs, scales, **kwargs)

    def test_empty_period_index(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            NowcastDistribution(pd.PeriodIndex([], freq="Q"), [], [])


# ---------------------------------------------------------------------- moments
class TestMoments:
    def test_gaussian_moments(self, gaussian: NowcastDistribution) -> None:
        np.testing.assert_allclose(gaussian.mean, [0.5, 1.0, -0.2])
        np.testing.assert_allclose(gaussian.std, [0.2, 0.4, 1.0])
        np.testing.assert_allclose(gaussian.variance, [0.04, 0.16, 1.0])
        np.testing.assert_allclose(gaussian.median, [0.5, 1.0, -0.2], atol=1e-12)
        pd.testing.assert_series_equal(gaussian.point, gaussian.mean.rename("point"))

    def test_mixture_moments_law_of_total_variance(self, mixture: NowcastDistribution) -> None:
        locs, scales = mixture.locs, mixture.scales
        np.testing.assert_allclose(mixture.mean, locs.mean(axis=1))
        expected = (scales**2).mean(axis=1) + locs.var(axis=1)
        np.testing.assert_allclose(mixture.variance, expected)
        dec = mixture.variance_decomposition
        np.testing.assert_allclose(dec["filtering"], (scales**2).mean(axis=1))
        np.testing.assert_allclose(dec["parameter"], locs.var(axis=1))
        np.testing.assert_allclose(dec["total"], expected)

    def test_moments_vs_monte_carlo(self, mixture: NowcastDistribution) -> None:
        draws = mixture.sample(200_000, random_state=1)
        np.testing.assert_allclose(draws.mean().to_numpy(), mixture.mean, atol=0.02)
        np.testing.assert_allclose(draws.std().to_numpy(), mixture.std, rtol=0.01)
        q = mixture.quantiles([0.1, 0.5, 0.9])
        emp = draws.quantile([0.1, 0.5, 0.9]).T.to_numpy()
        np.testing.assert_allclose(emp, q.to_numpy(), atol=0.03)

    def test_explicit_point_and_decomposition(self) -> None:
        dec = pd.DataFrame({"filtering": [1.0], "parameter": [0.0], "total": [1.0]})
        d = NowcastDistribution(
            ["2020Q1"], [0.0], [1.0], point=[0.3], variance_decomposition=dec, info={"a": 1}
        )
        assert d.point.iloc[0] == 0.3
        assert d.mean.iloc[0] == 0.0
        pd.testing.assert_frame_equal(d.variance_decomposition, dec)
        assert d.info == {"a": 1}


# ---------------------------------------------------------------------- pdf / cdf
class TestDensity:
    def test_gaussian_matches_scipy(self, gaussian: NowcastDistribution) -> None:
        x = np.array([0.1, 1.3, -2.0])
        mu, sd = gaussian.locs[:, 0], gaussian.scales[:, 0]
        np.testing.assert_allclose(gaussian.cdf(x), scipy.stats.norm.cdf(x, mu, sd))
        np.testing.assert_allclose(gaussian.pdf(x), scipy.stats.norm.pdf(x, mu, sd))
        np.testing.assert_allclose(gaussian.logpdf(x), scipy.stats.norm.logpdf(x, mu, sd))

    def test_broadcast_grid(self, gaussian: NowcastDistribution) -> None:
        grid = np.linspace(-3, 3, 11)[:, None]
        assert gaussian.cdf(grid).shape == (11, 3)
        single = gaussian["2020Q2"]
        assert single.pdf(np.linspace(-1, 1, 5)).shape == (5,)

    def test_mixture_pdf_integrates_to_one(self, mixture: NowcastDistribution) -> None:
        grid = np.linspace(-10, 14, 20001)[:, None]
        dens = mixture.pdf(grid)
        integral = np.trapezoid(dens, grid[:, 0], axis=0)
        np.testing.assert_allclose(integral, 1.0, atol=1e-6)

    def test_mixture_cdf_vs_scipy(self, mixture: NowcastDistribution) -> None:
        x = np.array([0.4, 2.5])
        expected = scipy.stats.norm.cdf(x[:, None], mixture.locs, mixture.scales).mean(axis=1)
        np.testing.assert_allclose(mixture.cdf(x), expected)

    def test_zero_weight_component(self) -> None:
        d = NowcastDistribution(["2020Q1"], [[0.0, 5.0]], [[1.0, 1.0]], weights=[1.0, 0.0])
        np.testing.assert_allclose(d.logpdf(0.0), scipy.stats.norm.logpdf(0.0))


# ---------------------------------------------------------------------- quantiles
class TestQuantiles:
    def test_gaussian_quantiles(self, gaussian: NowcastDistribution) -> None:
        q = gaussian.quantiles([0.05, 0.95])
        assert list(q.columns) == [0.05, 0.95]
        expected = scipy.stats.norm.ppf(0.95, gaussian.locs[:, 0], gaussian.scales[:, 0])
        np.testing.assert_allclose(q[0.95], expected)

    def test_mixture_ppf_inverts_cdf(self, mixture: NowcastDistribution) -> None:
        probs = np.array([0.001, 0.05, 0.37, 0.5, 0.92, 0.999])
        q = mixture.ppf(probs)
        for j, p in enumerate(probs):
            np.testing.assert_allclose(mixture.cdf(q[:, j]), p, atol=1e-10)

    def test_interval(self, mixture: NowcastDistribution) -> None:
        band = mixture.interval(0.8)
        np.testing.assert_allclose(mixture.cdf(band["lower"].to_numpy()), 0.1, atol=1e-10)
        np.testing.assert_allclose(mixture.cdf(band["upper"].to_numpy()), 0.9, atol=1e-10)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, [0.5, 1.2], [], np.ones((2, 2)) * 0.5])
    def test_invalid_probabilities(self, gaussian: NowcastDistribution, bad: object) -> None:
        with pytest.raises(ValueError):
            gaussian.quantiles(bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0.0, 1.0, [], [0.5, 1.5]])
    def test_invalid_levels(self, gaussian: NowcastDistribution, bad: object) -> None:
        with pytest.raises(ValueError):
            gaussian.to_frame(bad)  # type: ignore[arg-type]

    @given(
        locs=st.lists(st.floats(-5, 5), min_size=1, max_size=5),
        log_scales=st.lists(st.floats(-2, 1), min_size=5, max_size=5),
        p=st.floats(0.001, 0.999),
    )
    def test_property_ppf_cdf_roundtrip(
        self, locs: list[float], log_scales: list[float], p: float
    ) -> None:
        k = len(locs)
        d = NowcastDistribution(["2020Q1"], [locs], [np.exp(log_scales[:k])])
        q = d.ppf([p])[0, 0]
        assert d.cdf(q)[0] == pytest.approx(p, abs=1e-9)
        lo, hi = d.interval(0.5).to_numpy()[0]
        assert lo <= hi
        if p < 0.25:
            assert q <= lo + 1e-9
        elif p > 0.75:
            assert q >= hi - 1e-9


# ---------------------------------------------------------------------- sampling / selection
class TestSampleSelect:
    def test_sample_reproducible(self, mixture: NowcastDistribution) -> None:
        a = mixture.sample(50, random_state=3)
        b = mixture.sample(50, random_state=3)
        pd.testing.assert_frame_equal(a, b)
        assert list(a.columns) == list(mixture.index)

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True])
    def test_sample_invalid(self, gaussian: NowcastDistribution, bad: object) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            gaussian.sample(bad)  # type: ignore[arg-type]

    def test_select_and_getitem(self, gaussian: NowcastDistribution) -> None:
        sub = gaussian.select(["2020Q3", 0])
        assert [str(p) for p in sub.index] == ["2020Q3", "2020Q1"]
        assert gaussian[-1].mean.iloc[0] == -0.2
        assert gaussian[pd.Period("2020-05", freq="M")].mean.iloc[0] == 1.0
        with pytest.raises(KeyError):
            gaussian["2019Q1"]
        with pytest.raises(KeyError):
            gaussian[5]

    def test_select_keeps_decomposition(self) -> None:
        dec = pd.DataFrame(
            {"filtering": [1.0, 2.0], "parameter": [0.1, 0.2], "total": [1.1, 2.2]},
            index=pd.period_range("2020Q1", periods=2, freq="Q"),
        )
        d = NowcastDistribution(dec.index, [0, 0], [1, 1], variance_decomposition=dec)
        assert d["2020Q2"].variance_decomposition["total"].tolist() == [2.2]


# ---------------------------------------------------------------------- export / plot
class TestExport:
    def test_to_frame(self, gaussian: NowcastDistribution) -> None:
        frame = gaussian.to_frame()
        labels = ["50", "68", "90"]
        expected = ["point", "mean", "std", "median"]
        expected += [f"{b}_{lab}" for lab in labels for b in ("lower", "upper")]
        assert list(frame.columns) == expected
        assert DEFAULT_LEVELS == (0.5, 0.68, 0.9)
        np.testing.assert_allclose(frame["upper_90"], gaussian.interval(0.9)["upper"])
        assert list(gaussian.to_frame(0.955).columns)[-1] == "upper_95.5"

    def test_fan_chart_frame(self, gaussian: NowcastDistribution) -> None:
        bands = gaussian.fan_chart_frame([0.5, 0.9])
        assert len(bands) == 6
        assert bands["level"].tolist() == [0.9] * 3 + [0.5] * 3
        assert (bands["lower"] <= bands["median"]).all()
        assert (bands["median"] <= bands["upper"]).all()

    def test_plot(self, gaussian: NowcastDistribution) -> None:
        history = pd.Series(
            [0.1, np.nan, 0.3], index=pd.period_range("2019Q2", periods=3, freq="Q")
        )
        ax = gaussian.plot(history=history, levels=[0.5, 0.9], title="t")
        assert len(ax.collections) == 2
        assert len(ax.lines) == 2
        assert ax.get_title() == "t"
        plt.close("all")

    def test_plot_on_axes_without_history(self, gaussian: NowcastDistribution) -> None:
        _, ax = plt.subplots()
        out = gaussian.plot(ax=ax, history=pd.Series(dtype=float))
        assert out is ax
        assert len(gaussian.plot().lines) == 1
        assert out.get_title() == "Nowcast density: gdp"
        plt.close("all")

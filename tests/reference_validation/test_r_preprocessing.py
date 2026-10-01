"""Pre-processing vs the R package ``nowcasting`` 1.1.2 (``Bpanel``, ``month2qtr``, ``qtr2month``).

Fixtures: ``scripts/reference_fixtures/r_preprocessing.R`` (black-box calls). Tolerance of
plan section 10.4 for transformations and aggregations: 1e-10 (relative to
``max(1, |x|)`` because some BRGDP levels are of order 1e10 and are stored with 15
significant digits).

Findings (see ``docs/validation/preprocessing.md``):

* the codes 0-7 coincide exactly on monthly series and on quarterly series with codes 6/7;
* ``Bpanel`` **always** corrects outliers (also with ``NA.replace = FALSE``); the
  detection rule (``|x - median| > 4 IQR``) is identical to
  :func:`nowcastbox.preprocessing.detect_outliers`, the replacement is not (R: centred
  7-term moving average of the median-filled series; nowcastbox: centred 3-term moving
  median of the clean values) - ``reference_divergence``;
* ``Bpanel(aggregate = TRUE)`` applies the **unnormalised** Mariano-Murasawa filter
  (1, 2, 3, 2, 1) *before* the outlier correction; nowcastbox's ``prepare_panel`` divides
  the weights by 3 and filters after the corrections;
* ``Bpanel`` fills the leading values lost to the transformation lags, nowcastbox leaves
  them missing (it never fills outside the observed span).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from nowcastbox.preprocessing import (
    aggregate_panel,
    apply_transforms,
    detect_outliers,
    month_to_quarter,
    prepare_panel,
    quarter_to_month,
    replace_outliers,
)
from tests.reference_validation._bpanel import (
    correct_series,
    emulate_bpanel,
    iqr_flags,
    transform_monthly,
)
from tests.reference_validation._helpers import (
    TOL_TRANSFORM,
    codes,
    fixture_path,
    legend,
    max_abs,
    max_rel,
    read_json,
    read_periods,
    same_missing,
)

pytestmark = pytest.mark.reference_validation


@pytest.fixture(scope="module")
def brgdp_raw() -> pd.DataFrame:
    return read_periods("inputs/brgdp_base.csv.gz")


@pytest.fixture(scope="module")
def brgdp_transformed(brgdp_raw: pd.DataFrame) -> pd.DataFrame:
    return transform_monthly(brgdp_raw, codes("brgdp"))


def _r_corrected(ours: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Apply the R outlier rule to our transformed series (cells R changed)."""
    return pd.DataFrame(
        {c: correct_series(ours[c].to_numpy(), na_replace=False) for c in reference.columns},
        index=ours.index,
    )


# ---------------------------------------------------------------------- codes 0-7
def test_brgdp_codes_match_on_non_outliers(brgdp_transformed: pd.DataFrame) -> None:
    reference = read_periods("r/bpanel_transform_brgdp.csv.gz")
    ours = brgdp_transformed[reference.columns]
    assert same_missing(ours, reference)
    flagged = detect_outliers(ours, 4.0, frequency="M")
    assert max_rel(ours.where(~flagged), reference.where(~flagged)) < TOL_TRANSFORM


def test_every_code_0_to_7(brgdp_raw: pd.DataFrame) -> None:
    leg = pd.read_csv(fixture_path("r/bpanel_transform_codes_legend.csv"))
    reference = read_periods("r/bpanel_transform_codes.csv.gz")
    spec = dict(zip(leg["name"], leg["transformation"].astype(int), strict=True))
    assert sorted(set(spec.values())) == list(range(8))
    ours = apply_transforms(brgdp_raw[list(spec)], spec, frequency="M")
    assert same_missing(ours[reference.columns], reference)
    emulated = _r_corrected(ours, reference)
    assert max_rel(emulated, reference) < TOL_TRANSFORM


def test_nyfed_mixed_frequency_codes() -> None:
    """Monthly codes 0/1/2 and quarterly codes 6 (yoy) / 7 (qoq) stored in the 3rd month."""
    raw = read_periods("inputs/nyfed_base.csv.gz")
    leg = legend("nyfed")
    freq = {n: ("Q" if f == 4 else "M") for n, f in zip(leg["name"], leg["frequency"], strict=True)}
    reference = read_periods("r/bpanel_transform_nyfed.csv.gz")
    ours = apply_transforms(raw, codes("nyfed"), frequency=freq)[reference.columns]
    assert same_missing(ours, reference)
    flagged = detect_outliers(ours, 4.0, frequency=freq)
    assert max_rel(ours.where(~flagged), reference.where(~flagged)) < TOL_TRANSFORM
    # quarterly growth rates: R corrects their outliers (e.g. 2008Q4) on the monthly grid
    emulated = _r_corrected(ours, reference[["GDPC1", "ULCNFB"]])
    assert max_rel(emulated, reference[["GDPC1", "ULCNFB"]]) < TOL_TRANSFORM


@pytest.mark.reference_divergence
def test_quarterly_series_with_monthly_codes() -> None:
    """Codes 1/2 on a quarterly series stored on the monthly grid.

    R differences with a lag of one *month*, which is always missing: the series becomes
    empty and Bpanel drops it. nowcastbox lags by one *native* period (lacuna 5), giving
    the quarter-on-quarter change.
    """
    info = read_json("r/bpanel_quarterly_monthly_codes.json")
    assert "GDPC1" not in info["output_columns"]
    raw = read_periods("inputs/nyfed_base.csv.gz")[["GDPC1"]]
    ours = apply_transforms(raw, {"GDPC1": 1}, frequency={"GDPC1": "Q"})
    expected = raw["GDPC1"].dropna().pct_change().dropna()
    got = ours["GDPC1"].dropna()
    assert len(got) == len(expected) > 100
    assert max_abs(got.to_numpy(), expected.to_numpy()) < 1e-14


# ---------------------------------------------------------------------- outliers
def test_outlier_detection_rule_is_identical(brgdp_transformed: pd.DataFrame) -> None:
    """The cells Bpanel changed are exactly the cells nowcastbox flags."""
    reference = read_periods("r/bpanel_transform_brgdp.csv.gz")
    ours = brgdp_transformed[reference.columns]
    changed = (reference - ours).abs() / np.maximum(1.0, ours.abs()) > 1e-9
    flagged = detect_outliers(ours, 4.0, frequency="M")
    assert int(flagged.to_numpy().sum()) > 100
    assert bool((changed.to_numpy() == flagged.to_numpy()).all())
    emulated = np.column_stack([iqr_flags(ours[c].to_numpy()) for c in ours.columns])
    assert bool((emulated == flagged.to_numpy()).all())


def test_r_outlier_replacement_emulated(brgdp_transformed: pd.DataFrame) -> None:
    """The documented R replacement rule (``?Bpanel``) reproduces R to machine precision."""
    reference = read_periods("r/bpanel_transform_brgdp.csv.gz")
    emulated = _r_corrected(brgdp_transformed, reference)
    assert max_rel(emulated, reference) < 1e-13


@pytest.mark.reference_divergence
def test_outlier_replacement_differs(brgdp_transformed: pd.DataFrame) -> None:
    """Same flags, different replacement values (moving median vs moving average)."""
    reference = read_periods("r/bpanel_transform_brgdp.csv.gz")
    ours_in = brgdp_transformed[reference.columns]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours, mask = replace_outliers(ours_in, 4.0, window=3, frequency="M", return_mask=True)
    scale = ours_in.std()
    diff = ((ours - reference).abs() / scale).where(mask)
    assert float(np.nanmax(diff.to_numpy())) < 5.0  # documented: max 2.43 sd, median 0.41 sd
    assert max_rel(ours.where(~mask), reference.where(~mask)) < TOL_TRANSFORM


# ---------------------------------------------------------------------- Bpanel defaults
def test_default_bpanel_emulation(brgdp_raw: pd.DataFrame) -> None:
    kept = read_json("r/bpanel_default_brgdp_columns.json")["kept"]
    subset = read_periods("r/bpanel_default_brgdp_subset.csv.gz")
    emulated = emulate_bpanel(brgdp_raw, codes("brgdp"), h=0)
    assert list(emulated.columns) == kept
    assert same_missing(emulated[subset.columns], subset)
    assert max_rel(emulated[subset.columns], subset) < TOL_TRANSFORM


def test_drop_rule_matches(brgdp_raw: pd.DataFrame) -> None:
    """Series with more than 1/3 missing values: same set dropped as ``Bpanel``."""
    kept = read_json("r/bpanel_default_brgdp_columns.json")["kept"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, report = prepare_panel(
            brgdp_raw, codes("brgdp"), frequency="M", max_na_prop=1 / 3, return_report=True
        )
    assert sorted(report.dropped) == sorted(set(brgdp_raw.columns) - set(kept))


@pytest.mark.reference_divergence
def test_prepare_panel_vs_bpanel(brgdp_raw: pd.DataFrame) -> None:
    """Observed non-outlying values agree; R also fills the leading lag-lost values."""
    spec = codes("brgdp")
    emulated = emulate_bpanel(brgdp_raw, spec, h=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = prepare_panel(brgdp_raw, spec, frequency="M")
    transformed = transform_monthly(brgdp_raw, spec)
    common = [c for c in emulated.columns if c in ours.columns]
    ours, ref = ours[common], emulated[common].reindex(ours.index)
    flagged = detect_outliers(transformed[common], 4.0, frequency="M").reindex(ours.index)
    plain = transformed[common].reindex(ours.index).notna() & ~flagged
    assert max_rel(ours.where(plain), ref.where(plain)) < TOL_TRANSFORM
    filled_by_r_only = int((ours.isna() & ref.notna()).to_numpy().sum())
    assert filled_by_r_only > 0  # documented: 1197 BRGDP cells
    assert int((ours.notna() & ref.isna()).to_numpy().sum()) == 0


# ---------------------------------------------------------------------- aggregation
def test_mariano_murasawa_aggregation(brgdp_transformed: pd.DataFrame) -> None:
    """Bpanel(aggregate=TRUE): unnormalised (1,2,3,2,1) filter, then outlier correction."""
    reference = read_periods("r/bpanel_aggregate_brgdp_subset.csv.gz")
    filtered = aggregate_panel(
        brgdp_transformed[reference.columns],
        "mariano_murasawa",
        low_frequency="Q",
        normalize=False,
        frequency="M",
    )
    assert same_missing(filtered, reference)
    flagged = detect_outliers(filtered, 4.0, frequency="M")
    assert max_rel(filtered.where(~flagged), reference.where(~flagged)) < TOL_TRANSFORM
    emulated = _r_corrected(filtered, reference)
    assert max_rel(emulated, reference) < TOL_TRANSFORM


@pytest.mark.reference_divergence
def test_normalised_aggregation_is_one_third(brgdp_transformed: pd.DataFrame) -> None:
    """nowcastbox's default (``normalize=True``) is R's filter divided by 3."""
    cols = list(brgdp_transformed.columns[:10])
    norm = aggregate_panel(brgdp_transformed[cols], low_frequency="Q", frequency="M")
    raw = aggregate_panel(
        brgdp_transformed[cols], low_frequency="Q", normalize=False, frequency="M"
    )
    assert max_abs(3.0 * norm, raw) < 1e-12


def test_usgdp_bpanel_subsets() -> None:
    """The emulated USGDP panels (model inputs of the two-step tests) match R's subsets."""
    raw = read_periods("inputs/usgdp_base.csv.gz")
    spec = codes("usgdp")
    x_codes = {k: v for k, v in spec.items() if k != "RGDPGR"}
    agg = emulate_bpanel(raw.drop(columns="RGDPGR"), x_codes, aggregate=True)
    ref = read_periods("r/usgdp_2s_panel_subset.csv.gz")
    assert same_missing(agg[ref.columns], ref)
    assert max_rel(agg[ref.columns], ref) < TOL_TRANSFORM
    plain = emulate_bpanel(raw, spec)
    ref = read_periods("r/usgdp_2s_agg_panel_subset.csv.gz")
    assert same_missing(plain[ref.columns], ref)
    assert max_rel(plain[ref.columns], ref) < TOL_TRANSFORM


# ---------------------------------------------------------------------- frequency conversion
@pytest.mark.parametrize(("ref", "how"), [("1", 1), ("2", 2), ("3", 3), ("mean", "mean")])
def test_month_to_quarter(brgdp_raw: pd.DataFrame, ref: str, how: object) -> None:
    reference = read_periods(f"r/month2qtr_{ref}.csv", "Q")
    ours = month_to_quarter(brgdp_raw[reference.columns], how).reindex(reference.index)
    assert same_missing(ours, reference)
    assert max_rel(ours, reference) < TOL_TRANSFORM


@pytest.mark.parametrize(("ref", "how"), [(1, "start"), (2, "middle"), (3, "end")])
def test_quarter_to_month(ref: int, how: str) -> None:
    gdp = read_periods("inputs/brgdp_gdp.csv", "Q")["V1"]
    reference = read_periods(f"r/qtr2month_{ref}_false.csv")["V1"]
    ours = quarter_to_month(gdp, how).reindex(reference.index)  # type: ignore[arg-type]
    assert same_missing(ours, reference)
    assert max_abs(ours, reference) < TOL_TRANSFORM


def test_quarter_to_month_linear_interpolation() -> None:
    gdp = read_periods("inputs/brgdp_gdp.csv", "Q")["V1"]
    reference = read_periods("r/qtr2month_3_true.csv")["V1"]
    ours = quarter_to_month(gdp, "linear").reindex(reference.index)
    assert same_missing(ours, reference)
    assert max_abs(ours, reference) < TOL_TRANSFORM


@pytest.mark.reference_divergence
@pytest.mark.parametrize("ref", [1, 2])
def test_interpolation_anchored_in_month_1_or_2(ref: int) -> None:
    """R interpolates between values placed in month 1/2; nowcastbox only between quarter ends."""
    gdp = read_periods("inputs/brgdp_gdp.csv", "Q")["V1"]
    reference = read_periods(f"r/qtr2month_{ref}_true.csv")["V1"]
    anchors = quarter_to_month(gdp, "start" if ref == 1 else "middle").reindex(reference.index)
    assert max_abs(anchors, reference) < TOL_TRANSFORM  # the anchors themselves agree
    ours = quarter_to_month(gdp, "linear").reindex(reference.index)
    assert max_abs(ours, reference) > 1.0  # documented: 5.29 (ref 1), 2.65 (ref 2)

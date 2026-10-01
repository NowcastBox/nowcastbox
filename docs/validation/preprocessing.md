# Pre-processing vs `Bpanel`, `month2qtr`, `qtr2month`

Tests: `tests/reference_validation/test_r_preprocessing.py`. Fixtures:
`scripts/reference_fixtures/r_preprocessing.R`.

## Transformations (codes 0-7)

| Panel | Codes | Cells compared | Max rel. error |
|---|---|---|---:|
| BRGDP (99 series x 216 months) | 0-5 | every non-outlying observation | 2.5e-16 |
| 16 BRGDP series, two per code | 0-7 | every observation (through R's outlier rule) | < 1e-10 |
| NYFED (25 series) | monthly 0/1/2, quarterly 6 (yoy) and 7 (qoq) in the 3rd month | every observation | < 1e-10 |

Errors are relative to `max(1, |x|)`, because some BRGDP levels are around 1e10 and the
inputs are stored with 15 significant digits. The missing-value patterns are identical.

**Divergence: monthly codes on a quarterly series.** For codes 1 and 2, R lags by one
*month* on the monthly grid, so the series becomes empty and `Bpanel` drops it.
nowcastbox lags by one *native* period, which gives the quarter-on-quarter change
(lacuna 5).

## Outliers

`Bpanel` **always** corrects outliers, even with `NA.replace = FALSE`. Its detection rule,
|x − median| > 4·IQR with R's type-7 quantiles, is exactly the rule of
`nowcastbox.preprocessing.detect_outliers`: both flag the same 138 BRGDP cells.

The replacement differs (marked `reference_divergence`):

| | R `Bpanel` | nowcastbox `replace_outliers` |
|---|---|---|
| Replacement | outliers (and, with `NA.replace`, missing values) set to the median, then a centred 7-term (k = 3) moving average, edge-padded, ignoring missing values | centred 3-term moving median of the non-outlying values |
| \|difference\| / sd at outlier cells | | max **2.43**, median **0.44** |

The R rule above was taken from `?Bpanel` and pinned down by comparing candidate rules
with the outputs: it reproduces R to 2.5e-16. The tests use it as an emulation
(`tests/reference_validation/_bpanel.py`).

## Missing values and dropped series

- The series dropped for having more than 1/3 missing values are identical (14 in BRGDP,
  3 in USGDP).
- `Bpanel` fills every missing value before the last observation, **including the leading
  values lost to the transformation lags** (1,197 BRGDP cells). `prepare_panel` fills only
  interior gaps, so those cells stay missing. It also drops the leading rows in which
  nothing is observed. The ragged edge is never filled by either library by default.
- On observed, non-outlying cells the outputs of `prepare_panel` and `Bpanel` agree to
  < 1e-10.

## Mariano-Murasawa aggregation

`Bpanel(aggregate = TRUE)` applies the **unnormalised** filter (1, 2, 3, 2, 1) to the
transformed series and corrects outliers *afterwards*.
`aggregate_panel(..., normalize=False)` reproduces the filtered series (< 1e-10). Applying
R's outlier rule on top reproduces `Bpanel` exactly.

By default, nowcastbox normalises the weights by the frequency ratio, ⅓(1, 2, 3, 2, 1), and
`prepare_panel` filters after the corrections. Standardisation removes the factor 3, but
the order of the operations changes the values at the outlier cells.

## Frequency conversion

| R | nowcastbox | Max abs error |
|---|---|---:|
| `month2qtr(x, 1/2/3)` | `month_to_quarter(x, 1/2/3)` | 1.4e-14 |
| `month2qtr(x, "mean")` | `month_to_quarter(x, "mean")` | 4.3e-14 |
| `qtr2month(x, 1/2/3)` | `quarter_to_month(x, "start"/"middle"/"end")` | 0 |
| `qtr2month(x, 3, interpolation = TRUE)` | `quarter_to_month(x, "linear")` | 4.8e-12 |

**Divergence:** `qtr2month(x, 1 or 2, interpolation = TRUE)` interpolates between values
placed in the first or second month. nowcastbox only interpolates between quarter-end
values. The anchor values agree; the interpolated months differ by up to 5.3.

R-side note: `month2qtr(reference_month = "mean")` fails on a multivariate `mts` in
`nowcasting` 1.1.2, so the fixtures call it series by series.

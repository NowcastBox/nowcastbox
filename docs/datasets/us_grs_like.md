# `load_us_grs_like()` — GRS (2008)-like US panel built from FRED-MD

Documented approximation of the Giannone, Reichlin & Small (2008) panel (~200 monthly US series + quarterly GDP, 1982-2004): the FRED-MD series with their FRED-MD transformations and real GDP, restricted by default to 1982-01..2004-12, with stylised release delays.

## At a glance

| | |
|---|---|
| Loader | `nowcastbox.datasets.load_us_grs_like` |
| Default target | `GDPC1` |
| N series | 118 |
| Start | 1959-01 |
| End | 2026-07 |
| Frequencies | M: 117, Q: 1 |
| Vintage | 2026-08 (revised) |
| Built | 2026-10-01 (`scripts/build_datasets/build_us_fred_md.py`) |

## Usage

```python
import nowcastbox as nb
from nowcastbox.datasets import load_us_grs_like

ds = load_us_grs_like()                      # 1982-01 .. 2004-12
res = nb.TwoStepDFM(n_factors=2, factor_lags=2).fit(nb.prepare_panel(ds.data), target="GDPC1")
```

The variable dictionary is that of [`load_us_fred_md`](us_fred_md.md).

## Sources and license

- FRED-MD vintage 2026-08 (revised): https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/2026-rev-08-md.csv
- FRED-QD vintage 2026-08 (revised): https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/quarterly/2026-rev-08-qd.csv

**License / terms of use.** FRED-MD/FRED-QD are distributed freely by the Federal Reserve Bank of St. Louis for research. The shipped series come from US government agencies (BEA, BLS, Census, Federal Reserve Board, Treasury; public domain) or are constructed by McCracken & Ng; series under third-party copyright are excluded (see 'excluded_for_license').

**Excluded for licensing reasons:**

- S&P 500: S&P Dow Jones Indices copyright
- S&P div yield: S&P Dow Jones Indices copyright
- S&P PE ratio: S&P Dow Jones Indices copyright
- AAA: Moody's copyright
- BAA: Moody's copyright
- AAAFFM: Moody's copyright (spread built on AAA)
- BAAFFM: Moody's copyright (spread built on BAA)
- UMCSENTx: University of Michigan Surveys of Consumers copyright
- VIXCLSx: Cboe copyright

## Notes

The original GRS (2008) replication files are not available from a primary source (the authors' former web pages are offline; copies inside GPL software are not used). This dataset is therefore NOT the original panel: it has ~120 instead of ~200 series, current (revised) vintages and FRED-MD definitions. It shares the data file of 'us_fred_md'.

## Citation

McCracken, M. W. & Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic Research. Journal of Business & Economic Statistics, 34(4), 574-589. McCracken, M. W. & Ng, S. (2020). FRED-QD: A Quarterly Database for Macroeconomic Research. NBER Working Paper 26872.

## Files and integrity

| key | file | SHA-256 |
|---|---|---|
| data | `us_fred_md.csv.gz` | `4dc86d0cc140ecca9deddfc42554cac22800d2f1cae2a3a43297c7effe94ae22` |

Digests are verified on first load (`verify=True`); rebuild with `python3 scripts/build_datasets/build_all.py`.

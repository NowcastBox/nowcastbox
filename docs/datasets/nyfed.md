# `load_nyfed()` — NY Fed Staff Nowcast replication panel (US)

The public example panel of the FRBNY Staff Nowcast (Bok, Caratelli, Giannone, Sbordone & Tambalotti, 2018): monthly and quarterly US series (1985 onwards) with the NY Fed block structure (global, soft, real, labor), transformations and categories, data vintage 2017-01-27. Target: real GDP (GDPC1).

## At a glance

| | |
|---|---|
| Loader | `nowcastbox.datasets.load_nyfed` |
| Default target | `GDPC1` |
| N series | 29 |
| Start | 1985-01 |
| End | 2017-01 |
| Frequencies | M: 26, Q: 3 |
| Vintage | 2017-01-27 |
| Built | 2026-10-01 (`scripts/build_datasets/build_nyfed.py`) |

## Usage

```python
import nowcastbox as nb
from nowcastbox.datasets import load_nyfed

ds = load_nyfed()
x = nb.prepare_panel(ds.data)
res = nb.MixedFreqDFM(n_factors=1, factor_lags=1, blocks=ds.blocks).fit(x, target="GDPC1")
```

**Blocks:** `global` (29), `labor` (4), `real` (16), `soft` (2)

**Categories:** `hard` (27), `soft` (2)

## Variable dictionary

| name | description | source | source_code | frequency | transform | delay_days | blocks | category | units |
|---|---|---|---|---|---|---|---|---|---|
| PAYEMS | Payroll Employment | FRBNY Nowcasting repository (data from FRED) | PAYEMS | M | diff | 5 | global;labor | hard | Thousands of Persons |
| JTSJOL | Job Openings | FRBNY Nowcasting repository (data from FRED) | JTSJOL | M | diff | 40 | global;labor | hard | Thousands |
| GDPC1 | Real Gross Domestic Product | FRBNY Nowcasting repository (data from FRED) | GDPC1 | Q | log\|diff(1)\|scale(400) | 28 | global;real | hard | Chained $, Billions |
| CPIAUCSL | Consumer Price Index | FRBNY Nowcasting repository (data from FRED) | CPIAUCSL | M | pct_change(1)\|scale(100) | 14 | global | hard | Index |
| DGORDER | Durable Goods Orders | FRBNY Nowcasting repository (data from FRED) | DGORDER | M | pct_change(1)\|scale(100) | 27 | global;real | hard | $, Millions |
| HSN1F | New Home Sales | FRBNY Nowcasting repository (data from FRED) | HSN1F | M | pct_change(1)\|scale(100) | 25 | global;real | hard | Thousands |
| RSAFS | Retail Sales | FRBNY Nowcasting repository (data from FRED) | RSAFS | M | pct_change(1)\|scale(100) | 14 | global;real | hard | $, Millions |
| UNRATE | Unemployment Rate | FRBNY Nowcasting repository (data from FRED) | UNRATE | M | diff | 5 | global;labor | hard | % |
| HOUST | Housing Starts | FRBNY Nowcasting repository (data from FRED) | HOUST | M | pct_change(1)\|scale(100) | 18 | global;real | hard | Thousands of Units |
| INDPRO | Industrial Production | FRBNY Nowcasting repository (data from FRED) | INDPRO | M | pct_change(1)\|scale(100) | 16 | global;real | hard | Index |
| PPIFIS | Producer Price Index | FRBNY Nowcasting repository (data from FRED) | PPIFIS | M | pct_change(1)\|scale(100) | 13 | global | hard | Index |
| DSPIC96 | Personal Income | FRBNY Nowcasting repository (data from FRED) | DSPIC96 | M | pct_change(1)\|scale(100) | 29 | global;real | hard | Chained $, Billions |
| BOPTEXP | Exports | FRBNY Nowcasting repository (data from FRED) | BOPTEXP | M | pct_change(1)\|scale(100) | 35 | global;real | hard | $, Millions |
| BOPTIMP | Imports | FRBNY Nowcasting repository (data from FRED) | BOPTIMP | M | pct_change(1)\|scale(100) | 35 | global;real | hard | $, Millions |
| WHLSLRIMSA | Wholesale Inventories | FRBNY Nowcasting repository (data from FRED) | WHLSLRIMSA | M | pct_change(1)\|scale(100) | 40 | global;real | hard | $, Millions |
| TTLCONS | Construction Spending | FRBNY Nowcasting repository (data from FRED) | TTLCONS | M | pct_change(1)\|scale(100) | 31 | global;real | hard | $, Millions |
| IR | Import Price Index | FRBNY Nowcasting repository (data from FRED) | IR | M | pct_change(1)\|scale(100) | 14 | global | hard | Index |
| CPILFESL | Core Consumer Price Index | FRBNY Nowcasting repository (data from FRED) | CPILFESL | M | pct_change(1)\|scale(100) | 14 | global | hard | Index |
| PCEPILFE | Core PCE Price Index | FRBNY Nowcasting repository (data from FRED) | PCEPILFE | M | pct_change(1)\|scale(100) | 29 | global | hard | Index |
| PCEPI | PCE Price Index | FRBNY Nowcasting repository (data from FRED) | PCEPI | M | pct_change(1)\|scale(100) | 29 | global | hard | Index |
| PERMIT | Building Permits | FRBNY Nowcasting repository (data from FRED) | PERMIT | M | diff | 18 | global;real | hard | Thousands of Units |
| TCU | Capacity Utilization Rate | FRBNY Nowcasting repository (data from FRED) | TCU | M | diff | 16 | global;real | hard | % |
| BUSINV | Business Inventories | FRBNY Nowcasting repository (data from FRED) | BUSINV | M | pct_change(1)\|scale(100) | 44 | global;real | hard | $, Millions |
| ULCNFB | Unit Labor Cost | FRBNY Nowcasting repository (data from FRED) | ULCNFB | Q | log\|diff(1)\|scale(400) | 35 | global;labor | hard | Index |
| IQ | Export Price Index | FRBNY Nowcasting repository (data from FRED) | IQ | M | pct_change(1)\|scale(100) | 14 | global | hard | Index |
| GACDISA066MSFRBNY | Empire State Mfg Index | FRBNY Nowcasting repository (data from FRED) | GACDISA066MSFRBNY | M | level | 0 | global;soft | soft | Index |
| PCEC96 | Real Consumption Spending | FRBNY Nowcasting repository (data from FRED) | PCEC96 | M | pct_change(1)\|scale(100) | 29 | global;real | hard | Chained $, Billions |
| A261RX1Q020SBEA | Real Gross Domestic Income | FRBNY Nowcasting repository (data from FRED) | A261RX1Q020SBEA | Q | log\|diff(1)\|scale(400) | 58 | global;real | hard | Chained $, Billions |
| GACDFSA066MSFRBPHI | Philadelphia Fed Mfg Index | FRBNY Nowcasting repository (data from FRED) | GACDFSA066MSFRBPHI | M | level | 0 | global;soft | soft | Index |

## Sources and license

- https://github.com/FRBNY-TimeSeriesAnalysis/Nowcasting - Spec_US_example.xls and data/US/2017-01-27.xls (data downloaded by the NY Fed from FRED, Federal Reserve Bank of St. Louis)

**License / terms of use.** Repository files: BSD 3-Clause License, Copyright (c) 2018, Federal Reserve Bank of New York (redistribution with this notice). Underlying series: FRED, from US government agencies and regional Federal Reserve Banks.

> Copyright (c) 2018, Federal Reserve Bank of New York. All rights reserved. Redistribution and use in source and binary forms, with or without modification, are permitted provided that the conditions of the BSD 3-Clause License are met (see the repository LICENSE file).

## Notes

The NY Fed note that these example files do not exactly reproduce the published Staff Nowcast (data redistribution restrictions). 'nyfed_model' = 1 marks the 25 series of the example model (Model column of the specification); load_nyfed(model_only=True) keeps only those. Transformations: lin -> level, chg -> diff, pch -> pct_change(1)|scale(100), pca (compounded annual rate) -> log|diff(1)|scale(400), its first-order approximation; the original code is in 'nyfed_transformation'. Delays are stylised typical US publication lags (not part of the NY Fed files).

## Citation

Bok, B., Caratelli, D., Giannone, D., Sbordone, A. M. & Tambalotti, A. (2018). Macroeconomic Nowcasting and Forecasting with Big Data. Annual Review of Economics, 10, 615-643 (Federal Reserve Bank of New York Staff Report 830).

## Files and integrity

| key | file | SHA-256 |
|---|---|---|
| data | `nyfed.csv.gz` | `b03f91bd761ddc2aa9fa45e22e6c6bbd5f4bef60604e1255ead684ca429d38bb` |

Digests are verified on first load (`verify=True`); rebuild with `python3 scripts/build_datasets/build_all.py`.

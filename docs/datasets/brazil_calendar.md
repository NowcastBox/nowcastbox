# `load_brazil_calendar()` — Release calendar of the Brazilian nowcasting panel

Typical publication delays (days after the end of the reference period) of every series of brazil_nowcast, plus the actual release dates of the IBGE quarterly national accounts (GDP) since 2010Q1.

## At a glance

| | |
|---|---|
| Loader | `nowcastbox.datasets.load_brazil_calendar` |
| N series | 99 |
| Start | 2010Q1 |
| End | 2026Q2 |
| Built | 2026-10-01 (`scripts/build_datasets/build_brazil_calendar.py`) |

## Usage

```python
from nowcastbox.datasets import (
    load_brazil_calendar, load_brazil_gdp_releases, load_brazil_nowcast,
)

cal = load_brazil_calendar()                 # ReleaseCalendar
cal.release_date("pib", "2024Q1")            # Timestamp('2024-06-04')
ds = load_brazil_nowcast()
cal.releases_between("2024-05-01", "2024-06-30", ds.data)  # what was published
load_brazil_calendar(as_frame=True)          # delay table
load_brazil_gdp_releases()                   # GDP release dates and their provenance
```

## Release programmes

| Programme | Typical delay (days) | Rule |
|---|---|---|
| IBGE Contas Nacionais Trimestrais (GDP) | 62 | about 9 weeks after the quarter; exact dates since 2010Q1 in the releases file |
| BCB IBC-Br | 46 | mid-month, about 6-7 weeks after the reference month |
| IBGE PIM-PF (industrial production) | 35 | first week of month m+2 |
| IBGE PMC (retail) | 42 | second week of month m+2 |
| IBGE PMS (services) | 43 | second week of month m+2 |
| IBGE IPCA / INPC | 10 | around the 10th of month m+1 |
| IBGE IPCA-15 | 0 | released around the 25th of the reference month (delay set to 0) |
| IBGE IPP | 30 | end of month m+1 |
| IBGE PNAD Continua (rolling quarter) | 30 | end of month m+1 for the rolling quarter ending in m |
| IPEA monthly PNAD (seasonally adjusted) | 33 | a few days after the PNAD release |
| Novo Caged (Ministry of Labour) | 30 | end of month m+1 |
| BCB monetary and credit statistics | 28 | about four weeks after the reference month (money aggregates: 25) |
| BCB fiscal statistics | 30 | end of month m+1 |
| BCB external sector statistics | 25 | fourth week of month m+1 |
| SECEX trade balance | 5 | first business days of month m+1 |
| Receita Federal tax revenue | 25 | fourth week of month m+1 |
| EPE electricity consumption, ANP oil and fuels | 30 | about one month after the reference month |
| IPEA GFCF indicator / apparent consumption | 95 | GFCF about three months after; apparent consumption about two months (63) |
| Financial market data (Selic, PTAX, reserves, IC-Br) | 1 | monthly averages/end-of-month values known within days (IC-Br: 5; REER: 75) |
| BCB Focus survey (monthly average of weekly medians) | 0 | known at the end of the reference month |

## Sources and license

- IBGE and BCB release calendars 2024-2026 (typical delays)
- IBGE quarterly national accounts publications (GDP release dates)

**License / terms of use.** Calendar information compiled from IBGE/BCB public calendars.

## Notes

Delays are stylised: they ignore weekends, holidays and calendar shifts of individual months. GDP and its components use the explicit release dates for 2010Q1 onwards (date provenance in the releases file) and the 62-day delay before that and for future quarters.

## Citation

Compiled by nowcastbox from IBGE and BCB release calendars.

## Files and integrity

| key | file | SHA-256 |
|---|---|---|
| releases | `brazil_gdp_releases.csv.gz` | `9e143e93a67f38f0f94a5461d20bf530026048be6cd972ae5d0de083984eb9db` |

Digests are verified on first load (`verify=True`); rebuild with `python3 scripts/build_datasets/build_all.py`.

# `load_us_fred_md()` — FRED-MD monthly US macro database + FRED-QD real GDP

FRED-MD (McCracken & Ng, 2016): ~120 monthly US macroeconomic series from 1959 with the official transformation codes, mapped to named nowcastbox transforms (tcode 1 level, 2 diff, 3 diff|diff, 4 log, 5 dlog, 6 log|diff|diff, 7 pct_change|diff), plus quarterly real GDP (GDPC1) from FRED-QD stored in the third month of each quarter. Values in levels as in the vintage file.

## At a glance

| | |
|---|---|
| Loader | `nowcastbox.datasets.load_us_fred_md` |
| Default target | `GDPC1` |
| N series | 118 |
| Start | 1959-01 |
| End | 2026-07 |
| Frequencies | M: 117, Q: 1 |
| Vintage | 2026-08 (revised) |
| Built | 2026-10-01 (`scripts/build_datasets/build_us_fred_md.py`) |

## Usage

```python
from nowcastbox.datasets import load_us_fred_md

ds = load_us_fred_md(start="1985-01")
ds.legend[["transform", "fred_md_tcode", "fred_md_group"]]
```

**Blocks:** `global` (118), `output_income` (17), `consumption_orders` (9), `labor` (31), `housing` (10), `money_credit` (13), `rates_fx` (18), `prices` (20)

**Categories:** `hard` (87), `financial` (31)

## Variable dictionary

| name | description | source | source_code | frequency | transform | delay_days | blocks | category | units |
|---|---|---|---|---|---|---|---|---|---|
| RPI | Real Personal Income | FRED-MD | RPI | M | dlog | 29 | global;output_income | hard | see FRED series notes |
| W875RX1 | Real personal income ex transfer receipts | FRED-MD | W875RX1 | M | dlog | 29 | global;output_income | hard | see FRED series notes |
| DPCERA3M086SBEA | Real personal consumption expenditures | FRED-MD | DPCERA3M086SBEA | M | dlog | 29 | global;consumption_orders | hard | see FRED series notes |
| CMRMTSPLx | Real Manu.  and Trade Industries Sales | FRED-MD | CMRMTSPLx | M | dlog | 45 | global;consumption_orders | hard | see FRED series notes |
| RETAILx | Retail and Food Services Sales | FRED-MD | RETAILx | M | dlog | 14 | global;consumption_orders | hard | see FRED series notes |
| INDPRO | IP Index | FRED-MD | INDPRO | M | dlog | 16 | global;output_income | hard | see FRED series notes |
| IPFPNSS | IP: Final Products and Nonindustrial Supplies | FRED-MD | IPFPNSS | M | dlog | 16 | global;output_income | hard | see FRED series notes |
| IPFINAL | IP: Final Products (Market Group) | FRED-MD | IPFINAL | M | dlog | 16 | global;output_income | hard | see FRED series notes |
| IPCONGD | IP: Consumer Goods | FRED-MD | IPCONGD | M | dlog | 16 | global;output_income | hard | see FRED series notes |
| IPDCONGD | IP: Durable Consumer Goods | FRED-MD | IPDCONGD | M | dlog | 16 | global;output_income | hard | see FRED series notes |
| IPNCONGD | IP: Nondurable Consumer Goods | FRED-MD | IPNCONGD | M | dlog | 16 | global;output_income | hard | see FRED series notes |
| IPBUSEQ | IP: Business Equipment | FRED-MD | IPBUSEQ | M | dlog | 16 | global;output_income | hard | see FRED series notes |
| IPMAT | IP: Materials | FRED-MD | IPMAT | M | dlog | 16 | global;output_income | hard | see FRED series notes |
| IPDMAT | IP: Durable Materials | FRED-MD | IPDMAT | M | dlog | 16 | global;output_income | hard | see FRED series notes |
| IPNMAT | IP: Nondurable Materials | FRED-MD | IPNMAT | M | dlog | 16 | global;output_income | hard | see FRED series notes |
| IPMANSICS | IP: Manufacturing (SIC) | FRED-MD | IPMANSICS | M | dlog | 16 | global;output_income | hard | see FRED series notes |
| IPB51222S | IP: Residential Utilities | FRED-MD | IPB51222S | M | dlog | 16 | global;output_income | hard | see FRED series notes |
| IPFUELS | IP: Fuels | FRED-MD | IPFUELS | M | dlog | 16 | global;output_income | hard | see FRED series notes |
| CUMFNS | Capacity Utilization:  Manufacturing | FRED-MD | CUMFNS | M | diff | 16 | global;output_income | hard | see FRED series notes |
| HWI | Help-Wanted Index for United States | FRED-MD | HWI | M | diff | 40 | global;labor | hard | see FRED series notes |
| HWIURATIO | Ratio of Help Wanted/No.  Unemployed | FRED-MD | HWIURATIO | M | diff | 40 | global;labor | hard | see FRED series notes |
| CLF16OV | Civilian Labor Force | FRED-MD | CLF16OV | M | dlog | 5 | global;labor | hard | see FRED series notes |
| CE16OV | Civilian Employment | FRED-MD | CE16OV | M | dlog | 5 | global;labor | hard | see FRED series notes |
| UNRATE | Civilian Unemployment Rate | FRED-MD | UNRATE | M | diff | 5 | global;labor | hard | see FRED series notes |
| UEMPMEAN | Average Duration of Unemployment (Weeks) | FRED-MD | UEMPMEAN | M | diff | 5 | global;labor | hard | see FRED series notes |
| UEMPLT5 | Civilians Unemployed - Less Than 5 Weeks | FRED-MD | UEMPLT5 | M | dlog | 5 | global;labor | hard | see FRED series notes |
| UEMP5TO14 | Civilians Unemployed for 5-14 Weeks | FRED-MD | UEMP5TO14 | M | dlog | 5 | global;labor | hard | see FRED series notes |
| UEMP15OV | Civilians Unemployed - 15 Weeks & Over | FRED-MD | UEMP15OV | M | dlog | 5 | global;labor | hard | see FRED series notes |
| UEMP15T26 | Civilians Unemployed for 15-26 Weeks | FRED-MD | UEMP15T26 | M | dlog | 5 | global;labor | hard | see FRED series notes |
| UEMP27OV | Civilians Unemployed for 27 Weeks and Over | FRED-MD | UEMP27OV | M | dlog | 5 | global;labor | hard | see FRED series notes |
| CLAIMSx | Initial Claims | FRED-MD | CLAIMSx | M | dlog | 4 | global;labor | hard | see FRED series notes |
| PAYEMS | All Employees:  Total nonfarm | FRED-MD | PAYEMS | M | dlog | 5 | global;labor | hard | see FRED series notes |
| USGOOD | All Employees:  Goods-Producing Industries | FRED-MD | USGOOD | M | dlog | 5 | global;labor | hard | see FRED series notes |
| CES1021000001 | All Employees:  Mining and Logging:  Mining | FRED-MD | CES1021000001 | M | dlog | 5 | global;labor | hard | see FRED series notes |
| USCONS | All Employees:  Construction | FRED-MD | USCONS | M | dlog | 5 | global;labor | hard | see FRED series notes |
| MANEMP | All Employees:  Manufacturing | FRED-MD | MANEMP | M | dlog | 5 | global;labor | hard | see FRED series notes |
| DMANEMP | All Employees:  Durable goods | FRED-MD | DMANEMP | M | dlog | 5 | global;labor | hard | see FRED series notes |
| NDMANEMP | All Employees:  Nondurable goods | FRED-MD | NDMANEMP | M | dlog | 5 | global;labor | hard | see FRED series notes |
| SRVPRD | All Employees:  Service-Providing Industries | FRED-MD | SRVPRD | M | dlog | 5 | global;labor | hard | see FRED series notes |
| USTPU | All Employees:  Trade, Transportation & Utilities | FRED-MD | USTPU | M | dlog | 5 | global;labor | hard | see FRED series notes |
| USWTRADE | All Employees:  Wholesale Trade | FRED-MD | USWTRADE | M | dlog | 5 | global;labor | hard | see FRED series notes |
| USTRADE | All Employees:  Retail Trade | FRED-MD | USTRADE | M | dlog | 5 | global;labor | hard | see FRED series notes |
| USFIRE | All Employees:  Financial Activities | FRED-MD | USFIRE | M | dlog | 5 | global;labor | hard | see FRED series notes |
| USGOVT | All Employees:  Government | FRED-MD | USGOVT | M | dlog | 5 | global;labor | hard | see FRED series notes |
| CES0600000007 | Avg Weekly Hours :  Goods-Producing | FRED-MD | CES0600000007 | M | level | 5 | global;labor | hard | see FRED series notes |
| AWOTMAN | Avg Weekly Overtime Hours :  Manufacturing | FRED-MD | AWOTMAN | M | diff | 5 | global;labor | hard | see FRED series notes |
| AWHMAN | Avg Weekly Hours :  Manufacturing | FRED-MD | AWHMAN | M | level | 5 | global;labor | hard | see FRED series notes |
| HOUST | Housing Starts:  Total New Privately Owned | FRED-MD | HOUST | M | log | 18 | global;housing | hard | see FRED series notes |
| HOUSTNE | Housing Starts, Northeast | FRED-MD | HOUSTNE | M | log | 18 | global;housing | hard | see FRED series notes |
| HOUSTMW | Housing Starts, Midwest | FRED-MD | HOUSTMW | M | log | 18 | global;housing | hard | see FRED series notes |
| HOUSTS | Housing Starts, South | FRED-MD | HOUSTS | M | log | 18 | global;housing | hard | see FRED series notes |
| HOUSTW | Housing Starts, West | FRED-MD | HOUSTW | M | log | 18 | global;housing | hard | see FRED series notes |
| PERMIT | New Private Housing Permits (SAAR) | FRED-MD | PERMIT | M | log | 18 | global;housing | hard | see FRED series notes |
| PERMITNE | New Private Housing Permits, Northeast (SAAR) | FRED-MD | PERMITNE | M | log | 18 | global;housing | hard | see FRED series notes |
| PERMITMW | New Private Housing Permits, Midwest (SAAR) | FRED-MD | PERMITMW | M | log | 18 | global;housing | hard | see FRED series notes |
| PERMITS | New Private Housing Permits, South (SAAR) | FRED-MD | PERMITS | M | log | 18 | global;housing | hard | see FRED series notes |
| PERMITW | New Private Housing Permits, West (SAAR) | FRED-MD | PERMITW | M | log | 18 | global;housing | hard | see FRED series notes |
| ACOGNO | New Orders for Consumer Goods | FRED-MD | ACOGNO | M | dlog | 35 | global;consumption_orders | hard | see FRED series notes |
| AMDMNOx | New Orders for Durable Goods | FRED-MD | AMDMNOx | M | dlog | 35 | global;consumption_orders | hard | see FRED series notes |
| ANDENOx | New Orders for Nondefense Capital Goods | FRED-MD | ANDENOx | M | dlog | 35 | global;consumption_orders | hard | see FRED series notes |
| AMDMUOx | Unfilled Orders for Durable Goods | FRED-MD | AMDMUOx | M | dlog | 35 | global;consumption_orders | hard | see FRED series notes |
| BUSINVx | Total Business Inventories | FRED-MD | BUSINVx | M | dlog | 45 | global;consumption_orders | hard | see FRED series notes |
| ISRATIOx | Total Business:  Inventories to Sales Ratio | FRED-MD | ISRATIOx | M | diff | 45 | global;consumption_orders | hard | see FRED series notes |
| M1SL | M1 Money Stock | FRED-MD | M1SL | M | log\|diff\|diff | 25 | global;money_credit | financial | see FRED series notes |
| M2SL | M2 Money Stock | FRED-MD | M2SL | M | log\|diff\|diff | 25 | global;money_credit | financial | see FRED series notes |
| M2REAL | Real M2 Money Stock | FRED-MD | M2REAL | M | dlog | 25 | global;money_credit | financial | see FRED series notes |
| BOGMBASE | Monetary Base | FRED-MD | BOGMBASE | M | log\|diff\|diff | 25 | global;money_credit | financial | see FRED series notes |
| TOTRESNS | Total Reserves of Depository Institutions | FRED-MD | TOTRESNS | M | log\|diff\|diff | 25 | global;money_credit | financial | see FRED series notes |
| NONBORRES | Reserves Of Depository Institutions | FRED-MD | NONBORRES | M | pct_change\|diff | 25 | global;money_credit | financial | see FRED series notes |
| BUSLOANS | Commercial and Industrial Loans | FRED-MD | BUSLOANS | M | log\|diff\|diff | 10 | global;money_credit | financial | see FRED series notes |
| REALLN | Real Estate Loans at All Commercial Banks | FRED-MD | REALLN | M | log\|diff\|diff | 10 | global;money_credit | financial | see FRED series notes |
| NONREVSL | Total Nonrevolving Credit | FRED-MD | NONREVSL | M | log\|diff\|diff | 38 | global;money_credit | financial | see FRED series notes |
| CONSPI | Nonrevolving consumer credit to Personal Income | FRED-MD | CONSPI | M | diff | 38 | global;money_credit | financial | see FRED series notes |
| FEDFUNDS | Effective Federal Funds Rate | FRED-MD | FEDFUNDS | M | diff | 1 | global;rates_fx | financial | see FRED series notes |
| CP3Mx | 3-Month AA Financial Commercial Paper Rate | FRED-MD | CP3Mx | M | diff | 1 | global;rates_fx | financial | see FRED series notes |
| TB3MS | 3-Month Treasury Bill: | FRED-MD | TB3MS | M | diff | 1 | global;rates_fx | financial | see FRED series notes |
| TB6MS | 6-Month Treasury Bill: | FRED-MD | TB6MS | M | diff | 1 | global;rates_fx | financial | see FRED series notes |
| GS1 | 1-Year Treasury Rate | FRED-MD | GS1 | M | diff | 1 | global;rates_fx | financial | see FRED series notes |
| GS5 | 5-Year Treasury Rate | FRED-MD | GS5 | M | diff | 1 | global;rates_fx | financial | see FRED series notes |
| GS10 | 10-Year Treasury Rate | FRED-MD | GS10 | M | diff | 1 | global;rates_fx | financial | see FRED series notes |
| COMPAPFFx | 3-Month Commercial Paper Minus FEDFUNDS | FRED-MD | COMPAPFFx | M | level | 1 | global;rates_fx | financial | see FRED series notes |
| TB3SMFFM | 3-Month Treasury C Minus FEDFUNDS | FRED-MD | TB3SMFFM | M | level | 1 | global;rates_fx | financial | see FRED series notes |
| TB6SMFFM | 6-Month Treasury C Minus FEDFUNDS | FRED-MD | TB6SMFFM | M | level | 1 | global;rates_fx | financial | see FRED series notes |
| T1YFFM | 1-Year Treasury C Minus FEDFUNDS | FRED-MD | T1YFFM | M | level | 1 | global;rates_fx | financial | see FRED series notes |
| T5YFFM | 5-Year Treasury C Minus FEDFUNDS | FRED-MD | T5YFFM | M | level | 1 | global;rates_fx | financial | see FRED series notes |
| T10YFFM | 10-Year Treasury C Minus FEDFUNDS | FRED-MD | T10YFFM | M | level | 1 | global;rates_fx | financial | see FRED series notes |
| TWEXAFEGSMTHx | Trade Weighted U.S. Dollar Index | FRED-MD | TWEXAFEGSMTHx | M | dlog | 1 | global;rates_fx | financial | see FRED series notes |
| EXSZUSx | Switzerland / U.S. Foreign Exchange Rate | FRED-MD | EXSZUSx | M | dlog | 1 | global;rates_fx | financial | see FRED series notes |
| EXJPUSx | Japan / U.S. Foreign Exchange Rate | FRED-MD | EXJPUSx | M | dlog | 1 | global;rates_fx | financial | see FRED series notes |
| EXUSUKx | U.S. / U.K. Foreign Exchange Rate | FRED-MD | EXUSUKx | M | dlog | 1 | global;rates_fx | financial | see FRED series notes |
| EXCAUSx | Canada / U.S. Foreign Exchange Rate | FRED-MD | EXCAUSx | M | dlog | 1 | global;rates_fx | financial | see FRED series notes |
| WPSFD49207 | PPI: Finished Goods | FRED-MD | WPSFD49207 | M | log\|diff\|diff | 14 | global;prices | hard | see FRED series notes |
| WPSFD49502 | PPI: Finished Consumer Goods | FRED-MD | WPSFD49502 | M | log\|diff\|diff | 14 | global;prices | hard | see FRED series notes |
| WPSID61 | PPI: Intermediate Materials | FRED-MD | WPSID61 | M | log\|diff\|diff | 14 | global;prices | hard | see FRED series notes |
| WPSID62 | PPI: Crude Materials | FRED-MD | WPSID62 | M | log\|diff\|diff | 14 | global;prices | hard | see FRED series notes |
| OILPRICEx | Crude Oil, spliced WTI and Cushing | FRED-MD | OILPRICEx | M | log\|diff\|diff | 1 | global;prices | hard | see FRED series notes |
| PPICMM | PPI: Metals and metal products: | FRED-MD | PPICMM | M | log\|diff\|diff | 14 | global;prices | hard | see FRED series notes |
| CPIAUCSL | CPI : All Items | FRED-MD | CPIAUCSL | M | log\|diff\|diff | 14 | global;prices | hard | see FRED series notes |
| CPIAPPSL | CPI : Apparel | FRED-MD | CPIAPPSL | M | log\|diff\|diff | 14 | global;prices | hard | see FRED series notes |
| CPITRNSL | CPI : Transportation | FRED-MD | CPITRNSL | M | log\|diff\|diff | 14 | global;prices | hard | see FRED series notes |
| CPIMEDSL | CPI : Medical Care | FRED-MD | CPIMEDSL | M | log\|diff\|diff | 14 | global;prices | hard | see FRED series notes |
| CUSR0000SAC | CPI : Commodities | FRED-MD | CUSR0000SAC | M | log\|diff\|diff | 14 | global;prices | hard | see FRED series notes |
| CUSR0000SAD | CPI : Durables | FRED-MD | CUSR0000SAD | M | log\|diff\|diff | 14 | global;prices | hard | see FRED series notes |
| CUSR0000SAS | CPI : Services | FRED-MD | CUSR0000SAS | M | log\|diff\|diff | 14 | global;prices | hard | see FRED series notes |
| CPIULFSL | CPI : All Items Less Food | FRED-MD | CPIULFSL | M | log\|diff\|diff | 14 | global;prices | hard | see FRED series notes |
| CUSR0000SA0L2 | CPI : All items less shelter | FRED-MD | CUSR0000SA0L2 | M | log\|diff\|diff | 14 | global;prices | hard | see FRED series notes |
| CUSR0000SA0L5 | CPI : All items less medical care | FRED-MD | CUSR0000SA0L5 | M | log\|diff\|diff | 14 | global;prices | hard | see FRED series notes |
| PCEPI | Personal Cons.  Expend.:  Chain Index | FRED-MD | PCEPI | M | log\|diff\|diff | 29 | global;prices | hard | see FRED series notes |
| DDURRG3M086SBEA | Personal Cons.  Exp:  Durable goods | FRED-MD | DDURRG3M086SBEA | M | log\|diff\|diff | 29 | global;prices | hard | see FRED series notes |
| DNDGRG3M086SBEA | Personal Cons.  Exp:  Nondurable goods | FRED-MD | DNDGRG3M086SBEA | M | log\|diff\|diff | 29 | global;prices | hard | see FRED series notes |
| DSERRG3M086SBEA | Personal Cons.  Exp:  Services | FRED-MD | DSERRG3M086SBEA | M | log\|diff\|diff | 29 | global;prices | hard | see FRED series notes |
| CES0600000008 | Avg Hourly Earnings :  Goods-Producing | FRED-MD | CES0600000008 | M | log\|diff\|diff | 5 | global;labor | hard | see FRED series notes |
| CES2000000008 | Avg Hourly Earnings :  Construction | FRED-MD | CES2000000008 | M | log\|diff\|diff | 5 | global;labor | hard | see FRED series notes |
| CES3000000008 | Avg Hourly Earnings :  Manufacturing | FRED-MD | CES3000000008 | M | log\|diff\|diff | 5 | global;labor | hard | see FRED series notes |
| DTCOLNVHFNM | Consumer Motor Vehicle Loans Outstanding | FRED-MD | DTCOLNVHFNM | M | log\|diff\|diff | 10 | global;money_credit | financial | see FRED series notes |
| DTCTHFNM | Total Consumer Loans and Leases Outstanding | FRED-MD | DTCTHFNM | M | log\|diff\|diff | 10 | global;money_credit | financial | see FRED series notes |
| INVEST | Securities in Bank Credit at All Commercial Banks | FRED-MD | INVEST | M | log\|diff\|diff | 10 | global;money_credit | financial | see FRED series notes |
| GDPC1 | Real Gross Domestic Product (chained dollars) | FRED-QD | GDPC1 | Q | dlog | 28 | global;output_income | hard | billions of chained dollars, SAAR |

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

Blocks: 'global' plus the FRED-MD group of the series. Delays are stylised typical US publication lags by group/series (days after the end of the month). The 'fred_md_tcode' legend column keeps the original code.

## Citation

McCracken, M. W. & Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic Research. Journal of Business & Economic Statistics, 34(4), 574-589. McCracken, M. W. & Ng, S. (2020). FRED-QD: A Quarterly Database for Macroeconomic Research. NBER Working Paper 26872.

## Files and integrity

| key | file | SHA-256 |
|---|---|---|
| data | `us_fred_md.csv.gz` | `4dc86d0cc140ecca9deddfc42554cac22800d2f1cae2a3a43297c7effe94ae22` |

Digests are verified on first load (`verify=True`); rebuild with `python3 scripts/build_datasets/build_all.py`.

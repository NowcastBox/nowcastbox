# `load_brazil_vintages()` — Real-time vintages of Brazilian quarterly GDP (IBGE)

Seasonally adjusted chained volume indices (1995=100) of GDP, value added by sector and demand components as published in each IBGE quarterly national accounts release since 2010Q1 (one vintage per release), for real-time evaluation and revision analysis.

## At a glance

| | |
|---|---|
| Loader | `nowcastbox.datasets.load_brazil_vintages` |
| Default target | `pib` |
| N series | 10 |
| Start | 1996Q1 |
| End | 2026Q2 |
| N vintages | 65 |
| Frequencies | Q: 10 |
| Built | 2026-10-01 (`scripts/build_datasets/build_brazil_vintages.py`) |

## Usage

```python
from nowcastbox.datasets import load_brazil_vintages

store = load_brazil_vintages()               # VintageStore
store.as_of("2020-08-31")                    # GDP as known on that date
store.nth_release(0)["pib"]                  # first releases
store.revision_summary()                     # Aruoba (2008) statistics
```

## Variable dictionary

| name | description | frequency |
|---|---|---|
| pib_agropecuaria | Gross value added: agriculture - chained quarterly volume index, seasonally adjusted (1995=100) | Q |
| pib_industria | Gross value added: industry - chained quarterly volume index, seasonally adjusted (1995=100) | Q |
| pib_servicos | Gross value added: services - chained quarterly volume index, seasonally adjusted (1995=100) | Q |
| va_pb | Gross value added at basic prices - chained quarterly volume index, seasonally adjusted (1995=100) | Q |
| pib | GDP at market prices - chained quarterly volume index, seasonally adjusted (1995=100) | Q |
| consumo_familias | Household consumption - chained quarterly volume index, seasonally adjusted (1995=100) | Q |
| consumo_governo | Government consumption - chained quarterly volume index, seasonally adjusted (1995=100) | Q |
| fbcf | Gross fixed capital formation - chained quarterly volume index, seasonally adjusted (1995=100) | Q |
| exportacoes_cn | Exports of goods and services - chained quarterly volume index, seasonally adjusted (1995=100) | Q |
| importacoes_cn | Imports of goods and services - chained quarterly volume index, seasonally adjusted (1995=100) | Q |

## Sources and license

- IBGE - Contas Nacionais Trimestrais, Indicadores de Volume e Valores Correntes (one publication per quarter), Tabela 6

**License / terms of use.** IBGE public data; free reuse with attribution of the source (IBGE).

**Releases without a parsed vintage:**

- 2021Q2: last row 2021Q1 != release quarter 2021Q2

## Notes

Values are the seasonally adjusted index levels; growth rates are computed by the user (e.g. QoQ = pct_change). Because IBGE re-estimates the seasonal adjustment each quarter, the whole history changes from release to release. Release dates: see the 'releases' file and its 'date_source' column. Releases 2006Q4-2009Q4 (Word documents with embedded tables) are not included.

## Citation

IBGE - Instituto Brasileiro de Geografia e Estatística. Contas Nacionais Trimestrais: Indicadores de Volume e Valores Correntes, releases 2010Q1 to 2026Q2. Compiled by nowcastbox.

## Files and integrity

| key | file | SHA-256 |
|---|---|---|
| data | `brazil_vintages.csv.gz` | `2bff39f2c46e91fa88822f7d9713c62517dd699d449a76fc1a1511900aa22192` |
| releases | `brazil_gdp_releases.csv.gz` | `9e143e93a67f38f0f94a5461d20bf530026048be6cd972ae5d0de083984eb9db` |

Digests are verified on first load (`verify=True`); rebuild with `python3 scripts/build_datasets/build_all.py`.

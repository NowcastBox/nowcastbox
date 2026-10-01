# Datasets

Built-in datasets are rebuilt from their **primary sources** by the scripts in `scripts/build_datasets/` and shipped as compressed CSV with a metadata YAML (description, sources, license, citation, SHA-256 digests and the series legend). Loaders work offline and verify the digests on first load.

Every `Dataset` holds the panel **in levels** (`ds.data`, a `MixedFrequencyData` with quarterly values in the third month of the quarter) and a legend with the columns `name`, `description`, `source`, `source_code`, `frequency`, `transform` (named `nowcastbox.preprocessing` transformation), `legacy_code` (equivalent code 0-7, if any), `delay_days` (typical publication lag after the end of the period), `blocks`, `category` (`hard`/`soft`/`financial`) and `units`.

| Dataset | Loader | Series | Sample | Target |
|---|---|---|---|---|
| [Release calendar of the Brazilian nowcasting panel](brazil_calendar.md) | `load_brazil_calendar()` | 99 | 2010Q1 – 2026Q2 |  |
| [Brazilian nowcasting panel (BCB, IBGE, IPEA)](brazil_nowcast.md) | `load_brazil_nowcast()` | 99 | 2003-01 – 2026-09 | pib |
| [Real-time vintages of Brazilian quarterly GDP (IBGE)](brazil_vintages.md) | `load_brazil_vintages()` | 10 | 1996Q1 – 2026Q2 | pib |
| [NY Fed Staff Nowcast replication panel (US)](nyfed.md) | `load_nyfed()` | 29 | 1985-01 – 2017-01 | GDPC1 |
| [Simulated mixed-frequency dynamic factor model](simulated_dfm.md) | `load_simulated_dfm()` | 21 | 2000-01 – 2019-12 | gdp |
| [FRED-MD monthly US macro database + FRED-QD real GDP](us_fred_md.md) | `load_us_fred_md()` | 118 | 1959-01 – 2026-07 | GDPC1 |
| [GRS (2008)-like US panel built from FRED-MD](us_grs_like.md) | `load_us_grs_like()` | 118 | 1982-01 – 2004-12 | GDPC1 |

```python
import nowcastbox.datasets as nbd
nbd.list_datasets()
```

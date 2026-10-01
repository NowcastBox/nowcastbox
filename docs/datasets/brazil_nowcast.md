# `load_brazil_nowcast()` — Brazilian nowcasting panel (BCB, IBGE, IPEA)

Monthly and quarterly Brazilian indicators from 2003 to the latest release for nowcasting quarterly GDP: national accounts (target: seasonally adjusted GDP volume index, QoQ growth), IBC-Br, industrial production, retail and services surveys, labour market, external trade, prices, money, credit, fiscal, financial variables and Focus survey expectations. Values are in levels as published; the legend gives the named stationarity transformation.

## At a glance

| | |
|---|---|
| Loader | `nowcastbox.datasets.load_brazil_nowcast` |
| Default target | `pib` |
| N series | 99 |
| Start | 2003-01 |
| End | 2026-09 |
| Frequencies | Q: 10, M: 89 |
| Built | 2026-10-01 (`scripts/build_datasets/build_brazil_nowcast.py`) |

## Usage

```python
import nowcastbox as nb
from nowcastbox.datasets import load_brazil_nowcast, load_brazil_calendar

ds = load_brazil_nowcast()
print(ds.summary())
panel = nb.prepare_panel(ds.data)            # applies the legend transforms
res = nb.MixedFreqDFM(n_factors=1, blocks=ds.blocks).fit(panel, target="pib")

# pseudo real-time vintage with the actual GDP release dates
vintage = nb.pseudo_real_time(ds.data, calendar=load_brazil_calendar(), vintage="2024-05-15")
```

**Blocks:** `global` (99), `real` (57), `nominal` (21), `financial` (19), `soft` (2)

**Categories:** `hard` (73), `financial` (24), `soft` (2)

## Variable dictionary

| name | description | source | source_code | frequency | transform | delay_days | blocks | category | units |
|---|---|---|---|---|---|---|---|---|---|
| pib | GDP at market prices - chained quarterly volume index, seasonally adjusted (nowcasting target; transform gives QoQ SA growth) | IBGE/SIDRA | t1621/v584/c11255=90707 | Q | qoq | 62 | global;real | hard | index (1995=100), seasonally adjusted |
| pib_agropecuaria | Gross value added: agriculture - chained quarterly volume index, seasonally adjusted | IBGE/SIDRA | t1621/v584/c11255=90687 | Q | qoq | 62 | global;real | hard | index (1995=100), seasonally adjusted |
| pib_industria | Gross value added: industry - chained quarterly volume index, seasonally adjusted | IBGE/SIDRA | t1621/v584/c11255=90691 | Q | qoq | 62 | global;real | hard | index (1995=100), seasonally adjusted |
| pib_servicos | Gross value added: services - chained quarterly volume index, seasonally adjusted | IBGE/SIDRA | t1621/v584/c11255=90696 | Q | qoq | 62 | global;real | hard | index (1995=100), seasonally adjusted |
| consumo_familias | Household consumption - chained quarterly volume index, seasonally adjusted | IBGE/SIDRA | t1621/v584/c11255=93404 | Q | qoq | 62 | global;real | hard | index (1995=100), seasonally adjusted |
| consumo_governo | Government consumption - chained quarterly volume index, seasonally adjusted | IBGE/SIDRA | t1621/v584/c11255=93405 | Q | qoq | 62 | global;real | hard | index (1995=100), seasonally adjusted |
| fbcf | Gross fixed capital formation - chained quarterly volume index, seasonally adjusted | IBGE/SIDRA | t1621/v584/c11255=93406 | Q | qoq | 62 | global;real | hard | index (1995=100), seasonally adjusted |
| exportacoes_cn | Exports of goods and services - chained quarterly volume index, seasonally adjusted | IBGE/SIDRA | t1621/v584/c11255=93407 | Q | qoq | 62 | global;real | hard | index (1995=100), seasonally adjusted |
| importacoes_cn | Imports of goods and services - chained quarterly volume index, seasonally adjusted | IBGE/SIDRA | t1621/v584/c11255=93408 | Q | qoq | 62 | global;real | hard | index (1995=100), seasonally adjusted |
| pib_nsa | GDP at market prices - chained quarterly volume index, not seasonally adjusted | IBGE/SIDRA | t1620/v583/c11255=90707 | Q | yoy | 62 | global;real | hard | index (1995=100) |
| ibc_br | IBC-Br, Central Bank economic activity index, seasonally adjusted | BCB/SGS | 24364 | M | dlog | 46 | global;real | hard | index (2002=100), SA |
| ibc_br_agropecuaria | IBC-Br agriculture component, seasonally adjusted | BCB/SGS | 29602 | M | dlog | 46 | global;real | hard | index, SA |
| ibc_br_industria | IBC-Br industry component, seasonally adjusted | BCB/SGS | 29604 | M | dlog | 46 | global;real | hard | index, SA |
| ibc_br_servicos | IBC-Br services component, seasonally adjusted | BCB/SGS | 29606 | M | dlog | 46 | global;real | hard | index, SA |
| pim_geral | Industrial production (PIM-PF) - general industry, seasonally adjusted | IBGE/SIDRA | t8888/v12607/c544=129314 | M | dlog | 35 | global;real | hard | index (2022=100), seasonally adjusted |
| pim_extrativa | Industrial production (PIM-PF) - extractive industries, seasonally adjusted | IBGE/SIDRA | t8888/v12607/c544=129315 | M | dlog | 35 | global;real | hard | index (2022=100), seasonally adjusted |
| pim_transformacao | Industrial production (PIM-PF) - manufacturing, seasonally adjusted | IBGE/SIDRA | t8888/v12607/c544=129316 | M | dlog | 35 | global;real | hard | index (2022=100), seasonally adjusted |
| pim_alimentos | Industrial production (PIM-PF) - food products, seasonally adjusted | IBGE/SIDRA | t8888/v12607/c544=129317 | M | dlog | 35 | global;real | hard | index (2022=100), seasonally adjusted |
| pim_derivados_petroleo | Industrial production (PIM-PF) - coke and petroleum products, seasonally adjusted | IBGE/SIDRA | t8888/v12607/c544=129326 | M | dlog | 35 | global;real | hard | index (2022=100), seasonally adjusted |
| pim_quimicos | Industrial production (PIM-PF) - chemicals, seasonally adjusted | IBGE/SIDRA | t8888/v12607/c544=56689 | M | dlog | 35 | global;real | hard | index (2022=100), seasonally adjusted |
| pim_metalurgia | Industrial production (PIM-PF) - basic metals, seasonally adjusted | IBGE/SIDRA | t8888/v12607/c544=129333 | M | dlog | 35 | global;real | hard | index (2022=100), seasonally adjusted |
| pim_maquinas | Industrial production (PIM-PF) - machinery and equipment, seasonally adjusted | IBGE/SIDRA | t8888/v12607/c544=129337 | M | dlog | 35 | global;real | hard | index (2022=100), seasonally adjusted |
| pim_veiculos | Industrial production (PIM-PF) - motor vehicles, trailers and bodies, seasonally adjusted | IBGE/SIDRA | t8888/v12607/c544=129338 | M | dlog | 35 | global;real | hard | index (2022=100), seasonally adjusted |
| pim_bens_capital | Industrial production (PIM-PF) - capital goods, seasonally adjusted | IBGE/SIDRA | t8887/v12607/c543=129278 | M | dlog | 35 | global;real | hard | index (2022=100), seasonally adjusted |
| pim_bens_intermediarios | Industrial production (PIM-PF) - intermediate goods, seasonally adjusted | IBGE/SIDRA | t8887/v12607/c543=129283 | M | dlog | 35 | global;real | hard | index (2022=100), seasonally adjusted |
| pim_bens_consumo_duraveis | Industrial production (PIM-PF) - durable consumer goods, seasonally adjusted | IBGE/SIDRA | t8887/v12607/c543=129301 | M | dlog | 35 | global;real | hard | index (2022=100), seasonally adjusted |
| pim_bens_consumo_semi_nao_duraveis | Industrial production (PIM-PF) - semi-durable and non-durable consumer goods, seasonally adjusted | IBGE/SIDRA | t8887/v12607/c543=129305 | M | dlog | 35 | global;real | hard | index (2022=100), seasonally adjusted |
| pmc_varejo | Retail trade sales volume (PMC), seasonally adjusted | IBGE/SIDRA | t8880/v7170/c11046=56734 | M | dlog | 42 | global;real | hard | index (2022=100), SA |
| pmc_ampliado | Extended retail trade sales volume (PMC, incl. vehicles and construction materials), seasonally adjusted | IBGE/SIDRA | t8881/v7170/c11046=56736 | M | dlog | 42 | global;real | hard | index (2022=100), SA |
| pmc_supermercados | Retail sales volume: hyper/supermarkets, food, beverages and tobacco, SA | IBGE/SIDRA | t8882/v7170/c11046=56734/c85=90672 | M | dlog | 42 | global;real | hard | index (2022=100), SA |
| pmc_moveis_eletro | Retail sales volume: furniture and household appliances, SA | IBGE/SIDRA | t8882/v7170/c11046=56734/c85=2759 | M | dlog | 42 | global;real | hard | index (2022=100), SA |
| pmc_combustiveis | Retail sales volume: fuels and lubricants, SA | IBGE/SIDRA | t8882/v7170/c11046=56734/c85=90671 | M | dlog | 42 | global;real | hard | index (2022=100), SA |
| pms_volume | Services volume (PMS), seasonally adjusted | IBGE/SIDRA | t5906/v7168/c11046=56726 | M | dlog | 43 | global;real | hard | index (2022=100), SA |
| ipea_fbcf | IPEA monthly indicator of gross fixed capital formation, SA | IPEADATA | GAC12_INDFBCFDESSAZ12 | M | dlog | 95 | global;real | hard | index (1995=100), SA |
| ipea_fbcf_construcao | IPEA monthly GFCF indicator: construction, SA | IPEADATA | GAC12_INDFBCFCCDESSAZ12 | M | dlog | 95 | global;real | hard | index (1995=100), SA |
| ipea_consumo_aparente_bk | IPEA apparent consumption of capital goods (machinery), SA | IPEADATA | GAC12_FBKFCAMIDESSAZ12 | M | dlog | 63 | global;real | hard | index (2012=100), SA |
| ipea_consumo_aparente_industria | IPEA apparent consumption of industrial goods, SA | IPEADATA | GAC12_CAIGDESSAZ12 | M | dlog | 63 | global;real | hard | index (2012=100), SA |
| energia_consumo_total | Electricity consumption, total (EPE/Eletrobras, via IPEADATA) | IPEADATA | ELETRO12_CEET12 | M | dlog_yoy | 30 | global;real | hard | GWh |
| energia_consumo_industria | Electricity consumption, industry (EPE/Eletrobras, via IPEADATA) | IPEADATA | ELETRO12_CEEIND12 | M | dlog_yoy | 30 | global;real | hard | GWh |
| energia_consumo_comercio | Electricity consumption, commerce (EPE/Eletrobras, via IPEADATA) | IPEADATA | ELETRO12_CEECOM12 | M | dlog_yoy | 30 | global;real | hard | GWh |
| energia_consumo_residencial | Electricity consumption, residential (EPE/Eletrobras, via IPEADATA) | IPEADATA | ELETRO12_CEERES12 | M | dlog_yoy | 30 | global;real | hard | GWh |
| anp_diesel | Diesel oil consumption, daily average (ANP, via IPEADATA) | IPEADATA | ANP12_COLDIE12 | M | dlog_yoy | 30 | global;real | hard | barrels/day |
| anp_petroleo_producao | Oil production, daily average (ANP, via IPEADATA) | IPEADATA | ANP12_PDPET12 | M | dlog_yoy | 30 | global;real | hard | barrels/day |
| arrecadacao_federal | Federal tax revenue, gross (Receita Federal, via IPEADATA) | IPEADATA | SRF12_TOTGER12 | M | dlog_yoy | 25 | global;real | hard | R$ (current) |
| desocupacao | Unemployment rate, PNAD Continua (rolling quarter ending in the month) | IBGE/SIDRA | t6381/v4099 | M | diff | 30 | global;real | hard | % |
| desocupacao_sa | Unemployment rate, PNAD Continua, monthly and seasonally adjusted (IPEA) | IPEADATA | PNADC12_TDESOCMD12 | M | diff | 33 | global;real | hard | % |
| ocupacao_sa | Employment-to-population ratio, PNAD Continua, monthly and seasonally adjusted (IPEA) | IPEADATA | PNADC12_NOCUPMD12 | M | diff | 33 | global;real | hard | % |
| massa_salarial_sa | Real usual earnings mass, PNAD Continua, monthly and seasonally adjusted (IPEA) | IPEADATA | PNADC12_MRTHMD12 | M | dlog | 33 | global;real | hard | R$ |
| caged_saldo | Formal job creation, net (Novo Caged, with late declarations; Ministry of Labour) | IPEADATA | CAGED12_SALDONAJU12 | M | level | 30 | global;real | hard | persons |
| exportacoes_bp | Exports of goods, balance of payments (BPM6) | BCB/SGS | 22708 | M | dlog_yoy | 25 | global;real | hard | US$ million |
| importacoes_bp | Imports of goods, balance of payments (BPM6) | BCB/SGS | 22709 | M | dlog_yoy | 25 | global;real | hard | US$ million |
| balanca_comercial | Trade balance (goods), balance of payments (BPM6) | BCB/SGS | 22707 | M | annual_diff | 25 | global;real | hard | US$ million |
| transacoes_correntes | Current account balance (BPM6) | BCB/SGS | 22701 | M | annual_diff | 25 | global;real | hard | US$ million |
| secex_exportacoes | Exports, FOB (SECEX/MDIC, via IPEADATA) | IPEADATA | SECEX12_XVTOT12 | M | dlog_yoy | 5 | global;real | hard | US$ (FOB) |
| secex_importacoes | Imports, FOB (SECEX/MDIC, via IPEADATA) | IPEADATA | SECEX12_MVTOT12 | M | dlog_yoy | 5 | global;real | hard | US$ (FOB) |
| secex_importacoes_bk | Imports of capital goods, FOB (SECEX/MDIC, via IPEADATA) | IPEADATA | SECEX12_MBENCAPGCE12 | M | dlog_yoy | 5 | global;real | hard | US$ (FOB) |
| secex_importacoes_intermediarios | Imports of intermediate goods, FOB (SECEX/MDIC, via IPEADATA) | IPEADATA | SECEX12_MINTGCE12 | M | dlog_yoy | 5 | global;real | hard | US$ (FOB) |
| ipca | IPCA consumer price index | IBGE/SIDRA | t1737/v2266 | M | dlog | 10 | global;nominal | hard | index (Dec 1993=100) |
| inpc | INPC consumer price index | IBGE/SIDRA | t1736/v2289 | M | dlog | 10 | global;nominal | hard | index (Dec 1993=100) |
| ipca15 | IPCA-15 consumer price index (released within the reference month) | IBGE/SIDRA | t3065/v1117 | M | dlog | 0 | global;nominal | hard | index (Dec 1993=100) |
| ipp_industria | Producer price index (IPP), general industry | IBGE/SIDRA | t6903/v10008/c842=46608 | M | dlog | 30 | global;nominal | hard | index (Dec 2018=100) |
| ipca_livres | IPCA market (free) prices, monthly change | BCB/SGS | 11428 | M | level | 10 | global;nominal | hard | % m/m |
| ipca_monitorados | IPCA administered prices, monthly change | BCB/SGS | 4449 | M | level | 10 | global;nominal | hard | % m/m |
| ipca_servicos | IPCA services, monthly change | BCB/SGS | 10844 | M | level | 10 | global;nominal | hard | % m/m |
| ipca_comercializaveis | IPCA tradables, monthly change | BCB/SGS | 4447 | M | level | 10 | global;nominal | hard | % m/m |
| ipca_nao_comercializaveis | IPCA non-tradables, monthly change | BCB/SGS | 4448 | M | level | 10 | global;nominal | hard | % m/m |
| ipca_duraveis | IPCA durable goods, monthly change | BCB/SGS | 10843 | M | level | 10 | global;nominal | hard | % m/m |
| ipca_nucleo_ma | IPCA core: smoothed trimmed means, monthly change | BCB/SGS | 4466 | M | level | 10 | global;nominal | hard | % m/m |
| ipca_nucleo_ex | IPCA core: exclusion (ex administered and food at home), monthly change | BCB/SGS | 11427 | M | level | 10 | global;nominal | hard | % m/m |
| ipca_difusao | IPCA diffusion index | BCB/SGS | 21379 | M | diff | 10 | global;nominal | hard | % |
| base_monetaria | Monetary base (end of period) | BCB/SGS | 1788 | M | dlog_yoy | 25 | global;nominal | financial | R$ thousand |
| m1 | M1 money supply (end of period, new methodology) | BCB/SGS | 27791 | M | dlog_yoy | 25 | global;nominal | financial | R$ thousand |
| m2 | M2 money supply (end of period, new methodology) | BCB/SGS | 27810 | M | dlog_yoy | 25 | global;nominal | financial | R$ thousand |
| m3 | M3 money supply (end of period, new methodology) | BCB/SGS | 27813 | M | dlog_yoy | 25 | global;nominal | financial | R$ thousand |
| m4 | M4 money supply (end of period, new methodology) | BCB/SGS | 27815 | M | dlog_yoy | 25 | global;nominal | financial | R$ thousand |
| dlsp_pib | Net public sector debt, consolidated public sector | BCB/SGS | 4513 | M | diff | 30 | global;nominal | hard | % of GDP |
| dbgg_pib | Gross general government debt (2008 methodology) | BCB/SGS | 13762 | M | diff | 30 | global;nominal | hard | % of GDP |
| resultado_primario_12m | Public sector borrowing requirement, primary result, 12-month sum, consolidated public sector (positive = deficit) | BCB/SGS | 5793 | M | diff | 30 | global;nominal | hard | % of GDP |
| selic | Selic policy rate, monthly accumulated, annualised (252-day basis) | BCB/SGS | 4189 | M | diff | 1 | global;financial | financial | % p.a. |
| cambio_ptax | Exchange rate R$/US$ (PTAX, selling), monthly average | BCB/SGS | 3698 | M | dlog | 1 | global;financial | financial | R$/US$ |
| cambio_efetivo_real | Real effective exchange rate index (IPCA-deflated) | BCB/SGS | 11752 | M | dlog | 75 | global;financial | financial | index (Jun 1994=100) |
| reservas_internacionais | International reserves, liquidity concept, end of month | BCB/SGS | 3546 | M | dlog | 3 | global;financial | financial | US$ million |
| icbr | IC-Br commodity price index (R$), total | BCB/SGS | 27574 | M | dlog | 5 | global;financial | financial | index (Dec 2005=100) |
| icbr_agro | IC-Br commodity price index (R$), agriculture | BCB/SGS | 27575 | M | dlog | 5 | global;financial | financial | index (Dec 2005=100) |
| icbr_metal | IC-Br commodity price index (R$), metals | BCB/SGS | 27576 | M | dlog | 5 | global;financial | financial | index (Dec 2005=100) |
| icbr_energia | IC-Br commodity price index (R$), energy | BCB/SGS | 27577 | M | dlog | 5 | global;financial | financial | index (Dec 2005=100) |
| credito_saldo_total | Credit outstanding, total | BCB/SGS | 20539 | M | dlog | 28 | global;financial | financial | R$ million |
| credito_saldo_pj | Credit outstanding, non-financial corporations | BCB/SGS | 20540 | M | dlog | 28 | global;financial | financial | R$ million |
| credito_saldo_pf | Credit outstanding, households | BCB/SGS | 20541 | M | dlog | 28 | global;financial | financial | R$ million |
| credito_saldo_livres | Credit outstanding, non-earmarked (free) credit | BCB/SGS | 20542 | M | dlog | 28 | global;financial | financial | R$ million |
| credito_concessoes_sa | New credit operations, total, seasonally adjusted | BCB/SGS | 24439 | M | dlog | 28 | global;financial | financial | R$ million, SA |
| credito_concessoes_pf_sa | New credit operations, households, seasonally adjusted | BCB/SGS | 24441 | M | dlog | 28 | global;financial | financial | R$ million, SA |
| credito_concessoes_pj_sa | New credit operations, corporations, seasonally adjusted | BCB/SGS | 24440 | M | dlog | 28 | global;financial | financial | R$ million, SA |
| credito_taxa_juros | Average interest rate on new credit operations, total | BCB/SGS | 20714 | M | diff | 28 | global;financial | financial | % p.a. |
| credito_spread | Average spread on new credit operations, total | BCB/SGS | 20783 | M | diff | 28 | global;financial | financial | p.p. |
| credito_inadimplencia | Non-performing loans (90+ days), total | BCB/SGS | 21082 | M | diff | 28 | global;financial | financial | % |
| investimento_direto_pais | Direct investment liabilities (FDI into Brazil), net, monthly | BCB/SGS | 22885 | M | annual_diff | 25 | global;financial | financial | US$ million |
| focus_pib | Focus survey: median expected real GDP growth, constant 12-month horizon (weighted current/next calendar year), monthly average | BCB/Focus (Olinda) | ExpectativasMercadoAnuais:PIB Total | M | diff | 0 | global;soft | soft | % (annual growth) |
| focus_ipca_12m | Focus survey: median expected IPCA inflation over the next 12 months (smoothed), monthly average | BCB/Focus (Olinda) | ExpectativasMercadoInflacao12Meses:IPCA | M | diff | 0 | global;soft | soft | % (12 months ahead) |

## Sources and license

- Banco Central do Brasil - SGS (dadosabertos.bcb.gov.br), ODbL
- Banco Central do Brasil - Focus market expectations (Olinda API), ODbL
- IBGE - SIDRA (sidra.ibge.gov.br), free use with attribution
- IPEADATA (ipeadata.gov.br), free use citing IPEA and the original source (EPE/Eletrobras, ANP, SECEX/MDIC, Receita Federal, Ministry of Labour)

**License / terms of use.** Data: BCB series under the Open Data Commons Open Database License (ODbL 1.0) as published on dadosabertos.bcb.gov.br; IBGE and IPEADATA public data, free reuse with attribution of the source. Only open-license sources are included.

**Excluded for licensing reasons:**

- FGV confidence indices, IGP-M/IGP-DI/IPA: FGV terms restrict redistribution
- CNI ICEI, industrial survey indicators: third-party (CNI) terms unclear
- ANFAVEA vehicle production/sales (SGS 1373, 1378): third-party (ANFAVEA) terms unclear
- FENABRAVE dealer sales (SGS 7384-7389): third-party (FENABRAVE) terms unclear
- Fecomercio consumer confidence (SGS 4393-4395): third-party (Fecomercio-SP) terms unclear
- ABPO corrugated cardboard, Aco Brasil steel (SGS 7357): third-party associations, terms unclear

## Notes

Delays are stylised typical publication lags (days after the end of the reference month/quarter) from the 2024-2026 release calendars; GDP release dates are available exactly in load_brazil_calendar(). Seasonally adjusted series are used where the source publishes them; for unadjusted series the legend proposes 12-month transformations (dlog_yoy, annual_diff). The seasonal adjustment and the whole history of each series correspond to the latest vintage (no real-time data, see load_brazil_vintages for GDP). Focus constant-horizon expectations weight the current and next calendar year forecasts by (13-m)/12 and (m-1)/12 (Dovern, Fritsche & Slacalek, 2012).

## Citation

Banco Central do Brasil (SGS, Focus); IBGE (SIDRA: Contas Nacionais Trimestrais, PIM-PF, PMC, PMS, IPCA, INPC, IPP, PNAD Continua); IPEA (IPEADATA). Compiled by nowcastbox (scripts/build_datasets/build_brazil_nowcast.py).

## Files and integrity

| key | file | SHA-256 |
|---|---|---|
| data | `brazil_nowcast.csv.gz` | `bf67431b13b44189c5d221274748b4f362deebf5f3cb9948979166dd1fe6b503` |

Digests are verified on first load (`verify=True`); rebuild with `python3 scripts/build_datasets/build_all.py`.

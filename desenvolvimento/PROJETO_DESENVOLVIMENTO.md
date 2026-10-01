# NowcastBox — Projeto de Desenvolvimento

> **Nowcasting com Modelos de Fatores Dinâmicos em Python**
> Biblioteca **original** em Python, **inspirada** no pacote R [`nmecsys/nowcasting`](https://github.com/nmecsys/nowcasting) (de Valk, de Mattos & Ferreira, 2019).
> Não é um *port*: os métodos são reimplementados a partir da literatura, com arquitetura e
> API próprias, e o projeto vai além em escopo, desempenho e metodologia (§3.2).

| Campo | Valor |
|---|---|
| Nome do pacote (PyPI) | `nowcastbox` (nome provisório, segue o padrão `*box` do ecossistema: panelbox, kalmanbox, forecastbox, chronobox) |
| Autores | **Gustavo Haase** (gustavo.haase@gmail.com) · **Alexandre Leão Sanches** (a.leaosanches@gmail.com) |
| Afiliação | Programa de Pós-Graduação em Economia — Universidade Católica de Brasília (UCB). Ambos mestres e doutorandos em Economia. |
| Licença | **MIT** (implementação independente, escrita a partir da literatura — §11) |
| Python | ≥ 3.10 |
| Referência de qualidade | `panelbox` (v1.0.2) |
| Data do documento | 2026-10-01 |
| Status | Planejamento |

---

## Sumário

1. [Motivação e objetivos](#1-motivação-e-objetivos)
2. [Análise do pacote R original](#2-análise-do-pacote-r-original)
3. [Escopo funcional do NowcastBox](#3-escopo-funcional-do-nowcastbox)
4. [Fundamentação metodológica](#4-fundamentação-metodológica)
5. [Arquitetura e estrutura de diretórios](#5-arquitetura-e-estrutura-de-diretórios)
6. [Design da API](#6-design-da-api)
7. [Datasets](#7-datasets)
8. [Documentação](#8-documentação)
9. [Exemplos de aplicação](#9-exemplos-de-aplicação)
10. [Padrões de qualidade, testes e validação](#10-padrões-de-qualidade-testes-e-validação)
11. [Licenciamento e atribuição](#11-licenciamento-e-atribuição)
12. [Roadmap e cronograma](#12-roadmap-e-cronograma)
13. [Paper](#13-paper)
14. [Divisão de trabalho](#14-divisão-de-trabalho)
15. [Riscos e mitigação](#15-riscos-e-mitigação)
16. [Referências](#16-referências)

---

## 1. Motivação e objetivos

### 1.1 Problema

Indicadores macroeconômicos centrais (PIB, em especial) são divulgados com defasagem
(no Brasil, o PIB trimestral do IBGE sai ~60 dias após o fim do trimestre) e em
frequência mais baixa que centenas de indicadores mensais, semanais e diários
(produção industrial, varejo, confiança, crédito, mercado de trabalho, preços).
*Nowcasting* é a estimação do presente, do passado recente e do futuro próximo da
variável-alvo a partir desse fluxo de dados de frequência mista e com bordas
irregulares (*ragged/jagged edges*).

### 1.2 Lacuna no ecossistema Python

| Ferramenta | Linguagem | Limitação |
|---|---|---|
| `nowcasting` (nmecsys/FGV) | R | Sem manutenção desde 2019; dependências legadas (`matlab`, `RMySQL`, `RCurl`); `RTDB` depende de banco MySQL desativado; decomposição de *news* implementada mas **não exportada** |
| `statsmodels.tsa.DynamicFactorMQ` | Python | Apenas EM (Bańbura-Modugno); sem método two-step de Giannone et al. (2008); sem critérios de informação de Bai-Ng; sem construção de vintages; sem datasets de replicação; API de baixo nível para o economista aplicado |
| Códigos de replicação (NY Fed, BCE) | MATLAB | Não empacotados, sem testes, licença proprietária do ambiente |
| `midasml`, `nowcast_lstm` | Python/R | Abordagens distintas (MIDAS/ML), não DFM |

Não existe, em Python, uma biblioteca **completa, validada e de fluxo de ponta a ponta**
(dados → transformações → vintages → seleção de fatores → estimação → nowcast →
decomposição de *news* → avaliação fora da amostra → relatório) para nowcasting com DFM.

### 1.3 Objetivos

1. **Cobrir os métodos clássicos** (two-step de Giannone et al., 2008; EM de frequências
   mistas de Bańbura & Modugno, 2014; critérios de Bai & Ng) com implementação própria,
   escrita a partir dos artigos. O pacote R serve como *benchmark* de resultados
   (§10.4), não como modelo de código ou de API.
2. **Inovar** onde o R e o `statsmodels` param (§3.2): frequências arbitrárias,
   estimação escalável, robustez a outliers (pandemia), nowcasts de densidade,
   explicabilidade via *news*, seleção de variáveis, vintages reais e pipeline de produção.
3. **Avaliação rigorosa em tempo real**: backtesting em tempo pseudo-real e com vintages
   reais, *benchmarks* (bridge, MIDAS, AR, random walk, ML) e testes de comparação de previsões.
4. **Foco no Brasil**: integração com APIs públicas (BCB/SGS, IBGE/SIDRA, IPEADATA),
   calendário de divulgação brasileiro e dataset atualizado para nowcasting do PIB brasileiro.
5. **Qualidade de software nível panelbox**: tipagem, cobertura ≥ 90 %, docstrings
   NumPy ≥ 95 %, documentação MkDocs Material, CI, datasets embutidos, notebooks.
6. **Produção acadêmica**: artigo de software (JSS/JOSS) e artigo aplicado sobre
   nowcasting do PIB brasileiro (§13).

---

## 2. Pacote R de referência (inspiração)

Esta seção mapeia o estado da arte e as lacunas que motivam o projeto. O NowcastBox
**não replica** a estrutura interna, os nomes nem as assinaturas do R; a coluna da
direita indica como o mesmo problema é tratado aqui.

Repositório: `https://github.com/nmecsys/nowcasting` — versão 1.1.2 (2019-06-04), GPL-3,
~3.400 linhas de R. Autores: Daiane Marcolino de Mattos, Pedro Costa Ferreira,
Serge de Valk, Guilherme Branco Gomes (FGV/IBRE).

### 2.1 Funções exportadas

| Função R | Arquivo | Descrição | Abordagem no NowcastBox |
|---|---|---|---|
| `nowcast(formula, data, r, q, p, method, blocks, frequency)` | `nowcast.R` | Estimação DFM: `"2s"`, `"2s_agg"`, `"EM"` | `nb.nowcast()` + classes `TwoStepDFM`, `MixedFreqDFM` |
| `Bpanel(base, trans, NA.replace, aggregate, k.ma, na.prop, h)` | `Bpanel.R` | Transformações estacionárias (códigos 0–7), remoção de séries com > 1/3 NA, correção de outliers e NAs, agregação Mariano-Murasawa | `nb.prepare_panel()` com transformações nomeadas |
| `PRTDB(mts, delay, vintage)` | `PRTDB.R` | Base pseudo tempo-real a partir de defasagens de divulgação (dias) | `nb.pseudo_real_time()` / `Vintage` |
| `ICfactors(x, rmax, type)` | `ICfactors.R` | Critérios de Bai & Ng (2002) IC1/IC2/IC3 para nº de fatores | `nb.select_factors()` |
| `ICshocks(x, r, p, delta, m)` | `ICshocks.R` | Critério de Bai & Ng (2007) para nº de choques dinâmicos | `nb.select_shocks()` |
| `month2qtr(x, reference_month)` | `month2qtr.R` | Mensal → trimestral (mês 1/2/3 ou média) | `nb.utils.month_to_quarter()` |
| `qtr2month(x, reference_month, interpolation)` | `qtr2month.r` | Trimestral → mensal | `nb.utils.quarter_to_month()` |
| `nowcast.plot(out, type)` | `nowcast.plot.R` | Gráficos: `fcst`, `factors`, `eigenvalues`, `eigenvectors` | `results.plot(kind=...)` (Plotly + Matplotlib) |

### 2.2 Funções internas relevantes

| Função R | Arquivo | Papel |
|---|---|---|
| `FactorExtraction`, `pcatodfm` | `method_2s.R` | PCA → VAR(p) dos fatores → filtro/suavizador de Kalman (Giannone et al., 2008; Doz et al., 2011) |
| `bridge` | `method_2s.R` | Regressão ponte de `y` trimestral nos fatores |
| `kalman_filter_diag`, `kalman_smoother_diag` | `method_2s.R` | Kalman com R diagonal e NAs |
| `outliers_correction` | `method_2s.R` | Outliers > 4×IQR substituídos por mediana móvel (k.ma) |
| `EM_DFM_SS_block_idioQARMA_restrMQ`, `InitCond`, `EMstep` | `method_EM.r` | EM com blocos, idiossincráticos AR(1), restrição Mariano-Murasawa para trimestrais (Bańbura & Modugno, 2014) |
| `SKF`, `FIS`, `MissData`, `runKF` | `method_EM.r` | Filtro/suavizador de Kalman com dados faltantes |
| `remNaNs_spline` | `method_EM.r` | Preenchimento inicial de NAs por spline |
| `News_DFM_ML` | `News_DFM_ML.R` | Decomposição de *news* (**não exportada**) |
| `impact` | `impact.R` | Impacto de revisões entre vintages (**não exportada**) |
| `base_extraction`, `get.series.bacen` | `base_extraction.R` | Download de séries do SGS/BCB (HTTP sem TLS) |
| `RTDB` | `RTDB.R` | Vintages reais via MySQL (servidor desativado; **não exportada**) |

### 2.3 Códigos de transformação (`Bpanel`)

| Código | Transformação | Fórmula |
|---|---|---|
| 0 | Nível | $x_t$ |
| 1 | Variação mensal | $(x_t - x_{t-1})/x_{t-1}$ |
| 2 | Diferença mensal | $x_t - x_{t-1}$ |
| 3 | Diferença mensal da variação interanual | $\Delta_1\left[(x_t - x_{t-12})/x_{t-12}\right]$ |
| 4 | Diferença mensal da diferença anual | $\Delta_1 \Delta_{12} x_t$ |
| 5 | Diferença anual | $x_t - x_{t-12}$ |
| 6 | Variação anual | $(x_t - x_{t-12})/x_{t-12}$ |
| 7 | Variação trimestral | $(x_t - x_{t-3})/x_{t-3}$ |

No NowcastBox, transformações são **nomeadas e composáveis** (`"dlog"`, `"yoy"`,
`"mom"`, `Diff(12) | PctChange(1)` …), com inversa definida (para reconstruir níveis
a partir do nowcast) e cientes da frequência da série. Os códigos numéricos acima são
aceitos apenas como atalho, para reproduzir especificações da literatura.

### 2.4 Lacunas identificadas (oportunidades)

- `warnings()` usado no lugar de `warning()` ao validar `r`, `p`, `q` (validação silenciosa).
- Índices de fatores no EM fixos em blocos de `r*5` estados (acoplado à restrição trimestral).
- Mistura de `ts`/`mts`/`xts`; datas implícitas via `start()`/`frequency()`.
- Dependência do pacote `matlab` para emular funções MATLAB.
- `RTDB` inoperante; HTTP sem TLS na API do BCB.
- `p ≤ 5` no EM é limitação de implementação, não teórica.
- Sem testes automatizados no repositório.

---

## 3. Escopo funcional do NowcastBox

### 3.1 Núcleo (v0.1 — métodos clássicos)

- **Pré-processamento**: transformações (códigos 0–7), detecção/correção de outliers,
  tratamento de NAs (fora das bordas irregulares), padronização, agregação Mariano-Murasawa.
- **Frequências mistas**: mensal + trimestral no v0.1; arquitetura já preparada para frequências arbitrárias (§3.2).
- **Vintages**: bases pseudo tempo-real a partir de um calendário de divulgação.
- **Seleção de modelo**: Bai-Ng (2002) IC_p1/IC_p2/IC_p3 e PC_p; Bai-Ng (2007) choques.
- **Estimadores**:
  - `TwoStepDFM` — Giannone, Reichlin & Small (2008) / Doz, Giannone & Reichlin (2011);
    variantes `aggregate="variables"` (`"2s"`) e `aggregate="factors"` (`"2s_agg"`).
  - `MixedFreqDFM` (EM) — Bańbura & Modugno (2014), com blocos, idiossincráticos AR(1)
    e restrições de agregação trimestral.
- **Previsão**: nowcast, backcast e forecast de `y` e preenchimento de `X`.
- **Visualização**: série observada × in-sample × out-of-sample, fatores, autovalores
  (scree), autovetores/cargas.

### 3.2 Inovações

O que diferencia o NowcastBox do pacote R e do `statsmodels.DynamicFactorMQ`:

| # | Inovação | O que entrega | Base na literatura |
|---|---|---|---|
| I1 | **Frequências arbitrárias** | Diária, semanal, mensal, trimestral e anual no mesmo modelo; restrições de agregação genéricas (fluxo × estoque, nível × variação), não só a regra mensal→trimestral fixa | Mariano & Murasawa (2003) generalizado; Bańbura et al. (2011) |
| I2 | **EM escalável** | Tratamento univariado de observações (Kalman sem inversão de matrizes N×N), colapso do vetor de observações, kernels Numba; meta: ≥ 10× mais rápido que `DynamicFactorMQ` com N ≈ 200 | Koopman & Durbin (2000); Jungbacker & Koopman (2015) |
| I3 | **Robustez a outliers e à pandemia** | Idiossincráticos *t*-Student, detecção/tratamento automático de outliers dentro do EM, opções explícitas para 2020–2021 sem descartar amostra | Antolin-Diaz, Drechsel & Petrella (2017) |
| I4 | **Média de longo prazo variante no tempo** | Tendência local no crescimento do PIB (evita viés do nowcast quando o crescimento potencial muda — relevante para o Brasil) | Antolin-Diaz, Drechsel & Petrella (2017) |
| I5 | **Nowcasts de densidade** | Distribuição preditiva (incerteza de filtragem + parâmetros via bootstrap); avaliação por CRPS, log score, PIT e cobertura | Gneiting & Raftery (2007) |
| I6 | **Explicabilidade** | *News* por série, bloco e categoria (*hard*/*soft*/financeiro), separando novas divulgações × revisões; *nowcast tracker* ao longo do trimestre; contribuição de cada série ao **nível** do nowcast | Bańbura & Modugno (2014) |
| I7 | **Seleção de variáveis** | *Targeted predictors*, pré-seleção por elastic net, seleção de blocos validada em tempo pseudo-real | Bai & Ng (2008) |
| I8 | **Vintages reais** | `VintageStore` com API `as_of(data)`; vintages reais do PIB/IBGE e IBC-Br/BCB e ALFRED (EUA) | — |
| I9 | **Diagnósticos de DFM** | Estabilidade das cargas (quebras estruturais), convergência do EM, contribuição de cada fator, alertas de qualidade dos dados | Breitung & Eickmeier (2011) |
| I10 | **Pipeline de produção** | Especificação declarativa em YAML, CLI, snapshots versionados de cada nowcast, relatório HTML automático | — |
| I11 | **Interoperabilidade** | Qualquer estimador compatível com scikit-learn entra no backtest como *benchmark*; exportação de resultados para pandas/Parquet | — |

### 3.3 Módulos de extensão (v0.2–v1.0)

| Módulo | Conteúdo |
|---|---|
| `news` | Decomposição de *news* (Bańbura & Modugno, 2014): impacto de cada divulgação, separação *news* × revisões, pesos por série/bloco, gráfico waterfall |
| `evaluation` | Backtesting em tempo pseudo-real (janela expansiva/rolante), RMSFE por horizonte de nowcast, Diebold-Mariano, Giacomini-White, *Model Confidence Set* |
| `benchmarks` | AR(p), random walk, média histórica, bridge equations, U-MIDAS e MIDAS (Almon/Beta) |
| `calendar` | Calendário de divulgação (defasagens por série); simulação do fluxo diário de dados |
| `data` | Conectores BCB/SGS (HTTPS), IBGE/SIDRA, IPEADATA, FRED (opcional), com cache local |
| `density` | Intervalos de nowcast (incerteza de filtragem + parâmetros via bootstrap) |
| `reports` | Relatório HTML de nowcast (Jinja2 + Plotly), no padrão `panelbox.report` |
| `experiment` | `NowcastExperiment` para comparar especificações (padrão `panelbox.experiment`) |

### 3.4 Fora de escopo (v1.0)

DFM bayesiano completo (Gibbs), volatilidade estocástica, modelos não-lineares e redes
neurais próprias (modelos de ML entram apenas como *benchmarks* via I11). Candidatos a v1.x (§12).

---

## 4. Fundamentação metodológica

### 4.1 Modelo de fatores dinâmicos

$$
x_t = \Lambda f_t + \varepsilon_t, \qquad \varepsilon_t \sim N(0, \Psi)
$$

$$
f_t = \sum_{i=1}^{p} A_i f_{t-i} + B u_t, \qquad u_t \sim N(0, I_q)
$$

com $x_t$ ($n \times 1$) padronizado, $f_t$ ($r \times 1$) fatores comuns, $q \le r$ choques.

### 4.2 Two-step (Giannone et al., 2008; Doz et al., 2011)

1. Painel balanceado → PCA → $\hat\Lambda$, $\hat f_t$.
2. VAR(p) em $\hat f_t$ → $\hat A_i$; decomposição espectral de $\Sigma_u$ → $\hat B$ (posto $q$).
3. Filtro e suavizador de Kalman sobre o painel desbalanceado (bordas irregulares) →
   $\hat f_{t|T}$, com $\Psi$ diagonal.
4. Equação ponte: $y_t^Q = \alpha + \beta' \bar f_t^Q + e_t$, com $\bar f^Q$ obtido por
   agregação dos fatores (`2s_agg`, filtro $\tfrac{1}{9}(1,2,3,2,1)$) ou com as
   variáveis já agregadas (`2s`).

### 4.3 EM com frequências mistas (Bańbura & Modugno, 2014)

Agregação de Mariano & Murasawa (2003) para variação trimestral de uma série em log:

$$
y_t^Q \approx \tfrac{1}{3}\left(y_t + 2y_{t-1} + 3y_{t-2} + 2y_{t-3} + y_{t-4}\right)
$$

impondo restrições lineares nas cargas das séries trimestrais (matriz `Rconstr`).
Estimação por EM (Shumway & Stoffer, 1982; Doz et al., 2012) com dados faltantes de
padrão arbitrário, estrutura de blocos (global, real, nominal, *soft* etc.) e
componentes idiossincráticos AR(1) no estado.

### 4.4 Critérios de informação

- Bai & Ng (2002): $IC_{p,k}(r) = \ln V(r) + r\, g_k(n, T)$, $k = 1, 2, 3$.
- Bai & Ng (2007): nº de choques primitivos $q$ a partir dos autovalores da matriz de
  covariância dos resíduos do VAR dos fatores, com parâmetros $\delta \in (0, 1/2)$ e $m > 0$.

### 4.5 Decomposição de *news*

Para vintages $\Omega_v \subset \Omega_{v+1}$:

$$
\mathbb{E}[y_t \mid \Omega_{v+1}] - \mathbb{E}[y_t \mid \Omega_v]
= \sum_{j \in I_{v+1}} b_{j}\left(x_{j} - \mathbb{E}[x_{j} \mid \Omega_v]\right)
$$

isto é, a revisão do nowcast é uma soma ponderada das surpresas (*news*) de cada
divulgação. Cada página de teoria da documentação (§8) trará a derivação completa.

---

## 5. Arquitetura e estrutura de diretórios

Segue a organização do `panelbox` e do `kalmanbox`:

```
nowcastbox/
├── PROJETO_DESENVOLVIMENTO.md      # este documento
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CITATION.cff
├── LICENSE
├── CLAUDE.md                       # guia para agentes (padrão kalmanbox)
├── pyproject.toml
├── mkdocs.yml
├── codecov.yml
├── .pre-commit-config.yaml
├── .github/
│   ├── workflows/                  # tests.yml, docs.yml, release.yml, r-validation.yml
│   ├── ISSUE_TEMPLATE/
│   └── pull_request_template.md
├── nowcastbox/
│   ├── __init__.py                 # API pública
│   ├── __version__.py
│   ├── _logging.py
│   ├── py.typed
│   ├── core/
│   │   ├── base.py                 # BaseNowcaster (ABC): fit / predict / news
│   │   ├── results.py              # NowcastResults, FactorResults
│   │   ├── frequency.py            # Frequency enum, mapeamento de calendário
│   │   ├── data.py                 # MixedFrequencyData (DataFrame mensal + metadados)
│   │   └── formula.py              # parsing "y ~ ." / "y ~ x1 + x2"
│   ├── preprocessing/
│   │   ├── transforms.py           # códigos 0–7 + transformações nomeadas
│   │   ├── outliers.py             # correção por IQR + mediana móvel
│   │   ├── missing.py              # spline, mediana móvel, máscara de bordas
│   │   ├── aggregation.py          # Mariano-Murasawa, month_to_quarter, quarter_to_month
│   │   └── panel.py                # prepare_panel: transformação, outliers, NAs
│   ├── statespace/
│   │   ├── kalman.py               # filtro com NAs (numba), log-verossimilhança
│   │   ├── smoother.py             # RTS / suavizador de intervalo fixo, lag-1 cov
│   │   └── representation.py       # matrizes T, Z, R, Q, H e restrições
│   ├── models/
│   │   ├── two_step.py             # TwoStepDFM ("2s", "2s_agg")
│   │   ├── em.py                   # MixedFreqDFM (EM, blocos, AR(1) idio)
│   │   ├── bridge.py               # BridgeEquation
│   │   └── _init_conditions.py     # InitCond, PCA inicial
│   ├── selection/
│   │   ├── bai_ng_factors.py       # select_factors (Bai & Ng, 2002)
│   │   └── bai_ng_shocks.py        # select_shocks (Bai & Ng, 2007)
│   ├── vintages/
│   │   ├── pseudo_real_time.py     # pseudo_real_time a partir do calendário
│   │   ├── calendar.py             # ReleaseCalendar
│   │   └── vintage_store.py        # armazenamento/consulta de vintages reais (ALFRED, BCB)
│   ├── news/
│   │   ├── decomposition.py        # news por série/bloco/categoria (I6)
│   │   └── revisions.py            # efeito de revisões de dados
│   ├── benchmarks/                 # AR, RW, U-MIDAS, MIDAS
│   ├── evaluation/
│   │   ├── backtest.py             # PseudoRealTimeBacktest
│   │   ├── metrics.py              # RMSFE, MAE, por horizonte
│   │   └── tests.py                # Diebold-Mariano, Giacomini-White, MCS
│   ├── data_sources/
│   │   ├── bcb.py                  # SGS (HTTPS)
│   │   ├── ibge.py                 # SIDRA
│   │   ├── ipea.py                 # IPEADATA
│   │   ├── fred.py                 # FRED/ALFRED (chave opcional)
│   │   └── cache.py
│   ├── datasets/
│   │   ├── load.py                 # load_us_grs2008(), load_nyfed(), load_brazil_nowcast() ...
│   │   ├── metadata/               # YAML: descrição, fonte, transformação, defasagem, bloco
│   │   └── data/                   # CSV/Parquet comprimidos
│   ├── visualization/              # Plotly (interativo) + Matplotlib (publicação)
│   ├── reports/                    # HTML (Jinja2), padrão panelbox.report
│   ├── experiment/                 # NowcastExperiment
│   ├── pipeline/
│   │   ├── spec.py                 # especificação declarativa (YAML → objetos)
│   │   └── snapshots.py            # nowcasts versionados (I10)
│   └── cli/                        # `nowcastbox run config.yaml`
├── tests/
│   ├── conftest.py
│   ├── unit/ ...                   # espelha nowcastbox/
│   ├── integration/
│   ├── property/                   # hypothesis
│   ├── r_validation/               # fixtures geradas no R + scripts .R
│   ├── matlab_validation/          # opcional: códigos NY Fed/Bańbura-Modugno (Octave)
│   └── benchmarks/                 # pytest-benchmark
├── docs/                           # MkDocs Material (§8)
├── examples/                       # notebooks e scripts (§9)
├── benchmarks/                     # performance Python × R × statsmodels
├── scripts/                        # geração de fixtures R, atualização de datasets
├── desenvolvimento/                # planejamento interno, logs, relatórios de qualidade
└── (papers: repositório privado NowcastBox/nowcastbox-papers, §13)
```

### 5.1 Princípios de design

1. **pandas-first**: dados em `pd.DataFrame` com `PeriodIndex` mensal (`freq="M"`);
   séries trimestrais armazenadas no 3º mês do trimestre (convenção do R/NY Fed),
   com metadados de frequência num objeto `MixedFrequencyData`.
2. **Três camadas de API**: (a) orientada a objetos estilo scikit-learn/statsmodels
   (`Model(...).fit(data) → Results`); (b) função de alto nível `nb.nowcast(...)` para
   uso rápido; (c) pipeline declarativo em YAML + CLI para produção.
3. **Núcleo numérico isolado** (`statespace/`): NumPy + Numba, sem dependência de
   `statsmodels` no caminho crítico; `statsmodels` e `kalmanbox` usados como
   referências de teste.
4. **Resultados imutáveis** com `summary()`, `to_frame()`, `plot()`, `to_html()`, `save()/load()`.
5. **Reprodutibilidade**: `random_state` em toda aleatoriedade; datasets versionados com hash.

### 5.2 Dependências

| Obrigatórias | Opcionais (`extras`) |
|---|---|
| numpy, pandas ≥ 2.0, scipy, numba, statsmodels (VAR, OLS), matplotlib, plotly, jinja2, pyyaml, requests | `[data]`: requests-cache, `[fred]`: fredapi, `[dev]`, `[docs]`, `[test]`, `[r]`: rpy2 (validação) |

---

## 6. Design da API

### 6.1 API orientada a objetos

```python
import nowcastbox as nb

# 1. Dados (replicação Giannone, Reichlin & Small, 2008)
ds = nb.datasets.load_us_grs2008()      # Dataset: .data, .legend, .frequency, .transform
panel = nb.prepare_panel(
    ds.data,
    transform=ds.legend["Transformation"],
    aggregate=False,
)

# 2. Seleção do número de fatores e choques
ic = nb.select_factors(panel.drop(columns="RGDPGR"), rmax=10, criterion="IC2")
q = nb.select_shocks(panel.drop(columns="RGDPGR"), n_factors=ic.r_star, factor_lags=2)

# 3. Estimação
model = nb.TwoStepDFM(n_factors=2, factor_lags=2, n_shocks=2, aggregate="factors")
res = model.fit(panel, target="RGDPGR", frequency=ds.frequency)

# 4. Resultados
print(res.summary())
res.nowcast                 # DataFrame: y, in_sample, out_of_sample
res.factors                 # fatores suavizados
res.loadings, res.A, res.BB, res.Psi
res.bridge.summary()        # regressão ponte (statsmodels OLS)
res.plot("forecast"); res.plot("factors"); res.plot("eigenvalues"); res.plot("loadings")
```

```python
# EM com blocos — replicação NY Fed Staff Nowcast
ds = nb.datasets.load_nyfed()
x = nb.prepare_panel(ds.data, transform=ds.legend["Transformation"],
                      replace_na=False, max_na_prop=1.0)
model = nb.MixedFreqDFM(n_factors=1, factor_lags=1, blocks=ds.blocks, max_iter=500, tol=1e-4)
res = model.fit(x, target="GDPC1", frequency=ds.legend["Frequency"])
```

```python
# Vintages, news e backtest — PIB Brasil
ds = nb.datasets.load_brazil_nowcast()
v_old = nb.pseudo_real_time(ds.data, delay=ds.delay, vintage="2015-05-01")
v_new = nb.pseudo_real_time(ds.data, delay=ds.delay, vintage="2015-06-01")

news = res.news(old=v_old, new=v_new, target_period="2015Q2")
news.summary(); news.plot("waterfall")

bt = nb.evaluation.PseudoRealTimeBacktest(
    model=nb.MixedFreqDFM(n_factors=1, factor_lags=1, blocks=ds.blocks, idiosyncratic="student_t"),
    data=ds.data, delay=ds.delay, target="PIB",
    start="2012-01-01", end="2019-12-01", step="M",
    benchmarks=[nb.benchmarks.AR(p=1), nb.benchmarks.UMIDAS()],
)
out = bt.run(n_jobs=-1)
out.rmsfe_by_horizon(); out.diebold_mariano(reference="AR")
```

### 6.2 API de alto nível e pipeline declarativo

```python
# Uma linha: escolhe r por Bai-Ng, estima, devolve nowcast com intervalo
res = nb.nowcast(data, target="PIB", method="em", blocks="auto", density=True)
res.nowcast.tail()          # mediana, intervalos 68 % / 90 %
```

```yaml
# nowcast_pib.yaml  →  `nowcastbox run nowcast_pib.yaml`
target: PIB
data:
  source: brazil_nowcast        # dataset embutido ou conectores BCB/IBGE/IPEA
  vintage: today
model:
  type: MixedFreqDFM
  factors: {global: 1, real: 1, soft: 1}
  idiosyncratic: student_t
  long_run_mean: time_varying
outputs: [nowcast, news, density, report_html]
snapshot_dir: ./snapshots
```

### 6.3 Convenções

- Parâmetros em `snake_case`, nomes descritivos (`n_factors`, `factor_lags`, `blocks`), sem herdar a nomenclatura do R.
- Docstrings NumPy com seções *Parameters, Returns, Raises, See Also, Notes,
  References, Examples* (exemplos executados via doctest).
- Erros explícitos (`ValueError`, `NowcastDataError`) — nunca falha silenciosa.
- Warnings próprios (`ConvergenceWarning`, `DataQualityWarning`).

---

## 7. Datasets

Padrão panelbox: `nowcastbox/datasets/data/` + metadados YAML + loaders tipados
retornando um objeto `Dataset` (dados, legenda, frequência, transformação, defasagem,
blocos, fonte, licença, citação).

### 7.1 Datasets de replicação (reconstruídos das fontes primárias)

| Loader | Fonte primária | Conteúdo | Uso |
|---|---|---|---|
| `load_us_grs2008()` | Arquivos de replicação de Giannone et al. (2008) | 193 séries mensais + PIB EUA, 312 obs. | Replicar Giannone et al. (2008) |
| `load_us_bgr2011()` | Arquivos de replicação de Bańbura et al. (2011) | 26 séries, 358 obs. | Replicar Bańbura et al. (2011) |
| `load_nyfed()` | Código/dados públicos do NY Fed Staff Nowcast (FRBNY) | 25 séries (mensal + trimestral), 385 obs., blocos | Replicar NY Fed Staff Nowcast (Bok et al., 2018) |
| `load_brazil_2000_2019()` | BCB/SGS, IBGE/SIDRA | ≈ 100 séries brasileiras 2000–2019, transformações, defasagens, PIB | Vintages pseudo tempo-real (Brasil) |

Construção: scripts em `scripts/build_datasets/` baixam e montam cada dataset a partir
da fonte primária → Parquet/CSV comprimido, hash SHA-256 registrado no YAML. Os `.rda`
do pacote R são usados apenas para conferência cruzada dos valores, não como fonte.

### 7.2 Novos datasets

| Loader | Conteúdo | Fonte |
|---|---|---|
| `load_brazil_nowcast()` | Painel atualizado (≈ 80–120 séries, 2003–2026) para nowcast do PIB brasileiro: IBC-Br, PIM-PF, PMC, PMS, CAGED/PNAD, confiança (fontes de licença aberta), crédito, fiscal, comércio exterior, preços, financeiros | BCB/SGS, IBGE/SIDRA, IPEADATA |
| `load_brazil_calendar()` | Defasagens típicas de divulgação por série | Calendários IBGE/BCB |
| `load_brazil_vintages()` | Vintages **reais** do PIB e do IBC-Br (revisões) | Séries históricas de revisões do IBGE/BCB |
| `load_us_fred_md()` | FRED-MD (McCracken & Ng, 2016) com códigos de transformação | FRED-MD |
| `load_simulated_dfm()` | DFM simulado com parâmetros conhecidos | Gerador `nb.simulate.dfm()` (para testes e tutoriais) |

Cada dataset terá página na documentação (`docs/datasets/*.md`) com dicionário de
variáveis, transformações, defasagens, blocos e citação.

---

## 8. Documentação

MkDocs Material + mkdocstrings (mesma configuração do panelbox: abas, MathJax,
copy-code, tema claro/escuro), publicada no Read the Docs / GitHub Pages.

```
docs/
├── index.md
├── getting-started/     installation · quickstart · core-concepts · choosing-a-method
├── user-guide/
│   ├── data/            mixed-frequency-data · transformations · outliers-missing · aggregation
│   ├── vintages/        pseudo-real-time · release-calendar · real-vintages
│   ├── models/          two-step-dfm · em-dfm · blocks · bridge · benchmarks
│   ├── selection/       number-of-factors · number-of-shocks
│   ├── news/            news-decomposition · revisions
│   ├── evaluation/      backtesting · metrics · forecast-comparison-tests
│   ├── data-sources/    bcb · ibge · ipea · fred
│   └── visualization/   plots · reports
├── theory/              dfm · kalman-filter-smoother · em-algorithm · mariano-murasawa ·
│                        bai-ng · news · forecast-evaluation · references
├── tutorials/           (notebooks renderizados — §9)
├── datasets/            uma página por dataset
├── comparison/          r-nowcasting.md · statsmodels-dynamicfactormq.md (conceitos equivalentes e diferenças)
├── validation/          benchmarks contra R, statsmodels e MATLAB (tabelas de erro e desempenho)
├── api/                 referência automática (mkdocstrings)
├── faq/
└── contributing/
```

Metas: 100 % da API pública documentada; toda página de teoria com notação
consistente com §4; toda página de user-guide com exemplo executável.

---

## 9. Exemplos de aplicação

`examples/` no padrão panelbox (notebooks + `solutions/` + `utils/` + `outputs/`),
todos executados no CI (`nbmake`) com datasets embutidos (sem rede).

| # | Notebook | Conteúdo |
|---|---|---|
| 01 | `01_quickstart.ipynb` | Primeiro nowcast em 10 linhas |
| 02 | `02_data_preparation.ipynb` | Transformações, outliers, bordas irregulares, `prepare_panel` |
| 03 | `03_replicate_giannone_2008.ipynb` | Replicação GRS (2008) com `USGDP`, ACF dos resíduos 1985–2004 |
| 04 | `04_selecting_factors_shocks.ipynb` | Bai-Ng (2002, 2007), sensibilidade a `rmax`, `delta`, `m` |
| 05 | `05_em_nyfed_replication.ipynb` | Replicação NY Fed com blocos (`NYFED`) |
| 06 | `06_brazil_gdp_vintages.ipynb` | `BRGDP`: vintages pseudo tempo-real e nowcast do PIB |
| 07 | `07_news_decomposition.ipynb` | Decomposição de *news* entre vintages, waterfall por bloco |
| 08 | `08_pseudo_real_time_evaluation.ipynb` | Backtest 2012–2025, RMSFE por horizonte, DM vs. AR/MIDAS |
| 09 | `09_brazil_live_nowcast.ipynb` | Pipeline completo com APIs BCB/IBGE (marcado `network`) |
| 10 | `10_covid_robustness.ipynb` | Efeito da pandemia: outliers, janelas de estimação, dummies |
| 11 | `11_production_pipeline.ipynb` | Pipeline YAML, CLI, snapshots e nowcast tracker (I6, I10) |
| 12 | `12_reports_and_experiments.ipynb` | `NowcastExperiment` e relatório HTML |
| 13 | `13_density_nowcasts.ipynb` | Nowcasts de densidade, CRPS e PIT (I5) |
| 14 | `14_weekly_daily_data.ipynb` | Dados semanais/diários no mesmo modelo (I1) |
| 15 | `15_comparison_r_statsmodels.ipynb` | Resultados e tempo de execução vs. `nowcasting` (R) e `DynamicFactorMQ` |

Além disso: `examples/scripts/` (versões `.py`), `examples/R/` (scripts R equivalentes
para comparação) e `examples/cheatsheets/` (cheat sheet PDF de 1 página).

---

## 10. Padrões de qualidade, testes e validação

### 10.1 Ferramentas (mesmas do panelbox/kalmanbox)

| Aspecto | Ferramenta | Meta |
|---|---|---|
| Lint + format | ruff (line-length 100, convenção NumPy) | 0 erros |
| Tipagem | pyright (`strict` em `core/`, `statespace/`; `basic` no restante) | 0 erros |
| Cobertura | pytest-cov (branch) | **≥ 90 %** (núcleo numérico ≥ 95 %) |
| Docstrings | interrogate | **≥ 95 %** |
| Complexidade | radon / ruff mccabe | ≤ 10 por função (≤ 15 justificado) |
| Segurança | bandit, gitleaks | 0 achados ≥ médio |
| Mutação | mutmut (núcleo `statespace/`, `models/`) | score ≥ 80 % |
| Propriedades | hypothesis | invariantes do Kalman, transformações inversíveis |
| Performance | pytest-benchmark | sem regressão > 10 % |
| Notebooks | nbmake | todos executam |
| Pre-commit | ruff, pyright, yaml/toml, whitespace, large files | obrigatório |

### 10.2 CI (GitHub Actions)

- `tests.yml`: matriz Python 3.10–3.13 × Ubuntu/macOS/Windows; cobertura → Codecov.
- `reference-validation.yml`: instala R (`nowcasting`) e Octave (código NY Fed),
  regenera fixtures e compara com Python (agendado semanalmente).
- `docs.yml`: build MkDocs com `--strict`.
- `release.yml`: build + publicação PyPI via *trusted publishing*; DOI Zenodo.

### 10.3 Estratégia de testes

1. **Unitários** por módulo (transformações, agregação, filtro de Kalman, passos do EM).
2. **Analíticos**: Kalman vs. solução fechada em modelos pequenos; EM em DFM
   simulado recupera parâmetros (até rotação) com $T$ grande.
3. **Propriedades**: log-verossimilhança do EM monotonicamente não-decrescente;
   suavizador ≡ filtro no último período; `quarter_to_month ∘ month_to_quarter` = id.
4. **Validação cruzada**: filtro/suavizador vs. `statsmodels` e `kalmanbox`;
   EM vs. `statsmodels.DynamicFactorMQ` em especificações equivalentes.
5. **Benchmark contra implementações de referência** (§10.4).
6. **Integração**: notebooks de replicação reproduzem números publicados
   (GRS 2008; de Valk et al. 2019).

### 10.4 Benchmark contra implementações de referência

Para os **métodos clássicos** (mesma especificação estatística), os resultados devem
coincidir com as implementações de referência — R `nowcasting`, `statsmodels.DynamicFactorMQ`
e código MATLAB do NY Fed (via Octave). Pasta `tests/reference_validation/` (padrão
`panelbox/tests/r_validation`): scripts externos geram fixtures, Python compara.

| Componente | Tolerância (máx. erro absoluto) |
|---|---|
| Transformações e agregações | 1e-10 |
| Critérios de Bai-Ng | `r*`/`q*` exatos; IC 1e-8 |
| Two-step: fatores, cargas, VAR, previsões | 1e-6 (até sinal/rotação dos fatores) |
| EM: log-verossimilhança, previsões de `y` | 1e-4 |
| *News*: impactos e pesos | 1e-5 |

Onde o NowcastBox faz escolhas melhores (inicialização, critério de convergência,
tratamento de outliers), a diferença é documentada em `docs/validation/` e marcada com
`@pytest.mark.reference_divergence`. As inovações (§3.2) são validadas por simulação
(recuperação de parâmetros conhecidos) e por avaliação fora da amostra.

---

## 11. Licenciamento e atribuição

**Licença: MIT** (decidido), coerente com o panelbox.

Como o pacote R é GPL-3, o NowcastBox é uma **implementação independente**:

- O código é escrito a partir dos **artigos** e da formulação de §4, nunca traduzido do R.
- O pacote R entra apenas como *benchmark* externo de resultados (§10.4), executado
  fora da biblioteca para gerar fixtures; nenhum código R é distribuído.
- Os datasets são reconstruídos das fontes primárias (§7.1).
- `desenvolvimento/` registra, por módulo, as referências usadas na implementação.

Atribuição: o pacote R e o artigo do R Journal (de Valk et al., 2019) são citados como
trabalho relacionado no README, na documentação e nos papers, junto com
`statsmodels.DynamicFactorMQ` e o código do NY Fed.

Dados: verificar a licença de cada fonte primária (BCB/IBGE/IPEA: dados públicos);
séries com restrição de redistribuição ficam fora dos datasets embutidos e são
acessadas apenas via conector.

---

## 12. Roadmap e cronograma

Início estimado: outubro/2026. Estimativas para dois desenvolvedores em tempo parcial.

| Fase | Versão | Período | Entregas |
|---|---|---|---|
| 0. Fundação | — | out/2026 (3 sem.) | Repositório, `pyproject.toml` (MIT), CI, pre-commit, MkDocs, scripts de construção dos datasets a partir das fontes primárias, scripts de fixtures de referência |
| 1. Dados | 0.1.0a | nov/2026 | `MixedFrequencyData` com frequências arbitrárias (I1), transformações nomeadas e inversíveis, outliers, agregações genéricas, vintages pseudo tempo-real |
| 2. Núcleo de estado | 0.1.0b | nov–dez/2026 | Kalman univariado com NAs (Numba, I2), suavizador, log-verossimilhança; validação vs. statsmodels/kalmanbox |
| 3. Two-step | 0.1.0 | dez/2026–jan/2027 | `TwoStepDFM`, bridge, Bai-Ng, plots; notebooks 01–04; **release PyPI** |
| 4. EM | 0.2.0 | fev–mar/2027 | `MixedFreqDFM` com blocos, colapso do estado (I2), benchmark de desempenho; notebooks 05–06 |
| 5. News & avaliação | 0.3.0 | abr–mai/2027 | `news/` com tracker e contribuições (I6), `evaluation/`, `benchmarks/` + interface scikit-learn (I11), densidade (I5); notebooks 07–08, 13 |
| 6. Robustez | 0.4.0 | jun–jul/2027 | *t*-Student e outliers no EM (I3), média de longo prazo variante (I4), diagnósticos (I9), seleção de variáveis (I7); notebook 10 |
| 7. Brasil | 0.5.0 | ago–set/2027 | `data_sources/` (BCB, IBGE, IPEA), `load_brazil_nowcast`, calendário, `VintageStore` com vintages reais (I8); notebooks 09, 14 |
| 8. Produção | 0.6.0 | out/2027 | Pipeline YAML, CLI, snapshots (I10), relatórios HTML, `NowcastExperiment`; notebooks 11–12, 15 |
| 9. Estabilização | 1.0.0 | nov–dez/2027 | Auditoria de qualidade (padrão `panelbox/desenvolvimento/qualidade`), cobertura ≥ 90 %, docs completas, DOI Zenodo, submissão do paper de software |

Pós-1.0 (ideias): DFM bayesiano, volatilidade estocástica, nowcast regional (UF),
dados alternativos (texto, Google Trends, transações), integração com `forecastbox`
e `kalmanbox`.

---

## 13. Paper

Repositório privado `NowcastBox/nowcastbox-papers` (ver o README de lá). Dois artigos planejados:

1. **Artigo de software** — *"NowcastBox: Nowcasting with Dynamic Factor Models in Python"*,
   Gustavo Haase & Alexandre Leão Sanches (UCB).
   Alvos: *Journal of Statistical Software* (template JSS, como o panelbox) ou
   *Journal of Open Source Software* (rápido, complementar). Conteúdo: motivação,
   metodologia, **inovações (§3.2)**, arquitetura, validação contra R/MATLAB/statsmodels,
   replicações (GRS 2008, NY Fed), desempenho, aplicação ao PIB do Brasil.
2. **Artigo aplicado** — nowcasting do PIB brasileiro em tempo (pseudo-)real
   2012–2026: contribuição relativa de dados *hard* × *soft*, efeito da pandemia com
   DFM robusto (I3), média de longo prazo variante (I4), nowcasts de densidade (I5),
   comparação DFM × MIDAS × bridge × ML × Focus/BCB.
   Alvos: *Revista Brasileira de Economia*, *Brazilian Review of Econometrics*,
   *Economia Aplicada*, *International Journal of Forecasting*. Possível vínculo
   com as teses de doutorado dos autores na UCB.

Também: apresentação no Encontro da ANPEC / SBE e SciPy (como o panelbox).

---

## 14. Divisão de trabalho

Proposta inicial (a ajustar entre os autores):

| Área | Responsável principal | Revisão |
|---|---|---|
| Arquitetura, infraestrutura, CI, qualidade, docs | Gustavo Haase | Alexandre Leão Sanches |
| Núcleo numérico (Kalman, EM, two-step) | Gustavo Haase | Alexandre Leão Sanches |
| Metodologia econométrica, teoria (§4, docs/theory) | Alexandre Leão Sanches | Gustavo Haase |
| Datasets brasileiros, calendário, fontes de dados | Alexandre Leão Sanches | Gustavo Haase |
| News, avaliação, benchmarks | Conjunto | — |
| Paper de software | Gustavo Haase (1º autor) | Alexandre Leão Sanches |
| Paper aplicado | A definir | — |

Fluxo: branches por feature, PR com revisão obrigatória do coautor, CHANGELOG
(*Keep a Changelog*), versionamento semântico, commits convencionais.

---

## 15. Riscos e mitigação

| Risco | Impacto | Mitigação |
|---|---|---|
| Divergência numérica com referências no EM (inicialização, critério de parada) | Médio | Testes com inicialização idêntica para isolar diferenças; documentar divergências intencionais |
| Contaminação de código GPL | Alto | Implementação a partir dos artigos; R usado só como caixa-preta (§11); revisão cruzada entre autores |
| Inovações aumentarem demais o escopo | Alto | Inovações priorizadas por fase (§12); cada uma atrás de opção explícita, sem quebrar o núcleo |
| APIs públicas instáveis (BCB/IBGE) | Médio | Cache local, retries, testes com *mocks*; datasets embutidos para tudo que roda no CI |
| Desempenho do EM com N grande | Médio | Kalman univariado e colapso do estado desde o v0.2 (I2); benchmarks no CI |
| Sobreposição com `statsmodels.DynamicFactorMQ` | Médio | Diferenciação pelas inovações I1–I11, two-step, Bai-Ng, avaliação em tempo real e foco Brasil |
| Tempo dos autores (doutorado) | Médio | Escopo por versões; v0.1 já útil e publicável |
| Restrição de licença de algumas séries | Baixo | Acesso só via conector; datasets embutidos apenas com dados abertos |

---

## 16. Referências

- Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor models. *Econometrica*, 70(1), 191–221.
- Antolin-Diaz, J., Drechsel, T., & Petrella, I. (2017). Tracking the slowdown in long-run GDP growth. *Review of Economics and Statistics*, 99(2), 343–356.
- Bai, J., & Ng, S. (2007). Determining the number of primitive shocks in factor models. *Journal of Business & Economic Statistics*, 25(1), 52–60.
- Bai, J., & Ng, S. (2008). Forecasting economic time series using targeted predictors. *Journal of Econometrics*, 146(2), 304–317.
- Bańbura, M., Giannone, D., & Reichlin, L. (2011). Nowcasting. In M. P. Clements & D. F. Hendry (Eds.), *Oxford Handbook of Economic Forecasting* (pp. 193–224). Oxford University Press.
- Bańbura, M., & Modugno, M. (2014). Maximum likelihood estimation of factor models on datasets with arbitrary pattern of missing data. *Journal of Applied Econometrics*, 29(1), 133–160.
- Bańbura, M., & Rünstler, G. (2011). A look into the factor model black box: Publication lags and the role of hard and soft data in forecasting GDP. *International Journal of Forecasting*, 27(2), 333–346.
- Bok, B., Caratelli, D., Giannone, D., Sbordone, A. M., & Tambalotti, A. (2018). Macroeconomic nowcasting and forecasting with big data. *Annual Review of Economics*, 10, 615–643.
- Breitung, J., & Eickmeier, S. (2011). Testing for structural breaks in dynamic factor models. *Journal of Econometrics*, 163(1), 71–84.
- Dahlhaus, T., Guénette, J.-D., & Vasishtha, G. (2017). Nowcasting BRIC+M in real time. *International Journal of Forecasting*, 33(4), 775–788.
- de Valk, S., de Mattos, D., & Ferreira, P. (2019). Nowcasting: An R package for predicting economic variables using dynamic factor models. *The R Journal*, 11(1), 230–244.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Doz, C., Giannone, D., & Reichlin, L. (2011). A two-step estimator for large approximate dynamic factor models based on Kalman filtering. *Journal of Econometrics*, 164(1), 188–205.
- Doz, C., Giannone, D., & Reichlin, L. (2012). A quasi-maximum likelihood approach for large, approximate dynamic factor models. *Review of Economics and Statistics*, 94(4), 1014–1024.
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press.
- Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004). The MIDAS touch: Mixed data sampling regression models. Working paper, UCLA/UNC.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
- Giannone, D., Reichlin, L., & Small, D. (2008). Nowcasting: The real-time informational content of macroeconomic data. *Journal of Monetary Economics*, 55(4), 665–676.
- Jungbacker, B., & Koopman, S. J. (2015). Likelihood-based dynamic factor analysis for measurement and forecasting. *Econometrics Journal*, 18(2), C1–C21.
- Koopman, S. J., & Durbin, J. (2000). Fast filtering and smoothing for multivariate state space models. *Journal of Time Series Analysis*, 21(3), 281–296.
- Mariano, R. S., & Murasawa, Y. (2003). A new coincident index of business cycles based on monthly and quarterly series. *Journal of Applied Econometrics*, 18(4), 427–443.
- McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. *Journal of Business & Economic Statistics*, 34(4), 574–589.
- Shumway, R. H., & Stoffer, D. S. (1982). An approach to time series smoothing and forecasting using the EM algorithm. *Journal of Time Series Analysis*, 3(4), 253–264.
- Stock, J. H., & Watson, M. W. (2002). Forecasting using principal components from a large number of predictors. *Journal of the American Statistical Association*, 97(460), 1167–1179.

> As referências devem ser conferidas (páginas, DOIs) antes de entrarem no `.bib` final do paper.

# NowcastBox — Plano de paridade com o *ECB Nowcasting Toolbox*

| Campo | Valor |
|---|---|
| Referência | Linzenich, J. & Meunier, B. (2024). *Nowcasting made easier: a toolbox for economists*. ECB Working Paper No. 3004 (código MATLAB/R: `baptiste-meunier/Nowcasting_toolbox`) |
| Versão de partida | `nowcastbox` 0.1.2 (PyPI, DOI 10.5281/zenodo.23102804) |
| Versões-alvo | 0.2.0 (Fase 1), 0.3.0 (Fase 2), 0.4.0 (Fase 3) |
| Data do documento | 2026-10-02 |
| Status | **Plano concluído: Fases 1, 2 e 3 integradas** na branch `ecb-parity` (2026-10-02); releases 0.2.0/0.3.0/0.4.0 ainda não publicados |

---

## 0. Objetivo e regras

Cobrir todas as funcionalidades do *ECB Nowcasting Toolbox* que o `nowcastbox` ainda não tem,
para que a seção "Related software" do paper JSS possa afirmar paridade (e as vantagens já
existentes: contribuições exatas, news com reestimação, vintages reais, densidades, testes
formais, frequências semanal/diária, MIDAS).

**Clean room (vale para todo o plano).** Implementar a partir dos papers listados em cada
item. O código do toolbox do ECB **não declara licença**: pode-se ler o paper e o README, mas
**não** o código-fonte MATLAB/R. Continua valendo a regra do `CLAUDE.md` sobre o pacote R
`nowcasting` (GPL) e o `statsmodels` (só como dependência/referência numérica). Registrar as
fontes de cada item em `FONTES_POR_MODULO.md`.

**Padrões de qualidade (CI).** Cobertura ≥ 90 % (≥ 95 % em `core/` e `statespace/`),
docstrings NumPy com exemplo executável (interrogate ≥ 95 %), complexidade ≤ 10, ruff e
pyright limpos. Cada item novo entra em: API do subpacote (`__init__.py`), docs (teoria +
guia do usuário), `CHANGELOG.md`, `FONTES_POR_MODULO.md`, e — quando fizer sentido — no
pipeline YAML/CLI e no relatório HTML. O `nowcastbox/__init__.py` de topo só é alterado no
passo de integração de cada fase.

### Visão geral

| # | Item | Fase | Esforço | Módulos principais | Situação |
|---|---|---|---|---|---|
| 1 | FDA (acurácia direcional) + teste de Pesaran-Timmermann | 1 | S | `evaluation/metrics.py`, `evaluation/tests.py`, `evaluation/backtest.py` | **Concluído** |
| 2 | Métricas por subperíodo | 1 | S | `evaluation/backtest.py` | **Concluído** |
| 3 | Bandas de erro empírico (Reifschneider-Tulip) | 1 | S–M | `density/empirical.py` (novo) | **Concluído** |
| 4 | Heatmap de z-scores dos indicadores | 1 | S | `diagnostics/zscores.py` (novo), `visualization/heatmap.py` (novo) | **Concluído** |
| 5 | Parcela dos dados já divulgados | 1 | S | `core/data.py`, `visualization/data_flow.py`, `news/tracker.py` | **Concluído** |
| 6 | Nowcasts de modelos alternativos (sem 1–2 grupos) | 1 | S–M | `experiment/alternatives.py` (novo) | **Concluído** |
| 7 | Templates Excel | 1 | S | `pipeline/data.py`, `pipeline/spec.py`, `pipeline/templates/example.xlsx`, `cli/main.py` | **Concluído** |
| 8 | Pré-seleção completa (SIS, LARS, score agregado) | 2 | M | `selection/targeted.py`, `selection/preselection.py` (novo) | **Concluído** |
| 9 | Busca aleatória de especificações + robustez à Covid | 2 | M | `selection/search.py` (novo), `experiment/` | **Concluído** |
| 10 | Combinação de bridge equations | 2 | M | `models/bridge_combination.py` (novo), `benchmarks/` | **Concluído** |
| 11 | BVAR grande (Cimadomo et al. 2022) | 3 | L | `models/bvar.py`, `models/_bvar_prior.py`, `models/_bvar_blocking.py`, `models/bvar_extrapolation.py` (novos) | **Concluído** |

Esforço: S ≈ até 1 dia, M ≈ 2–5 dias, L ≈ 2–3 semanas (inclui testes, docs e validação).

---

## Fase 1 — Saídas de avaliação e de conjuntura (v0.2.0)

Objetivo: anular as vantagens "operacionais" do toolbox antes da submissão do paper JSS.
Todos os itens são independentes entre si e podem ser feitos em paralelo.

### Item 1 — FDA e teste de Pesaran-Timmermann (S)

**Definição (WP 3004, nota 9).** \(FDA = N^{-1}\sum_t I_t\), com \(I_t = 1\) se
\((y_t - y_{t-1})(\hat y_t - y_{t-1}) > 0\): o modelo acerta se a variável acelera ou
desacelera em relação ao último valor observado.

**Referências.** Pesaran & Timmermann (1992, *JBES* 10(4)) — teste de acurácia direcional;
Blaskowitz & Herwartz (2011, *IJF* 27(4)) — erros direcionais; Linzenich & Meunier (2024).

**API.**
- `evaluation.directional_accuracy(actual, forecast, previous) -> float` e
  `evaluation.pesaran_timmermann(actual, forecast, previous) -> PesaranTimmermannResult`
  (estatística, p-valor, taxa de acerto, n).
- `BacktestResults.metrics(..., metrics=("rmsfe", "mae", "bias", "fda", "n"))`: `"fda"` passa
  a ser aceito. Como as métricas atuais recebem só erros, `metric_by_horizon` ganha um caminho
  para métricas que precisam de `actual`/`forecast`/`previous` (registro separado
  `DIRECTIONAL_METRICS`, sem mudar a assinatura das existentes).
- `BacktestResults.directional_accuracy(horizon=..., test=True)` → tabela com FDA e p-valor PT.

**Decisão de desenho.** \(y_{t-1}\) é o valor do período anterior **conhecido na data da
vintage** (com safras reais, a primeira divulgação; em pseudo tempo real, a série final). O
backtest já guarda a vintage; adicionar a coluna `previous_actual` a `FORECAST_COLUMNS` (sem
quebrar `to_parquet`/`from` antigos: coluna opcional).

**Testes.** Casos à mão; FDA = 1 quando a previsão é perfeita; PT sob H0 (simulação de
tamanho); antissimetria; NaNs.

### Item 2 — Métricas por subperíodo (S)

**API.** `BacktestResults.metrics(..., periods=None)`, idem em `rmsfe_by_horizon`,
`relative_to`, `diebold_mariano`, `clark_west`, `giacomini_white` e `mcs`:

```python
res.metrics(horizon="kind", periods={
    "pre-Covid": ("2012Q1", "2019Q4"),
    "Covid": ("2020Q1", "2021Q4"),
    "post-Covid": ("2022Q1", None),
})
```

- Índice ganha o nível `period`. Atalho `periods="covid"` usa a janela padrão do
  `models/robust.py` (2020-03..2021-12, convertida para o período-alvo), mais `"ex-Covid"`.
- Implementar como filtro sobre `target_period` reaproveitando `evaluable()`; um helper
  privado `_by_periods(func, periods, ...)` evita duplicar lógica nos 7 métodos.

**Testes.** Igualdade com o filtro manual (o que os papers fazem hoje em `e03_evaluate.py`).

### Item 3 — Bandas de erro empírico (S–M)

**Método.** Reifschneider & Tulip (2019, *IJF* 35(4)); ECB (2009, *New procedure for
constructing Eurosystem and ECB staff projection ranges*); WP 3004 §2.3: banda = previsão ±
MAE dos erros passados **no mesmo horizonte** (meses até o fim do trimestre), janela móvel
de 10 anos; ±1 MAE ≈ 57,5 % sob normalidade; opção de ajuste de outliers nos erros
passados (excluir/winsorizar). Generalizar para qualquer nível: com erros gaussianos,
MAE = σ√(2/π), logo o quantil q usa σ̂ = MAE·√(π/2); opção não-paramétrica (quantis
empíricos dos erros).

**API.**
```python
from nowcastbox.density import empirical_bands
bands = empirical_bands(results, backtest, window="10Y", method="mae",  # "mae" | "rmse" | "quantile"
                        levels=(0.575, 0.9), outliers="winsorize")
```
- Retorna `NowcastDistribution` gaussiana (métodos `mae`/`rmse`) ou um objeto de quantis
  (`quantile`), para reaproveitar `plot("fan")`, `interval`, CRPS etc.
- Atalho: `NowcastResults.distribution(method="empirical", backtest=...)`.
- Só usa erros de vintages **anteriores** à data do nowcast (sem look-ahead): teste específico.

**Testes.** Cobertura nominal em simulação (erros gaussianos), igualdade MAE·√(π/2) = σ,
ausência de look-ahead, horizonte sem erros suficientes → aviso + NaN.

### Item 4 — Heatmap de z-scores (S)

**Método.** WP 3004 §3.4: z-score de cada série transformada em relação à média e desvio
de longo prazo; séries mensais suavizadas por média móvel de 5 meses (pesos
Mariano-Murasawa 1-2-3-2-1, Mariano & Murasawa 2003) para ficarem comparáveis ao trimestre;
agregação por grupo = média dos z-scores do grupo.

**API.**
```python
from nowcastbox.diagnostics import indicator_zscores
z = indicator_zscores(data, smooth="mm", window=None, by=None)   # by="category" | "block" | dict
z.plot(kind="heatmap", last=24)
results.plot("indicator_heatmap")                                # registrado via register_plot
```
- `window=None` = amostra toda; aceitar janela móvel. Média/desvio só com dados até a data
  da vintage (opção `as_of=`).
- Grupos: `SeriesMetadata.category`, `blocks` ou mapeamento livre.

**Testes.** Z-score conhecido em série sintética, pesos MM somando 9, grupos, NaN na borda.

### Item 5 — Parcela dos dados já divulgados (S)

**Método.** Para o trimestre-alvo e a data da vintage: proporção de observações mensais do
trimestre já disponíveis (por série, por grupo e total; opcionalmente ponderada pelos pesos
do modelo na previsão).

**API.** `MixedFrequencyData.released_share(period, by=None, weights=None)` reaproveitando
`data_availability` (`visualization/data_flow.py:62`), e coluna `released_share` no
`nowcast_tracker` e no relatório HTML.

**Testes.** Contagem manual em painel com borda irregular conhecida.

### Item 6 — Nowcasts de modelos alternativos (S–M)

**Método.** WP 3004 §3.5: reestimar (ou só refiltrar com os parâmetros fixos) o modelo
retirando 1 ou 2 grupos de variáveis; a amplitude dos nowcasts alternativos mostra a
dependência do resultado em grupos específicos. Referência para combinação: Granger & Jeon
(2004, *J. Policy Modeling*); Timmermann (2006, *Handbook of Economic Forecasting*).

**API.**
```python
from nowcastbox.experiment import alternative_models
alt = alternative_models(model, data, target="gdp", by="category",
                         drop=(1, 2), refit=True, n_jobs=-1)
alt.table()        # nowcast por combinação retirada
alt.range()        # min / max / mediana por período-alvo
alt.plot()         # leque de modelos alternativos sobre o nowcast base
```
- `refit=False`: filtra com os parâmetros do modelo base (rápido; usa `NowcastResults`
  com a série removida tratada como ausente).
- Paralelismo com `joblib` (como `PseudoRealTimeBacktest`).

**Testes.** Número de combinações; `refit=False` igual a marcar as séries como NaN; grupo
inexistente → erro.

### Item 7 — Templates Excel (S)

**API.**
- `pipeline.DataSpec(source="excel", path=..., sheets={"monthly": ..., "quarterly": ...,
  "metadata": ...})`, lendo com `pandas.read_excel` (extra opcional `[excel]` com
  `openpyxl`).
- Planilha de metadados com as colunas do `SeriesMetadata` (frequência, código de
  transformação 0–7 via `transform_from_code`, atraso em dias, grupo/categoria, blocos).
- `nowcastbox init --excel` (CLI) gera um template vazio; `pipeline/templates/` ganha o
  `.xlsx` de exemplo; exportação de resultados para Excel em `OutputsSpec`.

**Testes.** Ida e volta (escrever template → ler → mesmo `MixedFrequencyData`); erro claro
sem `openpyxl`.

### Integração da Fase 1

> **Situação (2026-10-02): concluída**, exceto os itens de paper e o release (ver abaixo).
> Feito: `nb.empirical_bands`, `nb.indicator_zscores`, `nb.alternative_models`,
> `nb.AlternativeNowcasts`, `nb.pesaran_timmermann` no topo; plots `"indicator_heatmap"`,
> `"released_share"` e `"density"` com `method="empirical"`; relatório HTML com bandas
> empíricas (tiles + leque 57,5/68/90 %), modelos alternativos, parcela divulgada (por
> categoria, padrão), heatmap (`heatmap=`) e tabela de acurácia (`backtest_metrics=`);
> pipeline com saídas `empirical_bands`, `heatmap`, `alternatives`, `excel` (escrita de
> fato do workbook) e opções `metrics`/`periods` em `outputs.backtest` (as chaves
> `evaluation.periods`/`evaluation.metrics` propostas abaixo ficaram dentro de
> `outputs.backtest`, que é onde o backtest é configurado); `CHANGELOG`, `FONTES`, nav
> do MkDocs, `CONTRATOS`. Pendentes: tabelas do paper do PIB / ilustração do JSS e o
> release 0.2.0 (bump de versão e tag ficam para a publicação).
>
> Decisões em aberto resolvidas: (1) FDA com o valor anterior disponível na vintage,
> `previous="final"` como opção; (2) bandas com 57,5 % + 68 % + 90 % por padrão; (3) LARS
> com implementação própria (Fase 2); (4) Excel via extra opcional `[excel]`
> (`openpyxl`); (5) BVAR neste ciclo de desenvolvimento (Fase 3).

- `NowcastResults`: métodos/plots novos (`plot("indicator_heatmap")`, `distribution(method=
  "empirical")`).
- Relatório HTML (`reports/`): seções "heatmap", "bandas empíricas", "modelos alternativos",
  "dados divulgados".
- Pipeline YAML: `outputs.heatmap`, `outputs.empirical_bands`, `outputs.alternatives`,
  `evaluation.periods`, `evaluation.metrics: [rmsfe, fda]`.
- Paper do PIB (`02_brazil_gdp_nowcast`): usar FDA, subperíodos e bandas empíricas nas
  tabelas; paper JSS: nova ilustração curta.
- Release **0.2.0**.

---

## Fase 2 — Construção de modelos (v0.3.0)

### Item 8 — Pré-seleção completa (M)

**Método (WP 3004 §2.1).** Três rankings dos indicadores contra o alvo, combinados num score:
1. **t-stat** (Bair et al. 2006, *JASA* 101(473); Bai & Ng 2008, *J. Econometrics*
   146(2)): já existe em `hard_threshold` (`selection/targeted.py:321`), com lags do alvo
   e HAC.
2. **SIS** (Fan & Lv 2008, *JRSS-B* 70(5)): correlação marginal; expor como
   `sis(x, y, ...)` em vez de depender de `hard_threshold(y_lags=0)`.
3. **LARS** (Efron, Hastie, Johnstone & Tibshirani 2004, *Ann. Statist.* 32(2)): ordem de
   entrada no caminho LARS. Implementação própria (algoritmo do paper; ~150 linhas) ou
   `lars_path` do scikit-learn **só se** o sklearn já for dependência (hoje é opcional:
   preferir implementação própria, validada contra o sklearn nos testes).

**Extras.** Leads/lags dos regressores (`x_lags=`), score agregado ponderado
(`weights={"tstat": 1, "sis": 1, "lars": 1}`), e tabela final com atraso de publicação,
grupo e frequência (a partir do `SeriesMetadata`/`release_table`). Referência aplicada:
Chinn, Meunier & Stumpner (2023, ECB WP 2808).

**API.**
```python
from nowcastbox.selection import preselect
pre = preselect(data, target="gdp", methods=("tstat", "sis", "lars"),
                x_lags=(0, 1), weights=None, top=30)
pre.table()            # ranking por método, score agregado, atraso, grupo, frequência
pre.selected           # nomes
data_small = data.select(pre.selected + ["gdp"])
```

**Testes.** LARS vs. `sklearn.linear_model.lars_path` (teste marcado `reference_validation`),
SIS vs. correlação manual, score agregado em exemplo à mão, uso só de dados até `as_of`.

### Item 9 — Busca de especificações e robustez à Covid (M)

**Método (WP 3004 §2.2–2.3).** Sortear especificações dentro de limites (nº de fatores,
lags do VAR dos fatores, nº/subconjunto de variáveis — "estratégia funil" pelo ranking do
Item 8 —, data inicial da amostra, blocos), avaliar cada uma em pseudo tempo real, e
ordenar por um score ponderado por horizonte (back/now/forecast) e métrica (RMSE, FDA).
Passo de robustez: repetir as melhores especificações com cada tratamento da Covid
(nenhum, dummies, excluir observações, correção de outliers) e avaliar no pós-2021.

**API.**
```python
from nowcastbox.selection import SpecificationSearch
search = SpecificationSearch(
    model=nb.MixedFreqDFM, data=data, target="gdp",
    space={"n_factors": [1, 2, 3], "factor_lags": [1, 2], "n_series": (20, 80),
           "start": ["2005-01", "2010-01"]},
    n_draws=200, ranking=pre,                     # Item 8 (funil)
    backtest={"start": "2015-01-15", "end": "2024-12-15", "delay": delays},
    score={"rmsfe": 0.7, "fda": 0.3}, horizon_weights={"nowcast": 0.5, "backcast": 0.25,
                                                        "forecast": 0.25},
    random_state=0, n_jobs=-1,
)
out = search.run()
out.table()                                       # uma linha por especificação, ordenado
out.covid_robustness(top=5, treatments=("none", "dummy", "mask", "outliers"),
                     evaluate_from="2022Q1")
out.best_model()
```
- Reaproveitar `PseudoRealTimeBacktest`, `select_blocks`/`select_variables`
  (`selection/blocks.py`) e `NowcastExperiment` (`experiment/experiment.py`).
- Checkpoint em disco (Parquet) para retomar buscas longas; seed fixa por sorteio.
- Grid exaustivo como caso particular (`n_draws=None`).

**Testes.** Reprodutibilidade com seed, score ponderado à mão, checkpoint/retomada, espaço
inválido → erro; teste `slow` de ponta a ponta com painel simulado.

### Item 10 — Combinação de bridge equations (M)

**Método.** Bańbura, Belousova, Bodnár & Tóth (2023, ECB WP 2815): estimar **todas** as
equações-ponte com 1–2 indicadores mensais e 0–1 trimestral, extrapolar os mensais até o
fim do trimestre e combinar as previsões (média simples; opções: mediana, pesos pelo
inverso do MSE passado — Stock & Watson 2004, *J. Forecasting* 23(6); Timmermann 2006).
Extrapolação dos indicadores: AR (já existe em `ar_extend`), e — após o Item 11 — BVAR.
Baffigi, Golinelli & Parigi (2004) e Diron (2008) para as equações-ponte individuais.

**API.**
```python
from nowcastbox.models import BridgeCombination
model = BridgeCombination(max_monthly=2, max_quarterly=1, target_lags=1,
                          combine="mean",          # "median" | "inverse_mse"
                          extrapolation="ar")      # "bvar" depois do Item 11
res = model.fit(data, target="gdp")
res.nowcast, res.equations()                      # previsões de cada equação
```
- Usa `BridgeEquation` (`models/bridge.py:574`) por equação; cache dos indicadores
  extrapolados (cada série é extrapolada uma vez por vintage).
- Benchmark no `benchmarks/` (`BridgeCombinationBenchmark`) para entrar no backtest.
- `news` aproximada por diferença de nowcasts entre vintages (documentar que não é a
  decomposição exata do DFM).

**Testes.** Contagem de equações (\(\binom{N_m}{1}+\binom{N_m}{2}\))·(1+N_q); com 1 equação
iguala `BridgeEquation`; pesos inverse-MSE somam 1; desempenho com N_m = 50 (≈1.275
equações) abaixo de um limite.

### Integração da Fase 2

> **Situação (2026-10-02): concluída**, exceto o paper JSS e o release 0.3.0.
> Feito: `nb.preselect`, `nb.PreselectionResult`, `nb.SpecificationSearch`,
> `nb.SearchResults`, `nb.CovidRobustness`, `nb.SpecifiedModel`, `nb.BridgeCombination`,
> `nb.BridgeCombinationResults` no topo; pipeline YAML com a seção `selection`
> (`selection.preselect` — aplicada ao modelo e refeita por vintage no backtest —,
> `selection.search` com `covid_robustness` e `apply`) e `model.type:
> bridge_combination` (módulo novo `pipeline/selection.py`; template `model_building`
> para `nowcastbox init`); tabelas `preselection`/`search`/`robustness` no snapshot, no
> Excel e no relatório HTML (`NowcastReport(selection=...)`); guia "Building a model from
> scratch" (`docs/user-guide/model-building.md`), páginas de teoria/guia dos itens 8–10,
> nav do MkDocs, referências; `CHANGELOG`, `FONTES`, `CONTRATOS`. Consistência: o argumento
> `evaluate_periods` da busca passou a `periods` (como em `BacktestResults`). Pendentes:
> tabela de paridade com o ECB no paper JSS e o release 0.3.0.
>
> Gancho da Fase 3: `models/bvar.py` deve chamar `register_extrapolator("bvar", factory)`
> (a fábrica recebe `**extrapolation_options` e devolve `(data, columns, end) -> dict`
> de séries na grade nativa até `end`; ver `extrapolation.native_until`), levantar
> `NowcastDataError` quando não puder ajustar (permite o recuo série a série) e ser
> importado em `models/__init__.py`.

Pipeline YAML (`selection.preselect`, `selection.search`, `model: bridge_combination`), docs
(guia "Construindo um modelo do zero" no fluxo do ECB: pré-seleção → busca → robustez),
paper JSS (tabela de paridade com o ECB), release **0.3.0**.

---

## Fase 3 — BVAR grande (v0.4.0)

### Item 11 — BVAR de frequência mista para nowcasting (L)

**Método.** Cimadomo, Giannone, Lenza, Monti & Sokol (2022, *J. Econometrics* 231(2)) —
BVAR grande para nowcasting com dados mistos e borda irregular (abordagem por blocos /
"blocking" para frequências mistas); prior Normal-Inverse-Wishart com Minnesota
(Litterman 1986, *JBES* 4(1); Doan, Litterman & Sims 1984), *sum-of-coefficients* e
*dummy-initial-observation* (Sims & Zha 1998; Bańbura, Giannone & Reichlin 2010, *JAE*
25(1)); escolha hierárquica do *overall tightness* (Giannone, Lenza & Primiceri 2015,
*REStat* 97(2)); previsão condicional para a borda irregular (Waggoner & Zha 1999, *REStat*
81(4); Bańbura, Giannone & Lenza 2015, *IJF* 31(3)).

**Escopo.**
- `models/_bvar_prior.py`: dummies de Minnesota/SoC/DIO, verossimilhança marginal fechada
  (GLP 2015) e otimização dos hiperparâmetros.
- `models/bvar.py`: `LargeBVAR(lags=..., prior="glp", blocking="quarterly")` com
  `fit(data, target) -> NowcastResults`; densidades por simulação da posterior (integra com
  `density/`); previsão condicional via Kalman (reaproveitar `statespace/` com os dados
  ausentes) para a borda irregular.
- Opção de extrapolação `extrapolation="bvar"` no Item 10.
- News: decomposição via forma de espaço de estados (o BVAR condicional é linear-gaussiano
  dado os parâmetros, então `news/` pode ser reaproveitado).

**Validação.** Verossimilhança marginal vs. fórmula fechada em casos pequenos; recuperação
de parâmetros em VAR simulado; posterior NIW vs. solução analítica e Monte Carlo (o
`statsmodels` não tem BVAR, então não há referência numérica externa em Python); tempo com
N = 50–100 séries.

**Release 0.4.0**; no paper JSS, se a Fase 3 não estiver pronta na submissão, citar como
trabalho futuro.

### Integração da Fase 3

> **Situação (2026-10-02): concluída**, exceto o release 0.4.0 e o paper JSS.
> Feito: `_bvar_prior.py` (prior NIW de Minnesota + SoC/DIO, verossimilhança marginal
> fechada por QR, gradiente analítico, escolha hierárquica GLP, retiradas exatas da
> posterior); `_bvar_blocking.py` (blocagem, forma companheira, previsão condicional por
> Kalman e em forma fechada); `bvar.py` (`LargeBVAR`/`LargeBVARResults`: nowcast pela
> previsão condicional na média da posterior, densidade por mistura sobre retiradas
> NIW, *news* e contribuições exatas via gancho `linear_nowcast_model()`, `predict` por
> vintage, aviso quando a amostra balanceada descarta trimestres); `bvar_extrapolation.py`
> (extrapolador `"bvar"` do Item 10: VAR mensal com indicadores só mensais, VAR por blocos
> com trimestrais, cache por vintage, reaproveitado pelo `BridgeCombinationBenchmark`);
> `nb.LargeBVAR`, `nb.LargeBVARResults`, `nb.BVARExtrapolator` no topo; pipeline
> `model.type: large_bvar` (com *news*, densidade pela posterior, backtest e relatório
> HTML); páginas de teoria/guia, nav, referências, comparação; `CHANGELOG`, `FONTES`,
> `CONTRATOS`, `STATUS`.
>
> Limitações registradas: amostra de estimação = maior trecho balanceado (séries que
> começam tarde encurtam a amostra; `DataQualityWarning`); hiperparâmetros na moda da
> posterior (sem o passo de Metropolis do GLP); `nowcast_tracker` indisponível para o
> BVAR; só grades mensais com séries mensais/trimestrais; sem estudo em dados reais.

---

## Cronograma sugerido

| Semana | Entrega |
|---|---|
| 1 | Itens 1, 2, 5 (avaliação) + item 4 (heatmap) |
| 2 | Itens 3, 6, 7 + integração (relatório, pipeline, docs) → **0.2.0** |
| 3 | Item 8 (pré-seleção) + item 10 (bridge combination com AR) |
| 4 | Item 9 (busca + robustez Covid) + integração → **0.3.0** |
| 5–7 | Item 11 (BVAR) + extrapolação BVAR no item 10 → **0.4.0** |

Paper JSS: submeter depois da 0.3.0 (paridade completa exceto BVAR), com a tabela comparativa
R `nowcasting` | `statsmodels` DFMQ | NY Fed | ECB toolbox | `nowcastbox`.

## Riscos

| Risco | Mitigação |
|---|---|
| Ambiguidade de definição (FDA com \(y_{t-1}\) final vs. em tempo real; bandas com 10 anos de erros quando o backtest é curto) | Documentar a escolha e expor como parâmetro; aviso quando a janela tiver poucos erros |
| Custo computacional (busca de especificações, ~1.300 bridges, modelos alternativos com reestimação) | `joblib`, cache de extrapolações, `refit=False`, checkpoint |
| Look-ahead em z-scores, bandas e pré-seleção | Argumento `as_of`/uso só da vintage; testes dedicados como os do `PseudoRealTimeBacktest` |
| BVAR grande é o item mais arriscado (prior, otimização, condicionamento) | Fase separada; validar por fórmulas fechadas e simulação antes de integrar |
| Clean room (código do ECB sem licença) | Só paper e README; fontes registradas em `FONTES_POR_MODULO.md` |

## Decisões em aberto

1. FDA: usar \(y_{t-1}\) da primeira divulgação (tempo real) ou da série final? Proposta:
   padrão = valor disponível na vintage; opção `previous="final"`.
2. Bandas empíricas: incluir o nível 57,5 % por padrão (convenção do ECB) além de 68 %/90 %?
3. LARS: implementação própria (sem nova dependência) — confirmar.
4. Excel: extra opcional `[excel]` com `openpyxl` — confirmar.
5. Fase 3 (BVAR) antes ou depois da submissão do JSS. (Fase 3 concluída em 2026-10-02;
   resta decidir se a submissão cita a 0.4.0.)

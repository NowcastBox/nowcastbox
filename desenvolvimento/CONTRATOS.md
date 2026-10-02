# NowcastBox — Contratos do núcleo (`nowcastbox.core`)

> Documento de referência para o desenvolvimento em paralelo. Tudo que está aqui é
> **contrato**: outros módulos podem depender destes nomes, atributos e convenções.
> Mudanças exigem atualização deste arquivo, do `CHANGELOG.md` e dos testes em
> `tests/core/`.
>
> Versão do contrato: **0.1.0.dev0** (fase 0 — fundação; revisado na integração da onda 1:
> `SeriesMetadata.transform_applied`, API de topo, `tests/conftest.py` com `--run-network`
> e BLAS em 1 thread; revisado na integração da onda 2: métodos de análise de
> `NowcastResults`, tipos de gráfico registrados, `MixedFreqDFMResults.state_space_model()`,
> `filter_method="auto"` e mapa de módulos §9; revisado na integração da onda 3: grades
> semanais/diárias com agregação de calendário (§1, §3), atrasos de divulgação negativos
> (§4), `state_space_model(n_periods)`, *news*/densidade/diagnósticos para modelos de
> calendário e two-step `"variables"`, multi-start do EM, convenção de pré-amostra (§5),
> `pipeline/`, `cli/`, `simulate/` implementados (§9)). Plano: `PROJETO_DESENVOLVIMENTO.md`.

---

## 1. Convenções gerais

| Tema | Convenção |
|---|---|
| Grade base | `pandas.PeriodIndex` **contígua** com a frequência base (mensal `"M"`; semanal `"W"` ou diária `"D"` para I1). Lacunas são preenchidas com NaN (com `DataQualityWarning`). Índices são ordenados. |
| Frequências mais baixas | Valor armazenado no **último período base** do período nativo: trimestre → 3º mês (mar/jun/set/dez); ano → dezembro; em grades semanais, a última semana que **termina** no período. Convenção de calendário (pandas): um período base pertence ao período nativo que contém o seu **último dia** (a semana 2020-01-27/2020-02-02 é de fevereiro). Demais células da coluna são NaN estruturais; valor fora do "slot" → `NowcastDataError`. |
| Frequências mais altas que a base | Proibidas (`NowcastDataError`). Para dados diários/semanais use uma grade base mais fina (arquitetura pronta para I1). |
| Valores | `float64`; faltante = `NaN`; `inf` → `NowcastDataError`. |
| Nomes de séries | `str` não vazia, únicas. |
| Imutabilidade | `MixedFrequencyData` e `NowcastResults` nunca expõem o DataFrame interno por referência; métodos retornam objetos novos. |
| Padronização | `z = (x - média) / desvio`, por coluna, sobre valores **observados**, `ddof=1` por padrão. Estatísticas guardadas em `StandardizationStats`. Série constante ou com < `ddof+1` observações → `NowcastDataError`. |
| Unidades dos resultados | `NowcastResults.nowcast` está nas **unidades originais do alvo como passado ao `fit`** (após transformações de `preprocessing`, antes da padronização). |
| Erros | Sempre explícitos: `NowcastDataError` (dados), `FormulaError` (fórmula), `ModelNotFittedError`, `ValueError` (hiperparâmetros). Nunca falhar silenciosamente. |
| Warnings | `ConvergenceWarning` (EM/otimizadores), `DataQualityWarning` (dados suspeitos). Ambos herdam de `NowcastBoxWarning`. |
| Aleatoriedade | Todo método aleatório recebe `random_state` (int, `numpy.random.Generator` ou None). |
| Parâmetros | `snake_case` descritivo: `n_factors`, `factor_lags`, `n_shocks`, `blocks`, `max_iter`, `tol`. |
| Logging | `from nowcastbox._logging import get_logger; logger = get_logger(__name__)`. Nunca `print`. |
| Testes | `tests/<subpacote>/` (sem `__init__.py`; pytest usa `--import-mode=importlib` e `pythonpath=["."]`, então nomes de arquivo podem se repetir entre pastas e helpers são importáveis como `tests.<subpacote>.<arquivo>`). Fixture global `rng` em `tests/conftest.py`, que também registra `--run-network` (ou `NOWCASTBOX_RUN_NETWORK=1`; testes `network` são pulados sem ele) e limita o BLAS a 1 thread na sessão (`threadpoolctl`; `NOWCASTBOX_TEST_BLAS_THREADS` altera); perfil hypothesis `HYPOTHESIS_PROFILE=ci\|dev`. Testes de desempenho (pytest-benchmark) em `tests/performance/`; comparação com referências em `tests/reference_validation/`; integração em `tests/integration/`. |
| Exportação | Cada subpacote exporta seus nomes públicos no próprio `__init__.py` (`__all__`). O `nowcastbox/__init__.py` de topo é editado **somente** pela etapa de integração (onda 1: reexporta `prepare_panel`, `select_factors`/`select_shocks`, `pseudo_real_time`, `ReleaseCalendar`, `VintageStore`, `TwoStepDFM`, `MixedFreqDFM`, transformações, `StateSpace`/Kalman, subpacotes como atributos e `nowcast` de `nowcastbox/api.py`). |

---

## 2. `core.exceptions`

```
NowcastBoxError(Exception)
├── NowcastDataError(NowcastBoxError, ValueError)
├── ModelNotFittedError(NowcastBoxError, AttributeError)
└── FormulaError(NowcastBoxError, ValueError)

NowcastBoxWarning(UserWarning)
├── ConvergenceWarning
└── DataQualityWarning
```

Uso: `warnings.warn(msg, ConvergenceWarning, stacklevel=2)`.

---

## 3. `core.frequency`

### `Frequency(str, Enum)`

| Membro | Valor | `periods_per_year` | `pandas_freq` |
|---|---|---|---|
| `DAILY` | `"D"` | 365 | `"D"` |
| `WEEKLY` | `"W"` | 52 | `"W"` |
| `MONTHLY` | `"M"` | 12 | `"M"` |
| `QUARTERLY` | `"Q"` | 4 | `"Q"` |
| `ANNUAL` | `"A"` | 1 | `"Y"` |

- `Frequency.from_value(x)` aceita membro, string (`"M"`, `"monthly"`, `"Q-DEC"`, `"QE"`,
  `"Y"`, `"annual"`, ...) ou inteiro de períodos por ano (12, 4, 1, 52, 365 — convenção
  `ts` do R). `FrequencyLike = Frequency | str | int`.
- `Frequency.from_index(period_index)`; `.is_lower_than(other)`, `.is_higher_than(other)`, `.label`.

### Funções

| Função | Descrição |
|---|---|
| `aggregation_ratio(high, low) -> int` | Razão fixa (M/Q=3, M/A=12, Q/A=4, D/W=7, igual=1); razões variáveis (D/M, W/M, W/Q...) → `ValueError` (use os auxiliares de calendário). |
| `is_fixed_ratio(high, low) -> bool` | Se o par tem razão fixa (onda 3). |
| `max_periods_per(high, low) -> int` | Máximo de períodos base por período nativo (W/M=5, W/Q=14, D/M=31...). |
| `native_period_bounds(index, base)` | Primeiro e último período base de cada período nativo (convenção de calendário). |
| `calendar_position(index, freq) -> (posição, comprimento)` | Posição de cada período base dentro do seu período nativo e o número de períodos base desse período. |
| `period_to_base(period, base, how="end")` | `2020Q1` → `2020-03` (ou `2020-01` com `how="start"`). |
| `base_to_native(index, freq)` | Cada período base → período nativo que o contém. |
| `native_to_base(index, base)` | Índice nativo → slots na grade base (fim do período). |
| `is_period_end(index, freq) -> ndarray[bool]` | Máscara dos slots de armazenamento de `freq` na grade base. |
| `infer_frequency(series)` | Heurística: frequência mais baixa (todas as frequências mais baixas, inclusive pares de calendário) cujos slots contêm todas as observações (≥ 2 obs.). |

### `AggregationType(str, Enum)`

Pesos `w` sobre `(x_t, x_{t-1}, ...)`, `t` = último período de alta frequência; `k` = razão:

| Membro | Valor | Pesos |
|---|---|---|
| `FLOW` | `"flow"` | `(1, …, 1)` (k) |
| `AVERAGE` | `"average"` | `(1, …, 1)/k` |
| `STOCK` | `"stock"` | `(1, 0, …, 0)` (k) |
| `GROWTH_RATE` | `"mariano_murasawa"` | triangular de comprimento `2k-1`; k=3: `(1,2,3,2,1)`; `normalize=True` divide por `k` (⅓(1,2,3,2,1), §4.3) |

`AggregationType.from_value("mm")`, `.weights(k, normalize=False)`, `.n_lags(k)`,
`.calendar_weights(n_current, n_previous=None)` (onda 3: pesos exatos de fluxo/média/
estoque e da taxa de crescimento da média do período para períodos de comprimentos
desiguais; igual a `weights(k, normalize=True)` quando os comprimentos são iguais).
Em `preprocessing.aggregation`: `CalendarAggregation`, `calendar_aggregation(high, low,
aggregation)` e `calendar_weight_matrix(index, low, aggregation)` (pesos por período,
mais recente primeiro).
`SeriesMetadata.aggregation=None` significa "o modelo usa seu default documentado"
(para `MixedFreqDFM`/trimestrais: `GROWTH_RATE`).

---

## 4. `core.data`

### `SeriesMetadata` (dataclass congelada)

| Campo | Tipo | Significado |
|---|---|---|
| `name` | `str` | nome da série |
| `frequency` | `Frequency` | frequência nativa |
| `transform` | `int \| str \| None` | código 0–7 ou nome da transformação (`preprocessing`); ver `transform_applied` |
| `release_delay` | `int \| None` | defasagem de divulgação em **dias** após o fim do período de referência; negativa (> −366) quando o valor sai antes do fim do período (pesquisas regionais, onda 3) |
| `blocks` | `tuple[str, ...]` | blocos de fatores (ex.: `("global", "real")`) |
| `category` | `SeriesCategory \| None` | `HARD`, `SOFT`, `FINANCIAL`, `OTHER` |
| `description` | `str` | descrição |
| `aggregation` | `AggregationType \| None` | relação com a série latente de alta frequência |
| `units` | `str` | unidade |
| `transform_applied` | `bool` (default `False`) | `False`: `transform` é a transformação **a aplicar** (valores em nível, como numa legenda); `True`: os valores **já estão transformados** e `transform` registra o que foi aplicado. `apply_transforms`/`prepare_panel` marcam `True`; com `transform=None` séries já transformadas não são transformadas de novo (idempotente); `invert_transforms` usa o registro e volta para `False`. Adicionado na integração da onda 1. |

`.replace(**changes)`, `.to_dict()`.

### `MixedFrequencyData`

Construtor:

```python
MixedFrequencyData(
    data: pd.DataFrame,                 # PeriodIndex (ou DatetimeIndex + base_frequency)
    frequencies=None,                   # dict | Series (ex.: legenda 12/4) | lista alinhada | escalar
    *, metadata=None,                   # {nome: SeriesMetadata | dict}
    transforms=None, release_delays=None, blocks=None,   # blocks: dict ou DataFrame 0/1 (série x bloco)
    categories=None, descriptions=None, aggregations=None,
    base_frequency=None,
)
MixedFrequencyData.from_series({"gdp": serie_trimestral, "ip": serie_mensal}, base_frequency="M", ...)
as_mixed_frequency_data(data, frequency=None, **kw)   # coerção canônica usada por todo fit()
```

Precedência: argumentos por campo (`release_delays=...`) sobrescrevem `metadata`;
séries sem frequência informada têm a frequência inferida (`infer_frequency`).
Metadados para séries inexistentes → `NowcastDataError`.

Propriedades: `base_frequency`, `index`, `columns`, `n_series`, `n_periods`, `shape`,
`start`, `end`, `data` (cópia), `values` (cópia ndarray T×N), `metadata`
(`dict[str, SeriesMetadata]`), `frequencies`, `transforms`, `release_delays` (`Int64`),
`categories`, `block_names`, `blocks` (DataFrame bool série×bloco), `is_mixed_frequency`,
`monthly_columns`, `quarterly_columns`.

Consultas: `columns_with_frequency(freq)`, `series_by_frequency()` (da maior para a
menor frequência), `frequency_ratio(col)`, `observation_mask()`, `slot_mask()`,
`missing_mask()` (slot sem valor), `ragged_edge_mask()` (slots após a última
observação de cada série), `last_observed()`, `n_observations()`, `to_native(col)`
(série no índice nativo, ex. trimestral), `to_frame()`, `metadata_frame()`.

Derivação (sempre retorna novo objeto validado): `copy()`, `select(cols)`, `drop(cols)`,
`truncate(start, end)` (aceita `"2020Q1"`: início → jan, fim → mar), `extend(n)`
(acrescenta períodos vazios — horizonte de previsão), `with_data(df)` (novos valores,
mesmos metadados — usado por transformações), `with_metadata(col, **changes)`,
`as_of(vintage, release_delays=None)`, `standardize(stats=None, ddof=1) -> (mfd, stats)`,
`destandardize(stats)`, `standardization_stats(ddof)`, `equals(other, atol)`.
Blocos não usados por nenhuma série são removidos em `select`/`with_metadata`.

Paridade ECB (0.2.0): `released_share(period, *, by=None, weights=None, series=None,
as_of=None) -> DataFrame` — por série/grupo (`"category"`, `"block"`, `"frequency"` ou
mapeamento `{série: grupo ou [grupos]}`) e linha `"total"`, colunas `released`,
`expected`, `weight`, `share`; observações esperadas = *slots* da série dentro do período
(períodos-base além do fim da grade contam como esperados e não divulgados).

#### Regra de divulgação (`as_of`)

Observação do período nativo `p` é divulgada em `fim(p) + release_delay` dias e é mantida
se essa data ≤ `vintage` (inclusivo). Sem revisões (vintages reais ficam em
`nowcastbox.vintages.VintageStore`). Grade inalterada. Série sem `release_delay` →
`NowcastDataError`.

### `StandardizationStats` (dataclass congelada)

`mean: pd.Series`, `std: pd.Series` (> 0), `ddof: int`; `transform(df)`,
`inverse_transform(df)`, `inverse_series(values, column, scale_only=False)` —
`scale_only=True` para desvios-padrão, erros e impactos de *news* (sem a média).

---

## 5. `core.results`

### `NowcastResults` (dataclass `frozen=True, kw_only=True`)

| Campo | Tipo | Conteúdo |
|---|---|---|
| `target` | `str` | alvo |
| `nowcast` | `DataFrame` | índice = `PeriodIndex` **na frequência nativa do alvo** (ex. trimestral); colunas obrigatórias `observed`, `in_sample`, `out_of_sample` (+ extras como `std`, `lower_68`, `upper_68`, `lower_90`, `upper_90`) |
| `model_name` | `str` | ex. `"TwoStepDFM"` |
| `model_params` | `dict` | `estimator.get_params()` |
| `factors` | `DataFrame \| None` | fatores suavizados na grade base, colunas `f1, f2, …` (para blocos: `"<bloco>_f1"`) |
| `loadings` | `DataFrame \| None` | linhas = séries, colunas = fatores (mesmos nomes de `factors`) |
| `params` | `dict` | parâmetros estimados específicos do modelo (matrizes etc.) |
| `loglikelihood` | `float \| None` | |
| `n_iter`, `converged` | `int \| None`, `bool \| None` | |
| `data` | `MixedFrequencyData \| None` | painel usado na estimação |
| `standardization` | `StandardizationStats \| None` | estatísticas usadas |
| `info` | `dict` | extras (tempo, diagnósticos) |

Semântica de `nowcast`: para cada período com estimativa, **exatamente uma** de
`in_sample` (alvo observado) ou `out_of_sample` (backcast/nowcast/forecast) é preenchida.
Use `build_nowcast_frame(observed, estimate, extra=None)` para montá-lo.

Métodos: `target_frequency`, `observed`, `in_sample`, `out_of_sample`, `estimate`
(combinação), `n_factors`, `get_nowcast(period=None)` (padrão: 1º período após a última
observação do alvo), `replace(**changes)`, `to_frame()`, `summary(n_periods=8)`,
`plot(kind, **kw)`, `save(path)`, `load(path)` (pickle; somente arquivos confiáveis).

Extensão de `summary`: sobrescrever `_summary_sections()` retornando
`[(titulo, [linhas])]` (chamar `super()`).

### `FactorResults(NowcastResults)`

Campos extras: `transition` (`[A_1 … A_p]`, shape `(r, r·p)`), `shock_loadings` (`B`,
`(r, q)`), `factor_lags` (`p`), `n_shocks` (`q`), `idiosyncratic_variance` (diag. de Ψ,
`Series` indexada por série). `transition_matrices() -> [A_1, …, A_p]`.
Resultados do two-step e do EM devem herdar de `FactorResults` (ou de `NowcastResults`
para modelos não fatoriais como bridge).

### Registro de gráficos

```python
from nowcastbox.core.results import register_plot, NowcastResults

@register_plot("forecast")                    # vale para NowcastResults e subclasses
def plot_forecast(results, backend="plotly", **kwargs): ...

@register_plot("loadings", FactorResults)     # específico de uma subclasse (sobrepõe)
def plot_loadings(results, **kwargs): ...
```

`results.plot(kind)` procura pelo MRO; se não achar, importa `nowcastbox.visualization`
(que deve registrar seus plots no import) e tenta novamente; senão `NotImplementedError`.
Tipos registrados (integração da onda 2; `nowcastbox/visualization/registry.py`):

| `kind` | Classe | Função |
|---|---|---|
| `"forecast"` | `NowcastResults` | `plot_forecast` (observado, in/out-of-sample, faixas 68/90 %) |
| `"fan"` | `NowcastResults` (precisa de `std`) | `plot_fan_chart` (quantis gaussianos da coluna `std`) |
| `"density"` | `NowcastResults` (precisa de `std`) | `plot_fan_chart(results.distribution(...))` (I5; aceita `n_boot`); com `method="empirical", backtest=...`, `plot_empirical_bands` nos níveis das bandas (57,5/68/90 %) |
| `"data_availability"` / `"ragged_edge"` | `NowcastResults` (precisa de `data`) | `plot_data_availability` |
| `"released_share"` | `NowcastResults` (precisa de `data`) | `plot_released_share` (parcela divulgada de um período-alvo; 0.2.0) |
| `"indicator_heatmap"` | `NowcastResults` (precisa de `data`) | `plot_indicator_heatmap` (z-scores dos indicadores; 0.2.0) |
| `"factors"`, `"eigenvalues"`, `"loadings"` | `FactorResults` | `plot_factors`, `plot_eigenvalues`, `plot_loadings` (`style=` heatmap/bar) |
| `"loglikelihood"` | `MixedFreqDFMResults` | `plot_loglikelihood` |

Todas aceitam `backend="plotly"` (padrão) ou `"matplotlib"` e `theme=`. Os objetos de
*news* não são `NowcastResults`: têm registro próprio (`nowcastbox.news.register_news_plot`)
com os tipos `"waterfall"` (`NewsResults`), `"path"` (`NowcastTracker`) e `"bar"`
(`LevelContributions`); `"waterfall"`/`"path"` usam o gráfico Matplotlib de `news` por
padrão e o de `visualization` com `backend=` ou `theme=`.

### Métodos de análise (integração da onda 2)

`NowcastResults` delega, com import preguiçoso (os subpacotes dependem de `core`, nunca o
contrário em tempo de execução):

| Método | Delegação | Retorno |
|---|---|---|
| `news(old, new, target_period=None, **kw)` | `nowcastbox.news.news_decomposition` | `NewsResults` |
| `nowcast_tracker(data, calendar=None, target_period=None, start=None, end=None, **kw)` | `nowcastbox.news.nowcast_tracker` | `NowcastTracker` |
| `level_contributions(data=None, target_period=None, **kw)` | `nowcastbox.news.level_contributions` | `LevelContributions` |
| `distribution(**kw)` | `nowcastbox.density.nowcast_distribution`; com `method="empirical"` (exige `backtest=`; `empirical_method=` vira `method`), `nowcastbox.density.empirical_bands` | `NowcastDistribution` (`EmpiricalGaussianDistribution` para `mae`/`rmse`) ou `EmpiricalQuantileDistribution` |
| `diagnostics(data=None, **kw)` | `nowcastbox.diagnostics.run_diagnostics` | `DiagnosticsReport` |

O `NowcastTracker` (0.2.0) tem a coluna `released_share`: parcela das observações do
período-alvo dos preditores do modelo (colunas de `results.data` menos o alvo)
divulgadas em cada vintage.

`news`/`nowcast_tracker`/`level_contributions` exigem modelos de espaço de estados
suportados (`MixedFreqDFM` em qualquer grade, inclusive de calendário; `TwoStepDFM` com
`aggregate="factors"` ou `"variables"` — neste caso cada vintage passa pelos filtros
ajustados, `TwoStepResults.filter_panel`; `TypeError` nos demais). O bootstrap
paramétrico da densidade cobre os mesmos modelos; o bootstrap por blocos exige razão
fixa entre a grade base e o alvo. `nowcastbox.news.attach(cls)` continua disponível para
classes de resultados *duck-typed* (não sobrescreve métodos nativos).

`MixedFreqDFMResults.state_space_model(n_periods=None) -> (StateSpace, StateLayout, grid)`
é o acesso público ao modelo estimado (usado por `news`, `density`, `diagnostics`); para
agregações de calendário (grades semanais/diárias) a equação de observação é variante no
tempo (`StateSpace(obs_index=...)`) e `n_periods` reconstrói o modelo para outra extensão
de grade (use `designs(n)`/`design_at(t)`, não `Z`). `params["design"]` é 3-D
`(n, N, m)` nesses modelos e `params["aggregation_weight_paths"]` guarda os pesos por
período (DataFrame por série). `info["initialization"]` registra o ponto de partida do EM
(`init="pca"`: com vários blocos, melhor log-verossimilhança inicial entre as ordens de
`BLOCK_ORDERS`). Convenção de pré-amostra (onda 3): períodos do alvo anteriores à sua
primeira observação não têm `out_of_sample` (o EM mantém a estimativa em `common`).
Campos adicionados
na onda 2: `observation_weights`, `student_t_df`, `outlier_flags`,
`excluded_observations`, `interventions`, `state_offset`, `long_run_mean` (I3/I4); coluna
`long_run_mean` em `nowcast`; `filter_method` (padrão `"auto"`: suavizador estruturado
exato no E-step e na suavização final quando mais barato, I2).

---

## 6. `core.base`

### `ParamsMixin`

Estilo scikit-learn: todo argumento de `__init__` é guardado como atributo homônimo,
sem validação no `__init__`, sem `*args/**kwargs`. Fornece `get_params(deep=True)`,
`set_params(**p)` (aninhado `inner__param`; descarta estado ajustado), `clone()`, `__repr__`.

### `BaseNowcaster(ParamsMixin, ABC)`

```python
class MeuModelo(BaseNowcaster):
    def __init__(self, n_factors: int = 1, factor_lags: int = 1, tol: float = 1e-4):
        self.n_factors = n_factors; self.factor_lags = factor_lags; self.tol = tol

    def _validate_params(self) -> None: ...        # levanta ValueError
    def _fit(self, data: MixedFrequencyData, target: str, **fit_kwargs) -> NowcastResults: ...
```

`fit(data, target, *, frequency=None, **fit_kwargs)`:
1. `_reset()` e `_validate_params()`;
2. `as_mixed_frequency_data(data, frequency)`;
3. `target` pode ser nome **ou fórmula** (`"gdp ~ ."`, `"gdp ~ ip + pmi"`, `"gdp ~ . - x"`);
   o painel passado a `_fit` contém só o alvo e os preditores (ordem original das colunas);
4. `_fit` deve retornar `NowcastResults` (senão `TypeError`); fica em `results_`.

Atributos: `is_fitted`, `results_` (`ModelNotFittedError` antes do fit), `_check_fitted()`.
Métodos adicionais (ex. `news`, `predict`) devem chamar `self._check_fitted()`.

### `BaseBenchmark(ParamsMixin, ABC)` e `BenchmarkForecaster` (Protocol)

```python
bench.fit(data, target, *, frequency=None) -> self   # usa só a informação do vintage
bench.predict(periods) -> pd.Series                  # períodos na frequência nativa do alvo
```

Subclasses implementam `_fit(data, target) -> None` e `_predict(periods: PeriodIndex)
-> Series` (índice exatamente igual a `periods`). Atributos: `name`, `is_fitted`,
`target_`, `target_frequency_`. Qualquer objeto com `fit`/`predict` compatíveis satisfaz
`BenchmarkForecaster` (I11, adaptadores scikit-learn).

---

## 7. `core.formula`

`parse_formula(formula, columns=None) -> ParsedFormula(target, regressors, excluded, uses_dot)`;
`ParsedFormula.resolve(columns)`; `resolve_target(target_or_formula, columns) -> (target, regressors)`;
`is_formula(text)`. Gramática: `alvo ~ termo (+|- termo)*`, `termo = . | nome | `nome com espaço``.
Sem interações/transformações.

---

## 8. Recomendações para módulos numéricos (não vinculantes ao core)

- **statespace**: notação de Durbin & Koopman (2012), como no kalmanbox:
  `α_{t+1} = T α_t + R η_t`, `η_t ~ N(0, Q)`; `y_t = Z α_t + ε_t`, `ε_t ~ N(0, H)`.
  Observações como `ndarray (n_periods, n_series)` com `NaN` = faltante. Funções puras
  (NumPy/Numba), sem pandas no caminho crítico.
- **models**: trabalham com `data.standardize()`; fatores/cargas em DataFrames rotulados;
  resultados em unidades originais via `StandardizationStats.inverse_series`.
- **preprocessing**: transformações recebem/retornam `MixedFrequencyData` (via `with_data`)
  ou `DataFrame` na grade base, preservando a convenção de slots.
- **vintages/news/evaluation**: usar `MixedFrequencyData.as_of` como gancho de tempo
  pseudo-real; datas de divulgação = fim do período nativo + `release_delay` dias.

---

## 9. Mapa de módulos (plano §5) — estado após a integração da onda 3

| Subpacote | Arquivos | Conteúdo | Depende de | Testes |
|---|---|---|---|---|
| `core/` | `base.py`, `results.py`, `frequency.py`, `data.py`, `formula.py`, `exceptions.py` | contratos (este documento) — **fundação**; métodos de análise de `NowcastResults` (§5) | — (imports preguiçosos de news/density/diagnostics) | `tests/core/` |
| `_logging.py` | — | `get_logger`, `set_log_level` — **fundação** | — | `tests/test_logging.py` |
| `preprocessing/` | `transforms.py`, `outliers.py`, `missing.py`, `aggregation.py`, `panel.py` | códigos 0–7 e transformações nomeadas inversíveis, outliers (IQR + mediana móvel), NAs fora da borda, Mariano-Murasawa, agregação de calendário (`CalendarAggregation`, `calendar_weight_matrix`), `month_to_quarter`/`quarter_to_month`, `prepare_panel` | core | `tests/preprocessing/` |
| `statespace/` | `representation.py`, `kalman.py`, `_kernels.py`, `smoother.py`, `collapse.py`, `simulate.py`, `structure.py`, `structured.py`, `_band_kernels.py` | Kalman com NAs (univariado, Numba; T esparsa), log-verossimilhança, RTS + cov. lag-1, `StateSpace(obs_index=...)` com Z/H/d variantes no tempo, inicialização difusa aproximada, **suavizador estruturado exato** (`structured_smoother`, `smoothed_moments`, `detect_structure`; I2) | core (exceções) | `tests/statespace/`, `tests/performance/` |
| `models/` | `two_step.py`, `em.py`, `_em_steps.py`, `bridge.py`, `_init_conditions.py`, `_pca.py`, `robust.py`, `long_run.py` | `TwoStepDFM` (+ `model_data`/`filter_panel`/`prefiltered`), `MixedFreqDFM` (EM, blocos, AR(1)/iid/Student-t, outliers, COVID mask/dummy, média de longo prazo variante; E-step estruturado; grades de calendário; GEM monótono; multi-start por ordem dos blocos), `BridgeEquation` | core, statespace, preprocessing, selection | `tests/models/` |
| `selection/` | `bai_ng_factors.py`, `bai_ng_shocks.py`, `targeted.py`, `blocks.py`, `_panel.py`, `_plot.py` | `select_factors`, `select_shocks`, preditores *targeted*, `select_blocks`/`select_variables` (I7, RMSFE pseudo tempo-real) | core, evaluation (preguiçoso) | `tests/selection/` |
| `vintages/` | `pseudo_real_time.py`, `calendar.py`, `vintage_store.py`, `_utils.py` | `pseudo_real_time`, `ReleaseCalendar`, `VintageStore.as_of` | core | `tests/vintages/` |
| `news/` | `decomposition.py`, `revisions.py`, `contributions.py`, `tracker.py`, `_linear.py`, `_model.py`, `plotting.py` | *news* (Bańbura-Modugno) por série/bloco/categoria/divulgação, revisões, reestimação, *nowcast tracker*, contribuições ao nível, suavizador linear em lote, registro de plots próprio | core, statespace, models (resultados) | `tests/news/` |
| `density/` | `distribution.py`, `bootstrap.py`, `predictive.py`, `empirical.py` | `NowcastDistribution` (gaussiana/mistura), bootstrap paramétrico e por blocos com reestimação, `nowcast_distribution` (I5); bandas de erro empírico (`empirical_bands`, `EmpiricalGaussianDistribution`, `EmpiricalQuantileDistribution`, Reifschneider-Tulip/BCE; 0.2.0) | core, statespace, models | `tests/density/` |
| `benchmarks/` | `ar.py`, `random_walk.py`, `mean.py`, `bridge.py`, `midas.py`, `sklearn_adapter.py`, `_base.py`, `_utils.py` | AR(p), RW, média, bridge, U-MIDAS, MIDAS (Almon/Beta), adaptador scikit-learn (I11) — `BaseBenchmark` | core, models.bridge (`BridgeEquation`, `ar_extend`, `aggregate_to_target`, `resolve_aggregation_weights`) | `tests/benchmarks/` |
| `evaluation/` | `backtest.py`, `metrics.py`, `tests.py`, `scoring.py` | `PseudoRealTimeBacktest` / `BacktestResults`, RMSFE/MAE por horizonte, DM (HLN), GW, MCS, Clark-West; FDA + Pesaran-Timmermann (coluna opcional `previous_actual`, `previous="vintage"\|"final"`) e `periods=` (subperíodos, atalhos `"covid"`/`"ex-covid"`) nos métodos de `BacktestResults` (0.2.0); scores de densidade (CRPS, log score, PIT, Berkowitz, KS, cobertura, Christoffersen, quantile score) | core, vintages, benchmarks (`SklearnBenchmark`) | `tests/evaluation/` |
| `diagnostics/` | `stability.py`, `convergence.py`, `contribution.py`, `data_quality.py`, `residuals.py`, `report.py`, `zscores.py`, `_common.py` | estabilidade das cargas (Breitung-Eickmeier), convergência do EM, contribuição dos fatores, qualidade dos dados, Ljung-Box/Jarque-Bera, `run_diagnostics -> DiagnosticsReport` (I9); `indicator_zscores -> IndicatorZScores` (z-scores com suavização Mariano-Murasawa, por série/grupo, `as_of=`; 0.2.0) | core, preprocessing (outliers, pesos de agregação), models (resultados, *duck typing*) | `tests/diagnostics/` |
| `data_sources/` | `bcb.py`, `ibge.py`, `ipea.py`, `fred.py`, `cache.py`, `_http.py`, `_parsing.py` | conectores HTTPS com cache (testes com mocks; rede marcada `network`); SIDRA com `dash_as` | core | `tests/data_sources/` |
| `datasets/` | `dataset.py`, `load.py`, `_io.py`, `_simulated.py`, `metadata/*.yaml`, `data/*.csv.gz` | `Dataset` + loaders (`load_brazil_nowcast`, `load_brazil_calendar`, `load_brazil_vintages`, `load_us_fred_md`, `load_nyfed`, `load_us_grs_like`, `load_simulated_dfm`); scripts em `scripts/build_datasets/` | core, vintages | `tests/datasets/` |
| `simulate/` | `dfm.py` | `dfm()` (mensal + trimestral) e `weekly_dfm()` (semanal + mensal + trimestral) → `SimulatedDFM(data, truth)` | core, datasets, preprocessing | `tests/simulate/` |
| `visualization/` | `themes.py`, `_common.py`, `forecast.py`, `factors.py`, `data_flow.py`, `heatmap.py`, `selection.py`, `news.py`, `evaluation.py`, `diagnostics.py`, `registry.py` | Plotly + Matplotlib, temas; registra os plots de §5 (resultados e objetos de *news*) | core, models (classes de resultado), news (registro) | `tests/visualization/` |
| `reports/` | `html.py`, `templates/nowcast_report.html` | `NowcastReport` (Jinja2 + Plotly): manchete, trajetória, *news*/*tracker*, fluxo de dados, diagnósticos (`diagnostics=True`); 0.2.0: `bands=`, `alternatives=`, `heatmap=`, `backtest_metrics=`, parcela de dados divulgados | core, visualization, diagnostics (preguiçoso) | `tests/reports/` |
| `experiment/` | `experiment.py`, `alternatives.py` | `NowcastExperiment` (comparação; backtest padrão = `PseudoRealTimeBacktest`); `alternative_models -> AlternativeNowcasts` (sem 1–2 grupos, `refit=True\|False`; 0.2.0) | core, models, evaluation (preguiçoso), visualization | `tests/experiment/` |
| `pipeline/` | `spec.py`, `data.py`, `runner.py`, `snapshots.py`, `examples.py`, `templates/*.yaml`, `templates/example.xlsx` | `NowcastSpec`/`load_spec` (YAML validado), `run_pipeline` → `PipelineRun`, `SnapshotStore` (snapshots versionados, histórico, diff), modelos de spec (I10); 0.2.0: fonte `excel` (extra `[excel]`), saídas `empirical_bands`, `heatmap`, `alternatives`, `excel`, `backtest.metrics`/`backtest.periods` | todos | `tests/pipeline/` |
| `cli/` | `main.py`, `__main__.py` | `nowcastbox run|validate|init|datasets|snapshots` (códigos de saída 0/1/2) | pipeline | `tests/cli/` |
| `api.py` | — | `nb.nowcast(..., idiosyncratic=, long_run_mean=, outliers=, density=, n_boot=)`, `add_density`; `preprocess=` não limpa o alvo (salvo `clean_target=True`); grades de calendário só com `method="em"` | models, selection, preprocessing, density | `tests/test_api.py` |
| `__init__.py` (topo) | — | API pública (subpacotes + entradas principais) — **somente etapa de integração** | todos | `tests/test_package.py`, `tests/test_api.py` |

Regras de trabalho em paralelo: criar/modificar arquivos apenas no(s) diretório(s)
atribuído(s); registrar os artigos usados em `desenvolvimento/FONTES_POR_MODULO.md`
(regra *clean-room* MIT: nunca ler/traduzir o código do pacote R `nowcasting` nem copiar
código do statsmodels).

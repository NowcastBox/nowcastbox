# NowcastBox — Status do desenvolvimento

> Atualizado na **integração da onda 3 (final)** (2026-10-01). Plano:
> `PROJETO_DESENVOLVIMENTO.md` (§12 roadmap). Contratos: `CONTRATOS.md`. Fontes:
> `FONTES_POR_MODULO.md`. Versão: `0.1.0.dev0` (candidata a 0.1.0; nada publicado no PyPI).

## Resumo

| Item | Valor |
|---|---|
| Testes (sem rede) | **3260 passaram, 0 falharam** (`python3 -m pytest -q -x`, inclui `slow`: exemplos da documentação, README e os 15 notebooks) |
| Pulados | 23 (com `-n 12 --cov`): 7 de rede (`--run-network`; 1 exige `FRED_API_KEY`), 15 páginas de docs sem Python, 1 de desempenho (razão de tempo vs. statsmodels pulada sob cobertura/xdist; passou na execução serial `python3 -m pytest -q -x`: 3259 passaram, 22 pulados, 7 min 40 s) |
| Cobertura total (branch) | **99,87 %** (`--cov=nowcastbox`; mínimo do projeto 90 %, núcleo 95 %) |
| `ruff check` / `ruff format` | limpos (`nowcastbox/`, `tests/`, `benchmarks/`, `scripts/`, `examples/`; papers no repositório `nowcastbox-papers`) |
| `pyright nowcastbox/` (modo `standard`) | 0 erros, 0 avisos |
| `interrogate` (docstrings) | 99,5 % (mínimo 95 %) |
| `bandit -c pyproject.toml -r nowcastbox -ll` | 0 achados médios/altos (1 `Markup` de HTML gerado pelo Plotly justificado com `# nosec B704`) |
| `radon cc nowcastbox -a` | média **A (3,7)** em 1922 blocos; nenhum bloco D–F |
| `mkdocs build --strict` | limpo; nav inclui Datasets e Validation completos |
| Notebooks (`examples/notebooks`) | 15/15 reexecutados com saídas salvas (`python3 examples/utils/build_notebooks.py`, 4–28 s cada); `tests/integration/test_notebooks.py` (slow) e job de CI |
| Paper JSS | `replicate_all.py --latex` regenerou tabelas/figuras (incl. comparação com o R e o novo `exp07_weekly`); PDF compilado com `pdflatex`+`bibtex` |

## Andamento por fase do roadmap (§12)

| Fase | Situação | Detalhes |
|---|---|---|
| 0. Fundação | **Concluída** | `pyproject.toml` (MIT), CI (testes, qualidade, docs, exemplos, validação de referência semanal, release), pre-commit, MkDocs, contratos; scripts de datasets (`scripts/build_datasets/`) e de fixtures de referência (`scripts/reference_fixtures/`, R como caixa-preta). |
| 1. Dados | **Concluída** | `MixedFrequencyData` com grades mensal/semanal/diária e agregação de calendário (I1), transformações nomeadas e inversíveis, outliers, NAs, vintages pseudo tempo-real (atrasos negativos aceitos). |
| 2. Núcleo de estado | **Concluída** | Kalman univariado/multivariado com NAs (Numba), suavizador, colapso, Z/H/d variantes no tempo, inicialização difusa aproximada, suavizador estruturado exato (I2) com heurística `"auto"` recalibrada. Falta: difusa exata; comparação com kalmanbox. |
| 3. Two-step | **Concluída** | `TwoStepDFM` (`"factors"`/`"variables"` com `model_data`/`filter_panel`), bridge, Bai-Ng, plots; notebooks 01–04. Release PyPI **não** feito (fora do escopo desta onda). |
| 4. EM | **Concluída** | `MixedFreqDFM` com blocos, GEM monótono com o termo inicial exato, multi-start por ordem dos blocos, grades de calendário; notebooks 05–06. |
| 5. News & avaliação | **Concluída** | `news/` (também calendário e two-step `"variables"`), `evaluation/` (`model_name=`, MCS por horizonte corrigido), `benchmarks/`, `density/`; notebooks 07–08, 13. |
| 6. Robustez | **Concluída** | I3, I4, I9 e I7 (`select_blocks`/`select_variables`); notebook 10. |
| 7. Brasil | **Concluída (I8 parcial)** | `data_sources/`, `load_brazil_nowcast` (99 séries), calendário real do PIB, vintages do PIB/IBGE; notebooks 09, 14. Faltam vintages do IBC-Br e conector BCB Olinda (Focus). |
| 8. Produção | **Concluída** | `pipeline/` (spec YAML, `run_pipeline`, snapshots), CLI `nowcastbox`, relatório HTML, `NowcastExperiment`; notebooks 11–12, 15. |
| 9. Estabilização | **Em andamento** | Auditoria de qualidade desta integração (acima), docs completas, README, CHANGELOG 0.1.0 (rascunho). Faltam: DOI Zenodo, publicação no PyPI, submissão do paper, `mutmut`. |

## Inovações I1–I11 (situação final desta versão)

| # | Situação | Onde |
|---|---|---|
| I1 Frequências arbitrárias | **Concluída no EM** | `core/frequency.py` (`is_fixed_ratio`, `calendar_position`, `AggregationType.calendar_weights`...), `preprocessing/aggregation.py` (`CalendarAggregation`), `MixedFreqDFM` em grades semanais/diárias; *news*, tracker, contribuições, densidade paramétrica e diagnósticos também. `TwoStepDFM`, `BridgeEquation`, benchmarks e MIDAS continuam só com razão fixa (M/Q/A; erro explícito em `nb.nowcast(method="two_step")`). |
| I2 EM escalável | **Atingida** | Ver Benchmark. |
| I3 Robustez | **Concluída** | `models/robust.py`. |
| I4 Média de longo prazo | **Concluída** | `models/long_run.py`. |
| I5 Densidade | **Concluída** | `density/` (paramétrico também para calendário e two-step `"variables"`), `evaluation/scoring.py`. |
| I6 Explicabilidade | **Concluída** | `news/` (modelos de calendário e two-step `"variables"` desde a onda 3). |
| I7 Seleção de variáveis | **Concluída** | `selection/targeted.py`, `selection/blocks.py` (`select_blocks`, `select_variables`, RMSFE pseudo tempo-real ou critério de informação). |
| I8 Vintages reais | **Parcial** | `VintageStore`, ALFRED, vintages do PIB/IBGE; falta IBC-Br. |
| I9 Diagnósticos | **Concluída** | `diagnostics/` (pesos por período em grades de calendário; resíduos filtrados no two-step `"variables"`). |
| I10 Produção | **Concluída** | `pipeline/`, `cli/`, `reports/`, `experiment/`. |
| I11 Interoperabilidade | **Concluída** | `benchmarks.SklearnBenchmark`, `BacktestResults.to_parquet`. |

## Benchmarks (re-medidos na integração da onda 3, máquina ociosa, 1 thread BLAS)

**I2** — `python3 benchmarks/bench_em.py` (N = 200: 180 mensais + 20 trimestrais, T = 300,
1 fator, AR(1), Mariano-Murasawa, 285 estados):

| Medida | nowcastbox (`"auto"`, estruturado) | nowcastbox denso (`"univariate"`) | statsmodels `DynamicFactorMQ` |
|---|---:|---:|---:|
| Segundos por iteração EM | **0,373** | 4,728 (12,7×) | 5,026 (**13,5×**) |
| `fit` completo | 1,96 s (3 iterações) | 28,01 s | 28,85 s (`fit_em`, 4 iterações) |

Log-verossimilhanças denso × estruturado: 9,5e-11; nowcasts: 3,9e-12. O teste
`tests/performance` (≥ 4× por iteração em N = 100) é pulado sob cobertura e dentro de
workers do `pytest-xdist` (concorrência distorce o tempo marginal do statsmodels).

**R `nowcasting` 1.1.2** — `python3 benchmarks/bench_vs_r.py --repeat 3` (tempos do R da
fixture, máquina ociosa): EM NYFED 56,4 s (R) vs 1,61 s (**35×**); two-step USGDP 3,9–4,3×;
Bai-Ng IC 30×; vintage pseudo tempo-real 0,060 s vs 0,007 s (**8×**, antes 0,4×); limpeza
do painel 1,9–2,6×. Tabela completa em `docs/validation/performance.md`.

**statsmodels `DynamicFactorMQ`** (NY Fed-like, 4 blocos, sem ruído de medida, tol 1e-6):
mesma função de verossimilhança (1e-12); ótimo do statsmodels −9347,64; nowcastbox
−9393,68 com o multi-start (antes −9765,28 com o início sequencial).

## Integração da onda 3 — o que foi feito

- **Frequências (I1) nos módulos de análise**: `news` (`batched_functional` com Z/d/H
  variantes; `LinearNowcastModel.model_for(n)`/`gain_at`; slot do alvo pela convenção de
  calendário), `density` (simulação e momentos com o modelo de calendário; bootstrap por
  blocos recusado com erro explícito), `diagnostics` (pesos por período,
  `aggregation_weight_paths`), `nb.nowcast` (erro explícito para two-step em grades de
  calendário), `preprocessing._utils` (`native_to_base`). Notebook 14 ganhou a seção de
  *news* semanal.
- **Lacuna 7 resolvida**: *news*/tracker/contribuições, bootstrap paramétrico e resíduos
  dos diagnósticos para `TwoStepDFM(aggregate="variables")` (painel filtrado).
- **EM multi-start** (achado da validação de referência): `init="pca"` avalia 4 ordens de
  blocos (`BLOCK_ORDERS`) pela log-verossimilhança inicial; `info["initialization"]`;
  `init="pca_<ordem>"` força uma. NY Fed: ótimo +372 pontos; distância ao statsmodels de
  418 para 46 pontos.
- **Vintages**: atrasos negativos (> −366 dias; NYFED com as 25 séries idêntico ao
  `PRTDB`); máscara de divulgação ~18× mais rápida; `delay` como Series nomeada com
  séries extras.
- **Núcleo**: `slot_mask`/validação de slots calculados uma vez por frequência;
  exportações de `core`/`preprocessing`.
- **Heurística `filter_method="auto"`** recalibrada com tempos do E-step (custos fixos
  por chamada e por grupo): denso até N ≈ 35, estruturado a partir de N ≈ 50 (T = 300).
- **API**: `nb.nowcast(preprocess=...)` não limpa mais o alvo (`clean_target=True`
  restaura); pré-amostra do alvo sem `out_of_sample` (EM e two-step);
  `PseudoRealTimeBacktest(model_name=)`; `mcs(horizon="months_to_end")` corrigido;
  `nb.simulate.dfm`/`weekly_dfm`; topo exporta pipeline/CLI/simulate/seleção de blocos.
- **Pipeline**: templates embutidos resolvem caminhos no diretório de trabalho (nunca
  dentro do pacote); `snapshot=False` ainda lê o snapshot anterior para *news*;
  `package-data` inclui `pipeline/templates/*.yaml`.
- **Testes novos**: `tests/news/test_calendar_news.py`, `tests/density/test_calendar_density.py`,
  `tests/diagnostics/test_calendar_diagnostics.py`, `tests/simulate/`,
  `tests/integration/test_notebooks.py`, `tests/docs/test_readme.py`, além de casos em
  `test_em.py`, `test_two_step.py`, `test_decomposition.py`, `test_report.py`,
  `test_backtest.py`, `test_calendar.py`, `test_api.py`, `test_runner.py`, `test_spec.py`.
- **Docs, README, CHANGELOG (0.1.0, rascunho), CONTRATOS**, CI (job de exemplos e
  notebooks; validação de referência com os scripts na ordem documentada), paper JSS
  (tabelas do R a partir das fixtures, `exp07_weekly`, PDF recompilado).

## Lacunas conhecidas / decisões registradas

1. **I1 fora do EM**: `TwoStepDFM`, `BridgeEquation`, benchmarks (`resolve_aggregation_weights`
   → `aggregation_ratio`) e MIDAS exigem pares de razão fixa (M/Q/A). O bootstrap por
   blocos da densidade também (unidades de comprimento variável). O rótulo
   `months_to_end` do backtest é em meses de calendário (use `days_to_end` em grades
   semanais/diárias).
2. **Inicialização difusa exata** não implementada (a aproximada existe).
3. **Suavizador estruturado**: custo O(N T² w) + O((T w)³); exige H diagonal estritamente
   positiva (`obs_noise_var=0` usa o denso). A heurística `"auto"` foi calibrada numa
   máquina (WSL2); em outras, force o caminho se necessário.
4. **BLAS multithread**: resolvida a documentação (`docs/user-guide/performance.md`); a
   suíte limita a 1 thread.
5. **Códigos legados 1–4** usam defasagem de 1 período **nativo**; códigos 3–7 e nomes
   usam janelas de calendário.
6. **Bai-Ng — resolvida**: defaults conferidos com os artigos (VAR(2), normalização
   Λ'Λ/N = I, `m` por matriz/estatística, aviso em `select_factors` quando
   min(N, T) < 20). `ICfactors` idêntico a `select_factors` (18/18 r*, 6e-15);
   `ICshocks` do R usa o limite m/min(N,T)^{1/(2−δ)} (diverge de Bai & Ng, 2007);
   nowcastbox mantém o do artigo (2 de 9 casos divergentes documentados).
7. **Two-step `"variables"` — resolvida** (news, densidade paramétrica, resíduos). Em
   `aggregate="factors"`, divulgações do próprio alvo têm impacto zero em `news` (entra
   só pela ponte).
8. **Monotonicidade do EM — resolvida**: passo GEM com meia-passo contra a esperança
   exata da log-verossimilhança completa incluindo o termo inicial estacionário; teste
   de propriedade a 1e-8 relativo; `info["n_loglikelihood_decreases"]` permanece.
   `DynamicFactorMQ.fit_em(...).llf` (statsmodels 0.14) difere de `loglike(params)`.
9. **Robustez**: Student-t só com idiossincrático branco (`"iid"`); E-step Student-t
   monta um sistema por período (memória n × N × m); `smooth/predict` em novos vintages
   usam o modelo gaussiano.
10. **Densidade**: `NowcastDistribution.sample` sorteia períodos independentemente; EM
    fornece `std` só fora da amostra.
11. **Diagnósticos**: p-valores dos testes sup por Monte Carlo (cache); teste binomial
    supõe independência entre séries.
12. **Datasets**: GRS (2008) original indisponível (`load_us_grs_like` aproxima);
    `load_us_bgr2011`/`load_brazil_2000_2019` não implementados; vintages do PIB/IBGE
    2010T1–2026T2; sem conector BCB Olinda (Focus); FRED-MD depende de raspagem.
13. **`simulate/`, `pipeline/`, `cli` — resolvida** (onda 3). Limitações do I7: com
    `n_jobs>1` o *warm start* não é compartilhado entre processos; o escore por critério
    de informação conta grupos (não séries) e é uma triagem.
14. **Validação de referência — resolvida**: 90+ testes contra o R `nowcasting` 1.1.2
    (caixa-preta) e o statsmodels (`tests/reference_validation/`, `docs/validation/`).
    Divergências documentadas (`reference_divergence`): substituição de outliers (mediana
    móvel × média móvel do R), padronização/ddof do two-step, pontos de partida/parada do
    EM (o multi-start chega a um máximo local diferente e mais alto que o do R no NY
    Fed).
15. **EM com blocos aninhados**: mesmo com o multi-start há máximos locais (NY Fed: 46
    pontos abaixo do statsmodels). Warm start (`init=EMParameters`) permite partir de
    qualquer ponto.
16. **Pendências para autores**: agradecimentos e conferência do `.bib` (entradas com
    `verify`) do paper JSS; itens `[TODO]` de pesquisa do paper 2; `quarter_to_month`
    com âncora no 1º/2º mês (como `qtr2month(reference_month=1|2)` do R) não existe.

## Pós-auditoria (correção final)

- **Viés de antecipação no backtest do pipeline — resolvido.** A auditoria econométrica mostrou que
  `outputs.backtest` usava o painel já limpo com a amostra inteira (outliers por IQR, mediana móvel
  centrada, spline). `PseudoRealTimeBacktest(preprocess=...)` aplica a limpeza a cada vintage e o
  pipeline agora faz o backtest sobre os dados brutos de cada vintage
  (`tests/evaluation/test_backtest.py::test_preprocess_is_applied_to_each_vintage_without_look_ahead`).
- Metadado do NYFED aponta para `datasets/licenses/NYFED-BSD-3-Clause.txt`; `python -m nowcastbox`.
- Pendências abertas da auditoria (não bloqueiam 0.1.0): agrupamento de *news* quando séries somem
  entre vintages; `res.nowcast` devolve o DataFrame interno (mutável); nomes `frequencies=` vs
  `frequency=`; validação tardia em `generate_vintages`; links relativos no README (PyPI);
  MANIFEST.in do sdist; I1 só no `MixedFreqDFM` (two-step/benchmarks exigem razões fixas);
  múltiplos máximos locais no EM com vários blocos; vintages do IBC-Br e conector Focus (I8).


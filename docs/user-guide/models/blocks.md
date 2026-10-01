# Blocks

In large panels groups of series share co-movement beyond the business cycle: real
activity, prices, financial conditions, surveys. A **block structure** gives each group
its own factors on top of a global one (Bańbura, Giannone & Reichlin, 2011; Bańbura &
Modugno, 2014). The loading matrix has zeros where a series does not belong to a block:

$$
x_t =
\begin{bmatrix}
\Lambda_{G,\text{real}} & \Lambda_{R} & 0 \\
\Lambda_{G,\text{fin}} & 0 & \Lambda_{F}
\end{bmatrix}
\begin{bmatrix} f^G_t \\ f^R_t \\ f^F_t \end{bmatrix} + \varepsilon_t ,
$$

and each block's factors follow their own VAR (independent across blocks).

## Specifying blocks

`MixedFreqDFM(blocks=...)` accepts:

| Value | Meaning |
|---|---|
| `None` | one `"global"` block with every series |
| `"data"` | the `blocks` metadata of the panel (built-in datasets carry it) |
| mapping | `{series: block or [blocks]}` |
| DataFrame | series × blocks, 0/1 or bool |
| array | `(n_series, n_blocks)` aligned with the columns (blocks `block1`, `block2`, ...) |

Every series must load on at least one block. `n_factors` is either one number for every
block or a mapping `{block: r}`.

```python
import nowcastbox as nb

COLUMNS = ["pib", "ibc_br", "pim_geral", "pmc_varejo", "pms_volume", "energia_consumo_total",
           "anp_diesel", "secex_exportacoes", "ipca", "selic", "credito_concessoes_sa",
           "icbr", "focus_pib"]
ds = nb.load_brazil_nowcast(columns=COLUMNS, start="2012-01", end="2024-12")
ds.blocks                       # global / real / nominal / financial / soft

panel = nb.prepare_panel(ds.data, ds.transform, keep="pib")
vintage = nb.pseudo_real_time(panel, delay=ds.delay, vintage="2024-11-15")

res = nb.MixedFreqDFM(n_factors=1, blocks="data").fit(vintage, target="pib")
res.factors.columns.tolist()    # ['global_f1', 'real_f1', 'nominal_f1', 'financial_f1', 'soft_f1']
res.block_factors("real").tail(3)
```

### A custom structure

```python
activity = ["pib", "ibc_br", "pim_geral", "pmc_varejo", "pms_volume"]
financial = ["selic", "credito_concessoes_sa", "icbr"]
blocks = {
    name: ["global"]
    + (["activity"] if name in activity else [])
    + (["financial"] if name in financial else [])
    for name in vintage.columns
}
custom = nb.MixedFreqDFM(
    n_factors={"global": 1, "activity": 1, "financial": 1}, blocks=blocks
).fit(vintage, target="pib")
custom.factors.columns.tolist(), custom.get_nowcast("2024Q4")
```

Blocks with no series are dropped with a `DataQualityWarning`. A block needs several
series to identify its factor; with one series the block factor just absorbs that
series' idiosyncratic component.

## Blocks in the analysis

News and level contributions can be grouped by block (a series in several blocks is
labelled with the `+`-joined names):

```python
old = nb.pseudo_real_time(panel, delay=ds.delay, vintage="2024-10-15")
res.news(old, vintage, "2024Q4").to_frame(by="block")
```

## Choosing blocks

- Start from economic groups: real activity, labour, prices, financial, surveys.
- Compare specifications out of sample with an [experiment](../evaluation/experiments.md)
  or a [backtest](../evaluation/backtesting.md); blocks rarely hurt the point nowcast
  and make the news decomposition more informative.
- `nb.nowcast(..., blocks="auto")` uses the panel's block metadata when present and a
  single global block otherwise.

## Starting values with several blocks

With nested blocks (a global block that contains every series plus thematic blocks) the
EM has several local maxima, and the sequential block principal components of Bańbura &
Modugno (2014) depend on the order of the blocks. The default `init="pca"` therefore
computes the block components in four orders (`"given"`, `"reversed"`,
`"specific_first"`, `"independent"`), evaluates the log-likelihood of each start with
one Kalman filter pass and starts the EM from the best; `init="pca_given"` (or another
`"pca_<order>"`) forces one order. On the NY Fed specification this raises the final
log-likelihood by 372 points (see [validation](../../validation/statsmodels.md)).

```python
res.info["initialization"]["block_order"], res.info["initialization"]["loglikelihood"]
```

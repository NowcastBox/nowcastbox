# Indicator heatmap (z-scores)

Before reading a nowcast it helps to see where the indicators stand. The **indicator
heatmap** (Linzenich & Meunier, 2024, ECB WP 3004, §3.4) shows every transformed series
as a z-score: how many standard deviations it sits above or below its long-run mean. Red
cells are readings below normal and blue cells are readings above normal, so you can
compare series measured in very different units on one colour scale.

## Method

For a transformed indicator $x_{i,t}$ (for example a growth rate or a diffusion index):

1. **Smoothing (monthly series).** The five-month moving average with Mariano &
   Murasawa (2003) weights,
   $s_{i,t} = \tfrac19 (x_{i,t} + 2x_{i,t-1} + 3x_{i,t-2} + 2x_{i,t-3} + x_{i,t-4})$,
   turns monthly growth rates into the growth rate of the quarterly average. This makes
   monthly series comparable with quarterly ones and removes most of the month-to-month
   noise. Series of other frequencies are not smoothed. If any of the five months is
   missing, the smoothed value is NaN (this happens at the start of the sample and at
   the ragged edge).
2. **Standardisation.** $z_{i,t} = (s_{i,t} - \bar s_i)/\hat\sigma_i$, with the mean and
   standard deviation (`ddof=1`) computed over the whole sample by default, or over a
   trailing rolling window (`window=`). Smoothing comes before standardisation, so the
   z-scores that are displayed have unit variance.
3. **Groups.** A group z-score is the mean of the z-scores of the group's series in that
   period. At the ragged edge, only the series already available enter the mean.

**No look-ahead.** With `as_of="YYYY-MM-DD"`, the panel is first cut to the data released
by that date (`MixedFrequencyData.as_of`, which uses the release delays). The smoothing,
the mean and the standard deviation then use only that information set. A rolling window
is always trailing.

## Usage

```python
import nowcastbox as nb
from nowcastbox.diagnostics import indicator_zscores

ds = nb.load_brazil_nowcast(start="2012-01", end="2024-12")
panel = nb.prepare_panel(ds.data, ds.transform)

z = indicator_zscores(panel, by="category", as_of="2024-11-15")
z.table("group", last=12)          # rows = groups, columns = months
z.table("series", frequency="Q")   # one column per quarter (third month)
z.latest()                         # latest z-score of each series and its period
print(z.summary())

fig = z.plot(last=24)                                   # Plotly heatmap of the groups
fig = z.plot(level="series", backend="matplotlib", annotate=True)
```

The heatmap is also registered on fitted results. These use the estimation panel without
the target:

```python
res = nb.MixedFreqDFM(n_factors=2).fit(panel, "pib")
res.plot("indicator_heatmap", by="block", last=18)
```

## Options

| Argument | Default | Meaning |
|---|---|---|
| `smooth` | `"mm"` | `"mm"` (1-2-3-2-1)/9, `None`/`"none"`, or custom weights (most recent first, normalised) |
| `window` | `None` | `None`: whole-sample moments; an integer: trailing rolling window in base periods |
| `by` | `None` | `"category"`, `"block"` (a series counts in every block it belongs to), `"frequency"` or `{series: group or [groups]}` |
| `as_of` | `None` | information date (requires release delays) |
| `series` | all | subset of series |
| `ddof`, `min_periods` | `1`, `max(2, ddof+1)` | moments of the standardisation |

Plot options: `level="auto" | "series" | "group"`, `last=24`, `frequency="Q"`, `zmax=2`
(where the colour scale saturates), `annotate`, `backend`, `theme`.

A series with too few observations or with zero variance gets no z-scores, and a
`DataQualityWarning` names it.

## See also

- [Share of released data](../data/released-share.md) for the information available on
  the nowcast period.
- [HTML reports](reports.md) (`heatmap=True`) and the pipeline output `heatmap`
  ([Pipeline and CLI](../pipeline-cli.md)).

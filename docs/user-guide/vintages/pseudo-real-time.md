# Pseudo real-time data

A **pseudo real-time vintage** is the final dataset truncated as it would have looked at
a past date, using each series' publication delay (Giannone, Reichlin & Small, 2008).
It reproduces the ragged edge of the past without access to the historical databases;
it ignores data **revisions** (for those, see [Real-time vintages](real-vintages.md)).

## The release rule

The value of series $i$ for native period $p$ is released at

$$
d_i(p) = \operatorname{end}(p) + \delta_i ,
$$

where $\operatorname{end}(p)$ is the last calendar day of the period (31 March for
2024Q1) and $\delta_i$ the delay in days. It belongs to the vintage of date $v$ iff
$d_i(p) \le v$ (inclusive). Explicit release dates of a
[`ReleaseCalendar`](release-calendar.md) override the rule where available.

## One vintage

```python
import nowcastbox as nb

ds = nb.load_brazil_nowcast(
    columns=["pib", "ibc_br", "pim_geral", "pmc_varejo", "ipca", "focus_pib"],
    start="2015-01",
    end="2024-12",
)
ds.delay                                   # days after the end of the period

panel = nb.prepare_panel(ds.data, ds.transform, keep="pib")
v = nb.pseudo_real_time(panel, delay=ds.delay, vintage="2024-08-20")
v.last_observed()
```

On 20 August 2024 GDP for 2024Q2 (released about 62 days after the end of June) is not
yet available, industrial production stops in June and inflation in July. The grid is
unchanged; only values are removed.

`delay` accepts a `ReleaseCalendar`, a mapping or `Series` of days by series, a sequence
aligned with the columns (as in a legend) or one number for every series. With `None`
the `release_delays` metadata of the panel is used — the same as `panel.as_of(date)`:

```python
same = panel.as_of("2024-08-20")
same.equals(v)
```

## Many vintages

`generate_vintages` iterates over dates between `start` and `end` with a regular `step`
(`"M"`, `"2W"`, `"Q"`, an integer number of days...) or, with `step="release"`, one
vintage per release date of the calendar:

```python
for vintage in nb.generate_vintages(panel, ds.delay, start="2024-07-01", end="2024-10-01", step="M"):
    print(vintage.date.date(), vintage.data.last_observed()["pib"])
```

```python
calendar = ds.calendar()
dates = nb.vintages.vintage_dates("2024-07-01", "2024-07-31", step="release", calendar=calendar)
len(dates), dates[:5]
```

## Using vintages

- Nowcast as of a date: `MixedFreqDFM().fit(v, "pib")`.
- Explain the change between two vintages: [news decomposition](../news/news-decomposition.md).
- Follow the nowcast through a quarter: [nowcast tracker](../news/tracker.md).
- Evaluate out of sample: [backtesting](../evaluation/backtesting.md), which builds the
  vintages itself from `delay` or `calendar`.

```python
res = nb.MixedFreqDFM(n_factors=1).fit(v, target="pib")
res.nowcast.tail(3)[["observed", "out_of_sample", "std"]]
```

!!! note "Pseudo versus real time"
    Pseudo real-time exercises use today's revised data. For Brazilian GDP, revisions
    between the first release and the latest vintage are sizeable (see
    `VintageStore.revision_summary`); a real-time evaluation with
    [real vintages](real-vintages.md) gives a more honest picture of the accuracy a
    forecaster could have achieved.

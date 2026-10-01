# Release calendar

`ReleaseCalendar` answers *when* each observation is published. It combines two rules:

- a **delay rule**: days after the end of the reference period (one number per series);
- **explicit dates** for the periods where the true release date is known (e.g. the
  actual IBGE GDP release dates), which override the delay rule.

## Building a calendar

```python
import pandas as pd
import nowcastbox as nb

calendar = nb.ReleaseCalendar(
    {"gdp": 60, "ip": 40, "pmi": 1},
    frequencies={"gdp": "Q", "ip": "M", "pmi": "M"},
)
calendar.release_date("gdp", "2024Q1"), calendar.release_date("ip", "2024-03")
calendar.available_at("2024-06-15")          # last period of each series known then
```

Explicit dates come as a mapping or as a long table with `series`,
`reference_period` and `release_date` columns:

```python
table = pd.DataFrame(
    {
        "series": ["gdp", "gdp"],
        "reference_period": ["2024Q1", "2024Q2"],
        "release_date": ["2024-06-04", "2024-09-03"],
    }
)
explicit = nb.ReleaseCalendar.from_frame(table, delays={"gdp": 60, "ip": 40, "pmi": 1},
                                         frequencies={"gdp": "Q", "ip": "M", "pmi": "M"})
explicit.release_date("gdp", "2024Q2"), explicit.has_explicit_dates("gdp")
```

From a panel's metadata: `ReleaseCalendar.from_data(panel)`.

## The Brazilian calendar

`load_brazil_calendar()` returns the calendar of the Brazilian nowcasting panel: a
stylised delay per series and the actual GDP release dates of IBGE since 2010.

```python
br = nb.load_brazil_calendar()
br
br.release_date("pib", "2024Q1")                # actual IBGE release
br.to_frame().head()
br.releases_between("2024-05-01", "2024-05-10")  # what came out in early May 2024
```

## Queries

| Method | Returns |
|---|---|
| `release_date(series, period)` | release timestamp of one observation |
| `release_dates_for(series, periods)` | release timestamps of several periods |
| `available_at(date)` | last available period of each series at a date |
| `release_mask(data, vintage)` | boolean panel of what is published at `vintage` |
| `releases_between(start, end, data=None)` | long table of releases in a window |
| `release_dates(start, end, data=None)` | distinct release dates (vintage dates of a tracker) |
| `schedule(data)` | release date of every cell of a panel |
| `with_delays(...)`, `with_frequencies(...)` | modified copies |

```python
ds = nb.load_brazil_nowcast(columns=["pib", "ibc_br", "ipca"], start="2023-01", end="2024-12")
br.schedule(ds.data).tail(3)
br.release_mask(ds.data, "2024-11-15").tail(3)
```

A calendar can replace `delay` everywhere: `pseudo_real_time(panel, calendar=br, vintage=...)`,
`generate_vintages(panel, br, step="release", ...)`, `PseudoRealTimeBacktest(calendar=br)`
and `res.nowcast_tracker(panel, br, ...)`.

```python
v = nb.pseudo_real_time(ds.data, calendar=br, vintage="2024-09-02")
v.last_observed()          # 2024Q2 GDP was released on 3 September 2024
```

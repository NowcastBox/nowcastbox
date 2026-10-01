# Pseudo real-time vintages vs `PRTDB`

Tests: `tests/reference_validation/test_r_vintages.py`. Fixtures:
`scripts/reference_fixtures/r_prtdb.R`.

`PRTDB(mts, delay, vintage)` keeps the original values and blanks those not yet released
at the vintage date. A vintage is therefore summarised, per series, by the number of
values kept and the position of the last one. Vintages were taken **daily around month
ends** and monthly over a decade. The daily vintages distinguish "end of month + delay"
from "start of next month + delay".

| Panel | Series | Vintages | Mismatches |
|---|---|---:|---:|
| BRGDP | 100 monthly, delays 0-84 days | 191 (2008-2017 monthly + 2017-08-25 to 2017-11-05 daily) | **0** |
| NYFED | 25 (monthly and quarterly, delays −14 to 63 days) | 216 (2005-2016 monthly + 2016-11-20 to 2017-01-31 daily) | **0** |

Release rule, identical in both libraries: an observation of period p is available
when end(p) + delay ≤ vintage (inclusive). For quarterly series the end is the end of the
quarter. `pseudo_real_time` never changes a value.

Differences:

- `PRTDB` **truncates the rows** after the vintage month. nowcastbox keeps the grid
  (those rows are empty), so models can still nowcast the current quarter.
- **Negative delays.** NYFED's Empire State and Philadelphia Fed surveys have delay −14:
  they are published before the end of their reference month. Both libraries accept
  them (nowcastbox since the wave-3 integration: delays must be integers larger than
  −366 days) and agree on every vintage.
- Speed: 0.02 s per BRGDP vintage against 0.06 s in R since the release mask computes
  the period ends once per frequency (see [Performance](performance.md)).

# Data sources

`nowcastbox.data_sources` downloads public macroeconomic data over HTTPS, with retries
and a local cache, and returns pandas objects that plug directly into
`MixedFrequencyData`.

| Source | Functions | Content | Key |
|---|---|---|---|
| [BCB / SGS](bcb.md) | `fetch_sgs`, `fetch_sgs_series` | Banco Central do Brasil time-series system: IBC-Br, credit, interest and exchange rates, prices, fiscal | no |
| [IBGE / SIDRA](ibge.md) | `fetch_sidra`, `fetch_sidra_raw`, `sidra_path` | national accounts, industrial production, retail, services, labour market, inflation | no |
| [IPEADATA](ipea.md) | `fetch_ipeadata`, `fetch_ipeadata_series`, `fetch_ipeadata_metadata` | IPEA's database (thousands of series from many agencies) | no |
| [FRED / ALFRED](fred.md) | `fetch_fred`, `fetch_fred_series` | St. Louis Fed US data, real-time vintages (ALFRED) | `FRED_API_KEY` |

## Conventions

- Every `fetch_*` function returns `float64` data indexed by a `pandas.PeriodIndex` at
  the series' **native** frequency (monthly, quarterly...).
- With `base_frequency="M"` series of different frequencies are put on one monthly grid,
  lower-frequency values in the last month of their period — the
  `MixedFrequencyData` convention — ready for `MixedFrequencyData(frame, frequencies)`.
- Requests use HTTPS (IPEADATA's official endpoint is HTTP), a descriptive User-Agent,
  timeouts, and exponential back-off on HTTP 429/5xx (`max_retries`, `backoff_factor`).
  Errors are explicit: `DataSourceError`, `HTTPStatusError`, `MissingAPIKeyError`,
  `NowcastDataError` for empty or malformed answers.

## Cache

Responses are cached on disk for one day by default, so notebooks and backtests do not
hit the APIs repeatedly.

| Setting | Default |
|---|---|
| directory | `$NOWCASTBOX_CACHE_DIR`, else `$XDG_CACHE_HOME/nowcastbox`, else the user cache directory |
| time to live | `$NOWCASTBOX_CACHE_TTL` seconds (`none` = never expires), default 86 400 |
| disable | `NOWCASTBOX_DISABLE_CACHE=1`, or `cache=False` in a call |

```python
import tempfile

from nowcastbox.data_sources import DiskCache, get_default_cache, set_default_cache

cache = DiskCache(tempfile.mkdtemp(), ttl=3600)          # one hour
cache.set("https://example.org/series?id=1", '{"value": 1}')
cache.get("https://example.org/series?id=1")
"https://example.org/other" in cache

set_default_cache(cache)          # used by every fetch_* call without cache=
get_default_cache().ttl
set_default_cache(True)           # back to the environment defaults
```

## Working offline and testing

Every connector accepts a `session` with a `requests`-like `get(url, params=..., ...)`
method. Passing a fake session is how the test-suite runs without network, and how you
can unit-test your own data pipelines:

```python
import json

import nowcastbox as nb
from nowcastbox.data_sources import fetch_sgs


class Response:
    def __init__(self, payload):
        self.status_code = 200
        self.text = json.dumps(payload)
        self.headers = {}


class OfflineSession:
    """Answer every request with canned SGS data."""

    def get(self, url, params=None, **kwargs):
        rows = [{"data": f"01/{m:02d}/2024", "valor": f"{0.1 * m:.2f}"} for m in range(1, 7)]
        return Response(rows)

    def close(self):
        pass


frame = fetch_sgs({"ipca": 433}, start="2024-01", end="2024-06",
                  session=OfflineSession(), cache=False)
nb.MixedFrequencyData(frame, frequencies={"ipca": "M"})
```

!!! note "Built-in datasets"
    The [built-in datasets](../../datasets/index.md) were built with these connectors
    (scripts in `scripts/build_datasets/`) and ship with the package, so the tutorials
    and the test-suite never need the network.

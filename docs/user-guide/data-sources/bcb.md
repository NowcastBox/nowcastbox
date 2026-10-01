# BCB (SGS)

The *Sistema Gerenciador de Séries Temporais* (SGS) of the Banco Central do Brasil
publishes thousands of series identified by a numeric code. `fetch_sgs` queries the open
API (`https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados`) and splits long daily
requests into the 10-year windows the API accepts.

## Usage

<!-- skip-test -->
```python
from nowcastbox.data_sources import fetch_sgs, fetch_sgs_series

ipca = fetch_sgs_series(433, start="2015-01", name="ipca")            # monthly inflation
panel = fetch_sgs(
    {"ibc_br": 24363, "ipca": 433, "selic": 4390, "gdp": 22099},     # name -> SGS code
    start="2010-01",
    base_frequency="M",            # quarterly GDP index in the 3rd month of the quarter
)
```

| Argument | Meaning |
|---|---|
| `codes` | one code, a list, or a mapping `{name: code}` (names become the columns) |
| `start`, `end` | date-likes (`"2010-01"`, `"2010-01-01"`, `Timestamp`, `Period`) |
| `native_frequency` | override of the inferred frequency (scalar or per name) |
| `base_frequency` | output grid; required when the series have different frequencies |
| `cache`, `session`, `timeout`, `max_retries`, `backoff_factor` | HTTP options |

Some useful codes (check the [SGS portal](https://www3.bcb.gov.br/sgspub/) before use):

| Code | Series | Frequency |
|---|---|---|
| 24363 | IBC-Br (economic activity index) | M |
| 433 | IPCA, monthly % change | M |
| 4390 | Selic, accumulated in the month (% p.m.) | M |
| 22099 | GDP at market prices, quarterly chained volume index | Q |
| 20539 | credit outstanding, total | M |

## From SGS to a nowcast

The example runs offline with a canned response (see [Working offline](index.md#working-offline-and-testing));
with network access drop the `session=` argument.

```python
import json

import numpy as np
import pandas as pd
import nowcastbox as nb
from nowcastbox.data_sources import fetch_sgs


class Response:
    def __init__(self, payload):
        self.status_code = 200
        self.text = json.dumps(payload)
        self.headers = {}


class CannedSGS:
    """Monthly series for codes 24363 / 433, a quarterly one for 22099."""

    def get(self, url, params=None, **kwargs):
        rng = np.random.default_rng(len(url))
        if "22099" in url:
            dates = pd.date_range("2015-01-01", "2023-10-01", freq="QS")
        else:
            dates = pd.date_range("2015-01-01", "2024-03-01", freq="MS")
        values = 100 + rng.normal(0, 1, len(dates)).cumsum()
        rows = [{"data": f"{d:%d/%m/%Y}", "valor": f"{v:.2f}"} for d, v in zip(dates, values)]
        return Response(rows)

    def close(self):
        pass


frame = fetch_sgs({"ibc_br": 24363, "ipca": 433, "gdp": 22099}, start="2015-01",
                  base_frequency="M", session=CannedSGS(), cache=False)
data = nb.MixedFrequencyData(
    frame,
    frequencies={"ibc_br": "M", "ipca": "M", "gdp": "Q"},
    transforms={"ibc_br": "dlog", "ipca": "level", "gdp": "dlog"},
    release_delays={"ibc_br": 45, "ipca": 10, "gdp": 62},
)
panel = nb.prepare_panel(data, keep="gdp")
res = nb.MixedFreqDFM(n_factors=1).fit(panel, "gdp")
res.nowcast.tail(2)[["out_of_sample", "std"]]
```

SGS values are published as strings with a decimal point; empty values and the API's
error pages are reported explicitly (`NowcastDataError`, `HTTPStatusError`).

Market expectations (Focus survey) are served by another BCB API (Olinda); the Focus
series of the Brazilian dataset were built with it (`scripts/build_datasets/`).

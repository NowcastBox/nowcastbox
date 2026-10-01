# FRED / ALFRED

FRED (Federal Reserve Economic Data, St. Louis Fed) serves US macro series; ALFRED
serves their **real-time vintages**. The API needs a free key: set the `FRED_API_KEY`
environment variable or pass `api_key=` (a `MissingAPIKeyError` is raised otherwise).
The key never appears in logs or error messages.

## Usage

<!-- skip-test -->
```python
from nowcastbox.data_sources import fetch_fred, fetch_fred_series

gdp = fetch_fred_series("GDPC1", start="2000-01-01")                  # quarterly
panel = fetch_fred({"gdp": "GDPC1", "ip": "INDPRO", "payems": "PAYEMS"},
                   start="2000-01-01", base_frequency="M")

# ALFRED: the data as published on a given date
as_of_2020 = fetch_fred({"gdp": "GDPC1", "ip": "INDPRO"}, base_frequency="M",
                        vintage_date="2020-07-15")
```

## Building real-time vintages

A sequence of ALFRED requests becomes a [`VintageStore`](../vintages/real-vintages.md):

<!-- skip-test -->
```python
import nowcastbox as nb

dates = ["2020-04-30", "2020-05-29", "2020-06-30", "2020-07-31"]
vintages = {
    date: fetch_fred({"gdp": "GDPC1", "ip": "INDPRO"}, start="2015-01-01",
                     base_frequency="M", vintage_date=date)
    for date in dates
}
store = nb.VintageStore.from_vintages(vintages, frequencies={"gdp": "Q", "ip": "M"})
store.revisions("gdp").tail()
```

## Offline example

```python
import json

import pandas as pd
import nowcastbox as nb
from nowcastbox.data_sources import fetch_fred


class Response:
    def __init__(self, payload):
        self.status_code = 200
        self.text = json.dumps(payload)
        self.headers = {}


class CannedFRED:
    def get(self, url, params=None, **kwargs):
        quarterly = params["series_id"] == "GDPC1"
        dates = pd.date_range("2023-01-01", periods=4 if quarterly else 12,
                              freq="QS" if quarterly else "MS")
        start = 22000 if quarterly else 102
        rows = [{"realtime_start": "2024-01-01", "realtime_end": "2024-01-01",
                 "date": f"{d:%Y-%m-%d}", "value": str(start + i)} for i, d in enumerate(dates)]
        return Response({"observations": rows})

    def close(self):
        pass


frame = fetch_fred({"gdp": "GDPC1", "ip": "INDPRO"}, base_frequency="M",
                   api_key="0123456789abcdef0123456789abcdef",
                   session=CannedFRED(), cache=False)
nb.MixedFrequencyData(frame, frequencies={"gdp": "Q", "ip": "M"})
```

FRED's missing-value marker `"."` becomes NaN. For the FRED-MD database (McCracken & Ng,
2016) use the built-in `nb.load_us_fred_md()` (see [Datasets](../../datasets/index.md)).

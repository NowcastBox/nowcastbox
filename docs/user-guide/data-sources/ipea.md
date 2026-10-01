# IPEADATA

IPEADATA, maintained by the Instituto de Pesquisa Econômica Aplicada, aggregates series
from many Brazilian agencies (energy consumption from EPE, fuel sales from ANP, foreign
trade from SECEX, labour data...). Series are identified by an alphanumeric code
(`SERCODIGO`), and the OData v4 API also serves their metadata.

## Usage

<!-- skip-test -->
```python
from nowcastbox.data_sources import (
    fetch_ipeadata,
    fetch_ipeadata_metadata,
    fetch_ipeadata_series,
)

selic = fetch_ipeadata_series("BM12_TJOVER12", start="2015-01")
panel = fetch_ipeadata({"selic": "BM12_TJOVER12", "gdp": "SCN104_PIBPM104"},
                       base_frequency="M")
fetch_ipeadata_metadata("SCN104_PIBPM104")["PERNOME"]          # 'Trimestral'
```

| Argument | Meaning |
|---|---|
| `codes` | one code, a list or `{name: code}` |
| `start`, `end` | date window |
| `territorial_level`, `territory_code` | regional series (e.g. `"Estados"`, `"35"`); national by default |
| `native_frequency`, `base_frequency` | frequency override and output grid |
| `base_url` | API root (`http://www.ipeadata.gov.br/api/odata4`) |

## Offline example

```python
import json

import pandas as pd
from nowcastbox.data_sources import fetch_ipeadata


class Response:
    def __init__(self, payload):
        self.status_code = 200
        self.text = json.dumps(payload)
        self.headers = {}


class CannedIPEA:
    def get(self, url, params=None, **kwargs):
        code = url.split("SERCODIGO='")[1].rstrip("')")
        dates = pd.date_range("2024-01-01", periods=6, freq="MS")
        rows = [
            {"SERCODIGO": code, "VALDATA": f"{d:%Y-%m-%d}T00:00:00-03:00",
             "VALVALOR": 0.8 + 0.01 * i, "NIVNOME": "", "TERCODIGO": ""}
            for i, d in enumerate(dates)
        ]
        return Response({"value": rows})

    def close(self):
        pass


fetch_ipeadata({"selic": "BM12_TJOVER12"}, session=CannedIPEA(), cache=False)
```

Dates come with a time-zone offset (`-03:00`) and are converted to periods of the native
frequency; regional series are filtered by `territorial_level` / `territory_code`.

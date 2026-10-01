# IBGE (SIDRA)

SIDRA is IBGE's table system: national accounts, industrial production (PIM-PF), retail
(PMC), services (PMS), labour market (PNAD Contínua), inflation (IPCA)... A query names
a **table**, a **variable**, optional **classifications** (category filters), the
territorial level and the periods. `sidra_path` shows the API path of a query:

```python
from nowcastbox.data_sources import sidra_path

sidra_path(8888, 12606, {"c544": 129314}, periods="last 12")    # PIM-PF, general industry
sidra_path(1620, 583, {"c11255": 90707})                        # quarterly GDP, volume index
```

## Usage

<!-- skip-test -->
```python
from nowcastbox.data_sources import fetch_sidra, fetch_sidra_raw

pim = fetch_sidra(8888, 12606, {"c544": 129314}, name="pim", start="2012-01")
gdp = fetch_sidra(1620, 583, {"c11255": 90707}, name="gdp")       # quarterly
raw = fetch_sidra_raw(1620, 583, {"c11255": 90707}, periods="last 4")  # the API's table
```

| Argument | Meaning |
|---|---|
| `table`, `variable` | SIDRA table and variable codes |
| `classifications` | `{"c544": 129314}` or `{544: [129314, 129315]}` |
| `territorial_level`, `territories` | `"n1"` (Brazil, default), `"n3"` (states)... and their codes |
| `periods`, `start`, `end` | `"all"`, `"last 12"`, explicit codes, or a date window |
| `name` | column name (default: built from the query) |
| `dash_as` | value of SIDRA's `-` sign (structural zero by default; `numpy.nan` to treat as missing) |
| `base_frequency` | output grid (e.g. `"M"` for quarterly tables) |

SIDRA's special signs are handled explicitly: `..` (not applicable), `...` (not
available) and `X` (confidential) become NaN; `-` (an absolute zero) follows `dash_as`.

## Offline example

```python
import json

import nowcastbox as nb
from nowcastbox.data_sources import fetch_sidra

HEADER = {
    "NC": "Nível Territorial (Código)", "NN": "Nível Territorial",
    "MC": "Unidade de Medida (Código)", "MN": "Unidade de Medida", "V": "Valor",
    "D1C": "Brasil (Código)", "D1N": "Brasil",
    "D2C": "Variável (Código)", "D2N": "Variável",
    "D3C": "Mês (Código)", "D3N": "Mês",
}


class Response:
    def __init__(self, payload):
        self.status_code = 200
        self.text = json.dumps(payload)
        self.headers = {}


class CannedSIDRA:
    def __init__(self):
        self.urls = []

    def get(self, url, params=None, **kwargs):
        self.urls.append(url)
        rows = [
            {"NC": "1", "NN": "Brasil", "MC": "30", "MN": "Número-índice", "V": str(100 + i),
             "D1C": "1", "D1N": "Brasil", "D2C": "12606", "D2N": "PIMPF",
             "D3C": f"2024{month:02d}", "D3N": f"{month}/2024"}
            for i, month in enumerate(range(1, 7))
        ]
        return Response([HEADER, *rows])

    def close(self):
        pass


session = CannedSIDRA()
pim = fetch_sidra(8888, 12606, name="pim", session=session, cache=False)
session.urls[0]
pim
nb.apply_transforms(pim, "dlog", frequency={"pim": "M"}).round(4)
```

"""Build ``brazil_nowcast``: Brazilian monthly/quarterly panel for nowcasting GDP.

Primary sources: BCB/SGS (``nowcastbox.data_sources.fetch_sgs_series``), IBGE/SIDRA
(``fetch_sidra``), IPEADATA (``fetch_ipeadata_series``) and the BCB Focus market
expectations survey (Olinda OData API, queried directly because ``data_sources`` has no
Olinda connector). Every series in :mod:`brazil_spec` is downloaded; series that fail
(unknown code, empty answer, too short) are dropped and recorded in the metadata.

Usage::

    python3 scripts/build_datasets/build_brazil_nowcast.py
"""

from __future__ import annotations

import datetime as dt
import time
import urllib.parse
import warnings
from typing import Any

import pandas as pd
import requests
from _common import LOG, legacy_code, setup_logging, write_dataset
from brazil_spec import EXCLUDED_FOR_LICENSE, FOCUS, IPEA, SERIES, SGS, SIDRA

from nowcastbox.data_sources import fetch_ipeadata_series, fetch_sgs_series, fetch_sidra

START = "2003-01"
MIN_OBS = 24
OLINDA = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"


def _olinda(resource: str, flt: str, select: str) -> pd.DataFrame:
    # Olinda rejects "+" for spaces: percent-encode the OData query explicitly.
    query = {"$filter": flt, "$select": select, "$format": "json", "$top": "1000000"}
    url = OLINDA + resource + "?" + urllib.parse.urlencode(query, quote_via=urllib.parse.quote)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    rows = resp.json().get("value", [])
    if not rows:
        raise ValueError(f"Olinda {resource} returned no rows for {flt!r}.")
    return pd.DataFrame(rows)


def focus_fixed_horizon(indicator: str) -> pd.Series:
    """Constant-horizon annual expectation from fixed-event Focus forecasts.

    For a survey date in month ``m`` of year ``Y`` the expectation for the next 12
    months is approximated by ``(13 - m)/12 * E[Y] + (m - 1)/12 * E[Y + 1]`` (Dovern,
    Fritsche & Slacalek, 2012), then averaged over the survey dates of the month.
    """
    flt = f"Indicador eq '{indicator}' and baseCalculo eq 0 and Data ge '{START}-01'"
    raw = _olinda("ExpectativasMercadoAnuais", flt, "Data,DataReferencia,Mediana")
    raw["Data"] = pd.to_datetime(raw["Data"])
    raw["ref"] = pd.to_numeric(raw["DataReferencia"], errors="coerce")
    raw = raw.dropna(subset=["ref", "Mediana"])
    raw["year"] = raw["Data"].dt.year
    cur = raw[raw["ref"] == raw["year"]].groupby("Data")["Mediana"].mean()
    nxt = raw[raw["ref"] == raw["year"] + 1].groupby("Data")["Mediana"].mean()
    both = pd.concat({"cur": cur, "nxt": nxt}, axis=1).dropna()
    month = both.index.month  # pyright: ignore[reportAttributeAccessIssue]
    w = (13 - month) / 12.0
    fixed = w * both["cur"] + (1 - w) * both["nxt"]
    monthly = fixed.groupby(both.index.to_period("M")).mean()  # pyright: ignore[reportAttributeAccessIssue]
    return monthly.astype(float)


def focus_ipca_12m() -> pd.Series:
    """Monthly average of the smoothed median 12-month-ahead IPCA expectation."""
    flt = f"Indicador eq 'IPCA' and Suavizada eq 'S' and baseCalculo eq 0 and Data ge '{START}-01'"
    raw = _olinda("ExpectativasMercadoInflacao12Meses", flt, "Data,Mediana")
    raw["Data"] = pd.to_datetime(raw["Data"])
    return raw.groupby(raw["Data"].dt.to_period("M"))["Mediana"].mean().astype(float)


def fetch_with_retry(spec: dict[str, Any], attempts: int = 3) -> pd.Series:
    """Call :func:`fetch_one`, retrying transient API errors without the cache."""
    for attempt in range(attempts):
        try:
            return fetch_one(spec, cache=None if attempt == 0 else False)
        except Exception as err:
            if attempt == attempts - 1:
                raise
            LOG.info("Retrying %s after error: %s", spec["name"], err)
            time.sleep(5.0 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_one(spec: dict[str, Any], cache: bool | None = None) -> pd.Series:
    """Download one series of the spec on its native grid."""
    source, code, query = spec["source"], spec["code"], spec["query"]
    if source == SGS:
        return fetch_sgs_series(code, start=f"{START}-01", cache=cache)
    if source == IPEA:
        return fetch_ipeadata_series(code, start=START, cache=cache)
    if source == SIDRA:
        frame = fetch_sidra(
            query["table"],
            query["variable"],
            query.get("classifications") or None,
            name=spec["name"],
            start=START,
            cache=cache,
        )
        if frame.shape[1] != 1:
            raise ValueError(f"SIDRA query returned {frame.shape[1]} columns.")
        return frame.iloc[:, 0]
    if source == FOCUS:
        resource, indicator = str(code).split(":", 1)
        if resource == "ExpectativasMercadoAnuais":
            return focus_fixed_horizon(indicator)
        return focus_ipca_12m()
    raise ValueError(f"Unknown source {source!r}.")


def to_monthly(series: pd.Series, frequency: str) -> pd.Series:
    """Place a native-frequency series on the monthly grid (quarterly -> 3rd month)."""
    series = series.dropna()
    idx = series.index
    if not isinstance(idx, pd.PeriodIndex):
        raise ValueError("Expected a PeriodIndex.")
    if frequency == "Q":
        if idx.freqstr[0] != "Q":
            raise ValueError(f"Expected quarterly data, got {idx.freqstr}.")
        series.index = idx.asfreq("M", how="end")
    elif idx.freqstr != "M":
        raise ValueError(f"Expected monthly data, got {idx.freqstr}.")
    return series[series.index >= pd.Period(START, "M")]


def check_delay(name: str, series: pd.Series, delay: int, today: pd.Timestamp) -> None:
    """Warn when the stylised delay is inconsistent with what is published today."""
    last = series.index.max()
    step = (
        3 if name.startswith(("pib", "consumo_", "fbcf", "exportacoes_cn", "importacoes_cn")) else 1
    )
    nxt = last + step
    due = nxt.to_timestamp(how="end").normalize() + pd.Timedelta(days=delay)
    if due < today - pd.Timedelta(days=10):
        LOG.warning(
            "%s: period %s due on %s (delay %d) but not published", name, nxt, due.date(), delay
        )


def main() -> None:
    """Download, verify and write the dataset."""
    setup_logging()
    today = pd.Timestamp(dt.date.today())
    columns: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    for spec in SERIES:
        name = spec["name"]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = fetch_with_retry(spec)
            series = to_monthly(raw, spec["frequency"])
            if "log" in spec["transform"] and (series <= 0).any():
                # SIDRA encodes "not available" cells of index series as "-" (= 0).
                LOG.warning("%s: %d non-positive values set to NaN", name, int((series <= 0).sum()))
                series = series[series > 0]
            if series.notna().sum() < (MIN_OBS // 3 if spec["frequency"] == "Q" else MIN_OBS):
                raise ValueError(f"only {series.notna().sum()} observations since {START}")
        except Exception as err:
            LOG.warning("Dropping %s (%s %s): %s", name, spec["source"], spec["code"], err)
            dropped.append(
                {
                    "name": name,
                    "source": spec["source"],
                    "code": str(spec["code"]),
                    "reason": str(err)[:200],
                }
            )
            continue
        check_delay(name, series, spec["delay_days"], today)
        columns[name] = series.rename(name)
        rows.append(
            {
                "name": name,
                "description": spec["description"],
                "source": spec["source"],
                "source_code": str(spec["code"]),
                "frequency": spec["frequency"],
                "transform": spec["transform"],
                "legacy_code": legacy_code(spec["transform"]),
                "delay_days": spec["delay_days"],
                "blocks": spec["blocks"],
                "category": spec["category"],
                "units": spec["units"],
                "first_observation": str(series.index.min()),
            }
        )
        LOG.info("%-36s %s..%s (%d obs)", name, series.index.min(), series.index.max(), len(series))
    frame = pd.concat(columns.values(), axis=1)
    frame = frame.reindex(pd.period_range(START, frame.index.max(), freq="M"))
    meta = {
        "title": "Brazilian nowcasting panel (BCB, IBGE, IPEA)",
        "description": (
            "Monthly and quarterly Brazilian indicators from 2003 to the latest release for "
            "nowcasting quarterly GDP: national accounts (target: seasonally adjusted GDP "
            "volume index, QoQ growth), IBC-Br, industrial production, retail and services "
            "surveys, labour market, external trade, prices, money, credit, fiscal, "
            "financial variables and Focus survey expectations. Values are in levels as "
            "published; the legend gives the named stationarity transformation."
        ),
        "loader": "load_brazil_nowcast",
        "target": "pib",
        "url": "https://www.bcb.gov.br/estatisticas ; https://sidra.ibge.gov.br ; http://www.ipeadata.gov.br",
        "sources": [
            "Banco Central do Brasil - SGS (dadosabertos.bcb.gov.br), ODbL",
            "Banco Central do Brasil - Focus market expectations (Olinda API), ODbL",
            "IBGE - SIDRA (sidra.ibge.gov.br), free use with attribution",
            "IPEADATA (ipeadata.gov.br), free use citing IPEA and the original source "
            "(EPE/Eletrobras, ANP, SECEX/MDIC, Receita Federal, Ministry of Labour)",
        ],
        "license": (
            "Data: BCB series under the Open Data Commons Open Database License (ODbL "
            "1.0) as published on dadosabertos.bcb.gov.br; IBGE and IPEADATA public data, "
            "free reuse with attribution of the source. Only open-license sources are "
            "included."
        ),
        "citation": (
            "Banco Central do Brasil (SGS, Focus); IBGE (SIDRA: Contas Nacionais "
            "Trimestrais, PIM-PF, PMC, PMS, IPCA, INPC, IPP, PNAD Continua); IPEA "
            "(IPEADATA). Compiled by nowcastbox (scripts/build_datasets/build_brazil_nowcast.py)."
        ),
        "notes": (
            "Delays are stylised typical publication lags (days after the end of the "
            "reference month/quarter) from the 2024-2026 release calendars; GDP release "
            "dates are available exactly in load_brazil_calendar(). Seasonally adjusted "
            "series are used where the source publishes them; for unadjusted series the "
            "legend proposes 12-month transformations (dlog_yoy, annual_diff). The "
            "seasonal adjustment and the whole history of each series correspond to the "
            "latest vintage (no real-time data, see load_brazil_vintages for GDP). "
            "Focus constant-horizon expectations weight the current and next calendar "
            "year forecasts by (13-m)/12 and (m-1)/12 (Dovern, Fritsche & Slacalek, 2012)."
        ),
        "excluded_for_license": EXCLUDED_FOR_LICENSE,
        "dropped": dropped,
        "built": {"retrieved": today.date().isoformat()},
    }
    write_dataset("brazil_nowcast", frame, rows, meta)
    if dropped:
        LOG.warning("Dropped %d series: %s", len(dropped), [d["name"] for d in dropped])


if __name__ == "__main__":
    main()

"""Build ``us_fred_md`` (FRED-MD + FRED-QD real GDP) and the ``us_grs_like`` metadata.

Primary source: the FRED-MD / FRED-QD databases of McCracken & Ng (2016, 2020),
St. Louis Fed (https://www.stlouisfed.org/research/economists/mccracken/fred-databases).
The script scrapes that page for the most recent monthly vintage (``YYYY-MM-md.csv`` or
``YYYY-rev-MM-md.csv``; the legacy ``files.stlouisfed.org/.../current.csv`` URL answers
HTTP 403 and the ``current.csv`` mirror on the new site can be stale), reads the
``Transform:`` row (tcodes 1-7), the series descriptions and groups of the official
appendix, and adds real GDP (``GDPC1``) from the matching FRED-QD vintage.

Series whose underlying data are subject to third-party copyright (S&P 500 indices,
Moody's corporate bond yields, the University of Michigan sentiment index and the CBOE
VIX) are excluded from the shipped file; they are listed in the metadata.

The ``us_grs_like`` dataset (a documented FRED-MD-based approximation of the
Giannone, Reichlin & Small, 2008, panel, whose original replication files are not
available from a primary source) reuses this data file; only its metadata is written.

Usage::

    python3 scripts/build_datasets/build_us_fred_md.py
"""

from __future__ import annotations

import io
import re
import zipfile
from typing import Any

import pandas as pd
import requests
from _common import LOG, legacy_code, setup_logging, write_dataset, write_metadata_only

from nowcastbox.datasets._io import read_metadata

BASE = "https://www.stlouisfed.org"
PAGE = BASE + "/research/economists/mccracken/fred-databases"

TCODE_TRANSFORM = {
    1: "level",
    2: "diff",
    3: "diff|diff",
    4: "log",
    5: "dlog",
    6: "log|diff|diff",
    7: "pct_change|diff",
}
"""FRED-MD tcodes (McCracken & Ng, 2016, appendix) as named nowcastbox transforms."""

GROUPS = {
    1: "output_income",
    2: "labor",
    3: "housing",
    4: "consumption_orders",
    5: "money_credit",
    6: "rates_fx",
    7: "prices",
    8: "stock_market",
}
GROUP_CATEGORY = {5: "financial", 6: "financial", 8: "financial"}
GROUP_DELAY = {1: 16, 2: 5, 3: 18, 4: 35, 5: 25, 6: 1, 7: 14, 8: 1}
"""Stylised publication delays (days after the end of the month) by FRED-MD group."""

SERIES_DELAY = {
    "RPI": 29, "W875RX1": 29, "HWI": 40, "HWIURATIO": 40, "CLAIMSx": 4,
    "DPCERA3M086SBEA": 29, "CMRMTSPLx": 45, "RETAILx": 14, "BUSINVx": 45, "ISRATIOx": 45,
    "BUSLOANS": 10, "REALLN": 10, "DTCOLNVHFNM": 10, "DTCTHFNM": 10, "INVEST": 10,
    "NONREVSL": 38, "CONSPI": 38, "OILPRICEx": 1, "PCEPI": 29, "DDURRG3M086SBEA": 29,
    "DNDGRG3M086SBEA": 29, "DSERRG3M086SBEA": 29, "GDPC1": 28,
}  # fmt: skip
"""Series-specific stylised delays (BEA/BLS/Census/Federal Reserve release schedules)."""

RESTRICTED = {
    "S&P 500": "S&P Dow Jones Indices copyright",
    "S&P div yield": "S&P Dow Jones Indices copyright",
    "S&P PE ratio": "S&P Dow Jones Indices copyright",
    "AAA": "Moody's copyright",
    "BAA": "Moody's copyright",
    "AAAFFM": "Moody's copyright (spread built on AAA)",
    "BAAFFM": "Moody's copyright (spread built on BAA)",
    "UMCSENTx": "University of Michigan Surveys of Consumers copyright",
    "VIXCLSx": "Cboe copyright",
}


def latest_vintage_urls() -> tuple[str, str, str, str]:
    """Return (vintage label, monthly CSV URL, quarterly CSV URL, appendix ZIP URL)."""
    html = requests.get(PAGE, timeout=120).text
    links = set(re.findall(r'href="([^"]+)"', html))
    monthly: dict[tuple[int, int, int], str] = {}
    for link in links:
        m = re.search(r"/monthly/(\d{4})-(rev-)?(\d{2})-md\.csv", link)
        if m:
            key = (int(m.group(1)), int(m.group(3)), int(bool(m.group(2))))
            monthly[key] = BASE + link.split("?")[0].replace("&amp;", "&")
    if not monthly:
        raise RuntimeError("No FRED-MD monthly vintage link found on " + PAGE)
    key = max(monthly)
    label = f"{key[0]}-{key[1]:02d}" + (" (revised)" if key[2] else "")
    md_url = monthly[key]
    qd_url = md_url.replace("/monthly/", "/quarterly/").replace("-md.csv", "-qd.csv")
    appendix = next(
        BASE + link.replace("&amp;", "&") for link in links if "fred-md_appendix" in link
    )
    return label, md_url, qd_url, appendix


def read_fred_csv(url: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Download a FRED-MD/QD CSV; return (data on a monthly grid, tcodes)."""
    text = requests.get(url, timeout=300).text
    raw = pd.read_csv(io.StringIO(text))
    first = raw.columns[0]
    labels = raw[first].astype(str).str.lower()
    tcode_row = raw[labels.str.startswith("transform")].iloc[0]
    tcodes = {c: int(float(tcode_row[c])) for c in raw.columns[1:]}
    body = raw[pd.to_datetime(raw[first], format="%m/%d/%Y", errors="coerce").notna()]
    dates = pd.to_datetime(body[first], format="%m/%d/%Y")
    frame = body.drop(columns=first).apply(pd.to_numeric, errors="coerce")
    frame.index = pd.PeriodIndex(dates, freq="M")
    return frame.astype(float), tcodes


def read_appendix(url: str) -> pd.DataFrame:
    """Return the official FRED-MD appendix (id, tcode, fred, description, group)."""
    content = requests.get(url, timeout=300).content
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = next(n for n in zf.namelist() if n.endswith("updated_appendix.csv"))
        table = pd.read_csv(zf.open(name), encoding="latin1")
    table["key"] = table["fred"].str.upper()
    return table.set_index("key")


def legend_row(name: str, tcode: int, info: Any, frequency: str = "M") -> dict[str, Any]:
    """Legend row of one FRED-MD/QD series."""
    group = int(info["group"]) if info is not None else 1
    transform = TCODE_TRANSFORM[tcode]
    description = str(info["description"]) if info is not None else name
    return {
        "name": name,
        "description": description,
        "source": "FRED-MD" if frequency == "M" else "FRED-QD",
        "source_code": name,
        "frequency": frequency,
        "transform": transform,
        "legacy_code": legacy_code(transform),
        "delay_days": SERIES_DELAY.get(name, GROUP_DELAY[group]),
        "blocks": f"global;{GROUPS[group]}",
        "category": GROUP_CATEGORY.get(group, "hard"),
        "units": "see FRED series notes",
        "fred_md_tcode": tcode,
        "fred_md_group": GROUPS[group],
    }


def main() -> None:
    """Download the latest vintage and write the datasets."""
    setup_logging()
    label, md_url, qd_url, app_url = latest_vintage_urls()
    LOG.info("FRED-MD vintage %s: %s", label, md_url)
    md, tcodes = read_fred_csv(md_url)
    qd, qcodes = read_fred_csv(qd_url)
    appendix = read_appendix(app_url)
    keep = [c for c in md.columns if c not in RESTRICTED]
    rows = [
        legend_row(c, tcodes[c], appendix.loc[c.upper()] if c.upper() in appendix.index else None)
        for c in keep
    ]
    gdp = qd["GDPC1"].dropna()
    gdp.index = gdp.index.asfreq("Q").asfreq("M", how="end")
    gdp_info = {"group": 1, "description": "Real Gross Domestic Product (chained dollars)"}
    rows.append(legend_row("GDPC1", qcodes["GDPC1"], gdp_info, frequency="Q"))
    rows[-1]["units"] = "billions of chained dollars, SAAR"
    frame = md[keep].copy()
    full = pd.period_range(min(frame.index.min(), gdp.index.min()), frame.index.max(), freq="M")
    frame = frame.reindex(full)
    frame["GDPC1"] = gdp.reindex(full)
    frame = frame.dropna(how="all")
    frame = frame.reindex(pd.period_range(frame.index.min(), frame.index.max(), freq="M"))
    common = {
        "url": PAGE,
        "license": (
            "FRED-MD/FRED-QD are distributed freely by the Federal Reserve Bank of St. Louis "
            "for research. The shipped series come from US government agencies (BEA, BLS, "
            "Census, Federal Reserve Board, Treasury; public domain) or are constructed by "
            "McCracken & Ng; series under third-party copyright are excluded (see "
            "'excluded_for_license')."
        ),
        "citation": (
            "McCracken, M. W. & Ng, S. (2016). FRED-MD: A Monthly Database for "
            "Macroeconomic Research. Journal of Business & Economic Statistics, 34(4), "
            "574-589. McCracken, M. W. & Ng, S. (2020). FRED-QD: A Quarterly Database for "
            "Macroeconomic Research. NBER Working Paper 26872."
        ),
        "sources": [f"FRED-MD vintage {label}: {md_url}", f"FRED-QD vintage {label}: {qd_url}"],
        "excluded_for_license": [{"series": k, "reason": v} for k, v in RESTRICTED.items()],
    }
    meta = {
        "title": "FRED-MD monthly US macro database + FRED-QD real GDP",
        "description": (
            "FRED-MD (McCracken & Ng, 2016): ~120 monthly US macroeconomic series from 1959 "
            "with the official transformation codes, mapped to named nowcastbox transforms "
            "(tcode 1 level, 2 diff, 3 diff|diff, 4 log, 5 dlog, 6 log|diff|diff, 7 "
            "pct_change|diff), plus quarterly real GDP (GDPC1) from FRED-QD stored in the "
            "third month of each quarter. Values in levels as in the vintage file."
        ),
        "loader": "load_us_fred_md",
        "target": "GDPC1",
        **common,
        "notes": (
            "Blocks: 'global' plus the FRED-MD group of the series. Delays are stylised "
            "typical US publication lags by group/series (days after the end of the "
            "month). The 'fred_md_tcode' legend column keeps the original code."
        ),
        "vintage": label,
        "build_script": "build_us_fred_md.py",
    }
    write_dataset("us_fred_md", frame, rows, meta)
    fred_meta = read_metadata("us_fred_md")
    write_metadata_only(
        "us_grs_like",
        {
            "title": "GRS (2008)-like US panel built from FRED-MD",
            "description": (
                "Documented approximation of the Giannone, Reichlin & Small (2008) panel "
                "(~200 monthly US series + quarterly GDP, 1982-2004): the FRED-MD series "
                "with their FRED-MD transformations and real GDP, restricted by default to "
                "1982-01..2004-12, with stylised release delays."
            ),
            "loader": "load_us_grs_like",
            "target": "GDPC1",
            **common,
            "notes": (
                "The original GRS (2008) replication files are not available from a primary "
                "source (the authors' former web pages are offline; copies inside GPL "
                "software are not used). This dataset is therefore NOT the original panel: "
                "it has ~120 instead of ~200 series, current (revised) vintages and FRED-MD "
                "definitions. It shares the data file of 'us_fred_md'."
            ),
            "default_sample": {"start": "1982-01", "end": "2004-12"},
            "files": {"data": fred_meta["files"]["data"]},
            "summary": fred_meta["summary"],
            "vintage": label,
            "built": fred_meta["built"],
        },
    )


if __name__ == "__main__":
    main()

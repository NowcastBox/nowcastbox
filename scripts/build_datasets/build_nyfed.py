"""Build ``nyfed``: the FRBNY Staff Nowcast public replication panel (Bok et al., 2018).

Primary source: the Federal Reserve Bank of New York repository
https://github.com/FRBNY-TimeSeriesAnalysis/Nowcasting (BSD 3-Clause license), files
``Spec_US_example.xls`` (series, frequency, blocks, transformation, units, category)
and ``data/US/<vintage>.xls`` (the example data, downloaded by the NY Fed from FRED).
Only the data and the specification are used; none of the repository's MATLAB code.

Reading the legacy ``.xls`` files needs ``xlrd`` at build time only (it is not a
nowcastbox dependency): ``pip install xlrd``.

Usage::

    python3 scripts/build_datasets/build_nyfed.py [VINTAGE]   # default 2017-01-27
"""

from __future__ import annotations

import io
import sys
from typing import Any

import pandas as pd
import requests
from _common import LOG, legacy_code, setup_logging, write_dataset

RAW = "https://raw.githubusercontent.com/FRBNY-TimeSeriesAnalysis/Nowcasting/master/"
REPO = "https://github.com/FRBNY-TimeSeriesAnalysis/Nowcasting"
DEFAULT_VINTAGE = "2017-01-27"

TRANSFORMS = {
    "lin": "level",
    "chg": "diff",
    "pch": "pct_change(1)|scale(100)",
    # Compounded annual rate of change of a quarterly series, 100*((x_t/x_{t-1})^4 - 1),
    # approximated by the annualised log difference (exact to first order).
    "pca": "log|diff(1)|scale(400)",
}
"""NY Fed / FRED ``units`` codes as named nowcastbox transforms."""

DELAYS = {
    "PAYEMS": 5, "UNRATE": 5, "JTSJOL": 40, "GDPC1": 28, "A261RX1Q020SBEA": 58,
    "ULCNFB": 35, "CPIAUCSL": 14, "CPILFESL": 14, "PPIFIS": 13, "IR": 14, "IQ": 14,
    "DGORDER": 27, "HSN1F": 25, "RSAFS": 14, "HOUST": 18, "PERMIT": 18, "INDPRO": 16,
    "TCU": 16, "DSPIC96": 29, "PCEC96": 29, "PCEPI": 29, "PCEPILFE": 29, "BOPTEXP": 35,
    "BOPTIMP": 35, "WHLSLRIMSA": 40, "TTLCONS": 31, "BUSINV": 44,
    "GACDISA066MSFRBNY": 0, "GACDFSA066MSFRBPHI": 0,
}  # fmt: skip
"""Stylised publication delays in days after the end of the reference period.

Regional Fed manufacturing surveys are released within the reference month (delay 0).
"""

BLOCK_COLUMNS = {
    "Block1-Global": "global",
    "Block2-Soft": "soft",
    "Block3-Real": "real",
    "Block4-Labor": "labor",
}


def read_xls(path: str) -> pd.DataFrame:
    """Download and parse an ``.xls`` file of the repository."""
    resp = requests.get(RAW + path, timeout=120)
    resp.raise_for_status()
    return pd.read_excel(io.BytesIO(resp.content), engine="xlrd")


def legend_rows(spec: pd.DataFrame) -> list[dict[str, Any]]:
    """Translate the NY Fed specification into legend rows."""
    rows = []
    for _, r in spec.iterrows():
        name = str(r["SeriesID"]).strip()
        code = str(r["Transformation"]).strip().lower()
        transform = TRANSFORMS[code]
        blocks = [b for col, b in BLOCK_COLUMNS.items() if int(r[col]) == 1]
        category_label = str(r["Category"]).strip()
        rows.append(
            {
                "name": name,
                "description": str(r["SeriesName"]).strip(),
                "source": "FRBNY Nowcasting repository (data from FRED)",
                "source_code": name,
                "frequency": str(r["Frequency"]).strip().upper(),
                "transform": transform,
                "legacy_code": legacy_code(transform),
                "delay_days": DELAYS[name],
                "blocks": ";".join(blocks),
                "category": "soft" if category_label.lower() == "surveys" else "hard",
                "units": str(r["Units"]).strip(),
                "nyfed_transformation": code,
                "nyfed_category": category_label,
                "nyfed_model": int(r["Model"]),
            }
        )
    return rows


def main(vintage: str = DEFAULT_VINTAGE) -> None:
    """Download the specification and one data vintage and write the dataset."""
    setup_logging()
    spec = read_xls("Spec_US_example.xls")
    data = read_xls(f"data/US/{vintage}.xls")
    data.index = pd.PeriodIndex(pd.to_datetime(data.pop("Date")), freq="M")
    rows = legend_rows(spec)
    names = [r["name"] for r in rows]
    missing = sorted(set(names) - set(data.columns))
    if missing:
        raise RuntimeError(f"Series of the specification missing from the data: {missing}")
    frame = data[names].astype(float)
    quarterly = [r["name"] for r in rows if r["frequency"] == "Q"]
    for col in quarterly:  # NY Fed files already store quarterly values in the 3rd month
        bad = frame[col].dropna().index.month % 3 != 0
        if bad.any():
            raise RuntimeError(f"{col}: quarterly values outside the 3rd month.")
    meta = {
        "title": "NY Fed Staff Nowcast replication panel (US)",
        "description": (
            "The public example panel of the FRBNY Staff Nowcast (Bok, Caratelli, Giannone, "
            "Sbordone & Tambalotti, 2018): monthly and quarterly US series (1985 onwards) "
            "with the NY Fed block structure (global, soft, real, labor), transformations "
            f"and categories, data vintage {vintage}. Target: real GDP (GDPC1)."
        ),
        "loader": "load_nyfed",
        "target": "GDPC1",
        "url": REPO,
        "sources": [
            f"{REPO} - Spec_US_example.xls and data/US/{vintage}.xls (data downloaded by "
            "the NY Fed from FRED, Federal Reserve Bank of St. Louis)"
        ],
        "license": (
            "Repository files: BSD 3-Clause License, Copyright (c) 2018, Federal Reserve "
            "Bank of New York (redistribution with this notice). Underlying series: FRED, "
            "from US government agencies and regional Federal Reserve Banks."
        ),
        "citation": (
            "Bok, B., Caratelli, D., Giannone, D., Sbordone, A. M. & Tambalotti, A. (2018). "
            "Macroeconomic Nowcasting and Forecasting with Big Data. Annual Review of "
            "Economics, 10, 615-643 (Federal Reserve Bank of New York Staff Report 830)."
        ),
        "notes": (
            "The NY Fed note that these example files do not exactly reproduce the "
            "published Staff Nowcast (data redistribution restrictions). 'nyfed_model' = 1 "
            "marks the 25 series of the example model (Model column of the "
            "specification); load_nyfed(model_only=True) keeps only those. Transformations: "
            "lin -> level, chg -> diff, pch -> pct_change(1)|scale(100), pca (compounded "
            "annual rate) -> log|diff(1)|scale(400), its first-order approximation; the "
            "original code is in 'nyfed_transformation'. Delays are stylised typical US "
            "publication lags (not part of the NY Fed files)."
        ),
        "vintage": vintage,
        "license_notice": (
            "Copyright (c) 2018, Federal Reserve Bank of New York. All rights reserved. "
            "Redistribution and use in source and binary forms, with or without "
            "modification, are permitted provided that the conditions of the BSD 3-Clause "
            "License are met (see nowcastbox/datasets/licenses/NYFED-BSD-3-Clause.txt)."
        ),
        "build_script": "build_nyfed.py",
    }
    write_dataset("nyfed", frame, rows, meta)
    LOG.info("nyfed: %d series, %s..%s", len(names), frame.index.min(), frame.index.max())


if __name__ == "__main__":
    main(*sys.argv[1:2])

"""Build ``brazil_vintages``: real-time vintages of Brazilian quarterly GDP (IBGE).

Primary source: the quarterly national accounts publication *Contas Nacionais
Trimestrais - Indicadores de Volume e Valores Correntes* of each release, kept by IBGE
at https://ftp.ibge.gov.br/Contas_Nacionais/Contas_Nacionais_Trimestrais/Fasciculo_Indicadores_IBGE/.
From every release (2010Q1 onwards, PDF) the script extracts *Tabela 6 - Série
Encadeada do Índice de Volume Trimestral com Ajuste Sazonal* (1995 = 100): GDP, the
three supply-side sectors, gross value added and the five demand components, i.e.
the full seasonally adjusted history **as published in that release**.

Release dates come, in order of preference, from the "Publicado em dd/mm/aaaa" line of
the publication, from the FTP listing time stamp of releases uploaded at publication
time (09:00/10:00), or from the time stamp of the file inside the release ZIP / the PDF
creation date (next business day when produced after 09:00). Dates outside 50-80 days
after the end of the quarter are rejected; if no candidate survives, the date is
estimated as 65 days after the quarter end. The provenance of every date is kept in the
``releases`` table.

Releases 2006Q4-2009Q4 are Word files whose tables are embedded OLE objects and are not
parsed (documented gap).

Requirements (build time only): ``pdftotext``/``pdfinfo`` (poppler-utils).

Usage::

    python3 scripts/build_datasets/build_brazil_vintages.py [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from _common import LOG, setup_logging

from nowcastbox.datasets._io import DATA_DIR, METADATA_DIR, write_csv_gz, write_metadata

FTP = (
    "https://ftp.ibge.gov.br/Contas_Nacionais/Contas_Nacionais_Trimestrais/"
    "Fasciculo_Indicadores_IBGE/"
)
FIRST_YEAR = 2010
COLUMNS = [
    "pib_agropecuaria",
    "pib_industria",
    "pib_servicos",
    "va_pb",
    "pib",
    "consumo_familias",
    "consumo_governo",
    "fbcf",
    "exportacoes_cn",
    "importacoes_cn",
]
DESCRIPTIONS = {
    "pib_agropecuaria": "Gross value added: agriculture",
    "pib_industria": "Gross value added: industry",
    "pib_servicos": "Gross value added: services",
    "va_pb": "Gross value added at basic prices",
    "pib": "GDP at market prices",
    "consumo_familias": "Household consumption",
    "consumo_governo": "Government consumption",
    "fbcf": "Gross fixed capital formation",
    "exportacoes_cn": "Exports of goods and services",
    "importacoes_cn": "Imports of goods and services",
}
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}
ROW_RE = re.compile(r"^\s*(\d{4})\.(IV|III|II|I)\s+(.+?)\s*$")
FILE_RE = re.compile(r"(\d{4})(0[1-4])caderno")
LIST_RE = re.compile(
    r'href="([^"?/]*caderno[^"]*\.(?:pdf|zip))">.*?<td align="right">(\d{4}-\d{2}-\d{2}) (\d{2}):(\d{2})'
)
MIN_LAG, MAX_LAG, DEFAULT_LAG = 50, 80, 65


def listing(url: str) -> list[tuple[str, pd.Timestamp]]:
    """Files of an FTP directory listing with their time stamps."""
    html = requests.get(url, timeout=120).text
    return [
        (m.group(1), pd.Timestamp(f"{m.group(2)} {m.group(3)}:{m.group(4)}"))
        for m in LIST_RE.finditer(html)
    ]


def discover() -> dict[pd.Period, dict[str, Any]]:
    """Map each reference quarter to its PDF/ZIP URLs and listing time stamps."""
    found: dict[pd.Period, dict[str, Any]] = {}
    years = list(range(FIRST_YEAR, dt.date.today().year + 1))
    dirs = [FTP + f"{y}/" for y in years] + [FTP]
    for base in dirs:
        for name, stamp in listing(base):
            m = FILE_RE.search(name)
            if not m:
                continue
            quarter = pd.Period(f"{m.group(1)}Q{int(m.group(2))}", freq="Q")
            entry = found.setdefault(quarter, {})
            kind = "pdf" if name.endswith(".pdf") else "zip"
            entry.setdefault(kind, (base + name, stamp))
    return {q: e for q, e in sorted(found.items()) if q.year >= FIRST_YEAR}


def fetch(url: str, cache_dir: Path) -> Path:
    """Download a file once into ``cache_dir``."""
    target = cache_dir / url.rsplit("/", 1)[-1]
    if not target.exists():
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        target.write_bytes(resp.content)
    return target


def pdf_from_entry(entry: dict[str, Any], cache_dir: Path) -> tuple[Path, pd.Timestamp | None]:
    """Return the release PDF (extracting it from the ZIP if needed) and the ZIP member time."""
    member_time = None
    if "zip" in entry:
        zpath = fetch(entry["zip"][0], cache_dir)
        with zipfile.ZipFile(zpath) as zf:
            pdfs = [i for i in zf.infolist() if i.filename.lower().endswith(".pdf")]
            if pdfs:
                member_time = pd.Timestamp(dt.datetime(*pdfs[0].date_time))
                if "pdf" not in entry:
                    out = cache_dir / Path(pdfs[0].filename).name
                    out.write_bytes(zf.read(pdfs[0]))
                    return out, member_time
    if "pdf" not in entry:
        raise FileNotFoundError("release has no PDF")
    return fetch(entry["pdf"][0], cache_dir), member_time


def pdf_text(path: Path) -> str:
    """Layout-preserving text of a PDF (poppler ``pdftotext``)."""
    return subprocess.run(
        ["pdftotext", "-layout", str(path), "-"], capture_output=True, check=True, text=True
    ).stdout


def pdf_creation(path: Path) -> pd.Timestamp | None:
    """PDF creation time stamp (``pdfinfo -isodates``)."""
    out = subprocess.run(
        ["pdfinfo", "-isodates", str(path)], capture_output=True, text=True, check=False
    ).stdout
    m = re.search(r"CreationDate:\s+(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})", out)
    return pd.Timestamp(f"{m.group(1)} {m.group(2)}:{m.group(3)}") if m else None


def _number(token: str) -> float:
    return float(token.replace(".", "").replace(",", "."))


def parse_table6(text: str) -> pd.DataFrame:
    """Rows of *Tabela 6* (SA chained volume index) as a quarterly DataFrame."""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if re.search(r"Tabela 6\s*-\s*S", ln)]
    if not starts:
        raise ValueError("Tabela 6 not found")
    rows: dict[pd.Period, list[float]] = {}
    for ln in lines[starts[0] + 1 :]:
        if re.search(r"Tabela\s+7\s*-", ln):
            break
        m = ROW_RE.match(ln)
        if not m:
            continue
        tokens = m.group(3).split()
        if len(tokens) != len(COLUMNS):
            raise ValueError(f"unexpected row layout: {ln.strip()!r}")
        period = pd.Period(f"{m.group(1)}Q{ROMAN[m.group(2)]}", freq="Q")
        rows.setdefault(period, [_number(t) for t in tokens])
    if not rows:
        raise ValueError("Tabela 6 has no rows")
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=COLUMNS).sort_index()
    expected = pd.period_range(frame.index[0], frame.index[-1], freq="Q")
    if len(expected) != len(frame):
        raise ValueError("Tabela 6 has gaps")
    return frame


def _plausible(quarter: pd.Period, date: pd.Timestamp) -> bool:
    lag = (date.normalize() - quarter.end_time.normalize()).days
    return MIN_LAG <= lag <= MAX_LAG


def _published_date(text: str, quarter: pd.Period) -> pd.Timestamp | None:
    m = re.search(r"Publicado em (\d{2})/(\d{2})/(\d{4})", text[:5000])
    if not m:
        return None
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    for y in (year, quarter.year, quarter.year + 1):  # fixes year typos in a few releases
        try:
            cand = pd.Timestamp(year=y, month=month, day=day)
        except ValueError:
            continue
        if _plausible(quarter, cand):
            return cand
    return None


def _next_release_day(stamp: pd.Timestamp) -> pd.Timestamp:
    day = stamp.normalize()
    if stamp.hour >= 9 or day.dayofweek >= 5:
        day += pd.offsets.BDay(1)
    return pd.Timestamp(day)


def release_date(
    quarter: pd.Period,
    text: str,
    entry: dict[str, Any],
    member_time: pd.Timestamp | None,
    creation: pd.Timestamp | None,
) -> tuple[pd.Timestamp, str]:
    """Best available release date of a publication and its provenance."""
    published = _published_date(text, quarter)
    if published is not None:
        return published, "publication ('Publicado em')"
    for kind in ("pdf", "zip"):
        if kind in entry:
            stamp = entry[kind][1]
            if stamp.hour in (9, 10) and stamp.minute == 0 and _plausible(quarter, stamp):
                return stamp.normalize(), f"FTP listing time stamp ({kind})"
    for label, stamp in (("ZIP member time stamp", member_time), ("PDF creation date", creation)):
        if stamp is not None:
            cand = _next_release_day(stamp)
            if _plausible(quarter, cand):
                return cand, label
    est = quarter.end_time.normalize() + pd.Timedelta(days=DEFAULT_LAG)
    return pd.Timestamp(est), f"estimated ({DEFAULT_LAG} days after quarter end)"


def changes_only(records: pd.DataFrame) -> pd.DataFrame:
    """Keep a row only when a value is first published or differs from the previous vintage."""
    records = records.sort_values(["series", "reference_period", "vintage_date"], kind="stable")
    prev = records.groupby(["series", "reference_period"])["value"].shift(1)
    keep = prev.isna() | (records["value"] != prev)
    return records[keep].reset_index(drop=True)


def main() -> None:
    """Download every release, parse it and write the dataset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", default=str(Path.home() / ".cache/nowcastbox/ibge_cnt"))
    args = parser.parse_args()
    setup_logging()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    releases: list[dict[str, Any]] = []
    parts: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    for quarter, entry in discover().items():
        try:
            pdf, member_time = pdf_from_entry(entry, cache_dir)
            text = pdf_text(pdf)
        except Exception as err:
            LOG.warning("Skipping release %s: %s", quarter, err)
            skipped.append({"reference_quarter": str(quarter), "reason": str(err)[:200]})
            continue
        date, how = release_date(quarter, text, entry, member_time, pdf_creation(pdf))
        releases.append(
            {
                "reference_quarter": str(quarter),
                "release_date": date.date().isoformat(),
                "date_source": how,
                "table_parsed": True,
                "file": (entry.get("pdf") or entry["zip"])[0],
            }
        )
        try:
            table = parse_table6(text)
            if table.index[-1] != quarter:
                raise ValueError(f"last row {table.index[-1]} != release quarter {quarter}")
        except Exception as err:
            LOG.warning("No vintage for release %s: %s", quarter, err)
            skipped.append({"reference_quarter": str(quarter), "reason": str(err)[:200]})
            releases[-1]["table_parsed"] = False
            continue
        LOG.info("%s released %s (%s), %d quarters", quarter, date.date(), how, len(table))
        long = table.reset_index(names="reference_period").melt(
            id_vars="reference_period", var_name="series", value_name="value"
        )
        long["vintage_date"] = date.date().isoformat()
        parts.append(long)
    records = pd.concat(parts, ignore_index=True)
    records["reference_period"] = records["reference_period"].astype(str)
    records = changes_only(records)
    records.insert(1, "frequency", "Q")
    records = records[["series", "frequency", "reference_period", "vintage_date", "value"]]
    records["series"] = pd.Categorical(records["series"], COLUMNS)
    records = records.sort_values(["series", "reference_period", "vintage_date"], kind="stable")
    records["series"] = records["series"].astype(str)
    release_table = pd.DataFrame(releases)
    files = {
        "data": {
            "path": "brazil_vintages.csv.gz",
            "sha256": write_csv_gz(records, DATA_DIR / "brazil_vintages.csv.gz"),
            "format": "csv.gz (long: series, frequency, reference_period, vintage_date, value; "
            "a row only when a value is first published or revised)",
            "n_rows": len(records),
        },
        "releases": {
            "path": "brazil_gdp_releases.csv.gz",
            "sha256": write_csv_gz(release_table, DATA_DIR / "brazil_gdp_releases.csv.gz"),
            "format": "csv.gz (reference_quarter, release_date, date_source, table_parsed, file)",
            "n_rows": len(release_table),
        },
    }
    meta = {
        "name": "brazil_vintages",
        "title": "Real-time vintages of Brazilian quarterly GDP (IBGE)",
        "description": (
            "Seasonally adjusted chained volume indices (1995=100) of GDP, value added by "
            "sector and demand components as published in each IBGE quarterly national "
            "accounts release since 2010Q1 (one vintage per release), for real-time "
            "evaluation and revision analysis."
        ),
        "loader": "load_brazil_vintages",
        "target": "pib",
        "url": FTP,
        "sources": [
            "IBGE - Contas Nacionais Trimestrais, Indicadores de Volume e Valores Correntes "
            "(one publication per quarter), Tabela 6",
        ],
        "license": "IBGE public data; free reuse with attribution of the source (IBGE).",
        "citation": (
            "IBGE - Instituto Brasileiro de Geografia e Estatística. Contas Nacionais "
            "Trimestrais: Indicadores de Volume e Valores Correntes, releases 2010Q1 to "
            f"{releases[-1]['reference_quarter']}. Compiled by nowcastbox."
        ),
        "notes": (
            "Values are the seasonally adjusted index levels; growth rates are computed by "
            "the user (e.g. QoQ = pct_change). Because IBGE re-estimates the seasonal "
            "adjustment each quarter, the whole history changes from release to release. "
            "Release dates: see the 'releases' file and its 'date_source' column. "
            "Releases 2006Q4-2009Q4 (Word documents with embedded tables) are not included."
        ),
        "skipped": skipped,
        "built": {
            "date": dt.date.today().isoformat(),
            "script": "scripts/build_datasets/build_brazil_vintages.py",
            "n_vintages": len(parts),
        },
        "files": files,
        "summary": {
            "n_series": len(COLUMNS),
            "start": str(min(records["reference_period"])),
            "end": str(max(records["reference_period"])),
            "frequencies": {"Q": len(COLUMNS)},
            "n_vintages": len(parts),
            "first_vintage": releases[0]["release_date"],
            "last_vintage": releases[-1]["release_date"],
        },
        "series": [
            {
                "name": c,
                "description": DESCRIPTIONS[c]
                + " - chained quarterly volume index, seasonally adjusted (1995=100)",
                "frequency": "Q",
            }
            for c in COLUMNS
        ],
    }
    write_metadata(meta, METADATA_DIR / "brazil_vintages.yaml")
    LOG.info("Wrote %d records from %d vintages", len(records), len(releases))


if __name__ == "__main__":
    main()

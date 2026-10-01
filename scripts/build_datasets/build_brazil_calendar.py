"""Build the ``brazil_calendar`` metadata (release delays of the Brazilian panel).

The calendar combines

* the typical publication delay of every series of ``brazil_nowcast`` (days after the
  end of the reference period, from its legend), documented per release programme
  below; and
* the **actual release dates** of the IBGE quarterly national accounts since 2010Q1
  (``brazil_gdp_releases.csv.gz``, produced by ``build_brazil_vintages.py``), used as
  explicit dates for GDP and its components.

No download is needed: run it after ``build_brazil_nowcast.py`` and
``build_brazil_vintages.py``.

Usage::

    python3 scripts/build_datasets/build_brazil_calendar.py
"""

from __future__ import annotations

import datetime as dt

from _common import setup_logging, write_metadata_only

from nowcastbox.datasets._io import read_metadata, read_table

PROGRAMMES = [
    {"programme": "IBGE Contas Nacionais Trimestrais (GDP)", "typical_delay_days": 62,
     "rule": "about 9 weeks after the quarter; exact dates since 2010Q1 in the releases file"},
    {"programme": "BCB IBC-Br", "typical_delay_days": 46,
     "rule": "mid-month, about 6-7 weeks after the reference month"},
    {"programme": "IBGE PIM-PF (industrial production)", "typical_delay_days": 35,
     "rule": "first week of month m+2"},
    {"programme": "IBGE PMC (retail)", "typical_delay_days": 42,
     "rule": "second week of month m+2"},
    {"programme": "IBGE PMS (services)", "typical_delay_days": 43,
     "rule": "second week of month m+2"},
    {"programme": "IBGE IPCA / INPC", "typical_delay_days": 10,
     "rule": "around the 10th of month m+1"},
    {"programme": "IBGE IPCA-15", "typical_delay_days": 0,
     "rule": "released around the 25th of the reference month (delay set to 0)"},
    {"programme": "IBGE IPP", "typical_delay_days": 30, "rule": "end of month m+1"},
    {"programme": "IBGE PNAD Continua (rolling quarter)", "typical_delay_days": 30,
     "rule": "end of month m+1 for the rolling quarter ending in m"},
    {"programme": "IPEA monthly PNAD (seasonally adjusted)", "typical_delay_days": 33,
     "rule": "a few days after the PNAD release"},
    {"programme": "Novo Caged (Ministry of Labour)", "typical_delay_days": 30,
     "rule": "end of month m+1"},
    {"programme": "BCB monetary and credit statistics", "typical_delay_days": 28,
     "rule": "about four weeks after the reference month (money aggregates: 25)"},
    {"programme": "BCB fiscal statistics", "typical_delay_days": 30, "rule": "end of month m+1"},
    {"programme": "BCB external sector statistics", "typical_delay_days": 25,
     "rule": "fourth week of month m+1"},
    {"programme": "SECEX trade balance", "typical_delay_days": 5,
     "rule": "first business days of month m+1"},
    {"programme": "Receita Federal tax revenue", "typical_delay_days": 25,
     "rule": "fourth week of month m+1"},
    {"programme": "EPE electricity consumption, ANP oil and fuels", "typical_delay_days": 30,
     "rule": "about one month after the reference month"},
    {"programme": "IPEA GFCF indicator / apparent consumption", "typical_delay_days": 95,
     "rule": "GFCF about three months after; apparent consumption about two months (63)"},
    {"programme": "Financial market data (Selic, PTAX, reserves, IC-Br)", "typical_delay_days": 1,
     "rule": "monthly averages/end-of-month values known within days (IC-Br: 5; REER: 75)"},
    {"programme": "BCB Focus survey (monthly average of weekly medians)", "typical_delay_days": 0,
     "rule": "known at the end of the reference month"},
]  # fmt: skip


def main() -> None:
    """Write ``metadata/brazil_calendar.yaml``."""
    setup_logging()
    panel = read_metadata("brazil_nowcast")
    vintages = read_metadata("brazil_vintages")
    releases = read_table("brazil_vintages", "releases")
    write_metadata_only(
        "brazil_calendar",
        {
            "title": "Release calendar of the Brazilian nowcasting panel",
            "description": (
                "Typical publication delays (days after the end of the reference period) "
                "of every series of brazil_nowcast, plus the actual release dates of the "
                "IBGE quarterly national accounts (GDP) since 2010Q1."
            ),
            "loader": "load_brazil_calendar",
            "url": "https://www.ibge.gov.br/calendario-de-divulgacao ; "
            "https://www.bcb.gov.br/estatisticas/calendario",
            "sources": [
                "IBGE and BCB release calendars 2024-2026 (typical delays)",
                "IBGE quarterly national accounts publications (GDP release dates)",
            ],
            "license": "Calendar information compiled from IBGE/BCB public calendars.",
            "citation": "Compiled by nowcastbox from IBGE and BCB release calendars.",
            "notes": (
                "Delays are stylised: they ignore weekends, holidays and calendar shifts "
                "of individual months. GDP and its components use the explicit release "
                "dates for 2010Q1 onwards (date provenance in the releases file) and the "
                "62-day delay before that and for future quarters."
            ),
            "programmes": PROGRAMMES,
            "explicit_dates_for": [
                s["name"]
                for s in panel["series"]
                if s["name"] in {v["name"] for v in vintages["series"]}
            ],
            "files": {"releases": vintages["files"]["releases"]},
            "summary": {
                "n_series": len(panel["series"]),
                "start": str(releases["reference_quarter"].iloc[0]),
                "end": str(releases["reference_quarter"].iloc[-1]),
                "n_release_dates": vintages["files"]["releases"]["n_rows"],
            },
            "built": {
                "date": dt.date.today().isoformat(),
                "script": "scripts/build_datasets/build_brazil_calendar.py",
            },
        },
    )


if __name__ == "__main__":
    main()

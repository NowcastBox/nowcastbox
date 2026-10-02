"""Packaging checks from the final software-quality audit."""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "nowcastbox"

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    tomllib = pytest.importorskip("tomli")


def _package_data_globs() -> list[str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return list(config["tool"]["setuptools"]["package-data"]["nowcastbox"])


def _matches(relative: str, globs: list[str]) -> bool:
    # setuptools globs: "**/" may match zero directories
    candidates = set(globs) | {g.replace("**/", "") for g in globs}
    return any(fnmatch.fnmatch(relative, g) for g in candidates)


def test_every_data_file_is_shipped_in_the_wheel() -> None:
    globs = _package_data_globs()
    missing = [
        str(path.relative_to(PACKAGE))
        for path in PACKAGE.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".py", ".pyc"}
        and path.name != ".gitkeep"
        and not _matches(str(path.relative_to(PACKAGE).as_posix()), globs)
    ]
    assert missing == []


def test_nyfed_bsd_license_text_is_shipped() -> None:
    text = (PACKAGE / "datasets" / "licenses" / "NYFED-BSD-3-Clause.txt").read_text(
        encoding="utf-8"
    )
    assert "Copyright (c) 2018, Federal Reserve Bank of New York" in text
    assert "Redistributions in binary form must reproduce" in text
    assert "AS IS" in text


def test_license_metadata_uses_spdx_expression() -> None:
    # setuptools deprecates the TOML-table license and license classifiers (hard error
    # from 2027-02-18): use the PEP 639 SPDX expression and license-files instead.
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = config["project"]
    assert project["license"] == "MIT"
    assert "LICENSE" in project["license-files"]
    assert not [c for c in project["classifiers"] if c.startswith("License ::")]
    requires = config["build-system"]["requires"]
    assert any(r.replace(" ", "").startswith("setuptools>=77") for r in requires)


# import name -> distribution; None = optional, imported lazily with an explicit error
_DISTRIBUTION = {"yaml": "pyyaml", "markupsafe": "jinja2", "sklearn": None, "openpyxl": None}


def test_every_third_party_import_is_a_declared_dependency() -> None:
    import ast
    import re

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        re.split(r"[<>=!~\[; ]", req, maxsplit=1)[0].lower()
        for req in config["project"]["dependencies"]
    }
    imported: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
    third_party = imported - set(sys.stdlib_module_names) - {"nowcastbox"}
    missing = sorted(
        name
        for name in third_party
        if _DISTRIBUTION.get(name, name) is not None
        and _DISTRIBUTION.get(name, name) not in declared
    )
    assert missing == []


def test_optional_excel_extra_is_declared() -> None:
    # openpyxl is imported lazily (Excel templates) and ships as the [excel] extra
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra = config["project"]["optional-dependencies"]["excel"]
    assert any(req.startswith("openpyxl") for req in extra)
    assert "pipeline/templates/*.xlsx" in config["tool"]["setuptools"]["package-data"]["nowcastbox"]

"""``nowcastbox init --excel``: Excel data templates (plan item 7)."""

from __future__ import annotations

import sys

import pandas as pd
import pytest

from nowcastbox.cli import EXIT_ERROR, EXIT_OK, EXIT_USAGE, build_parser, main
from nowcastbox.pipeline.data import example_workbook_path, read_excel_panel

pytest.importorskip("openpyxl")


def test_parser_flags():
    args = build_parser().parse_args(["init", "--excel", "--example"])
    assert args.excel and args.example and args.path is None


def test_init_excel_default_path(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init", "--excel"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Wrote empty Excel data template nowcast.xlsx" in out
    assert "source: excel, path: nowcast.xlsx" in out
    assert pd.ExcelFile(tmp_path / "nowcast.xlsx").sheet_names == [
        "monthly",
        "quarterly",
        "metadata",
        "readme",
    ]
    assert main(["init", "--excel"]) == EXIT_USAGE
    assert "already exists" in capsys.readouterr().err
    assert main(["init", "--excel", "--force"]) == EXIT_OK


def test_init_excel_example(tmp_path, capsys):
    target = tmp_path / "sub" / "data.xlsx"
    assert main(["init", str(target), "--excel", "--example"]) == EXIT_OK
    assert "Wrote example Excel workbook" in capsys.readouterr().out
    assert read_excel_panel(target).equals(read_excel_panel(example_workbook_path()))


def test_init_excel_usage_errors(tmp_path, capsys):
    assert main(["init", str(tmp_path / "data.csv"), "--excel"]) == EXIT_USAGE
    assert "not an .xlsx file" in capsys.readouterr().err
    assert main(["init", str(tmp_path / "s.yaml"), "--example"]) == EXIT_USAGE
    assert "only valid with --excel" in capsys.readouterr().err


def test_init_excel_without_openpyxl(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    assert main(["init", str(tmp_path / "t.xlsx"), "--excel"]) == EXIT_ERROR
    assert "nowcastbox[excel]" in capsys.readouterr().err


def test_init_yaml_default_path(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init", "-t", "csv"]) == EXIT_OK
    assert (tmp_path / "nowcast.yaml").is_file()
    assert "Wrote example spec nowcast.yaml" in capsys.readouterr().out

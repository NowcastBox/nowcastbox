"""Tests of the ``nowcastbox`` command-line interface."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from nowcastbox.__version__ import __version__
from nowcastbox.cli import EXIT_ERROR, EXIT_OK, EXIT_USAGE, build_parser, main
from nowcastbox.pipeline import template_text

pytestmark = [
    pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.ConvergenceWarning"),
    pytest.mark.filterwarnings("ignore::nowcastbox.core.exceptions.DataQualityWarning"),
]


def small_spec(path: Path, **changes) -> Path:
    spec = {
        "name": "sim",
        "target": "gdp",
        "data": {"source": "simulated_dfm", "columns": ["x01", "x02", "x03", "x04"]},
        "vintage": "2019-11-15",
        "preprocessing": False,
        "model": {"type": "MixedFreqDFM", "factors": 1, "max_iter": 20},
        "outputs": ["nowcast", "news", "density"],
        "snapshot_dir": "snapshots",
    }
    spec.update(changes)
    path.write_text(yaml.safe_dump(spec))
    return path


@pytest.fixture(autouse=True)
def _restore_logging():
    logger = logging.getLogger("nowcastbox")
    level, handlers = logger.level, list(logger.handlers)
    yield
    logger.setLevel(level)
    logger.handlers[:] = handlers


@pytest.fixture(scope="module")
def two_snapshots(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cli")
    spec = small_spec(folder / "sim.yaml")
    assert main(["run", str(spec)]) == EXIT_OK
    assert main(["run", str(spec), "--vintage", "2019-12-20"]) == EXIT_OK
    return spec


# ---------------------------------------------------------------- basics
def test_version(capsys):
    assert main(["--version"]) == EXIT_OK
    assert capsys.readouterr().out.strip() == f"nowcastbox {__version__}"


def test_no_command_is_usage_error(capsys):
    assert main([]) == EXIT_USAGE
    assert "COMMAND" in capsys.readouterr().err


def test_help(capsys):
    assert main(["--help"]) == EXIT_OK
    assert "snapshots" in capsys.readouterr().out


def test_parser():
    args = build_parser().parse_args(["snapshots", "diff", "dir"])
    assert (args.old, args.new) == ("latest~1", "latest")


def test_main_uses_sys_argv(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["nowcastbox", "init", "--list"])
    assert main() == EXIT_OK
    assert "simulated" in capsys.readouterr().out


# ---------------------------------------------------------------- init / validate
def test_init(tmp_path, capsys):
    target = tmp_path / "sub" / "nowcast.yaml"
    assert main(["init", str(target), "--template", "simulated"]) == EXIT_OK
    assert target.read_text() == template_text("simulated")
    assert "Wrote example spec" in capsys.readouterr().out
    assert main(["init", str(target)]) == EXIT_USAGE
    assert "already exists" in capsys.readouterr().err
    assert main(["init", str(target), "--force"]) == EXIT_OK
    assert target.read_text() == template_text("brazil_pib")
    assert main(["init", str(tmp_path / "x.yaml"), "-t", "nope"]) == EXIT_USAGE
    assert "Unknown template" in capsys.readouterr().err


def test_validate(tmp_path, capsys):
    spec = small_spec(tmp_path / "s.yaml")
    assert main(["validate", str(spec)]) == EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith("OK:") and "outputs   : nowcast, news, density" in out
    assert main(["validate", str(spec), "--check-data", "--show"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "data check: 5 series" in out and "target: gdp" in out


def test_validate_errors(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("target: gdp\ndata: {source: simulatd_dfm}\nmodel: {type: DFM}\n")
    assert main(["validate", str(bad)]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "2 problems" in err and "did you mean 'simulated_dfm'" in err
    assert main(["validate", str(tmp_path / "missing.yaml")]) == EXIT_USAGE
    assert "Spec file not found" in capsys.readouterr().err
    assert main(["run", str(tmp_path / "missing.yaml")]) == EXIT_USAGE


# ---------------------------------------------------------------- run
def test_run_json_no_snapshot(tmp_path, capsys):
    spec = small_spec(tmp_path / "s.yaml")
    assert main(["-q", "run", str(spec), "--json", "--no-snapshot"]) == EXIT_OK
    info = json.loads(capsys.readouterr().out)
    assert info["headline_period"] == "2019Q4" and info["snapshot"] is None
    assert not (tmp_path / "snapshots").exists()


def test_run_snapshot_dir_override(tmp_path, capsys):
    spec = small_spec(tmp_path / "s.yaml")
    other = tmp_path / "other"
    assert main(["-v", "run", str(spec), "--snapshot-dir", str(other)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Nowcast pipeline 'sim'" in out and str(other) in out
    assert len(list(other.iterdir())) == 1


def test_run_data_error(tmp_path, capsys):
    spec = small_spec(tmp_path / "s.yaml", vintage="1990-01-01")
    assert main(["run", str(spec)]) == EXIT_USAGE
    assert "before the data start" in capsys.readouterr().err


def test_run_runtime_error_from_data(tmp_path, capsys, monkeypatch):
    import nowcastbox.pipeline.runner as runner

    def broken(spec):
        raise OSError("disk on fire")

    monkeypatch.setattr(runner, "load_data", broken)
    spec = small_spec(tmp_path / "s.yaml")
    assert main(["run", str(spec)]) == EXIT_ERROR
    assert "OSError: disk on fire" in capsys.readouterr().err


def test_run_runtime_error(tmp_path, capsys):
    spec = small_spec(tmp_path / "s.yaml", model={"factors": 1, "idiosyncratic": "bogus"})
    assert main(["run", str(spec)]) == EXIT_ERROR
    assert "nowcastbox: error: ValueError" in capsys.readouterr().err


# ---------------------------------------------------------------- datasets
def test_datasets(capsys):
    assert main(["datasets", "list"]) == EXIT_OK
    assert "brazil_nowcast" in capsys.readouterr().out
    assert main(["datasets", "list", "--json"]) == EXIT_OK
    rows = json.loads(capsys.readouterr().out)
    assert {"name", "title", "target"} <= set(rows[0])
    assert main(["datasets", "info", "nyfed"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "29 series" in out and "GDPC1" in out
    assert main(["datasets", "info", "simulated_dfm"]) == EXIT_OK
    assert "simulated_dfm:" in capsys.readouterr().out
    assert main(["datasets", "info", "nyfed", "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["target"] == "GDPC1"
    assert main(["datasets", "info", "nope"]) == EXIT_ERROR
    assert "Unknown dataset" in capsys.readouterr().err


# ---------------------------------------------------------------- snapshots
def test_snapshots_list(two_snapshots, capsys):
    folder = two_snapshots.parent / "snapshots"
    assert main(["snapshots", "list", str(two_snapshots)]) == EXIT_OK
    out = capsys.readouterr().out
    assert out.count("_sim") == 2
    assert main(["snapshots", "list", str(folder), "--json"]) == EXIT_OK
    rows = json.loads(capsys.readouterr().out)
    assert [r["vintage"] for r in rows] == ["2019-11-15", "2019-12-20"]
    assert main(["snapshots", "list", str(folder), "--name", "nope"]) == EXIT_OK
    assert "(none)" in capsys.readouterr().out


def test_snapshots_show(two_snapshots, capsys):
    assert main(["snapshots", "show", str(two_snapshots)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "vintage    : 2019-12-20" in out and "News against" in out
    assert main(["snapshots", "show", str(two_snapshots), "latest~1", "--json"]) == EXIT_OK
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["vintage"] == "2019-11-15"
    assert main(["snapshots", "show", str(two_snapshots), "latest~1"]) == EXIT_OK
    assert "News against" not in capsys.readouterr().out
    assert main(["snapshots", "show", str(two_snapshots), "zzz"]) == EXIT_ERROR
    assert "No snapshot matches 'zzz'" in capsys.readouterr().err


def test_snapshots_diff_and_history(two_snapshots, capsys):
    assert main(["snapshots", "diff", str(two_snapshots)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "data changed     : yes" in out and "headline change" in out
    assert main(["snapshots", "diff", str(two_snapshots), "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["data_changed"] is True
    assert main(["snapshots", "history", str(two_snapshots)]) == EXIT_OK
    assert "2019Q4" in capsys.readouterr().out
    assert main(["snapshots", "history", str(two_snapshots), "--period", "2019Q3", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["period"] for r in rows] == ["2019Q3", "2019Q3"]


def test_snapshots_location_errors(tmp_path, capsys):
    assert main(["snapshots", "list", str(tmp_path / "nope")]) == EXIT_ERROR
    assert "neither a snapshot directory" in capsys.readouterr().err
    spec = tmp_path / "s.yaml"
    spec.write_text("target: gdp\ndata: {source: simulated_dfm}\n")
    assert main(["snapshots", "list", str(spec)]) == EXIT_USAGE
    assert "has no snapshot_dir" in capsys.readouterr().err
    assert main(["snapshots", "show", str(tmp_path)]) == EXIT_ERROR


# ---------------------------------------------------------------- subprocess smoke tests
def test_module_entry_point_subprocess(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "nowcastbox.cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"nowcastbox {__version__}"


def test_console_script_subprocess(tmp_path):
    exe = shutil.which("nowcastbox")
    if exe is None:
        pytest.skip("console script not installed on PATH")
    spec = tmp_path / "sim.yaml"
    init = subprocess.run(  # noqa: S603
        [exe, "init", str(spec), "-t", "simulated"], capture_output=True, text=True, check=False
    )
    assert init.returncode == 0, init.stderr
    valid = subprocess.run(  # noqa: S603
        [exe, "validate", str(spec)], capture_output=True, text=True, check=False
    )
    assert valid.returncode == 0, valid.stderr
    assert valid.stdout.startswith("OK:")

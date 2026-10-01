"""The README quickstart must run against the current API."""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[2] / "README.md"


def _python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)


def test_readme_has_quickstart_and_authors() -> None:
    text = README.read_text(encoding="utf-8")
    assert "## Quickstart" in text and "Alexandre Leão Sanches" in text and "MIT" in text
    assert len(_python_blocks()) >= 2


@pytest.mark.slow
def test_readme_python_blocks_run(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for block in _python_blocks():
            exec(compile(block, str(README), "exec"), namespace)  # noqa: S102 - our README
    res = namespace["res"]
    assert math.isfinite(res.get_nowcast("2024Q3"))  # type: ignore[attr-defined]
    assert namespace["news"].check_identity()

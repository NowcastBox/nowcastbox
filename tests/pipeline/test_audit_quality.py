"""Regression tests from the final software-quality audit (pipeline)."""

from __future__ import annotations

import pytest

from nowcastbox.pipeline import NowcastSpec, template_text


def test_from_yaml_given_yaml_text_points_to_from_string() -> None:
    text = template_text("simulated")  # long multi-line text: OSError on stat()
    with pytest.raises(FileNotFoundError, match="from_string"):
        NowcastSpec.from_yaml(text)


def test_from_yaml_missing_file_message(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Spec file not found"):
        NowcastSpec.from_yaml(tmp_path / "missing.yaml")

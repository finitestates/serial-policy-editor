from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import trajectory_editor

pytestmark = pytest.mark.optional

def steering_cli():
    vectors_src = Path(__file__).resolve().parents[2] / "vector" / "src"
    package_path = str(vectors_src / "trajectory_editor")
    if package_path not in trajectory_editor.__path__:
        trajectory_editor.__path__.append(package_path)
    return importlib.import_module("trajectory_editor.steering_vector_cli")


def test_public_vector_command_exposes_only_hidden_state_steering():
    parser = steering_cli()._parser()
    help_text = parser.format_help()

    assert "hidden-state" in help_text
    assert "import-cvector" in help_text
    assert "output-head" not in help_text
    assert "token-preference" not in help_text

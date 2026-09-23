"""Keep renderer selection at the terminal boundary as commands evolve."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.fakes import ScriptedIO, ScriptedTextIO
from trajectory_editor.terminal_contracts import TerminalProtocol

pytestmark = pytest.mark.current_workflow

SOURCE = Path(__file__).resolve().parents[2] / "core" / "src" / "trajectory_editor"
RENDERERS = {"plain_tui", "persistent_tui", "live_tui", "edge_tui"}
PRESENTATION = RENDERERS | {"tui", "tui_views", "candidate_columns", "ui_themes"}
PLAIN_FORBIDDEN = {
    "teacher_commands", "edge_commands", "episode_cli", "episode_ui",
    "episode_store", "episode_engine", "episode_session", "ephemeral_runtime",
    "episode_backend_loader", "run_loop",
}


def test_production_imports_keep_renderers_at_the_terminal_boundary():
    violations = []
    for path in SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = {(node.module or "").rsplit(".", 1)[-1]}
                imported.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
            elif isinstance(node, ast.Import):
                imported = {alias.name.rsplit(".", 1)[-1] for alias in node.names}
            else:
                continue
            if "plain_tui" in imported and path.stem != "tui":
                violations.append(f"{path.name}:{node.lineno}: plain_tui import")
            if path.stem not in PRESENTATION and imported & RENDERERS:
                violations.append(f"{path.name}:{node.lineno}: renderer import")
            if path.stem == "plain_tui" and imported & PLAIN_FORBIDDEN:
                violations.append(f"{path.name}:{node.lineno}: runtime import")
    assert not violations, "\n".join(violations)


def test_runtime_does_not_branch_on_renderer_or_probe_terminal_methods():
    violations = []
    for path in SOURCE.rglob("*.py"):
        if path.stem in PRESENTATION or path.stem == "terminal_contracts":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.IfExp, ast.While)):
                condition = node.test
            elif isinstance(node, ast.Match):
                condition = node.subject
            else:
                condition = None
            if condition is not None and any(
                isinstance(part, ast.Attribute)
                and part.attr in {"plain_ui", "live_views", "live_choices"}
                for part in ast.walk(condition)
            ):
                violations.append(f"{path.name}:{node.lineno}: renderer selection")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id in {"getattr", "hasattr"} \
                    and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) \
                    and node.args[1].value in {"read_choice", "read_edge", "prompt", "capabilities"}:
                violations.append(f"{path.name}:{node.lineno}: terminal method probe")
    assert not violations, "\n".join(violations)


def test_scripted_terminal_is_an_explicit_request_adapter():
    text_only = ScriptedTextIO([])
    assert not hasattr(text_only, "read_choice")
    terminal: TerminalProtocol = ScriptedIO([])
    assert terminal.capabilities.live_views is False
    assert terminal.terminal_size() is None
    with terminal.session():
        terminal.write("ready")
    assert terminal.output == ["ready\n"]

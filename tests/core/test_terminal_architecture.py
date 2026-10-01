"""Keep renderer selection at the terminal boundary as commands evolve."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.fakes import ScriptedIO, ScriptedTextIO
from trajectory_editor.terminal_contracts import TerminalProtocol

pytestmark = pytest.mark.current_workflow

SOURCE = Path(__file__).resolve().parents[2] / "core" / "src" / "trajectory_editor"
RENDERERS = {"plain_tui", "term"}
PRESENTATION = RENDERERS | {"tui", "tui_render", "candidate_columns", "ui_themes"}
PLAIN_FORBIDDEN = {
    "teacher_commands", "edge_commands", "episode_cli", "episode_ui",
    "episode_store", "episode_engine", "episode_session", "session_runtime",
    "episode_backend_loader", "run_loop",
}


def _presentation(path: Path) -> bool:
    return path.stem in PRESENTATION or path.parent.name == "term"


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
                violations.append(f"{path.name}:{node.lineno}: plain renderer import")
            if not _presentation(path) and imported & RENDERERS:
                violations.append(f"{path.name}:{node.lineno}: renderer import")
            if path.stem == "plain_tui" and imported & PLAIN_FORBIDDEN:
                violations.append(f"{path.name}:{node.lineno}: runtime import")
    assert not violations, "\n".join(violations)


def test_live_ui_is_lazy_and_the_plain_fallback_stays_isolated():
    tui_source = (SOURCE / "tui.py").read_text()
    tree = ast.parse(tui_source)
    module_imports = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    module_imports.update(
        (node.module or "").split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    )
    assert "term" not in {(node.module or "") for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert "textual" not in module_imports

    plain_tree = ast.parse((SOURCE / "plain_tui.py").read_text())
    plain_imports = set()
    for node in ast.walk(plain_tree):
        if isinstance(node, ast.Import):
            plain_imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            plain_imports.add((node.module or "").split(".", 1)[0])
    assert "textual" not in plain_imports
    assert "term" not in plain_imports


def test_runtime_does_not_branch_on_renderer_or_probe_terminal_methods():
    violations = []
    for path in SOURCE.rglob("*.py"):
        if _presentation(path) or path.stem == "terminal_contracts":
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
    with terminal.session():
        terminal.write("ready")
    assert terminal.output == ["ready\n"]

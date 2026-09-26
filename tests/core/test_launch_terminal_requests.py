"""Startup and plain prompt paths around the persistent terminal boundary."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_cli import main
from trajectory_editor.session_runtime import session_edge_menu
from trajectory_editor.episode_session import LiveSession
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_store import EpisodeStore

pytestmark = pytest.mark.current_workflow

class StartupIO(ScriptedIO):
    def __init__(self, responses, events):
        super().__init__(responses)
        self.events = events
        self.active = False

    @contextmanager
    def session(self):
        self.events.append("session entered")
        self.active = True
        try:
            yield self
        finally:
            self.active = False
            self.events.append("session restored")

    def prompt(self, request):
        assert self.active
        if request.multiline:
            self.events.append("compose")
        return super().prompt(request)


@pytest.mark.parametrize("ephemeral", [False, True])
def test_interactive_launch_composes_inside_session_before_backend(tmp_path, ephemeral):
    events = []
    io = StartupIO(["P", "q", "q"], events)
    tty_sys = SimpleNamespace(
        stdin=SimpleNamespace(isatty=lambda: True),
        stdout=SimpleNamespace(isatty=lambda: True),
        stderr=sys.stderr,
    )

    def load_backend(args, *unused):
        assert io.active and args.new_prompt == "P"
        events.append("backend loaded")
        backend = ConformingFakeBackend()
        return backend if ephemeral else (backend, backend.provenance(), False)

    with patch("trajectory_editor.episode_cli.sys", tty_sys), patch(
        "trajectory_editor.episode_cli.TerminalIO", return_value=io,
    ), patch(
        "trajectory_editor.episode_backend_loader.load_backend" if ephemeral
        else "trajectory_editor.episode_backend_loader.load_episode_backend",
        side_effect=load_backend,
    ):
        assert main([
            "--workspace", str(tmp_path / "episodes.db"), "--model", "fake",
            *(["--ephemeral"] if ephemeral else []),
        ]) == 0
    assert events == ["session entered", "compose", "backend loaded", "session restored"]
    with EpisodeStore(tmp_path / "episodes.db") as store:
        assert store.workspace_list(include_finished=True) == "No open episodes."


def test_launch_cancellation_restores_session_without_loading_backend(tmp_path, capsys):
    events = []
    io = StartupIO([None], events)
    tty_sys = SimpleNamespace(
        stdin=SimpleNamespace(isatty=lambda: True),
        stdout=SimpleNamespace(isatty=lambda: True),
        stderr=sys.stderr,
    )
    with patch("trajectory_editor.episode_cli.sys", tty_sys), patch(
        "trajectory_editor.episode_cli.TerminalIO", return_value=io,
    ), patch(
        "trajectory_editor.episode_backend_loader.load_episode_backend",
        side_effect=AssertionError("backend loaded after cancel"),
    ):
        assert main(["--workspace", str(tmp_path / "episodes.db"), "--model", "fake"]) == 2
    assert events == ["session entered", "compose", "session restored"]
    assert capsys.readouterr().err.count("error: prompt entry cancelled") == 1


def test_missing_source_on_piped_input_never_opens_terminal(tmp_path, capsys):
    with patch("trajectory_editor.episode_cli.TerminalIO",
               side_effect=AssertionError("opened terminal")):
        assert main(["--workspace", str(tmp_path / "episodes.db"), "--model", "fake"]) == 2
    assert "no episode source supplied" in capsys.readouterr().err


def test_noninteractive_commands_do_not_open_terminal(tmp_path, capsys):
    path = tmp_path / "episodes.db"
    engine = EpisodeEngine(
        ConformingFakeBackend(), sampling=SamplerConfig(),
        initial_text="P", initial_token_ids=[7],
    )
    with EpisodeStore(path) as store:
        store.create_episode(
            episode_id="source", initial_text="P", initial_token_ids=[7],
            sampling=engine.sampling, stream_fingerprint=engine.stream_fingerprint,
            max_tokens=None, backend={},
        )
    exported = tmp_path / "source.jsonl"
    with patch("trajectory_editor.episode_cli.TerminalIO",
               side_effect=AssertionError("opened terminal")):
        with pytest.raises(SystemExit) as help_exit:
            main(["--help"])
        assert help_exit.value.code == 0
        assert main(["--workspace", str(path), "--list"]) == 0
        assert main(["--workspace", str(path), "--projector", "source"]) == 0
        assert main(["--workspace", str(path), "--export-teacher-plan", "source", str(exported)]) == 0
    assert exported.exists()
    assert "source" in capsys.readouterr().out


def test_prompt_file_keeps_multiline_text_and_original_line_endings(tmp_path):
    path = tmp_path / "episodes.db"
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_bytes(b"P\r\nQ\n")
    io = ScriptedIO(["q", f"save {path} prompt-file", "q"])
    with patch("trajectory_editor.episode_backend_loader.load_backend",
               return_value=ConformingFakeBackend()), patch(
        "trajectory_editor.episode_cli.TerminalIO", return_value=io,
    ):
        assert main([
            "--workspace", str(path), "--model", "fake",
            "--new-prompt-file", str(prompt_file), "--episode-id", "prompt-file",
        ]) == 0
    with EpisodeStore(path) as store:
        assert store.get_episode("prompt-file")["initial_text"] == "P\r\nQ\n"


def test_bare_new_cancellation_returns_to_the_session_edge(tmp_path):
    io = ScriptedIO(["new", None, "q"])
    engine = EpisodeEngine(
        ConformingFakeBackend(), sampling=SamplerConfig(),
        initial_text="P", initial_token_ids=[7],
    )
    with EpisodeStore(tmp_path / "episodes.db") as store:
        store.create_episode(
            episode_id="source", initial_text="P", initial_token_ids=[7],
            sampling=engine.sampling, stream_fingerprint=engine.stream_fingerprint,
            max_tokens=None, backend={},
        )
        session = LiveSession(engine)
        assert session_edge_menu(io, session, store=store) == ("quit", None)
        assert store.workspace_list().count("#") == 1

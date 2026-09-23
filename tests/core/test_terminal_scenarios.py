"""The same command flow reaches durable and ephemeral owners through either terminal."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.core.runtime_helpers import LiveScriptedIO
from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.episode_cli import main
from trajectory_editor.episode_replay_source import replay_procedure
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.teacher_plan import load_teacher_tape_jsonl

pytestmark = pytest.mark.current_workflow

@pytest.mark.parametrize("ephemeral", [False, True])
def test_shared_command_scenario_through_live_and_plain_adapters(tmp_path, ephemeral):
    records = []
    for live in (False, True):
        suffix = "live" if live else "plain"
        workspace = tmp_path / f"{suffix}.sqlite3"
        export = tmp_path / f"{suffix}.jsonl"
        commands = ["1", "q"]
        commands += [f"export {export}", "q"] if ephemeral else ["q"]
        terminal = LiveScriptedIO(commands) if live else ScriptedIO(commands)
        flags = ["--ephemeral"] if ephemeral else ["--workspace", str(workspace)]
        if not live:
            flags.append("--plain-ui")
        with patch(
            "trajectory_editor.episode_backend_loader.load_backend",
            side_effect=lambda _args: ConformingFakeBackend(),
        ), patch("trajectory_editor.episode_cli.TerminalIO", return_value=terminal):
            assert main(["--model", "fake", "--new-prompt", "P", *flags]) == 0

        assert terminal.choice_requests
        assert terminal.edge_requests
        assert all(request.mode == ("session" if ephemeral else "episode")
                   for request in terminal.edge_requests)
        assert not terminal.responses
        if live:
            assert terminal.entered == 1
        if ephemeral:
            assert not workspace.exists()
            actions = [step.action.kind for step in load_teacher_tape_jsonl(export).plan]
        else:
            with EpisodeStore(workspace) as store:
                actions = [step["action"].kind for step in replay_procedure(
                    store, store.resolve_id("#1"))]
        records.append((actions, terminal.choice_requests[0].choice.proposal_token_id))

    assert records[0] == records[1]

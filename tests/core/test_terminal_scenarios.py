"""The same command flow reaches live and workspace-backed sessions through either terminal."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from tests.core.runtime_helpers import LiveScriptedIO
from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.beam import BeamSearch, beam_menu
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_cli import main
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_replay_source import replay_procedure
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.terminal_contracts import BeamInput
from trajectory_editor.teacher_plan import load_teacher_tape_jsonl

pytestmark = pytest.mark.current_workflow


def test_plain_beam_menu_renders_frontier_and_replays_typed_branch():
    engine = EpisodeEngine(
        ConformingFakeBackend(),
        initial_text="P",
        sampling=SamplerConfig(temperature=0.0),
    )
    beam = BeamSearch(engine, width=2)
    selected = beam.ordered_paths()[0]
    selected_label = selected.label
    expected_token_ids = tuple(selected.engine.token_ids)
    terminal = ScriptedIO([selected_label])

    result, actions = beam_menu(terminal, beam)

    assert result == "select"
    assert actions
    request = terminal.prompt_requests[0]
    assert request.isolated
    assert "Shared context (last 4 lines):" in request.body
    assert f"{selected_label}" in request.body
    assert "Survivors:" in request.body

    replay = EpisodeEngine(
        ConformingFakeBackend(),
        initial_text="P",
        sampling=SamplerConfig(temperature=0.0),
    )
    for action in actions:
        replay.apply(action)
    assert tuple(replay.token_ids) == expected_token_ids


def test_advancing_beam_keeps_the_selected_leaderboard_row_in_place():
    engine = EpisodeEngine(
        ConformingFakeBackend(),
        initial_text="P",
        sampling=SamplerConfig(temperature=0.0),
    )
    beam = BeamSearch(engine, width=2)

    class Terminal:
        def __init__(self):
            self.states = []

        def read_beam(self, state):
            self.states.append(state)
            if len(self.states) == 1:
                # Right-arrow advances one step while carrying the highlighted
                # row's branch ID back through the presentation boundary.
                return BeamInput("advance 1", state.rows[1].label)
            return BeamInput("return", state.selected_label)

    terminal = Terminal()
    beam_menu(terminal, beam)

    before, after = terminal.states
    assert before.rows[1].state == "LIVE"
    assert after.selected_label == after.rows[1].label


def test_beam_recent_steps_roll_forward_to_the_newest_five_tokens():
    class CyclicBackend(ConformingFakeBackend):
        def last_logits(self):
            next_token = {
                1: 2,
                2: 3,
                3: 4,
                4: 5,
                5: 6,
                6: 1,
                7: 1,
            }[self.tokens[-1]]
            logits = np.full(self.vocabulary_size(), -30.0, dtype=np.float32)
            logits[next_token] = 10.0
            logits[0] = 0.0
            return logits

    engine = EpisodeEngine(
        CyclicBackend(),
        initial_text="P",
        sampling=SamplerConfig(temperature=0.0),
    )
    beam = BeamSearch(engine, width=1)
    assert beam.advance(7)

    live_row = next(row for row in beam.view_state().rows if row.state == "LIVE")
    recent_tokens = tuple(
        step.split("”", 1)[0].removeprefix("“") for step in live_row.recent_steps
    )
    assert recent_tokens == (" hello", "!", "?", " A", " B")


@pytest.mark.parametrize("workspace_enabled", [False, True])
def test_shared_command_scenario_through_live_and_plain_adapters(tmp_path, workspace_enabled):
    records = []
    for live in (False, True):
        suffix = "live" if live else "plain"
        workspace = tmp_path / f"{suffix}.sqlite3"
        export = tmp_path / f"{suffix}.jsonl"
        commands = ["1", "q"]
        commands += (
            [f"export {export}", "q"] if not workspace_enabled
            else [f"save {workspace} saved", "q"]
        )
        terminal = LiveScriptedIO(commands) if live else ScriptedIO(commands)
        flags = ["--workspace", str(workspace)] if workspace_enabled else []
        if not live:
            flags.append("--plain-ui")
        with patch(
            "trajectory_editor.episode_backend_loader.load_backend",
            side_effect=lambda _args: ConformingFakeBackend(),
        ), patch("trajectory_editor.episode_cli.TerminalIO", return_value=terminal):
            assert main(["--model", "fake", "--new-prompt", "P", *flags]) == 0

        assert terminal.choice_requests
        assert terminal.edge_requests
        assert all(request.mode == "session"
                   for request in terminal.edge_requests)
        assert not terminal.responses
        if live:
            assert terminal.entered == 1
        if not workspace_enabled:
            assert not workspace.exists()
            actions = [step.action.kind for step in load_teacher_tape_jsonl(export).plan]
        else:
            with EpisodeStore(workspace) as store:
                actions = [step["action"].kind for step in replay_procedure(store, "saved")]
        records.append((actions, terminal.choice_requests[0].choice.proposal_token_id))

    assert records[0] == records[1]


def test_beam_kill_backfills_same_depth_and_preserves_survivor_ids():
    runtime = EpisodeEngine(ConformingFakeBackend(), initial_token_ids=[7], sampling=SamplerConfig())
    beam = BeamSearch(runtime, width=2)
    survivors = {beam._path_token_ids(path): path.label for path in beam.active}
    victim = beam.active[0]
    killed = beam._path_token_ids(victim)
    survivor = beam.active[1]
    assert beam.kill(victim.label)
    assert len(beam.active) == 2
    assert all(len(beam._path_token_ids(path)) == 1 for path in beam.active)
    assert killed not in {beam._path_token_ids(path) for path in beam.active}
    assert next(path.label for path in beam.active if beam._path_token_ids(path) == beam._path_token_ids(survivor)) == survivors[beam._path_token_ids(survivor)]
    assert len(beam._history) == 1
    # Repeated pruning must reach candidates beyond the original per-parent pool.
    seen = {killed}
    for _ in range(4):
        victim = beam.active[0]
        seen.add(beam._path_token_ids(victim))
        assert beam.kill(victim.label)
        assert seen.isdisjoint({beam._path_token_ids(path) for path in beam.active})
    beam.discard()


def test_beam_refill_survives_advance_rewind_and_protection():
    runtime = EpisodeEngine(ConformingFakeBackend(), initial_token_ids=[7], sampling=SamplerConfig())
    beam = BeamSearch(runtime, width=3)
    protected = beam.active[-1]
    protected_tokens = beam._path_token_ids(protected)
    beam.toggle_protection(protected.label)
    assert beam.kill(beam.active[0].label)
    assert protected_tokens in {beam._path_token_ids(path) for path in beam.active}
    assert beam.advance(2)
    assert all(len(beam._path_token_ids(path)) == 3 for path in beam.active)
    assert beam.rewind()
    assert all(len(beam._path_token_ids(path)) == 1 for path in beam.active)
    assert beam.kill(beam.active[0].label)
    assert len(beam.active) == 3
    assert all(len(beam._path_token_ids(path)) == 1 for path in beam.active)
    beam.discard()

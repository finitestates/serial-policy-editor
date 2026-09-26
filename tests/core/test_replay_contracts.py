from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.core.actions import Accept, EndGeneration, Hold, Phrase, Write
from trajectory_editor.core.results import ReplayExpectation
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_cli import main
from trajectory_editor.episode_lifecycle import _restore_engine
from trajectory_editor.episode_materializer import materialize_live_branch
from trajectory_editor.episode_replay_source import build_source_replay_recipe, replay_procedure
from trajectory_editor.run_loop import (
    ReplayContext, ReplayPlan, TapeStep, run_plan,
)
from trajectory_editor.episode_session import LiveSession
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_ui import InteractivePolicy
from trajectory_editor.run_loop import EdgeRequested
from trajectory_editor.spr_recipe import (
    ReplayControlPolicy,
    ReplayPlacement,
    compose_replay_plan,
)
from tests.core.runtime_helpers import PhraseBackend

pytestmark = pytest.mark.invariant

class SequenceBackend(ConformingFakeBackend):
    pieces = {
        0: "<eog>",
        1: "word",
        2: ".",
        3: " next",
        4: "\n",
        5: "more",
        6: "?",
        7: "P",
    }

    def __init__(self, sequence):
        super().__init__()
        self.sequence = sequence

    def last_logits(self):
        position = len(self.tokens) - 1
        token = self.sequence[min(position, len(self.sequence) - 1)]
        logits = np.full(8, -100.0)
        logits[token] = 100.0
        return logits


def runtime(sequence):
    return EpisodeEngine(
        SequenceBackend(sequence),
        initial_token_ids=[7],
        sampling=SamplerConfig(temperature=0.0),
    )


def create(store, episode_id, episode):
    return store.create_episode(
        episode_id=episode_id,
        initial_text=episode.initial_text,
        initial_token_ids=list(episode.initial_token_ids),
        sampling=episode.sampling,
        stream_fingerprint=episode.stream_fingerprint,
        max_tokens=episode.max_tokens,
        backend=episode.backend.provenance(),
    )


def engine(prefix=7):
    return EpisodeEngine(
        ConformingFakeBackend(),
        initial_token_ids=[prefix],
        sampling=SamplerConfig(temperature=0.0),
    )


def create_legacy(store, episode, identifier="test"):
    return store.create_episode(
        episode_id=identifier,
        initial_text=episode.text,
        initial_token_ids=list(episode.initial_token_ids),
        sampling=episode.sampling,
        stream_fingerprint=episode.stream_fingerprint,
        max_tokens=None,
        backend={},
    )


def tape(store, episode_id):
    return [
        TapeStep(step["action"], step["expectation"])
        for step in replay_procedure(store, episode_id)
    ]


class NeverChoose:
    def choose(self, *args):
        raise AssertionError("replay should reach the live edge before requesting input")


def test_live_session_executes_without_a_store():
    session = LiveSession(runtime([1, 3, 5]))

    class LiveWrite:
        def choose(self, *args):
            return Hold(1)

    result = run_plan(session, divergence_policy="handoff",
        live_policy=LiveWrite(),
        max_live_actions=1,
    )

    assert result.replayed_actions == 0
    assert len(result.outcomes) == 1
    assert session.history_visible_token_ids == (1,)


def test_interactive_bias_before_first_action_survives_save_and_replay(tmp_path):
    session = LiveSession(engine())
    result = run_plan(session, divergence_policy="handoff",
        live_policy=InteractivePolicy(io=ScriptedIO(["2+100", "h 1"]), menu_size=3),
        max_live_actions=1,
    )

    assert result.outcomes[0].visible_token_ids == (2,)
    assert [(boundary, sampling.bias_rules) for boundary, sampling, _ in session.sampler_states] == [
        (0, session.sampler.bias_rules),
    ]

    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        source_id = materialize_live_branch(
            store, session, session.branch_state(), {}, episode_id="biased"
        )
        segments = store.sampler_segments(source_id)
        assert [segment["start_boundary"] for segment in segments] == [0]
        recipe = build_source_replay_recipe(store, source_id)

    plan = compose_replay_plan(
        recipe, ReplayPlacement.SOURCE_ROOT, ReplayControlPolicy.FOLLOW_SOURCE
    )
    replay = run_plan(LiveSession(engine()), divergence_policy="handoff", tape=plan)

    assert replay.replayed_actions == 1
    assert replay.outcomes[0].visible_token_ids == (2,)
    assert not replay.handed_off


def test_execution_history_stays_in_memory_until_explicit_save(tmp_path):
    session = LiveSession(runtime([1, 3, 5]))

    class LiveWrite:
        def choose(self, *args):
            return Hold(1)

    workspace = tmp_path / "episodes.sqlite3"
    with EpisodeStore(workspace) as store:
        result = run_plan(
            session,
            divergence_policy="handoff",
            live_policy=LiveWrite(),
            max_live_actions=1,
        )
        assert result.outcomes[0].visible_token_ids == (1,)
        assert session.history_visible_token_ids == (1,)
        assert store.workspace_list(include_finished=True) == "No open episodes."

        identifier = materialize_live_branch(
            store, session, session.branch_state(), {}, episode_id="saved"
        )
        assert identifier == "saved"
        assert [row["token_id"] for row in store.tokens(identifier)] == [1]
        assert len(store.actions(identifier)) == 1


def test_teacher_input_eof_yields_to_an_unsealed_in_memory_edge():
    session = LiveSession(runtime([1, 3, 5]))
    with pytest.raises(EdgeRequested):
        run_plan(
            session,
            divergence_policy="handoff",
            live_policy=InteractivePolicy(io=ScriptedIO([None])),
        )
    assert session.status == "open"
    assert session.history_outcomes == ()


@pytest.mark.parametrize(
    "action, reason",
    [(Accept(), "teacher-eog"), (Hold(1), "model-eog")],
)
def test_genuine_eog_is_saved_only_after_explicit_materialization(
    tmp_path, action, reason
):
    class ChooseAction:
        def choose(self, *args):
            return action

    session = LiveSession(runtime([0]))
    result = run_plan(
        session,
        divergence_policy="handoff",
        live_policy=ChooseAction(),
    )
    assert session.engine.ended
    assert session.engine.terminal_reason == reason

    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        assert store.workspace_list(include_finished=True) == "No open episodes."
        identifier = materialize_live_branch(
            store, session, session.branch_state(), {}, episode_id="eog"
        )
        saved = store.get_episode(identifier)
        assert saved["status"] == "completed"
        assert saved["terminal_reason"] == reason


def test_cli_persists_only_when_the_edge_save_command_is_used(tmp_path):
    workspace = tmp_path / "episodes.sqlite3"
    io = ScriptedIO(["h 1", "q", f"save {workspace}", "q"])
    with patch(
        "trajectory_editor.episode_backend_loader.load_backend",
        side_effect=lambda _args: SequenceBackend([1]),
    ), patch("trajectory_editor.episode_cli.TerminalIO", return_value=io):
        assert main([
            "--workspace", str(workspace), "--model", "fake", "--plain-ui",
            "--new-prompt", "P", "--episode-id", "terminal",
        ]) == 0

    with EpisodeStore(workspace) as store:
        saved = store.get_episode("terminal")
        assert saved["status"] == "open"
        assert saved["visible_text"] == "word"
        assert len(store.actions("terminal")) == 1


def test_replay_plan_uses_source_sampling_on_an_inactive_branch():
    session = LiveSession(runtime([1, 3, 5]), branch_id="root")
    session.generate(Accept())
    child = session.fork(boundary=1, branch_id="child")
    session.activate("root")

    first = SamplerConfig(temperature=0.0, seed=11)
    second = SamplerConfig(temperature=0.0, seed=22)
    final = SamplerConfig(temperature=0.0, seed=33)
    plan = ReplayPlan(
        steps=(TapeStep(Hold(1), None), TapeStep(Hold(1), None)),
        context=ReplayContext(sampling=(first, second)),
        final_sampling=final,
    )

    result = run_plan(child, divergence_policy="handoff", tape=plan, live_policy=NeverChoose())

    assert result.replayed_actions == 2
    assert result.replay_exhausted
    assert len(result.outcomes) == 2
    assert child.history_visible_token_ids == (1, 3, 5)
    assert [(boundary, sampling) for boundary, sampling, _ in child.sampler_states] == [
        (0, SamplerConfig(temperature=0.0)),
        (1, first),
        (2, second),
        (3, final),
    ]
    session.activate("root")
    assert session.history_visible_token_ids == (1,)


def test_r01_exact_replay_reproduces_the_recorded_visible_prefix(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        source = runtime([1, 3, 5])
        source_id = create(store, "source", source)
        source_outcome = source.apply(Hold(2))
        store.record_action(source_id, 0, source_outcome)

        target = LiveSession(runtime([1, 3, 5]))
        result = run_plan(target, divergence_policy="handoff", tape=tape(store, source_id))

    assert result.replayed_actions == 1
    assert target.engine.visible_token_ids == [1, 3]
    assert result.outcomes[0].expectation() == source_outcome.expectation()


def test_r05_check_and_force_replay_as_recorded_writes(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        source = EpisodeEngine(
            PhraseBackend(),
            initial_token_ids=[7],
            sampling=SamplerConfig(temperature=0.0),
        )
        source_id = create(store, "source", source)
        outcome = source.apply(Phrase("C!", mode="exact", force=True, max_shift=0.5))
        store.record_action(source_id, 0, outcome)
        target = LiveSession(EpisodeEngine(
            PhraseBackend(),
            initial_token_ids=[7],
            sampling=SamplerConfig(temperature=0.0),
        ))
        result = run_plan(target, divergence_policy="handoff", tape=tape(store, source_id))

    assert result.outcomes[0].action.kind == "force-phrase"
    assert target.engine.visible_token_ids == [3, 5]


def test_r06_replay_eog_reaches_the_live_edge_without_committing_terminal(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        target = LiveSession(runtime([0]))
        result = run_plan(
            target, divergence_policy="handoff",
            tape=[TapeStep(EndGeneration(), ReplayExpectation((), 0, "eog"))],
            live_policy=NeverChoose(),
        )

    assert result.handed_off
    assert target.engine.visible_token_ids == []
    assert target.engine.terminal_token_id is None
    assert result.outcomes[0].stop_reason == "replay-eog"


def test_r07_editorial_moves_never_become_replay_steps(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        source = runtime([1, 3, 5])
        source_id = create(store, "source", source)
        store.record_action(source_id, 0, source.apply(Accept()))
        store.record_interaction(source_id, 1, "search", {"query": "word"})
        store.record_interaction(source_id, 1, "fork-requested", {"boundary": 0})
        store.record_interaction(source_id, 1, "rewind-requested", {"boundary": 0})

        tape = replay_procedure(store, source_id)

    assert len(tape) == 1
    assert tape[0]["action"] == Accept()


@pytest.mark.parametrize("as_plan", [False, True])
def test_r08_exhausted_replay_yields_to_a_usable_live_edge(tmp_path, as_plan):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        source = runtime([1, 3, 5])
        source_id = create(store, "source", source)
        store.record_action(source_id, 0, source.apply(Accept()))
        target = LiveSession(runtime([1, 3, 5]))

        class LiveWrite:
            def choose(self, *args):
                return Write("hello", mode="exact")

        steps = tape(store, source_id)
        result = run_plan(
            target, divergence_policy="handoff",
            tape=ReplayPlan(tuple(steps)) if as_plan else steps,
            live_policy=NeverChoose(),
            max_live_actions=1,
        )
        assert result.replay_exhausted
        assert target.engine.visible_token_ids == [1]
        assert len(result.outcomes) == 1

        resumed = run_plan(
            target, divergence_policy="handoff",
            live_policy=LiveWrite(), max_live_actions=1
        )

    assert not resumed.replay_exhausted
    assert target.engine.visible_token_ids == [1, 4]
    assert len(resumed.outcomes) == 1


def test_r09_replay_never_mutates_the_recorded_source_prefix(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        source = runtime([1, 3, 5])
        source_id = create(store, "source", source)
        store.record_action(source_id, 0, source.apply(Accept()))
        before = store.tokens(source_id)

        target = LiveSession(runtime([2, 3, 5]))
        result = run_plan(
            target, divergence_policy="handoff", tape=tape(store, source_id)
        )
        after = store.tokens(source_id)

    assert result.handed_off
    assert after == before


@pytest.mark.parametrize(
    "expected",
    [
        ReplayExpectation((1,), None, "requested-length"),
        ReplayExpectation((1, 3, 5, 1, 3), None, "requested-length"),
        ReplayExpectation((5, 5, 5), None, "requested-length"),
        ReplayExpectation((1, 3, 5), None, "newline-boundary"),
    ],
)
def test_r04_ballistic_replay_uses_teacher_actions_after_divergence(expected):
    baseline = runtime([1, 3, 5]).apply(Hold(3))
    result = runtime([1, 3, 5]).apply(
        Hold(3),
        replay=True,
        expectation=expected,
        divergence_policy="ballistic",
    )

    assert result.visible_token_ids == baseline.visible_token_ids == (1, 3, 5)
    assert result.stop_reason == baseline.stop_reason == "requested-length"
    assert result.status == "completed-with-divergence"


@pytest.mark.parametrize(
    "expected, committed",
    [
        ((1,), (1,)),
        ((1, 5, 5), (1,)),
        ((1, 3, 5, 1), (1, 3, 5)),
    ],
)
def test_r03_handoff_replay_stops_at_the_first_divergent_token(expected, committed):
    result = runtime([1, 3, 5]).apply(
        Hold(3),
        replay=True,
        expectation=ReplayExpectation(expected),
        divergence_policy="handoff",
    )

    assert result.visible_token_ids == committed
    assert result.status == "handed-off"
    assert result.divergence is not None


@pytest.mark.parametrize(
    "boundary, sequence, committed",
    [
        ("newline", [1, 4, 5], (1, 4)),
        ("sentence", [1, 2, 3], (1, 2)),
    ],
)
def test_r10_replay_observes_current_text_stop_conditions(
    boundary, sequence, committed
):
    baseline = runtime(sequence).apply(Hold(6, boundary))
    result = runtime(sequence).apply(
        Hold(6, boundary),
        replay=True,
        expectation=ReplayExpectation((1,), None, "requested-length"),
        divergence_policy="ballistic",
    )

    assert result.visible_token_ids == baseline.visible_token_ids == committed
    assert result.stop_reason == baseline.stop_reason == boundary + "-boundary"

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
from trajectory_editor.episode_runner import (
    EpisodeRunner,
    LiveSessionRunner,
    ReplayContext,
    ReplayPlan,
    TapeStep,
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
from tests.core.test_lifecycle_contracts import PhraseBackend


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
        coordinate_offset=episode.coordinate_offset,
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
        coordinate_offset=0,
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


def test_r00_ephemeral_runner_uses_the_same_execution_path_without_a_store():
    session = LiveSession(runtime([1, 3, 5]))

    class LiveWrite:
        def choose(self, *args):
            return Hold(1)

    result = LiveSessionRunner(session).run(
        live_policy=LiveWrite(),
        max_live_actions=1,
    )

    assert result.replayed_actions == 0
    assert len(result.outcomes) == 1
    assert session.history_visible_token_ids == (1,)


def test_interactive_bias_before_first_action_survives_save_and_replay(tmp_path):
    session = LiveSession(engine())
    result = LiveSessionRunner(session).run(
        live_policy=InteractivePolicy(io=ScriptedIO(["2+100", "h 1"]), menu_size=3),
        max_live_actions=1,
    )

    assert result.outcomes[0].visible_token_ids == (2,)
    assert [(boundary, sampling.bias_rules) for boundary, sampling, _, _ in session.sampler_states] == [
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
    replay = LiveSessionRunner(LiveSession(engine())).run(tape=plan)

    assert replay.replayed_actions == 1
    assert replay.outcomes[0].visible_token_ids == (2,)
    assert not replay.handed_off


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, RuntimeError, EOFError])
def test_aborted_run_leaves_recorded_durable_actions_resumable(tmp_path, error_type):
    class AbortAfterOneAction:
        def __init__(self):
            self.choices = 0

        def choose(self, *args):
            self.choices += 1
            if self.choices == 2:
                raise error_type("run stopped")
            return Hold(1)

    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        first = runtime([1, 3, 5])
        episode_id = create(store, "interrupted", first)
        with pytest.raises(error_type):
            EpisodeRunner(first, store, episode_id).run(
                live_policy=AbortAfterOneAction()
            )

        episode = store.get_episode(episode_id)
        assert episode["status"] == "running"
        assert episode["finished_at"] is None
        assert [token["token_id"] for token in store.tokens(episode_id)] == [1]

        restored = _restore_engine(
            store,
            episode_id,
            SequenceBackend([1, 3, 5]),
            max_tokens=None,
            sampling_override=None,
        )
        assert restored.visible_token_ids == [1]

        class ContinueOneAction:
            def choose(self, *args):
                return Hold(1)

        EpisodeRunner(restored, store, episode_id).run(
            live_policy=ContinueOneAction(), max_live_actions=1
        )
        assert restored.visible_token_ids == [1, 3]
        assert len(store.actions(episode_id)) == 2


def test_mid_action_error_resumes_from_last_recorded_boundary(tmp_path):
    class FailingBackend(SequenceBackend):
        def last_logits(self):
            if len(self.tokens) > 1:
                raise RuntimeError("model stopped mid-action")
            return super().last_logits()

    class HoldTwo:
        def choose(self, *args):
            return Hold(2)

    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode = EpisodeEngine(
            FailingBackend([1, 3, 5]),
            initial_token_ids=[7],
            sampling=SamplerConfig(temperature=0.0),
        )
        episode_id = create(store, "partial", episode)
        with pytest.raises(RuntimeError, match="mid-action"):
            EpisodeRunner(episode, store, episode_id).run(live_policy=HoldTwo())

        assert episode.visible_token_ids == [1]
        assert store.actions(episode_id) == []
        assert store.tokens(episode_id) == []
        assert store.get_episode(episode_id)["status"] == "running"

        restored = _restore_engine(
            store,
            episode_id,
            SequenceBackend([1, 3, 5]),
            max_tokens=None,
            sampling_override=None,
        )
        assert restored.visible_token_ids == []


def test_teacher_input_eof_yields_to_an_unsealed_edge(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode = runtime([1, 3, 5])
        episode_id = create(store, "input-eof", episode)

        with pytest.raises(EdgeRequested):
            EpisodeRunner(episode, store, episode_id).run(
                live_policy=InteractivePolicy(io=ScriptedIO([None]))
            )

        saved = store.get_episode(episode_id)
        assert saved["status"] == "open"
        assert saved["finished_at"] is None
        assert saved["terminal_reason"] is None


@pytest.mark.parametrize(
    "action, reason",
    [(Accept(), "teacher-eog"), (Hold(1), "model-eog")],
)
def test_genuine_eog_completes_and_seals_durable_episode(tmp_path, action, reason):
    class ChooseAction:
        def choose(self, *args):
            return action

    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode = runtime([0])
        episode_id = create(store, "eog", episode)
        EpisodeRunner(episode, store, episode_id).run(live_policy=ChooseAction())

        saved = store.get_episode(episode_id)
        assert saved["status"] == "completed"
        assert saved["finished_at"] is not None
        assert saved["terminal_reason"] == reason


@pytest.mark.parametrize(
    "commands, sequence, reason",
    [(["q", "end"], [1], "menu-end"), (["h 1"], [0], "model-eog")],
)
def test_cli_completes_durable_episode_once(tmp_path, commands, sequence, reason):
    finishes = []

    class CountingStore(EpisodeStore):
        def finish_episode(self, *args, **kwargs):
            finishes.append(args[0])
            return super().finish_episode(*args, **kwargs)

    workspace = tmp_path / "episodes.sqlite3"
    io = ScriptedIO(commands)
    with patch("trajectory_editor.episode_cli.EpisodeStore", CountingStore), patch(
        "trajectory_editor.episode_backend_loader.load_backend",
        side_effect=lambda _args: SequenceBackend(sequence),
    ), patch("trajectory_editor.episode_cli.TerminalIO", return_value=io):
        assert main([
            "--workspace", str(workspace), "--model", "fake", "--plain-ui",
            "--new-prompt", "P", "--episode-id", "terminal",
        ]) == 0

    with EpisodeStore(workspace) as store:
        saved = store.get_episode("terminal")
        assert saved["status"] == "completed"
        assert saved["terminal_reason"] == reason
    assert finishes == ["terminal"]


@pytest.mark.parametrize(
    "raises_eof, expected_exit, expected_status",
    [(False, 0, "open"), (True, 2, "running")],
)
def test_cli_input_eof_never_seals(tmp_path, raises_eof, expected_exit, expected_status):
    class ClosedInput(ScriptedIO):
        def read(self, prompt):
            raise EOFError("terminal input closed")

    workspace = tmp_path / "episodes.sqlite3"
    io = ClosedInput([]) if raises_eof else ScriptedIO([None, None])
    with patch(
        "trajectory_editor.episode_backend_loader.load_backend",
        side_effect=lambda _args: SequenceBackend([1]),
    ), patch("trajectory_editor.episode_cli.TerminalIO", return_value=io):
        assert main([
            "--workspace", str(workspace), "--model", "fake", "--plain-ui",
            "--new-prompt", "P", "--episode-id", "input-eof",
        ]) == expected_exit

    with EpisodeStore(workspace) as store:
        saved = store.get_episode("input-eof")
        assert saved["status"] == expected_status
        assert saved["finished_at"] is None
        assert saved["terminal_reason"] is None


def test_ephemeral_replay_plan_uses_source_sampling_on_an_inactive_branch():
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

    result = LiveSessionRunner(child).run(tape=plan, live_policy=NeverChoose())

    assert result.episode_id == "child"
    assert result.replayed_actions == 2
    assert result.replay_exhausted
    assert len(result.outcomes) == 2
    assert child.history_visible_token_ids == (1, 3, 5)
    assert [(boundary, sampling) for boundary, sampling, _, _ in child.sampler_states] == [
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

        target = runtime([1, 3, 5])
        target_id = create(store, "target", target)
        result = EpisodeRunner(target, store, target_id).run(
            tape=tape(store, source_id)
        )

    assert result.replayed_actions == 1
    assert target.visible_token_ids == [1, 3]
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
        target = EpisodeEngine(
            PhraseBackend(),
            initial_token_ids=[7],
            sampling=SamplerConfig(temperature=0.0),
        )
        target_id = create(store, "target", target)
        result = EpisodeRunner(target, store, target_id).run(
            tape=tape(store, source_id)
        )

    assert result.outcomes[0].action.kind == "force-phrase"
    assert target.visible_token_ids == [3, 5]


def test_r06_replay_eog_reaches_the_live_edge_without_committing_terminal(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        target = runtime([0])
        target_id = create(store, "target", target)
        result = EpisodeRunner(target, store, target_id).run(
            tape=[TapeStep(EndGeneration(), ReplayExpectation((), 0, "eog"))],
            live_policy=NeverChoose(),
        )

    assert result.handed_off
    assert target.visible_token_ids == []
    assert target.terminal_token_id is None
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
        target = runtime([1, 3, 5])
        target_id = create(store, "target", target)

        class LiveWrite:
            def choose(self, *args):
                return Write("hello", mode="exact")

        steps = tape(store, source_id)
        result = EpisodeRunner(target, store, target_id).run(
            tape=ReplayPlan(tuple(steps)) if as_plan else steps,
            live_policy=NeverChoose(),
            max_live_actions=1,
        )
        assert result.replay_exhausted
        assert target.visible_token_ids == [1]
        assert len(result.outcomes) == 1
        assert store.get_episode(target_id)["status"] == "replay-edge"

        resumed = EpisodeRunner(target, store, target_id).run(
            live_policy=LiveWrite(), max_live_actions=1
        )

    assert not resumed.replay_exhausted
    assert target.visible_token_ids == [1, 4]
    assert len(resumed.outcomes) == 1


def test_r09_replay_never_mutates_the_recorded_source_prefix(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        source = runtime([1, 3, 5])
        source_id = create(store, "source", source)
        store.record_action(source_id, 0, source.apply(Accept()))
        before = store.tokens(source_id)

        target = runtime([2, 3, 5])
        target_id = create(store, "target", target)
        result = EpisodeRunner(
            target, store, target_id, divergence_policy="handoff"
        ).run(
            tape=tape(store, source_id),
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

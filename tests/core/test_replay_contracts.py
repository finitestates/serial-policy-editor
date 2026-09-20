from __future__ import annotations

import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor.core.actions import Accept, EndGeneration, Hold, Phrase, Write
from trajectory_editor.core.results import ReplayExpectation
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_runner import EpisodeRunner, LiveSessionRunner, TapeStep
from trajectory_editor.episode_session import LiveSession
from trajectory_editor.episode_store import EpisodeStore
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
        TapeStep(action, expectation)
        for action, expectation in store.replay_tape(episode_id)
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


def test_r02_replay_tape_contains_only_action_and_optional_result(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        source = runtime([1, 3, 5])
        source_id = create(store, "source", source)
        outcome = source.apply(Hold(1))
        store.record_action(source_id, 0, outcome)
        store.record_interaction(source_id, 1, "search", {"query": "word"})
        store.record_interaction(source_id, 1, "rewind-requested", {"boundary": 0})
        tape = store.replay_tape(source_id)

    assert len(tape) == 1
    assert isinstance(tape[0][0], Hold)
    assert tape[0][1] == outcome.expectation()


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

        tape = store.replay_tape(source_id)

    assert len(tape) == 1
    assert tape[0][0] == Accept()


def test_r08_exhausted_replay_yields_to_a_usable_live_edge(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        source = runtime([1, 3, 5])
        source_id = create(store, "source", source)
        store.record_action(source_id, 0, source.apply(Accept()))
        target = runtime([1, 3, 5])
        target_id = create(store, "target", target)

        class LiveWrite:
            def choose(self, *args):
                return Write("hello", mode="exact")

        result = EpisodeRunner(target, store, target_id).run(
            tape=tape(store, source_id),
            live_policy=LiveWrite(),
            stop_after_tape=False,
            max_live_actions=1,
        )

    assert result.replay_exhausted
    assert target.visible_token_ids == [1, 4]
    assert len(result.outcomes) == 2


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

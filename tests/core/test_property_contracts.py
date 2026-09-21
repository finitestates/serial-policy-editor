from __future__ import annotations

import random
from dataclasses import replace

import pytest

from tests.core.test_lifecycle_contracts import NoEogBackend, create as create_episode
from tests.core.test_replay_contracts import runtime
from trajectory_editor.core.actions import Accept, Hold
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.core.results import ReplayExpectation
from trajectory_editor.episode_cli import _fork_engine, _rewind_episode
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_replay_source import replay_tape
from trajectory_editor.episode_store import EpisodeStore


def test_q01_generated_core_sampler_records_round_trip_exactly():
    randomizer = random.Random(20260918)
    for _ in range(50):
        config = SamplerConfig(
            temperature=randomizer.choice((0.0, 0.25, 0.8, 1.0, 1.7)),
            top_k=randomizer.randint(1, 16),
            top_p=randomizer.choice((0.1, 0.5, 0.95, 1.0)),
            min_p=randomizer.choice((0.0, 0.01, 0.2)),
            typical_p=randomizer.choice((0.35, 0.8, 1.0)),
            tail_free_z=randomizer.choice((0.35, 0.8, 1.0)),
            draw_kernel=randomizer.choice(("categorical", "gumbel-max")),
            repeat_penalty=randomizer.choice((1.0, 1.1, 1.5)),
            repeat_last_n=randomizer.choice((0, 4, 16, -1)),
            presence_penalty=randomizer.choice((0.0, 0.2)),
            frequency_penalty=randomizer.choice((0.0, 0.4)),
            seed=randomizer.randint(-1000, 1000),
        )
        assert SamplerConfig.from_record(config.to_dict()) == config


def test_q02_generated_replay_prefixes_are_never_changed_by_replay():
    for count in range(1, 5):
        source = runtime([1, 3, 5])
        outcome = source.apply(Hold(count))
        source_prefix = list(source.visible_token_ids)

        target = runtime([1, 3, 5])
        replayed = target.apply(
            Hold(count),
            replay=True,
            expectation=outcome.expectation(),
        )

        assert source.visible_token_ids == source_prefix
        assert target.visible_token_ids == source_prefix
        assert replayed.divergence is None


def test_q03_rewind_then_replay_reproduces_each_retained_prefix(tmp_path):
    for boundary in range(4):
        path = tmp_path / f"rewind-{boundary}.sqlite3"
        with EpisodeStore(path) as store:
            source = runtime([1, 3, 5])
            identifier = create_episode(store, "episode", source)
            outcome = source.apply(Hold(3))
            store.record_action(identifier, 0, outcome)
            store.update_episode(identifier, visible_text="word next more", max_tokens=None)

            _rewind_episode(store, identifier, source, boundary)
            retained = tuple(source.visible_token_ids)
            replay = runtime([1, 3, 5])
            tape = replay_tape(store, identifier)
            if tape:
                replay_action, expectation = tape[0]
                replay.apply(replay_action, replay=True, expectation=expectation)

        assert retained == tuple(replay.visible_token_ids)


@pytest.mark.parametrize("boundary", [0, 1, 2, 3])
def test_q04_fork_at_every_generated_boundary_preserves_the_prefix(tmp_path, boundary):
    with EpisodeStore(tmp_path / f"fork-{boundary}.sqlite3") as store:
        parent = EpisodeEngine(
            NoEogBackend(),
            initial_token_ids=[7],
            sampling=SamplerConfig(temperature=0.0),
        )
        identifier = create_episode(store, "parent", parent)
        outcome = parent.apply(Hold(3))
        store.record_action(identifier, 0, outcome)
        store.update_episode(identifier, visible_text=parent.text, max_tokens=None)

        child = _fork_engine(
            store, identifier, parent, boundary, backend=NoEogBackend(), max_tokens=None
        )

    assert child.initial_token_ids == (7,)
    assert child.visible_token_ids == list(outcome.visible_token_ids[:boundary])


def test_q05_generated_divergence_always_handoffs_or_goes_ballistic():
    for mode in ("handoff", "ballistic"):
        for expected_token, actual_token in ((1, 2), (2, 1), (3, 5)):
            expected = ReplayExpectation((expected_token,))
            result = runtime([actual_token]).apply(
                Accept(),
                replay=True,
                expectation=expected,
                divergence_policy=mode,
            )
            assert result.divergence is not None
            if mode == "handoff":
                assert result.status == "handed-off"
                assert result.visible_token_ids == ()
            else:
                assert result.status == "completed-with-divergence"
                assert len(result.visible_token_ids) == 1

from __future__ import annotations

import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor.core.actions import Accept, Write
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.results import ReplayExpectation
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_live_history import (
    history_from_live,
    truncate_live_history,
)
from trajectory_editor.episode_runner import TapeStep


def engine() -> EpisodeEngine:
    return EpisodeEngine(
        ConformingFakeBackend(),
        initial_token_ids=[7],
        sampling=SamplerConfig(temperature=0.0),
    )


def test_truncate_live_history_replaces_only_the_partial_attempt():
    runtime = engine()
    outcome = runtime.apply(Write(" A B", mode="exact"))
    step = TapeStep(outcome.action, outcome.expectation())

    prefix = truncate_live_history((step,), (outcome,), 1)

    assert prefix.retained_tape == (
        TapeStep(Write(" A", mode="exact"), ReplayExpectation((1,), None, "completed")),
    )
    assert prefix.retained_outcomes[0].visible_token_ids == (1,)
    assert prefix.discarded_tape == (step,)
    assert prefix.discarded_outcomes == (outcome,)


def test_truncate_live_history_discards_a_handoff_at_its_root_boundary():
    runtime = engine()
    outcome = runtime.apply(
        Accept(),
        expectation=ReplayExpectation((2,)),
        replay=True,
    )
    step = TapeStep(Accept(), ReplayExpectation((2,)))

    prefix = truncate_live_history((step,), (outcome,), 0)

    assert prefix.retained_tape == ()
    assert prefix.retained_outcomes == ()
    assert prefix.discarded_tape == (step,)
    assert prefix.discarded_outcomes == (outcome,)


def test_live_history_adapter_rejects_misaligned_records():
    with pytest.raises(EditorError, match="must align"):
        history_from_live((TapeStep(Accept(), None),), ())

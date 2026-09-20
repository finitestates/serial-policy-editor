from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_controls import (
    BudgetState,
    ControlState,
    ControlTimeline,
    ControlTransition,
    SamplerState,
    append_transition,
    effective_state,
    truncate_after,
    validate_budget_pair,
)


FINGERPRINT = "a" * 64


def _state(
    *,
    temperature: float = 1.0,
    coordinate_offset: int = 0,
    allowance: int | None = 8,
    checkpoint_boundary: int | None = 8,
) -> ControlState:
    return ControlState(
        SamplerState(
            SamplerConfig(temperature=temperature),
            FINGERPRINT,
            coordinate_offset,
        ),
        BudgetState(allowance, checkpoint_boundary),
    )


def test_transition_is_effective_at_its_own_boundary():
    initial = _state()
    changed = _state(temperature=0.7)
    timeline = ControlTimeline.from_state(initial).append_transition(3, changed)

    assert timeline.effective_at(2) == initial
    assert timeline.effective_at(3) == changed
    assert effective_state(timeline, 100) == changed


def test_identical_adjacent_state_is_not_recorded():
    initial = _state()
    timeline = ControlTimeline.from_state(initial)

    assert timeline.append_transition(2, initial) is timeline
    assert append_transition(timeline, 4, initial) is timeline
    assert len(timeline.transitions) == 1


def test_same_boundary_change_replaces_the_transition_without_duplicates():
    initial = _state()
    first_change = _state(temperature=0.8)
    second_change = _state(temperature=0.6)
    timeline = ControlTimeline.from_state(initial).append_transition(4, first_change)
    replaced = timeline.append_transition(4, second_change)

    assert [(item.start_boundary, item.state) for item in replaced.transitions] == [
        (0, initial),
        (4, second_change),
    ]


def test_sampler_and_budget_transitions_can_change_at_different_boundaries():
    initial = _state()
    sampler_changed = _state(temperature=0.5)
    budget_changed = _state(allowance=3, checkpoint_boundary=9)
    combined = ControlState(sampler_changed.sampler, budget_changed.budget)
    timeline = (
        ControlTimeline.from_state(initial)
        .append_sampler_transition(2, sampler_changed.sampler)
        .append_budget_transition(5, budget_changed.budget)
    )

    assert timeline.effective_at(4) == sampler_changed
    assert timeline.effective_at(5) == combined
    assert timeline.effective_at(5).sampling.temperature == 0.5
    assert timeline.effective_at(5).allowance == 3


def test_truncation_keeps_retained_boundary_and_root_relative_coordinate():
    initial = _state(coordinate_offset=40)
    changed = _state(temperature=0.5, coordinate_offset=47)
    future = _state(temperature=0.2, coordinate_offset=99)
    timeline = (
        ControlTimeline.from_state(initial)
        .append_transition(3, changed)
        .append_transition(8, future)
    )

    truncated = timeline.truncate_after(3)

    assert truncate_after(timeline, 3) == truncated
    assert [item.start_boundary for item in truncated.transitions] == [0, 3]
    assert truncated.effective_at(3).coordinate_offset == 47
    assert truncated.effective_at(100).coordinate_offset == 47


def test_unlimited_budget_is_an_explicit_valid_state():
    unlimited = BudgetState()
    assert unlimited.unlimited
    assert unlimited.allowance is None
    assert unlimited.checkpoint_boundary is None
    assert validate_budget_pair(12, None, None) == unlimited


def test_timeline_requires_a_root_control_state():
    with pytest.raises(EditorError, match="root transition"):
        ControlTimeline()
    with pytest.raises(EditorError, match="boundary zero"):
        ControlTimeline((ControlTransition(1, _state()),))


def test_value_records_are_immutable():
    state = _state()
    with pytest.raises(FrozenInstanceError):
        state.sampler = state.sampler  # type: ignore[misc]


@pytest.mark.parametrize(
    ("allowance", "checkpoint_boundary"),
    [
        (None, 4),
        (4, None),
        (0, 4),
        (-1, 4),
        (True, 4),
    ],
)
def test_invalid_budget_pairs_are_rejected(allowance, checkpoint_boundary):
    with pytest.raises(EditorError):
        BudgetState(allowance, checkpoint_boundary)


def test_checkpoint_must_not_precede_the_transition_boundary():
    with pytest.raises(EditorError):
        ControlTimeline.from_state(_state()).append_transition(
            9, _state(allowance=2, checkpoint_boundary=8)
        )
    with pytest.raises(EditorError):
        validate_budget_pair(9, 2, 8)


def test_fingerprint_coordinate_and_boundary_inputs_are_validated():
    with pytest.raises(EditorError):
        SamplerState(SamplerConfig(), "not-a-fingerprint", 0)
    with pytest.raises(EditorError):
        SamplerState(SamplerConfig(), FINGERPRINT, -1)
    with pytest.raises(EditorError):
        ControlTimeline.from_state(_state()).append_transition(-1, _state())
    with pytest.raises(EditorError):
        ControlTimeline.from_state(_state()).effective_at(-1)
    with pytest.raises(EditorError):
        ControlTimeline.from_state(_state()).truncate_after(-1)


def test_transitions_must_be_ordered_and_new_future_history_requires_truncation():
    initial = _state()
    changed = _state(temperature=0.3)
    with pytest.raises(EditorError):
        ControlTimeline((
            # Duplicate starts would make the effective state ambiguous.
            ControlTimeline.from_state(initial).transitions[0],
            ControlTimeline.from_state(changed).transitions[0],
        ))

    timeline = ControlTimeline.from_state(initial).append_transition(5, changed)
    with pytest.raises(EditorError):
        timeline.append_transition(2, initial)

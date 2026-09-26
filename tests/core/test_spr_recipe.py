from __future__ import annotations

from dataclasses import replace

import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor import EpisodeEngine, LiveSession, SamplerConfig
from trajectory_editor.core.actions import Write
from trajectory_editor.core.results import ReplayExpectation
from trajectory_editor.episode_controls import (
    BudgetState,
    ControlState,
    ControlTimeline,
    SamplerState,
)
from trajectory_editor.run_loop import TapeStep, run_plan
from trajectory_editor.surviving_procedure import (
    ProcedureStep,
    SurvivingProcedure,
)
from trajectory_editor.spr_recipe import (
    ReplayControlPolicy,
    ReplayPlacement,
    SourceReplayRecipe,
    compose_replay_plan,
)

pytestmark = pytest.mark.invariant

FINGERPRINT = "a" * 64


def control_timeline(
    initial: SamplerConfig,
    transitions: tuple[tuple[int, SamplerConfig], ...] = (),
) -> ControlTimeline:
    timeline = ControlTimeline.from_state(
        ControlState(SamplerState(initial, FINGERPRINT), BudgetState())
    )
    for boundary, sampler in transitions:
        timeline = timeline.append_sampler_transition(
            boundary,
            SamplerState(sampler, FINGERPRINT),
        )
    return timeline


def procedure(*items: tuple[int, str, tuple[int, ...]]) -> SurvivingProcedure:
    return SurvivingProcedure(
        tuple(
            ProcedureStep(
                TapeStep(Write(text, mode="exact"), ReplayExpectation(tokens)),
                boundary,
                source_index,
            )
            for source_index, (boundary, text, tokens) in enumerate(items)
        )
    )


def recipe(
    *,
    prompt: str = "source prompt",
    steps: SurvivingProcedure | None = None,
    controls_value: ControlTimeline | None = None,
    visible_boundary: int | None = None,
    end: int = 0,
    source_id: str | None = "source-1",
) -> SourceReplayRecipe:
    return SourceReplayRecipe(
        source_prompt=prompt,
        procedure=steps if steps is not None else SurvivingProcedure(),
        controls=(
            controls_value
            if controls_value is not None
            else control_timeline(SamplerConfig(temperature=0.1))
        ),
        source_visible_boundary=end if visible_boundary is None else visible_boundary,
        source_end_boundary=end,
        source_id=source_id,
    )


def test_source_root_has_no_prompt_step_and_keeps_procedure_order():
    plan = compose_replay_plan(
        recipe(
            steps=procedure(
                (0, " A", (1,)),
                (1, " B", (2,)),
            ),
            controls_value=control_timeline(
                SamplerConfig(temperature=0.1),
                ((2, SamplerConfig(temperature=0.2)),),
            ),
            end=2,
        ),
        ReplayPlacement.SOURCE_ROOT,
        ReplayControlPolicy.FOLLOW_SOURCE,
    )

    assert [step.action for step in plan.steps] == [
        Write(" A", mode="exact"),
        Write(" B", mode="exact"),
    ]
    assert len(plan.context.sampling) == len(plan.steps) == 2
    assert plan.final_sampling == SamplerConfig(temperature=0.2)


def test_append_prepends_one_exact_prompt_and_preserves_source_order():
    plan = compose_replay_plan(
        recipe(
            prompt="source prompt",
            steps=procedure((0, " C", (3,))),
            end=1,
        ),
        ReplayPlacement.APPEND_TO_CURRENT_BRANCH,
        ReplayControlPolicy.PRESERVE_DESTINATION,
    )

    assert [step.action for step in plan.steps] == [
        Write("source prompt", mode="exact"),
        Write(" C", mode="exact"),
    ]
    assert plan.steps[0].expectation is None
    assert plan.follow_source_sampling is False
    assert plan.final_sampling is None
    assert plan.context.sampling == (None, None)


def test_source_control_transitions_and_overrides_cover_steps_and_trailing_final():
    first = SamplerConfig(temperature=0.1, top_k=11)
    second = SamplerConfig(temperature=0.2, top_k=22)
    trailing = SamplerConfig(temperature=0.3, top_k=33)
    plan = compose_replay_plan(
        recipe(
            steps=procedure(
                (0, " A B", (1, 2)),
                (2, " C", (3,)),
            ),
            controls_value=control_timeline(
                first,
                ((2, second), (3, trailing)),
            ),
            end=3,
        ),
        ReplayPlacement.SOURCE_ROOT,
        ReplayControlPolicy.FOLLOW_SOURCE,
        sampler_overrides={"temperature": 0.9, "top_k": 7},
    )

    assert plan.context.sampling == (
        replace(first, temperature=0.9, top_k=7),
        replace(second, temperature=0.9, top_k=7),
    )
    assert plan.final_sampling == replace(trailing, temperature=0.9, top_k=7)


def test_preserve_destination_has_aligned_none_context_and_no_final_sampler():
    plan = compose_replay_plan(
        recipe(
            prompt="source",
            steps=procedure((0, " A", (1,))),
            end=1,
        ),
        ReplayPlacement.APPEND_TO_CURRENT_BRANCH,
        ReplayControlPolicy.PRESERVE_DESTINATION,
        sampler_overrides={"temperature": 0.2},
    )

    assert plan.context.sampling == (None, None)
    assert len(plan.context.sampling) == len(plan.steps)
    assert plan.final_sampling is None


def test_empty_prompt_and_empty_procedure_produce_empty_aligned_plans():
    source = recipe(prompt="", end=0)

    root = compose_replay_plan(
        source,
        ReplayPlacement.SOURCE_ROOT,
        ReplayControlPolicy.FOLLOW_SOURCE,
    )
    append = compose_replay_plan(
        source,
        ReplayPlacement.APPEND_TO_CURRENT_BRANCH,
        ReplayControlPolicy.PRESERVE_DESTINATION,
    )

    assert root.steps == append.steps == ()
    assert root.context.sampling == ()
    assert append.context.sampling == ()
    assert root.final_sampling == SamplerConfig(temperature=0.1)
    assert append.final_sampling is None


def test_explicit_source_boundary_allows_unobserved_steps_past_root():
    unobserved = SurvivingProcedure(
        (
            ProcedureStep(
                TapeStep(Write(" A", mode="exact"), None),
                boundary=3,
                source_index=0,
            ),
        )
    )
    plan = compose_replay_plan(
        recipe(steps=unobserved, visible_boundary=3, end=3),
        ReplayPlacement.SOURCE_ROOT,
        ReplayControlPolicy.FOLLOW_SOURCE,
    )

    assert plan.steps == unobserved.tape
    assert plan.context.sampling == (SamplerConfig(temperature=0.1),)
    assert plan.final_sampling == SamplerConfig(temperature=0.1)


def test_invalid_boundaries_and_unknown_sampler_overrides_are_rejected():
    with pytest.raises(ValueError, match="must not exceed"):
        compose_replay_plan(
            recipe(visible_boundary=0, end=1),
            ReplayPlacement.SOURCE_ROOT,
            ReplayControlPolicy.FOLLOW_SOURCE,
        )

    with pytest.raises(ValueError, match="after the source end boundary"):
        compose_replay_plan(
            recipe(
                steps=procedure((2, " A", (1,))),
                end=1,
            ),
            ReplayPlacement.SOURCE_ROOT,
            ReplayControlPolicy.FOLLOW_SOURCE,
        )

    with pytest.raises(ValueError, match="extends past"):
        compose_replay_plan(
            recipe(
                steps=procedure((0, " A B", (1, 2))),
                end=1,
            ),
            ReplayPlacement.SOURCE_ROOT,
            ReplayControlPolicy.FOLLOW_SOURCE,
        )

    with pytest.raises(ValueError, match="unknown replay sampler override"):
        compose_replay_plan(
            recipe(),
            ReplayPlacement.SOURCE_ROOT,
            ReplayControlPolicy.FOLLOW_SOURCE,
            sampler_overrides={"not_a_sampler_field": 1},
        )


def test_append_prompt_is_ordinary_rewindable_multi_token_destination_history():
    backend = ConformingFakeBackend()
    destination_prompt = "destination root"
    destination = LiveSession(
        EpisodeEngine(
            backend,
            initial_text=destination_prompt,
            initial_token_ids=[7],
            sampling=SamplerConfig(temperature=0.1),
        ),
        prompt=destination_prompt,
        branch_id="destination",
    )
    destination.generate(Write(" A", mode="exact"))
    original_initial_ids = destination.engine.initial_token_ids

    plan = compose_replay_plan(
        recipe(
            prompt=" A B",
            steps=procedure((0, "C", (3,))),
            end=1,
        ),
        ReplayPlacement.APPEND_TO_CURRENT_BRANCH,
        ReplayControlPolicy.PRESERVE_DESTINATION,
    )
    result = run_plan(destination, divergence_policy="handoff", tape=plan)

    assert result.replayed_actions == 2
    assert destination.engine.initial_token_ids == original_initial_ids == (7,)
    assert destination.prompt == destination_prompt
    assert destination.history_visible_token_ids == (1, 1, 2, 3)

    destination.rewind(2)
    assert destination.history_visible_token_ids == (1, 1)
    assert destination.history_tape[0].action == Write(" A", mode="exact")
    assert destination.history_tape[1] == TapeStep(
        Write(" A", mode="exact"),
        ReplayExpectation((1,), None, "completed"),
    )
    assert destination.engine.backend.render(list(destination.engine.visible_token_ids)) == " A A"

    destination.rewind(1)
    assert destination.history_visible_token_ids == (1,)
    assert len(destination.history_tape) == 1
    assert destination.engine.backend.render(list(destination.engine.visible_token_ids)) == " A"

    destination.rewind(0)
    assert destination.history_visible_token_ids == ()
    assert destination.engine.initial_token_ids == original_initial_ids == (7,)
    assert destination.prompt == destination_prompt

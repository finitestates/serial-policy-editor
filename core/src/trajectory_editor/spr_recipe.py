"""Pure composition of a selected source procedure into a replay plan.

This module is deliberately a small seam between source-history projection and
runtime execution.  It receives a prompt, an already projected procedure, and
root-relative source controls; it does not know how any of those values were
stored or selected.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Any

from .core.actions import Write
from .core.sampler_config import SamplerConfig
from .episode_controls import ControlTimeline
from .run_loop import ReplayContext, ReplayOrigin, ReplayPlan, TapeStep
from .surviving_procedure import SurvivingProcedure


class ReplayPlacement(str, Enum):
    """Where the selected source procedure enters the destination."""

    SOURCE_ROOT = "source-root"
    APPEND_TO_CURRENT_BRANCH = "append-to-current-branch"


class ReplayControlPolicy(str, Enum):
    """Which sampler context the replay plan carries."""

    FOLLOW_SOURCE = "follow-source"
    PRESERVE_DESTINATION = "preserve-destination"


@dataclass(frozen=True)
class SourceReplayRecipe:
    """Selected, storage-neutral source values needed to compose a plan."""

    source_prompt: str
    procedure: SurvivingProcedure
    controls: ControlTimeline
    source_visible_boundary: int
    source_end_boundary: int
    source_id: str | None = None
    incomplete_handoff_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_prompt, str):
            raise TypeError("source prompt must be a string")
        if not isinstance(self.procedure, SurvivingProcedure):
            raise TypeError("source procedure must be a SurvivingProcedure")
        if not isinstance(self.controls, ControlTimeline):
            raise TypeError("source controls must be a ControlTimeline")
        if (
            type(self.source_visible_boundary) is not int
            or self.source_visible_boundary < 0
        ):
            raise ValueError(
                "source visible boundary must be a nonnegative integer"
            )
        if type(self.source_end_boundary) is not int or self.source_end_boundary < 0:
            raise ValueError("source end boundary must be a nonnegative integer")
        if self.source_end_boundary > self.source_visible_boundary:
            raise ValueError(
                "source end boundary must not exceed source visible boundary"
            )
        if self.source_id is not None and not isinstance(self.source_id, str):
            raise TypeError("source id must be a string or None")
        if self.incomplete_handoff_reason is not None and not isinstance(
            self.incomplete_handoff_reason, str
        ):
            raise TypeError("incomplete handoff reason must be a string or None")

_SAMPLER_FIELDS = frozenset(
    field.name for field in fields(SamplerConfig) if field.init
)


def _validate_overrides(
    sampler_overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if sampler_overrides is None:
        return {}
    if not isinstance(sampler_overrides, Mapping):
        raise TypeError("sampler overrides must be a mapping")
    overrides = dict(sampler_overrides)
    unknown = set(overrides) - _SAMPLER_FIELDS
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ValueError(f"unknown replay sampler override field(s): {names}")
    if any(not isinstance(name, str) for name in overrides):
        raise ValueError("replay sampler override fields must be strings")
    return overrides


def _validate_recipe_boundaries(recipe: SourceReplayRecipe) -> None:
    end = recipe.source_end_boundary

    for step in recipe.procedure.steps:
        if type(step.boundary) is not int or step.boundary < 0:
            raise ValueError("procedure step boundary must be a nonnegative integer")
        if step.boundary > end:
            raise ValueError(
                "surviving procedure contains a step after the source end boundary"
            )
        expectation = step.expectation
        if expectation is not None and step.boundary + len(expectation.token_ids) > end:
            raise ValueError(
                "surviving procedure step extends past the source end boundary"
            )


def _resolve_sampler(
    recipe: SourceReplayRecipe,
    boundary: int,
    overrides: Mapping[str, Any],
) -> SamplerConfig:
    return replace(recipe.controls.effective_at(boundary).sampling, **overrides)


def compose_replay_plan(
    recipe: SourceReplayRecipe,
    placement: ReplayPlacement,
    control_policy: ReplayControlPolicy,
    *,
    sampler_overrides: Mapping[str, Any] | None = None,
) -> ReplayPlan:
    """Compose one selected source procedure into a storage-neutral plan.

    ``SOURCE_ROOT`` leaves the source prompt out of the tape because lifecycle
    code uses it as the engine entrance.  ``APPEND_TO_CURRENT_BRANCH`` makes
    that prompt an ordinary exact write, preserving all destination root and
    coordinate state.  The control policy is intentionally required at every
    call site: following source controls and preserving destination controls
    are different placement semantics.
    """

    if not isinstance(recipe, SourceReplayRecipe):
        raise TypeError("recipe must be a SourceReplayRecipe")
    if not isinstance(placement, ReplayPlacement):
        raise TypeError("placement must be a ReplayPlacement")
    if not isinstance(control_policy, ReplayControlPolicy):
        raise TypeError("control policy must be a ReplayControlPolicy")

    _validate_recipe_boundaries(recipe)
    overrides = _validate_overrides(sampler_overrides)

    steps: list[TapeStep] = []
    origins: list[ReplayOrigin | None] = []
    source_samplers: list[SamplerConfig | None] = []

    if (
        placement is ReplayPlacement.APPEND_TO_CURRENT_BRANCH
        and recipe.source_prompt
    ):
        steps.append(TapeStep(Write(recipe.source_prompt, mode="exact"), None))
        origins.append(ReplayOrigin(recipe.source_id, 0, "prompt"))
        source_samplers.append(None)

    for procedure_step in recipe.procedure.steps:
        steps.append(procedure_step.tape_step)
        origins.append(ReplayOrigin(recipe.source_id, procedure_step.boundary))
        if control_policy is ReplayControlPolicy.FOLLOW_SOURCE:
            source_samplers.append(
                _resolve_sampler(recipe, procedure_step.boundary, overrides)
            )
        else:
            source_samplers.append(None)

    if control_policy is ReplayControlPolicy.FOLLOW_SOURCE:
        final_sampling = (
            _resolve_sampler(recipe, recipe.source_end_boundary, overrides)
            if recipe.incomplete_handoff_reason is None else None
        )
        follow_source_sampling = True
    else:
        final_sampling = None
        follow_source_sampling = False

    return ReplayPlan(
        steps=tuple(steps),
        follow_source_sampling=follow_source_sampling,
        final_sampling=final_sampling,
        incomplete_handoff_reason=recipe.incomplete_handoff_reason,
        context=ReplayContext(
            sampling=tuple(source_samplers),
            origins=tuple(origins),
        ),
    )


__all__ = [
    "ReplayControlPolicy",
    "ReplayPlacement",
    "SourceReplayRecipe",
    "compose_replay_plan",
]

"""Compose a selected source procedure into an executable replay plan."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Any

from .core.actions import Reroll, SetSampler, Write
from .core.errors import EditorError
from .core.sampler_config import SamplerConfig
from .run_loop import ReplayPlan, TapeStep
from .surviving_procedure import SurvivingProcedure


class ReplayPlacement(str, Enum):
    SOURCE_ROOT = "source-root"
    APPEND_TO_CURRENT_BRANCH = "append-to-current-branch"


class ReplaySamplerPolicy(str, Enum):
    FOLLOW_SOURCE = "follow-source"
    PRESERVE_DESTINATION = "preserve-destination"


@dataclass(frozen=True)
class SourceReplayRecipe:
    source_prompt: str
    procedure: SurvivingProcedure
    initial_sampling: SamplerConfig
    stream_fingerprint: str
    source_visible_boundary: int
    source_end_boundary: int
    source_id: str | None = None
    incomplete_handoff_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_prompt, str):
            raise TypeError("source prompt must be a string")
        if not isinstance(self.procedure, SurvivingProcedure):
            raise TypeError("source procedure must be a SurvivingProcedure")
        if not isinstance(self.initial_sampling, SamplerConfig):
            raise TypeError("source recipe requires its initial sampler settings")
        if not isinstance(self.stream_fingerprint, str) or not self.stream_fingerprint:
            raise TypeError("source recipe requires its root stream fingerprint")
        if type(self.source_visible_boundary) is not int or self.source_visible_boundary < 0:
            raise ValueError("source visible boundary must be nonnegative")
        if type(self.source_end_boundary) is not int or not 0 <= self.source_end_boundary <= self.source_visible_boundary:
            raise ValueError("source end boundary is outside the visible source")
        if self.source_id is not None and not isinstance(self.source_id, str):
            raise TypeError("source id must be a string or None")


_SAMPLER_FIELDS = frozenset(field.name for field in fields(SamplerConfig) if field.init)


def _validate_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise TypeError("sampler overrides must be a mapping")
    result = dict(overrides)
    if any(not isinstance(name, str) for name in result):
        raise ValueError("sampler override fields must be strings")
    unknown = set(result) - _SAMPLER_FIELDS
    if unknown:
        raise ValueError(f"unknown sampler override field(s): {', '.join(sorted(unknown))}")
    return result


def _validate_recipe_boundaries(recipe: SourceReplayRecipe) -> None:
    for step in recipe.procedure.steps:
        if type(step.boundary) is not int or step.boundary < 0:
            raise ValueError("procedure step boundary must be nonnegative")
        if step.boundary > recipe.source_end_boundary:
            raise ValueError("procedure contains a step after its selected endpoint")
        expectation = step.expectation
        if expectation is not None and step.boundary + len(expectation.token_ids) > recipe.source_end_boundary:
            raise ValueError("procedure step extends past its selected endpoint")


def compose_replay_plan(
    recipe: SourceReplayRecipe,
    placement: ReplayPlacement,
    sampler_policy: ReplaySamplerPolicy,
    *,
    sampler_overrides: Mapping[str, Any] | None = None,
) -> ReplayPlan:
    """Compose prompt placement and sampler-command policy into one tape."""
    if not isinstance(recipe, SourceReplayRecipe):
        raise TypeError("recipe must be a SourceReplayRecipe")
    if not isinstance(placement, ReplayPlacement):
        raise TypeError("placement must be a ReplayPlacement")
    if not isinstance(sampler_policy, ReplaySamplerPolicy):
        raise TypeError("sampler policy must be a ReplaySamplerPolicy")
    _validate_recipe_boundaries(recipe)
    overrides = _validate_overrides(sampler_overrides)

    steps: list[TapeStep] = []
    if placement is ReplayPlacement.APPEND_TO_CURRENT_BRANCH:
        if recipe.source_prompt:
            steps.append(TapeStep(Write(recipe.source_prompt, mode="exact"), None))
        if sampler_policy is ReplaySamplerPolicy.FOLLOW_SOURCE:
            steps.append(
                TapeStep(SetSampler(replace(recipe.initial_sampling, **overrides)), None)
            )

    for procedure_step in recipe.procedure.steps:
        tape_step = procedure_step.tape_step
        if isinstance(tape_step.action, SetSampler):
            if sampler_policy is ReplaySamplerPolicy.PRESERVE_DESTINATION:
                continue
            tape_step = TapeStep(
                SetSampler(replace(tape_step.action.sampling, **overrides)),
                tape_step.expectation,
            )
        elif (
            isinstance(tape_step.action, Reroll)
            and sampler_policy is ReplaySamplerPolicy.PRESERVE_DESTINATION
        ):
            continue
        elif isinstance(tape_step.action, Reroll) and "seed" in overrides:
            tape_step = TapeStep(Reroll(overrides["seed"]), tape_step.expectation)
        steps.append(tape_step)

    return ReplayPlan(
        steps=tuple(steps),
        incomplete_handoff_reason=recipe.incomplete_handoff_reason,
    )


__all__ = [
    "ReplayPlacement",
    "ReplaySamplerPolicy",
    "SourceReplayRecipe",
    "compose_replay_plan",
]

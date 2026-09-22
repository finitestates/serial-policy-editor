"""Adapt durable episode records into source-relative replay semantics.

This module turns an episode reader's primitive action, token, and control
records into a surviving procedure and a :class:`SourceReplayRecipe`.  It
does not know whether those records came from SQLite, an import, or another
persistence adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .core.actions import Hold, Phrase, PolicyAction, Write, action_from_dict
from .core.errors import EditorError
from .core.results import ReplayExpectation
from .core.sampler_config import SamplerConfig
from .episode_controls import ControlState, ControlTimeline, ControlTransition
from .episode_runner import TapeStep
from .spr_recipe import SourceReplayRecipe
from .surviving_procedure import (
    ProcedureRecord,
    ProcedureStep,
    SurvivingProcedure,
    project_surviving_procedure,
)


class EpisodeReplayReader(Protocol):
    """Primitive reads required to construct a durable replay source."""

    def get_episode(self, episode_id: str) -> Mapping[str, Any]: ...

    def actions(self, episode_id: str) -> list[Mapping[str, Any]]: ...

    def tokens(self, episode_id: str) -> list[Mapping[str, Any]]: ...

    def sampler_segments(self, episode_id: str) -> list[Mapping[str, Any]]: ...

    def budget_segments(self, episode_id: str) -> list[Mapping[str, Any]]: ...


@dataclass(frozen=True)
class _ProjectedSource:
    procedure: SurvivingProcedure
    action_rows: tuple[Mapping[str, Any], ...]
    tokens_by_action: tuple[tuple[Mapping[str, Any], ...], ...]
    visible_boundary: int


def _required_int(record: Mapping[str, Any], field: str) -> int:
    value = record.get(field)
    if type(value) is not int:
        raise EditorError(f"saved {field} must be an integer")
    return value


def _controls(
    reader: EpisodeReplayReader,
    episode_id: str,
    *,
    sampling_factory: Callable[[Mapping[str, Any]], SamplerConfig],
) -> ControlTimeline:
    sampler_segments = tuple(reader.sampler_segments(episode_id))
    budget_segments = tuple(reader.budget_segments(episode_id))
    if not sampler_segments:
        raise EditorError(f"episode {episode_id!r} has no sampler segment")
    if not budget_segments:
        raise EditorError(f"episode {episode_id!r} has no budget segment")

    sampler_starts = tuple(
        _required_int(segment, "start_boundary") for segment in sampler_segments
    )
    budget_starts = tuple(
        _required_int(segment, "start_boundary") for segment in budget_segments
    )
    if sampler_starts[0] != 0 or budget_starts[0] != 0:
        raise EditorError("episode controls must begin at root boundary zero")
    if tuple(sorted(set(sampler_starts))) != sampler_starts:
        raise EditorError("saved sampler segments must have ordered unique boundaries")
    if tuple(sorted(set(budget_starts))) != budget_starts:
        raise EditorError("saved budget segments must have ordered unique boundaries")

    transitions: list[ControlTransition] = []
    sampler_index = 0
    budget_index = 0
    for boundary in sorted({*sampler_starts, *budget_starts}):
        while (
            sampler_index + 1 < len(sampler_segments)
            and _required_int(sampler_segments[sampler_index + 1], "start_boundary")
            <= boundary
        ):
            sampler_index += 1
        while (
            budget_index + 1 < len(budget_segments)
            and _required_int(budget_segments[budget_index + 1], "start_boundary")
            <= boundary
        ):
            budget_index += 1

        sampler = sampler_segments[sampler_index]
        budget = budget_segments[budget_index]
        sampling_record = sampler.get("sampling")
        if not isinstance(sampling_record, Mapping):
            raise EditorError("saved sampler settings must be an object")
        fingerprint = sampler.get("stream_fingerprint")
        if not isinstance(fingerprint, str):
            raise EditorError("saved sampler segment must have a stream fingerprint")
        transitions.append(
            ControlTransition(
                boundary,
                ControlState.from_parts(
                    sampling_factory(sampling_record),
                    fingerprint,
                    _required_int(sampler, "coordinate_offset"),
                    budget.get("max_tokens"),
                    budget.get("checkpoint_boundary"),
                ),
            )
        )
    return ControlTimeline(tuple(transitions))


def _projected_source(
    reader: EpisodeReplayReader,
    episode_id: str,
    *,
    controls: ControlTimeline,
) -> _ProjectedSource:
    action_rows = tuple(reader.actions(episode_id))
    token_rows = reader.tokens(episode_id)
    visible_boundary = sum(bool(row.get("realized_visible")) for row in token_rows)
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for token in token_rows:
        grouped.setdefault(_required_int(token, "action_ordinal"), []).append(token)
    tokens_by_action = tuple(
        tuple(grouped.get(_required_int(row, "ordinal"), ()))
        for row in action_rows
    )

    records: list[ProcedureRecord] = []
    for row, tokens in zip(action_rows, tokens_by_action):
        arguments = row.get("arguments")
        if not isinstance(arguments, Mapping):
            raise EditorError("saved action arguments must be an object")
        boundary = _required_int(row, "boundary_before")
        visible = tuple(
            _required_int(token, "token_id")
            for token in tokens
            if bool(token.get("realized_visible"))
        )
        terminal = next(
            (
                _required_int(token, "token_id")
                for token in tokens
                if bool(token.get("is_eog"))
            ),
            None,
        )
        action = action_from_dict(dict(arguments))
        stop_reason = str(row["stop_reason"])
        if (
            isinstance(action, Hold)
            and stop_reason == "checkpoint"
            and len(visible) == action.limit
            and terminal is None
        ):
            # The budget also expired, but the finite hold completed. Replay
            # may use a different budget, so use the hold's completion reason.
            stop_reason = "requested-length"
        records.append(
            ProcedureRecord(
                action=action,
                expectation=ReplayExpectation(visible, terminal, stop_reason),
                status=str(row["status"]),
                visible_token_ids=visible,
                visible_text="",
                boundary_before=boundary,
                sampling=controls.effective_at(boundary).sampling,
            )
        )

    return _ProjectedSource(
        procedure=project_surviving_procedure(
            records,
            normalize_for_replay=False,
        ),
        action_rows=action_rows,
        tokens_by_action=tokens_by_action,
        visible_boundary=visible_boundary,
    )


def _select_through(
    source: _ProjectedSource,
    end_boundary: int,
) -> SurvivingProcedure:
    """Select a procedure prefix without rebasing its source coordinates."""

    selected: list[ProcedureStep] = []
    partial = set(source.procedure.partial_source_indices)
    for step in source.procedure.steps:
        if step.boundary >= end_boundary:
            break
        tokens = source.tokens_by_action[step.source_index]
        visible = tuple(token for token in tokens if bool(token.get("realized_visible")))
        count = end_boundary - step.boundary
        action = step.action
        expectation = step.expectation
        cut = len(visible) > count
        # A Hold ending exactly at the selected boundary is made finite so
        # the selected procedure yields at that live edge.
        make_finite = cut or (
            len(visible) == count and isinstance(action, Hold)
        )
        if make_finite:
            retained = visible[:count]
            visible_ids = tuple(_required_int(token, "token_id") for token in retained)
            if isinstance(action, (Write, Phrase)):
                action = Write(
                    "".join(str(token.get("text", "")) for token in retained),
                    mode="exact",
                )
                reason = "completed"
            else:
                action = Hold(count)
                reason = "requested-length"
            expectation = ReplayExpectation(visible_ids, None, reason)
            if cut:
                partial.add(step.source_index)
        selected.append(
            replace(
                step,
                tape_step=TapeStep(action, expectation),
                partial=step.partial or cut,
            )
        )

    return SurvivingProcedure(
        steps=tuple(selected),
        skipped_source_indices=source.procedure.skipped_source_indices,
        partial_source_indices=tuple(sorted(partial)),
    )


def build_source_replay_recipe(
    reader: EpisodeReplayReader,
    episode_id: str,
    until: int | None = None,
    *,
    sampling_factory: Callable[[Mapping[str, Any]], SamplerConfig] = SamplerConfig.from_record,
) -> SourceReplayRecipe:
    """Transform one durable source and selected endpoint into an SPR recipe."""

    episode = reader.get_episode(episode_id)
    controls = _controls(reader, episode_id, sampling_factory=sampling_factory)
    source = _projected_source(reader, episode_id, controls=controls)
    end_boundary = source.visible_boundary if until is None else until
    if type(end_boundary) is not int or not 0 <= end_boundary <= source.visible_boundary:
        raise EditorError(f"Replay boundary must be 0..{source.visible_boundary}.")
    prompt = episode.get("initial_text")
    if not isinstance(prompt, str):
        raise EditorError("saved initial prompt must be a string")
    return SourceReplayRecipe(
        source_prompt=prompt,
        procedure=_select_through(source, end_boundary),
        controls=controls.truncate_after(end_boundary),
        source_visible_boundary=source.visible_boundary,
        source_end_boundary=end_boundary,
        source_id=episode_id,
    )


def replay_procedure(
    reader: EpisodeReplayReader,
    episode_id: str,
    *,
    sampling_factory: Callable[[Mapping[str, Any]], SamplerConfig] = SamplerConfig.from_record,
) -> list[dict[str, Any]]:
    """Return the source procedure plus durable evidence for presentation."""

    controls = _controls(reader, episode_id, sampling_factory=sampling_factory)
    source = _projected_source(reader, episode_id, controls=controls)
    return [
        {
            "action": step.action,
            "expectation": step.expectation,
            "sampling": step.sampling,
            "boundary": step.boundary,
            "tokens": list(source.tokens_by_action[step.source_index]),
        }
        for step in source.procedure.steps
    ]


def replay_tape(
    reader: EpisodeReplayReader,
    episode_id: str,
    *,
    sampling_factory: Callable[[Mapping[str, Any]], SamplerConfig] = SamplerConfig.from_record,
) -> list[tuple[PolicyAction, ReplayExpectation | None]]:
    """Return the executable action/expectation pairs for a source episode."""

    return [
        (step["action"], step["expectation"])
        for step in replay_procedure(
            reader, episode_id, sampling_factory=sampling_factory
        )
    ]


def replay_tape_with_sampling(
    reader: EpisodeReplayReader,
    episode_id: str,
    *,
    sampling_factory: Callable[[Mapping[str, Any]], SamplerConfig] = SamplerConfig.from_record,
) -> list[tuple[PolicyAction, ReplayExpectation | None, SamplerConfig]]:
    """Return executable steps with the sampler effective at each source step."""

    return [
        (step["action"], step["expectation"], step["sampling"])
        for step in replay_procedure(
            reader, episode_id, sampling_factory=sampling_factory
        )
    ]


def final_sampling(
    reader: EpisodeReplayReader,
    episode_id: str,
    *,
    sampling_factory: Callable[[Mapping[str, Any]], SamplerConfig] = SamplerConfig.from_record,
) -> SamplerConfig:
    """Return the latest stored source sampler setting."""

    controls = _controls(reader, episode_id, sampling_factory=sampling_factory)
    return controls.transitions[-1].state.sampling


__all__ = [
    "EpisodeReplayReader",
    "build_source_replay_recipe",
    "final_sampling",
    "replay_procedure",
    "replay_tape",
    "replay_tape_with_sampling",
]

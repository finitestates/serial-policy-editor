"""Project durable ordered actions into storage-neutral replay semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .core.actions import (
    Hold, Phrase, UnsupportedPolicyActionKind, Write, action_from_dict,
)
from .core.errors import EditorError
from .core.results import ReplayExpectation
from .core.sampler_config import SamplerConfig
from .run_loop import TapeStep
from .spr_recipe import SourceReplayRecipe
from .surviving_procedure import (
    ProcedureRecord,
    ProcedureStep,
    SurvivingProcedure,
    project_surviving_procedure,
)


class EpisodeReplayReader(Protocol):
    def get_episode(self, episode_id: str) -> Mapping[str, Any]: ...
    def actions(self, episode_id: str) -> list[Mapping[str, Any]]: ...
    def tokens(self, episode_id: str) -> list[Mapping[str, Any]]: ...


@dataclass(frozen=True)
class _ProjectedSource:
    procedure: SurvivingProcedure
    action_rows: tuple[Mapping[str, Any], ...]
    tokens_by_action: tuple[tuple[Mapping[str, Any], ...], ...]
    visible_boundary: int
    unsupported_boundary: int | None = None
    handoff_reason: str | None = None


def _required_int(record: Mapping[str, Any], field: str) -> int:
    value = record.get(field)
    if type(value) is not int:
        raise EditorError(f"saved {field} must be an integer")
    return value


def _projected_source(
    reader: EpisodeReplayReader, episode_id: str
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
    unsupported_boundary: int | None = None
    handoff_reason: str | None = None
    for row, tokens in zip(action_rows, tokens_by_action):
        arguments = row.get("arguments")
        if not isinstance(arguments, Mapping):
            raise EditorError("saved action arguments must be an object")
        boundary = _required_int(row, "boundary_before")
        visible = tuple(
            _required_int(token, "token_id")
            for token in tokens if bool(token.get("realized_visible"))
        )
        terminal = next(
            (
                _required_int(token, "token_id")
                for token in tokens if bool(token.get("is_eog"))
            ),
            None,
        )
        try:
            action = action_from_dict(arguments)
        except UnsupportedPolicyActionKind as exc:
            unsupported_boundary = boundary
            ordinal = _required_int(row, "ordinal")
            handoff_reason = (
                f"Warning: replay stopped before source step {ordinal + 1} "
                f"at boundary {boundary}: {exc}."
            )
            break
        stop_reason = row.get("stop_reason")
        if stop_reason is not None and not isinstance(stop_reason, str):
            raise EditorError("saved stop_reason must be a string or null")
        expectation = (
            ReplayExpectation(visible, terminal, stop_reason)
            if stop_reason is not None else None
        )
        records.append(
            ProcedureRecord(
                action=action,
                expectation=expectation,
                status=str(row["status"]),
                visible_token_ids=visible,
                visible_text="",
                boundary_before=boundary,
            )
        )

    return _ProjectedSource(
        procedure=project_surviving_procedure(records, normalize_for_replay=False),
        action_rows=action_rows,
        tokens_by_action=tokens_by_action,
        visible_boundary=visible_boundary,
        unsupported_boundary=unsupported_boundary,
        handoff_reason=handoff_reason,
    )


def _select_through(
    source: _ProjectedSource, end_boundary: int, *, full_source: bool
) -> SurvivingProcedure:
    """Select a command prefix; an explicit endpoint precedes its commands."""
    selected: list[ProcedureStep] = []
    partial = set(source.procedure.partial_source_indices)
    for step in source.procedure.steps:
        if step.boundary > end_boundary or (
            step.boundary == end_boundary and not full_source
        ):
            break
        tokens = source.tokens_by_action[step.source_index]
        visible = tuple(token for token in tokens if bool(token.get("realized_visible")))
        count = end_boundary - step.boundary
        action = step.action
        expectation = step.expectation
        cut = len(visible) > count
        make_finite = count > 0 and (
            cut or (len(visible) == count and isinstance(action, Hold))
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
            expectation = (
                ReplayExpectation(visible_ids, None, reason)
                if step.expectation is not None else None
            )
            if cut:
                partial.add(step.source_index)
        selected.append(
            replace(step, tape_step=TapeStep(action, expectation), partial=step.partial or cut)
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
) -> SourceReplayRecipe:
    episode = reader.get_episode(episode_id)
    source = _projected_source(reader, episode_id)
    end_boundary = source.visible_boundary if until is None else until
    if type(end_boundary) is not int or not 0 <= end_boundary <= source.visible_boundary:
        raise EditorError(f"Replay boundary must be 0..{source.visible_boundary}.")
    prompt = episode.get("initial_text")
    if not isinstance(prompt, str):
        raise EditorError("saved initial prompt must be a string")
    initial_record = episode.get("initial_sampling")
    if not isinstance(initial_record, Mapping):
        raise EditorError("saved initial sampler settings must be an object")
    return SourceReplayRecipe(
        source_prompt=prompt,
        procedure=_select_through(source, end_boundary, full_source=until is None),
        initial_sampling=SamplerConfig.from_record(initial_record),
        stream_fingerprint=str(episode["initial_stream_fingerprint"]),
        source_visible_boundary=source.visible_boundary,
        source_end_boundary=end_boundary,
        source_id=episode_id,
        incomplete_handoff_reason=(
            source.handoff_reason
            if source.unsupported_boundary is not None
            and (until is None or source.unsupported_boundary < end_boundary)
            else None
        ),
    )


def replay_procedure(
    reader: EpisodeReplayReader,
    episode_id: str,
) -> list[dict[str, Any]]:
    """Return surviving tape actions and their durable token evidence."""
    source = _projected_source(reader, episode_id)
    if source.handoff_reason is not None:
        raise EditorError(source.handoff_reason)
    return [
        {
            "action": step.action,
            "expectation": step.expectation,
            "boundary": step.boundary,
            "tokens": list(source.tokens_by_action[step.source_index]),
        }
        for step in source.procedure.steps
    ]


__all__ = ["EpisodeReplayReader", "build_source_replay_recipe", "replay_procedure"]

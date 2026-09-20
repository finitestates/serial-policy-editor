"""Storage-neutral construction of root-relative fork prefixes.

Forking is a semantic operation: retain a visible prefix, keep the original
root prompt, and make the retained history ordinary destination history.  The
durable adapter serializes the result; the live-session adapter keeps the same
shape in memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .core.actions import Hold, PolicyAction, Write
from .core.errors import EditorError
from .core.results import ActionOutcome
from .episode_runner import TapeStep


@dataclass(frozen=True)
class StoredForkAction:
    """One source action projected into a retained root-relative prefix."""

    ordinal: int
    boundary_before: int
    boundary_after: int
    kind: str
    arguments: Mapping[str, Any]
    resolved_text: str
    status: str
    stop_reason: str
    mismatch: Mapping[str, Any] | None
    tokens: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class StoredForkPrefix:
    """The persistence-neutral result of materializing a stored prefix."""

    actions: tuple[StoredForkAction, ...]
    sampler_segments: tuple[Mapping[str, Any], ...]
    budget_segments: tuple[Mapping[str, Any], ...]
    source_boundary: int


def visible_text_prefix(
    tokens: Sequence[Mapping[str, Any]],
    boundary: int,
) -> str:
    """Return visible source text strictly before a root-relative boundary."""
    if type(boundary) is not int or boundary < 0:
        raise EditorError("fork boundary must be a nonnegative integer")
    return "".join(
        str(token["text"])
        for token in tokens
        if bool(token["realized_visible"]) and int(token["boundary"]) < boundary
    )


def materialize_stored_prefix(
    actions: Sequence[Mapping[str, Any]],
    tokens: Sequence[Mapping[str, Any]],
    sampler_segments: Sequence[Mapping[str, Any]],
    budget_segments: Sequence[Mapping[str, Any]],
    boundary: int,
) -> StoredForkPrefix:
    """Project stored source records into a root-relative fork prefix.

    The input mappings are deliberately persistence-shaped but contain no
    SQLite objects or serialization concerns.  This lets the database remain
    an adapter while the partial-action rules are shared with other stores.
    """
    if type(boundary) is not int or boundary < 0:
        raise EditorError("fork boundary must be a nonnegative integer")

    visible_count = sum(bool(token["realized_visible"]) for token in tokens)
    if boundary > visible_count:
        raise EditorError(f"fork boundary must be between 0 and {visible_count}")

    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for token in tokens:
        grouped.setdefault(int(token["action_ordinal"]), []).append(token)

    projected: list[StoredForkAction] = []
    for source in sorted(actions, key=lambda row: int(row["ordinal"])):
        before = int(source["boundary_before"])
        after = int(source["boundary_after"])
        if before >= boundary:
            break
        source_tokens = grouped.get(int(source["ordinal"]), [])
        if after <= boundary:
            projected.append(_stored_action(source, source_tokens))
            continue

        # The boundary cuts through this action.  Match rewind semantics:
        # writes become exact writes and holds become finite holds.
        retained = tuple(
            token
            for token in source_tokens
            if int(token["boundary"]) < boundary
        )
        arguments = dict(source["arguments"])
        resolved_text = visible_text_prefix(retained, boundary)
        kind = str(source["kind"])
        if kind in {"write", "check-phrase", "force-phrase"}:
            original_key = "original_write" if kind == "write" else "original_action"
            arguments = {
                **arguments,
                "kind": "write",
                "mode": "exact",
                "text": resolved_text,
                original_key: arguments.get(original_key, dict(arguments)),
            }
            kind = "write"
            stop_reason = "completed"
        else:
            arguments = {
                **arguments,
                "limit": boundary - before,
                "boundary": None,
            }
            stop_reason = "requested-length"
        projected.append(
            StoredForkAction(
                ordinal=int(source["ordinal"]),
                boundary_before=before,
                boundary_after=boundary,
                kind=kind,
                arguments=arguments,
                resolved_text=resolved_text,
                status="completed",
                stop_reason=stop_reason,
                mismatch=None,
                tokens=retained,
            )
        )
        break

    return StoredForkPrefix(
        actions=tuple(projected),
        sampler_segments=tuple(
            dict(segment)
            for segment in sampler_segments
            if int(segment["start_boundary"]) <= boundary
        ),
        budget_segments=tuple(
            dict(segment)
            for segment in budget_segments
            if int(segment["start_boundary"]) <= boundary
        ),
        source_boundary=boundary,
    )


def trim_live_prefix(
    tape: Sequence[TapeStep],
    outcomes: Sequence[ActionOutcome],
    boundary: int,
    backend: Any,
) -> tuple[list[TapeStep], list[ActionOutcome], list[TapeStep], list[ActionOutcome]]:
    """Trim live-session records to a root-relative fork boundary."""
    if type(boundary) is not int or boundary < 0:
        raise EditorError("fork boundary must be a nonnegative integer")
    if len(tape) != len(outcomes):
        raise EditorError("live fork tape and outcomes must align")

    kept_tape: list[TapeStep] = []
    kept_outcomes: list[ActionOutcome] = []
    removed_tape: list[TapeStep] = []
    removed_outcomes: list[ActionOutcome] = []
    for step, outcome in zip(tape, outcomes):
        if outcome.boundary_before >= boundary:
            removed_tape.append(step)
            removed_outcomes.append(outcome)
        elif outcome.boundary_after <= boundary:
            kept_tape.append(step)
            kept_outcomes.append(outcome)
        elif outcome.boundary_before < boundary < outcome.boundary_after:
            partial = _partial_live_outcome(outcome, boundary, backend)
            if partial is not None:
                kept_tape.append(TapeStep(partial.action, partial.expectation()))
                kept_outcomes.append(partial)
            removed_tape.append(step)
            removed_outcomes.append(outcome)
        else:
            removed_tape.append(step)
            removed_outcomes.append(outcome)
    return kept_tape, kept_outcomes, removed_tape, removed_outcomes


def _stored_action(
    source: Mapping[str, Any],
    tokens: Sequence[Mapping[str, Any]],
) -> StoredForkAction:
    return StoredForkAction(
        ordinal=int(source["ordinal"]),
        boundary_before=int(source["boundary_before"]),
        boundary_after=int(source["boundary_after"]),
        kind=str(source["kind"]),
        arguments=dict(source["arguments"]),
        resolved_text=str(source["resolved_text"]),
        status=str(source["status"]),
        stop_reason=str(source["stop_reason"]),
        mismatch=source.get("mismatch"),
        tokens=tuple(tokens),
    )


def _partial_live_outcome(
    outcome: ActionOutcome,
    boundary: int,
    backend: Any,
) -> ActionOutcome | None:
    count = boundary - outcome.boundary_before
    visible = outcome.visible_token_ids[:count]
    if not visible:
        return None
    if isinstance(outcome.action, Hold):
        action: PolicyAction = Hold(len(visible))
        stop_reason = "requested-length"
    else:
        action = Write(backend.render(list(visible)), mode="exact")
        stop_reason = "completed"
    evidence = tuple(
        item
        for item in outcome.evidence
        if item.realized_visible and item.boundary < boundary
    )
    return replace(
        outcome,
        action=action,
        boundary_after=boundary,
        resolved_text=backend.render(list(visible)),
        resolved_token_ids=tuple(visible),
        visible_token_ids=tuple(visible),
        terminal_token_id=None,
        stop_reason=stop_reason,
        evidence=evidence,
        status="completed",
        divergence=None,
        replay_eog_token_id=None,
    )


__all__ = [
    "StoredForkAction",
    "StoredForkPrefix",
    "materialize_stored_prefix",
    "visible_text_prefix",
    "trim_live_prefix",
]

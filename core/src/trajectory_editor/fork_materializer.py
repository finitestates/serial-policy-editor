"""Storage-neutral construction of root-relative durable fork prefixes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .core.errors import EditorError


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


__all__ = [
    "StoredForkAction",
    "StoredForkPrefix",
    "materialize_stored_prefix",
    "visible_text_prefix",
]

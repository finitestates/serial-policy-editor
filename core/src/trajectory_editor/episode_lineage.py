"""Storage-neutral episode lineage facts and graph projection.

This module deliberately accepts already-loaded relation facts.  Persistence
adapters can load those facts from SQLite (or another source) later, while
renderers can consume :class:`LineageView` without knowing how the facts were
stored.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


def _validate_relation_id(
    value: object,
    field: str,
    *,
    optional: bool = False,
) -> str | None:
    if value is None:
        if optional:
            return None
        raise ValueError(f"{field} must be a nonempty relation ID")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string or None")
    if not value or any(character.isspace() for character in value):
        raise ValueError(f"{field} must be a nonempty string without whitespace")
    return value


def _validate_label(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value or value != value.strip():
        raise ValueError(
            f"{field} must be a nonempty label without surrounding whitespace"
        )
    return value


def _validate_nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be a nonnegative integer")
    if value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class EpisodeRelation:
    """The typed relation facts needed to place one episode in a lineage.

    ``parent_id`` is the persisted parent/context reference.  It is used for
    ordinary family edges only when the referenced record is also ordinary.
    ``spr_source_id`` identifies an SPR provenance source and never makes the
    record an ordinary family node.

    Database field names are translated by the storage adapter.
    """

    episode_id: str
    parent_id: str | None = None
    fork_boundary: int | None = None
    mode: str = "interactive"
    spr_source_id: str | None = None
    status: str = "unknown"
    creation_key: Any = 0
    terminal_reason: str | None = None
    visible_token_count: int = 0
    model_change_source_id: str | None = None
    model_change_boundary: int | None = None

    def __post_init__(self) -> None:
        _validate_relation_id(self.episode_id, "episode_id")
        _validate_relation_id(self.parent_id, "parent_id", optional=True)
        _validate_relation_id(self.spr_source_id, "spr_source_id", optional=True)
        _validate_relation_id(
            self.model_change_source_id, "model_change_source_id", optional=True
        )
        if self.fork_boundary is not None:
            _validate_nonnegative_int(self.fork_boundary, "fork_boundary")
        if self.model_change_boundary is not None:
            _validate_nonnegative_int(
                self.model_change_boundary, "model_change_boundary"
            )
        _validate_label(self.mode, "mode")
        _validate_label(self.status, "status")
        _validate_nonnegative_int(self.visible_token_count, "visible_token_count")
        if self.terminal_reason is not None and not isinstance(self.terminal_reason, str):
            raise TypeError("terminal_reason must be a string or None")

    @property
    def is_replay(self) -> bool:
        """Whether this record represents replay provenance."""

        return self.mode == "serial-policy-replay" or self.spr_source_id is not None

@dataclass(frozen=True, slots=True)
class LineageNode:
    """One ordinary family node and its finite, ordered descendants."""

    record: EpisodeRelation
    children: tuple["LineageNode", ...] = ()

    @property
    def episode_id(self) -> str:
        return self.record.episode_id

    @property
    def parent_id(self) -> str | None:
        return self.record.parent_id


@dataclass(frozen=True, slots=True)
class LineageView:
    """Immutable result of projecting flat relation facts into lineage."""

    selected_record: EpisodeRelation
    ordinary_family_root: EpisodeRelation | None
    ordinary_fork_tree: LineageNode | None
    related_replays: tuple[EpisodeRelation, ...]
    replay_derived_forks: tuple[EpisodeRelation, ...]
    model_change_related: tuple[EpisodeRelation, ...] = ()

    @property
    def ordinary_family_root_id(self) -> str | None:
        if self.ordinary_family_root is None:
            return None
        return self.ordinary_family_root.episode_id

def _sortable(value: Any) -> tuple[Any, ...]:
    """Make even malformed mixed creation keys sortable and deterministic."""

    if value is None:
        return (0, "")
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, (int, float, str)):
        return (2, type(value).__name__, value)
    if isinstance(value, tuple):
        return (3, tuple(_sortable(item) for item in value))
    return (4, type(value).__name__, repr(value))


def _record_order(record: EpisodeRelation) -> tuple[Any, ...]:
    return (_sortable(record.creation_key), record.episode_id)


def _index_records(records: Iterable[EpisodeRelation]) -> dict[str, EpisodeRelation]:
    indexed: dict[str, EpisodeRelation] = {}
    for record in records:
        if not isinstance(record, EpisodeRelation):
            raise TypeError("lineage facts must be EpisodeRelation instances")
        if record.episode_id in indexed:
            raise ValueError(f"duplicate episode relation ID {record.episode_id!r}")
        indexed[record.episode_id] = record
    return indexed


def _nearest_ordinary_seed(
    selected: EpisodeRelation,
    records: dict[str, EpisodeRelation],
) -> str | None:
    """Find the nearest ordinary record by parent/source traversal.

    Parent is considered before SPR source at each distance.  A visited set
    makes corrupt replay-reference cycles finite without discarding the
    selected record from the eventual result.
    """

    pending: deque[str] = deque([selected.episode_id])
    seen: set[str] = set()
    while pending:
        identifier = pending.popleft()
        if identifier in seen:
            continue
        seen.add(identifier)
        record = records.get(identifier)
        if record is None:
            continue
        if not record.is_replay:
            return identifier
        for reference in (record.parent_id, record.spr_source_id):
            if (
                isinstance(reference, str)
                and reference in records
                and reference not in seen
            ):
                pending.append(reference)
    return None


def _ordinary_root_id(
    identifier: str | None,
    ordinary: dict[str, EpisodeRelation],
) -> str | None:
    if identifier is None or identifier not in ordinary:
        return None
    current = identifier
    path: dict[str, int] = {}
    while current in ordinary:
        if current in path:
            cycle = tuple(path)[path[current] :]
            return min(cycle, key=lambda value: _record_order(ordinary[value]))
        path[current] = len(path)
        parent = ordinary[current].parent_id
        if not isinstance(parent, str) or parent not in ordinary:
            return current
        current = parent
    return current


def _tree(
    identifier: str,
    records: dict[str, EpisodeRelation],
    children_by_parent: dict[str, tuple[str, ...]],
    path: frozenset[str] = frozenset(),
) -> LineageNode:
    record = records[identifier]
    if identifier in path:
        return LineageNode(record)
    next_path = path | {identifier}
    children = tuple(
        _tree(child, records, children_by_parent, next_path)
        for child in children_by_parent.get(identifier, ())
        if child not in next_path
    )
    return LineageNode(record, children)


def build_lineage(
    records: Iterable[EpisodeRelation],
    selected_episode_id: str,
) -> LineageView:
    """Build a deterministic typed lineage view from flat relation facts.

    Replay records are provenance, not ordinary tree nodes.  Missing parent or
    source references simply terminate the relevant traversal, and ordinary
    parent cycles are represented by a finite tree with the cyclic back-edge
    omitted.
    """

    indexed = _index_records(records)
    try:
        selected = indexed[selected_episode_id]
    except KeyError as exc:
        raise KeyError(f"unknown lineage episode {selected_episode_id!r}") from exc

    ordinary = {
        identifier: record
        for identifier, record in indexed.items()
        if not record.is_replay
    }
    seed_id = _nearest_ordinary_seed(selected, indexed)
    root_id = _ordinary_root_id(seed_id, ordinary)

    family_ids = {
        identifier
        for identifier in ordinary
        if _ordinary_root_id(identifier, ordinary) == root_id
    }
    children: dict[str, list[str]] = {}
    for identifier in family_ids:
        parent = ordinary[identifier].parent_id
        if isinstance(parent, str) and parent in family_ids:
            children.setdefault(parent, []).append(identifier)
    ordered_children = {
        parent: tuple(
            sorted(child_ids, key=lambda child: _record_order(ordinary[child]))
        )
        for parent, child_ids in children.items()
    }

    family_root = ordinary[root_id] if root_id is not None else None
    tree = (
        _tree(root_id, indexed, ordered_children)
        if root_id is not None
        else None
    )

    related_replay_ids = {
        identifier
        for identifier, record in indexed.items()
        if record.is_replay
        and (
            identifier == selected_episode_id
            or record.parent_id in family_ids
            or record.spr_source_id in family_ids
        )
    }
    # Preserve the context of an ordinary record whose direct parent is a
    # replay, including in workspaces with incomplete replay source metadata.
    for identifier in family_ids:
        parent = ordinary[identifier].parent_id
        if isinstance(parent, str) and parent in indexed and indexed[parent].is_replay:
            related_replay_ids.add(parent)

    related_replays = tuple(
        sorted(
            (indexed[identifier] for identifier in related_replay_ids),
            key=_record_order,
        )
    )
    replay_derived_forks = tuple(
        sorted(
            (
                record
                for record in indexed.values()
                if not record.is_replay and record.parent_id in related_replay_ids
            ),
            key=_record_order,
        )
    )

    model_change_context_ids = family_ids | {selected_episode_id}
    model_change_related = tuple(
        sorted(
            (
                record
                for record in indexed.values()
                if record.mode == "model-change"
                and record.model_change_source_id is not None
                and record.parent_id != record.model_change_source_id
                and (
                    record.episode_id in model_change_context_ids
                    or record.model_change_source_id in model_change_context_ids
                )
            ),
            key=_record_order,
        )
    )

    return LineageView(
        selected_record=selected,
        ordinary_family_root=family_root,
        ordinary_fork_tree=tree,
        related_replays=related_replays,
        replay_derived_forks=replay_derived_forks,
        model_change_related=model_change_related,
    )


__all__ = [
    "EpisodeRelation",
    "LineageNode",
    "LineageView",
    "build_lineage",
]

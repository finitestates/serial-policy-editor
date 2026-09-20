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


_UNSET = object()


@dataclass(frozen=True, slots=True, init=False)
class EpisodeRelation:
    """The typed relation facts needed to place one episode in a lineage.

    ``parent_id`` is the persisted parent/context reference.  It is used for
    ordinary family edges only when the referenced record is also ordinary.
    ``spr_source_id`` identifies an SPR provenance source and never makes the
    record an ordinary family node.

    The keyword aliases accepted by ``__init__`` keep this seam convenient for
    adapters whose field names still include the older ``*_episode_id``
    spelling.  The stored value remains one immutable typed record.
    """

    episode_id: str
    parent_id: str | None
    fork_boundary: int | None
    mode: str
    spr_source_id: str | None
    status: str
    creation_key: Any
    terminal_reason: str | None
    visible_token_count: int

    def __init__(
        self,
        episode_id: str,
        parent_id: str | None | object = _UNSET,
        fork_boundary: int | None = None,
        mode: str = "interactive",
        spr_source_id: str | None | object = _UNSET,
        status: str = "unknown",
        creation_key: Any = _UNSET,
        terminal_reason: str | None = None,
        visible_token_count: int | object = _UNSET,
        *,
        parent_episode_id: str | None | object = _UNSET,
        spr_source_episode_id: str | None | object = _UNSET,
        replay_source_episode_id: str | None | object = _UNSET,
        creation_order_key: Any = _UNSET,
        visible_tokens: int | object = _UNSET,
    ) -> None:
        parent = _coalesce_alias(
            parent_id,
            parent_episode_id,
            canonical="parent_id",
            alias="parent_episode_id",
            default=None,
        )
        source_alias = _coalesce_alias(
            spr_source_episode_id,
            replay_source_episode_id,
            canonical="spr_source_episode_id",
            alias="replay_source_episode_id",
            default=_UNSET,
        )
        source = _coalesce_alias(
            spr_source_id,
            source_alias,
            canonical="spr_source_id",
            alias="spr_source_episode_id/replay_source_episode_id",
            default=None,
        )
        ordering = _coalesce_alias(
            creation_key,
            creation_order_key,
            canonical="creation_key",
            alias="creation_order_key",
            default=0,
        )
        visible_count = _coalesce_alias(
            visible_token_count,
            visible_tokens,
            canonical="visible_token_count",
            alias="visible_tokens",
            default=0,
        )
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "parent_id", parent)
        object.__setattr__(self, "fork_boundary", fork_boundary)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "spr_source_id", source)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "creation_key", ordering)
        object.__setattr__(self, "terminal_reason", terminal_reason)
        object.__setattr__(self, "visible_token_count", visible_count)

    @property
    def is_replay(self) -> bool:
        """Whether this record represents replay provenance."""

        return self.mode == "serial-policy-replay" or self.spr_source_id is not None

    @property
    def parent_episode_id(self) -> str | None:
        """Persistence-shaped spelling for an adapter boundary."""

        return self.parent_id

    @property
    def spr_source_episode_id(self) -> str | None:
        """Persistence-shaped spelling for the SPR source reference."""

        return self.spr_source_id

    @property
    def replay_source_episode_id(self) -> str | None:
        """Alias used by the existing durable lineage projection."""

        return self.spr_source_id

    @property
    def creation_order_key(self) -> Any:
        return self.creation_key

    @property
    def visible_tokens(self) -> int:
        return self.visible_token_count


def _coalesce_alias(
    canonical_value: Any,
    alias_value: Any,
    *,
    canonical: str,
    alias: str,
    default: Any,
) -> Any:
    canonical_set = canonical_value is not _UNSET
    alias_set = alias_value is not _UNSET
    if canonical_set and alias_set and canonical_value != alias_value:
        raise TypeError(f"{canonical} and {alias} disagree")
    if canonical_set:
        return canonical_value
    if alias_set:
        return alias_value
    return default


LineageRecord = EpisodeRelation


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


ForkTreeNode = LineageNode


@dataclass(frozen=True, slots=True)
class LineageView:
    """Immutable result of projecting flat relation facts into lineage."""

    selected_record: EpisodeRelation
    ordinary_family_root: EpisodeRelation | None
    ordinary_fork_tree: LineageNode | None
    related_replays: tuple[EpisodeRelation, ...]
    replay_derived_forks: tuple[EpisodeRelation, ...]

    @property
    def selected(self) -> EpisodeRelation:
        return self.selected_record

    @property
    def selected_episode_id(self) -> str:
        return self.selected_record.episode_id

    @property
    def ordinary_family_root_id(self) -> str | None:
        if self.ordinary_family_root is None:
            return None
        return self.ordinary_family_root.episode_id

    @property
    def family_root(self) -> EpisodeRelation | None:
        return self.ordinary_family_root

    @property
    def family_root_id(self) -> str | None:
        return self.ordinary_family_root_id

    @property
    def tree(self) -> LineageNode | None:
        return self.ordinary_fork_tree

    @property
    def ordinary_tree(self) -> LineageNode | None:
        return self.ordinary_fork_tree

    @property
    def replays(self) -> tuple[EpisodeRelation, ...]:
        return self.related_replays

    @property
    def replay_forks(self) -> tuple[EpisodeRelation, ...]:
        return self.replay_derived_forks


EpisodeLineage = LineageView


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


def _duplicate_order(record: EpisodeRelation) -> tuple[Any, ...]:
    return (
        _record_order(record),
        repr(
            (
                record.parent_id,
                record.fork_boundary,
                record.mode,
                record.spr_source_id,
                record.status,
                record.terminal_reason,
                record.visible_token_count,
            )
        ),
    )


def _index_records(records: Iterable[EpisodeRelation]) -> dict[str, EpisodeRelation]:
    indexed: dict[str, EpisodeRelation] = {}
    for record in records:
        if not isinstance(record, EpisodeRelation):
            raise TypeError("lineage facts must be EpisodeRelation instances")
        previous = indexed.get(record.episode_id)
        if previous is None or _duplicate_order(record) < _duplicate_order(previous):
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

    return LineageView(
        selected_record=selected,
        ordinary_family_root=family_root,
        ordinary_fork_tree=tree,
        related_replays=related_replays,
        replay_derived_forks=replay_derived_forks,
    )


def build_episode_lineage(
    records: Iterable[EpisodeRelation],
    selected_episode_id: str,
) -> LineageView:
    """Descriptive alias for :func:`build_lineage`."""

    return build_lineage(records, selected_episode_id)


def lineage_view(
    records: Iterable[EpisodeRelation],
    selected_episode_id: str,
) -> LineageView:
    """Short alias for callers that treat the result as a view."""

    return build_lineage(records, selected_episode_id)


__all__ = [
    "EpisodeLineage",
    "EpisodeRelation",
    "ForkTreeNode",
    "LineageNode",
    "LineageRecord",
    "LineageView",
    "build_episode_lineage",
    "build_lineage",
    "lineage_view",
]

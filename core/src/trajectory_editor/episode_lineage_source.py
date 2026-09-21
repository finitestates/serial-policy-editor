"""Adapt primitive episode relation rows into a typed lineage view."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .core.errors import EditorError
from .episode_lineage import EpisodeRelation, LineageView, build_lineage


class EpisodeLineageReader(Protocol):
    """Primitive relation read needed to construct a lineage view."""

    def episode_relation_rows(self) -> Sequence[Mapping[str, Any]]: ...


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = row.get("metadata_json", "{}")
    if not isinstance(raw, str):
        raise EditorError("saved episode metadata must be JSON text")
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EditorError("saved episode metadata must be valid JSON") from exc
    return metadata if isinstance(metadata, Mapping) else {}


def _relation(row: Mapping[str, Any]) -> EpisodeRelation:
    metadata = _metadata(row)
    mode = metadata.get("mode")
    if not isinstance(mode, str) or not mode or mode != mode.strip():
        mode = "interactive"
    raw_source = metadata.get("spr_source")
    source = raw_source
    if source is not None and not isinstance(source, str):
        source = str(source)
    if isinstance(source, str) and (not source or any(char.isspace() for char in source)):
        # Older stores permitted arbitrary JSON metadata.  Keep such records
        # inspectable as replays without passing an invalid relation ID into
        # the typed graph.
        mode = "serial-policy-replay"
        source = None
    if mode == "serial-policy-replay" and raw_source is None:
        source = row.get("parent_episode_id")
    try:
        return EpisodeRelation(
            episode_id=row.get("episode_id"),
            parent_id=row.get("parent_episode_id"),
            fork_boundary=row.get("fork_boundary"),
            mode=mode,
            spr_source_id=source,
            status=row.get("status"),
            creation_key=row.get("created_at"),
            terminal_reason=row.get("terminal_reason"),
            visible_token_count=row.get("visible_token_count"),
        )
    except (TypeError, ValueError) as exc:
        raise EditorError(f"saved episode relation is invalid: {exc}") from exc


def build_lineage_view(
    reader: EpisodeLineageReader,
    selected_episode_id: str,
) -> LineageView:
    """Transform one reader's flat relation rows into a lineage view."""

    try:
        return build_lineage(
            (_relation(row) for row in reader.episode_relation_rows()),
            selected_episode_id,
        )
    except KeyError as exc:
        raise EditorError(f"unknown episode {selected_episode_id!r}") from exc


__all__ = ["EpisodeLineageReader", "build_lineage_view"]

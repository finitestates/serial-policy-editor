from __future__ import annotations

import pytest

from trajectory_editor.core.errors import EditorError
from trajectory_editor.episode_lineage_source import build_lineage_view


class MemoryReader:
    """Primitive relation reader; deliberately not an EpisodeStore."""

    def episode_relation_rows(self):
        return [
            {
                "episode_id": "root",
                "parent_episode_id": None,
                "fork_boundary": None,
                "status": "completed",
                "created_at": "2026-01-01T00:00:00+00:00",
                "terminal_reason": None,
                "metadata_json": "{}",
                "visible_token_count": 1,
            },
            {
                "episode_id": "child",
                "parent_episode_id": "root",
                "fork_boundary": 1,
                "status": "completed",
                "created_at": "2026-01-02T00:00:00+00:00",
                "terminal_reason": None,
                "metadata_json": '{"mode":"fork"}',
                "visible_token_count": 2,
            },
            {
                "episode_id": "replay",
                "parent_episode_id": "root",
                "fork_boundary": 0,
                "status": "completed",
                "created_at": "2026-01-03T00:00:00+00:00",
                "terminal_reason": "end-of-generation",
                "metadata_json": '{"mode":"serial-policy-replay"}',
                "visible_token_count": 3,
            },
            {
                "episode_id": "from-replay",
                "parent_episode_id": "replay",
                "fork_boundary": 2,
                "status": "open",
                "created_at": "2026-01-04T00:00:00+00:00",
                "terminal_reason": None,
                "metadata_json": '{}',
                "visible_token_count": 4,
            },
        ]


def test_reader_adapter_builds_typed_lineage_without_a_store():
    view = build_lineage_view(MemoryReader(), "child")

    assert view.ordinary_family_root_id == "root"
    assert view.ordinary_fork_tree.episode_id == "root"
    assert [node.episode_id for node in view.ordinary_fork_tree.children] == ["child"]
    assert [record.episode_id for record in view.related_replays] == ["replay"]
    assert view.related_replays[0].spr_source_id == "root"
    assert [record.episode_id for record in view.replay_derived_forks] == ["from-replay"]


def test_reader_adapter_keeps_durable_unknown_episode_error():
    with pytest.raises(EditorError, match="unknown episode 'missing'"):
        build_lineage_view(MemoryReader(), "missing")

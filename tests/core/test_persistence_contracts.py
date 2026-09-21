from __future__ import annotations

from dataclasses import replace

import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor.core.actions import Accept, Hold, Write
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.projector import (
    project_episode,
    project_fork_map,
    project_lineage,
    project_procedure,
)
from trajectory_editor.episode_replay_source import replay_tape
from trajectory_editor.episode_store import EpisodeStore


def runtime(*, sampling=None):
    from trajectory_editor.episode_engine import EpisodeEngine

    return EpisodeEngine(
        ConformingFakeBackend(),
        initial_text="P",
        initial_token_ids=[7],
        sampling=sampling or SamplerConfig(temperature=0.0),
    )


def create(store, episode_id, episode, *, parent=None, boundary=None, metadata=None):
    return store.create_episode(
        episode_id=episode_id,
        initial_text=episode.initial_text,
        initial_token_ids=list(episode.initial_token_ids),
        sampling=episode.sampling,
        stream_fingerprint=episode.stream_fingerprint,
        coordinate_offset=episode.coordinate_offset,
        max_tokens=episode.max_tokens,
        backend=episode.backend.provenance(),
        parent_episode_id=parent,
        fork_boundary=boundary,
        metadata=metadata,
    )


def save_live(store, episode_id, episode):
    store.update_episode(
        episode_id,
        visible_text=episode.backend.render(episode.visible_token_ids),
        max_tokens=episode.max_tokens,
    )


def test_p01_episode_actions_persist_and_reload(tmp_path):
    path = tmp_path / "episodes.sqlite3"
    sampling = SamplerConfig(temperature=0.4, seed=17)
    episode = runtime(sampling=sampling)
    with EpisodeStore(path) as store:
        identifier = create(store, "persisted", episode)
        outcome = episode.apply(Accept())
        store.record_action(identifier, 0, outcome)
        save_live(store, identifier, episode)

    with EpisodeStore(path) as reopened:
        record = reopened.get_episode("persisted")
        actions = reopened.actions("persisted")
        tokens = reopened.tokens("persisted")

    assert record["visible_text"] == " A"
    assert record["status"] == "open"
    assert actions[0]["kind"] == "accept"
    assert actions[0]["boundary_after"] == 1
    assert tokens[0]["token_id"] == 1


def test_p02_persisted_tape_excludes_editorial_interactions(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode = runtime()
        identifier = create(store, "tape", episode)
        store.record_action(identifier, 0, episode.apply(Hold(1)))
        store.record_interaction(identifier, 1, "search", {"query": "word"})
        store.record_interaction(identifier, 1, "rewind-requested", {"boundary": 0})

        rows = store.connection.execute(
            "SELECT kind, arguments_json, mismatch_json FROM actions WHERE episode_id = ?",
            (identifier,),
        ).fetchall()
        tape = replay_tape(store, identifier)

    assert len(rows) == 1
    assert rows[0]["kind"] == "hold"
    assert rows[0]["mismatch_json"] is None
    assert len(tape) == 1
    assert tape[0][0] == Hold(1)


def test_p03_fork_maps_preserve_exact_visible_boundaries_including_zero(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode = runtime()
        identifier = create(store, "fork-map", episode)
        assert project_fork_map(store, identifier) == "P|0|"

        first = episode.apply(Accept())
        store.record_action(identifier, 0, first)
        second = episode.apply(Accept())
        store.record_action(identifier, 1, second)
        save_live(store, identifier, episode)

        rendered = project_fork_map(store, identifier)

    assert rendered == "P|0| A|1| B|2|"


def test_p04_exports_preserve_procedure_and_lineage_semantics(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        parent = runtime()
        parent_id = create(store, "parent", parent)
        first = parent.apply(Write(" A", mode="exact"))
        store.record_action(parent_id, 0, first)
        save_live(store, parent_id, parent)

        child = runtime(sampling=replace(parent.sampling, seed=91))
        child_id = create(
            store,
            "child",
            child,
            parent=parent_id,
            boundary=1,
            metadata={"mode": "fork"},
        )
        plain = project_episode(store, parent_id)
        procedure = project_procedure(store, parent_id)
        lineage = project_lineage(store, child_id)
        relation_rows = store.episode_relation_rows()

    assert plain.text == "P A"
    assert "P       : P" in procedure
    assert "0 : x  A" in procedure
    assert "fork family:" in lineage
    assert "child" in lineage
    assert [
        {key: value for key, value in row.items() if key != "created_at"}
        for row in relation_rows
    ] == [
        {
            "episode_id": "parent",
            "parent_episode_id": None,
            "fork_boundary": None,
            "status": "open",
            "terminal_reason": None,
            "metadata_json": "{}",
            "visible_token_count": 1,
        },
        {
            "episode_id": "child",
            "parent_episode_id": "parent",
            "fork_boundary": 1,
            "status": "running",
            "terminal_reason": None,
            "metadata_json": '{"mode":"fork"}',
            "visible_token_count": 0,
        },
    ]
    assert all(isinstance(row["created_at"], str) for row in relation_rows)


@pytest.mark.parametrize("failure", ["invalid allowance", "budget insert"])
def test_p05_failed_initial_budget_rolls_back_episode_creation(tmp_path, failure):
    path = tmp_path / "episodes.sqlite3"
    episode = runtime()
    with EpisodeStore(path) as store:
        if failure == "budget insert":
            store.connection.execute("""
                CREATE TRIGGER reject_initial_budget BEFORE INSERT ON budget_segments
                BEGIN SELECT RAISE(ABORT, 'budget insert failed'); END
            """)
        error = ("invalid budget allowance or checkpoint" if failure == "invalid allowance"
                 else "budget insert failed")
        with pytest.raises(EditorError, match=error):
            store.create_episode(
                episode_id="failed",
                initial_text=episode.initial_text,
                initial_token_ids=episode.initial_token_ids,
                sampling=episode.sampling,
                stream_fingerprint=episode.stream_fingerprint,
                coordinate_offset=episode.coordinate_offset,
                max_tokens=0 if failure == "invalid allowance" else 5,
                backend=episode.backend.provenance(),
            )

    with EpisodeStore(path) as store:
        for table in ("episodes", "episode_names", "sampler_segments", "budget_segments"):
            assert store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0

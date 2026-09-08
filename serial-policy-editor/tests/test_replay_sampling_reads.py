from dataclasses import replace

import pytest

from tests.test_replay_eog import create, engine
from trajectory_editor.domain import EditorError
from trajectory_editor.episode_actions import Write
from trajectory_editor.episode_store import EpisodeStore


@pytest.mark.parametrize("count", [1, 40])
def test_replay_loads_sampler_segments_once_and_preserves_transitions(tmp_path, count):
    runtime = engine()
    with EpisodeStore(tmp_path / "episode.sqlite3") as store:
        identifier = create(store, runtime)
        # Include exact boundaries, a transition within the episode, and one
        # beyond its final action. Each action still uses its boundary's policy.
        for boundary in (1, 7, 23, 100):
            store.record_sampling_segment(
                identifier, start_boundary=boundary,
                sampling=replace(runtime.sampling, seed=boundary),
                stream_fingerprint=runtime.stream_fingerprint, coordinate_offset=0,
            )
        for ordinal in range(count):
            store.record_action(identifier, ordinal, runtime.apply(Write("hello", "exact")))
        expected = [store.sampling_segment(identifier, at)["sampling"] for at in range(count)]
        queries = []
        store.connection.set_trace_callback(queries.append)
        try:
            steps = store.replay_procedure(identifier)
        finally:
            store.connection.set_trace_callback(None)
        assert [step["sampling"].to_dict() for step in steps] == expected
        assert [step["boundary"] for step in steps] == list(range(count))
        assert all(step["expectation"].token_ids == (4,) for step in steps)
        selects = [query for query in queries if query.lstrip().upper().startswith("SELECT")]
        assert len(selects) == 5  # Two episode checks plus actions, tokens, and segments.

        # The cache belongs to this read, so subsequent edits remain visible.
        changed = replace(runtime.sampling, seed=999)
        store.record_sampling_segment(
            identifier, start_boundary=0, sampling=changed,
            stream_fingerprint=runtime.stream_fingerprint, coordinate_offset=0,
        )
        assert store.replay_procedure(identifier)[0]["sampling"] == changed


def test_replay_without_applicable_sampler_retains_error(tmp_path):
    runtime = engine()
    with EpisodeStore(tmp_path / "episode.sqlite3") as store:
        identifier = create(store, runtime)
        store.record_action(identifier, 0, runtime.apply(Write("hello", "exact")))
        with store.transaction() as db:
            db.execute("DELETE FROM sampler_segments WHERE episode_id = ?", (identifier,))
        with pytest.raises(EditorError, match="no sampler segment"):
            store.replay_procedure(identifier)

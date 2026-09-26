from __future__ import annotations

import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor.core.actions import Accept
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_lifecycle import _restore_engine
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_store import EpisodeStore

pytestmark = pytest.mark.invariant

class NoEogBackend(ConformingFakeBackend):
    def last_logits(self):
        logits = super().last_logits()
        logits[0] = -100.0
        return logits



def runtime(backend=None, *, max_tokens=None, sampling=None):
    return EpisodeEngine(
        backend or NoEogBackend(),
        initial_text="P",
        initial_token_ids=[7],
        sampling=sampling or SamplerConfig(temperature=0.0),
        max_tokens=max_tokens,
    )


def create(store, episode_id, episode):
    return store.create_episode(
        episode_id=episode_id,
        initial_text=episode.initial_text,
        initial_token_ids=list(episode.initial_token_ids),
        sampling=episode.sampling,
        stream_fingerprint=episode.stream_fingerprint,
        max_tokens=episode.max_tokens,
        backend=episode.backend.provenance(),
    )


def test_l01_resume_reconstructs_an_open_episode_and_continues(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        source = runtime()
        identifier = create(store, "resume", source)
        first = source.apply(Accept())
        store.record_action(identifier, 0, first)
        store.update_episode(identifier, visible_text=source.text, max_tokens=None)

        restored = _restore_engine(
            store, identifier, NoEogBackend(), max_tokens=None, sampling_override=None
        )
        next_outcome = restored.apply(Accept())

    assert restored.visible_token_ids == [1, 2]
    assert next_outcome.visible_token_ids == (2,)


@pytest.mark.parametrize("sealed_status", ["completed", "failed"])
def test_sealed_episodes_reject_all_record_writes(tmp_path, sealed_status):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode = runtime()
        identifier = create(store, "sealed", episode)
        outcome = episode.apply(Accept())
        store.finish_episode(
            identifier,
            visible_text=episode.text,
            terminal_token_id=None,
            terminal_reason="teacher-end",
        )
        if sealed_status == "failed":
            store.connection.execute(
                "UPDATE episodes SET status = 'failed' WHERE episode_id = ?",
                (identifier,),
            )
            store.connection.commit()

        before = (
            store.get_episode(identifier),
            store.budget_segments(identifier),
            store.sampler_segments(identifier),
            store.actions(identifier),
            store.interactions(identifier),
        )
        writes = (
            lambda: store.record_budget(identifier, 0, 1, 1),
            lambda: store.record_sampling_segment(
                identifier,
                start_boundary=0,
                sampling=episode.sampling,
                stream_fingerprint=episode.stream_fingerprint,
            ),
            lambda: store.record_action(identifier, 0, outcome),
            lambda: store.record_interaction(identifier, 0, "search", {"query": "word"}),
        )
        for write in writes:
            with pytest.raises(EditorError, match="sealed"):
                write()

        assert before == (
            store.get_episode(identifier),
            store.budget_segments(identifier),
            store.sampler_segments(identifier),
            store.actions(identifier),
            store.interactions(identifier),
        )

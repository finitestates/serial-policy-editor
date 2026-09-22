from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor.core.actions import Accept, Hold, Phrase, Write
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_lifecycle import (
    _create_episode,
    _fork_engine,
    _model_continuation,
    _restore_engine,
    _rewind_episode,
)
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_replay_source import replay_tape
from trajectory_editor.projector import project_fork_map, project_lineage
from trajectory_editor.episode_store import EpisodeStore


class NoEogBackend(ConformingFakeBackend):
    def last_logits(self):
        logits = super().last_logits()
        logits[0] = -100.0
        return logits


class PhraseBackend(NoEogBackend):
    def tokenize(self, text, *, add_bos=False, special=False):
        if not add_bos and text == "C!":
            return [3, 5]
        return super().tokenize(text, add_bos=add_bos, special=special)


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
        coordinate_offset=episode.coordinate_offset,
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


def test_l02_rewind_restores_any_retained_token_boundary(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode = runtime()
        identifier = create(store, "rewind", episode)
        outcome = episode.apply(Write(" A B", mode="exact"))
        store.record_action(identifier, 0, outcome)
        store.update_episode(identifier, visible_text=episode.text, max_tokens=None)

        _rewind_episode(store, identifier, episode, 1)
        retained = store.tokens(identifier)

    assert episode.visible_token_ids == [1]
    assert retained[0]["token_id"] == 1


def test_l03_rewind_can_cut_inside_a_checked_multitoken_write(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode = runtime(PhraseBackend())
        identifier = create(store, "phrase-rewind", episode)
        outcome = episode.apply(Phrase("C!", mode="exact", max_shift=100.0))
        store.record_action(identifier, 0, outcome)
        store.update_episode(identifier, visible_text=episode.text, max_tokens=None)

        _rewind_episode(store, identifier, episode, 1)
        action, expectation = replay_tape(store, identifier)[0]

    assert action == Write(" C", mode="exact")
    assert expectation.token_ids == (3,)
    assert episode.visible_token_ids == [3]


def test_l04_fork_preserves_exactly_the_requested_visible_prefix(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode = runtime()
        identifier = create(store, "parent", episode)
        outcome = episode.apply(Hold(2))
        store.record_action(identifier, 0, outcome)
        store.update_episode(identifier, visible_text=episode.text, max_tokens=None)

        child = _fork_engine(
            store, identifier, episode, 1, backend=NoEogBackend(), max_tokens=None
        )

    assert child.visible_token_ids == [1]
    assert child.initial_token_ids == (7,)
    assert child.text == "P A"


@pytest.mark.parametrize("cfg_prefix_tokens", [3, 5])
def test_l04_cfg_fork_counts_inherited_tokens_once(tmp_path, cfg_prefix_tokens):
    sampling = SamplerConfig(
        temperature=0.0,
        cfg_unconditional_prompt="P",
        cfg_scale=2.0,
        cfg_prefix_tokens=cfg_prefix_tokens,
    )
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        parent = EpisodeEngine(
            NoEogBackend(),
            initial_text="P",
            initial_token_ids=[7],
            sampling=sampling,
            guidance_backend=NoEogBackend(),
        )
        parent_id = create(store, "parent", parent)
        outcome = parent.apply(Write(" A B", mode="exact"))
        store.record_action(parent_id, 0, outcome)

        parent_observation = parent.observe()
        child = _fork_engine(
            store,
            parent_id,
            parent,
            2,
            backend=NoEogBackend(),
            max_tokens=None,
            guidance_backend=NoEogBackend(),
        )
        child_observation = child.observe()

    assert child_observation.proposal_token_id == parent_observation.proposal_token_id
    np.testing.assert_array_equal(child_observation.logits, parent_observation.logits)
    assert child.guidance_backend.tokens == [7, 1, 2]


def test_l04a_persisted_fork_keeps_root_relative_history_and_can_rewind_to_zero(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        parent = runtime()
        parent_id = create(store, "parent", parent)
        outcome = parent.apply(Hold(2))
        store.record_action(parent_id, 0, outcome)
        store.update_episode(
            parent_id,
            visible_text=parent.backend.render(parent.visible_token_ids),
            max_tokens=None,
        )

        child_engine = _fork_engine(
            store, parent_id, parent, 1, backend=NoEogBackend(), max_tokens=None
        )
        child_id = _create_episode(
            store,
            child_engine,
            backend_provenance={},
            parent_episode_id=parent_id,
            fork_boundary=1,
            mode="fork",
        )
        store.copy_prefix(
            parent_id,
            child_id,
            1,
            visible_text=child_engine.backend.render(child_engine.visible_token_ids),
            max_tokens=child_engine.max_tokens,
        )

        child = _restore_engine(
            store, child_id, NoEogBackend(), max_tokens=None, sampling_override=None
        )
        assert store.get_episode(child_id)["initial_text"] == "P"
        assert store.get_episode(child_id)["initial_token_ids"] == [7]
        assert child.visible_token_ids == [1]
        assert project_fork_map(store, child_id) == "P|0| A|1|"

        _rewind_episode(store, child_id, child, 0)
        assert child.visible_token_ids == []
        assert store.actions(child_id) == []
        assert store.tokens(child_id) == []
        assert project_fork_map(store, child_id) == "P|0|"


def test_l04b_nested_persisted_forks_keep_the_same_root_coordinates(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        parent = runtime()
        parent_id = create(store, "parent", parent)
        outcome = parent.apply(Hold(2))
        store.record_action(parent_id, 0, outcome)
        store.update_episode(
            parent_id,
            visible_text=parent.backend.render(parent.visible_token_ids),
            max_tokens=None,
        )

        child_engine = _fork_engine(
            store, parent_id, parent, 1, backend=NoEogBackend(), max_tokens=None
        )
        child_id = _create_episode(
            store,
            child_engine,
            backend_provenance={},
            parent_episode_id=parent_id,
            fork_boundary=1,
            mode="fork",
        )
        store.copy_prefix(
            parent_id,
            child_id,
            1,
            visible_text=child_engine.backend.render(child_engine.visible_token_ids),
            max_tokens=child_engine.max_tokens,
        )

        grandchild_engine = _fork_engine(
            store,
            child_id,
            child_engine,
            1,
            backend=NoEogBackend(),
            max_tokens=None,
        )
        grandchild_id = _create_episode(
            store,
            grandchild_engine,
            backend_provenance={},
            parent_episode_id=child_id,
            fork_boundary=1,
            mode="fork",
        )
        store.copy_prefix(
            child_id,
            grandchild_id,
            1,
            visible_text=grandchild_engine.backend.render(
                grandchild_engine.visible_token_ids
            ),
            max_tokens=grandchild_engine.max_tokens,
        )

        restored = _restore_engine(
            store,
            grandchild_id,
            NoEogBackend(),
            max_tokens=None,
            sampling_override=None,
        )

        assert store.get_episode(grandchild_id)["initial_text"] == "P"
        assert restored.initial_token_ids == (7,)
        assert restored.visible_token_ids == [1]
        assert project_fork_map(store, grandchild_id) == "P|0| A|1|"


def test_l05_fork_lineage_and_fork_map_keep_editorial_boundaries(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        root = runtime()
        create(store, "root", root)
        store.update_episode("root", visible_text="P", max_tokens=None)
        store.create_episode(
            episode_id="child",
            initial_text="P",
            initial_token_ids=[7],
            sampling=root.sampling,
            stream_fingerprint=root.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=None,
            backend={"backend": "fake"},
            parent_episode_id="root",
            fork_boundary=1,
            metadata={"mode": "fork"},
        )
        store.record_action("root", 0, root.apply(Accept()))
        store.update_episode("root", visible_text=" A", max_tokens=None)
        fork_map = project_fork_map(store, "root")
        lineage = project_lineage(store, "child")

    assert fork_map.startswith("P|0|")
    assert "fork family:" in lineage
    assert "child" in lineage


def test_l06_rewind_restores_sampler_transition_at_the_selected_boundary(tmp_path):
    original = SamplerConfig(temperature=1.7, top_k=8, top_p=1.0, min_p=0.0, seed=19)
    changed = replace(original, temperature=0.4, top_k=2, seed=83)
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode = runtime(sampling=original)
        identifier = create(store, "sampler", episode)
        store.record_action(identifier, 0, episode.apply(Hold(1)))
        episode.sampling = changed
        episode.stream_fingerprint = "b" * 64
        episode.coordinate_offset = 29
        store.record_sampling_segment(
            identifier,
            start_boundary=1,
            sampling=changed,
            stream_fingerprint=episode.stream_fingerprint,
            coordinate_offset=29,
        )

        _rewind_episode(store, identifier, episode, 0)

    assert episode.sampling == original
    assert episode.coordinate_offset == 0
    assert episode.stream_fingerprint != "b" * 64


@pytest.mark.parametrize("target, expected_remaining", [(0, 2), (1, 1)])
def test_l07_budget_state_follows_the_retained_boundary(tmp_path, target, expected_remaining):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode = runtime(max_tokens=2)
        identifier = _create_episode(store, episode, backend_provenance={})
        outcome = episode.apply(Hold(2))
        store.record_action(identifier, 0, outcome)
        store.update_episode(identifier, visible_text=episode.text, max_tokens=episode.max_tokens)

        _rewind_episode(store, identifier, episode, target)

    assert episode.remaining == expected_remaining
    assert not episode.ended


def test_l08_model_continuation_preserves_visible_text_and_sampler_state(tmp_path):
    class NewTokenizer(NoEogBackend):
        def tokenize(self, text, **kwargs):
            self.received_text = text
            if kwargs.get("add_bos"):
                return [7]
            return [7, 4]

    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        source = runtime(max_tokens=10)
        identifier = create(store, "source", source)
        outcome = source.apply(Write(" A", mode="exact"))
        store.record_action(identifier, 0, outcome)
        store.update_episode(identifier, visible_text=" A", max_tokens=source.max_tokens)

        backend = NewTokenizer()
        continued, child_id = _model_continuation(store, identifier, backend, {})
        child = store.get_episode(child_id)
        assert child["initial_text"] == "P"
        assert child["initial_token_ids"] == [7]
        assert len(store.actions(child_id)) == 1

    assert backend.received_text == " A"
    assert continued.initial_text == "P"
    assert continued.initial_token_ids == (7,)
    assert continued.visible_token_ids == [7, 4]
    assert continued.sampling == source.sampling
    assert child_id != identifier

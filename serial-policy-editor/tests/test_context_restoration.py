from unittest.mock import patch

import pytest

from tests.fakes import BranchingFakeBackend, ConformingFakeBackend
from tests.test_episode_runtime import engine
from tests.test_replay_eog import create
from trajectory_editor.episode_actions import Accept, Write
from trajectory_editor.episode_cli import _restore_engine
from trajectory_editor.episode_store import EpisodeStore


@pytest.mark.parametrize("backend_type", [ConformingFakeBackend, BranchingFakeBackend])
@pytest.mark.parametrize("boundary", [0, 1, 2])
def test_rewind_repositions_once_and_replays_next_move(backend_type, boundary):
    backend = backend_type()
    runtime = engine(backend, max_tokens=10)
    runtime.apply(Write(" A B", "exact"))
    reference = engine(max_tokens=10)
    if boundary:
        reference.apply(Write(" A" if boundary == 1 else " A B", "exact"))
    expected = reference.apply(Accept())
    with patch.object(backend, "reset", wraps=backend.reset) as reset, patch.object(
        backend, "eval", wraps=backend.eval
    ) as evaluate:
        runtime.rewind_to(boundary)
        assert evaluate.call_count == 0
        if isinstance(backend, BranchingFakeBackend):
            assert backend.branch_prefixes == [[7, *[1, 2][:boundary]]]
            assert reset.call_count == 0
        else:
            reset.assert_called_once_with([7, *[1, 2][:boundary]])
    assert runtime.initial_token_ids == (7,)
    assert runtime.remaining == 10 - boundary
    result = runtime.apply(Accept(), expectation=expected.expectation())
    assert result.divergence is None
    assert result.resolved_token_ids == expected.resolved_token_ids


@pytest.mark.parametrize("visible", [False, True])
def test_resume_batches_known_tokens_and_replays_next_move(tmp_path, visible):
    source = engine(max_tokens=10)
    with EpisodeStore(tmp_path / "episode.sqlite3") as store:
        identifier = create(store, source)
        if visible:
            store.record_action(identifier, 0, source.apply(Write(" A B", "exact")))
        backend = ConformingFakeBackend()
        with patch.object(backend, "eval", wraps=backend.eval) as evaluate:
            restored = _restore_engine(
                store, identifier, backend, max_tokens=None, sampling_override=None
            )
            if visible:
                evaluate.assert_called_once_with([1, 2])
            else:
                evaluate.assert_not_called()
        assert restored.initial_token_ids == source.initial_token_ids
        assert restored.visible_token_ids == source.visible_token_ids
        assert restored.coordinate_offset == source.coordinate_offset
        assert restored.stream_fingerprint == source.stream_fingerprint
        expected = source.apply(Accept())
        result = restored.apply(Accept(), expectation=expected.expectation())
        assert result.divergence is None
        assert result.resolved_token_ids == expected.resolved_token_ids

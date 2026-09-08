from dataclasses import replace

import pytest

from tests.test_episode_runtime import NoEogBackend
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.episode_actions import Hold
from trajectory_editor.episode_cli import _fork_engine, _restore_engine, _rewind_episode
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_store import EpisodeStore


@pytest.mark.parametrize("boundary", [0, 2, 3, 5])
def test_rewind_restores_historical_sampler_and_hold_matches_fork(tmp_path, boundary):
    original = SamplingConfig(temperature=1.7, top_k=8, top_p=1.0, min_p=0.0, seed=19)
    runtime = EpisodeEngine(
        NoEogBackend(), sampling=original, initial_token_ids=[7],
        stream_fingerprint="a" * 64, coordinate_offset=11,
    )
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        identifier = store.create_episode(
            initial_text=runtime.text, initial_token_ids=[7], sampling=original,
            stream_fingerprint=runtime.stream_fingerprint, coordinate_offset=11,
            max_tokens=None, backend={},
        )
        store.record_action(identifier, 0, runtime.apply(Hold(2)))
        runtime.sampling = replace(original, seed=83, temperature=0.4, top_k=2)
        runtime.stream_fingerprint = "b" * 64
        runtime.coordinate_offset = 29
        store.record_sampling_segment(
            identifier, start_boundary=2, sampling=runtime.sampling,
            stream_fingerprint=runtime.stream_fingerprint, coordinate_offset=29,
        )
        store.record_action(identifier, 1, runtime.apply(Hold(3)))
        expected_segment = store.sampling_segment(identifier, boundary)
        fork = _fork_engine(
            store, identifier, runtime, boundary, backend=NoEogBackend(), max_tokens=None,
        )
        # Populate an observation at the future boundary to catch stale evidence.
        runtime.observe()
        _rewind_episode(store, identifier, runtime, boundary)
        assert runtime.sampling == SamplingConfig.from_mapping(expected_segment["sampling"])
        assert runtime.stream_fingerprint == expected_segment["stream_fingerprint"]
        assert runtime.coordinate_offset == expected_segment["coordinate_offset"]
        assert runtime.observe().sampling_coordinate == fork.observe().sampling_coordinate
        with EpisodeStore(store.path) as reopened:
            restored = _restore_engine(
                reopened, identifier, NoEogBackend(), max_tokens=None, sampling_override=None,
            )
        expected = fork.apply(Hold(5))
        actual = runtime.apply(Hold(5), expectation=expected.expectation())
        resumed = restored.apply(Hold(5), expectation=expected.expectation())
        assert actual.divergence is None
        assert resumed.divergence is None
        assert actual.resolved_token_ids == expected.resolved_token_ids == resumed.resolved_token_ids

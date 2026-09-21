import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor.bias_presets import load_bias_preset, project_biases
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_cli import build_parser
from trajectory_editor.token_preference_features import (
    DEFAULT_PROJECTION_CHUNK_SIZE,
    project_token_embeddings,
)


FEATURES = np.asarray(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.95, 0.05],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [0.2, 0.0],
        [0.0, 0.0],
    ],
    dtype=np.float32,
)


class TokenPreferenceBackend(ConformingFakeBackend):
    def token_preference_features(self, *, feature_dimension: int, projection_seed: int):
        del projection_seed
        assert feature_dimension == FEATURES.shape[1]
        return FEATURES


def test_fixed_projection_is_deterministic_and_normalizes_rows():
    embeddings = np.arange(20, dtype=np.float32).reshape(5, 4)
    first = project_token_embeddings(embeddings, feature_dimension=3, projection_seed=7)
    second = project_token_embeddings(embeddings, feature_dimension=3, projection_seed=7)

    assert np.array_equal(first, second)
    assert np.all(np.linalg.norm(first, axis=1) <= 1.0 + 1.0e-6)
    assert not first.flags.writeable


def test_projection_chunking_preserves_features():
    embeddings = np.arange(60, dtype=np.float32).reshape(15, 4)
    full = project_token_embeddings(
        embeddings,
        feature_dimension=3,
        projection_seed=7,
        projection_chunk_size=len(embeddings),
    )
    chunked = project_token_embeddings(
        embeddings,
        feature_dimension=3,
        projection_seed=7,
        projection_chunk_size=2,
    )

    assert np.allclose(full, chunked, rtol=1.0e-6, atol=1.0e-7)


def test_full_bias_export_includes_token_preference_state(tmp_path):
    from trajectory_editor.episode_lifecycle import _create_episode

    sampling = SamplingConfig(
        token_preference_vector=(0.5, -0.25),
        token_preference_strength=0.75,
    )
    runtime = EpisodeEngine(TokenPreferenceBackend(), initial_token_ids=[7], sampling=sampling)
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode_id = _create_episode(
            store,
            runtime,
            backend_provenance=runtime.backend.provenance(),
        )
        path = tmp_path / "biases.json"
        path.write_text(project_biases(store, episode_id), encoding="utf-8")
        restored = load_bias_preset(
            path, runtime.backend, runtime.backend.provenance()
        )

    assert restored.token_preference_vector == sampling.token_preference_vector
    assert restored.token_preference_strength == sampling.token_preference_strength


def test_token_preference_cli_controls_are_explicit_and_default_off():
    parser = build_parser()
    defaults = parser.parse_args([])
    assert defaults.token_preference is False
    assert defaults.token_preference_projection_chunk_size == DEFAULT_PROJECTION_CHUNK_SIZE
    args = parser.parse_args([
        "--token-preference",
        "--token-preference-dimension", "32",
        "--token-preference-learning-rate", "0.1",
        "--token-preference-strength", "0.7",
        "--token-preference-max-step", "0.03",
        "--token-preference-max-norm", "1.5",
        "--token-preference-projection-chunk-size", "1024",
    ])
    assert args.token_preference is True
    assert args.token_preference_dimension == 32
    assert args.token_preference_learning_rate == 0.1
    assert args.token_preference_strength == 0.7
    assert args.token_preference_max_step == 0.03
    assert args.token_preference_max_norm == 1.5
    assert args.token_preference_projection_chunk_size == 1024

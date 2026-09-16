import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from trajectory_editor.domain import EditorError, SamplingConfig
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.token_preference_features import TokenPreferenceCoordinateIdentity
from trajectory_editor.vector_artifacts import (
    FORMAT,
    TokenPreferenceVectorArtifact,
    assert_compatible,
    blend_artifacts,
)
from trajectory_editor.vector_cli import main
from tests.fakes import ConformingFakeBackend


IDENTITY = TokenPreferenceCoordinateIdentity(
    model_fingerprint=None,
    embedding_width=None,
    dimension=2,
    projection_seed=7,
    feature_scheme="random-projection-unit-v1",
    whitening_scheme="none-v1",
    whitening_ridge=0.0,
)


def artifact(vector=(0.1, 0.2), *, strength=1.0, fast=(), fast_strength=0.0):
    return TokenPreferenceVectorArtifact(
        model={"backend": "fake", "vocabulary_size": 8},
        coordinate_identity=IDENTITY,
        token_preference_vector=vector,
        token_preference_fast_vector=fast,
        token_preference_strength=strength,
        token_preference_fast_strength=fast_strength,
    )


def test_artifact_round_trip_is_explicit_and_versioned(tmp_path):
    path = tmp_path / "preference.json"
    original = artifact(fast=(0.3, -0.1), fast_strength=0.5)
    original.write(path)

    loaded = TokenPreferenceVectorArtifact.from_path(path)

    assert json.loads(path.read_text())['format'] == FORMAT
    assert loaded == original


def test_artifact_rejects_legacy_or_wrong_vector_shapes():
    with pytest.raises(EditorError, match="format"):
        TokenPreferenceVectorArtifact.from_mapping({"format": "spe-bias-rules-v4"})
    with pytest.raises(EditorError, match="same dimension"):
        TokenPreferenceVectorArtifact(
            model={},
            coordinate_identity=IDENTITY,
            token_preference_vector=(0.1, 0.2),
            token_preference_fast_vector=(0.3,),
        )


def test_blend_uses_effective_actuation_and_requires_compatible_coordinates():
    result = blend_artifacts(
        [artifact((0.1, 0.2), strength=2.0), artifact((-0.5, 0.4))],
        [1.0, 0.5],
    )

    assert result.token_preference_strength == 1.0
    assert result.token_preference_vector == pytest.approx((-0.05, 0.6))
    with pytest.raises(EditorError, match="coordinates"):
        assert_compatible(
            artifact(),
            TokenPreferenceVectorArtifact(
                model={"backend": "fake", "vocabulary_size": 8},
                coordinate_identity=replace(IDENTITY, projection_seed=8),
                token_preference_vector=(0.1, 0.2),
            ),
        )


class VectorBackend(ConformingFakeBackend):
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [0.5, 0.5],
            [-0.5, -0.5],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    def token_preference_features(self, *, feature_dimension, projection_seed):
        assert (feature_dimension, projection_seed) == (2, 7)
        return self.features

    def token_preference_coordinate_identity(
        self, *, feature_dimension, projection_seed, feature_scheme, whitening_ridge
    ):
        del feature_scheme, whitening_ridge
        return TokenPreferenceCoordinateIdentity(
            model_fingerprint=None,
            embedding_width=None,
            dimension=feature_dimension,
            projection_seed=projection_seed,
            feature_scheme="random-projection-unit-v1",
            whitening_scheme="none-v1",
            whitening_ridge=0.0,
        )


def test_cli_extract_inspect_explain_validate_and_apply(tmp_path, capsys):
    workspace = tmp_path / "episodes.sqlite3"
    with EpisodeStore(workspace) as store:
        store.create_episode(
            initial_text="P",
            initial_token_ids=[7],
            sampling=SamplingConfig(
                token_preference_vector=(0.1, 0.2),
                token_preference_projection_seed=7,
            ),
            stream_fingerprint="0" * 64,
            coordinate_offset=0,
            max_tokens=None,
            backend={"backend": "fake", "vocabulary_size": 8},
            episode_id="source",
        )
    artifact_path = tmp_path / "preference.json"
    assert main([
        "token-preference", "extract", "--workspace", str(workspace),
        "--episode", "source", "--output", str(artifact_path),
    ]) == 0
    assert main(["token-preference", "inspect", str(artifact_path)]) == 0
    assert "dimension: 2" in capsys.readouterr().out

    with patch("trajectory_editor.vector_cli.create_backend", return_value=VectorBackend()):
        assert main([
            "token-preference", "validate", str(artifact_path),
            "--model", "fake.model",
        ]) == 0
        capsys.readouterr()
        assert main([
            "token-preference", "explain", str(artifact_path),
            "--model", "fake.model", "--format", "json", "--top", "2",
        ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["top_positive"][0]["token_id"] == 2
    assert report["top_negative"][0]["token_id"] == 4

    preset = tmp_path / "biases.json"
    preset.write_text(json.dumps({
        "format": "spe-bias-rules-v4",
        "model": {"backend": "fake", "vocabulary_size": 8},
        "bias_rules": [],
        "bias_groups": [],
    }))
    combined = tmp_path / "combined.json"
    assert main([
        "token-preference", "apply", str(artifact_path),
        "--biases", str(preset), "--output", str(combined),
    ]) == 0
    applied = json.loads(combined.read_text())
    assert applied["token_preference_vector"] == [0.1, 0.2]
    assert applied["token_preference_projection_seed"] == 7

    extracted = tmp_path / "extracted.json"
    assert main([
        "token-preference", "extract", "--preset", str(combined),
        "--output", str(extracted),
    ]) == 0
    assert TokenPreferenceVectorArtifact.from_path(extracted).token_preference_vector == pytest.approx((0.1, 0.2))

    second = tmp_path / "second.json"
    artifact((-0.2, 0.4)).write(second)
    blended = tmp_path / "blended.json"
    assert main([
        "token-preference", "blend", str(artifact_path), str(second),
        "--weights", "1", "0.5", "--output", str(blended),
    ]) == 0
    assert TokenPreferenceVectorArtifact.from_path(blended).token_preference_vector == pytest.approx((0.0, 0.4))

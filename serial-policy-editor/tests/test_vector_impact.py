from __future__ import annotations

import json

import numpy as np

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.episode_actions import Write
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_ui import InteractivePolicy
from trajectory_editor.token_preference_features import TokenPreferenceCoordinateIdentity
from trajectory_editor.tui import display_candidates
from trajectory_editor.vector_artifacts import TokenPreferenceVectorArtifact
from trajectory_editor.vector_cli import main
from trajectory_editor.vector_impact import impact_vector


IDENTITY = TokenPreferenceCoordinateIdentity(
    model_fingerprint=None,
    embedding_width=None,
    dimension=2,
    projection_seed=7,
    feature_scheme="random-projection-unit-v1",
    whitening_scheme="none-v1",
    whitening_ridge=0.0,
)


class ImpactBackend(ConformingFakeBackend):
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

    def token_preference_features(
        self,
        *,
        feature_dimension,
        projection_seed,
        projection_chunk_size=8192,
        feature_scheme="random-projection-unit-v1",
        whitening_ridge=1.0e-6,
    ):
        del projection_chunk_size, feature_scheme, whitening_ridge
        assert (feature_dimension, projection_seed) == (2, 7)
        return self.features

    def token_preference_coordinate_identity(
        self,
        *,
        feature_dimension,
        projection_seed,
        feature_scheme,
        whitening_ridge,
    ):
        del feature_dimension, projection_seed, feature_scheme, whitening_ridge
        return IDENTITY

    def provenance(self, *, include_model_sha256=True):
        del include_model_sha256
        return {"backend": "fake", "vocabulary_size": 8}


def _artifact() -> TokenPreferenceVectorArtifact:
    return TokenPreferenceVectorArtifact(
        model={"backend": "fake", "vocabulary_size": 8},
        coordinate_identity=IDENTITY,
        token_preference_vector=(1.0, 0.0),
        token_preference_strength=1.0,
    )


def _episode(store: EpisodeStore, episode_id: str, texts: tuple[str, ...]) -> None:
    runtime = EpisodeEngine(
        ConformingFakeBackend(),
        sampling=SamplingConfig(temperature=1.0, top_k=8, top_p=1.0, min_p=0.0),
        max_tokens=20,
        initial_text="P",
        initial_token_ids=[7],
    )
    store.create_episode(
        episode_id=episode_id,
        initial_text="P",
        initial_token_ids=[7],
        sampling=runtime.sampling,
        stream_fingerprint=runtime.stream_fingerprint,
        coordinate_offset=0,
        max_tokens=20,
        backend={"backend": "fake", "vocabulary_size": 8},
    )
    for ordinal, text in enumerate(texts):
        outcome = runtime.apply(Write(text, "exact"))
        store.record_action(episode_id, ordinal, outcome)
    store.update_episode(
        episode_id,
        visible_text=runtime.backend.render(runtime.visible_token_ids),
        max_tokens=runtime.max_tokens,
    )


def test_new_sampling_defaults_to_unit_temperature():
    assert SamplingConfig().temperature == 1.0


def test_impact_measures_one_vector_across_matched_contexts(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        _episode(store, "context-a", (" A", " B"))
        _episode(store, "context-b", (" B", " A"))
        report = impact_vector(
            store,
            ["context-a", "context-b"],
            ImpactBackend(),
            _artifact(),
            strengths=(-1.0, 0.0, 1.0),
            top=3,
            include_vectors=True,
            rollout=True,
        )

    assert report["format"] == "spe-vector-impact-v1"
    assert len(report["contexts"]) == 2
    assert report["aggregate"]["strength_sweep"][0]["metrics"]["positions"] == 4
    zero = report["aggregate"]["strength_sweep"][1]["metrics"]
    assert zero["effective_delta"]["centered_rms"] == 0.0
    positive = report["aggregate"]["strength_sweep"][2]["metrics"]
    assert positive["top_positive"][0]["token_id"] == 1
    assert len(positive["mean_effective_delta_vector"]) == 8
    assert report["contexts"][0]["strength_sweep"][2]["trajectory"]["same_rollout"] is False


def test_logit_view_is_compact_and_toggleable():
    runtime = EpisodeEngine(
        ConformingFakeBackend(),
        sampling=SamplingConfig(temperature=0.0),
        initial_text="P",
        initial_token_ids=[7],
    )
    candidates = runtime.candidates(runtime.observe(), count=3)
    io = ScriptedIO([])
    display_candidates(io, candidates, heading=True, logit_view="all")
    assert "raw-logit" in io.output[0]
    assert "eff-logit" in io.output[0]
    assert "Δlogit" in io.output[0]
    assert "+10.000" in io.output[1]

    io = ScriptedIO(["l", "1"])
    InteractivePolicy(io=io).choose(runtime, runtime.observe())
    assert any("raw-logit" in line and "eff-logit" in line for line in io.output)


def test_vector_cli_impact_emits_json(tmp_path, capsys, monkeypatch):
    workspace = tmp_path / "episodes.sqlite3"
    with EpisodeStore(workspace) as store:
        _episode(store, "context", (" A",))
    artifact_path = tmp_path / "vector.json"
    _artifact().write(artifact_path)
    monkeypatch.setattr("trajectory_editor.vector_cli.create_backend", lambda *args, **kwargs: ImpactBackend())

    assert main([
        "impact",
        "--vector", str(artifact_path),
        "--workspace", str(workspace),
        "--episodes", "context",
        "--model", "fake.model",
        "--format", "json",
        "--strengths", "0", "1",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["format"] == "spe-vector-impact-v1"
    assert report["vector"]["kind"] == "token-preference"

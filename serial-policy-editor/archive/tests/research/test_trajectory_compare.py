from __future__ import annotations

import json

import numpy as np

from tests.fakes import ConformingFakeBackend
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.token_preference_features import TokenPreferenceCoordinateIdentity
from trajectory_editor.trajectory_compare import compare_episodes, render_compare_report
from trajectory_editor.vector_cli import main


IDENTITY = TokenPreferenceCoordinateIdentity(
    model_fingerprint=None,
    embedding_width=None,
    dimension=2,
    projection_seed=7,
    feature_scheme="random-projection-unit-v1",
    whitening_scheme="none-v1",
    whitening_ridge=0.0,
)


class CompareFeatureBackend(ConformingFakeBackend):
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
        del feature_scheme, whitening_ridge
        assert (feature_dimension, projection_seed) == (2, 7)
        return IDENTITY

    def activation_width(self):
        return 2

    def activation_snapshot(self, text, *, layer="output", position="last"):
        del layer, position
        return np.asarray([float(len(text)), float(sum(ord(char) for char in text) % 17)])

    def provenance(self, *, include_model_sha256=True):
        del include_model_sha256
        return {"backend": "fake", "vocabulary_size": 8}


def _create(
    store: EpisodeStore,
    episode_id: str,
    *,
    parent_episode_id: str | None = None,
    fork_boundary: int | None = None,
    sampling: SamplingConfig | None = None,
) -> EpisodeEngine:
    runtime = EpisodeEngine(
        ConformingFakeBackend(),
        sampling=sampling or SamplingConfig(temperature=0.0, top_k=8, top_p=1.0, min_p=0.0),
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
        parent_episode_id=parent_episode_id,
        fork_boundary=fork_boundary,
    )
    return runtime


def _write(store: EpisodeStore, episode_id: str, runtime: EpisodeEngine, text: str) -> None:
    from trajectory_editor.episode_actions import Write

    outcome = runtime.apply(Write(text, "exact"))
    store.record_action(episode_id, 0, outcome)
    store.update_episode(
        episode_id,
        visible_text=runtime.backend.render(runtime.visible_token_ids),
        max_tokens=runtime.max_tokens,
    )


def test_compare_is_useful_without_a_learner(tmp_path):
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        _create(store, "root")
        baseline = _create(store, "baseline", parent_episode_id="root", fork_boundary=0)
        nautical = _create(store, "nautical", parent_episode_id="root", fork_boundary=0)
        store.rename("baseline", "baseline")
        store.rename("nautical", "nautical mood")
        _write(store, "baseline", baseline, " A")
        _write(store, "nautical", nautical, " B")

        report = compare_episodes(store, ["baseline", "nautical"])
        rendered = render_compare_report(report)

    assert report["format"] == "spe-trajectory-compare-v1"
    assert report["context"]["same_initial_token_prefix"]
    assert report["context"]["equal_visible_token_span"]
    comparison = report["comparisons"][0]
    assert comparison["candidate_label"].endswith("nautical mood")
    assert comparison["continuation"]["first_divergence"]["offset"] == 0
    assert comparison["learner_vector"]["available"] is False
    assert "learner vector: unavailable" in rendered


def test_compare_reports_learner_deltas_and_aggregates(tmp_path):
    reference_sampling = SamplingConfig(
        token_preference_vector=(0.1, 0.2),
        token_preference_strength=2.0,
        token_preference_projection_seed=7,
    )
    candidate_sampling = SamplingConfig(
        token_preference_vector=(0.2, -0.1),
        token_preference_strength=1.0,
        token_preference_projection_seed=7,
    )
    third_sampling = SamplingConfig(
        token_preference_vector=(0.3, 0.0),
        token_preference_strength=1.0,
        token_preference_projection_seed=7,
    )
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        _create(store, "reference", sampling=reference_sampling)
        _create(store, "candidate", sampling=candidate_sampling)
        _create(store, "third", sampling=third_sampling)

        report = compare_episodes(
            store,
            ["reference", "candidate", "third"],
            include_vectors=True,
        )

    candidate_delta = report["comparisons"][0]["learner_vector"]["slow"]["delta_vector"]
    assert np.allclose(candidate_delta, [0.0, -0.5])
    aggregate = report["aggregate"]["learner_slow_delta"]
    assert aggregate["count"] == 2
    assert np.allclose(aggregate["mean_delta_vector"], [0.05, -0.45])


def test_compare_can_compute_content_and_hidden_state_contrasts(tmp_path):
    content_sampling = SamplingConfig(token_preference_projection_seed=7)
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        reference = _create(store, "reference", sampling=content_sampling)
        candidate = _create(store, "candidate", sampling=content_sampling)
        _write(store, "reference", reference, " A")
        _write(store, "candidate", candidate, " B")

        report = compare_episodes(
            store,
            ["reference", "candidate"],
            backend=CompareFeatureBackend(),
            include_content=True,
            include_hidden_state=True,
            feature_dimension=2,
            activation_strengths=(0.25, 1.0, -1.0),
            include_vectors=True,
        )

    comparison = report["comparisons"][0]
    assert comparison["content_feature_vector"]["delta_vector"] == [-1.0, 1.0]
    hidden_state = comparison["hidden_state_vector"]
    assert hidden_state["available"] is True
    assert hidden_state["vector"]
    assert [row["multiplier"] for row in hidden_state["strength_sweep"]] == [0.25, 1.0, -1.0]
    assert hidden_state["strength_sweep"][-1]["orientation"] == "opposite"


def test_vector_cli_compare_emits_json(tmp_path, capsys):
    workspace = tmp_path / "episodes.sqlite3"
    with EpisodeStore(workspace) as store:
        _create(store, "reference")
        _create(store, "candidate")

    assert main([
        "compare",
        "--workspace",
        str(workspace),
        "--episodes",
        "reference",
        "candidate",
        "--format",
        "json",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["format"] == "spe-trajectory-compare-v1"
    assert report["reference_episode_id"] == "reference"
    assert report["comparisons"][0]["candidate_episode_id"] == "candidate"

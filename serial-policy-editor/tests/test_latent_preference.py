from types import SimpleNamespace

import numpy as np
import pytest
import trajectory_editor.latent_preference as latent_preference_module

from tests.fakes import ConformingFakeBackend
from trajectory_editor.bias_presets import load_bias_preset, project_biases
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.episode_actions import SelectRawRank
from trajectory_editor.episode_engine import EpisodeEngine, ReplayExpectation
from trajectory_editor.episode_policy import EpisodeRunner, TapeStep
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_cli import build_parser
from trajectory_editor.latent_features import (
    DEFAULT_PROJECTION_CHUNK_SIZE,
    project_token_embeddings,
)
from trajectory_editor.latent_preference import (
    LatentPreferenceConfig,
    LatentPreferenceLearner,
)
from trajectory_editor.sampling import ObservationStatistics


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


class LatentBackend(ConformingFakeBackend):
    def latent_token_features(self, *, feature_dimension: int, projection_seed: int):
        del projection_seed
        assert feature_dimension == FEATURES.shape[1]
        return FEATURES


def _observation(sampling: SamplingConfig, logits=None):
    values = np.asarray(
        logits if logits is not None else [2.0, 1.0, 0.5, -1.0, -2.0, -3.0, -4.0, -5.0],
        dtype=np.float64,
    )
    statistics = ObservationStatistics(
        values,
        sampling,
        [],
        latent_features=FEATURES if sampling.latent_preference_z else None,
    )
    return SimpleNamespace(
        boundary=0,
        logits=values,
        statistics=statistics,
    )


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


def test_weighted_mean_uses_float64_output_without_upcasting_features(monkeypatch):
    observed: dict[str, np.dtype] = {}
    real_einsum = latent_preference_module.np.einsum

    def recording_einsum(subscripts, left, right, *, out, dtype, optimize):
        assert subscripts == "v,vd->d"
        observed["probabilities"] = left.dtype
        observed["features"] = right.dtype
        observed["output"] = out.dtype
        assert dtype is np.float64
        assert optimize is False
        return real_einsum(
            subscripts,
            left,
            right,
            out=out,
            dtype=dtype,
            optimize=optimize,
        )

    monkeypatch.setattr(latent_preference_module.np, "einsum", recording_einsum)
    LatentPreferenceLearner(
        FEATURES,
        dimension=2,
    ).update(_observation(SamplingConfig()), 1, SamplingConfig())

    assert observed == {
        "probabilities": np.dtype(np.float64),
        "features": np.dtype(np.float32),
        "output": np.dtype(np.float64),
    }


def test_latent_state_adjusts_policy_logits_and_round_trips():
    sampling = SamplingConfig(
        latent_preference_z=(1.0, 0.0),
        latent_strength=2.0,
    )
    observation = _observation(sampling, logits=[0.0] * len(FEATURES))

    assert observation.statistics.adjusted[1] == pytest.approx(2.0)
    assert observation.statistics.adjusted[3] == pytest.approx(-2.0)
    assert SamplingConfig.from_record(sampling.to_dict()) == sampling


def test_latent_update_moves_toward_the_chosen_feature_and_is_bounded():
    sampling = SamplingConfig()
    result = LatentPreferenceLearner(
        FEATURES,
        enabled=True,
        dimension=2,
        learning_rate=1.0,
        max_step=0.1,
        max_norm=0.15,
    ).update(_observation(sampling), 1, sampling)

    assert result.old_policy_rank == 2
    assert result.severity > 0.0
    assert result.new_z[0] > 0.0
    assert result.update_norm <= 0.1 + 1.0e-9
    assert result.z_norm <= 0.15 + 1.0e-9
    assert len(result.new_z) == 2


def test_unseen_similar_token_moves_up_without_token_specific_state():
    sampling = SamplingConfig()
    logits = [1.0, 0.9, 0.89, -1.0, -2.0, -3.0, -4.0, -5.0]
    before = _observation(sampling, logits=logits)
    learner = LatentPreferenceLearner(
        FEATURES,
        enabled=True,
        dimension=2,
        learning_rate=5.0,
        max_step=0.5,
        max_norm=2.0,
    )
    result = learner.update(before, 1, sampling)
    after = _observation(result.sampling, logits=before.logits)

    assert 2 != result.chosen_token_id
    assert after.statistics.policy_rank(2) < before.statistics.policy_rank(2)
    assert not hasattr(learner, "token_biases")


def test_runner_persists_latent_state_and_records_diagnostic(tmp_path):
    sampling = SamplingConfig()
    runtime = EpisodeEngine(LatentBackend(), initial_token_ids=[7], sampling=sampling)
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode_id = store.create_episode(
            episode_id="latent-live",
            initial_text=runtime.text,
            initial_token_ids=list(runtime.initial_token_ids),
            sampling=sampling,
            stream_fingerprint=runtime.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=None,
            backend={},
        )
        result = EpisodeRunner(
            runtime,
            store,
            episode_id,
            latent_learner=LatentPreferenceLearner(
                FEATURES,
                enabled=True,
                dimension=2,
                learning_rate=0.5,
            ),
        ).run(
            live_policy=type("Once", (), {"choose": lambda self, _e, _o: SelectRawRank(3)})(),
            max_live_actions=1,
        )

        assert result.outcomes[0].evidence[0].token_id == 3
        assert runtime.sampling.latent_preference_z
        saved = SamplingConfig.from_record(
            store.sampling_segment(episode_id, 1)["sampling"]
        )
        assert saved.latent_preference_z == runtime.sampling.latent_preference_z
        interaction = store.interactions(episode_id)[-1]
        assert interaction["kind"] == "latent-preference-update"
        assert interaction["boundary"] == 1
        assert interaction["payload"]["chosen_token_id"] == 3
        assert len(interaction["payload"]["new_z"]) == 2


def test_runner_combines_independent_group_and_latent_updates(tmp_path):
    from trajectory_editor.bias_rules import BiasGroup, BiasRule
    from trajectory_editor.online_learning import OnlineLearner

    group = BiasGroup("concrete", (BiasRule(routes=((3,),), bias=0.0),))
    sampling = SamplingConfig(bias_groups=(group,))
    runtime = EpisodeEngine(LatentBackend(), initial_token_ids=[7], sampling=sampling)
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode_id = store.create_episode(
            episode_id="both-live",
            initial_text=runtime.text,
            initial_token_ids=list(runtime.initial_token_ids),
            sampling=sampling,
            stream_fingerprint=runtime.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=None,
            backend={},
        )
        EpisodeRunner(
            runtime,
            store,
            episode_id,
            learner=OnlineLearner(enabled=True, learning_rate=0.5),
            latent_learner=LatentPreferenceLearner(
                FEATURES,
                enabled=True,
                dimension=2,
                learning_rate=0.5,
            ),
        ).run(
            live_policy=type("Once", (), {"choose": lambda self, _e, _o: SelectRawRank(3)})(),
            max_live_actions=1,
        )

        assert runtime.sampling.bias_groups[0].bias > 0.0
        assert runtime.sampling.latent_preference_z
        assert {item["kind"] for item in store.interactions(episode_id)} == {
            "online-learning-update",
            "latent-preference-update",
        }


def test_runner_does_not_learn_latent_state_during_replay(tmp_path):
    sampling = SamplingConfig(
        latent_preference_z=(0.5, 0.0),
        latent_strength=1.0,
    )
    runtime = EpisodeEngine(LatentBackend(), initial_token_ids=[7], sampling=sampling)
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode_id = store.create_episode(
            episode_id="latent-replay",
            initial_text=runtime.text,
            initial_token_ids=list(runtime.initial_token_ids),
            sampling=sampling,
            stream_fingerprint=runtime.stream_fingerprint,
            coordinate_offset=0,
            max_tokens=None,
            backend={},
        )
        EpisodeRunner(
            runtime,
            store,
            episode_id,
            latent_learner=LatentPreferenceLearner(
                FEATURES,
                enabled=True,
                dimension=2,
            ),
        ).run(
            tape=[TapeStep(SelectRawRank(3), ReplayExpectation((3,)))],
        )

        assert runtime.sampling == sampling
        assert store.interactions(episode_id) == []


def test_full_bias_export_includes_latent_state(tmp_path):
    from trajectory_editor.episode_lifecycle import _create_episode

    sampling = SamplingConfig(
        latent_preference_z=(0.5, -0.25),
        latent_strength=0.75,
    )
    runtime = EpisodeEngine(LatentBackend(), initial_token_ids=[7], sampling=sampling)
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

    assert restored.latent_preference_z == sampling.latent_preference_z
    assert restored.latent_strength == sampling.latent_strength


def test_latent_cli_controls_are_explicit_and_default_off():
    parser = build_parser()
    defaults = parser.parse_args([])
    assert defaults.latent_preference is False
    assert defaults.latent_projection_chunk_size == DEFAULT_PROJECTION_CHUNK_SIZE
    args = parser.parse_args([
        "--latent-preference",
        "--latent-dimension", "32",
        "--latent-learning-rate", "0.1",
        "--latent-strength", "0.7",
        "--latent-max-step", "0.03",
        "--latent-max-norm", "1.5",
        "--latent-projection-chunk-size", "1024",
    ])
    assert args.latent_preference is True
    assert args.latent_dimension == 32
    assert args.latent_learning_rate == 0.1
    assert args.latent_strength == 0.7
    assert args.latent_max_step == 0.03
    assert args.latent_max_norm == 1.5
    assert args.latent_projection_chunk_size == 1024

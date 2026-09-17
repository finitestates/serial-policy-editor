from __future__ import annotations

import json
import struct
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor.activation_vectors import (
    CONTROL_VECTOR_LAYER,
    CONTROL_VECTOR_POSITION,
    FORMAT,
    SteeringVectorArtifact,
    blend_artifacts,
)


def _write_cvector(path):
    def string(value):
        encoded = value.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    metadata = b"".join((
        string("general.architecture") + struct.pack("<I", 8) + string("controlvector"),
        string("controlvector.model_hint") + struct.pack("<I", 8) + string("llama"),
        string("controlvector.layer_count") + struct.pack("<I", 5) + struct.pack("<i", 2),
    ))
    tensors = []
    values = []
    for index, row in enumerate(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), 1):
        tensors.append(
            string(f"direction.{index}")
            + struct.pack("<I", 1)
            + struct.pack("<Q", 3)
            + struct.pack("<I", 0)
            + struct.pack("<Q", (index - 1) * 12)
        )
        values.append(struct.pack("<3f", *row))
    header = struct.pack("<4sIQQ", b"GGUF", 3, 2, 3) + metadata + b"".join(tensors)
    data_start = (len(header) + 31) // 32 * 32
    path.write_bytes(header + b"\0" * (data_start - len(header)) + b"".join(values))
from trajectory_editor.domain import EditorError, SamplingConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_lifecycle import _create_episode
from trajectory_editor.episode_policy import EpisodeRunner, TapeStep
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_actions import Hold
from trajectory_editor.vector_cli import main as vector_main


class ActivationBackend(ConformingFakeBackend):
    matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
            [-1.0, 0.5, 0.25],
            [0.5, -1.0, 0.5],
            [0.25, 0.5, -1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    def activation_width(self) -> int:
        return 3

    def activation_snapshot(self, text, *, layer="output", position="last"):
        del layer, position
        return {
            "A": np.asarray([1.0, 2.0, 4.0], dtype=np.float32),
            "B": np.asarray([0.0, 2.0, 2.0], dtype=np.float32),
        }[text]

    def activation_logit_adjustments(self, vector, *, layer="output", position="current"):
        assert layer == "output"
        assert position == "current"
        return self.matrix @ np.asarray(vector, dtype=np.float32)

    def provenance(self, *, include_model_sha256=True):
        del include_model_sha256
        return {
            "backend": "fake",
            "adapter": "activation-test",
            "vocabulary_size": self.vocabulary_size(),
        }


class ControlBackend(ActivationBackend):
    def __init__(self):
        self.control_calls = []
        self.clear_calls = 0
        super().__init__()

    def activation_control_vector_width(self) -> int:
        return 3

    def activation_control_vector_layer_count(self) -> int:
        return 2

    def set_activation_control_vector(self, vector, *, layer_start, layer_end, strength):
        self.control_calls.append((tuple(vector), layer_start, layer_end, strength))

    def clear_activation_control_vector(self):
        self.clear_calls += 1


def artifact(backend=None) -> SteeringVectorArtifact:
    backend = backend or ActivationBackend()
    return SteeringVectorArtifact.from_prompt_pair(
        backend,
        backend.provenance(),
        "A",
        "B",
    )


def test_prompt_pair_artifact_is_normalized_and_round_trips():
    value = artifact()

    assert value.model["hidden_state_width"] == 3
    assert value.vector == pytest.approx((1 / np.sqrt(5), 0.0, 2 / np.sqrt(5)))
    assert value.norm == pytest.approx(1.0)
    assert value.source["capture_position"] == "last"
    assert value.digest == value.to_dict()["digest"]
    assert SteeringVectorArtifact.from_mapping(json.loads(value.to_json())) == value
    document = value.to_dict()
    assert document["kind"] == "output-head-steering-vector"
    assert "layer" not in document


def test_ambiguous_legacy_artifacts_are_rejected_without_guessing():
    with pytest.raises(EditorError, match="ambiguous activation-vector artifact"):
        SteeringVectorArtifact.from_mapping({
            "format": "spe-activation-vector-v1",
            "kind": "activation",
        })


def test_activation_cli_create_inspect_and_validate(tmp_path, capsys):
    output = tmp_path / "activation.json"
    with patch("trajectory_editor.vector_cli.create_backend", return_value=ActivationBackend()):
        assert vector_main([
            "output-head", "create", "--model", "fake",
            "--backend", "llama.cpp", "--prompt-a", "A", "--prompt-b", "B",
            "--output", str(output),
        ]) == 0
        assert vector_main(["output-head", "inspect", str(output)]) == 0
        assert vector_main([
            "output-head", "validate", str(output), "--model", "fake",
        ]) == 0
    captured = capsys.readouterr().out
    assert "spe-steering-vector-v1" in captured
    assert "hidden_state_width=3" in captured


def test_activation_blend_requires_matching_coordinates():
    first = artifact()
    second = replace(first, layer="output")
    combined = blend_artifacts([first, second], [1.0, -0.5])

    assert combined.vector == pytest.approx(tuple(0.5 * value for value in first.vector))
    assert combined.method == "weighted-blend-v1"
    with pytest.raises(EditorError, match="incompatible activation models"):
        blend_artifacts([first, replace(first, model={"backend": "other"})], [1.0, 1.0])


def test_activation_adjustment_is_a_policy_surface_and_zero_strength_is_noop():
    backend = ActivationBackend()
    active = artifact(backend).apply_to_sampling(
        SamplingConfig(temperature=0.0, top_k=8, top_p=1.0, min_p=0.0)
    )
    runtime = EpisodeEngine(backend, initial_token_ids=[7], sampling=active)
    observation = runtime.observe()
    expected = backend.activation_logit_adjustments(active.activation_vector)

    assert observation.statistics.logits == pytest.approx(backend.last_logits())
    assert observation.statistics.activation_logit_adjustments == pytest.approx(expected)
    assert observation.statistics.adjusted == pytest.approx(
        backend.last_logits() + expected
    )
    assert observation.statistics.activation_diagnostics["digest"] == active.activation_vector_digest

    neutral = replace(active, activation_vector_strength=0.0)
    neutral_runtime = EpisodeEngine(
        ActivationBackend(), initial_token_ids=[7], sampling=neutral
    )
    neutral_observation = neutral_runtime.observe()
    assert neutral_observation.statistics.activation_logit_adjustments == pytest.approx(
        np.zeros(8)
    )
    assert neutral_observation.statistics.adjusted == pytest.approx(
        neutral_observation.statistics.logits
    )


def test_activation_state_persists_through_replay(tmp_path):
    initial = artifact().apply_to_sampling(
        SamplingConfig(temperature=0.0, top_k=8, top_p=1.0, min_p=0.0)
    )
    path = tmp_path / "episodes.sqlite3"
    with EpisodeStore(path) as store:
        source_runtime = EpisodeEngine(
            ActivationBackend(), initial_token_ids=[7], sampling=initial
        )
        source_id = _create_episode(
            store,
            source_runtime,
            backend_provenance=source_runtime.backend.provenance(),
        )
        outcome = source_runtime.apply(Hold(1))
        store.record_action(source_id, 0, outcome)
        saved = SamplingConfig.from_record(store.sampling_segment(source_id, 0)["sampling"])
        assert saved == initial

        replay_runtime = EpisodeEngine(
            ActivationBackend(), initial_token_ids=[7], sampling=saved
        )
        replay_id = _create_episode(
            store,
            replay_runtime,
            backend_provenance=replay_runtime.backend.provenance(),
        )
        result = EpisodeRunner(replay_runtime, store, replay_id).run(
            tape=[TapeStep(outcome.action, outcome.expectation())]
        )

        assert result.replayed_actions == 1
        assert replay_runtime.sampling.activation_vector_digest == initial.activation_vector_digest
        assert replay_runtime.visible_token_ids == source_runtime.visible_token_ids


def test_cvector_gguf_import_preserves_layerwise_directions(tmp_path, capsys):
    source = tmp_path / "control_vector.gguf"
    output = tmp_path / "control_vector.json"
    _write_cvector(source)

    assert vector_main([
        "hidden-state", "import-cvector", str(source), "--output", str(output)
    ]) == 0
    loaded = SteeringVectorArtifact.from_path(output)
    assert loaded.layer == CONTROL_VECTOR_LAYER
    assert loaded.position == CONTROL_VECTOR_POSITION
    assert loaded.layer_start == 2
    assert loaded.layer_end == 2
    assert loaded.vector == pytest.approx((1, 2, 3, 4, 5, 6))
    assert loaded.model["hidden_state_layer_count"] == 2
    assert vector_main(["hidden-state", "inspect", str(source)]) == 0
    assert "target: hidden-state layers range=2..2" in capsys.readouterr().out


def test_layerwise_cvector_is_installed_before_runtime_logits():
    backend = ControlBackend()
    cvector = SteeringVectorArtifact(
        model={"backend": "fake", "activation_width": 3, "activation_layer_count": 2},
        vector=(1, 2, 3, 4, 5, 6),
        layer=CONTROL_VECTOR_LAYER,
        position=CONTROL_VECTOR_POSITION,
        layer_start=1,
        layer_end=2,
    )
    sampling = cvector.apply_to_sampling(
        SamplingConfig(temperature=0.0, top_k=8, top_p=1.0, min_p=0.0)
    )

    runtime = EpisodeEngine(backend, initial_token_ids=[7], sampling=sampling)
    runtime.observe()

    assert backend.control_calls == [((1, 2, 3, 4, 5, 6), 1, 2, 1.0)]

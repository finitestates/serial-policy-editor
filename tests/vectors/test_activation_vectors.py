from __future__ import annotations

import struct

import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor.activation_vectors import (
    CONTROL_VECTOR_LAYER,
    CONTROL_VECTOR_POSITION,
    SteeringVectorArtifact,
    assert_model_compatible,
)
from trajectory_editor.model_hash import sha256_path

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
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.steering_vector_cli import main as vector_main

pytestmark = pytest.mark.optional

class ControlBackend(ConformingFakeBackend):
    def __init__(self):
        self.control_calls = []
        super().__init__()

    def activation_width(self) -> int:
        return 3

    def activation_control_vector_width(self) -> int:
        return 3

    def activation_control_vector_layer_count(self) -> int:
        return 2

    def set_activation_control_vector(self, vector, *, layer_start, layer_end, strength):
        self.control_calls.append((tuple(vector), layer_start, layer_end, strength))


def test_model_identity_requires_matching_hash_for_strong_artifacts(tmp_path):
    model = tmp_path / "same-name.gguf"
    model.write_bytes(b"first contents")
    expected = {
        "backend": "llama.cpp",
        "filename": model.name,
        "file_size_bytes": model.stat().st_size,
        "hidden_state_width": 3,
        "model_sha256": sha256_path(model),
    }
    assert_model_compatible(expected, dict(expected), label="loaded model")

    model.write_bytes(b"other contents")
    actual = dict(expected)
    actual["file_size_bytes"] = model.stat().st_size
    actual["model_sha256"] = sha256_path(model)
    with pytest.raises(EditorError, match="model_sha256"):
        assert_model_compatible(expected, actual, label="loaded model")


def test_legacy_metadata_only_identity_is_explicit_and_remains_readable():
    value = SteeringVectorArtifact(
        model={"backend": "fake", "filename": "model.gguf", "hidden_state_width": 3},
        vector=(1.0, 0.0, 0.0),
    )
    document = value.to_dict()
    assert document["schema_version"] == 2
    assert document["compatibility"] == {
        "model_identity": "metadata-only",
        "hash_algorithm": None,
    }

    legacy = dict(document)
    legacy.pop("schema_version")
    legacy.pop("compatibility")
    assert SteeringVectorArtifact.from_mapping(legacy).model == value.model


def test_model_identity_reports_architecture_and_width_mismatches():
    expected = {
        "backend": "llama.cpp",
        "model_type": "llama",
        "hidden_state_width": 3,
        "hidden_state_layer_count": 4,
    }
    actual = {**expected, "model_type": "qwen", "hidden_state_width": 4}
    with pytest.raises(EditorError, match="model_type"):
        assert_model_compatible(expected, actual, label="loaded model")


def test_ambiguous_legacy_artifacts_are_rejected_without_guessing():
    with pytest.raises(EditorError, match="ambiguous activation-vector artifact"):
        SteeringVectorArtifact.from_mapping({
            "format": "spe-activation-vector-v1",
            "kind": "activation",
        })


def test_cvector_gguf_import_preserves_layerwise_directions(tmp_path, capsys):
    source = tmp_path / "control_vector.gguf"
    output = tmp_path / "control_vector.json"
    _write_cvector(source)

    assert vector_main(["import-cvector", str(source), "--output", str(output)]) == 0
    loaded = SteeringVectorArtifact.from_path(output)
    assert loaded.layer == CONTROL_VECTOR_LAYER
    assert loaded.position == CONTROL_VECTOR_POSITION
    assert loaded.layer_start == 2
    assert loaded.layer_end == 3
    assert loaded.vector == pytest.approx((0, 0, 0, 1, 2, 3, 4, 5, 6))
    assert loaded.model["hidden_state_layer_count"] == 3
    assert vector_main(["inspect", str(source)]) == 0
    assert "layers: 2..3" in capsys.readouterr().out


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
        SamplerConfig(temperature=0.0, top_k=8, top_p=1.0, min_p=0.0)
    )

    runtime = EpisodeEngine(backend, initial_token_ids=[7], sampling=sampling)
    runtime.observe()

    assert backend.control_calls == [((1, 2, 3, 4, 5, 6), 1, 2, 1.0)]

from __future__ import annotations

import struct

import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor.activation_vectors import (
    CONTROL_VECTOR_LAYER,
    CONTROL_VECTOR_POSITION,
    SteeringVectorArtifact,
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


def test_current_artifact_drops_model_compatibility_metadata():
    value = SteeringVectorArtifact(vector=(1.0, 0.0, 0.0))
    document = value.to_dict()

    assert document["schema_version"] == 3
    assert "model" not in document
    assert "compatibility" not in document
    assert SteeringVectorArtifact.from_mapping(document).vector == value.vector

    with pytest.raises(EditorError, match="schema version must be 3"):
        SteeringVectorArtifact.from_mapping({**document, "schema_version": 2})


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
    assert vector_main(["inspect", str(source)]) == 0
    assert "layers: 2..3" in capsys.readouterr().out


def test_layerwise_cvector_is_installed_before_runtime_logits():
    backend = ControlBackend()
    cvector = SteeringVectorArtifact(
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

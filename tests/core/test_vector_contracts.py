from __future__ import annotations

import json
import struct
from dataclasses import asdict

import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor.activation_vectors import (
    CONTROL_VECTOR_LAYER,
    CONTROL_VECTOR_POSITION,
    SteeringVectorArtifact,
)
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.decoder import LlamaCppSettings
from trajectory_editor.transformers_backend import (
    TransformersSettings,
    _supports_logits_to_keep,
)

pytestmark = pytest.mark.current_workflow

def _artifact(**overrides):
    value = {
        "format": "spe-steering-vector-v1",
        "schema_version": 3,
        "kind": "output-head-steering-vector",
        "strength": 1.0,
        "method": "external-producer",
        "vector": [0.25, -0.5, 1.0],
    }
    value.update(overrides)
    return value


def _write_cvector(path):
    def string(value):
        encoded = value.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    metadata = b"".join((
        string("general.architecture") + struct.pack("<I", 8) + string("controlvector"),
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
    header = struct.pack("<4sIQQ", b"GGUF", 3, 2, 2) + metadata + b"".join(tensors)
    data_start = (len(header) + 31) // 32 * 32
    path.write_bytes(header + b"\0" * (data_start - len(header)) + b"".join(values))


class ControlVectorBackend(ConformingFakeBackend):
    def __init__(self):
        super().__init__()
        self.control_calls = []

    def set_activation_control_vector(self, vector, *, layer_start, layer_end, strength):
        self.control_calls.append((tuple(vector), layer_start, layer_end, strength))


class OutputVectorBackend:
    def activation_width(self):
        return 3

    def vocabulary_size(self):
        return 4

    def activation_logit_adjustments(self, vector, *, layer, position):
        assert layer == "output"
        assert position == "current"
        return np.asarray([sum(vector), 0.0, -sum(vector), 0.0])


def test_v01_core_loads_an_external_json_vector_without_model_metadata(tmp_path):
    path = tmp_path / "external.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")

    artifact = SteeringVectorArtifact.from_path(path)

    assert artifact.vector == (0.25, -0.5, 1.0)
    assert artifact.method == "external-producer"


def test_v02_core_maps_cvector_layers_to_the_canonical_runtime_order(tmp_path):
    path = tmp_path / "external.gguf"
    _write_cvector(path)

    artifact = SteeringVectorArtifact.from_cvector_path(path)
    backend = ControlVectorBackend()
    artifact.validate_against_backend(backend)

    assert artifact.layer == CONTROL_VECTOR_LAYER
    assert artifact.position == CONTROL_VECTOR_POSITION
    assert artifact.layer_start == 2
    assert artifact.layer_end == 3
    assert artifact.vector == pytest.approx((0, 0, 0, 1, 2, 3, 4, 5, 6))


def test_v03_malformed_or_dimensionally_unusable_vectors_fail_clearly():
    for value, message in (
        ({"format": "wrong"}, "format"),
        (_artifact(vector=[]), "must not be empty"),
        (_artifact(schema_version=2), "schema version must be 3"),
    ):
        with pytest.raises(EditorError, match=message):
            SteeringVectorArtifact.from_mapping(value)

    artifact = SteeringVectorArtifact.from_mapping(_artifact(vector=[0.25, 0.5]))
    with pytest.raises(EditorError, match="dimension"):
        artifact.validate_against_backend(OutputVectorBackend())


def test_v04_model_labels_do_not_gate_vector_application():
    artifact = SteeringVectorArtifact.from_mapping(
        _artifact(model={"backend": "different-model", "model_sha256": "unrelated"})
    )

    adjustment = artifact.validate_against_backend(OutputVectorBackend())

    assert adjustment.shape == (4,)
    assert "model" not in artifact.to_dict()

    hidden = SteeringVectorArtifact.from_mapping(
        _artifact(
            kind="hidden-state-vector",
            target={"site": "another-site", "coordinate": "another-coordinate"},
            position="layers",
            layer_start=1,
            layer_end=1,
        )
    )
    hidden.validate_against_backend(ControlVectorBackend())


def test_v05_output_vector_uses_backend_math_without_model_label_checks():
    artifact = SteeringVectorArtifact.from_mapping(_artifact())

    adjustment = artifact.validate_against_backend(OutputVectorBackend())
    sampling = artifact.apply_to_sampling(SamplerConfig())

    assert adjustment.shape == (4,)
    assert sampling.activation_vector == artifact.vector
    assert sampling.activation_vector_layer == "output"
    assert "seed" not in asdict(LlamaCppSettings())

    assert TransformersSettings(device="cpu", dtype="float32").device == "cpu"
    with pytest.raises(EditorError):
        TransformersSettings(dtype="float128")

    class SupportsFinalLogits:
        def forward(self, input_ids=None, logits_to_keep=0):
            del input_ids, logits_to_keep

    class UsesFullOutput:
        def forward(self, input_ids=None):
            del input_ids

    assert _supports_logits_to_keep(SupportsFinalLogits())
    assert not _supports_logits_to_keep(UsesFullOutput())

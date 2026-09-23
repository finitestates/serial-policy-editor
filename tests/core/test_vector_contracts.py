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
        "schema_version": 2,
        "kind": "output-head-steering-vector",
        "model": {},
        "strength": 1.0,
        "method": "external-producer",
        "vector": [0.25, -0.5, 1.0],
        "compatibility": {"model_identity": "metadata-only", "hash_algorithm": None},
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

    def activation_width(self) -> int:
        return 3

    def activation_control_vector_width(self) -> int:
        return 3

    def activation_control_vector_layer_count(self) -> int:
        return 3

    def provenance(self, *, include_model_sha256=True):
        del include_model_sha256
        return {
            "backend": "llama.cpp",
            "model_type": "llama",
            "hidden_state_width": 3,
            "hidden_state_layer_count": 3,
        }

    def set_activation_control_vector(self, vector, *, layer_start, layer_end, strength):
        self.control_calls.append((tuple(vector), layer_start, layer_end, strength))


class OutputVectorBackend:
    def __init__(self, backend_name):
        self.backend_name = backend_name

    def activation_width(self):
        return 3

    def vocabulary_size(self):
        return 4

    def provenance(self, *, include_model_sha256=True):
        del include_model_sha256
        return {"backend": self.backend_name, "hidden_state_width": 3}

    def activation_logit_adjustments(self, vector, *, layer, position):
        assert layer == "output"
        assert position == "current"
        return np.asarray([sum(vector), 0.0, -sum(vector), 0.0])


def test_v01_core_loads_an_external_json_vector_without_provenance(tmp_path):
    path = tmp_path / "external.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")

    artifact = SteeringVectorArtifact.from_path(path)

    assert artifact.model == {}
    assert artifact.vector == (0.25, -0.5, 1.0)
    assert artifact.method == "external-producer"


def test_v02_core_maps_cvector_layers_to_the_canonical_runtime_order(tmp_path):
    path = tmp_path / "external.gguf"
    _write_cvector(path)

    artifact = SteeringVectorArtifact.from_cvector_path(path)
    backend = ControlVectorBackend()
    artifact.validate_against_backend(backend, backend.provenance())

    assert artifact.layer == CONTROL_VECTOR_LAYER
    assert artifact.position == CONTROL_VECTOR_POSITION
    assert artifact.layer_start == 2
    assert artifact.layer_end == 3
    assert artifact.vector == pytest.approx((0, 0, 0, 1, 2, 3, 4, 5, 6))


def test_v03_malformed_or_dimensionally_unusable_vectors_fail_clearly():
    for value, message in (
        ({"format": "wrong"}, "format"),
        (_artifact(vector=[]), "must not be empty"),
        (_artifact(model=[]), "model metadata"),
        (_artifact(compatibility={}), "compatibility metadata"),
    ):
        with pytest.raises(EditorError, match=message):
            SteeringVectorArtifact.from_mapping(value)

    artifact = SteeringVectorArtifact.from_mapping(_artifact(vector=[0.25, 0.5]))
    with pytest.raises(EditorError, match="dimension"):
        artifact.validate_against_backend(
            OutputVectorBackend("llama.cpp"),
            {"backend": "llama.cpp", "hidden_state_width": 3},
        )


def test_v04_core_preserves_supplied_vector_metadata_without_inventing_provenance():
    supplied = {"backend": "independent-tool", "filename": "outside.gguf"}
    artifact = SteeringVectorArtifact.from_mapping(_artifact(model=supplied))

    assert artifact.model == supplied
    assert artifact.to_dict()["model"] == supplied
    assert "model_sha256" not in artifact.model


@pytest.mark.parametrize("backend_name", ["llama.cpp", "transformers"])
def test_v05_backends_share_the_common_output_vector_capability(backend_name):
    artifact = SteeringVectorArtifact.from_mapping(
        _artifact(model={"backend": backend_name, "hidden_state_width": 3})
    )

    adjustment = artifact.validate_against_backend(
        OutputVectorBackend(backend_name),
        {"backend": backend_name, "hidden_state_width": 3},
    )
    sampling = artifact.apply_to_sampling(SamplerConfig())

    assert adjustment.shape == (4,)
    assert sampling.activation_vector == artifact.vector
    assert sampling.activation_vector_layer == "output"

    if backend_name == "llama.cpp":
        assert "seed" not in asdict(LlamaCppSettings())
    else:
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

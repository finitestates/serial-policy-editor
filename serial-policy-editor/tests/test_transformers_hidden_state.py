from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from unittest.mock import patch

from trajectory_editor.activation_vectors import (
    HIDDEN_STATE_KIND,
    SteeringVectorArtifact,
)
from trajectory_editor.transformers_backend import (
    TransformersBackend,
    _is_multimodal_config,
    _text_config,
)
from trajectory_editor.vector_cli import main as vector_main


class TinyTokenizer:
    bos_token_id = 1
    all_special_ids = []
    model_max_length = 32

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [2 + (ord(char) % 4) for char in text]


class TinyBlock(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.full((4,), float(offset)))

    def forward(self, hidden_states, **kwargs):
        del kwargs
        return hidden_states + self.offset


class TinyBackbone(torch.nn.Module):
    def __init__(self, layer_count=3):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.layers = torch.nn.ModuleList(
            [TinyBlock(index + 1) for index in range(layer_count)]
        )
        self.norm = torch.nn.Identity()


class TinyModel(torch.nn.Module):
    def __init__(self, layer_count=3):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="tiny",
            num_hidden_layers=layer_count,
            hidden_size=4,
            vocab_size=8,
            max_position_embeddings=32,
        )
        self.model = TinyBackbone(layer_count)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, **kwargs):
        del kwargs
        hidden_states = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return SimpleNamespace(logits=self.lm_head(hidden_states))


def backend(model=None):
    model = model or TinyModel()
    result = object.__new__(TransformersBackend)
    result._torch = torch
    result._model = model
    result._text_config = _text_config(model.config)
    result._tokenizer = TinyTokenizer()
    result._input_device = torch.device("cpu")
    result._device = torch.device("cpu")
    result._vocabulary_size = 8
    result._context_limit = 32
    result._tokens = []
    result._last_logits = None
    result._hidden_state_control_handles = []
    result._hidden_state_control_key = None
    return result


def test_hidden_state_capabilities_describe_the_decoder_boundary():
    value = backend()

    assert value.hidden_state_width() == 4
    assert value.hidden_state_layer_count() == 3
    assert value.hidden_state_capabilities() == {
        "site": "decoder-block-output-residual",
        "layer_numbering": "one-based",
        "layer_count": 3,
        "width": 4,
        "position_policies": ["first", "last", "current", "all"],
        "layer_types": ["decoder", "decoder", "decoder"],
        "native_module_path": "model.layers[N-1]",
        "modality": "text",
    }


def test_hidden_state_snapshot_captures_arbitrary_layer_and_positions():
    value = backend()

    all_positions = value.hidden_state_snapshot("abc", layer=2, position="all")
    last = value.hidden_state_snapshot("abc", layer=2, position="last")
    first = value.hidden_state_snapshot("abc", layer=2, position="first")

    assert all_positions.shape == (4, 4)
    np.testing.assert_allclose(last, all_positions[-1])
    np.testing.assert_allclose(first, all_positions[0])
    with pytest.raises(RuntimeError, match="between 1 and 3"):
        value.hidden_state_snapshot("abc", layer=0)


def test_hidden_state_vector_hooks_only_selected_layers_and_can_be_cleared():
    value = backend()
    baseline = value.hidden_state_snapshot("abc", layer=2, position="last")
    direction = np.zeros(12, dtype=np.float32)
    direction[4:8] = [1.0, -2.0, 3.0, 4.0]

    value.set_hidden_state_vector(
        direction, layer_start=2, layer_end=2, strength=0.5
    )
    adjusted = value.hidden_state_snapshot("abc", layer=2, position="last")
    np.testing.assert_allclose(adjusted, baseline + direction[4:8] * 0.5)
    assert len(value._hidden_state_control_handles) == 1

    value.clear_hidden_state_vector()
    restored = value.hidden_state_snapshot("abc", layer=2, position="last")
    np.testing.assert_allclose(restored, baseline)
    assert value._hidden_state_control_handles == []


def test_hidden_state_vector_rejects_wrong_width_without_installing_hooks():
    value = backend()

    with pytest.raises(RuntimeError, match="one direction for every model layer"):
        value.set_hidden_state_vector(
            np.zeros(4, dtype=np.float32),
            layer_start=1,
            layer_end=1,
            strength=1.0,
        )
    assert value._hidden_state_control_handles == []


def test_hidden_state_prompt_pair_emits_full_layer_aligned_artifact():
    value = backend()
    value.provenance = lambda include_model_sha256=True: {
        "backend": "fake-transformers",
        "adapter": "hidden-state-test",
        "vocabulary_size": 8,
        "hidden_state_layer_count": 3,
    }

    artifact = SteeringVectorArtifact.from_hidden_state_prompt_pair(
        value,
        value.provenance(),
        "abc",
        "abd",
        layer_start=2,
        layer_end=3,
    )

    assert artifact.kind == HIDDEN_STATE_KIND
    assert artifact.layer_start == 2
    assert artifact.layer_end == 3
    assert len(artifact.vector) == 12
    np.testing.assert_allclose(artifact.vector[:4], np.zeros(4))
    assert np.linalg.norm(artifact.vector[4:8]) == pytest.approx(1.0)
    assert np.linalg.norm(artifact.vector[8:12]) == pytest.approx(1.0)
    round_trip = SteeringVectorArtifact.from_mapping(artifact.to_dict())
    assert round_trip == artifact


def test_hidden_state_cli_creates_a_selected_layer_artifact(tmp_path):
    value = backend()
    value.provenance = lambda include_model_sha256=True: {
        "backend": "fake-transformers",
        "adapter": "hidden-state-test",
        "vocabulary_size": 8,
        "hidden_state_layer_count": 3,
    }
    output = tmp_path / "hidden-state.json"

    with patch("trajectory_editor.vector_cli._load_backend", return_value=value):
        assert vector_main(
            [
                "hidden-state",
                "create",
                "--model",
                "fake.model",
                "--backend",
                "transformers",
                "--prompt-a",
                "abc",
                "--prompt-b",
                "abd",
                "--layer",
                "2",
                "--output",
                str(output),
            ]
        ) == 0

    loaded = SteeringVectorArtifact.from_path(output)
    assert loaded.layer_start == 2
    assert loaded.layer_end == 2
    assert loaded.model["hidden_state_layer_count"] == 3


def test_nested_text_config_and_multimodal_detection():
    text = SimpleNamespace(
        model_type="qwen3_5_text",
        vocab_size=248320,
        hidden_size=1024,
        num_hidden_layers=24,
    )
    config = SimpleNamespace(
        model_type="qwen3_5",
        text_config=text,
        vision_config=SimpleNamespace(model_type="qwen3_5_vision"),
    )

    assert _text_config(config) is text
    assert _is_multimodal_config(config)

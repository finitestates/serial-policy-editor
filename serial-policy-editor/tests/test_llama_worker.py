from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from trajectory_editor.activation_vectors import HIDDEN_STATE_KIND, SteeringVectorArtifact
from trajectory_editor.domain import EditorError
from trajectory_editor.vector_cli import main as vector_main
from trajectory_editor.llama_worker import capture_hidden_state_pair


def worker_response(filename: str = "model.gguf") -> dict:
    return {
        "protocol": "spe-llama-worker-v2",
        "operation": "hidden-state-pair",
        "backend": {
            "name": "llama.cpp",
            "version": "0.3.0-dev",
            "capture_api": "layer-input-c-api-plus-graph-output",
        },
        "model": {
            "filename": filename,
            "model_type": "llama",
            "vocabulary_size": 32,
            "hidden_state_width": 3,
            "hidden_state_layer_count": 4,
        },
        "target": {
            "site": "decoder-block-output-residual",
            "layer_numbering": "one-based",
            "coordinate": "canonical-decoder-block-output-v1",
            "layer_start": 2,
            "layer_end": 3,
            "position": "last",
        },
        "prompts": {"a_token_count": 3, "b_token_count": 4},
        "normalized": True,
        "directions": {
            "2": [1.0, 0.0, 0.0],
            "3": [0.0, 1.0, 0.0],
        },
        "raw_delta_norms": {"2": 2.0, "3": 3.0},
    }


def test_worker_response_becomes_a_runtime_compatible_artifact(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    artifact = SteeringVectorArtifact.from_llama_worker_response(
        worker_response(),
        model_path=model,
        worker_path=tmp_path / "spe-llama-worker",
        prompt_a="calm",
        prompt_b="angry",
        layer_start=2,
        layer_end=3,
    )

    assert artifact.kind == HIDDEN_STATE_KIND
    assert artifact.method == "hidden-state-prompt-pair-llama-worker-v1"
    assert artifact.model == {
        "backend": "llama.cpp",
        "filename": "model.gguf",
        "file_size_bytes": 5,
        "vocabulary_size": 32,
        "model_type": "llama",
        "hidden_state_width": 3,
        "hidden_state_layer_count": 4,
    }
    np.testing.assert_allclose(
        artifact.vector,
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    )
    assert artifact.source["prompt_token_counts"] == {
        "a_token_count": 3,
        "b_token_count": 4,
    }
    assert SteeringVectorArtifact.from_mapping(json.loads(artifact.to_json())) == artifact


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda value: value["directions"].update({"4": [0.0, 0.0, 1.0]}), "incomplete layer range"),
        (lambda value: value["directions"].update({"2": [float("nan"), 0.0, 0.0]}), "non-finite"),
        (lambda value: value["target"].update({"position": "first"}), "position does not match"),
    ],
)
def test_worker_response_validation_rejects_mismatches(tmp_path, change, message):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    response = worker_response()
    change(response)

    with pytest.raises(EditorError, match=message):
        SteeringVectorArtifact.from_llama_worker_response(
            response,
            model_path=model,
            worker_path=tmp_path / "worker",
            prompt_a="a",
            prompt_b="b",
            layer_start=2,
            layer_end=3,
        )


def test_worker_layer_one_capture_cannot_become_a_loadable_cvector(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    response = worker_response()
    response["target"]["layer_start"] = 1
    response["directions"]["1"] = [0.0, 0.0, 0.0]
    response["raw_delta_norms"]["1"] = 0.0

    with pytest.raises(EditorError, match="layer 1 is capture-only"):
        SteeringVectorArtifact.from_llama_worker_response(
            response,
            model_path=model,
            worker_path=tmp_path / "worker",
            prompt_a="a",
            prompt_b="b",
            layer_start=1,
            layer_end=3,
        )


def test_worker_cli_path_emits_the_normal_vector_artifact(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    output = tmp_path / "vector.json"
    with patch(
        "trajectory_editor.vector_cli.capture_hidden_state_pair",
        return_value=worker_response(),
    ) as capture:
        assert vector_main(
            [
                "hidden-state",
                "create",
                "--model",
                str(model),
                "--backend",
                "llama.cpp",
                "--worker",
                str(tmp_path / "worker"),
                "--prompt-a",
                "calm",
                "--prompt-b",
                "angry",
                "--layer-range",
                "2",
                "3",
                "--output",
                str(output),
            ]
        ) == 0

    artifact = SteeringVectorArtifact.from_path(output)
    assert artifact.model["filename"] == "model.gguf"
    capture.assert_called_once()
    assert capture.call_args.kwargs["normalize"] is True


def test_worker_process_reports_last_diagnostic_line():
    completed = type(
        "Completed",
        (),
        {"returncode": 2, "stdout": "", "stderr": "load failed\nfinal detail\n"},
    )()
    with patch("trajectory_editor.llama_worker.subprocess.run", return_value=completed):
        with pytest.raises(EditorError, match="final detail"):
            capture_hidden_state_pair(
                Path("worker"),
                Path("model.gguf"),
                "a",
                "b",
                layer_start=1,
                layer_end=1,
                position="last",
                normalize=True,
                n_ctx=128,
            )

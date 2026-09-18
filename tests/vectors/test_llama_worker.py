from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from trajectory_editor.activation_vectors import HIDDEN_STATE_KIND, SteeringVectorArtifact
from trajectory_editor.core.errors import EditorError
from trajectory_editor.llama_worker import (
    MAX_PROMPT_ARGUMENT_BYTES,
    WorkerModelError,
    WorkerProtocolError,
    WorkerStartupError,
    WorkerTimeoutError,
    capture_hidden_state_pair,
)


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
        "model_sha256": hashlib.sha256(b"model").hexdigest(),
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


def test_worker_process_validates_protocol_and_metadata():
    response = worker_response()
    completed = type(
        "Completed",
        (),
        {"returncode": 0, "stdout": json.dumps(response), "stderr": ""},
    )()
    with patch("trajectory_editor.llama_worker.subprocess.run", return_value=completed) as run:
        actual = capture_hidden_state_pair(
            Path("worker"),
            Path("model.gguf"),
            "a",
            "b",
            layer_start=2,
            layer_end=3,
            position="last",
            normalize=True,
            n_ctx=128,
            timeout_seconds=3,
        )

    assert actual == response
    assert run.call_args.kwargs["timeout"] == 3.0


def test_worker_process_distinguishes_timeout_and_malformed_output():
    with patch(
        "trajectory_editor.llama_worker.subprocess.run",
        side_effect=FileNotFoundError,
    ):
        with pytest.raises(WorkerStartupError, match="could not start"):
            capture_hidden_state_pair(
                Path("worker"), Path("model.gguf"), "a", "b",
                layer_start=2, layer_end=3, position="last", normalize=True, n_ctx=128,
            )

    with patch(
        "trajectory_editor.llama_worker.subprocess.run",
        side_effect=subprocess.TimeoutExpired(["worker"], 1),
    ):
        with pytest.raises(WorkerTimeoutError, match="timed out"):
            capture_hidden_state_pair(
                Path("worker"), Path("model.gguf"), "a", "b",
                layer_start=2, layer_end=3, position="last", normalize=True, n_ctx=128,
            )

    completed = type(
        "Completed",
        (),
        {"returncode": 0, "stdout": "not json", "stderr": ""},
    )()
    with patch("trajectory_editor.llama_worker.subprocess.run", return_value=completed):
        with pytest.raises(WorkerProtocolError, match="malformed JSON"):
            capture_hidden_state_pair(
                Path("worker"), Path("model.gguf"), "a", "b",
                layer_start=2, layer_end=3, position="last", normalize=True, n_ctx=128,
            )


def test_worker_process_rejects_oversized_prompt_and_model_failure():
    with pytest.raises(WorkerProtocolError, match="too large"):
        capture_hidden_state_pair(
            Path("worker"), Path("model.gguf"), "x" * (MAX_PROMPT_ARGUMENT_BYTES + 1), "b",
            layer_start=2, layer_end=3, position="last", normalize=True, n_ctx=128,
        )

    completed = type(
        "Completed",
        (),
        {"returncode": 2, "stdout": "", "stderr": "load failed\nmodel detail\n"},
    )()
    with patch("trajectory_editor.llama_worker.subprocess.run", return_value=completed):
        with pytest.raises(WorkerModelError, match="model detail"):
            capture_hidden_state_pair(
                Path("worker"), Path("model.gguf"), "a", "b",
                layer_start=2, layer_end=3, position="last", normalize=True, n_ctx=128,
            )

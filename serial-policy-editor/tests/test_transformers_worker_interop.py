"""Opt-in cross-backend hidden-state sanity check.

Set ``SPE_TRANSFORMERS_GGUF_MODEL`` to a local GGUF and build the SPE worker
before running this test.  Transformers de-quantizes the same file for an
independent reference; this is intentionally a slow conformance check, not a
normal runtime dependency of vector creation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from trajectory_editor.activation_vectors import SteeringVectorArtifact
from trajectory_editor.decoder import LlamaCppDecoder, LlamaCppSettings
from trajectory_editor.llama_worker import capture_hidden_state_pair


pytestmark = pytest.mark.transformers_gguf_smoke

PROMPT_A = "A calm cat watches the rain today"
PROMPT_B = "An angry dog chases the storm tonight"


@pytest.fixture(scope="module")
def gguf_model() -> Path:
    value = os.environ.get("SPE_TRANSFORMERS_GGUF_MODEL")
    if not value:
        pytest.skip(
            "Set SPE_TRANSFORMERS_GGUF_MODEL to a local GGUF to run the "
            "Transformers/worker interop check"
        )
    path = Path(value).resolve()
    if not path.is_file() or path.suffix.lower() != ".gguf":
        pytest.skip(f"configured GGUF model does not exist: {path}")
    return path


@pytest.fixture(scope="module")
def worker_path() -> Path:
    value = os.environ.get("SPE_LLAMA_WORKER")
    path = Path(value).resolve() if value else Path(__file__).parents[1] / "build" / "spe-llama-worker"
    if not path.is_file():
        pytest.skip(f"SPE llama worker is not built: {path}")
    return path


def _token_ids(tokenizer, native, text: str) -> tuple[list[int], list[int]]:
    hf = [int(tokenizer.bos_token_id)] + [
        int(value) for value in tokenizer.encode(text, add_special_tokens=False)
    ]
    llama = [
        int(value)
        for value in native.tokenize(
            text.encode("utf-8"), add_bos=True, special=False
        )
    ]
    return hf, llama


def _hidden_tensor(output):
    if hasattr(output, "detach") and hasattr(output, "shape"):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if hasattr(first, "detach") and hasattr(first, "shape"):
            return first
    raise AssertionError("Transformers decoder block returned no hidden-state tensor")


def _capture_reference_representations(model, input_ids):
    """Capture the same named sites without using hidden_states conventions."""

    torch = pytest.importorskip("torch")
    block_outputs = []
    output_head_inputs = []

    def capture_block(_module, _inputs, output):
        block_outputs.append(_hidden_tensor(output).detach().cpu())

    def capture_output_head_input(_module, inputs):
        assert inputs
        output_head_inputs.append(inputs[0].detach().cpu())

    block_handles = [
        layer.register_forward_hook(capture_block)
        for layer in model.model.layers
    ]
    head_handle = model.get_output_embeddings().register_forward_pre_hook(
        capture_output_head_input
    )
    try:
        with torch.inference_mode():
            embedding_output = model.model.embed_tokens(input_ids).detach().cpu()
            model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in block_handles:
            handle.remove()
        head_handle.remove()

    assert len(block_outputs) == len(model.model.layers)
    assert len(output_head_inputs) == 1
    return {
        "embedding": embedding_output,
        "blocks": tuple(block_outputs),
        "post_normalization": output_head_inputs[0],
    }


@pytest.fixture(scope="module")
def reference(gguf_model: Path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    llama_cpp = pytest.importorskip("llama_cpp")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = gguf_model.parent
    tokenizer = AutoTokenizer.from_pretrained(
        str(root),
        gguf_file=gguf_model.name,
        local_files_only=True,
    )
    native = llama_cpp.Llama(
        model_path=str(gguf_model), vocab_only=True, verbose=False
    )
    token_pairs = {
        text: _token_ids(tokenizer, native, text)
        for text in (PROMPT_A, PROMPT_B)
    }
    for text, (hf_ids, llama_ids) in token_pairs.items():
        assert hf_ids == llama_ids, f"tokenizer mismatch for {text!r}"

    model = AutoModelForCausalLM.from_pretrained(
        str(root),
        gguf_file=gguf_model.name,
        local_files_only=True,
        dtype=torch.float32,
    )
    model.eval()
    outputs = {
        text: _capture_reference_representations(
            model, torch.tensor([hf_ids])
        )
        for text, (hf_ids, _) in token_pairs.items()
    }
    return transformers.__version__, model, outputs, token_pairs


@pytest.fixture(scope="module")
def worker_response(gguf_model: Path, worker_path: Path, reference):
    _, hf_model, outputs, _ = reference
    return capture_hidden_state_pair(
        worker_path,
        gguf_model,
        PROMPT_A,
        PROMPT_B,
        layer_start=1,
        layer_end=len(outputs[PROMPT_A]["blocks"]),
        position="last",
        normalize=False,
        n_ctx=128,
    )


def test_worker_hidden_state_layers_match_transformers_reference(
    gguf_model: Path, worker_response, reference
):
    _, hf_model, outputs, _ = reference
    response = worker_response

    target = response["target"]
    assert response["protocol"] == "spe-llama-worker-v2"
    assert target["site"] == "decoder-block-output-residual"
    assert target["layer_numbering"] == "one-based"
    assert target["coordinate"] == "canonical-decoder-block-output-v1"

    hf_a = outputs[PROMPT_A]["blocks"]
    hf_b = outputs[PROMPT_B]["blocks"]
    cosines: dict[int, float] = {}
    relative_errors: dict[int, float] = {}
    for layer_text, values in response["directions"].items():
        layer = int(layer_text)
        expected = (
            hf_a[layer - 1][0, -1] - hf_b[layer - 1][0, -1]
        ).detach().cpu().numpy().astype(np.float64)
        actual = np.asarray(values, dtype=np.float64)
        denominator = np.linalg.norm(expected) * np.linalg.norm(actual)
        assert denominator > 0.0
        cosines[layer] = float(np.dot(expected, actual) / denominator)
        relative_errors[layer] = float(
            np.linalg.norm(expected - actual) / max(np.linalg.norm(expected), 1e-12)
        )

    assert set(cosines) == set(range(1, len(hf_a) + 1))
    threshold = 0.995 if "f16" in gguf_model.name.lower() else 0.95
    assert min(cosines.values()) >= threshold, json.dumps(cosines, indent=2)
    assert max(relative_errors.values()) < (0.05 if threshold > 0.99 else 0.5)

    site_directions = response["site_directions"]
    expected_sites = {
        "embedding-output": (
            outputs[PROMPT_A]["embedding"][0, -1]
            - outputs[PROMPT_B]["embedding"][0, -1]
        ),
        "pre-output-normalization-residual": (
            outputs[PROMPT_A]["blocks"][-1][0, -1]
            - outputs[PROMPT_B]["blocks"][-1][0, -1]
        ),
        "post-normalization-output-head-input": (
            outputs[PROMPT_A]["post_normalization"][0, -1]
            - outputs[PROMPT_B]["post_normalization"][0, -1]
        ),
    }
    for site, expected_tensor in expected_sites.items():
        actual = np.asarray(site_directions[site]["direction"], dtype=np.float64)
        expected = expected_tensor.detach().cpu().numpy().astype(np.float64)
        denominator = np.linalg.norm(expected) * np.linalg.norm(actual)
        assert denominator > 0.0
        cosine = float(np.dot(expected, actual) / denominator)
        error = float(np.linalg.norm(expected - actual) / max(np.linalg.norm(expected), 1e-12))
        assert cosine >= threshold, f"{site}: cosine={cosine}"
        assert error < (0.05 if threshold > 0.99 else 0.5), f"{site}: error={error}"


def test_worker_vector_is_applied_at_the_matching_runtime_site(
    gguf_model: Path, worker_path: Path, worker_response, reference
):
    torch = pytest.importorskip("torch")
    _, hf_model, _, token_pairs = reference
    layer_end = int(worker_response["target"]["layer_end"])
    runtime_response = json.loads(json.dumps(worker_response))
    runtime_response["target"]["layer_start"] = 2
    runtime_response["directions"] = {
        key: value
        for key, value in worker_response["directions"].items()
        if int(key) >= 2
    }
    runtime_response["raw_delta_norms"] = {
        key: value
        for key, value in worker_response["raw_delta_norms"].items()
        if int(key) >= 2
    }
    artifact = SteeringVectorArtifact.from_llama_worker_response(
        runtime_response,
        model_path=gguf_model,
        worker_path=worker_path,
        prompt_a=PROMPT_A,
        prompt_b=PROMPT_B,
        layer_start=2,
        layer_end=layer_end,
        capture_position="last",
        normalize=False,
    )
    tokens = token_pairs[PROMPT_A][0]
    input_ids = torch.tensor([tokens])
    with torch.inference_mode():
        base_hf = (
            hf_model(input_ids=input_ids, use_cache=False)
            .logits[0, -1]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    width = int(artifact.model["hidden_state_width"])
    handles = []
    for layer_number in range(artifact.layer_start, artifact.layer_end + 1):
        direction = np.asarray(
            artifact.vector[(layer_number - 1) * width : layer_number * width],
            dtype=np.float32,
        )

        def add_direction(_module, _inputs, output, direction=direction):
            return output + torch.as_tensor(
                direction, dtype=output.dtype, device=output.device
            )

        handles.append(
            hf_model.model.layers[layer_number - 1].register_forward_hook(add_direction)
        )
    try:
        with torch.inference_mode():
            steered_hf = (
                hf_model(input_ids=input_ids, use_cache=False)
                .logits[0, -1]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
    finally:
        for handle in handles:
            handle.remove()

    native = LlamaCppDecoder(
        gguf_model,
        LlamaCppSettings(n_ctx=128, n_batch=128, n_threads=4),
    )
    try:
        native.reset(tokens)
        base_native = native.last_logits().astype(np.float64)
        artifact.validate_against_backend(native, native.provenance(include_model_sha256=False))
        native.set_activation_control_vector(
            artifact.vector,
            layer_start=artifact.layer_start,
            layer_end=artifact.layer_end,
            strength=artifact.strength,
        )
        native.reset(tokens)
        steered_native = native.last_logits().astype(np.float64)
    finally:
        native.close()

    delta_hf = steered_hf - base_hf
    delta_native = steered_native - base_native
    denominator = np.linalg.norm(delta_hf) * np.linalg.norm(delta_native)
    cosine = float(np.dot(delta_hf, delta_native) / denominator)
    threshold = 0.995 if "f16" in gguf_model.name.lower() else 0.95
    assert np.linalg.norm(delta_hf) > 0.0
    assert np.linalg.norm(delta_native) > 0.0
    assert cosine >= threshold

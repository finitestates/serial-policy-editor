"""Process boundary for the native llama.cpp hidden-state worker."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .core.errors import EditorError


PROTOCOL = "spe-llama-worker-v2"
DEFAULT_TIMEOUT_SECONDS = 300.0
MAX_PROMPT_ARGUMENT_BYTES = 128 * 1024
MAX_STDOUT_BYTES = 64 * 1024 * 1024
MAX_STDERR_BYTES = 1024 * 1024


class WorkerStartupError(EditorError):
    """The worker process could not be started."""


class WorkerTimeoutError(EditorError):
    """The worker exceeded its bounded one-shot execution time."""


class WorkerModelError(EditorError):
    """The worker started but llama.cpp could not complete the model request."""


class WorkerProtocolError(EditorError):
    """The worker did not return the expected machine-readable response."""


def _text_size(value: str) -> int:
    return len(value.encode("utf-8"))


def _validate_prompt_argument(prompt: str, label: str) -> None:
    if not isinstance(prompt, str) or not prompt:
        raise WorkerProtocolError(f"{label} must be a nonempty string")
    if _text_size(prompt) > MAX_PROMPT_ARGUMENT_BYTES:
        raise WorkerProtocolError(
            f"{label} is too large for the worker argument protocol; "
            f"maximum is {MAX_PROMPT_ARGUMENT_BYTES} UTF-8 bytes; use the native "
            "worker's --prompt-*-file interface"
        )


def _as_text(value: Any, label: str) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkerProtocolError(f"worker {label} is not valid UTF-8") from exc
    if not isinstance(value, str):
        raise WorkerProtocolError(f"worker {label} is not text")
    if _text_size(value) > (MAX_STDOUT_BYTES if label == "stdout" else MAX_STDERR_BYTES):
        raise WorkerProtocolError(f"worker {label} exceeded its output limit")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise WorkerProtocolError(f"worker response {label} must be a positive integer")
    return int(value)


def _validate_response(
    response: Any,
    *,
    model: Path,
    layer_start: int,
    layer_end: int,
    position: str,
) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise WorkerProtocolError("worker response must be a JSON object")
    if response.get("protocol") != PROTOCOL:
        raise WorkerProtocolError(f"worker must use protocol {PROTOCOL}")
    if response.get("operation") != "hidden-state-pair":
        raise WorkerProtocolError("worker returned an unsupported operation")
    backend = response.get("backend")
    if not isinstance(backend, Mapping) or backend.get("name") != "llama.cpp":
        raise WorkerProtocolError("worker response has invalid backend metadata")
    model_data = response.get("model")
    if not isinstance(model_data, Mapping):
        raise WorkerProtocolError("worker response has invalid model metadata")
    filename = model_data.get("filename")
    if not isinstance(filename, str) or Path(filename).name != model.name:
        raise WorkerProtocolError("worker response model filename does not match the requested model")
    _positive_int(model_data.get("vocabulary_size"), "vocabulary_size")
    width = _positive_int(model_data.get("hidden_state_width"), "hidden_state_width")
    model_layers = _positive_int(
        model_data.get("hidden_state_layer_count"), "hidden_state_layer_count"
    )
    decoder_block_count = model_data.get("decoder_block_count")
    if decoder_block_count is not None and decoder_block_count != model_layers:
        raise WorkerProtocolError("worker response decoder layer metadata is inconsistent")
    target = response.get("target")
    if not isinstance(target, Mapping):
        raise WorkerProtocolError("worker response has invalid target metadata")
    if target.get("site") != "decoder-block-output-residual":
        raise WorkerProtocolError("worker response target site is unsupported")
    if target.get("layer_numbering") != "one-based":
        raise WorkerProtocolError("worker response target must use one-based layers")
    if target.get("coordinate") != "canonical-decoder-block-output-v1":
        raise WorkerProtocolError("worker response target coordinate is unsupported")
    target_layer_start = target.get("layer_start")
    target_layer_end = target.get("layer_end")
    if (
        type(target_layer_start) is not int
        or type(target_layer_end) is not int
        or target_layer_start != layer_start
        or target_layer_end != layer_end
    ):
        raise WorkerProtocolError("worker response target range does not match the request")
    if target.get("position") != position:
        raise WorkerProtocolError("worker response target position does not match the request")
    if layer_end > model_layers:
        raise WorkerProtocolError("worker response target range exceeds model layers")

    prompts = response.get("prompts")
    if not isinstance(prompts, Mapping):
        raise WorkerProtocolError("worker response has invalid prompt metadata")
    _positive_int(prompts.get("a_token_count"), "prompts.a_token_count")
    _positive_int(prompts.get("b_token_count"), "prompts.b_token_count")

    directions = response.get("directions")
    norms = response.get("raw_delta_norms")
    if not isinstance(directions, Mapping) or not isinstance(norms, Mapping):
        raise WorkerProtocolError("worker response has invalid direction metadata")
    for layer in range(layer_start, layer_end + 1):
        values = directions.get(str(layer))
        if not isinstance(values, list) or len(values) != width:
            raise WorkerProtocolError(f"worker response direction {layer} has the wrong width")
        if not all(type(value) in (int, float) and math.isfinite(float(value)) for value in values):
            raise WorkerProtocolError(f"worker response direction {layer} is not finite")
        raw_norm = norms.get(str(layer))
        if type(raw_norm) not in (int, float) or not math.isfinite(float(raw_norm)) or float(raw_norm) < 0.0:
            raise WorkerProtocolError(f"worker response norm {layer} is invalid")
    return response


def capture_hidden_state_pair(
    worker: Path,
    model: Path,
    prompt_a: str,
    prompt_b: str,
    *,
    layer_start: int,
    layer_end: int,
    position: str,
    normalize: bool,
    n_ctx: int,
    n_threads: int | None = None,
    n_gpu_layers: int | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run a one-shot native worker and parse its machine-readable response."""

    _validate_prompt_argument(prompt_a, "prompt A")
    _validate_prompt_argument(prompt_b, "prompt B")
    if (
        type(layer_start) is not int
        or type(layer_end) is not int
        or layer_start < 1
        or layer_end < layer_start
    ):
        raise WorkerProtocolError("worker layer range must be one-based and ordered")
    if type(timeout_seconds) not in (int, float) or not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
        raise WorkerTimeoutError("worker timeout must be a finite positive number")

    command = [
        str(worker),
        "--model",
        str(model),
        "--prompt-a",
        prompt_a,
        "--prompt-b",
        prompt_b,
        "--layer-range",
        str(layer_start),
        str(layer_end),
        "--position",
        position,
        "--n-ctx",
        str(n_ctx),
    ]
    if not normalize:
        command.append("--no-normalize")
    if n_threads is not None:
        command.extend(("--n-threads", str(n_threads)))
    if n_gpu_layers is not None:
        command.extend(("--n-gpu-layers", str(n_gpu_layers)))

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except FileNotFoundError as exc:
        raise WorkerStartupError(f"could not start llama.cpp worker: {worker} was not found") from exc
    except PermissionError as exc:
        raise WorkerStartupError(f"could not start llama.cpp worker: permission denied for {worker}") from exc
    except subprocess.TimeoutExpired as exc:
        raise WorkerTimeoutError(
            f"llama.cpp worker timed out after {float(timeout_seconds):g} seconds"
        ) from exc
    except OSError as exc:
        raise WorkerStartupError(f"could not start llama.cpp worker: {exc}") from exc
    stdout = _as_text(getattr(completed, "stdout", ""), "stdout")
    stderr = _as_text(getattr(completed, "stderr", ""), "stderr")
    if completed.returncode != 0:
        detail = stderr.strip().splitlines()
        message = detail[-1] if detail else f"worker exited with status {completed.returncode}"
        raise WorkerModelError(
            f"llama.cpp worker model error (exit status {completed.returncode}): {message}"
        )
    try:
        response = json.loads(stdout)
    except ValueError as exc:
        raise WorkerProtocolError("llama.cpp worker returned malformed JSON") from exc
    return _validate_response(
        response,
        model=Path(model),
        layer_start=layer_start,
        layer_end=layer_end,
        position=position,
    )

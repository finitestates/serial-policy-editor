"""Process boundary for the native llama.cpp hidden-state worker."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .domain import EditorError


PROTOCOL = "spe-llama-worker-v2"


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
) -> dict[str, Any]:
    """Run a one-shot native worker and parse its machine-readable response."""

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
        )
    except OSError as exc:
        raise EditorError(f"could not start llama.cpp worker: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        message = detail[-1] if detail else f"worker exited with status {completed.returncode}"
        raise EditorError(f"llama.cpp worker failed: {message}")
    try:
        response = json.loads(completed.stdout)
    except ValueError as exc:
        raise EditorError("llama.cpp worker returned malformed JSON") from exc
    if not isinstance(response, dict):
        raise EditorError("llama.cpp worker response must be a JSON object")
    if response.get("protocol") != PROTOCOL:
        raise EditorError(f"llama.cpp worker must use protocol {PROTOCOL}")
    if response.get("operation") != "hidden-state-pair":
        raise EditorError("llama.cpp worker returned an unsupported operation")
    return response

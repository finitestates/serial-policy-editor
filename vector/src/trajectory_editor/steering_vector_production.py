"""Construction helpers for conventional hidden-state steering vectors."""

from __future__ import annotations

import inspect
from typing import Any, Mapping

import numpy as np

from trajectory_editor.activation_vectors import (
    CONTROL_VECTOR_LAYER,
    CONTROL_VECTOR_POSITION,
    CAPTURE_POSITIONS,
    SteeringVectorArtifact,
)
from trajectory_editor.core.errors import EditorError


def _supported_kwargs(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return kwargs
    names = {parameter.name for parameter in parameters}
    return {name: value for name, value in kwargs.items() if name in names}


def create_hidden_state_prompt_pair(
    backend: Any,
    prompt_a: str,
    prompt_b: str,
    *,
    layer_start: int,
    layer_end: int,
    capture_position: str = "last",
    normalize: bool = True,
    strength: float = 1.0,
) -> SteeringVectorArtifact:
    """Create a residual-stream direction for the selected decoder layers."""

    if capture_position not in CAPTURE_POSITIONS:
        raise EditorError("hidden-state capture position must be first or last")
    if not isinstance(prompt_a, str) or not prompt_a:
        raise EditorError("prompt A must be nonempty")
    if not isinstance(prompt_b, str) or not prompt_b:
        raise EditorError("prompt B must be nonempty")
    capture = getattr(backend, "hidden_state_snapshot", None)
    width_method = getattr(backend, "hidden_state_width", None)
    layer_count_method = getattr(backend, "hidden_state_layer_count", None)
    if not callable(capture) or not callable(width_method) or not callable(layer_count_method):
        raise EditorError("the loaded backend does not expose arbitrary hidden-state snapshots")
    try:
        width = int(width_method())
        layer_count = int(layer_count_method())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise EditorError(f"could not read hidden-state coordinates: {exc}") from exc
    if width < 1 or layer_count < 1:
        raise EditorError("backend reported invalid hidden-state coordinates")
    if (
        type(layer_start) is not int
        or type(layer_end) is not int
        or layer_start < 1
        or layer_end < layer_start
        or layer_end > layer_count
    ):
        raise EditorError(f"hidden-state layer range must be between 1 and {layer_count}")
    capabilities_method = getattr(backend, "hidden_state_capabilities", None)
    if callable(capabilities_method):
        capabilities = capabilities_method()
        runtime_range = capabilities.get("runtime_layer_range")
        if runtime_range is not None:
            runtime_start, runtime_end = runtime_range
            if layer_start < int(runtime_start) or layer_end > int(runtime_end):
                raise EditorError(
                    "hidden-state vector layer range is outside the backend's canonical runtime range "
                    f"{runtime_start}..{runtime_end}"
                )

    directions = np.zeros((layer_count, width), dtype=np.float64)
    raw_norms: list[float] = []
    bulk_capture = getattr(backend, "hidden_state_snapshots", None)

    def capture_prompt(prompt: str) -> Mapping[int, Any]:
        if callable(bulk_capture):
            captured = bulk_capture(
                prompt,
                **_supported_kwargs(
                    bulk_capture,
                    {
                        "layer_start": layer_start,
                        "layer_end": layer_end,
                        "position": capture_position,
                    },
                ),
            )
            if not isinstance(captured, Mapping):
                raise ValueError("bulk hidden-state capture must return a mapping")
            return captured
        return {
            layer: capture(
                prompt,
                **_supported_kwargs(
                    capture, {"layer": layer, "position": capture_position}
                ),
            )
            for layer in range(layer_start, layer_end + 1)
        }

    try:
        first_states = capture_prompt(prompt_a)
        second_states = capture_prompt(prompt_b)
        for layer in range(layer_start, layer_end + 1):
            if layer not in first_states or layer not in second_states:
                raise ValueError(f"bulk hidden-state capture omitted layer {layer}")
            first = np.asarray(first_states[layer], dtype=np.float64)
            second = np.asarray(second_states[layer], dtype=np.float64)
            if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
                raise ValueError(
                    f"hidden-state layer {layer} snapshots must be equal one-dimensional vectors"
                )
            if first.shape[0] != width:
                raise ValueError(
                    f"hidden-state layer {layer} width {first.shape[0]} does not match backend width {width}"
                )
            if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
                raise ValueError("hidden-state snapshots must be finite")
            delta = first - second
            raw_norm = float(np.linalg.norm(delta))
            raw_norms.append(raw_norm)
            if normalize and raw_norm > 0.0:
                delta = delta / raw_norm
            directions[layer - 1] = delta
    except (RuntimeError, TypeError, ValueError) as exc:
        raise EditorError(f"could not capture hidden-state pair: {exc}") from exc

    return SteeringVectorArtifact(
        vector=tuple(float(value) for value in directions.reshape(-1)),
        layer=CONTROL_VECTOR_LAYER,
        position=CONTROL_VECTOR_POSITION,
        strength=strength,
        method="hidden-state-prompt-pair-v1",
        source={
            "type": "hidden-state-prompt-pair",
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "capture_position": capture_position,
            "normalized_per_layer": bool(normalize),
            "layer_start": layer_start,
            "layer_end": layer_end,
            "layer_delta_norms": raw_norms,
        },
        layer_start=layer_start,
        layer_end=layer_end,
    )


__all__ = ["create_hidden_state_prompt_pair"]

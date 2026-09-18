"""Research-only construction of post-output steering vectors.

The core runtime can load and apply a typed artifact, but prompt-pair capture
at the post-normalization output-head input is an experimental producer.  It
belongs with the research tools rather than the core artifact loader or the
optional hidden-state steering package.
"""

from __future__ import annotations

import inspect
from typing import Any, Mapping, Sequence

import numpy as np

from trajectory_editor.activation_vectors import (
    OUTPUT_LAYER,
    SteeringVectorArtifact,
    model_identity,
)
from trajectory_editor.core.errors import EditorError


CAPTURE_POSITIONS = {"first", "last"}


def _supported_kwargs(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return kwargs
    names = {parameter.name for parameter in parameters}
    return {name: value for name, value in kwargs.items() if name in names}


def create_output_head_prompt_pair(
    backend: Any,
    provenance: Mapping[str, Any],
    prompt_a: str,
    prompt_b: str,
    *,
    capture_position: str = "last",
    normalize: bool = True,
) -> SteeringVectorArtifact:
    """Create a post-normalization output-head direction from two prompts."""

    if capture_position not in CAPTURE_POSITIONS:
        raise EditorError("output-head capture position must be first or last")
    if not isinstance(prompt_a, str) or not prompt_a:
        raise EditorError("prompt A must be nonempty")
    if not isinstance(prompt_b, str) or not prompt_b:
        raise EditorError("prompt B must be nonempty")
    capture = getattr(backend, "activation_snapshot", None)
    width_method = getattr(backend, "activation_width", None)
    if not callable(capture) or not callable(width_method):
        raise EditorError(
            "the loaded backend does not expose post-normalization output-head snapshots"
        )
    try:
        kwargs = {"layer": OUTPUT_LAYER, "position": capture_position}
        first = np.asarray(
            capture(prompt_a, **_supported_kwargs(capture, kwargs)), dtype=np.float64
        )
        second = np.asarray(
            capture(prompt_b, **_supported_kwargs(capture, kwargs)), dtype=np.float64
        )
        width = int(width_method())
    except (RuntimeError, TypeError, ValueError) as exc:
        raise EditorError(f"could not capture activation pair: {exc}") from exc
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise EditorError("output-head snapshots must be equal one-dimensional vectors")
    if first.shape[0] != width:
        raise EditorError(
            f"output-head snapshot width {first.shape[0]} does not match backend width {width}"
        )
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise EditorError("output-head snapshots must be finite")
    delta = first - second
    raw_norm = float(np.linalg.norm(delta))
    if normalize and raw_norm > 0.0:
        delta = delta / raw_norm
    return SteeringVectorArtifact(
        model=model_identity(provenance, hidden_state_width=width),
        vector=tuple(float(value) for value in delta),
        source={
            "type": "prompt-pair",
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "capture_position": capture_position,
            "normalized": bool(normalize),
            "raw_delta_norm": raw_norm,
        },
    )


def create_output_head_prompt_pairs(
    backend: Any,
    provenance: Mapping[str, Any],
    pairs: Sequence[tuple[str, str]],
    *,
    capture_position: str = "last",
    normalize: bool = True,
    strength: float = 1.0,
    source: Mapping[str, Any] | None = None,
) -> SteeringVectorArtifact:
    """Average positive-minus-negative post-output directions."""

    if not pairs:
        raise EditorError("at least one output-head prompt pair is required")
    if capture_position not in CAPTURE_POSITIONS:
        raise EditorError("output-head capture position must be first or last")
    capture = getattr(backend, "activation_snapshot", None)
    width_method = getattr(backend, "activation_width", None)
    if not callable(capture) or not callable(width_method):
        raise EditorError(
            "the loaded backend does not expose post-normalization output-head snapshots"
        )

    deltas: list[np.ndarray] = []
    pair_norms: list[float] = []
    try:
        width = int(width_method())
        if width < 1:
            raise ValueError("final hidden-state width must be positive")
        kwargs = {"layer": OUTPUT_LAYER, "position": capture_position}
        for positive_prompt, negative_prompt in pairs:
            if not isinstance(positive_prompt, str) or not positive_prompt:
                raise ValueError("positive prompts must be nonempty strings")
            if not isinstance(negative_prompt, str) or not negative_prompt:
                raise ValueError("negative prompts must be nonempty strings")
            positive = np.asarray(
                capture(positive_prompt, **_supported_kwargs(capture, kwargs)),
                dtype=np.float64,
            )
            negative = np.asarray(
                capture(negative_prompt, **_supported_kwargs(capture, kwargs)),
                dtype=np.float64,
            )
            if (
                positive.ndim != 1
                or negative.ndim != 1
                or positive.shape != negative.shape
            ):
                raise ValueError(
                    "output-head snapshots must be equal one-dimensional vectors"
                )
            if positive.shape[0] != width:
                raise ValueError(
                    f"output-head snapshot width {positive.shape[0]} does not match backend width {width}"
                )
            if not np.all(np.isfinite(positive)) or not np.all(np.isfinite(negative)):
                raise ValueError("output-head snapshots must be finite")
            delta = positive - negative
            deltas.append(delta)
            pair_norms.append(float(np.linalg.norm(delta)))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise EditorError(f"could not capture activation pairs: {exc}") from exc

    aggregate = np.mean(np.stack(deltas, axis=0), axis=0)
    aggregate_norm = float(np.linalg.norm(aggregate))
    if normalize and aggregate_norm > 0.0:
        aggregate = aggregate / aggregate_norm
    source_payload = dict(source or {})
    source_payload.setdefault("type", "prompt-pairs")
    source_payload.update(
        {
            "pair_count": len(deltas),
            "capture_position": capture_position,
            "normalized": bool(normalize),
            "pair_delta_norms": pair_norms,
            "aggregate_delta_norm": aggregate_norm,
        }
    )
    return SteeringVectorArtifact(
        model=model_identity(provenance, hidden_state_width=width),
        vector=tuple(float(value) for value in aggregate),
        strength=strength,
        method="prompt-pairs-mean-v1",
        source=source_payload,
    )


__all__ = [
    "create_output_head_prompt_pair",
    "create_output_head_prompt_pairs",
]

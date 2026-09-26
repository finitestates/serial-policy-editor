"""Model and tokenizer identities used by durable episode boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .core.errors import EditorError


def tokenizer_id_for(backend: Any) -> str:
    """Return the backend's precomputed identity for its token-ID space."""

    identity = getattr(backend, "tokenizer_id", None)
    if not callable(identity):
        raise EditorError("backend must expose tokenizer_id()")
    identity = identity()
    if not isinstance(identity, str) or not identity:
        raise EditorError("backend tokenizer_id must be nonempty text")
    return identity


def model_id_for(
    backend: Any, provenance: Mapping[str, Any] | None = None
) -> str | None:
    """Return a content identity for model weights when the backend exposes one."""

    record = provenance or {}
    identity = record.get("model_id") or record.get("model_sha256")
    if identity is None:
        identity = getattr(backend, "model_id", None)
        if callable(identity):
            identity = identity()
    if identity is None:
        return None
    if not isinstance(identity, str) or not identity:
        raise EditorError("backend model_id must be nonempty text")
    return identity


def backend_provenance_with_identity(
    backend: Any, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    """Persist model and tokenizer IDs beside backend launch provenance."""

    result = dict(provenance)
    result["tokenizer_id"] = tokenizer_id_for(backend)
    model_id = model_id_for(backend, result)
    if model_id is not None:
        result["model_id"] = model_id
    return result


def recorded_model_id(provenance: Mapping[str, Any]) -> str | None:
    """Read the current model ID or the earlier model hash field."""

    value = provenance.get("model_id") or provenance.get("model_sha256")
    return value if isinstance(value, str) and value else None


__all__ = [
    "backend_provenance_with_identity",
    "model_id_for",
    "recorded_model_id",
    "tokenizer_id_for",
]


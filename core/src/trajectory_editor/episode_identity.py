"""Model and tokenizer identities used by durable episode boundaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .core.errors import EditorError


_TOKENIZER_PROBES = (
    "",
    "a",
    " A",
    " A B",
    "hello, world!",
    "\nindented text",
    "café 東京",
    "<|special-token-probe|>",
    # These also make the fallback useful for small backends that expose only
    # a deliberately limited tokenization surface (for example test adapters).
    "conditional",
    "other",
    "U",
    "B",
    "x",
    "y",
    "z",
    "xy",
    "<BOS>U",
)


def tokenizer_id_for(backend: Any) -> str:
    """Return the backend's stable identity for its token-ID space."""

    cached = getattr(backend, "_spe_tokenizer_id_cache", None)
    if isinstance(cached, str) and cached:
        return cached
    identity = getattr(backend, "tokenizer_id", None)
    if callable(identity):
        identity = identity()
    if identity is not None:
        if not isinstance(identity, str) or not identity:
            raise EditorError("backend tokenizer_id must be nonempty text")
        return identity

    # Minimal third-party backends can still identify themselves through their
    # public token text and tokenization contract. Native backends provide a
    # stronger identity from their complete tokenizer serialization.
    vocabulary_size = int(backend.vocabulary_size())
    pieces = [str(backend.token_text(token_id)) for token_id in range(vocabulary_size)]
    probes = []
    for text in _TOKENIZER_PROBES:
        try:
            token_ids = backend.tokenize(text, special=True)
        except Exception:
            # A minimal adapter may support only a subset of text inputs. Its
            # available probes still distinguish common tokenization rules;
            # production backends expose their full tokenizer serialization.
            continue
        probes.append((text, [int(token_id) for token_id in token_ids]))
    payload = {
        "pieces": pieces,
        "eog_token_ids": [int(value) for value in backend.eog_token_ids()],
        "probe_tokenizations": probes,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    identity = hashlib.sha256(
        b"serial-policy-editor-tokenizer-v1\0" + encoded
    ).hexdigest()
    try:
        setattr(backend, "_spe_tokenizer_id_cache", identity)
    except (AttributeError, TypeError):
        pass
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


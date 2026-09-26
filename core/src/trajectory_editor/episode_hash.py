"""Stable sampler stream identities and boundary validation."""

from __future__ import annotations

import hashlib
import re

from .core.errors import EditorError


def validate_boundary(value: int, name: str = "boundary") -> int:
    if type(value) is not int or value < 0:
        raise EditorError(f"{name} must be a nonnegative integer")
    return value


def validate_fingerprint(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise EditorError("stream_fingerprint must be a lowercase SHA-256 hex digest")
    return value


def token_prefix_sha256(
    token_ids: list[int] | tuple[int, ...], *, tokenizer_id: str | None = None
) -> str:
    if not isinstance(token_ids, (list, tuple)):
        raise EditorError("token IDs must be a list or tuple of integers")
    digest = hashlib.sha256()
    if tokenizer_id is not None:
        if not isinstance(tokenizer_id, str) or not tokenizer_id:
            raise EditorError("tokenizer_id must be nonempty text")
        tokenizer_bytes = tokenizer_id.encode("utf-8")
        digest.update(len(tokenizer_bytes).to_bytes(8, "little", signed=False))
        digest.update(tokenizer_bytes)
    for token_id in token_ids:
        if type(token_id) is not int or not 0 <= token_id < (1 << 63):
            raise EditorError("token IDs must be nonnegative signed-64-bit integers")
        digest.update(token_id.to_bytes(8, "little", signed=True))
    return digest.hexdigest()

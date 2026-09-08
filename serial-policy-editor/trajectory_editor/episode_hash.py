"""Stable sampler stream identities with explicit input validation."""

from __future__ import annotations

import hashlib
import re

from .domain import EditorError


def validate_coordinate(value: int, name: str = "coordinate") -> int:
    if type(value) is not int or value < 0:
        raise EditorError(f"{name} must be a nonnegative integer")
    return value


def validate_fingerprint(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise EditorError("stream_fingerprint must be a lowercase SHA-256 hex digest")
    return value


def token_prefix_sha256(token_ids: list[int] | tuple[int, ...]) -> str:
    if not isinstance(token_ids, (list, tuple)):
        raise EditorError("token IDs must be a list or tuple of integers")
    digest = hashlib.sha256()
    for token_id in token_ids:
        if type(token_id) is not int or not 0 <= token_id < (1 << 63):
            raise EditorError("token IDs must be nonnegative signed-64-bit integers")
        digest.update(token_id.to_bytes(8, "little", signed=True))
    return digest.hexdigest()

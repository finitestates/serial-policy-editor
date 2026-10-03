"""Versioned binary-framed messages for the Rust real-model worker."""

from __future__ import annotations

import json
import struct
from typing import BinaryIO


PROTOCOL_VERSION = 1
MAX_JSON_FRAME_BYTES = 16 * 1024 * 1024


class ProtocolError(ValueError):
    pass


def _read_exact(stream: BinaryIO, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        part = stream.read(length - len(chunks))
        if not part:
            raise ProtocolError(f"truncated protocol frame: wanted {length} bytes")
        chunks.extend(part)
    return bytes(chunks)


def read_json_frame(stream: BinaryIO) -> dict:
    prefix = stream.read(4)
    if not prefix:
        raise EOFError("protocol input closed")
    if len(prefix) != 4:
        prefix += _read_exact(stream, 4 - len(prefix))
    length = struct.unpack("<I", prefix)[0]
    if length == 0 or length > MAX_JSON_FRAME_BYTES:
        raise ProtocolError(f"invalid JSON frame length {length}")
    try:
        value = json.loads(_read_exact(stream, length))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON frame: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError("JSON frame must contain an object")
    return value


def write_json_frame(stream: BinaryIO, value: dict) -> int:
    try:
        payload = json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"cannot encode JSON frame: {exc}") from exc
    if not payload or len(payload) > MAX_JSON_FRAME_BYTES:
        raise ProtocolError(f"invalid JSON frame length {len(payload)}")
    frame = struct.pack("<I", len(payload)) + payload
    stream.write(frame)
    stream.flush()
    return len(frame)


def f64le_logits(logits, vocabulary_size: int) -> bytes:
    """Validate a backend row and encode the protocol's f64 little-endian body."""
    import numpy as np

    values = np.asarray(logits)
    if values.ndim != 1 or values.shape[0] != vocabulary_size:
        raise ProtocolError(
            f"backend logits must have shape ({vocabulary_size},), got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ProtocolError("backend logits contain a non-finite value")
    return np.asarray(values, dtype="<f8", order="C").tobytes(order="C")

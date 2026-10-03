"""Model-free checks for the isolated Rust/Python real-model protocol."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "terminal-ui/scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import real_model_worker  # noqa: E402
import worker_protocol  # noqa: E402


def test_json_frame_round_trip_and_truncated_input():
    buffer = BytesIO()
    size = worker_protocol.write_json_frame(
        buffer, {"version": 1, "request_id": 4, "op": "eval", "token_ids": [7]}
    )
    assert size == len(buffer.getvalue())
    buffer.seek(0)
    assert worker_protocol.read_json_frame(buffer) == {
        "version": 1, "request_id": 4, "op": "eval", "token_ids": [7]
    }

    with pytest.raises(worker_protocol.ProtocolError, match="truncated"):
        worker_protocol.read_json_frame(BytesIO(b"\x05\x00"))


def test_json_frame_rejects_invalid_lengths_and_non_object_values():
    with pytest.raises(worker_protocol.ProtocolError, match="length"):
        worker_protocol.read_json_frame(BytesIO(b"\x00\x00\x00\x00"))

    body = b"[]"
    malformed = len(body).to_bytes(4, "little") + body
    with pytest.raises(worker_protocol.ProtocolError, match="object"):
        worker_protocol.read_json_frame(BytesIO(malformed))


def test_worker_request_validation_and_error_response_shape():
    assert real_model_worker._request_shape({
        "version": 1, "request_id": 2, "op": "logits"
    }) == (2, "logits")
    for request in (
        {"version": 2, "request_id": 2, "op": "logits"},
        {"version": 1, "request_id": 0, "op": "logits"},
        {"version": 1, "request_id": 2, "op": ""},
    ):
        with pytest.raises(worker_protocol.ProtocolError):
            real_model_worker._request_shape(request)
    error = real_model_worker._response(2, "eval", error="backend failed")
    assert error["ok"] is False
    assert error["error"] == "backend failed"
    assert error["binary_bytes"] == 0


def test_logits_wire_payload_is_f64_little_endian_and_validated():
    logits = np.asarray([1.25, -2.5], dtype=np.float32)
    payload = worker_protocol.f64le_logits(logits, 2)
    assert payload == np.asarray([1.25, -2.5], dtype="<f8").tobytes()

    for invalid, vocab in (
        (np.zeros((1, 2), dtype=np.float32), 2),
        (np.asarray([1.0], dtype=np.float32), 2),
        (np.asarray([np.nan, 2.0], dtype=np.float32), 2),
    ):
        with pytest.raises(worker_protocol.ProtocolError):
            worker_protocol.f64le_logits(invalid, vocab)

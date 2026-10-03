#!/usr/bin/env python3
"""Persistent SPE inference worker for the standalone Rust Choice experiment."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from worker_protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    f64le_logits,
    read_json_frame,
    write_json_frame,
)


ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "core/src"), str(ROOT)]


def _protocol_stdout():
    # Keep a duplicate of the pipe before routing fd 1 to diagnostics. This
    # also catches native-library writes that bypass Python's sys.stdout.
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(protocol_fd, "wb", buffering=0)


def _jsonable(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _request_shape(request: dict) -> tuple[int, str]:
    if request.get("version") != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported request protocol version {request.get('version')!r}")
    request_id = request.get("request_id")
    op = request.get("op")
    if type(request_id) is not int or request_id < 1:
        raise ProtocolError("request_id must be a positive integer")
    if not isinstance(op, str) or not op:
        raise ProtocolError("op must be a nonempty string")
    return request_id, op


def _response(request_id: int, op: str, *, result=None, error=None,
              backend_wall_s: float = 0.0, binary: bytes = b"", **fields) -> dict:
    value = {
        "version": PROTOCOL_VERSION,
        "request_id": request_id,
        "op": op,
        "ok": error is None,
        "result": _jsonable(result),
        "error": None if error is None else str(error),
        "backend_wall_s": float(backend_wall_s),
        "binary_dtype": "f64le" if binary else None,
        "binary_count": len(binary) // 8 if binary else 0,
        "binary_bytes": len(binary),
        **fields,
    }
    return value


def _write_response(output, response: dict, binary: bytes = b"") -> None:
    write_json_frame(output, response)
    if binary:
        output.write(binary)
        output.flush()


def _effective_launch(profile: dict) -> dict:
    values = vars(profile["options"])
    names = (
        "cache", "temperature", "seed", "top_k", "top_p", "min_p",
        "typical_p", "tail_free_z", "n_ctx", "n_batch", "n_ubatch",
        "n_threads", "n_threads_batch", "n_gpu_layers", "transformers_device",
        "transformers_dtype", "transformers_torch_threads",
        "transformers_torch_interop_threads",
    )
    return {name: values[name] for name in names if name in values}


def run_worker(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--model-root", type=Path)
    args = parser.parse_args(argv)
    output = _protocol_stdout()

    backend = None
    profile = None
    try:
        from benchmarks.real_model import load_profile
        from trajectory_editor.episode_backend_loader import load_backend

        profile = load_profile(
            args.profile, args.model_root, None, None, None, None
        )
        started = perf_counter()
        # load_backend prints progress; keep every byte off the worker pipe.
        with redirect_stdout(sys.stderr):
            backend = load_backend(profile["options"])
            provenance = dict(backend.provenance(include_model_sha256=True))
            vocabulary_size = int(backend.vocabulary_size())
            tokenizer_id = str(backend.tokenizer_id())
        startup_wall_s = perf_counter() - started
        provenance.setdefault("backend", profile["backend"])
        provenance.setdefault("model_path", str(profile["model_path"]))
        provenance.setdefault("tokenizer_id", tokenizer_id)
        hello = {
            "profile": {
                "name": profile["name"],
                "path": profile["path"],
                "backend": profile["backend"],
                "model_path": str(profile["model_path"]),
                "launch": _jsonable(profile["launch"]),
                "effective_launch": _effective_launch(profile),
            },
            "vocabulary_size": vocabulary_size,
            "tokenizer_id": tokenizer_id,
            "provenance": _jsonable(provenance),
            "startup_wall_s": startup_wall_s,
        }
        _write_response(output, _response(
            0, "hello", result=hello, backend_wall_s=startup_wall_s
        ))
    except BaseException as exc:
        _write_response(output, _response(
            0, "hello", error=f"{type(exc).__name__}: {exc}"
        ))
        return 2

    assert backend is not None and profile is not None
    from trajectory_editor.episode_hash import token_prefix_sha256

    while True:
        request = None
        try:
            request = read_json_frame(sys.stdin.buffer)
            request_id, op = _request_shape(request)
            started = perf_counter()
            result = None
            binary = b""
            fields = {}
            if op == "tokenize":
                token_ids = backend.tokenize(
                    request["text"],
                    add_bos=request.get("add_bos") is True,
                    special=request.get("special") is True,
                )
                backend_wall_s = perf_counter() - started
                result = {
                    "token_ids": [int(value) for value in token_ids],
                    "tokenizer_id": str(backend.tokenizer_id()),
                    "stream_fingerprint": token_prefix_sha256(token_ids),
                }
            elif op == "reset":
                backend.reset([int(value) for value in request["token_ids"]])
                backend_wall_s = perf_counter() - started
                result = {"ok": True}
            elif op == "eval":
                token_ids = [int(value) for value in request["token_ids"]]
                backend.eval(token_ids)
                backend_wall_s = perf_counter() - started
                result = {"evaluated_token_ids": token_ids}
            elif op == "logits":
                import numpy as np

                values = backend.last_logits()
                backend_wall_s = perf_counter() - started
                binary = f64le_logits(values, int(backend.vocabulary_size()))
                result = {"shape": [int(backend.vocabulary_size())]}
                fields = {
                    "binary_dtype": "f64le",
                    "binary_count": int(backend.vocabulary_size()),
                    "binary_bytes": len(binary),
                    "source_dtype": str(np.asarray(values).dtype),
                }
            elif op == "render":
                result = {"text": str(backend.render(
                    [int(value) for value in request["token_ids"]],
                    special=request.get("special") is True,
                ))}
                backend_wall_s = perf_counter() - started
            elif op == "token_text":
                result = {"text": str(backend.token_text(int(request["token_id"])))}
                backend_wall_s = perf_counter() - started
            elif op == "is_eog":
                result = {"is_eog": bool(backend.is_eog(int(request["token_id"])))}
                backend_wall_s = perf_counter() - started
            elif op == "eog_token_ids":
                result = {"token_ids": [int(value) for value in backend.eog_token_ids()]}
                backend_wall_s = perf_counter() - started
            elif op == "shutdown":
                result = {"closed": True}
                backend_wall_s = perf_counter() - started
                _write_response(output, _response(
                    request_id, op, result=result, backend_wall_s=backend_wall_s
                ))
                return 0
            else:
                raise ProtocolError(f"unsupported operation {op!r}")
            response = _response(
                request_id, op, result=result, backend_wall_s=backend_wall_s,
                binary=binary, **fields
            )
            _write_response(output, response, binary)
        except EOFError:
            return 0
        except BaseException as exc:
            request_id = request.get("request_id", -1) if request else -1
            op = request.get("op", "unknown") if request else "unknown"
            _write_response(output, _response(
                request_id if type(request_id) is int else -1,
                op if isinstance(op, str) else "unknown",
                error=f"{type(exc).__name__}: {exc}",
            ))


if __name__ == "__main__":
    raise SystemExit(run_worker())

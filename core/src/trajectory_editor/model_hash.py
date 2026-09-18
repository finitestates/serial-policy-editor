"""Small, deterministic content hashing helpers for model provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_path(path: Path) -> str:
    """Hash one model file or a local model directory.

    Directory hashes include sorted relative filenames and file contents. The
    result is intentionally computed by callers at most once per backend
    lifecycle; it is not suitable for per-token work.
    """

    path = Path(path)
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        files = sorted(
            candidate for candidate in path.rglob("*") if candidate.is_file()
        )
        for candidate in files:
            relative = candidate.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
    raise OSError(f"model path does not exist: {path}")

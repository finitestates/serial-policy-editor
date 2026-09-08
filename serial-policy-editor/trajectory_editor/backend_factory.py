"""Backend registry and construction helpers.

The CLI is intentionally a frontend onto this small factory rather than the
owner of backend semantics.  Future launchers/TUIs can use the same entry point.
"""

from __future__ import annotations

from pathlib import Path

from .decoder import Decoder, LlamaCppDecoder, LlamaCppSettings
from .domain import EditorError
from .episode_backend import CacheMode, validate_cache_mode
from .transformers_backend import TransformersBackend, TransformersSettings


BACKEND_NAMES = ("llama.cpp", "transformers")


def normalize_backend_name(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "llama": "llama.cpp",
        "llama-cpp": "llama.cpp",
        "llama_cpp": "llama.cpp",
        "hf": "transformers",
        "huggingface": "transformers",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in BACKEND_NAMES:
        raise EditorError(
            f"unsupported backend {value!r}; choose one of {', '.join(BACKEND_NAMES)}"
        )
    return normalized


def create_backend(
    backend: str,
    model_path: Path,
    *,
    llama_settings: LlamaCppSettings | None = None,
    transformers_settings: TransformersSettings | None = None,
    cache_mode: CacheMode = "auto",
) -> Decoder:
    backend = normalize_backend_name(backend)
    cache_mode = validate_cache_mode(cache_mode)
    if backend == "llama.cpp":
        return LlamaCppDecoder(
            model_path,
            llama_settings or LlamaCppSettings(),
            cache_mode=cache_mode,
        )
    if backend == "transformers":
        return TransformersBackend(
            model_path,
            transformers_settings or TransformersSettings(),
            cache_mode=cache_mode,
        )
    raise AssertionError(f"unhandled backend {backend}")

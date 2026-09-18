"""Compatibility facade for the canonical core backend contract."""

from __future__ import annotations

from .core.backend import (
    CacheMode,
    InferenceBackend,
    require_inference_backend,
    validate_cache_mode as _validate_cache_mode,
)
from .core.errors import EditorError


EpisodeBackend = InferenceBackend


def validate_cache_mode(value: str) -> CacheMode:
    return _validate_cache_mode(value)


def require_episode_backend(backend: EpisodeBackend) -> None:
    require_inference_backend(backend)


__all__ = [
    "CacheMode",
    "EditorError",
    "EpisodeBackend",
    "InferenceBackend",
    "require_episode_backend",
    "require_inference_backend",
    "validate_cache_mode",
]

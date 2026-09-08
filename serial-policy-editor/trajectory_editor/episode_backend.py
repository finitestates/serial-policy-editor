"""Minimal inference contract required by the policy-episode core."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from .domain import EditorError


CacheMode = Literal["auto", "off"]


def validate_cache_mode(value: str) -> CacheMode:
    if value not in {"auto", "off"}:
        raise EditorError("cache mode must be auto or off")
    return value  # type: ignore[return-value]


@runtime_checkable
class EpisodeBackend(Protocol):
    def vocabulary_size(self) -> int: ...

    def reset(self, prefix_token_ids: list[int]) -> None: ...

    def eval(self, token_ids: list[int]) -> None: ...

    def last_logits(self) -> np.ndarray: ...

    def tokenize(
        self, text: str, *, add_bos: bool = False, special: bool = False
    ) -> list[int]: ...

    def render(self, token_ids: list[int], *, special: bool = False) -> str: ...

    def token_text(self, token_id: int) -> str: ...

    def is_eog(self, token_id: int) -> bool: ...

    def eog_token_ids(self) -> tuple[int, ...]: ...

    def provenance(self, *, include_model_sha256: bool = True) -> Mapping[str, Any]: ...

    # Backends may expose branch_to_prefix(prefix_token_ids) as a private
    # optimization. The episode core falls back to reset() when it is absent.


def require_episode_backend(backend: EpisodeBackend) -> None:
    """Fail clearly at the backend boundary."""
    if backend.vocabulary_size() < 1:
        raise RuntimeError("backend reported an empty vocabulary")
    logits = backend.last_logits
    if not callable(logits):
        raise TypeError("backend does not expose full-vocabulary logits")

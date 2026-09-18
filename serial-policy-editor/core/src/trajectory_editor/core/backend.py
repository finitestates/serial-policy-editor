"""Small inference boundary required by the episode runtime.

Concrete llama.cpp and Transformers adapters remain outside this package.
Optional research operations such as hidden-state capture do not belong in
this contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from .errors import EditorError


CacheMode = Literal["auto", "off"]


def validate_cache_mode(value: str) -> CacheMode:
    if value not in {"auto", "off"}:
        raise EditorError("cache mode must be auto or off")
    return value  # type: ignore[return-value]


@runtime_checkable
class InferenceBackend(Protocol):
    """Minimal model interface required by stepped episode editing."""

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


def require_inference_backend(backend: InferenceBackend) -> None:
    """Fail clearly at the backend boundary."""

    if backend.vocabulary_size() < 1:
        raise RuntimeError("backend reported an empty vocabulary")
    if not callable(backend.last_logits):
        raise TypeError("backend does not expose full-vocabulary logits")

"""Small inference boundary required by the episode runtime.

Concrete llama.cpp and Transformers adapters remain outside this package.
Optional research operations such as hidden-state capture do not belong in
this contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class BackendStateSnapshot:
    """Opaque, in-memory snapshot of one backend's exact inference position.

    Snapshots are tied to the backend instance and token prefix that created
    them. They are transient acceleration state, not episode or replay state;
    adapters may copy substantial KV-cache data when creating one.
    """

    backend_token: object = field(repr=False, compare=False)
    prefix_token_ids: tuple[int, ...]
    payload: Any = field(repr=False, compare=False)


@runtime_checkable
class SnapshotableInferenceBackend(Protocol):
    """Optional capability for temporarily saving and restoring model state.

    This is deliberately separate from :class:`InferenceBackend`, so minimal
    and third-party backends remain conforming without snapshot support.
    Implementations return `None` when an exact independent snapshot is not
    available. A snapshot should be restored only to the backend instance and
    prefix recorded in it, and should normally be held for one short-lived
    speculative operation.
    """

    def snapshot_state(self) -> BackendStateSnapshot | None: ...

    def restore_state(self, snapshot: BackendStateSnapshot) -> bool: ...


def require_inference_backend(backend: InferenceBackend) -> None:
    """Fail clearly at the backend boundary."""

    if backend.vocabulary_size() < 1:
        raise RuntimeError("backend reported an empty vocabulary")
    if not callable(backend.last_logits):
        raise TypeError("backend does not expose full-vocabulary logits")

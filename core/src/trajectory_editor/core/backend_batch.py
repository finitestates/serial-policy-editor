"""Shared lane bookkeeping for adapter-owned batched inference sessions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .backend import InferenceBackend
from .backend_position import BackendPosition


PrefixMap = Mapping[int, tuple[int, ...]]
AdvanceMap = Mapping[int, tuple[tuple[int, ...], tuple[int, ...]]]
LogitMap = Mapping[int, np.ndarray]


@dataclass
class _LaneState:
    token_ids: list[int]
    pending: list[int] = field(default_factory=list)
    logits: np.ndarray | None = None
    needs_rebuild: bool = False


class InferenceBatch:
    """Sequence lanes plus adapter callbacks for prefill and incremental work.

    Adapters own cache layout and model calls. This class keeps lane token
    ledgers, queues append-only ``eval`` calls, and exposes one backend-shaped
    view per lane to :class:`EpisodeEngine`.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        prefixes: Sequence[Sequence[int]],
        *,
        prefill: Callable[[PrefixMap], LogitMap],
        advance: Callable[[AdvanceMap], LogitMap],
        close: Callable[[], None] | None = None,
    ) -> None:
        if not prefixes:
            raise ValueError("an inference batch needs at least one lane")
        self._backend = backend
        self._prefill = prefill
        self._advance = advance
        self._close_callback = close
        self._closed = False
        self._lanes = [
            _LaneState([int(token_id) for token_id in prefix])
            for prefix in prefixes
        ]
        if any(not lane.token_ids for lane in self._lanes):
            raise ValueError("batch lane prefixes cannot be empty")
        self._install_logits(self._prefill(self._prefixes(range(len(self._lanes)))))

    def lane(self, lane_id: int) -> InferenceBackend:
        self._check_lane_id(lane_id)
        return _InferenceBatchLane(self, lane_id)

    def flush(self, active_lane_ids: Sequence[int]) -> None:
        self._ensure_open()
        ids = self._validate_lane_ids(active_lane_ids)
        if any(self._lanes[lane_id].needs_rebuild for lane_id in ids):
            self.rebuild(ids)
            return
        requests: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        for lane_id in ids:
            lane = self._lanes[lane_id]
            requests[lane_id] = (tuple(lane.token_ids), tuple(lane.pending))
        self._mark_inactive(ids)
        if not ids:
            return
        outputs = self._advance(requests)
        self._install_logits(outputs, expected_ids=ids)
        for lane_id in ids:
            self._lanes[lane_id].pending.clear()
            self._lanes[lane_id].needs_rebuild = False

    def rebuild(self, active_lane_ids: Sequence[int]) -> None:
        self._ensure_open()
        ids = self._validate_lane_ids(active_lane_ids)
        self._mark_inactive(ids)
        if not ids:
            return
        outputs = self._prefill(self._prefixes(ids))
        self._install_logits(outputs, expected_ids=ids)
        for lane_id in ids:
            lane = self._lanes[lane_id]
            lane.pending.clear()
            lane.needs_rebuild = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for lane in self._lanes:
            lane.logits = None
            lane.pending.clear()
        if self._close_callback is not None:
            self._close_callback()

    def _prefixes(self, lane_ids: Sequence[int]) -> dict[int, tuple[int, ...]]:
        return {
            lane_id: tuple(self._lanes[lane_id].token_ids)
            for lane_id in lane_ids
        }

    def _install_logits(
        self, outputs: LogitMap, *, expected_ids: Sequence[int] | None = None
    ) -> None:
        expected = set(range(len(self._lanes))) if expected_ids is None else set(expected_ids)
        if set(outputs) != expected:
            raise RuntimeError("batch adapter returned logits for the wrong lanes")
        vocabulary_size = self._backend.vocabulary_size()
        for lane_id, raw_logits in outputs.items():
            logits = np.asarray(raw_logits, dtype=np.float32)
            if logits.ndim != 1 or len(logits) != vocabulary_size:
                raise RuntimeError("batch adapter returned logits with the wrong vocabulary shape")
            if not np.all(np.isfinite(logits)):
                raise RuntimeError("batch adapter returned non-finite logits")
            lane = self._lanes[lane_id]
            lane.logits = logits.copy()

    def _mark_inactive(self, active_ids: Sequence[int]) -> None:
        active = set(active_ids)
        for lane_id, lane in enumerate(self._lanes):
            if lane_id not in active:
                lane.logits = None

    def _validate_lane_ids(self, lane_ids: Sequence[int]) -> tuple[int, ...]:
        ids = tuple(lane_ids)
        if any(type(lane_id) is not int for lane_id in ids):
            raise ValueError("active batch lane IDs must be integers")
        if len(set(ids)) != len(ids):
            raise ValueError("active batch lane IDs must be unique")
        for lane_id in ids:
            self._check_lane_id(lane_id)
        return ids

    def _check_lane_id(self, lane_id: int) -> None:
        if type(lane_id) is not int or not 0 <= lane_id < len(self._lanes):
            raise IndexError("batch lane ID is out of range")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("inference batch is closed")


class _InferenceBatchLane:
    """A single-sequence backend view backed by an ``InferenceBatch``."""

    def __init__(self, batch: InferenceBatch, lane_id: int) -> None:
        self._batch = batch
        self._lane_id = lane_id

    def __getattr__(self, name: str) -> Any:
        # Optional adapter capabilities (activation controls, rendering
        # streams, diagnostics) still belong to the underlying model adapter.
        return getattr(self._batch._backend, name)

    @property
    def _state(self) -> _LaneState:
        self._batch._ensure_open()
        return self._batch._lanes[self._lane_id]

    def vocabulary_size(self) -> int:
        return self._batch._backend.vocabulary_size()

    def reset(self, prefix_token_ids: list[int]) -> None:
        values = [int(token_id) for token_id in prefix_token_ids]
        if not values:
            raise RuntimeError("decoder prefix cannot be empty")
        lane = self._state
        lane.token_ids = values
        lane.pending.clear()
        lane.logits = None
        lane.needs_rebuild = True

    def eval(self, token_ids: list[int]) -> None:
        values = [int(token_id) for token_id in token_ids]
        if not values:
            return
        lane = self._state
        lane.token_ids.extend(values)
        if lane.needs_rebuild:
            lane.pending.clear()
        else:
            lane.pending.extend(values)
        lane.logits = None

    def last_logits(self) -> np.ndarray:
        logits = self._state.logits
        if logits is None:
            raise RuntimeError("batch lane logits are stale; flush or rebuild its session")
        return logits.copy()

    def tokenize(
        self, text: str, *, add_bos: bool = False, special: bool = False
    ) -> list[int]:
        return self._batch._backend.tokenize(text, add_bos=add_bos, special=special)

    def render(self, token_ids: list[int], *, special: bool = False) -> str:
        return self._batch._backend.render(token_ids, special=special)

    def token_text(self, token_id: int) -> str:
        return self._batch._backend.token_text(token_id)

    def is_eog(self, token_id: int) -> bool:
        return self._batch._backend.is_eog(token_id)

    def eog_token_ids(self) -> tuple[int, ...]:
        return self._batch._backend.eog_token_ids()

    def tokenizer_id(self) -> str:
        return self._batch._backend.tokenizer_id()

    def provenance(self, *, include_model_sha256: bool = True) -> Mapping[str, Any]:
        return self._batch._backend.provenance(
            include_model_sha256=include_model_sha256
        )

    def position(self) -> BackendPosition:
        lane = self._state
        return BackendPosition(
            token_ids=tuple(lane.token_ids),
            cursor=len(lane.token_ids),
            cache_start=None,
            cache_end=None,
            cache_reusable=False,
            logits_valid=lane.logits is not None and not lane.pending and not lane.needs_rebuild,
        )

    def branch_to_prefix(self, prefix_token_ids: list[int]) -> None:
        values = [int(token_id) for token_id in prefix_token_ids]
        if not values:
            raise RuntimeError("decoder prefix cannot be empty")
        lane = self._state
        if values == lane.token_ids:
            return
        if (
            not lane.needs_rebuild
            and len(values) >= len(lane.token_ids)
            and values[: len(lane.token_ids)] == lane.token_ids
        ):
            self.eval(values[len(lane.token_ids) :])
            return
        self.reset(values)

    def truncate_to(self, length: int) -> bool:
        lane = self._state
        if type(length) is not int or length < 1 or length > len(lane.token_ids):
            raise RuntimeError("batch lane truncation target is out of range")
        self.reset(lane.token_ids[:length])
        return True


__all__ = ["AdvanceMap", "InferenceBatch", "LogitMap", "PrefixMap"]

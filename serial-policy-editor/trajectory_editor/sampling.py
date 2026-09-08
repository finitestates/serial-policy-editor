"""Replayable sampling and ranked raw-logit menus."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .domain import RNG_SCHEME, SamplingConfig, EditorError, MIN_SEED, MAX_SEED
from .episode_hash import validate_coordinate, validate_fingerprint


@dataclass(frozen=True)
class SparseDistribution:
    ids: np.ndarray
    probabilities: np.ndarray

    def probability(self, token_id: int) -> float:
        matches = np.flatnonzero(self.ids == int(token_id))
        return float(self.probabilities[matches[0]]) if len(matches) else 0.0


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.array(values, dtype=np.float64, copy=True)
    if shifted.ndim != 1 or not len(shifted) or not np.all(np.isfinite(shifted)):
        raise ValueError("softmax values must be a finite nonempty vector")
    shifted -= np.max(shifted)
    exponentials = np.exp(shifted)
    denominator = float(np.sum(exponentials))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("softmax normalization failed")
    return exponentials / denominator


def _top_ids(values: np.ndarray, count: int) -> np.ndarray:
    """Return descending-logit ids with token-id ascending as the tie-break."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("decoder logits must be a finite nonempty vector")
    count = min(max(1, int(count)), len(values))
    if count == len(values):
        selected = np.arange(len(values), dtype=np.int64)
    else:
        threshold = float(np.partition(values, len(values) - count)[len(values) - count])
        above = np.flatnonzero(values > threshold)
        tied = np.flatnonzero(values == threshold)
        needed = count - len(above)
        selected = np.concatenate((above, tied[:needed])).astype(np.int64, copy=False)
    order = np.lexsort((selected, -values[selected]))
    return selected[order]


def _validated_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(
            "decoder logits must be a finite nonempty one-dimensional array"
        )
    return values


def _history_penalty_surface(
    values: np.ndarray,
    config: SamplingConfig,
    history_token_ids: list[int] | tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply the canonical reconstructable history policy to raw logits."""

    if history_token_ids is None:
        if config.history_penalties_active:
            raise ValueError("active history penalties require exact prefix token ids")
        return values.copy(), np.zeros(len(values), dtype=np.int64), 0
    history = np.asarray(history_token_ids, dtype=np.int64)
    if history.ndim != 1 or np.any(history < 0) or np.any(history >= len(values)):
        raise ValueError("history token ids must address the decoder vocabulary")
    if config.repeat_last_n == 0 or not len(history):
        considered = history[:0]
    elif config.repeat_last_n == -1:
        considered = history
    else:
        considered = history[-config.repeat_last_n :]
    counts = np.bincount(considered, minlength=len(values)).astype(
        np.int64, copy=False
    )
    adjusted = values.copy()
    present = counts > 0
    if float(config.repeat_penalty) != 1.0 and np.any(present):
        selected = adjusted[present]
        adjusted[present] = np.where(
            selected < 0.0,
            selected * float(config.repeat_penalty),
            selected / float(config.repeat_penalty),
        )
    if float(config.presence_penalty) != 0.0:
        adjusted[present] -= float(config.presence_penalty)
    if float(config.frequency_penalty) != 0.0:
        adjusted -= counts * float(config.frequency_penalty)
    if not np.all(np.isfinite(adjusted)):
        raise ValueError("history penalties produced non-finite policy logits")
    return adjusted, counts, int(len(considered))


def _rank(values: np.ndarray, token_id: int) -> int:
    token_id = int(token_id)
    if not 0 <= token_id < len(values):
        raise ValueError("token id is outside the decoder vocabulary")
    target = float(values[token_id])
    higher = int(np.count_nonzero(values > target))
    tied_before = int(np.count_nonzero(values[:token_id] == target))
    return 1 + higher + tied_before


def _stages_from_adjusted(
    adjusted: np.ndarray, config: SamplingConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | None]]:
    """Run decoder stages on an already prepared history-policy surface."""
    if config.temperature == 0.0:
        greedy = _top_ids(adjusted, 1)
        return adjusted, adjusted, {
            "after_temperature": greedy,
            "after_top_k": greedy,
            "after_top_p": greedy,
            "after_min_p": greedy,
        }

    scaled = adjusted / float(config.temperature)
    if not np.all(np.isfinite(scaled)):
        raise ValueError("temperature produced non-finite scaled logits")

    after_top_k = _top_ids(scaled, min(config.top_k, len(scaled)))
    after_top_p = after_top_k
    if config.top_p < 1.0:
        probabilities = _softmax(scaled[after_top_k])
        keep_count = int(
            np.searchsorted(np.cumsum(probabilities), config.top_p, side="left")
        ) + 1
        after_top_p = after_top_k[: max(1, keep_count)]

    after_min_p = after_top_p
    if config.min_p > 0.0:
        probabilities = _softmax(scaled[after_top_p])
        mask = probabilities >= config.min_p * float(np.max(probabilities))
        if np.any(mask):
            after_min_p = after_top_p[mask]

    return adjusted, scaled, {
        # None means the stage retains the full vocabulary. Avoid allocating a
        # vocabulary-sized id array when no filtering happens at this stage.
        "after_temperature": None,
        "after_top_k": after_top_k,
        "after_top_p": after_top_p,
        "after_min_p": after_min_p,
    }


def position_uniform(seed: int, stream_fingerprint: str, aligned_step: int) -> float:
    if type(seed) is not int or not MIN_SEED <= seed <= MAX_SEED:
        raise EditorError("seed must be a signed-64-bit integer")
    validate_fingerprint(stream_fingerprint)
    validate_coordinate(aligned_step, "sampling position")
    payload = f"{RNG_SCHEME}:{seed}:{stream_fingerprint}:{aligned_step}".encode()
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return (value + 0.5) / float(1 << 64)


def draw_token(
    distribution: SparseDistribution,
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
) -> int:
    draw = position_uniform(seed, stream_fingerprint, aligned_step)
    index = int(
        np.searchsorted(np.cumsum(distribution.probabilities), draw, side="right")
    )
    return int(distribution.ids[min(index, len(distribution.ids) - 1)])


def raw_rank(logits: np.ndarray, token_id: int) -> int:
    """Return the one-based full-vocabulary rank with the menu tie-break."""
    values = _validated_logits(logits)
    return _rank(values, token_id)


def top_raw_ids(logits: np.ndarray, count: int) -> list[int]:
    return [int(value) for value in _top_ids(np.asarray(logits), count)]


class ObservationStatistics:
    """Owned numeric snapshot, with selected-token ranks computed lazily."""

    def __init__(self, logits, config, history_token_ids):
        self.logits = _validated_logits(logits).copy()
        self.adjusted, _, _ = _history_penalty_surface(
            self.logits, config, history_token_ids
        )
        self.maximum = float(np.max(self.logits))
        self.denominator = float(np.sum(np.exp(self.logits - self.maximum)))
        self.log_z = self.maximum + float(np.log(self.denominator))
        self.policy_probabilities = _softmax(self.adjusted)
        _, scaled, stages = _stages_from_adjusted(self.adjusted, config)
        ids = stages["after_min_p"]
        assert ids is not None
        self.distribution = SparseDistribution(ids, _softmax(scaled[ids]))
        for array in (
            self.logits, self.adjusted, self.policy_probabilities,
            self.distribution.ids, self.distribution.probabilities,
        ):
            array.setflags(write=False)
        self._raw_ranks: dict[int, int] = {}
        self._policy_ranks: dict[int, int] = {}
        self._ordered: list[int] = []

    def raw_probabilities(self, token_ids):
        return np.exp(self.logits[list(token_ids)] - self.maximum) / self.denominator

    def raw_nll(self, token_id: int) -> float:
        return self.log_z - float(self.logits[token_id])

    def raw_rank(self, token_id: int) -> int:
        if token_id not in self._raw_ranks:
            self._raw_ranks[token_id] = _rank(self.logits, token_id)
        return self._raw_ranks[token_id]

    def policy_rank(self, token_id: int) -> int:
        if token_id not in self._policy_ranks:
            self._policy_ranks[token_id] = _rank(self.adjusted, token_id)
        return self._policy_ranks[token_id]

    def top_raw_ids(self, count: int) -> list[int]:
        if count > len(self._ordered):
            self._ordered = top_raw_ids(self.logits, count)
            self._raw_ranks.update(
                (token_id, rank) for rank, token_id in enumerate(self._ordered, 1)
            )
        return self._ordered[:count]

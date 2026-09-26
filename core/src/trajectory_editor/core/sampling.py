"""Dependency-light sampling kernels used by the episode editor.

This module contains the numeric sampling kernels: candidate filtering, rank
calculations, stable quantiles derived from draw coordinates, and the final token draw.
The core observer in observation.py assembles policy surfaces and controller
traces. Research-only policy surfaces live in the separate archive/research
package.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .errors import EditorError


RNG_SCHEME = "blake2b64-token-prefix-quantile-v2"
MIN_SEED = -(1 << 63)
MAX_SEED = (1 << 63) - 1


class SamplingFilterConfig(Protocol):
    """The configuration fields required by candidate filtering."""

    temperature: float
    top_k: int
    top_p: float
    min_p: float
    typical_p: float
    tail_free_z: float


@dataclass(frozen=True)
class SparseDistribution:
    ids: np.ndarray
    probabilities: np.ndarray
    scores: np.ndarray | None = None

    def probability(self, token_id: int) -> float:
        matches = np.flatnonzero(self.ids == int(token_id))
        return float(self.probabilities[matches[0]]) if len(matches) else 0.0


@dataclass(frozen=True)
class CandidateFilterResult:
    """The explicit hand-off between candidate filtering and token drawing."""

    scaled_logits: np.ndarray
    stages: dict[str, np.ndarray | None]
    diagnostics: dict[str, object] = field(default_factory=dict)


class StandardCandidateFilter:
    """Deterministic top-k/typical/tail-free/top-p/min-p filtering."""

    name = "standard"

    @staticmethod
    def _sorted(ids: np.ndarray, scaled: np.ndarray) -> np.ndarray:
        if len(ids) < 2:
            return np.asarray(ids, dtype=np.int64)
        order = np.lexsort((ids, -scaled[ids]))
        return np.asarray(ids[order], dtype=np.int64)

    @classmethod
    def _typical(
        cls,
        ids: np.ndarray,
        scaled: np.ndarray,
        typical_p: float,
    ) -> np.ndarray:
        ids = cls._sorted(ids, scaled)
        if typical_p >= 1.0 or len(ids) <= 1:
            return ids
        probabilities = _softmax(scaled[ids])
        entropy = float(
            -np.sum(probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny)))
        )
        surprisal = -np.log(np.maximum(probabilities, np.finfo(float).tiny))
        typicality = np.abs(surprisal - entropy)
        order = np.lexsort((ids, typicality))
        ordered_ids = ids[order]
        ordered_probabilities = probabilities[order]
        keep_count = int(
            np.searchsorted(np.cumsum(ordered_probabilities), typical_p, side="left")
        ) + 1
        return ordered_ids[: max(1, keep_count)]

    @classmethod
    def _tail_free(
        cls,
        ids: np.ndarray,
        scaled: np.ndarray,
        tail_free_z: float,
    ) -> np.ndarray:
        ids = cls._sorted(ids, scaled)
        if tail_free_z >= 1.0 or len(ids) < 3:
            return ids
        probabilities = _softmax(scaled[ids])
        first_derivative = np.abs(np.diff(probabilities))
        second_derivative = np.abs(np.diff(first_derivative))
        total = float(np.sum(second_derivative))
        if total <= 0.0 or not np.isfinite(total):
            return ids
        mass = second_derivative / total
        keep_count = int(np.searchsorted(np.cumsum(mass), tail_free_z, side="left")) + 2
        return ids[: max(1, min(len(ids), keep_count))]

    @classmethod
    def apply(
        cls,
        adjusted: np.ndarray,
        config: SamplingFilterConfig,
    ) -> CandidateFilterResult:
        if config.temperature == 0.0:
            greedy = _top_ids(adjusted, 1)
            stages = {
                "after_temperature": greedy,
                "after_top_k": greedy,
                "after_typical": greedy,
                "after_tail_free": greedy,
                "after_top_p": greedy,
                "after_min_p": greedy,
            }
            return CandidateFilterResult(adjusted, stages, {"filter": cls.name, "greedy": True})

        temperature = float(config.temperature)
        scaled = adjusted if temperature == 1.0 else adjusted / temperature
        if not np.all(np.isfinite(scaled)):
            raise ValueError("temperature produced non-finite scaled logits")
        after_top_k = _top_ids(scaled, min(config.top_k, len(scaled)))
        typical_p = float(config.typical_p)
        if typical_p >= 1.0:
            after_typical = after_top_k
        else:
            after_typical = cls._typical(after_top_k, scaled, typical_p)
        tail_free_z = float(config.tail_free_z)
        if typical_p >= 1.0 and tail_free_z >= 1.0:
            after_tail_free = after_top_k
        else:
            after_tail_free = cls._tail_free(after_typical, scaled, tail_free_z)
        after_top_p = after_tail_free
        if config.top_p < 1.0:
            probabilities = _softmax(scaled[after_top_p])
            keep_count = int(
                np.searchsorted(np.cumsum(probabilities), config.top_p, side="left")
            ) + 1
            after_top_p = after_top_p[: max(1, keep_count)]

        after_min_p = after_top_p
        if config.min_p > 0.0:
            probabilities = _softmax(scaled[after_top_p])
            mask = probabilities >= config.min_p * float(np.max(probabilities))
            if np.any(mask):
                after_min_p = after_top_p[mask]

        return CandidateFilterResult(
            scaled,
            {
                "after_temperature": None,
                "after_top_k": after_top_k,
                "after_typical": after_typical,
                "after_tail_free": after_tail_free,
                "after_top_p": after_top_p,
                "after_min_p": after_min_p,
            },
            {
                "filter": cls.name,
                "typical_p": float(config.typical_p),
                "tail_free_z": float(config.tail_free_z),
            },
        )


def apply_candidate_filter(
    adjusted: np.ndarray,
    config: SamplingFilterConfig,
) -> CandidateFilterResult:
    """Apply configured candidate filters before a draw kernel runs."""

    return StandardCandidateFilter.apply(np.asarray(adjusted, dtype=np.float64), config)


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
        raise ValueError("decoder logits must be a finite nonempty one-dimensional array")
    return values


def _rank(values: np.ndarray, token_id: int) -> int:
    token_id = int(token_id)
    if not 0 <= token_id < len(values):
        raise ValueError("token id is outside the decoder vocabulary")
    target = float(values[token_id])
    higher = int(np.count_nonzero(values > target))
    tied_before = int(np.count_nonzero(values[:token_id] == target))
    return 1 + higher + tied_before


def raw_rank(logits: np.ndarray, token_id: int) -> int:
    """Return the one-based full-vocabulary rank with the menu tie-break."""

    return _rank(_validated_logits(logits), token_id)


def top_raw_ids(logits: np.ndarray, count: int) -> list[int]:
    return [int(value) for value in _top_ids(np.asarray(logits), count)]


def _validate_fingerprint(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise EditorError("stream_fingerprint must be a lowercase SHA-256 hex digest")
    return value


def _validate_boundary(value: int, name: str = "boundary") -> int:
    if type(value) is not int or value < 0:
        raise EditorError(f"{name} must be a nonnegative integer")
    return value


def position_uniform(seed: int, stream_fingerprint: str, aligned_step: int) -> float:
    if type(seed) is not int or not MIN_SEED <= seed <= MAX_SEED:
        raise EditorError("seed must be a signed-64-bit integer")
    _validate_fingerprint(stream_fingerprint)
    _validate_boundary(aligned_step, "sampling boundary")
    payload = f"{RNG_SCHEME}:{seed}:{stream_fingerprint}:{aligned_step}".encode()
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return (value + 0.5) / float(1 << 64)


def position_uniform_token(
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    token_id: int,
) -> float:
    """Return a stable per-token quantile for order-independent draw kernels."""

    if type(token_id) is not int or token_id < 0:
        raise EditorError("token id must be a nonnegative integer")
    if type(seed) is not int or not MIN_SEED <= seed <= MAX_SEED:
        raise EditorError("seed must be a signed-64-bit integer")
    _validate_fingerprint(stream_fingerprint)
    _validate_boundary(aligned_step, "sampling boundary")
    payload = (
        f"{RNG_SCHEME}:gumbel-max:{seed}:{stream_fingerprint}:"
        f"{aligned_step}:{token_id}"
    ).encode()
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return (value + 0.5) / float(1 << 64)


def draw_token(
    distribution: SparseDistribution,
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    kernel: str = "categorical",
) -> int:
    """Draw one token using a replay-stable categorical or Gumbel kernel."""

    if kernel not in {"categorical", "gumbel-max"}:
        raise EditorError("unsupported draw kernel")
    if kernel == "gumbel-max":
        if distribution.scores is None:
            raise ValueError("gumbel-max requires candidate scores")
        uniforms = np.asarray(
            [
                position_uniform_token(seed, stream_fingerprint, aligned_step, int(token_id))
                for token_id in distribution.ids
            ],
            dtype=np.float64,
        )
        gumbels = -np.log(-np.log(uniforms))
        scores = np.asarray(distribution.scores, dtype=np.float64)
        if scores.shape != distribution.ids.shape:
            raise ValueError("candidate scores do not match candidate IDs")
        ranking = scores + gumbels
        best = np.flatnonzero(ranking == np.max(ranking))
        index = int(best[np.argmin(distribution.ids[best])])
        return int(distribution.ids[index])
    draw = position_uniform(seed, stream_fingerprint, aligned_step)
    index = int(np.searchsorted(np.cumsum(distribution.probabilities), draw, side="right"))
    return int(distribution.ids[min(index, len(distribution.ids) - 1)])


__all__ = [
    "CandidateFilterResult",
    "MAX_SEED",
    "MIN_SEED",
    "RNG_SCHEME",
    "SparseDistribution",
    "StandardCandidateFilter",
    "_rank",
    "_softmax",
    "_top_ids",
    "_validated_logits",
    "apply_candidate_filter",
    "draw_token",
    "position_uniform",
    "position_uniform_token",
    "raw_rank",
    "top_raw_ids",
]

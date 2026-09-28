"""Dependency-light sampling kernels used by the episode editor.

This module contains the numeric sampling kernels: candidate filtering, rank
calculations, stable quantiles derived from draw coordinates, and the final token draw.
``policy_calculations.py`` combines these kernels with the active policy adjustments.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .errors import EditorError


RNG_SCHEME = "blake2b64-token-prefix-quantile-v2"
GUMBEL_NOISE_ADDRESSES = ("token-id", "model-rank")
MIN_SEED = -(1 << 63)
MAX_SEED = (1 << 63) - 1


class SamplingFilterConfig(Protocol):
    """The configuration fields required by candidate filtering."""

    temperature: float
    top_k: int | None
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
        if (
            temperature == 1.0
            and config.top_k is None
            and config.typical_p == 1.0
            and config.tail_free_z == 1.0
            and config.top_p == 1.0
            and config.min_p == 0.0
        ):
            all_ids = np.arange(len(scaled), dtype=np.int64)
            return CandidateFilterResult(
                scaled,
                {
                    "after_temperature": None,
                    "after_top_k": all_ids,
                    "after_typical": all_ids,
                    "after_tail_free": all_ids,
                    "after_top_p": all_ids,
                    "after_min_p": all_ids,
                },
                {"filter": cls.name, "unfiltered": True},
            )

        after_top_k = (
            _top_ids(scaled, min(config.top_k, len(scaled)))
            if config.top_k is not None
            else np.arange(len(scaled), dtype=np.int64)
        )
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


def position_uniform_model_rank(
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    model_rank: int,
) -> float:
    """Return a stable Gumbel quantile addressed by one-based model rank."""

    if type(model_rank) is not int or model_rank < 1:
        raise EditorError("model rank must be a positive integer")
    if type(seed) is not int or not MIN_SEED <= seed <= MAX_SEED:
        raise EditorError("seed must be a signed-64-bit integer")
    _validate_fingerprint(stream_fingerprint)
    _validate_boundary(aligned_step, "sampling boundary")
    payload = (
        f"{RNG_SCHEME}:gumbel-max:model-rank:{seed}:{stream_fingerprint}:"
        f"{aligned_step}:{model_rank}"
    ).encode()
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return (value + 0.5) / float(1 << 64)


def conditional_gumbel_top_k(
    log_probabilities: np.ndarray,
    *,
    count: int,
    parent_score: float,
    parent_log_probability: float,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    prefix_token_ids: Sequence[int],
) -> tuple[tuple[int, float], ...]:
    """Return top child Gumbel scores conditioned on their maximum.

    This is the top-down Gumbel split used by stochastic beam search.  In
    exponential-race coordinates, the winning child is sampled from the
    conditional distribution and every later child arrives after an
    exponential waiting time over the remaining child mass.  The returned
    scores are therefore the highest ``count`` child scores conditioned on
    their maximum being exactly ``parent_score``.
    """

    values = np.asarray(log_probabilities, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("conditional Gumbel sampling requires finite log probabilities")
    if type(count) is not int or count < 1:
        raise EditorError("conditional Gumbel sample count must be a positive integer")
    if (
        type(parent_score) not in {int, float}
        or not math.isfinite(float(parent_score))
        or type(parent_log_probability) not in {int, float}
        or not math.isfinite(float(parent_log_probability))
    ):
        raise EditorError("conditional Gumbel parent scores must be finite")
    if type(seed) is not int or not MIN_SEED <= seed <= MAX_SEED:
        raise EditorError("seed must be a signed-64-bit integer")
    _validate_fingerprint(stream_fingerprint)
    _validate_boundary(aligned_step, "sampling boundary")
    if isinstance(prefix_token_ids, (str, bytes)):
        raise EditorError("stochastic beam prefixes must be token ID sequences")
    prefix = tuple(prefix_token_ids)
    if any(
        type(token_id) is not int
        or token_id < 0
        or token_id >= (1 << 64)
        for token_id in prefix
    ):
        raise EditorError("stochastic beam prefixes must contain nonnegative token IDs")

    log_normalizer = float(np.logaddexp.reduce(values))
    normalized = values - log_normalizer
    weights = np.exp(normalized)
    total_weight = float(np.sum(weights))
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("conditional Gumbel child mass is invalid")
    weights /= total_weight

    prefix_bytes = len(prefix).to_bytes(8, "big") + b"".join(
        token_id.to_bytes(8, "big") for token_id in prefix
    )
    address = (
        f"{RNG_SCHEME}:stochastic-beam-gumbel-top-k-v1:"
        f"{seed}:{stream_fingerprint}:{aligned_step}:"
    ).encode() + prefix_bytes
    parent_key = hashlib.blake2b(address, digest_size=16).digest()

    def uniform(lane: bytes, ordinal: int) -> float:
        payload = (
            b"stochastic-beam-child-v1:" + parent_key + b":" + lane + b":"
            + ordinal.to_bytes(8, "big")
        )
        value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
        return (value + 0.5) / float(1 << 64)

    def choose(remaining: np.ndarray, remaining_mass: float, quantile: float) -> int:
        target = quantile * remaining_mass
        index = int(np.searchsorted(np.cumsum(remaining), target, side="right"))
        if index >= len(remaining):
            index = int(np.flatnonzero(remaining > 0.0)[-1])
        return index

    remaining = weights.copy()
    remaining_mass = float(np.sum(remaining))
    winner = choose(remaining, remaining_mass, uniform(b"winner", 0))
    result: list[tuple[int, float]] = [(winner, float(parent_score))]
    remaining[winner] = 0.0

    for ordinal in range(1, min(count, len(values))):
        remaining_mass = float(np.sum(remaining))
        if remaining_mass <= 0.0:
            break
        log_rate = float(parent_log_probability) + math.log(remaining_mass)
        exponential_wait = -math.log(uniform(b"wait", ordinal))
        log_wait = math.log(exponential_wait) - log_rate
        log_arrival = float(np.logaddexp(-float(parent_score), log_wait))
        child_score = -log_arrival
        token_id = choose(
            remaining, remaining_mass, uniform(b"winner", ordinal)
        )
        result.append((token_id, child_score))
        remaining[token_id] = 0.0

    return tuple(result)


def gaussian_ranking_scores(
    distribution: SparseDistribution,
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    noise_std: float = 1.0,
) -> np.ndarray:
    """Add replay-stable, per-token Gaussian noise to the active scores."""

    if distribution.scores is None:
        raise ValueError("gaussian-max requires candidate scores")
    scores = np.asarray(distribution.scores, dtype=np.float64)
    ids = np.asarray(distribution.ids, dtype=np.int64)
    if scores.shape != ids.shape:
        raise ValueError("candidate scores do not match candidate IDs")
    if not len(ids):
        raise ValueError("gaussian-max requires at least one candidate")
    if type(noise_std) not in {int, float} or not math.isfinite(float(noise_std)):
        raise EditorError("gaussian_noise_std must be finite and nonnegative")
    if noise_std < 0.0:
        raise EditorError("gaussian_noise_std must be finite and nonnegative")
    if type(seed) is not int or not MIN_SEED <= seed <= MAX_SEED:
        raise EditorError("seed must be a signed-64-bit integer")
    _validate_fingerprint(stream_fingerprint)
    _validate_boundary(aligned_step, "sampling boundary")
    if noise_std == 0.0:
        return scores.copy()

    perturbations = np.empty(len(ids), dtype=np.float64)
    for index, token_id in enumerate(ids):
        prefix = (
            f"{RNG_SCHEME}:gaussian-max:{seed}:{stream_fingerprint}:"
            f"{aligned_step}:{int(token_id)}:"
        )
        uniforms = []
        for lane in (0, 1):
            payload = f"{prefix}{lane}".encode()
            value = int.from_bytes(
                hashlib.blake2b(payload, digest_size=8).digest(), "big"
            )
            uniforms.append((value + 0.5) / float(1 << 64))
        radius = math.sqrt(-2.0 * math.log(uniforms[0]))
        perturbations[index] = radius * math.cos(2.0 * math.pi * uniforms[1])
    ranking = scores + float(noise_std) * perturbations
    if not np.all(np.isfinite(ranking)):
        raise ValueError("gaussian noise produced non-finite ranking scores")
    return ranking


def draw_token(
    distribution: SparseDistribution,
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    kernel: str = "categorical",
    gaussian_noise_std: float = 1.0,
    gumbel_noise_address: str = "token-id",
    candidate_model_ranks: np.ndarray | None = None,
    gumbel_noise_scale: float = 1.0,
) -> int:
    """Draw one token using a replay-stable categorical, Gumbel, or Gaussian kernel."""

    if kernel not in {"categorical", "gumbel-max", "gaussian-max"}:
        raise EditorError("unsupported draw kernel")
    if kernel == "gumbel-max":
        scores = gumbel_ranking_scores(
            distribution,
            seed=seed,
            stream_fingerprint=stream_fingerprint,
            aligned_step=aligned_step,
            noise_address=gumbel_noise_address,
            candidate_model_ranks=candidate_model_ranks,
            gumbel_noise_scale=gumbel_noise_scale,
        )
        return gumbel_winner(distribution, scores)
    if kernel == "gaussian-max":
        scores = gaussian_ranking_scores(
            distribution,
            seed=seed,
            stream_fingerprint=stream_fingerprint,
            aligned_step=aligned_step,
            noise_std=gaussian_noise_std,
        )
        return gaussian_winner(distribution, scores)
    draw = position_uniform(seed, stream_fingerprint, aligned_step)
    index = int(np.searchsorted(np.cumsum(distribution.probabilities), draw, side="right"))
    return int(distribution.ids[min(index, len(distribution.ids) - 1)])


def gumbel_ranking_scores(
    distribution: SparseDistribution,
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    noise_address: str = "token-id",
    candidate_model_ranks: np.ndarray | None = None,
    gumbel_noise_scale: float = 1.0,
) -> np.ndarray:
    """Return effective scores plus the replay-stable Gumbel perturbations."""

    if distribution.scores is None:
        raise ValueError("gumbel-max requires candidate scores")
    scores = np.asarray(distribution.scores, dtype=np.float64)
    ids = np.asarray(distribution.ids, dtype=np.int64)
    if scores.shape != ids.shape:
        raise ValueError("candidate scores do not match candidate IDs")
    if not len(ids):
        raise ValueError("gumbel-max requires at least one candidate")
    if (
        type(gumbel_noise_scale) not in {int, float}
        or not math.isfinite(float(gumbel_noise_scale))
    ):
        raise EditorError("gumbel_noise_scale must be finite and nonnegative")
    if gumbel_noise_scale < 0.0:
        raise EditorError("gumbel_noise_scale must be finite and nonnegative")
    if noise_address not in GUMBEL_NOISE_ADDRESSES:
        raise EditorError("gumbel_noise_address must be token-id or model-rank")
    if gumbel_noise_scale == 0.0:
        return scores.copy()
    if noise_address == "model-rank":
        if candidate_model_ranks is None:
            raise ValueError("model-rank Gumbel noise requires candidate model ranks")
        ranks = np.asarray(candidate_model_ranks)
        if (
            ranks.shape != ids.shape
            or not np.issubdtype(ranks.dtype, np.integer)
            or np.any(ranks < 1)
            or len(np.unique(ranks)) != len(ranks)
        ):
            raise ValueError("candidate model ranks must be distinct positive integers")
        uniforms = np.asarray(
            [
                position_uniform_model_rank(
                    seed, stream_fingerprint, aligned_step, int(rank)
                )
                for rank in ranks
            ],
            dtype=np.float64,
        )
    else:
        uniforms = np.asarray(
            [
                position_uniform_token(seed, stream_fingerprint, aligned_step, int(token_id))
                for token_id in ids
            ],
            dtype=np.float64,
        )
    if gumbel_noise_scale == 1.0:
        return scores - np.log(-np.log(uniforms))
    ranking = scores - float(gumbel_noise_scale) * np.log(-np.log(uniforms))
    if not np.all(np.isfinite(ranking)):
        raise ValueError("Gumbel noise produced non-finite ranking scores")
    return ranking


def gumbel_winner(
    distribution: SparseDistribution, ranking_scores: np.ndarray
) -> int:
    """Return the maximum Gumbel score, breaking exact ties by token ID."""

    ids = np.asarray(distribution.ids, dtype=np.int64)
    scores = np.asarray(ranking_scores, dtype=np.float64)
    if scores.shape != ids.shape:
        raise ValueError("Gumbel scores do not match candidate IDs")
    if not len(ids):
        raise ValueError("gumbel-max requires at least one candidate")
    best = np.flatnonzero(scores == np.max(scores))
    return int(ids[best[np.argmin(ids[best])]])


def gaussian_winner(
    distribution: SparseDistribution, ranking_scores: np.ndarray
) -> int:
    """Return the maximum Gaussian-perturbed score, breaking ties by token ID."""

    ids = np.asarray(distribution.ids, dtype=np.int64)
    scores = np.asarray(ranking_scores, dtype=np.float64)
    if scores.shape != ids.shape:
        raise ValueError("Gaussian scores do not match candidate IDs")
    if not len(ids):
        raise ValueError("gaussian-max requires at least one candidate")
    best = np.flatnonzero(scores == np.max(scores))
    return int(ids[best[np.argmin(ids[best])]])


def gumbel_ranked_ids(
    distribution: SparseDistribution,
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    noise_address: str = "token-id",
    candidate_model_ranks: np.ndarray | None = None,
    gumbel_noise_scale: float = 1.0,
) -> np.ndarray:
    """Return eligible token IDs in deterministic Gumbel-Max order."""

    ids = np.asarray(distribution.ids, dtype=np.int64)
    ranking = gumbel_ranking_scores(
        distribution,
        seed=seed,
        stream_fingerprint=stream_fingerprint,
        aligned_step=aligned_step,
        noise_address=noise_address,
        candidate_model_ranks=candidate_model_ranks,
        gumbel_noise_scale=gumbel_noise_scale,
    )
    order = np.lexsort((ids, -ranking))
    return ids[order]


def find_seed_for_token(
    distribution: SparseDistribution,
    token_id: int,
    *,
    current_seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    kernel: str,
    gaussian_noise_std: float = 1.0,
    gumbel_noise_address: str = "token-id",
    candidate_model_ranks: np.ndarray | None = None,
    gumbel_noise_scale: float = 1.0,
    next_seed: Callable[[], int],
) -> tuple[int, int]:
    """Find a fresh seed whose normal draw selects an eligible token.

    Returns the seed and the number of candidate seeds checked.  The caller
    supplies seed generation so this numeric kernel stays independent of
    runtime configuration and can be exercised with deterministic sequences.
    """

    if type(token_id) is not int or token_id < 0:
        raise EditorError("draw token id must be a nonnegative integer")
    if not np.any(distribution.ids == token_id):
        raise EditorError(
            f"token {token_id} is outside the active truncated candidate set"
        )
    if kernel == "categorical" and distribution.probability(token_id) <= 0.0:
        raise EditorError(f"token {token_id} has no selectable categorical mass")
    if kernel == "gaussian-max" and gaussian_noise_std == 0.0:
        proposal = draw_token(
            distribution,
            seed=current_seed,
            stream_fingerprint=stream_fingerprint,
            aligned_step=aligned_step,
            kernel=kernel,
            gaussian_noise_std=0.0,
            gumbel_noise_address=gumbel_noise_address,
            candidate_model_ranks=candidate_model_ranks,
        )
        if proposal != token_id:
            raise EditorError(
                "gaussian_noise_std=0 makes this token impossible to select"
            )
    if kernel == "gumbel-max" and gumbel_noise_scale == 0.0:
        proposal = draw_token(
            distribution,
            seed=current_seed,
            stream_fingerprint=stream_fingerprint,
            aligned_step=aligned_step,
            kernel=kernel,
            gumbel_noise_address=gumbel_noise_address,
            gumbel_noise_scale=0.0,
        )
        if proposal != token_id:
            raise EditorError(
                "gumbel_noise_scale=0 makes this token impossible to select"
            )

    checked = 0
    while True:
        seed = next_seed()
        checked += 1
        if type(seed) is not int or not MIN_SEED <= seed <= MAX_SEED:
            raise EditorError("draw seed generator returned an invalid signed-64-bit seed")
        if seed == current_seed:
            continue
        if draw_token(
            distribution,
            seed=seed,
            stream_fingerprint=stream_fingerprint,
            aligned_step=aligned_step,
            kernel=kernel,
            gaussian_noise_std=gaussian_noise_std,
            gumbel_noise_address=gumbel_noise_address,
            candidate_model_ranks=candidate_model_ranks,
            gumbel_noise_scale=gumbel_noise_scale,
        ) == token_id:
            return seed, checked


__all__ = [
    "CandidateFilterResult",
    "GUMBEL_NOISE_ADDRESSES",
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
    "gaussian_ranking_scores",
    "gaussian_winner",
    "find_seed_for_token",
    "gumbel_ranking_scores",
    "gumbel_ranked_ids",
    "gumbel_winner",
    "position_uniform",
    "position_uniform_model_rank",
    "position_uniform_token",
    "raw_rank",
    "top_raw_ids",
]

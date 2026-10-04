"""Dependency-light sampling kernels used by the episode editor.

This module contains the numeric sampling kernels: candidate filtering, rank
calculations, stable quantiles derived from draw coordinates, and final argmax selection. Noise quantiles are not token CDF draws.
``policy_calculations.py`` combines these kernels with the active policy adjustments.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol

import numpy as np

from .errors import EditorError


RNG_SCHEME = "blake2b64-token-prefix-quantile-v2"
GUMBEL_NOISE_ADDRESSES = ("token-id", "model-rank")
PERTURB_MAX_KERNELS = (
    "logistic-max",
    "student-t-max",
    "laplace-max",
    "uniform-max",
)
DRAW_KERNELS = ("argmax", "gumbel-max", "gaussian-max", *PERTURB_MAX_KERNELS)
MIN_SEED = -(1 << 63)
MAX_SEED = (1 << 63) - 1


class SamplingFilterConfig(Protocol):
    """The configuration fields required by candidate filtering."""

    temperature: float
    top_k: int | None
    min_p: float


@dataclass(frozen=True)
class EligibleScores:
    """Eligible IDs and required pre-noise scores; softmax is diagnostic only."""

    ids: np.ndarray
    scores: np.ndarray

    @cached_property
    def softmax(self) -> np.ndarray:
        values = _softmax(self.scores)
        values.setflags(write=False)
        return values

    @cached_property
    def eligible_ids(self) -> frozenset[int]:
        return frozenset(int(token_id) for token_id in self.ids)

    def is_eligible(self, token_id: int) -> bool:
        return int(token_id) in self.eligible_ids

    def softmax_at(self, token_id: int) -> float:
        if not self.is_eligible(token_id):
            return 0.0
        index = int(np.flatnonzero(self.ids == int(token_id))[0])
        return float(self.softmax[index])


@dataclass(frozen=True)
class CandidateFilterResult:
    """The explicit hand-off between candidate filtering and token drawing."""

    scaled_logits: np.ndarray
    stages: dict[str, np.ndarray | None]
    diagnostics: dict[str, object] = field(default_factory=dict)


class StandardCandidateFilter:
    """Logit-only eligibility: temperature, top-k, and min-p logit gap."""

    name = "standard"

    @classmethod
    def apply(cls, adjusted: np.ndarray, config: SamplingFilterConfig) -> CandidateFilterResult:
        if config.temperature == 0.0:
            greedy = _top_ids(adjusted, 1)
            return CandidateFilterResult(adjusted, {
                "after_temperature": greedy, "after_top_k": greedy, "after_min_p": greedy,
            }, {"filter": cls.name, "greedy": True})
        scaled = adjusted if config.temperature == 1.0 else adjusted / float(config.temperature)
        if not np.all(np.isfinite(scaled)):
            raise ValueError("temperature produced non-finite scaled logits")
        ids = (_top_ids(scaled, min(config.top_k, len(scaled)))
               if config.top_k is not None else np.arange(len(scaled), dtype=np.int64))
        eligible = ids
        if config.min_p > 0.0:
            cutoff = float(np.max(scaled[ids])) + math.log(float(config.min_p))
            eligible = ids[scaled[ids] >= cutoff]
        return CandidateFilterResult(scaled, {
            "after_temperature": None, "after_top_k": ids, "after_min_p": eligible,
        }, {"filter": cls.name, "min_logit_gap": (
            -math.log(float(config.min_p)) if config.min_p > 0.0 else None
        )})


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


def _noise_indices(distribution: EligibleScores, count: int | None) -> np.ndarray:
    """Select top eligible pre-noise scores, with token-ID tie breaking."""
    if count is not None and (type(count) is not int or count < 1):
        raise EditorError("selective_noise_k must be a positive integer or null")
    ids = np.asarray(distribution.ids, dtype=np.int64)
    if distribution.scores is None:
        raise ValueError("perturb-and-argmax requires candidate scores")
    scores = np.asarray(distribution.scores, dtype=np.float64)
    if scores.shape != ids.shape or not len(ids) or not np.all(np.isfinite(scores)):
        raise ValueError("candidate scores must be finite and match nonempty candidate IDs")
    if count is None or count >= len(ids):
        return np.arange(len(ids))
    return np.lexsort((ids, -scores))[:count]


def _noise_subset(distribution: EligibleScores, indices: np.ndarray) -> EligibleScores:
    return EligibleScores(
        distribution.ids[indices], distribution.scores[indices],
    )


def gaussian_ranking_scores(
    distribution: EligibleScores,
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    noise_std: float = 1.0,
    selective_noise_k: int | None = None,
) -> np.ndarray:
    """Add replay-stable, per-token Gaussian noise to the active scores."""

    indices = _noise_indices(distribution, selective_noise_k)
    if len(indices) < len(distribution.ids):
        ranking = np.asarray(distribution.scores, dtype=np.float64).copy()
        ranking[indices] = gaussian_ranking_scores(
            _noise_subset(distribution, indices),
            seed=seed, stream_fingerprint=stream_fingerprint, aligned_step=aligned_step,
            noise_std=noise_std,
        )
        return ranking

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


def _perturbation_uniform(
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    token_id: int,
    lane: int,
) -> float:
    payload = (
        f"{RNG_SCHEME}:perturb-max:{seed}:{stream_fingerprint}:"
        f"{aligned_step}:{token_id}:{lane}"
    ).encode()
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    # 52-bit midpoints stay strictly inside (0, 1) in binary64.
    return ((value >> 12) + 0.5) / float(1 << 52)


def _normal_from_uniforms(first: float, second: float) -> float:
    radius = math.sqrt(-2.0 * math.log(first))
    return radius * math.cos(2.0 * math.pi * second)


def _log_gamma_ge_one(
    shape: float,
    uniform: Callable[[], float],
    normal: Callable[[], float],
) -> float:
    """Sample log Gamma(shape, 1) with Marsaglia-Tsang rejection sampling."""

    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    for _ in range(128):
        value = normal()
        base = 1.0 + c * value
        if base <= 0.0:
            continue
        cube = base * base * base
        draw = uniform()
        if draw < 1.0 - 0.0331 * value**4 or math.log(draw) < (
            0.5 * value * value + d * (1.0 - cube + math.log(cube))
        ):
            return math.log(d) + math.log(cube)
    raise ValueError("Student-t gamma draw did not converge")


def _log_gamma(
    shape: float,
    uniform: Callable[[], float],
    normal: Callable[[], float],
) -> float:
    """Sample log Gamma(shape, 1), including shapes in (0, 1)."""

    if shape >= 1.0:
        return _log_gamma_ge_one(shape, uniform, normal)
    if shape <= 0.0:
        # A positive df can underflow when halved only at the smallest
        # representable inputs; its gamma variate is then below float range.
        return -math.inf
    boosted = _log_gamma_ge_one(shape + 1.0, uniform, normal)
    try:
        return boosted + math.log(uniform()) / shape
    except OverflowError:
        return -math.inf


def _student_t_unit_noise(
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    token_id: int,
    degrees_of_freedom: float,
) -> float:
    """Draw t_df / sqrt(3), retaining the historical t3 stream exactly."""

    def uniform_at(lane: int) -> float:
        return _perturbation_uniform(
            seed=seed,
            stream_fingerprint=stream_fingerprint,
            aligned_step=aligned_step,
            token_id=token_id,
            lane=lane,
        )

    if degrees_of_freedom == 3.0:
        # Keep the original four-lane Box-Muller construction byte-for-byte:
        # existing df=3 episodes therefore replay the same perturbations.
        normals = []
        for offset in (0, 2):
            radius = math.sqrt(-2.0 * math.log(uniform_at(offset)))
            angle = 2.0 * math.pi * uniform_at(offset + 1)
            normals.extend((radius * math.cos(angle), radius * math.sin(angle)))
        return normals[0] / math.sqrt(
            normals[1] ** 2 + normals[2] ** 2 + normals[3] ** 2
        )

    lane = 0

    def next_uniform() -> float:
        nonlocal lane
        result = uniform_at(lane)
        lane += 1
        return result

    def next_normal() -> float:
        return _normal_from_uniforms(next_uniform(), next_uniform())

    normal = next_normal()
    log_gamma = _log_gamma(degrees_of_freedom / 2.0, next_uniform, next_normal)
    if normal == 0.0:
        return 0.0
    log_magnitude = (
        math.log(abs(normal))
        - 0.5 * (math.log(2.0) + log_gamma - math.log(degrees_of_freedom))
        - 0.5 * math.log(3.0)
    )
    if log_magnitude > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        return math.copysign(math.inf, normal)
    if log_magnitude < math.log(float.fromhex("0x0.0000000000001p-1022")):
        return math.copysign(0.0, normal)
    return math.copysign(math.exp(log_magnitude), normal)


def _validated_student_t_df(value: float) -> float:
    if type(value) not in {int, float}:
        raise EditorError("student_t_df must be finite and greater than 0")
    try:
        degrees_of_freedom = float(value)
    except OverflowError as exc:
        raise EditorError("student_t_df must be finite and greater than 0") from exc
    if not math.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0.0:
        raise EditorError("student_t_df must be finite and greater than 0")
    return degrees_of_freedom


def perturbation_ranking_scores(
    distribution: EligibleScores,
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    kernel: str,
    noise_std: float = 1.0,
    student_t_df: float = 3.0,
    selective_noise_k: int | None = None,
) -> np.ndarray:
    """Add replay-stable perturbations to the active candidate scores."""

    indices = _noise_indices(distribution, selective_noise_k)
    if len(indices) < len(distribution.ids):
        ranking = np.asarray(distribution.scores, dtype=np.float64).copy()
        ranking[indices] = perturbation_ranking_scores(
            _noise_subset(distribution, indices),
            seed=seed, stream_fingerprint=stream_fingerprint, aligned_step=aligned_step,
            kernel=kernel, noise_std=noise_std, student_t_df=student_t_df,
        )
        return ranking

    if kernel not in PERTURB_MAX_KERNELS:
        raise EditorError("unsupported perturb-and-argmax kernel")
    if kernel == "student-t-max":
        student_t_df = _validated_student_t_df(student_t_df)
    if distribution.scores is None:
        raise ValueError(f"{kernel} requires candidate scores")
    scores = np.asarray(distribution.scores, dtype=np.float64)
    ids = np.asarray(distribution.ids, dtype=np.int64)
    if scores.shape != ids.shape:
        raise ValueError("candidate scores do not match candidate IDs")
    if not len(ids):
        raise ValueError(f"{kernel} requires at least one candidate")
    if type(noise_std) not in {int, float} or not math.isfinite(float(noise_std)):
        raise EditorError("perturb_noise_std must be finite and nonnegative")
    if noise_std < 0.0:
        raise EditorError("perturb_noise_std must be finite and nonnegative")
    if type(seed) is not int or not MIN_SEED <= seed <= MAX_SEED:
        raise EditorError("seed must be a signed-64-bit integer")
    _validate_fingerprint(stream_fingerprint)
    _validate_boundary(aligned_step, "sampling boundary")
    if noise_std == 0.0:
        return scores.copy()

    perturbations = np.empty(len(ids), dtype=np.float64)
    for index, token_id in enumerate(ids):
        if kernel == "student-t-max":
            unit_noise = _student_t_unit_noise(
                seed=seed,
                stream_fingerprint=stream_fingerprint,
                aligned_step=aligned_step,
                token_id=int(token_id),
                degrees_of_freedom=float(student_t_df),
            )
        else:
            uniform = _perturbation_uniform(
                seed=seed,
                stream_fingerprint=stream_fingerprint,
                aligned_step=aligned_step,
                token_id=int(token_id),
                lane=0,
            )
            if kernel == "logistic-max":
                unit_noise = (math.sqrt(3.0) / math.pi) * (
                    math.log(uniform) - math.log1p(-uniform)
                )
            elif kernel == "laplace-max":
                unit_noise = (
                    math.log(2.0 * uniform)
                    if uniform < 0.5
                    else -math.log(2.0 * (1.0 - uniform))
                ) / math.sqrt(2.0)
            else:
                unit_noise = math.sqrt(3.0) * (2.0 * uniform - 1.0)
        perturbations[index] = float(noise_std) * unit_noise

    ranking = scores + perturbations
    if not np.all(np.isfinite(ranking)):
        if kernel == "student-t-max":
            raise ValueError(
                f"student-t-max at df={float(student_t_df):g} produced "
                "non-finite ranking scores"
            )
        raise ValueError(f"{kernel} noise produced non-finite ranking scores")
    return ranking


def unified_ranking_scores(
    distribution: EligibleScores,
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    kernel: str = "argmax",
    gaussian_noise_std: float = 1.0,
    perturb_noise_std: float = 1.0,
    student_t_df: float = 3.0,
    gumbel_noise_address: str = "token-id",
    candidate_model_ranks: np.ndarray | None = None,
    gumbel_noise_scale: float = 1.0,
    selective_noise_k: int | None = None,
) -> np.ndarray:
    """Return eligible scores with the selected replay-stable noise applied."""

    if kernel not in DRAW_KERNELS:
        raise EditorError("unsupported draw kernel")
    if kernel == "gumbel-max":
        scores = gumbel_ranking_scores(
            distribution,
            seed=seed,
            stream_fingerprint=stream_fingerprint,
            aligned_step=aligned_step,
            selective_noise_k=selective_noise_k,
            noise_address=gumbel_noise_address,
            candidate_model_ranks=candidate_model_ranks,
            gumbel_noise_scale=gumbel_noise_scale,
        )
        return scores
    if kernel == "gaussian-max":
        scores = gaussian_ranking_scores(
            distribution,
            seed=seed,
            stream_fingerprint=stream_fingerprint,
            aligned_step=aligned_step,
            selective_noise_k=selective_noise_k,
            noise_std=gaussian_noise_std,
        )
        return scores
    if kernel in PERTURB_MAX_KERNELS:
        scores = perturbation_ranking_scores(
            distribution,
            seed=seed,
            stream_fingerprint=stream_fingerprint,
            aligned_step=aligned_step,
            selective_noise_k=selective_noise_k,
            kernel=kernel,
            noise_std=perturb_noise_std,
            student_t_df=student_t_df,
        )
        return scores
    if distribution.scores is None:
        raise ValueError("argmax requires candidate scores")
    return distribution.scores


def draw_token(
    distribution: EligibleScores,
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    kernel: str = "argmax",
    gaussian_noise_std: float = 1.0,
    perturb_noise_std: float = 1.0,
    student_t_df: float = 3.0,
    gumbel_noise_address: str = "token-id",
    candidate_model_ranks: np.ndarray | None = None,
    gumbel_noise_scale: float = 1.0,
    selective_noise_k: int | None = None,
) -> int:
    """Select the argmax of the unified ranking scores."""

    scores = unified_ranking_scores(
        distribution,
        seed=seed,
        stream_fingerprint=stream_fingerprint,
        aligned_step=aligned_step,
        kernel=kernel,
        gaussian_noise_std=gaussian_noise_std,
        perturb_noise_std=perturb_noise_std,
        student_t_df=student_t_df,
        gumbel_noise_address=gumbel_noise_address,
        candidate_model_ranks=candidate_model_ranks,
        gumbel_noise_scale=gumbel_noise_scale,
        selective_noise_k=selective_noise_k,
    )
    return argmax_winner(distribution, scores)


def gumbel_ranking_scores(
    distribution: EligibleScores,
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    noise_address: str = "token-id",
    candidate_model_ranks: np.ndarray | None = None,
    gumbel_noise_scale: float = 1.0,
    selective_noise_k: int | None = None,
) -> np.ndarray:
    """Return effective scores plus the replay-stable Gumbel perturbations."""

    indices = _noise_indices(distribution, selective_noise_k)
    if len(indices) < len(distribution.ids):
        ranking = np.asarray(distribution.scores, dtype=np.float64).copy()
        ranking[indices] = gumbel_ranking_scores(
            _noise_subset(distribution, indices),
            seed=seed, stream_fingerprint=stream_fingerprint, aligned_step=aligned_step,
            noise_address=noise_address, gumbel_noise_scale=gumbel_noise_scale,
            candidate_model_ranks=(None if candidate_model_ranks is None
                                   else np.asarray(candidate_model_ranks)[indices]),
        )
        return ranking

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


def argmax_winner(
    distribution: EligibleScores, ranking_scores: np.ndarray
) -> int:
    """Return the maximum ranking score, breaking exact ties by token ID."""

    ids = np.asarray(distribution.ids, dtype=np.int64)
    scores = np.asarray(ranking_scores, dtype=np.float64)
    if scores.shape != ids.shape:
        raise ValueError("ranking scores do not match candidate IDs")
    if not len(ids):
        raise ValueError("argmax requires at least one candidate")
    best = np.flatnonzero(scores == np.max(scores))
    return int(ids[best[np.argmin(ids[best])]])


def gumbel_ranked_ids(
    distribution: EligibleScores,
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    noise_address: str = "token-id",
    candidate_model_ranks: np.ndarray | None = None,
    gumbel_noise_scale: float = 1.0,
    selective_noise_k: int | None = None,
) -> np.ndarray:
    """Return eligible token IDs in deterministic Gumbel-Max order."""

    ids = np.asarray(distribution.ids, dtype=np.int64)
    ranking = gumbel_ranking_scores(
        distribution,
        seed=seed,
        stream_fingerprint=stream_fingerprint,
        aligned_step=aligned_step,
        selective_noise_k=selective_noise_k,
        noise_address=noise_address,
        candidate_model_ranks=candidate_model_ranks,
        gumbel_noise_scale=gumbel_noise_scale,
    )
    order = np.lexsort((ids, -ranking))
    return ids[order]


def find_seed_for_token(
    distribution: EligibleScores,
    token_id: int,
    *,
    current_seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    kernel: str,
    gaussian_noise_std: float = 1.0,
    perturb_noise_std: float = 1.0,
    student_t_df: float = 3.0,
    gumbel_noise_address: str = "token-id",
    candidate_model_ranks: np.ndarray | None = None,
    gumbel_noise_scale: float = 1.0,
    next_seed: Callable[[], int],
    selective_noise_k: int | None = None,
) -> tuple[int, int]:
    """Find a fresh seed whose configured draw selects an eligible token.

    Returns the seed and the number of candidate seeds checked.  The caller
    supplies seed generation so this numeric kernel stays independent of
    runtime configuration and can be exercised with deterministic sequences.
    """

    if type(token_id) is not int or token_id < 0:
        raise EditorError("draw token id must be a nonnegative integer")
    if not distribution.is_eligible(token_id):
        raise EditorError(
            f"token {token_id} is outside the active truncated candidate set"
        )
    if selective_noise_k is not None:
        indices = _noise_indices(distribution, selective_noise_k)
        untouched = np.ones(len(distribution.ids), dtype=bool)
        untouched[indices] = False
        target_index = int(np.flatnonzero(distribution.ids == token_id)[0])
        if untouched[target_index]:
            untouched_ids = distribution.ids[untouched]
            untouched_scores = distribution.scores[untouched]
            best = np.lexsort((untouched_ids, -untouched_scores))[0]
            if int(untouched_ids[best]) != token_id:
                raise EditorError("an unchanged higher-ranked candidate makes this token impossible to select")
    if kernel == "argmax":
        raise EditorError("argmax is independent of seed; choose a noise kernel to search seeds")
    if kernel in PERTURB_MAX_KERNELS and (
        type(perturb_noise_std) not in {int, float}
        or not math.isfinite(float(perturb_noise_std))
        or perturb_noise_std < 0.0
    ):
        raise EditorError("perturb_noise_std must be finite and nonnegative")
    if kernel == "student-t-max":
        student_t_df = _validated_student_t_df(student_t_df)
    if kernel == "gaussian-max" and gaussian_noise_std == 0.0:
        proposal = draw_token(
            distribution,
            seed=current_seed,
            stream_fingerprint=stream_fingerprint,
            aligned_step=aligned_step,
            selective_noise_k=selective_noise_k,
            kernel=kernel,
            gaussian_noise_std=0.0,
            gumbel_noise_address=gumbel_noise_address,
            candidate_model_ranks=candidate_model_ranks,
        )
        if proposal != token_id:
            raise EditorError(
                "gaussian_noise_std=0 makes this token impossible to select"
            )
    if kernel in PERTURB_MAX_KERNELS and perturb_noise_std == 0.0:
        proposal = draw_token(
            distribution,
            seed=current_seed,
            stream_fingerprint=stream_fingerprint,
            aligned_step=aligned_step,
            selective_noise_k=selective_noise_k,
            kernel=kernel,
            perturb_noise_std=0.0,
            student_t_df=student_t_df,
        )
        if proposal != token_id:
            raise EditorError(
                f"perturb_noise_std=0 makes this token impossible to select"
            )
    if kernel == "uniform-max" and perturb_noise_std > 0.0:
        if distribution.scores is None:
            raise ValueError("uniform-max requires candidate scores")
        scores = np.asarray(distribution.scores, dtype=np.float64)
        ids = np.asarray(distribution.ids, dtype=np.int64)
        if scores.shape != ids.shape:
            raise ValueError("candidate scores do not match candidate IDs")
        radii = np.zeros(len(ids), dtype=np.float64)
        radii[_noise_indices(distribution, selective_noise_k)] = (
            math.sqrt(3.0) * float(perturb_noise_std)
        )
        other = ids != token_id
        if np.any(other):
            if (scores[ids == token_id] + radii[ids == token_id])[0] <= float(
                np.max((scores - radii)[other])
            ):
                raise EditorError(
                    "uniform-max bounded noise makes this token impossible to select"
                )
    if kernel == "gumbel-max" and gumbel_noise_scale == 0.0:
        proposal = draw_token(
            distribution,
            seed=current_seed,
            stream_fingerprint=stream_fingerprint,
            aligned_step=aligned_step,
            selective_noise_k=selective_noise_k,
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
            selective_noise_k=selective_noise_k,
            kernel=kernel,
            gaussian_noise_std=gaussian_noise_std,
            perturb_noise_std=perturb_noise_std,
            student_t_df=student_t_df,
            gumbel_noise_address=gumbel_noise_address,
            candidate_model_ranks=candidate_model_ranks,
            gumbel_noise_scale=gumbel_noise_scale,
        ) == token_id:
            return seed, checked


__all__ = [
    "CandidateFilterResult",
    "DRAW_KERNELS",
    "GUMBEL_NOISE_ADDRESSES",
    "MAX_SEED",
    "MIN_SEED",
    "PERTURB_MAX_KERNELS",
    "RNG_SCHEME",
    "EligibleScores",
    "StandardCandidateFilter",
    "_rank",
    "_softmax",
    "_top_ids",
    "_validated_logits",
    "apply_candidate_filter",
    "argmax_winner",
    "unified_ranking_scores",
    "draw_token",
    "gaussian_ranking_scores",
    "find_seed_for_token",
    "gumbel_ranking_scores",
    "gumbel_ranked_ids",
    "position_uniform_model_rank",
    "position_uniform_token",
    "perturbation_ranking_scores",
    "raw_rank",
    "top_raw_ids",
]

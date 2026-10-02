"""NumPy-compatible adapter for the experimental Rust sampler.

The extension accepts plain vectors and scalar values only. This module owns
shape/type checks and conversion at the Python boundary; numeric work is done
in Rust.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from . import rust_sampler_native as _native

try:
    from trajectory_editor.core.errors import EditorError
except ImportError:  # Allows inspecting the experiment before installing core.
    class EditorError(ValueError):
        """Fallback matching the core package's public validation exception."""


RNG_SCHEME = _native.RNG_SCHEME
MIN_SEED = _native.MIN_SEED
MAX_SEED = _native.MAX_SEED
DRAW_KERNELS = tuple(_native.DRAW_KERNELS)
PERTURB_MAX_KERNELS = tuple(_native.PERTURB_MAX_KERNELS)
GUMBEL_NOISE_ADDRESSES = ("token-id", "model-rank")


def _call(function, *args):
    try:
        return function(*args)
    except _native.NativeEditorError as exc:
        raise EditorError(str(exc)) from None


def _vector(value, name: str, dtype=None) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if dtype is not None:
        try:
            array = np.asarray(array, dtype=dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain numeric values") from exc
    return array


def _float_vector(value, name: str) -> np.ndarray:
    return _vector(value, name, np.float64)


def _id_vector(value) -> np.ndarray:
    array = _vector(value, "candidate IDs")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("candidate IDs must be a one-dimensional integer array")
    if array.dtype.kind == "u" and len(array) and int(np.max(array)) > MAX_SEED:
        raise ValueError("candidate IDs must fit in signed 64-bit integers")
    try:
        return np.asarray(array, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("candidate IDs must fit in signed 64-bit integers") from exc


def _seed(value) -> int:
    if type(value) is not int or not MIN_SEED <= value <= MAX_SEED:
        raise EditorError("seed must be a signed-64-bit integer")
    return value


def _boundary(value) -> str:
    if type(value) is not int or value < 0:
        raise EditorError("sampling boundary must be a nonnegative integer")
    return str(value)


def _fingerprint(value) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise EditorError("stream_fingerprint must be a lowercase SHA-256 hex digest")
    return value


def _finite_parameter(value, name: str, *, strictly_positive=False) -> float:
    if type(value) not in {int, float}:
        if name == "student_t_df":
            raise EditorError("student_t_df must be finite and greater than 0")
        raise EditorError(f"{name} must be finite and nonnegative")
    try:
        converted = float(value)
    except OverflowError as exc:
        if name == "student_t_df":
            raise EditorError("student_t_df must be finite and greater than 0") from exc
        raise EditorError(f"{name} must be finite and nonnegative") from exc
    invalid = not math.isfinite(converted) or (
        converted <= 0.0 if strictly_positive else converted < 0.0
    )
    if invalid:
        if name == "student_t_df":
            raise EditorError("student_t_df must be finite and greater than 0")
        raise EditorError(f"{name} must be finite and nonnegative")
    return converted


def _dist_parts(distribution: "SparseDistribution"):
    ids = _id_vector(distribution.ids)
    probabilities = _float_vector(distribution.probabilities, "candidate probabilities")
    scores = (
        None
        if distribution.scores is None
        else _float_vector(distribution.scores, "candidate scores")
    )
    return ids, probabilities, scores


def _aligned_probabilities(ids, probabilities):
    if len(ids) != len(probabilities):
        raise ValueError("candidate probabilities do not match candidate IDs")


def _ranks(values, *, validate_dtype=True):
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("candidate model ranks must be distinct positive integers")
    if validate_dtype and not np.issubdtype(array.dtype, np.integer):
        raise ValueError("candidate model ranks must be distinct positive integers")
    return [str(int(value)) for value in array]


class SamplingFilterConfig(Protocol):
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
        ids = _id_vector(self.ids)
        probabilities = _float_vector(self.probabilities, "candidate probabilities")
        _aligned_probabilities(ids, probabilities)
        token_id = int(token_id)
        if not MIN_SEED <= token_id <= MAX_SEED:
            return 0.0
        return float(_native.probability(ids.tolist(), probabilities.tolist(), token_id))


@dataclass(frozen=True)
class CandidateFilterResult:
    scaled_logits: np.ndarray
    stages: dict[str, np.ndarray | None]
    diagnostics: dict[str, object] = field(default_factory=dict)


class StandardCandidateFilter:
    """Deterministic top-k/typical/tail-free/top-p/min-p filtering."""

    name = "standard"

    @classmethod
    def apply(cls, adjusted: np.ndarray, config: SamplingFilterConfig) -> CandidateFilterResult:
        return apply_candidate_filter(adjusted, config)


def _softmax(values: np.ndarray) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("softmax values must be a finite nonempty vector") from exc
    if vector.ndim != 1:
        raise ValueError("softmax values must be a finite nonempty vector")
    result = _call(_native.softmax, vector.tolist())
    return np.asarray(result, dtype=np.float64)


def _top_ids(values: np.ndarray, count: int) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("decoder logits must be a finite nonempty vector") from exc
    if vector.ndim != 1:
        raise ValueError("decoder logits must be a finite nonempty vector")
    count = int(count)
    safe_count = max(1, min(count, len(vector))) if len(vector) else 1
    result = _call(_native.top_ids, vector.tolist(), safe_count)
    return np.asarray(result, dtype=np.int64)


def _validated_logits(logits: np.ndarray) -> np.ndarray:
    try:
        vector = np.asarray(logits, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("decoder logits must be a finite nonempty one-dimensional array") from exc
    if vector.ndim != 1:
        raise ValueError("decoder logits must be a finite nonempty one-dimensional array")
    result = _call(_native.validated_logits, vector.tolist())
    return np.asarray(result, dtype=np.float64)


def _rank(values: np.ndarray, token_id: int) -> int:
    vector = _float_vector(values, "decoder logits")
    token_id = int(token_id)
    if not -(1 << 63) <= token_id <= MAX_SEED:
        raise ValueError("token id is outside the decoder vocabulary")
    return _call(_native.rank, vector.tolist(), token_id)


def raw_rank(logits: np.ndarray, token_id: int) -> int:
    return _rank(_validated_logits(logits), token_id)


def top_raw_ids(logits: np.ndarray, count: int) -> list[int]:
    return [int(value) for value in _top_ids(logits, count)]


def position_uniform(seed: int, stream_fingerprint: str, aligned_step: int) -> float:
    return _call(
        _native.position_uniform,
        _seed(seed),
        _fingerprint(stream_fingerprint),
        _boundary(aligned_step),
    )


def position_uniform_token(
    seed: int, stream_fingerprint: str, aligned_step: int, token_id: int
) -> float:
    if type(token_id) is not int or token_id < 0:
        raise EditorError("token id must be a nonnegative integer")
    return _call(
        _native.position_uniform_token,
        _seed(seed),
        _fingerprint(stream_fingerprint),
        _boundary(aligned_step),
        str(token_id),
    )


def position_uniform_model_rank(
    seed: int, stream_fingerprint: str, aligned_step: int, model_rank: int
) -> float:
    if type(model_rank) is not int or model_rank < 1:
        raise EditorError("model rank must be a positive integer")
    return _call(
        _native.position_uniform_model_rank,
        _seed(seed),
        _fingerprint(stream_fingerprint),
        _boundary(aligned_step),
        str(model_rank),
    )


def apply_candidate_filter(adjusted: np.ndarray, config: SamplingFilterConfig) -> CandidateFilterResult:
    vector = _float_vector(adjusted, "adjusted logits")
    top_k = config.top_k
    if top_k is not None:
        top_k = int(top_k)
        top_k = max(1, min(top_k, len(vector))) if len(vector) else 1
    temperature = float(config.temperature)
    top_p = float(config.top_p)
    min_p = float(config.min_p)
    typical_p = float(config.typical_p)
    tail_free_z = float(config.tail_free_z)
    scaled, raw_stages, greedy, unfiltered = _call(
        _native.apply_filter,
        vector.tolist(),
        temperature,
        top_k,
        top_p,
        min_p,
        typical_p,
        tail_free_z,
    )
    names = (
        "after_temperature",
        "after_top_k",
        "after_typical",
        "after_tail_free",
        "after_top_p",
        "after_min_p",
    )
    stages = {
        name: None if values is None else np.asarray(values, dtype=np.int64)
        for name, values in zip(names, raw_stages)
    }
    if greedy:
        diagnostics = {"filter": StandardCandidateFilter.name, "greedy": True}
    elif unfiltered:
        diagnostics = {"filter": StandardCandidateFilter.name, "unfiltered": True}
    else:
        diagnostics = {
            "filter": StandardCandidateFilter.name,
            "typical_p": typical_p,
            "tail_free_z": tail_free_z,
        }
    return CandidateFilterResult(np.asarray(scaled, dtype=np.float64), stages, diagnostics)


def _draw_arguments(distribution, seed, stream_fingerprint, aligned_step, kernel,
                    gaussian_noise_std, perturb_noise_std, student_t_df,
                    gumbel_noise_address, candidate_model_ranks, gumbel_noise_scale):
    ids, probabilities, scores = _dist_parts(distribution)
    if kernel == "categorical":
        _aligned_probabilities(ids, probabilities)
    gumbel_noise_scale = (
        _finite_parameter(gumbel_noise_scale, "gumbel_noise_scale")
        if kernel == "gumbel-max" else 1.0
    )
    gumbel_noise_address = (
        gumbel_noise_address if isinstance(gumbel_noise_address, str) else ""
    ) if kernel == "gumbel-max" else "token-id"
    if kernel == "gumbel-max" and gumbel_noise_address not in GUMBEL_NOISE_ADDRESSES:
        raise EditorError("gumbel_noise_address must be token-id or model-rank")
    ranks = None
    if kernel == "gumbel-max" and gumbel_noise_scale != 0.0 and candidate_model_ranks is not None:
        ranks = _ranks(candidate_model_ranks)
    gaussian_noise_std = (
        _finite_parameter(gaussian_noise_std, "gaussian_noise_std")
        if kernel == "gaussian-max" else 1.0
    )
    perturb_noise_std = (
        _finite_parameter(perturb_noise_std, "perturb_noise_std")
        if kernel in PERTURB_MAX_KERNELS else 1.0
    )
    student_t_df = (
        _finite_parameter(student_t_df, "student_t_df", strictly_positive=True)
        if kernel == "student-t-max" else 3.0
    )
    if kernel == "gumbel-max" and gumbel_noise_scale == 0.0:
        native_seed, native_fingerprint, native_boundary = 0, "0" * 64, "0"
    else:
        native_seed = _seed(seed)
        native_fingerprint = _fingerprint(stream_fingerprint)
        native_boundary = _boundary(aligned_step)
    return (
        ids.tolist(),
        probabilities.tolist(),
        None if scores is None else scores.tolist(),
        native_seed,
        native_fingerprint,
        native_boundary,
        kernel,
        gaussian_noise_std,
        perturb_noise_std,
        student_t_df,
        gumbel_noise_address,
        ranks,
        gumbel_noise_scale,
    )


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
    try:
        values = np.asarray(log_probabilities, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("conditional Gumbel sampling requires finite log probabilities") from exc
    if values.ndim != 1:
        raise ValueError("conditional Gumbel sampling requires finite log probabilities")
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("conditional Gumbel sampling requires finite log probabilities")
    if type(count) is not int or count < 1:
        raise EditorError("conditional Gumbel sample count must be a positive integer")
    if type(parent_score) not in {int, float} or not math.isfinite(float(parent_score)):
        raise EditorError("conditional Gumbel parent scores must be finite")
    if type(parent_log_probability) not in {int, float} or not math.isfinite(float(parent_log_probability)):
        raise EditorError("conditional Gumbel parent scores must be finite")
    seed_value = _seed(seed)
    fingerprint = _fingerprint(stream_fingerprint)
    boundary = _boundary(aligned_step)
    if isinstance(prefix_token_ids, (str, bytes)):
        raise EditorError("stochastic beam prefixes must be token ID sequences")
    prefix = tuple(prefix_token_ids)
    if any(type(token_id) is not int or not 0 <= token_id < (1 << 64) for token_id in prefix):
        raise EditorError("stochastic beam prefixes must contain nonnegative token IDs")
    safe_count = min(count, len(values))
    result = _call(
        _native.conditional_gumbel_top_k,
        values.tolist(),
        safe_count,
        float(parent_score),
        float(parent_log_probability),
        seed_value,
        fingerprint,
        boundary,
        list(prefix),
    )
    return tuple((int(token_id), float(score)) for token_id, score in result)


def gaussian_ranking_scores(
    distribution: SparseDistribution,
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    noise_std: float = 1.0,
) -> np.ndarray:
    ids, probabilities, scores = _dist_parts(distribution)
    result = _call(
        _native.gaussian_ranking_scores,
        ids.tolist(),
        probabilities.tolist(),
        None if scores is None else scores.tolist(),
        _seed(seed),
        _fingerprint(stream_fingerprint),
        _boundary(aligned_step),
        _finite_parameter(noise_std, "gaussian_noise_std"),
    )
    return np.asarray(result, dtype=np.float64)


def perturbation_ranking_scores(
    distribution: SparseDistribution,
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    kernel: str,
    noise_std: float = 1.0,
    student_t_df: float = 3.0,
) -> np.ndarray:
    if kernel not in PERTURB_MAX_KERNELS:
        raise EditorError("unsupported perturb-and-argmax kernel")
    ids, probabilities, scores = _dist_parts(distribution)
    df = (
        _finite_parameter(student_t_df, "student_t_df", strictly_positive=True)
        if kernel == "student-t-max" else 3.0
    )
    result = _call(
        _native.perturbation_ranking_scores,
        ids.tolist(),
        probabilities.tolist(),
        None if scores is None else scores.tolist(),
        _seed(seed),
        _fingerprint(stream_fingerprint),
        _boundary(aligned_step),
        kernel,
        _finite_parameter(noise_std, "perturb_noise_std"),
        df,
    )
    return np.asarray(result, dtype=np.float64)


def draw_token(
    distribution: SparseDistribution,
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
    kernel: str = "categorical",
    gaussian_noise_std: float = 1.0,
    perturb_noise_std: float = 1.0,
    student_t_df: float = 3.0,
    gumbel_noise_address: str = "token-id",
    candidate_model_ranks: np.ndarray | None = None,
    gumbel_noise_scale: float = 1.0,
) -> int:
    if kernel not in DRAW_KERNELS:
        raise EditorError("unsupported draw kernel")
    args = _draw_arguments(
        distribution,
        seed,
        stream_fingerprint,
        aligned_step,
        kernel,
        gaussian_noise_std,
        perturb_noise_std,
        student_t_df,
        gumbel_noise_address,
        candidate_model_ranks,
        gumbel_noise_scale,
    )
    return int(_call(_native.draw_token, *args))


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
    ids, probabilities, scores = _dist_parts(distribution)
    scale = _finite_parameter(gumbel_noise_scale, "gumbel_noise_scale")
    if not isinstance(noise_address, str):
        noise_address = ""
    if noise_address not in GUMBEL_NOISE_ADDRESSES:
        raise EditorError("gumbel_noise_address must be token-id or model-rank")
    ranks = (
        _ranks(candidate_model_ranks)
        if scale != 0.0 and candidate_model_ranks is not None
        else None
    )
    result = _call(
        _native.gumbel_ranking_scores,
        ids.tolist(),
        probabilities.tolist(),
        None if scores is None else scores.tolist(),
        0 if scale == 0.0 else _seed(seed),
        "0" * 64 if scale == 0.0 else _fingerprint(stream_fingerprint),
        "0" if scale == 0.0 else _boundary(aligned_step),
        noise_address,
        ranks,
        scale,
    )
    return np.asarray(result, dtype=np.float64)


def _winner(distribution, ranking_scores, function):
    ids, _, _ = _dist_parts(distribution)
    scores = _float_vector(ranking_scores, "ranking scores")
    return int(_call(function, ids.tolist(), scores.tolist()))


def gumbel_winner(distribution: SparseDistribution, ranking_scores: np.ndarray) -> int:
    return _winner(distribution, ranking_scores, _native.gumbel_winner)


def gaussian_winner(distribution: SparseDistribution, ranking_scores: np.ndarray) -> int:
    return _winner(distribution, ranking_scores, _native.gaussian_winner)


def perturbation_winner(distribution: SparseDistribution, ranking_scores: np.ndarray) -> int:
    return _winner(distribution, ranking_scores, _native.perturbation_winner)


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
    ids, probabilities, scores = _dist_parts(distribution)
    scale = _finite_parameter(gumbel_noise_scale, "gumbel_noise_scale")
    if not isinstance(noise_address, str):
        noise_address = ""
    if noise_address not in GUMBEL_NOISE_ADDRESSES:
        raise EditorError("gumbel_noise_address must be token-id or model-rank")
    ranks = (
        _ranks(candidate_model_ranks)
        if scale != 0.0 and candidate_model_ranks is not None
        else None
    )
    result = _call(
        _native.gumbel_ranked_ids,
        ids.tolist(),
        probabilities.tolist(),
        None if scores is None else scores.tolist(),
        0 if scale == 0.0 else _seed(seed),
        "0" * 64 if scale == 0.0 else _fingerprint(stream_fingerprint),
        "0" if scale == 0.0 else _boundary(aligned_step),
        noise_address,
        ranks,
        scale,
    )
    return np.asarray(result, dtype=np.int64)


def find_seed_for_token(
    distribution: SparseDistribution,
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
) -> tuple[int, int]:
    if type(token_id) is not int or token_id < 0:
        raise EditorError("draw token id must be a nonnegative integer")
    ids, probabilities, scores = _dist_parts(distribution)
    if not np.any(ids == token_id):
        raise EditorError(f"token {token_id} is outside the active truncated candidate set")
    if token_id > MAX_SEED:
        raise EditorError(f"token {token_id} is outside the active truncated candidate set")
    ranks = None
    gaussian_noise_std = (
        _finite_parameter(gaussian_noise_std, "gaussian_noise_std")
        if kernel == "gaussian-max" else 1.0
    )
    perturb_noise_std = (
        _finite_parameter(perturb_noise_std, "perturb_noise_std")
        if kernel in PERTURB_MAX_KERNELS else 1.0
    )
    student_t_df = (
        _finite_parameter(student_t_df, "student_t_df", strictly_positive=True)
        if kernel == "student-t-max" else 3.0
    )
    gumbel_noise_scale = (
        _finite_parameter(gumbel_noise_scale, "gumbel_noise_scale")
        if kernel == "gumbel-max" else 1.0
    )
    gumbel_noise_address = (
        gumbel_noise_address if isinstance(gumbel_noise_address, str) else ""
    ) if kernel == "gumbel-max" else "token-id"
    if kernel == "gumbel-max" and gumbel_noise_address not in GUMBEL_NOISE_ADDRESSES:
        raise EditorError("gumbel_noise_address must be token-id or model-rank")

    # The source callback belongs to Python. Rust performs all eligibility,
    # draw, and match calculations one candidate at a time so this stays lazy.
    check_current_seed = (
        (kernel == "gaussian-max" and gaussian_noise_std == 0.0)
        or (kernel in PERTURB_MAX_KERNELS and perturb_noise_std == 0.0)
    )
    seed_for_draw = _seed(current_seed) if check_current_seed else 0
    native_fingerprint = _fingerprint(stream_fingerprint) if check_current_seed else "0" * 64
    native_boundary = _boundary(aligned_step) if check_current_seed else "0"
    args = (
        ids.tolist(),
        probabilities.tolist(),
        None if scores is None else scores.tolist(),
        token_id,
        seed_for_draw,
        native_fingerprint,
        native_boundary,
        kernel,
        gaussian_noise_std,
        perturb_noise_std,
        student_t_df,
        gumbel_noise_address,
        ranks,
        gumbel_noise_scale,
    )
    _call(_native.validate_seed_search_target, *args)

    checked = 0
    while True:
        seed = next_seed()
        checked += 1
        if type(seed) is not int or not MIN_SEED <= seed <= MAX_SEED:
            raise EditorError("draw seed generator returned an invalid signed-64-bit seed")
        if seed == current_seed:
            continue
        ranks = (
            _ranks(candidate_model_ranks)
            if candidate_model_ranks is not None
            and kernel == "gumbel-max"
            and gumbel_noise_scale != 0.0
            else None
        )
        matches = _call(
            _native.seed_search_candidate,
            ids.tolist(),
            probabilities.tolist(),
            None if scores is None else scores.tolist(),
            token_id,
            seed,
            _fingerprint(stream_fingerprint),
            _boundary(aligned_step),
            kernel,
            gaussian_noise_std,
            perturb_noise_std,
            student_t_df,
            gumbel_noise_address,
            ranks,
            gumbel_noise_scale,
        )
        if matches:
            return seed, checked


__all__ = [
    "CandidateFilterResult",
    "DRAW_KERNELS",
    "GUMBEL_NOISE_ADDRESSES",
    "MAX_SEED",
    "MIN_SEED",
    "PERTURB_MAX_KERNELS",
    "RNG_SCHEME",
    "SparseDistribution",
    "StandardCandidateFilter",
    "_rank",
    "_softmax",
    "_top_ids",
    "_validated_logits",
    "apply_candidate_filter",
    "conditional_gumbel_top_k",
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
    "perturbation_ranking_scores",
    "perturbation_winner",
    "raw_rank",
    "top_raw_ids",
]

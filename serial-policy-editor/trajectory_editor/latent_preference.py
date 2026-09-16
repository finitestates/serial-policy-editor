"""Experimental latent preference learning over fixed model token features."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .domain import MAX_SEED, MIN_SEED, EditorError, SamplingConfig
from .latent_features import (
    DEFAULT_LATENT_DIMENSION,
    DEFAULT_PROJECTION_SEED,
    coordinate_identity,
)
from .learning_controls import decay_applies, validate_controls, write_scale


_MIN_PROBABILITY = float.fromhex("0x1.0p-1022")
FeatureProvider = Callable[..., np.ndarray]


def choice_gradient(features, probabilities, chosen_token_id):
    """Gradient of log q(y) for a categorical log-linear choice."""
    values = np.asarray(features, dtype=np.float64)
    weights = np.asarray(probabilities, dtype=np.float64)
    mean = np.einsum("v,vd->d", weights, values, dtype=np.float64, optimize=False)
    return np.asarray(values[int(chosen_token_id)], dtype=np.float64) - mean


def pairwise_logistic_gradient(z, chosen_features, rejected_features):
    """Ascent gradient of log sigmoid(z·(f_y-f_r))."""
    delta = np.asarray(chosen_features, dtype=np.float64) - np.asarray(
        rejected_features, dtype=np.float64
    )
    margin = float(np.dot(np.asarray(z, dtype=np.float64), delta))
    if margin >= 0.0:
        coefficient = math.exp(-min(margin, 745.0))
        coefficient /= 1.0 + coefficient
    else:
        coefficient = 1.0 / (1.0 + math.exp(min(margin, 0.0)))
    return coefficient * delta


def fisher_matrix(features, probabilities):
    """Full categorical Fisher covariance for a feature matrix."""
    values = np.asarray(features, dtype=np.float64)
    weights = np.asarray(probabilities, dtype=np.float64)
    mean = np.einsum("v,vd->d", weights, values, dtype=np.float64, optimize=False)
    centered = values - mean
    covariance = np.einsum(
        "v,vi,vj->ij", weights, centered, centered,
        dtype=np.float64, optimize=False,
    )
    return (covariance + covariance.T) * 0.5


def canonical_policy(base_probabilities, features, z):
    """Return ``p0 * exp(F z)`` as a stable categorical distribution."""
    base = np.asarray(base_probabilities, dtype=np.float64)
    values = np.asarray(features, dtype=np.float64)
    vector = np.asarray(z, dtype=np.float64)
    if base.ndim != 1 or values.shape[0] != base.size or vector.shape != (values.shape[1],):
        raise ValueError("canonical policy inputs have incompatible shapes")
    if not np.all(np.isfinite(base)) or np.any(base < 0.0) or float(np.sum(base)) <= 0.0:
        raise ValueError("canonical policy base probabilities are invalid")
    base = base / float(np.sum(base))
    logits = np.log(np.maximum(base, _MIN_PROBABILITY)) + values @ vector
    shifted = logits - float(np.max(logits))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def canonical_policy_kl(base_probabilities, features, old_z, new_z) -> float:
    """Measure exact ``KL(q_new || q_old)`` in a canonical latent policy."""
    base = np.asarray(base_probabilities, dtype=np.float64)
    values = np.asarray(features, dtype=np.float64)
    old_vector = np.asarray(old_z, dtype=np.float64)
    new_vector = np.asarray(new_z, dtype=np.float64)
    if base.ndim != 1 or values.shape[0] != base.size:
        raise ValueError("canonical KL inputs have incompatible shapes")
    base = base / float(np.sum(base))
    log_base = np.log(np.maximum(base, _MIN_PROBABILITY))

    def normalized(scores):
        raw = log_base + scores
        maximum = float(np.max(raw))
        weights = np.exp(raw - maximum)
        log_normalizer = maximum + math.log(float(np.sum(weights)))
        return weights / float(np.sum(weights)), raw - log_normalizer

    old_probabilities, old_log_probabilities = normalized(values @ old_vector)
    new_probabilities, new_log_probabilities = normalized(values @ new_vector)
    return max(0.0, float(np.sum(new_probabilities * (
        new_log_probabilities - old_log_probabilities
    ))))


def exact_kl_line_search(
    base_probabilities,
    features,
    old_z,
    direction,
    target_kl,
    *,
    fisher=None,
    initial_alpha=None,
    max_iterations: int = 64,
    tolerance: float = 1.0e-10,
) -> tuple[float, float, float, int]:
    """Find the largest nonnegative scalar satisfying an exact KL budget.

    Fisher supplies the local scale only.  The returned exact KL is measured
    against the full canonical vocabulary, including when the Fisher itself
    was estimated on a truncated support.
    """
    target = max(0.0, float(target_kl))
    direction = np.asarray(direction, dtype=np.float64)
    old_z = np.asarray(old_z, dtype=np.float64)
    if target <= 0.0 or not np.any(np.abs(direction) > 1.0e-15):
        return 0.0, 0.0, 0.0, 0
    if initial_alpha is None:
        quadratic = (
            float(direction @ np.asarray(fisher, dtype=np.float64) @ direction)
            if fisher is not None else 0.0
        )
        initial_alpha = math.sqrt(2.0 * target / max(quadratic, 1.0e-24))
    initial_alpha = max(float(initial_alpha), 1.0e-12)

    def exact(alpha: float) -> float:
        return canonical_policy_kl(
            base_probabilities, features, old_z, old_z + alpha * direction
        )

    iterations = 0
    lower = 0.0
    upper = initial_alpha
    upper_kl = exact(upper)
    iterations += 1
    if upper_kl <= target:
        lower = upper
        # Expand deterministically until the budget is bracketed.  This makes
        # the result the maximal admissible scalar, not merely the Fisher guess.
        for _ in range(max_iterations // 2):
            candidate = upper * 2.0
            candidate_kl = exact(candidate)
            iterations += 1
            if candidate_kl > target or not math.isfinite(candidate_kl):
                upper = candidate
                break
            lower, upper, upper_kl = candidate, candidate, candidate_kl
        else:
            final_delta = lower * direction
            predicted = (
                0.5 * float(final_delta @ np.asarray(fisher) @ final_delta)
                if fisher is not None else 0.0
            )
            return float(lower), float(upper_kl), float(predicted), iterations
    else:
        upper = initial_alpha

    for _ in range(max_iterations):
        midpoint = (lower + upper) * 0.5
        midpoint_kl = exact(midpoint)
        iterations += 1
        if midpoint_kl <= target:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= tolerance * max(1.0, upper):
            break
    alpha = float(lower)
    exact_value = exact(alpha)
    predicted = 0.0
    if fisher is not None:
        final_delta = alpha * direction
        predicted = 0.5 * float(final_delta @ np.asarray(fisher) @ final_delta)
    return alpha, exact_value, predicted, iterations


def _finite_number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise EditorError(f"{name} must be a finite number")
    value = float(value)
    if nonnegative and value < 0.0:
        raise EditorError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class LatentPreferenceConfig:
    """Conservative controls for the optional latent learner."""

    enabled: bool = False
    dimension: int = DEFAULT_LATENT_DIMENSION
    learning_rate: float = 0.05
    latent_strength: float = 1.0
    max_step: float = 0.25
    max_norm: float = 4.0
    projection_seed: int = DEFAULT_PROJECTION_SEED
    decay: float = 0.0
    severity_cap: int = 1000
    no_severity_attenuation: bool = False
    dead_zone_rank: int = 1
    rejection_strength: float = 0.0
    fast_slow: bool = False
    fast_learning_rate: float | None = None
    fast_decay: float = 0.10
    fast_strength: float | None = None
    fast_max_step: float | None = None
    fast_max_norm: float | None = None
    learning_gate: str = "rank"
    decay_on: str = "update"
    write_reduction: str = "sum"
    rejection_target: str = "proposal"
    learning_scheme: str = "sgd-v1"
    latent_learning_scheme: str | None = None
    learning_metric: str = "euclidean"
    learning_kl: float = 0.05
    fast_learning_kl: float | None = None
    fisher_ridge: float = 1.0e-3
    fisher_mode: str = "diagonal"
    fisher_mass: float = 0.999
    fisher_max_support: int = 2048

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise EditorError("latent preference enabled must be a boolean")
        if self.learning_gate not in ("rank", "sampler"):
            raise EditorError("latent learning gate must be rank or sampler")
        if self.latent_learning_scheme is not None:
            if self.learning_scheme != "sgd-v1" and self.learning_scheme != self.latent_learning_scheme:
                raise EditorError("learning_scheme and latent_learning_scheme disagree")
            object.__setattr__(self, "learning_scheme", self.latent_learning_scheme)
        if self.learning_scheme not in ("sgd-v1", "fisher-kl-v2"):
            raise EditorError("unsupported latent learning scheme")
        if self.learning_metric not in ("euclidean", "fisher"):
            raise EditorError("latent learning_metric must be euclidean or fisher")
        if self.fisher_mode not in ("diagonal", "full"):
            raise EditorError("latent fisher_mode must be diagonal or full")
        validate_controls(self.decay_on, self.write_reduction, self.rejection_target)
        if type(self.dimension) is not int or self.dimension < 1:
            raise EditorError("latent preference dimension must be positive")
        _finite_number(self.learning_rate, "latent learning_rate", nonnegative=True)
        _finite_number(self.latent_strength, "latent strength", nonnegative=True)
        _finite_number(self.max_step, "latent max_step", nonnegative=True)
        _finite_number(self.max_norm, "latent max_norm", nonnegative=True)
        if (type(self.projection_seed) is not int
                or not MIN_SEED <= self.projection_seed <= MAX_SEED):
            raise EditorError("latent projection seed must be a signed 64-bit integer")
        if type(self.no_severity_attenuation) is not bool:
            raise EditorError("latent no_severity_attenuation must be a boolean")
        if type(self.fast_slow) is not bool:
            raise EditorError("latent fast_slow must be a boolean")
        for name in ("severity_cap", "dead_zone_rank"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise EditorError(f"latent {name} must be a positive integer")
        for name in ("decay", "fast_decay"):
            value = _finite_number(getattr(self, name), f"latent {name}")
            if not 0.0 <= value <= 1.0:
                raise EditorError(f"latent {name} must be between 0 and 1")
        _finite_number(self.rejection_strength, "latent rejection_strength", nonnegative=True)
        _finite_number(self.learning_kl, "latent learning_kl", nonnegative=True)
        if self.fast_learning_kl is not None:
            _finite_number(self.fast_learning_kl, "latent fast_learning_kl", nonnegative=True)
        _finite_number(self.fisher_ridge, "latent fisher_ridge", nonnegative=True)
        _finite_number(self.fisher_mass, "latent fisher_mass")
        if not 0.0 < self.fisher_mass <= 1.0:
            raise EditorError("latent fisher_mass must be in (0, 1]")
        if type(self.fisher_max_support) is not int or self.fisher_max_support < 1:
            raise EditorError("latent fisher_max_support must be positive")
        defaults = dict(fast_learning_rate=4.0 * self.learning_rate,
                        fast_strength=0.5 * self.latent_strength,
                        fast_max_step=self.max_step, fast_max_norm=min(1.0, self.max_norm))
        for name, default in defaults.items():
            value = default if getattr(self, name) is None else getattr(self, name)
            _finite_number(value, f"latent {name}", nonnegative=True)
            object.__setattr__(self, name, value)
        if self.fast_learning_kl is None:
            object.__setattr__(self, "fast_learning_kl", float(self.learning_kl))


@dataclass(frozen=True)
class LatentPreferenceResult:
    """One attempted live latent correction and its diagnostics."""

    sampling: SamplingConfig
    observation_boundary: int
    chosen_token_id: int
    old_policy_rank: int
    old_policy_probability: float
    severity: float
    loss: float
    old_z: tuple[float, ...]
    new_z: tuple[float, ...]
    delta: tuple[float, ...]
    policy_weighted_mean_features: tuple[float, ...]
    update_norm: float
    z_norm: float
    enabled: bool
    severity_cap: int = 1000
    no_severity_attenuation: bool = False
    dead_zone_rank: int = 1
    proposal_token_id: int | None = None
    proposal_rejected: bool = False
    rejection_strength: float = 0.0
    decay: float = 0.0
    learning_step_norm: float = 0.0
    decay_norm: float = 0.0
    learning_delta: tuple[float, ...] = ()
    learning_evidence: tuple[float, ...] = ()
    raw_learning_gradient: tuple[float, ...] = ()
    fisher_snapshot: tuple = ()
    canonical_base_probabilities: tuple[float, ...] = ()
    fast_learning_evidence: tuple[float, ...] = ()
    old_fast_z: tuple[float, ...] = ()
    new_fast_z: tuple[float, ...] = ()
    fast_delta: tuple[float, ...] = ()
    fast_update_norm: float = 0.0
    fast_z_norm: float = 0.0
    fast_decay: float = 0.0
    fast_strength: float = 0.0
    fast_learning_delta: tuple[float, ...] = ()
    fast_learning_step_norm: float = 0.0
    fast_decay_norm: float = 0.0
    learning_gate: str = "rank"
    sampler_eligible: bool | None = None
    sampler_probability: float | None = None
    decay_on: str = "update"
    effective_decay: float = 0.0
    effective_fast_decay: float = 0.0
    write_reduction: str = "sum"
    write_evidence_scale: float = 1.0
    write_evidence_tokens: int | None = None
    rejection_target: str = "proposal"
    learning_scheme: str = "sgd-v1"
    learning_step_kl: float = 0.0
    requested_learning_kl: float = 0.0
    predicted_fisher_kl: float = 0.0
    exact_learning_kl: float = 0.0
    kl_line_search_iterations: int = 0
    fast_requested_learning_kl: float = 0.0
    fast_predicted_fisher_kl: float = 0.0
    fast_exact_learning_kl: float = 0.0
    fast_kl_line_search_iterations: int = 0
    raw_gradient_norm: float = 0.0
    pairwise_margin: float | None = None
    pairwise_loss: float = 0.0
    rejection_gradient_norm: float = 0.0
    fisher_mode: str = "diagonal"
    fisher_condition_estimate: float = 0.0
    step_clipped: bool = False
    norm_clipped: bool = False
    learning_policy: str = "deployed"
    learning_policy_rank: int | None = None
    learning_policy_probability: float | None = None
    latent_rank_before: int | None = None
    latent_rank_after: int | None = None
    latent_pre_post_kl: float = 0.0
    latent_effective_logit_rms: float = 0.0
    latent_top_logit_min: float = 0.0
    latent_top_logit_max: float = 0.0
    latent_effective_gain: float = 0.0
    latent_user_multiplier: float = 1.0
    latent_slow_raw_rms: float = 0.0
    latent_fast_raw_rms: float = 0.0
    latent_combined_raw_rms: float = 0.0
    latent_deployment_kl: float = 0.0
    latent_gain_capped: bool = False
    latent_relative_fast_weight: float = 0.0

    @property
    def updated_sampling(self) -> SamplingConfig:
        return self.sampling

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_boundary": self.observation_boundary,
            "chosen_token_id": self.chosen_token_id,
            "old_policy_rank": self.old_policy_rank,
            "old_policy_probability": self.old_policy_probability,
            "severity": self.severity,
            "loss": self.loss,
            "old_z": list(self.old_z),
            "new_z": list(self.new_z),
            "delta": list(self.delta),
            "policy_weighted_mean_features": list(
                self.policy_weighted_mean_features
            ),
            "update_norm": self.update_norm,
            "z_norm": self.z_norm,
            "enabled": self.enabled,
            "severity_cap": self.severity_cap,
            "no_severity_attenuation": self.no_severity_attenuation,
            "dead_zone_rank": self.dead_zone_rank,
            "proposal_token_id": self.proposal_token_id,
            "proposal_rejected": self.proposal_rejected,
            "rejection_strength": self.rejection_strength,
            "decay": self.decay,
            "learning_step_norm": self.learning_step_norm,
            "decay_norm": self.decay_norm,
            "learning_delta": list(self.learning_delta),
            "learning_evidence": list(self.learning_evidence),
            "raw_learning_gradient": list(self.raw_learning_gradient),
            "fast_learning_evidence": list(self.fast_learning_evidence),
            "old_fast_z": list(self.old_fast_z),
            "new_fast_z": list(self.new_fast_z),
            "fast_delta": list(self.fast_delta),
            "fast_update_norm": self.fast_update_norm,
            "fast_z_norm": self.fast_z_norm,
            "fast_decay": self.fast_decay,
            "fast_strength": self.fast_strength,
            "fast_learning_delta": list(self.fast_learning_delta),
            "fast_learning_step_norm": self.fast_learning_step_norm,
            "learning_gate": self.learning_gate,
            "sampler_eligible": self.sampler_eligible,
            "sampler_probability": self.sampler_probability,
            "decay_on": self.decay_on,
            "effective_decay": self.effective_decay,
            "effective_fast_decay": self.effective_fast_decay,
            "write_reduction": self.write_reduction,
            "write_evidence_scale": self.write_evidence_scale,
            "write_evidence_tokens": self.write_evidence_tokens,
            "rejection_target": self.rejection_target,
            "learning_scheme": self.learning_scheme,
            "learning_step_kl": self.learning_step_kl,
            "requested_learning_kl": self.requested_learning_kl,
            "predicted_fisher_kl": self.predicted_fisher_kl,
            "exact_learning_kl": self.exact_learning_kl,
            "kl_line_search_iterations": self.kl_line_search_iterations,
            "fast_requested_learning_kl": self.fast_requested_learning_kl,
            "fast_predicted_fisher_kl": self.fast_predicted_fisher_kl,
            "fast_exact_learning_kl": self.fast_exact_learning_kl,
            "fast_kl_line_search_iterations": self.fast_kl_line_search_iterations,
            "raw_gradient_norm": self.raw_gradient_norm,
            "pairwise_margin": self.pairwise_margin,
            "pairwise_loss": self.pairwise_loss,
            "rejection_gradient_norm": self.rejection_gradient_norm,
            "fisher_mode": self.fisher_mode,
            "fisher_condition_estimate": self.fisher_condition_estimate,
            "step_clipped": self.step_clipped,
            "norm_clipped": self.norm_clipped,
            "learning_policy": self.learning_policy,
            "learning_policy_rank": self.learning_policy_rank,
            "learning_policy_probability": self.learning_policy_probability,
            "latent_rank_before": self.latent_rank_before,
            "latent_rank_after": self.latent_rank_after,
            "latent_pre_post_kl": self.latent_pre_post_kl,
            "latent_effective_logit_rms": self.latent_effective_logit_rms,
            "latent_top_logit_min": self.latent_top_logit_min,
            "latent_top_logit_max": self.latent_top_logit_max,
            "latent_effective_gain": self.latent_effective_gain,
            "latent_user_multiplier": self.latent_user_multiplier,
            "latent_slow_raw_rms": self.latent_slow_raw_rms,
            "latent_fast_raw_rms": self.latent_fast_raw_rms,
            "latent_combined_raw_rms": self.latent_combined_raw_rms,
            "latent_deployment_kl": self.latent_deployment_kl,
            "latent_gain_capped": self.latent_gain_capped,
            "latent_relative_fast_weight": self.latent_relative_fast_weight,
        }


class LatentPreferenceLearner:
    """Update anonymous slow and optional fast memory over fixed token features.

    Both memories and their coordinate seed live in SamplingConfig. The current
    sampler is authoritative on every update, including after rewind or replay;
    the learner's launch configuration must not override restored coordinates.
    """

    def __init__(
        self,
        feature_matrix: np.ndarray | None = None,
        *,
        feature_provider: FeatureProvider | None = None,
        config: LatentPreferenceConfig | None = None,
        **settings: Any,
    ) -> None:
        if config is not None and settings:
            raise TypeError("pass either config or latent-preference settings")
        if feature_matrix is None and feature_provider is None:
            raise TypeError("latent preference requires fixed token features")
        if feature_matrix is not None and feature_provider is not None:
            raise TypeError("pass either feature_matrix or feature_provider")
        self.config = config or LatentPreferenceConfig(**settings)
        self._feature_matrix = feature_matrix
        self._feature_provider = feature_provider

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _features(
        self, vocabulary_size: int, sampling: SamplingConfig, dimension: int
    ) -> np.ndarray:
        if self._feature_matrix is not None:
            values = self._feature_matrix
        else:
            assert self._feature_provider is not None
            kwargs = dict(
                feature_dimension=dimension,
                projection_seed=sampling.latent_projection_seed,
            )
            if sampling.latent_feature_scheme != "random-projection-unit-v1":
                kwargs.update(
                    feature_scheme=sampling.latent_feature_scheme,
                    whitening_ridge=sampling.latent_whitening_ridge,
                )
            values = self._feature_provider(**kwargs)
        values = np.asarray(values, dtype=np.float32)
        expected = (vocabulary_size, dimension)
        if values.shape != expected:
            raise EditorError(
                "latent token features must have shape "
                f"{expected}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise EditorError("latent token features must be finite")
        return values

    def _severity(self, policy_rank: int) -> float:
        if policy_rank <= self.config.dead_zone_rank:
            return 0.0
        if self.config.no_severity_attenuation:
            return 1.0
        return min(
            1.0,
            math.log1p(max(0, policy_rank - self.config.dead_zone_rank))
            / math.log1p(self.config.severity_cap),
        )

    @staticmethod
    def _loss(probability: float) -> float:
        return -math.log(max(float(probability), _MIN_PROBABILITY))

    @staticmethod
    def _sigmoid_negative(margin: float) -> float:
        if margin >= 0.0:
            value = math.exp(-min(float(margin), 745.0))
            return value / (1.0 + value)
        return 1.0 / (1.0 + math.exp(max(float(margin), -745.0)))

    @staticmethod
    def _pair_loss(margin: float) -> float:
        return float(np.logaddexp(0.0, -float(margin)))

    def _fisher_direction(self, features, probabilities, mean, direction):
        """Return a damped natural-gradient direction and Fisher diagnostics."""
        dimension = features.shape[1]
        if self.config.fisher_mode == "full":
            order = np.argsort(-probabilities, kind="stable")
            cumulative = np.cumsum(probabilities[order])
            count = int(np.searchsorted(cumulative, self.config.fisher_mass, side="left")) + 1
            ids = order[: min(count, self.config.fisher_max_support)]
            support_probabilities = probabilities[ids]
            support_probabilities = support_probabilities / float(np.sum(support_probabilities))
            support_features = np.asarray(features[ids], dtype=np.float64)
            support_mean = np.einsum(
                "v,vd->d", support_probabilities, support_features,
                dtype=np.float64, optimize=False,
            )
            centered = support_features - support_mean
            fisher = np.einsum(
                "v,vi,vj->ij", support_probabilities, centered, centered,
                dtype=np.float64, optimize=False,
            )
            mean = support_mean
        else:
            squared = np.asarray(features, dtype=np.float64) ** 2
            second = np.einsum(
                "v,vd->d", probabilities, squared,
                dtype=np.float64, optimize=False,
            )
            diagonal = np.maximum(second - mean * mean, 0.0)
            fisher = np.diag(diagonal)
        fisher = (fisher + fisher.T) * 0.5
        damped = fisher + float(self.config.fisher_ridge) * np.eye(dimension)
        try:
            natural = np.linalg.solve(damped, direction)
        except np.linalg.LinAlgError as exc:
            raise EditorError("latent Fisher solve failed") from exc
        eigenvalues = np.linalg.eigvalsh(damped)
        smallest = max(float(np.min(eigenvalues)), np.finfo(np.float64).tiny)
        condition = float(np.max(eigenvalues) / smallest)
        return natural, fisher, condition, mean

    def _fast_relative_weight(self, sampling: SamplingConfig) -> float:
        """Return the fast channel's relative weight for the current state.

        A fresh fast/slow learner has no saved fast vector yet, so the
        sampling default (zero) cannot be allowed to suppress the configured
        fast channel on its first teaching event.  Once fast memory exists,
        the replayed sampling value is authoritative.
        """
        if not self.config.fast_slow:
            return 0.0
        if not sampling.latent_preference_fast_z:
            return float(self.config.fast_strength)
        return float(sampling.latent_fast_strength)

    def _v2_direction(
        self, features, statistics, canonical_z, chosen_token_id, proposal, rejected,
    ):
        """Build the choice plus pairwise direction under the canonical policy."""
        probabilities = np.asarray(
            getattr(statistics, "learning_probabilities", ()),
            dtype=np.float64,
        )
        if probabilities.shape != (features.shape[0],):
            base = np.asarray(statistics.pre_latent_logits, dtype=np.float64)
            canonical_logits = base.copy()
            if canonical_z.size:
                canonical_logits += np.asarray(features @ canonical_z, dtype=np.float64)
            shifted = canonical_logits - np.max(canonical_logits)
            probabilities = np.exp(shifted)
            probabilities /= float(np.sum(probabilities))
        mean = np.einsum(
            "v,vd->d", probabilities, features,
            dtype=np.float64, optimize=False,
        )
        choice = choice_gradient(features, probabilities, chosen_token_id)
        pair_loss = 0.0
        pair_direction = np.zeros_like(choice)
        pair_margin = None
        if rejected and self.config.rejection_strength:
            chosen_features = np.asarray(features[chosen_token_id], dtype=np.float64)
            if self.config.rejection_target == "sampler":
                ids = statistics.distribution.ids
                weights = np.asarray(statistics.distribution.probabilities, dtype=np.float64)
                for token_id, weight in zip(ids, weights):
                    delta_features = chosen_features - np.asarray(features[int(token_id)], dtype=np.float64)
                    margin = float(np.dot(canonical_z, delta_features))
                    pair_direction += float(weight) * pairwise_logistic_gradient(
                        canonical_z, chosen_features, features[int(token_id)]
                    )
                    pair_loss += float(weight) * self._pair_loss(margin)
                    pair_margin = (
                        float(weight * margin) if pair_margin is None
                        else pair_margin + float(weight * margin)
                    )
            else:
                delta_features = chosen_features - np.asarray(features[proposal], dtype=np.float64)
                margin = float(np.dot(canonical_z, delta_features))
                pair_direction = pairwise_logistic_gradient(
                    canonical_z, chosen_features, features[proposal]
                )
                pair_loss = self._pair_loss(margin)
                pair_margin = margin
            choice += float(self.config.rejection_strength) * pair_direction
        natural, fisher, condition, mean = self._fisher_direction(
            features, probabilities, mean, choice,
        )
        quadratic = float(natural @ fisher @ natural)
        if self.config.learning_kl <= 0.0 or quadratic <= 1.0e-24:
            alpha = 0.0
        else:
            alpha = math.sqrt(2.0 * self.config.learning_kl / quadratic)
        return (
            probabilities, mean, natural, fisher, condition,
            float(alpha), pair_loss, pair_margin, choice, pair_direction,
        )

    def update(
        self,
        observation,
        chosen_token_id: int,
        sampling: SamplingConfig,
    ) -> LatentPreferenceResult:
        if type(chosen_token_id) is not int or not 0 <= chosen_token_id < len(
            observation.logits
        ):
            raise EditorError("chosen token is outside the observation vocabulary")

        statistics = observation.statistics
        old_policy_rank = statistics.policy_rank(chosen_token_id)
        old_policy_probability = float(
            statistics.policy_probabilities[chosen_token_id]
        )
        sampler_eligible = bool(chosen_token_id in statistics.distribution.ids)
        sampler_probability = statistics.distribution.probability(chosen_token_id)
        severity = (float(not sampler_eligible) if self.config.learning_gate == "sampler"
                    else self._severity(old_policy_rank))
        loss = self._loss(old_policy_probability)
        dimension = len(
            sampling.latent_preference_z or sampling.latent_preference_fast_z
        ) or self.config.dimension
        old_z = np.asarray(sampling.latent_preference_z, dtype=np.float64)
        if old_z.size == 0:
            old_z = np.zeros(dimension, dtype=np.float64)

        features = getattr(statistics, "latent_features", None)
        if features is None:
            features = self._features(len(observation.logits), sampling, dimension)
        else:
            features = np.asarray(features, dtype=np.float32)
            expected = (len(observation.logits), dimension)
            if features.shape != expected:
                raise EditorError(
                    "observation latent features do not match learner dimension"
                )
        # Persist the exact basis identity alongside newly learned memory.  A
        # later scheme/seed/ridge change can then reset this memory instead of
        # silently reading it in unrelated coordinates.
        provider_identity = None
        if self._feature_provider is not None:
            owner = getattr(self._feature_provider, "__self__", None)
            identity_method = getattr(owner, "latent_coordinate_identity", None)
            if callable(identity_method):
                provider_identity = identity_method(
                    feature_dimension=features.shape[1],
                    projection_seed=sampling.latent_projection_seed,
                    feature_scheme=sampling.latent_feature_scheme,
                    whitening_ridge=sampling.latent_whitening_ridge,
                )
        current_identity = getattr(statistics, "latent_coordinate_identity", None) or provider_identity or coordinate_identity(
            dimension=features.shape[1],
            projection_seed=sampling.latent_projection_seed,
            feature_scheme=sampling.latent_feature_scheme,
            whitening_ridge=sampling.latent_whitening_ridge,
        )
        latent_before_logits = np.asarray(
            getattr(statistics, "pre_latent_logits", statistics.logits),
            dtype=np.float64,
        )
        latent_after_logits = latent_before_logits + np.asarray(
            getattr(statistics, "latent_logit_adjustments", np.zeros(len(latent_before_logits))),
            dtype=np.float64,
        )
        latent_rank_before = 1 + int(np.count_nonzero(
            latent_before_logits > latent_before_logits[chosen_token_id]
        )) + int(np.count_nonzero(
            latent_before_logits[:chosen_token_id] == latent_before_logits[chosen_token_id]
        ))
        latent_rank_after = 1 + int(np.count_nonzero(
            latent_after_logits > latent_after_logits[chosen_token_id]
        )) + int(np.count_nonzero(
            latent_after_logits[:chosen_token_id] == latent_after_logits[chosen_token_id]
        ))
        latent_diagnostics = getattr(statistics, "latent_diagnostics", {})
        proposal = observation.proposal_token_id
        rejected = proposal != chosen_token_id
        scheme = (
            "fisher-kl-v2"
            if self.config.learning_metric == "fisher"
            else self.config.learning_scheme
        )
        if scheme == "sgd-v1" and sampling.latent_learning_scheme != "sgd-v1":
            # A restored v2 state remains v2 even when a caller constructs a
            # learner with only its legacy defaults.
            scheme = sampling.latent_learning_scheme
        old_fast_z = np.asarray(
            sampling.latent_preference_fast_z or (0.0,) * dimension,
            dtype=np.float64,
        )
        fast_relative_weight = self._fast_relative_weight(sampling)
        canonical_z = old_z + fast_relative_weight * old_fast_z
        fisher = None
        fisher_condition = 0.0
        pair_loss = 0.0
        pair_margin = None
        pair_direction = np.zeros(dimension, dtype=np.float64)
        raw_gradient = np.zeros(dimension, dtype=np.float64)
        requested_learning_kl = 0.0
        predicted_fisher_kl = 0.0
        exact_learning_kl = 0.0
        kl_line_search_iterations = 0
        fast_requested_learning_kl = 0.0
        fast_predicted_fisher_kl = 0.0
        fast_exact_learning_kl = 0.0
        fast_kl_line_search_iterations = 0
        learning_policy = "deployed"
        learning_policy_rank = old_policy_rank
        learning_policy_probability = old_policy_probability
        if scheme == "fisher-kl-v2":
            (
                probabilities, weighted_mean, natural, fisher,
                fisher_condition, alpha, pair_loss, pair_margin,
                raw_gradient, pair_direction,
            ) = self._v2_direction(
                features, statistics, canonical_z, chosen_token_id, proposal, rejected,
            )
            learning_rank = 1 + int(np.count_nonzero(
                probabilities > probabilities[chosen_token_id]
            )) + int(np.count_nonzero(
                probabilities[:chosen_token_id] == probabilities[chosen_token_id]
            ))
            learning_policy = "canonical"
            learning_policy_rank = learning_rank
            learning_policy_probability = float(probabilities[chosen_token_id])
            if self.config.learning_gate == "rank":
                severity = (
                    0.0 if learning_rank <= self.config.dead_zone_rank
                    else 1.0 if self.config.no_severity_attenuation
                    else min(
                        1.0,
                        math.log1p(max(0, learning_rank - self.config.dead_zone_rank))
                        / math.log1p(self.config.severity_cap),
                    )
                )
            raw_learning_delta = severity * alpha * natural
            requested_learning_kl = severity * severity * self.config.learning_kl
            effective_direction = severity * natural
            approximate = float(effective_direction @ fisher @ effective_direction)
            initial_alpha = (
                math.sqrt(2.0 * requested_learning_kl / approximate)
                if requested_learning_kl > 0.0 and approximate > 1.0e-24 else 0.0
            )
            alpha, exact_learning_kl, predicted_fisher_kl, kl_line_search_iterations = exact_kl_line_search(
                statistics.pre_latent_probabilities,
                features,
                canonical_z,
                effective_direction,
                requested_learning_kl,
                fisher=fisher,
                initial_alpha=initial_alpha or None,
            )
            raw_learning_delta = alpha * effective_direction
            loss = self._loss(float(probabilities[chosen_token_id])) + (
                self.config.rejection_strength * pair_loss
            )
        else:
            probabilities = np.asarray(statistics.policy_probabilities, dtype=np.float64)
            weighted_mean = np.empty(dimension, dtype=np.float64)
            # Accumulate in float64 without first materializing a float64 copy
            # of the full float32 feature matrix. Keep this v1 reduction
            # unchanged for replay compatibility.
            np.einsum(
                "v,vd->d",
                probabilities,
                features,
                out=weighted_mean,
                dtype=np.float64,
                optimize=False,
            )
            direction = features[chosen_token_id].astype(np.float64) - weighted_mean
            if severity == 0.0:
                direction = np.zeros_like(direction)
            elif rejected and self.config.rejection_strength:
                negative = features[proposal].astype(np.float64)
                if self.config.rejection_target == "sampler":
                    distribution = statistics.distribution
                    negative = np.einsum("v,vd->d", distribution.probabilities,
                                         features[distribution.ids], dtype=np.float64, optimize=False)
                direction += self.config.rejection_strength * (
                    weighted_mean - negative
                )
            raw_learning_delta = self.config.learning_rate * severity * direction
        decay_allowed = self.enabled and decay_applies(
            self.config.decay_on, rejected=rejected, severity=severity)
        effective_decay = self.config.decay if decay_allowed else 0.
        effective_fast_decay = self.config.fast_decay if decay_allowed and self.config.fast_slow else 0.
        new_z, learning_delta, decay_norm = self._channel(
            old_z, raw_learning_delta, effective_decay,
            self.config.max_step, self.config.max_norm,
        )
        step_norm_before_safety = float(np.linalg.norm(raw_learning_delta))
        step_clipped = bool((
            self.config.max_step == 0.0 and step_norm_before_safety > 0.0
        ) or (
            self.config.max_step > 0.0
            and step_norm_before_safety > self.config.max_step
        ))
        state_before_norm_clip = (1.0 - effective_decay) * old_z + learning_delta
        norm_clipped = bool((
            self.config.max_norm == 0.0 and np.linalg.norm(state_before_norm_clip) > 0.0
        ) or (
            self.config.max_norm > 0.0
            and np.linalg.norm(state_before_norm_clip) > self.config.max_norm
        ))
        if scheme == "fisher-kl-v2" and fisher is not None:
            # Re-measure after the safety channel.  In particular, this makes
            # the diagnostic describe the actual step when max_step clips it.
            measured_delta = (new_z - old_z) if norm_clipped else learning_delta
            measured_state = canonical_z + measured_delta
            exact_learning_kl = canonical_policy_kl(
                statistics.pre_latent_probabilities,
                features,
                canonical_z,
                measured_state,
            )
            predicted_fisher_kl = 0.5 * float(measured_delta @ fisher @ measured_delta)
        new_fast_z = old_fast_z.copy()
        fast_learning_delta = np.zeros_like(old_fast_z)
        fast_decay_norm = 0.0
        fast_raw_delta = np.zeros_like(old_fast_z)
        if self.config.fast_slow:
            fast_raw_delta = (
                raw_learning_delta
                if scheme == "fisher-kl-v2"
                else self.config.fast_learning_rate * severity * direction
            )
            if scheme == "fisher-kl-v2":
                fast_target = severity * severity * float(self.config.fast_learning_kl)
                fast_requested_learning_kl = fast_target
                fast_direction = fast_relative_weight * severity * natural
                fast_alpha, _fast_exact, _fast_predicted, _fast_iterations = exact_kl_line_search(
                    statistics.pre_latent_probabilities,
                    features,
                    canonical_z,
                    fast_direction,
                    fast_target,
                    fisher=fisher,
                )
                fast_kl_line_search_iterations = _fast_iterations
                # This is the parameter-space step for the fast memory.  The
                # line search was performed in the combined canonical policy,
                # so gamma is applied only while measuring its influence.
                fast_raw_delta = fast_alpha * severity * natural
            new_fast_z, fast_learning_delta, fast_decay_norm = self._channel(
                old_fast_z, fast_raw_delta,
                effective_fast_decay, self.config.fast_max_step, self.config.fast_max_norm,
            )
            if scheme == "fisher-kl-v2" and fisher is not None:
                fast_state_before_norm_clip = (
                    (1.0 - effective_fast_decay) * old_fast_z
                    + fast_learning_delta
                )
                fast_norm_clipped = bool(
                    np.linalg.norm(new_fast_z) + 1.0e-12
                    < np.linalg.norm(fast_state_before_norm_clip)
                )
                fast_measured_delta = (
                    new_fast_z - old_fast_z
                    if fast_norm_clipped else fast_learning_delta
                )
                fast_canonical_delta = fast_relative_weight * fast_measured_delta
                fast_exact_learning_kl = canonical_policy_kl(
                    statistics.pre_latent_probabilities,
                    features,
                    canonical_z,
                    canonical_z + fast_canonical_delta,
                )
                fast_predicted_fisher_kl = 0.5 * float(
                    fast_canonical_delta @ fisher @ fast_canonical_delta
                )
        fast_tuple = (tuple(float(v) for v in new_fast_z)
                      if sampling.latent_preference_fast_z or np.any(new_fast_z) else ())
        # Keep a zero-initialized learner inactive until it has a nonzero
        # correction, avoiding unnecessary embedding materialization.
        if not sampling.latent_preference_z and not np.any(new_z):
            new_z_tuple: tuple[float, ...] = ()
        else:
            new_z_tuple = tuple(float(value) for value in new_z)
        state_identity = (
            current_identity
            if self.enabled and (new_z_tuple or fast_tuple)
            else sampling.latent_coordinate_identity
        )
        effective_strength = (
            self.config.latent_strength
            if self.enabled
            else sampling.latent_strength
        )
        updated = replace(
            sampling,
            latent_preference_z=new_z_tuple,
            latent_strength=effective_strength,
            latent_learning_scheme=(scheme if self.enabled else sampling.latent_learning_scheme),
            latent_preference_fast_z=fast_tuple,
            latent_coordinate_identity=state_identity,
            latent_fast_strength=(self.config.fast_strength
                                  if self.enabled and self.config.fast_slow
                                  else sampling.latent_fast_strength),
        )
        actual_delta = new_z - old_z
        return LatentPreferenceResult(
            sampling=updated,
            observation_boundary=observation.boundary,
            chosen_token_id=chosen_token_id,
            old_policy_rank=old_policy_rank,
            old_policy_probability=old_policy_probability,
            severity=severity,
            loss=loss,
            old_z=tuple(float(value) for value in old_z),
            new_z=new_z_tuple,
            delta=tuple(float(value) for value in actual_delta),
            policy_weighted_mean_features=tuple(float(value) for value in weighted_mean),
            update_norm=float(np.linalg.norm(actual_delta)),
            z_norm=float(np.linalg.norm(new_z)),
            enabled=self.enabled,
            severity_cap=self.config.severity_cap,
            no_severity_attenuation=self.config.no_severity_attenuation,
            dead_zone_rank=self.config.dead_zone_rank,
            proposal_token_id=proposal,
            proposal_rejected=rejected,
            rejection_strength=self.config.rejection_strength,
            decay=self.config.decay,
            learning_delta=tuple(float(v) for v in learning_delta),
            learning_evidence=tuple(float(v) for v in raw_learning_delta),
            fast_learning_evidence=(
                tuple(float(v) for v in fast_raw_delta)
                if self.config.fast_slow else ()
            ),
            learning_step_norm=float(np.linalg.norm(learning_delta)),
            decay_norm=decay_norm,
            old_fast_z=tuple(float(v) for v in old_fast_z) if self.config.fast_slow else (),
            new_fast_z=fast_tuple if self.config.fast_slow else (),
            fast_delta=tuple(float(v) for v in new_fast_z - old_fast_z) if self.config.fast_slow else (),
            fast_update_norm=float(np.linalg.norm(new_fast_z - old_fast_z)),
            fast_z_norm=float(np.linalg.norm(new_fast_z)) if self.config.fast_slow else 0.0,
            fast_decay=self.config.fast_decay if self.config.fast_slow else 0.0,
            fast_strength=updated.latent_fast_strength if self.config.fast_slow else 0.0,
            fast_learning_delta=tuple(float(v) for v in fast_learning_delta) if self.config.fast_slow else (),
            fast_learning_step_norm=float(np.linalg.norm(fast_learning_delta)),
            fast_decay_norm=fast_decay_norm,
            learning_gate=self.config.learning_gate, sampler_eligible=sampler_eligible,
            sampler_probability=sampler_probability,
            decay_on=self.config.decay_on, effective_decay=effective_decay,
            effective_fast_decay=effective_fast_decay, write_reduction=self.config.write_reduction,
            rejection_target=self.config.rejection_target,
            learning_scheme=scheme,
            learning_step_kl=(
                exact_learning_kl if scheme == "fisher-kl-v2" else 0.0
            ),
            requested_learning_kl=requested_learning_kl,
            predicted_fisher_kl=predicted_fisher_kl,
            exact_learning_kl=exact_learning_kl,
            kl_line_search_iterations=kl_line_search_iterations,
            fast_requested_learning_kl=fast_requested_learning_kl,
            fast_predicted_fisher_kl=fast_predicted_fisher_kl,
            fast_exact_learning_kl=fast_exact_learning_kl,
            fast_kl_line_search_iterations=fast_kl_line_search_iterations,
            raw_gradient_norm=float(np.linalg.norm(raw_gradient)),
            raw_learning_gradient=tuple(float(v) for v in raw_gradient),
            fisher_snapshot=(
                tuple(tuple(float(v) for v in row) for row in fisher)
                if scheme == "fisher-kl-v2" and fisher is not None else ()
            ),
            canonical_base_probabilities=tuple(
                float(v) for v in statistics.pre_latent_probabilities
            ) if scheme == "fisher-kl-v2" else (),
            pairwise_margin=pair_margin,
            pairwise_loss=pair_loss,
            rejection_gradient_norm=float(np.linalg.norm(pair_direction)),
            fisher_mode=self.config.fisher_mode,
            fisher_condition_estimate=fisher_condition,
            step_clipped=step_clipped,
            norm_clipped=norm_clipped,
            learning_policy=learning_policy,
            learning_policy_rank=learning_policy_rank,
            learning_policy_probability=learning_policy_probability,
            latent_rank_before=latent_rank_before,
            latent_rank_after=latent_rank_after,
            latent_pre_post_kl=float(latent_diagnostics.get("pre_post_latent_kl", 0.0)),
            latent_effective_logit_rms=float(latent_diagnostics.get("effective_logit_rms", 0.0)),
            latent_top_logit_min=float(latent_diagnostics.get("top_latent_logit_min", 0.0)),
            latent_top_logit_max=float(latent_diagnostics.get("top_latent_logit_max", 0.0)),
            latent_effective_gain=float(latent_diagnostics.get("effective_gain", 0.0)),
            latent_user_multiplier=float(latent_diagnostics.get("user_multiplier", 1.0)),
            latent_slow_raw_rms=float(latent_diagnostics.get("slow_raw_logit_rms", 0.0)),
            latent_fast_raw_rms=float(latent_diagnostics.get("fast_raw_logit_rms", 0.0)),
            latent_combined_raw_rms=float(latent_diagnostics.get("combined_raw_logit_rms", 0.0)),
            latent_deployment_kl=float(latent_diagnostics.get("deployment_kl", 0.0)),
            latent_gain_capped=bool(latent_diagnostics.get("gain_capped", False)),
            latent_relative_fast_weight=float(latent_diagnostics.get("relative_fast_weight", 0.0)),
        )

    def _channel(self, old_z, raw_delta, decay, max_step, max_norm):
        """Clip new evidence, forget old memory, then bound the combined state."""
        if not self.enabled:
            return old_z.copy(), np.zeros_like(old_z), 0.0
        delta_norm = float(np.linalg.norm(raw_delta))
        if max_step == 0.0:
            step = np.zeros_like(raw_delta)
        elif delta_norm > max_step:
            step = raw_delta * (max_step / delta_norm)
        else:
            step = raw_delta
        new_z = (1.0 - decay) * old_z + step
        norm = float(np.linalg.norm(new_z))
        if max_norm == 0.0:
            new_z = np.zeros_like(new_z)
        elif norm > max_norm:
            new_z *= max_norm / norm
        return new_z, step, float(np.linalg.norm(decay * old_z))

    def aggregate(
        self,
        results: list[LatentPreferenceResult],
        sampling: SamplingConfig,
        *,
        base_probabilities=None,
        features=None,
    ) -> LatentPreferenceResult:
        """Reduce a Write's token evidence, then clip and conditionally decay once."""
        first = results[0]
        scale, evidence_tokens = write_scale(self.config.write_reduction, results)
        if first.learning_scheme == "fisher-kl-v2":
            # v2 Write learning is a batch update.  Token results carry the
            # un-preconditioned evidence and local Fisher snapshot; only this
            # aggregate performs the solve, KL line search, decay, and bounds.
            dimension = len(first.old_z or first.old_fast_z)
            old_z = np.asarray(first.old_z, dtype=np.float64)
            old_fast = np.asarray(first.old_fast_z, dtype=np.float64)
            if not old_z.size:
                old_z = np.zeros(dimension, dtype=np.float64)
            if not old_fast.size:
                old_fast = np.zeros(dimension, dtype=np.float64)
            admitted = [result for result in results if result.severity > 0.0]
            weighted_gradients = [
                np.asarray(result.raw_learning_gradient, dtype=np.float64)
                * float(result.severity)
                for result in admitted if result.raw_learning_gradient
            ]
            if weighted_gradients:
                evidence = np.sum(weighted_gradients, axis=0) * scale
            else:
                evidence = np.zeros(dimension, dtype=np.float64)
            fisher_items = [
                (
                    np.asarray(result.fisher_snapshot, dtype=np.float64),
                    float(result.severity),
                )
                for result in admitted if result.fisher_snapshot
            ]
            if fisher_items:
                fisher_weight = sum(weight for _snapshot, weight in fisher_items)
                fisher = sum(
                    weight * snapshot for snapshot, weight in fisher_items
                ) / max(fisher_weight, np.finfo(np.float64).tiny)
            else:
                fisher = np.zeros((dimension, dimension), dtype=np.float64)
            fisher = (fisher + fisher.T) * 0.5
            damped = fisher + float(self.config.fisher_ridge) * np.eye(dimension)
            natural = np.linalg.solve(damped, evidence)
            eigenvalues = np.linalg.eigvalsh(damped)
            condition = float(np.max(eigenvalues) / max(float(np.min(eigenvalues)), np.finfo(np.float64).tiny))
            requested = float(self.config.learning_kl)
            gamma = self._fast_relative_weight(sampling)
            canonical_z = old_z + gamma * old_fast
            if base_probabilities is None:
                base_probabilities = getattr(first, "canonical_base_probabilities", None)
            if features is None and base_probabilities is not None:
                features = self._features(len(base_probabilities), sampling, dimension)
            if base_probabilities is not None and features is not None:
                alpha, exact_value, predicted, iterations = exact_kl_line_search(
                    base_probabilities, features, canonical_z, natural, requested,
                    fisher=fisher,
                )
                raw_step = alpha * natural
            else:
                quadratic = float(natural @ fisher @ natural)
                alpha = math.sqrt(2.0 * requested / quadratic) if requested > 0 and quadratic > 1.0e-24 else 0.0
                raw_step = alpha * natural
                predicted = 0.5 * float(raw_step @ fisher @ raw_step)
                exact_value = 0.0
                iterations = 0
            decay_allowed = self.enabled and any(decay_applies(
                self.config.decay_on, rejected=r.proposal_rejected, severity=r.severity
            ) for r in results)
            effective_decay = self.config.decay if decay_allowed else 0.0
            effective_fast_decay = self.config.fast_decay if decay_allowed and self.config.fast_slow else 0.0
            new_z, step, decay_norm = self._channel(
                old_z, raw_step, effective_decay, self.config.max_step, self.config.max_norm
            )
            state_before_norm_clip = (1.0 - effective_decay) * old_z + step
            norm_clipped = bool(
                np.linalg.norm(new_z) + 1.0e-12 < np.linalg.norm(state_before_norm_clip)
            )
            if base_probabilities is not None and features is not None:
                measured_state = canonical_z + (
                    (new_z - old_z) if norm_clipped else step
                )
                exact_value = canonical_policy_kl(
                    base_probabilities, features, canonical_z, measured_state
                )
            measured_delta = (new_z - old_z) if norm_clipped else step
            predicted = 0.5 * float(measured_delta @ fisher @ measured_delta)
            step_clipped = bool(
                np.linalg.norm(step) + 1.0e-12 < np.linalg.norm(raw_step)
            )
            fast_z = old_fast.copy()
            fast_step = np.zeros_like(old_fast)
            fast_decay_norm = 0.0
            fast_evidence = np.zeros_like(old_fast)
            fast_requested = 0.0
            fast_predicted = 0.0
            fast_exact = 0.0
            fast_iterations = 0
            if self.config.fast_slow:
                fast_requested = float(self.config.fast_learning_kl)
                fast_direction = gamma * natural
                if base_probabilities is not None and features is not None:
                    fast_alpha, _fast_exact, _fast_predicted, _fast_iterations = exact_kl_line_search(
                        base_probabilities, features, canonical_z, fast_direction,
                        fast_requested, fisher=fisher,
                    )
                    fast_iterations = _fast_iterations
                    fast_evidence = fast_alpha * natural
                else:
                    fast_evidence = raw_step.copy()
                fast_z, fast_step, fast_decay_norm = self._channel(
                    old_fast, fast_evidence, effective_fast_decay,
                    self.config.fast_max_step, self.config.fast_max_norm,
                )
                fast_state_before_norm_clip = (
                    (1.0 - effective_fast_decay) * old_fast + fast_step
                )
                fast_norm_clipped = bool(
                    np.linalg.norm(fast_z) + 1.0e-12
                    < np.linalg.norm(fast_state_before_norm_clip)
                )
                fast_measured_delta = (
                    fast_z - old_fast if fast_norm_clipped else fast_step
                )
                fast_canonical_delta = gamma * fast_measured_delta
                if base_probabilities is not None and features is not None:
                    fast_exact = canonical_policy_kl(
                        base_probabilities,
                        features,
                        canonical_z,
                        canonical_z + fast_canonical_delta,
                    )
                    fast_predicted = 0.5 * float(
                        fast_canonical_delta @ fisher @ fast_canonical_delta
                    )
            z = (
                tuple(float(v) for v in new_z)
                if sampling.latent_preference_z or np.any(new_z) else ()
            )
            fast_tuple = (
                tuple(float(v) for v in fast_z)
                if sampling.latent_preference_fast_z or np.any(fast_z) else ()
            )
            updated = replace(
                sampling,
                latent_preference_z=z,
                latent_preference_fast_z=fast_tuple,
                latent_learning_scheme=(
                    "fisher-kl-v2" if self.enabled
                    else sampling.latent_learning_scheme
                ),
                latent_coordinate_identity=first.sampling.latent_coordinate_identity,
            )
            pair_margins = [
                r.pairwise_margin for r in admitted
                if r.pairwise_margin is not None
            ]
            pair_losses = [r.pairwise_loss for r in admitted]
            return replace(
                first,
                sampling=updated,
                new_z=z,
                delta=tuple(float(v) for v in new_z - old_z),
                update_norm=float(np.linalg.norm(new_z - old_z)),
                z_norm=float(np.linalg.norm(new_z)),
                decay_norm=decay_norm,
                learning_delta=tuple(float(v) for v in step),
                learning_evidence=tuple(float(v) for v in evidence),
                raw_learning_gradient=tuple(float(v) for v in evidence),
                fisher_snapshot=tuple(tuple(float(v) for v in row) for row in fisher),
                learning_step_norm=float(np.linalg.norm(step)),
                sampler_eligible=None,
                sampler_probability=None,
                proposal_rejected=any(r.proposal_rejected for r in results),
                effective_decay=effective_decay,
                effective_fast_decay=effective_fast_decay,
                write_evidence_scale=scale,
                write_evidence_tokens=evidence_tokens,
                requested_learning_kl=requested,
                predicted_fisher_kl=predicted,
                exact_learning_kl=exact_value,
                learning_step_kl=exact_value,
                kl_line_search_iterations=iterations,
                fast_requested_learning_kl=fast_requested,
                fast_predicted_fisher_kl=fast_predicted,
                fast_exact_learning_kl=fast_exact,
                fast_kl_line_search_iterations=fast_iterations,
                raw_gradient_norm=float(np.linalg.norm(evidence)),
                fisher_condition_estimate=condition,
                step_clipped=step_clipped,
                norm_clipped=norm_clipped,
                pairwise_margin=(float(np.mean(pair_margins)) if pair_margins else None),
                pairwise_loss=float(np.mean(pair_losses)) if pair_losses else 0.0,
                rejection_gradient_norm=float(np.mean([
                    r.rejection_gradient_norm for r in admitted
                ])) if admitted else 0.0,
                policy_weighted_mean_features=tuple(np.mean([
                    r.policy_weighted_mean_features for r in results
                ], axis=0)),
                **(
                    {
                        "new_fast_z": fast_tuple,
                        "fast_delta": tuple(float(v) for v in fast_z - old_fast),
                        "fast_update_norm": float(np.linalg.norm(fast_z - old_fast)),
                        "fast_learning_delta": tuple(float(v) for v in fast_step),
                        "fast_learning_evidence": tuple(float(v) for v in fast_evidence),
                        "fast_learning_step_norm": float(np.linalg.norm(fast_step)),
                        "fast_decay_norm": fast_decay_norm,
                        "fast_z_norm": float(np.linalg.norm(fast_z)),
                    } if self.config.fast_slow else {}
                ),
            )

        # v1 deliberately retains the historical “sum evidence, then clip”
        # path.  Saved v1 trajectories therefore do not acquire v2 geometry.
        evidence = np.sum([r.learning_evidence for r in results], axis=0) * scale
        decay_allowed = self.enabled and any(decay_applies(
            self.config.decay_on, rejected=r.proposal_rejected, severity=r.severity) for r in results)
        effective_decay = self.config.decay if decay_allowed else 0.
        effective_fast_decay = self.config.fast_decay if decay_allowed and self.config.fast_slow else 0.
        old_z = np.asarray(first.old_z)
        new_z, step, decay_norm = self._channel(
            old_z, evidence, effective_decay, self.config.max_step, self.config.max_norm)
        z = tuple(float(v) for v in new_z) if sampling.latent_preference_z or np.any(new_z) else ()
        updated = replace(
            first.sampling,
            latent_preference_z=z,
            latent_learning_scheme=(
                first.learning_scheme
                if self.enabled else first.sampling.latent_learning_scheme
            ),
        )
        extra = {}
        if self.config.fast_slow:
            old_fast = np.asarray(first.old_fast_z)
            fast_evidence = np.sum([r.fast_learning_evidence for r in results], axis=0) * scale
            new_fast, fast_step, fast_decay_norm = self._channel(
                old_fast, fast_evidence, effective_fast_decay,
                self.config.fast_max_step, self.config.fast_max_norm)
            fast_z = tuple(float(v) for v in new_fast) if sampling.latent_preference_fast_z or np.any(new_fast) else ()
            updated = replace(updated, latent_preference_fast_z=fast_z)
            extra = dict(new_fast_z=fast_z, fast_delta=tuple(new_fast - old_fast),
                         fast_update_norm=float(np.linalg.norm(new_fast - old_fast)),
                         fast_z_norm=float(np.linalg.norm(new_fast)),
                         fast_learning_delta=tuple(fast_step),
                         fast_learning_evidence=tuple(fast_evidence),
                         fast_learning_step_norm=float(np.linalg.norm(fast_step)),
                         fast_decay_norm=fast_decay_norm)
        return replace(first, sampling=updated, new_z=z, delta=tuple(new_z - old_z),
                       update_norm=float(np.linalg.norm(new_z - old_z)),
                       z_norm=float(np.linalg.norm(new_z)), decay_norm=decay_norm,
                       learning_delta=tuple(step),
                       learning_evidence=tuple(evidence),
                       learning_step_norm=float(np.linalg.norm(step)),
                       policy_weighted_mean_features=tuple(np.mean([r.policy_weighted_mean_features for r in results], axis=0)),
                       sampler_eligible=None, sampler_probability=None,
                       proposal_rejected=any(r.proposal_rejected for r in results),
                       effective_decay=effective_decay, effective_fast_decay=effective_fast_decay,
                       write_evidence_scale=scale, write_evidence_tokens=evidence_tokens,
                       **extra)

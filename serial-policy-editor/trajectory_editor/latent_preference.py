"""Experimental latent preference learning over fixed model token features."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .domain import MAX_SEED, MIN_SEED, EditorError, SamplingConfig
from .latent_features import DEFAULT_LATENT_DIMENSION, DEFAULT_PROJECTION_SEED
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

    def _v2_direction(
        self, features, statistics, old_z, chosen_token_id, proposal, rejected,
    ):
        """Build the choice plus pairwise direction under the canonical policy."""
        probabilities = np.asarray(
            getattr(statistics, "learning_probabilities", ()),
            dtype=np.float64,
        )
        if probabilities.shape != (features.shape[0],):
            base = np.asarray(statistics.pre_latent_logits, dtype=np.float64)
            canonical_logits = base.copy()
            if old_z.size:
                canonical_logits += np.asarray(features @ old_z, dtype=np.float64)
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
        if rejected and self.config.rejection_strength:
            chosen_features = np.asarray(features[chosen_token_id], dtype=np.float64)
            if self.config.rejection_target == "sampler":
                ids = statistics.distribution.ids
                weights = np.asarray(statistics.distribution.probabilities, dtype=np.float64)
                for token_id, weight in zip(ids, weights):
                    delta_features = chosen_features - np.asarray(features[int(token_id)], dtype=np.float64)
                    margin = float(np.dot(old_z, delta_features))
                    pair_direction += float(weight) * pairwise_logistic_gradient(
                        old_z, chosen_features, features[int(token_id)]
                    )
                    pair_loss += float(weight) * self._pair_loss(margin)
            else:
                delta_features = chosen_features - np.asarray(features[proposal], dtype=np.float64)
                margin = float(np.dot(old_z, delta_features))
                pair_direction = pairwise_logistic_gradient(
                    old_z, chosen_features, features[proposal]
                )
                pair_loss = self._pair_loss(margin)
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
            float(alpha), pair_loss,
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
        fisher = None
        fisher_condition = 0.0
        pair_loss = 0.0
        requested_learning_kl = 0.0
        learning_policy = "deployed"
        learning_policy_rank = old_policy_rank
        learning_policy_probability = old_policy_probability
        if scheme == "fisher-kl-v2":
            (
                probabilities, weighted_mean, natural, fisher,
                fisher_condition, alpha, pair_loss,
            ) = self._v2_direction(
                features, statistics, old_z, chosen_token_id, proposal, rejected,
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
        old_fast_z = np.asarray(sampling.latent_preference_fast_z or (0.0,) * dimension)
        new_fast_z = old_fast_z.copy()
        fast_learning_delta = np.zeros_like(old_fast_z)
        fast_decay_norm = 0.0
        if self.config.fast_slow:
            fast_raw_delta = (
                raw_learning_delta
                if scheme == "fisher-kl-v2"
                else self.config.fast_learning_rate * severity * direction
            )
            if scheme == "fisher-kl-v2":
                fast_raw_delta = fast_raw_delta * (
                    self.config.fast_learning_rate
                    / max(self.config.learning_rate, 1.0e-12)
                )
            new_fast_z, fast_learning_delta, fast_decay_norm = self._channel(
                old_fast_z, fast_raw_delta,
                effective_fast_decay, self.config.fast_max_step, self.config.fast_max_norm,
            )
        fast_tuple = (tuple(float(v) for v in new_fast_z)
                      if sampling.latent_preference_fast_z or np.any(new_fast_z) else ())
        # Keep a zero-initialized learner inactive until it has a nonzero
        # correction, avoiding unnecessary embedding materialization.
        if not sampling.latent_preference_z and not np.any(new_z):
            new_z_tuple: tuple[float, ...] = ()
        else:
            new_z_tuple = tuple(float(value) for value in new_z)
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
                0.5 * float(learning_delta @ fisher @ learning_delta)
                if scheme == "fisher-kl-v2" and fisher is not None else 0.0
            ),
            requested_learning_kl=requested_learning_kl,
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
        self, results: list[LatentPreferenceResult], sampling: SamplingConfig
    ) -> LatentPreferenceResult:
        """Reduce a Write's token evidence, then clip and conditionally decay once."""
        first = results[0]
        scale, evidence_tokens = write_scale(self.config.write_reduction, results)
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

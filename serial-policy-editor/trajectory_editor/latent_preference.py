"""Experimental latent preference learning over fixed model token features."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .domain import MAX_SEED, MIN_SEED, EditorError, SamplingConfig
from .latent_features import DEFAULT_LATENT_DIMENSION, DEFAULT_PROJECTION_SEED


_MIN_PROBABILITY = float.fromhex("0x1.0p-1022")
FeatureProvider = Callable[..., np.ndarray]


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

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise EditorError("latent preference enabled must be a boolean")
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
            "fast_decay_norm": self.fast_decay_norm,
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
            values = self._feature_provider(
                feature_dimension=dimension,
                projection_seed=sampling.latent_projection_seed,
            )
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
        severity = self._severity(old_policy_rank)
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
        probabilities = np.asarray(statistics.policy_probabilities, dtype=np.float64)
        weighted_mean = np.empty(dimension, dtype=np.float64)
        # Accumulate in float64 without first materializing a float64 copy of
        # the full float32 feature matrix.  Keep optimization off so this
        # remains a direct two-operand reduction with bounded workspace.
        np.einsum(
            "v,vd->d",
            probabilities,
            features,
            out=weighted_mean,
            dtype=np.float64,
            optimize=False,
        )
        proposal = observation.proposal_token_id
        rejected = proposal != chosen_token_id
        direction = features[chosen_token_id].astype(np.float64) - weighted_mean
        if severity == 0.0:
            direction = np.zeros_like(direction)
        elif rejected and self.config.rejection_strength:
            direction += self.config.rejection_strength * (
                weighted_mean - features[proposal].astype(np.float64)
            )
        new_z, learning_delta, decay_norm = self._channel(
            old_z, self.config.learning_rate * severity * direction,
            self.config.decay, self.config.max_step, self.config.max_norm,
        )
        old_fast_z = np.asarray(sampling.latent_preference_fast_z or (0.0,) * dimension)
        new_fast_z = old_fast_z.copy()
        fast_learning_delta = np.zeros_like(old_fast_z)
        fast_decay_norm = 0.0
        if self.config.fast_slow:
            new_fast_z, fast_learning_delta, fast_decay_norm = self._channel(
                old_fast_z, self.config.fast_learning_rate * severity * direction,
                self.config.fast_decay, self.config.fast_max_step, self.config.fast_max_norm,
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
            learning_evidence=tuple(self.config.learning_rate * severity * direction),
            fast_learning_evidence=(tuple(self.config.fast_learning_rate * severity * direction)
                                    if self.config.fast_slow else ()),
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
        """Sum a Write's token evidence, then clip and decay once atomically."""
        first = results[0]
        evidence = np.sum([r.learning_evidence for r in results], axis=0)
        old_z = np.asarray(first.old_z)
        new_z, step, decay_norm = self._channel(
            old_z, evidence, self.config.decay, self.config.max_step, self.config.max_norm)
        z = tuple(float(v) for v in new_z) if sampling.latent_preference_z or np.any(new_z) else ()
        updated = replace(first.sampling, latent_preference_z=z)
        extra = {}
        if self.config.fast_slow:
            old_fast = np.asarray(first.old_fast_z)
            fast_evidence = np.sum([r.fast_learning_evidence for r in results], axis=0)
            new_fast, fast_step, fast_decay_norm = self._channel(
                old_fast, fast_evidence, self.config.fast_decay,
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
                       **extra)

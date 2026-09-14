"""Experimental latent preference learning over fixed model token features."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .domain import EditorError, SamplingConfig
from .latent_features import DEFAULT_LATENT_DIMENSION, DEFAULT_PROJECTION_SEED


_SEVERITY_RANK_CAP = 1000
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

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise EditorError("latent preference enabled must be a boolean")
        if type(self.dimension) is not int or self.dimension < 1:
            raise EditorError("latent preference dimension must be positive")
        _finite_number(self.learning_rate, "latent learning_rate", nonnegative=True)
        _finite_number(self.latent_strength, "latent strength", nonnegative=True)
        _finite_number(self.max_step, "latent max_step", nonnegative=True)
        _finite_number(self.max_norm, "latent max_norm", nonnegative=True)
        if type(self.projection_seed) is not int:
            raise EditorError("latent projection seed must be an integer")


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
        }


class LatentPreferenceLearner:
    """Update one anonymous vector using fixed token embedding features.

    The feature matrix is fixed and supplied by the model backend.  The only
    evolving state is ``SamplingConfig.latent_preference_z``; this object does
    not maintain a token-keyed memory.
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

    def _features(self, vocabulary_size: int) -> np.ndarray:
        if self._feature_matrix is not None:
            values = self._feature_matrix
        else:
            assert self._feature_provider is not None
            values = self._feature_provider(
                feature_dimension=self.config.dimension,
                projection_seed=self.config.projection_seed,
            )
        values = np.asarray(values, dtype=np.float32)
        expected = (vocabulary_size, self.config.dimension)
        if values.shape != expected:
            raise EditorError(
                "latent token features must have shape "
                f"{expected}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise EditorError("latent token features must be finite")
        return values

    @staticmethod
    def _severity(policy_rank: int) -> float:
        return min(
            1.0,
            math.log1p(max(0, policy_rank - 1))
            / math.log1p(_SEVERITY_RANK_CAP),
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
        old_z = np.asarray(sampling.latent_preference_z, dtype=np.float64)
        if old_z.size == 0:
            old_z = np.zeros(self.config.dimension, dtype=np.float64)
        elif old_z.size != self.config.dimension:
            raise EditorError(
                "saved latent preference state does not match learner dimension"
            )

        features = getattr(statistics, "latent_features", None)
        if features is None:
            features = self._features(len(observation.logits))
        else:
            features = np.asarray(features, dtype=np.float32)
            expected = (len(observation.logits), self.config.dimension)
            if features.shape != expected:
                raise EditorError(
                    "observation latent features do not match learner dimension"
                )
        probabilities = np.asarray(statistics.policy_probabilities, dtype=np.float64)
        weighted_mean = np.empty(self.config.dimension, dtype=np.float64)
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
        raw_delta = self.config.learning_rate * severity * (
            features[chosen_token_id].astype(np.float64) - weighted_mean
        )
        if self.enabled:
            delta_norm = float(np.linalg.norm(raw_delta))
            if self.config.max_step == 0.0:
                applied_delta = np.zeros_like(raw_delta)
            elif delta_norm > self.config.max_step:
                applied_delta = raw_delta * (self.config.max_step / delta_norm)
            else:
                applied_delta = raw_delta
            new_z = old_z + applied_delta
            new_norm = float(np.linalg.norm(new_z))
            if self.config.max_norm == 0.0:
                new_z = np.zeros_like(new_z)
            elif new_norm > self.config.max_norm:
                new_z = new_z * (self.config.max_norm / new_norm)
        else:
            applied_delta = np.zeros_like(old_z)
            new_z = old_z.copy()

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
        )

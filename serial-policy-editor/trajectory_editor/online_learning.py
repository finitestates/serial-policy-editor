"""Small, opt-in online learning over named bias-group strengths.

The model, vocabulary, and group definitions remain fixed.  The only values
this module can change are the scalar ``BiasGroup.bias`` values already held by
the sampling configuration.  Counterfactual policy surfaces are evaluated from
the frozen logits captured in an :class:`Observation`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from .domain import EditorError, SamplingConfig
from .sampling import ObservationStatistics


_SEVERITY_RANK_CAP = 1000


def _finite_number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise EditorError(f"{name} must be a finite number")
    value = float(value)
    if nonnegative and value < 0.0:
        raise EditorError(f"{name} must be nonnegative")
    return value


@dataclass(frozen=True)
class OnlineLearningConfig:
    """Conservative controls for the optional live learner."""

    enabled: bool = False
    learning_rate: float = 0.05
    epsilon: float = 0.05
    max_step: float = 0.25
    min_bias: float = -4.0
    max_bias: float = 4.0
    learnable_groups: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise EditorError("online learning enabled must be a boolean")
        learning_rate = _finite_number(
            self.learning_rate, "learning_rate", nonnegative=True
        )
        epsilon = _finite_number(self.epsilon, "epsilon")
        max_step = _finite_number(self.max_step, "max_step", nonnegative=True)
        min_bias = _finite_number(self.min_bias, "min_bias")
        max_bias = _finite_number(self.max_bias, "max_bias")
        if epsilon <= 0.0:
            raise EditorError("epsilon must be positive")
        if min_bias > max_bias:
            raise EditorError("min_bias must not exceed max_bias")
        if self.learnable_groups is None:
            groups = None
        else:
            if isinstance(self.learnable_groups, str) or not hasattr(
                self.learnable_groups, "__iter__"
            ):
                raise EditorError("learnable_groups must be a list of names")
            groups = []
            for name in self.learnable_groups:
                if not isinstance(name, str):
                    raise EditorError("learnable_groups must contain names")
                groups.extend(name_part.strip() for name_part in name.split(","))
            if any(not name for name in groups):
                raise EditorError("learnable_groups must contain nonempty names")
            groups = tuple(dict.fromkeys(groups))
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "max_step", max_step)
        object.__setattr__(self, "min_bias", min_bias)
        object.__setattr__(self, "max_bias", max_bias)
        object.__setattr__(self, "learnable_groups", groups)


@dataclass(frozen=True)
class LearningResult:
    """One attempted live correction and the resulting sampler settings."""

    sampling: SamplingConfig
    observation_boundary: int
    chosen_token_id: int
    old_policy_rank: int
    old_policy_probability: float
    severity: float
    loss: float
    old_group_weights: dict[str, float]
    new_group_weights: dict[str, float]
    group_deltas: dict[str, float]
    gradients: dict[str, float]
    update_norm: float
    enabled: bool

    @property
    def updated_sampling(self) -> SamplingConfig:
        """Readable alias for callers that prefer an explicit name."""
        return self.sampling

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.observation_boundary,
            "chosen_token_id": self.chosen_token_id,
            "old_policy_rank": self.old_policy_rank,
            "old_policy_probability": self.old_policy_probability,
            "severity": self.severity,
            "loss": self.loss,
            "old_group_weights": dict(self.old_group_weights),
            "new_group_weights": dict(self.new_group_weights),
            "group_deltas": dict(self.group_deltas),
            "gradients": dict(self.gradients),
            "update_norm": self.update_norm,
            "enabled": self.enabled,
        }


class OnlineLearner:
    """Finite-difference learner for fixed named bias groups.

    This object is deliberately stateless.  Learned state lives only in the
    returned ``SamplingConfig`` and therefore remains replayable through the
    existing sampler-segment records.
    """

    def __init__(
        self,
        config: OnlineLearningConfig | None = None,
        **settings: Any,
    ) -> None:
        if config is not None and settings:
            raise TypeError("pass either config or online-learning settings")
        self.config = config or OnlineLearningConfig(**settings)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @staticmethod
    def _severity(policy_rank: int) -> float:
        return min(
            1.0,
            math.log1p(max(0, policy_rank - 1))
            / math.log1p(_SEVERITY_RANK_CAP),
        )

    @staticmethod
    def _loss(statistics: ObservationStatistics, token_id: int) -> float:
        probability = float(statistics.policy_probabilities[token_id])
        # Keep diagnostic payloads JSON-safe even for an extreme logit spread
        # that underflows a probability to zero.
        return -math.log(max(probability, float.fromhex("0x1.0p-1022")))

    @staticmethod
    def _with_group_bias(
        sampling: SamplingConfig, group_name: str, bias: float
    ) -> SamplingConfig:
        groups = tuple(
            replace(group, bias=bias) if group.name == group_name else group
            for group in sampling.bias_groups
        )
        return replace(sampling, bias_groups=groups)

    def _counterfactual(
        self,
        observation,
        sampling: SamplingConfig,
    ) -> ObservationStatistics:
        return ObservationStatistics(
            observation.logits,
            sampling,
            observation.prefix_token_ids,
            observation.statistics.boundaries,
            latent_features=getattr(observation.statistics, "latent_features", None),
        )

    def update(
        self,
        observation,
        chosen_token_id: int,
        sampling: SamplingConfig,
    ) -> LearningResult:
        """Apply one bounded correction using the pre-action observation."""
        if type(chosen_token_id) is not int or not 0 <= chosen_token_id < len(
            observation.logits
        ):
            raise EditorError("chosen token is outside the observation vocabulary")

        statistics = observation.statistics
        old_policy_rank = statistics.policy_rank(chosen_token_id)
        old_policy_probability = float(statistics.policy_probabilities[chosen_token_id])
        severity = self._severity(old_policy_rank)
        loss = self._loss(statistics, chosen_token_id)
        groups = tuple(sampling.bias_groups)
        old_weights = {group.name: float(group.bias) for group in groups}
        selected_names = (
            set(old_weights)
            if self.config.learnable_groups is None
            else set(self.config.learnable_groups)
        )
        unknown = selected_names - set(old_weights)
        if unknown:
            raise EditorError(
                "learnable group not present in sampling configuration: "
                + ", ".join(sorted(unknown))
            )

        gradients = {group.name: 0.0 for group in groups}
        new_weights = dict(old_weights)
        if self.enabled:
            for group in groups:
                if group.name not in selected_names:
                    continue
                plus = self._counterfactual(
                    observation,
                    self._with_group_bias(
                        sampling, group.name, group.bias + self.config.epsilon
                    ),
                )
                minus = self._counterfactual(
                    observation,
                    self._with_group_bias(
                        sampling, group.name, group.bias - self.config.epsilon
                    ),
                )
                gradient = (self._loss(plus, chosen_token_id) - self._loss(
                    minus, chosen_token_id
                )) / (2.0 * self.config.epsilon)
                gradients[group.name] = (
                    float(gradient) if math.isfinite(float(gradient)) else 0.0
                )
                raw_delta = (
                    -self.config.learning_rate * severity * gradients[group.name]
                )
                delta = max(-self.config.max_step, min(self.config.max_step, raw_delta))
                new_weights[group.name] = max(
                    self.config.min_bias,
                    min(self.config.max_bias, group.bias + delta),
                )

        deltas = {
            name: float(new_weights[name] - old_weights[name])
            for name in old_weights
        }
        update_norm = math.sqrt(sum(delta * delta for delta in deltas.values()))
        updated_groups = tuple(
            replace(group, bias=new_weights[group.name]) for group in groups
        )
        return LearningResult(
            sampling=replace(sampling, bias_groups=updated_groups),
            observation_boundary=observation.boundary,
            chosen_token_id=chosen_token_id,
            old_policy_rank=old_policy_rank,
            old_policy_probability=old_policy_probability,
            severity=severity,
            loss=loss,
            old_group_weights=old_weights,
            new_group_weights=new_weights,
            group_deltas=deltas,
            gradients=gradients,
            update_norm=update_norm,
            enabled=self.enabled,
        )

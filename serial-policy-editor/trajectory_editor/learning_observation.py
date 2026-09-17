"""Reusable numeric inputs for one teacher-learning observation.

The policy surface is already fully assembled by ``ObservationStatistics``.
This small wrapper keeps the learners from repeatedly normalizing and looking
up the same arrays while leaving the persisted episode state unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CompiledLearningObservation:
    """Read-only policy arrays plus ephemeral per-observation caches."""

    observation: Any
    policy_probabilities: np.ndarray
    learning_probabilities: np.ndarray
    distribution_ids: np.ndarray
    distribution_probabilities: np.ndarray
    token_preference_features: np.ndarray | None = None
    group_scales: dict[tuple[Any, ...], dict[int, float]] = field(default_factory=dict)

    @classmethod
    def from_observation(cls, observation: Any) -> "CompiledLearningObservation":
        statistics = observation.statistics
        return cls(
            observation=observation,
            policy_probabilities=np.asarray(statistics.policy_probabilities),
            learning_probabilities=np.asarray(statistics.learning_probabilities),
            distribution_ids=np.asarray(statistics.distribution.ids),
            distribution_probabilities=np.asarray(statistics.distribution.probabilities),
            token_preference_features=getattr(
                statistics, "token_preference_features", None
            ),
        )

    @property
    def statistics(self):
        return self.observation.statistics

    def sampler_eligible(self, token_id: int) -> bool:
        return bool(np.any(self.distribution_ids == int(token_id)))

    def sampler_probability(self, token_id: int) -> float:
        matches = np.flatnonzero(self.distribution_ids == int(token_id))
        return (
            float(self.distribution_probabilities[matches[0]])
            if len(matches) else 0.0
        )


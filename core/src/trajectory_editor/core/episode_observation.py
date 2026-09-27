"""Core decision snapshot assembled at one live episode boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .policy_calculations import PolicyCalculations
from .sampling import SparseDistribution


@dataclass(frozen=True)
class EpisodeObservation:
    boundary: int
    sampling_boundary: int
    prefix_token_ids: Sequence[int] = field(repr=False)
    proposal_token_id: int
    proposal_text: str
    proposal_raw_rank: int
    proposal_decoder_probability: float
    policy_calculations: PolicyCalculations = field(repr=False, compare=False)

    @property
    def logits(self) -> np.ndarray:
        return self.policy_calculations.logits

    @property
    def distribution(self) -> SparseDistribution:
        return self.policy_calculations.distribution

    @cached_property
    def proposal_raw_probability(self) -> float:
        """Model soft-max mass for the proposal; computed on first read."""
        return float(self.policy_calculations.raw_probabilities([self.proposal_token_id])[0])

    @cached_property
    def proposal_policy_rank(self) -> int:
        return self.policy_calculations.policy_rank(self.proposal_token_id)


__all__ = ["EpisodeObservation"]

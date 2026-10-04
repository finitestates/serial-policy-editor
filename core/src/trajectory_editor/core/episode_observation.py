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
    policy_calculations: PolicyCalculations = field(repr=False, compare=False)
    gumbel_scores: np.ndarray | None = field(default=None, repr=False, compare=False)
    ranking_scores: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def logits(self) -> np.ndarray:
        return self.policy_calculations.logits

    @property
    def distribution(self) -> SparseDistribution:
        return self.policy_calculations.distribution

    @cached_property
    def proposal_decoder_probability(self) -> float:
        """Eligible-score softmax diagnostic; never used to choose a winner."""
        return self.distribution.probability(self.proposal_token_id)

    @cached_property
    def noise_by_token(self) -> dict[int, float]:
        if self.ranking_scores is None or self.distribution.scores is None:
            return {}
        return {
            int(token_id): float(final - initial)
            for token_id, final, initial in zip(
                self.distribution.ids, self.ranking_scores, self.distribution.scores
            )
        }

    @cached_property
    def proposal_raw_probability(self) -> float:
        """Model soft-max mass for the proposal; computed on first read."""
        return float(self.policy_calculations.raw_probabilities([self.proposal_token_id])[0])

    @cached_property
    def proposal_policy_rank(self) -> int:
        return self.policy_calculations.policy_rank(self.proposal_token_id)

    @cached_property
    def gumbel_order(self) -> tuple[int, ...]:
        if self.gumbel_scores is None:
            return ()
        ids = self.distribution.ids
        scores = np.asarray(self.gumbel_scores, dtype=np.float64)
        if scores.shape != ids.shape:
            raise ValueError("Gumbel scores do not match the active candidate IDs")
        order = np.lexsort((ids, -scores))
        return tuple(int(token_id) for token_id in ids[order])

    @cached_property
    def gumbel_ranks(self) -> dict[int, int]:
        return {token_id: rank for rank, token_id in enumerate(self.gumbel_order, 1)}


__all__ = ["EpisodeObservation"]

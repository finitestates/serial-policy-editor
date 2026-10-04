"""Core candidate records used by the vocabulary menu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    # One-based model rank: the fixed keyboard address, even when rows are
    # displayed in policy or Gumbel order.
    rank: int
    token_id: int
    text: str
    eligible_softmax: float | None
    is_eog: bool
    raw_probability: float | None = None
    bias: float = 0.0
    policy_rank: int | None = None
    policy_probability: float | None = None
    raw_logit: float | None = None
    # One-based order within the active filtered candidate set.
    gumbel_rank: int | None = None
    noise: float | None = None
    eligible: bool = False

    @property
    def model_probability(self) -> float | None:
        """Canonical backend/model probability surface (None if not computed)."""

        return self.raw_probability

    @property
    def model_logit(self) -> float | None:
        return self.raw_logit

    def to_dict(self) -> dict[str, Any]:
        return {
            "bias": self.bias,
            "rank": self.rank,
            "token_id": self.token_id,
            "text": self.text,
            "raw_probability": self.raw_probability,
            "model_probability": self.model_probability,
            "eligible_softmax": self.eligible_softmax,
            "eligible": self.eligible,
            "is_eog": self.is_eog,
            "policy_rank": self.policy_rank,
            "policy_probability": self.policy_probability,
            "raw_logit": self.raw_logit,
            "model_logit": self.model_logit,
            "gumbel_rank": self.gumbel_rank,
            "noise": self.noise,
        }


__all__ = ["Candidate"]

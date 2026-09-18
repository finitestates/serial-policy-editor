"""Core candidate records used by the vocabulary menu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    rank: int
    token_id: int
    text: str
    raw_probability: float
    decoder_probability: float
    is_eog: bool
    bias: float = 0.0
    policy_rank: int | None = None
    policy_probability: float | None = None
    raw_logit: float | None = None

    @property
    def model_probability(self) -> float:
        """Canonical backend/model probability surface."""

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
            "decoder_probability": self.decoder_probability,
            "decoder_supported": self.decoder_probability > 0.0,
            "is_eog": self.is_eog,
            "policy_rank": self.policy_rank,
            "policy_probability": self.policy_probability,
            "raw_logit": self.raw_logit,
            "model_logit": self.model_logit,
        }


__all__ = ["Candidate"]

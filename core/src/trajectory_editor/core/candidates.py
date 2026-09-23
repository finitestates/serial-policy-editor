"""Core candidate records used by the vocabulary menu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    rank: int
    token_id: int
    text: str
    decoder_probability: float | None
    is_eog: bool
    raw_probability: float | None = None
    bias: float = 0.0
    policy_rank: int | None = None
    policy_probability: float | None = None
    raw_logit: float | None = None
    # logit[i] - logit[i+1] in the ordered menu list; None for last / unset.
    neighbor_margin: float | None = None
    # (logit - mean) / std over full-vocab raw logits; None when unset/undefined.
    logit_z: float | None = None

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
            "decoder_probability": self.decoder_probability,
            "decoder_supported": self.decoder_probability is not None and self.decoder_probability > 0.0,
            "is_eog": self.is_eog,
            "policy_rank": self.policy_rank,
            "policy_probability": self.policy_probability,
            "raw_logit": self.raw_logit,
            "model_logit": self.model_logit,
            "neighbor_margin": self.neighbor_margin,
            "logit_z": self.logit_z,
        }


__all__ = ["Candidate"]

"""Core presentation records used by the terminal episode editor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .candidates import Candidate
from .errors import EditorError


@dataclass(frozen=True)
class ChoiceSet:
    choice_set_id: str
    prompt_id: str
    aligned_step: int
    sampling_coordinate: int
    context_token_sha256: str
    context_text_tail: str
    proposal_token_id: int
    proposal_text: str
    proposal_raw_probability: float | None
    proposal_decoder_probability: float
    proposal_is_eog: bool
    candidates: tuple[Candidate, ...]
    vocabulary_size: int | None = None
    proposal_raw_rank: int | None = None
    proposal_policy_rank: int | None = None
    proposal_policy_probability: float | None = None
    raw_k1_logit: float | None = None


class ActionKind(str, Enum):
    ACCEPT = "accept"
    SELECT = "select"
    INSERT = "insert"


class InsertMode(str, Enum):
    CONTINUATION = "continuation"
    EXACT = "exact"


@dataclass(frozen=True)
class EditAction:
    kind: ActionKind
    selected_rank: int | None = None
    supplied_text: str | None = None
    insert_mode: InsertMode | None = None

    @classmethod
    def accept(cls) -> "EditAction":
        return cls(ActionKind.ACCEPT)

    @classmethod
    def select(cls, rank: int) -> "EditAction":
        if rank < 1:
            raise EditorError("selected rank must be positive")
        return cls(ActionKind.SELECT, selected_rank=rank)

    @classmethod
    def insert(cls, text: str, mode: InsertMode) -> "EditAction":
        if not text:
            raise EditorError("inserted text cannot be empty")
        return cls(ActionKind.INSERT, supplied_text=text, insert_mode=mode)


__all__ = ["ActionKind", "ChoiceSet", "EditAction", "InsertMode"]

"""Core presentation records used by the terminal episode editor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
from enum import Enum

from .candidates import Candidate
from .errors import EditorError


@dataclass(frozen=True, eq=False, slots=True)
class ContextText:
    """Persistent rendered context, extended by only newly decoded text."""

    parent: "ContextText | None"
    chunk: str
    character_count: int

    @classmethod
    def root(cls, text: str) -> "ContextText":
        return cls(None, text, len(text))

    def append(self, text: str) -> "ContextText":
        if not text:
            return self
        return ContextText(self, text, self.character_count + len(text))

    def iter_chunks(self) -> Iterator[str]:
        nodes: list[ContextText] = []
        current: ContextText | None = self
        while current is not None:
            if current.chunk:
                nodes.append(current)
            current = current.parent
        for node in reversed(nodes):
            yield node.chunk

    def nodes_since(
        self, ancestor: "ContextText | None"
    ) -> tuple["ContextText", ...] | None:
        nodes: list[ContextText] = []
        current: ContextText | None = self
        while current is not ancestor and current is not None:
            nodes.append(current)
            current = current.parent
        if current is not ancestor:
            return None
        return tuple(reversed(nodes))

    def chunks_since(self, ancestor: "ContextText | None") -> tuple[str, ...] | None:
        nodes = self.nodes_since(ancestor)
        if nodes is None:
            return None
        return tuple(node.chunk for node in nodes if node.chunk)

    def materialize(self) -> str:
        return "".join(self.iter_chunks())

    def iter_slices(self, start: int, stop: int) -> Iterator[str]:
        start = max(0, min(start, self.character_count))
        stop = max(start, min(stop, self.character_count))
        if start == stop:
            return
        pieces: list[tuple[ContextText, int, int]] = []
        current: ContextText | None = self
        while current is not None:
            chunk_start = current.character_count - len(current.chunk)
            if current.character_count <= start:
                break
            lower = max(start, chunk_start)
            upper = min(stop, current.character_count)
            if upper > lower:
                pieces.append((current, lower - chunk_start, upper - chunk_start))
            current = current.parent
        for node, lower, upper in reversed(pieces):
            yield node.chunk[lower:upper]

    def slice(self, start: int, stop: int) -> str:
        return "".join(self.iter_slices(start, stop))

    def tail(self, characters: int) -> str:
        if characters <= 0 or characters >= self.character_count:
            return self.materialize()
        chunks: list[str] = []
        remaining = characters
        current: ContextText | None = self
        while current is not None and remaining:
            if current.chunk:
                piece = current.chunk[-remaining:]
                chunks.append(piece)
                remaining -= len(piece)
            current = current.parent
        return "".join(reversed(chunks))

    def __str__(self) -> str:
        return self.materialize()

    def __repr__(self) -> str:
        return f"ContextText(character_count={self.character_count})"


@dataclass(frozen=True)
class ChoiceSet:
    choice_set_id: str
    prompt_id: str
    aligned_step: int
    sampling_boundary: int
    context_token_sha256: str
    context_text_tail: str | ContextText
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

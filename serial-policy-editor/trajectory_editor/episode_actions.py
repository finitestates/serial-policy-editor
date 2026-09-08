"""The small policy language understood by the episode runtime.

These are the only commands that can change a generative trajectory.  Search,
menu expansion, review, notes, and projection are interface operations and are
therefore deliberately absent from this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from .domain import EditorError


@dataclass(frozen=True)
class Accept:
    kind: str = "accept"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}


@dataclass(frozen=True)
class SelectRawRank:
    rank: int
    kind: str = "select-raw-rank"

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 1:
            raise EditorError("raw rank must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "rank": self.rank}


@dataclass(frozen=True)
class Write:
    text: str
    mode: str = "continuation"
    kind: str = "write"

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise EditorError("write text must be nonempty")
        if self.mode not in {"continuation", "exact"}:
            raise EditorError("write mode must be continuation or exact")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text, "mode": self.mode}


@dataclass(frozen=True)
class Hold:
    limit: int
    boundary: str | None = None
    kind: str = "hold"

    def __post_init__(self) -> None:
        if type(self.limit) is not int or self.limit < 1:
            raise EditorError("hold limit must be a positive integer")
        if self.boundary not in {None, "sentence", "newline"}:
            raise EditorError("hold boundary must be sentence, newline, or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "limit": self.limit,
            "boundary": self.boundary,
        }


@dataclass(frozen=True)
class Finish:
    kind: str = "finish"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}


@dataclass(frozen=True)
class EndGeneration:
    """Select the highest-raw-ranked terminal token at this boundary."""

    kind: str = "end-generation"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}


PolicyAction: TypeAlias = Accept | SelectRawRank | Write | Hold | Finish | EndGeneration


def action_from_dict(raw: Mapping[str, Any]) -> PolicyAction:
    kind = raw.get("kind")
    if kind == "accept":
        return Accept()
    if kind in {"select", "select-raw-rank"}:
        rank = raw.get("rank", raw.get("selected_rank"))
        if type(rank) is not int:
            raise EditorError("select action has no valid raw rank")
        return SelectRawRank(rank)
    if kind in {"insert", "write"}:
        text = raw.get("text", raw.get("supplied_text"))
        mode = raw.get("mode", raw.get("insert_mode", "continuation"))
        if not isinstance(text, str) or not isinstance(mode, str):
            raise EditorError("write action is malformed")
        return Write(text, mode)
    if kind == "hold":
        limit = raw.get("limit", raw.get("requested_visible_tokens"))
        if type(limit) is not int:
            raise EditorError("hold action has no valid limit")
        boundary = raw.get("boundary")
        return Hold(limit, str(boundary) if boundary is not None else None)
    if kind == "finish":
        return Finish()
    if kind in {"teacher-eog", "end-generation"}:
        return EndGeneration()
    raise EditorError(f"unsupported policy action kind {kind!r}")

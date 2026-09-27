"""The small policy language understood by the episode runtime.

These actions are the durable semantic boundary between the menu/application
and the runtime. Search, rendering, persistence, and research controls are
deliberately not part of this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from typing import Any, TypeAlias

from .errors import EditorError
from .sampler_config import SamplerConfig
from .sampling import MAX_SEED, MIN_SEED


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


PHRASE_DEFAULT_MAX_TOKENS = 16
PHRASE_DEFAULT_MAX_SHIFT = 6.0


@dataclass(frozen=True)
class Phrase:
    """A bounded check or force phrase operation."""

    text: str
    mode: str = "continuation"
    force: bool = False
    max_tokens: int = PHRASE_DEFAULT_MAX_TOKENS
    max_shift: float = PHRASE_DEFAULT_MAX_SHIFT

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise EditorError("phrase text must be nonempty")
        if self.mode not in {"continuation", "exact"}:
            raise EditorError("phrase mode must be continuation or exact")
        if type(self.force) is not bool:
            raise EditorError("phrase force flag must be boolean")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise EditorError("phrase max_tokens must be a positive integer")
        if type(self.max_shift) not in (int, float) or not math.isfinite(float(self.max_shift)):
            raise EditorError("phrase max_shift must be a finite number")
        if self.max_shift < 0.0:
            raise EditorError("phrase max_shift must be nonnegative")

    @property
    def kind(self) -> str:
        return "force-phrase" if self.force else "check-phrase"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "mode": self.mode,
            "max_tokens": self.max_tokens,
            "max_shift": float(self.max_shift),
        }


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
class EndGeneration:
    """Select the highest-raw-ranked terminal token at this boundary."""

    kind: str = "end-generation"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}


@dataclass(frozen=True)
class Reroll:
    """Replace the draw seed, leaving prefix, policy, and boundary intact.

    The next proposal is drawn under the new seed at the same coordinates,
    mirroring the reference kernel's ``Reroll`` event so replay reproduces
    the same draws.  ``seed`` is a resolved integer, never an instruction
    to obtain entropy.
    """

    seed: int
    kind: str = "reroll"

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise EditorError("reroll seed must be an integer")
        if not MIN_SEED <= self.seed <= MAX_SEED:
            raise EditorError(f"reroll seed must be between {MIN_SEED} and {MAX_SEED} inclusive")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "seed": self.seed}


@dataclass(frozen=True)
class SetSampler:
    """Replace the complete sampler configuration at this action boundary."""

    sampling: SamplerConfig
    kind: str = "set-sampler"

    def __post_init__(self) -> None:
        if not isinstance(self.sampling, SamplerConfig):
            raise EditorError("sampler change requires a SamplerConfig")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "sampling": self.sampling.to_dict()}


PolicyAction: TypeAlias = (
    Accept | SelectRawRank | Write | Phrase | Hold | EndGeneration | Reroll | SetSampler
)


def sampler_after_action(sampling: SamplerConfig, action: PolicyAction) -> SamplerConfig:
    """Apply the sampler-state effect of one action, if it has one."""
    if isinstance(action, SetSampler):
        return action.sampling
    if isinstance(action, Reroll):
        return replace(sampling, seed=action.seed)
    return sampling


class UnsupportedPolicyActionKind(EditorError):
    """A well-formed action names a kind this runtime cannot execute."""


def action_from_dict(raw: Mapping[str, Any]) -> PolicyAction:
    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise EditorError("policy action kind must be a nonempty string")
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
    if kind in {"check-phrase", "force-phrase"}:
        text = raw.get("text", raw.get("supplied_text"))
        mode = raw.get("mode", "continuation")
        if not isinstance(text, str) or not isinstance(mode, str):
            raise EditorError("phrase action is malformed")
        return Phrase(
            text,
            mode,
            force=kind == "force-phrase",
            max_tokens=raw.get("max_tokens", PHRASE_DEFAULT_MAX_TOKENS),
            max_shift=raw.get("max_shift", PHRASE_DEFAULT_MAX_SHIFT),
        )
    if kind == "hold":
        limit = raw.get("limit", raw.get("requested_visible_tokens"))
        if type(limit) is not int:
            raise EditorError("hold action has no valid limit")
        boundary = raw.get("boundary")
        return Hold(limit, str(boundary) if boundary is not None else None)
    if kind in {"teacher-eog", "end-generation"}:
        return EndGeneration()
    if kind == "reroll":
        seed = raw.get("seed")
        if type(seed) is not int:
            raise EditorError("reroll action has no valid seed")
        return Reroll(seed)
    if kind == "set-sampler":
        sampling = raw.get("sampling")
        if not isinstance(sampling, Mapping):
            raise EditorError("sampler change has no valid sampling configuration")
        return SetSampler(SamplerConfig.from_record(sampling))
    raise UnsupportedPolicyActionKind(f"unsupported policy action kind {kind!r}")

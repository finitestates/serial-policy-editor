"""Stable observations of actions and replay outcomes.

These records are part of the episode contract. They contain no persistence,
UI, or backend implementation details.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .actions import PolicyAction
from .errors import EditorError


@dataclass(frozen=True)
class ReplayExpectation:
    """Recorded result used to test whether an action still means the same thing."""

    token_ids: tuple[int, ...]
    terminal_token_id: int | None = None
    stop_reason: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ReplayExpectation":
        values = raw.get("token_ids") or ()
        if not isinstance(values, (list, tuple)) or any(
            type(value) is not int for value in values
        ):
            raise EditorError("replay expectation token ids are malformed")
        terminal = raw.get("terminal_token_id")
        if terminal is not None and type(terminal) is not int:
            raise EditorError("replay expectation terminal token is malformed")
        stop = raw.get("stop_reason")
        if stop is not None and not isinstance(stop, str):
            raise EditorError("replay expectation stop reason is malformed")
        return cls(tuple(int(value) for value in values), terminal, stop)

    @property
    def resolved_token_ids(self) -> tuple[int, ...]:
        if self.terminal_token_id is None:
            return self.token_ids
        return (*self.token_ids, self.terminal_token_id)


@dataclass(frozen=True)
class Divergence:
    boundary: int
    action_kind: str
    reason: str
    expected_token_id: int | None
    actual_token_id: int | None
    expected_stop_reason: str | None = None
    actual_stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary,
            "action_kind": self.action_kind,
            "reason": self.reason,
            "expected_token_id": self.expected_token_id,
            "actual_token_id": self.actual_token_id,
            "expected_stop_reason": self.expected_stop_reason,
            "actual_stop_reason": self.actual_stop_reason,
        }


@dataclass(frozen=True)
class TokenEvidence:
    boundary: int
    sampling_coordinate: int
    token_id: int
    text: str
    proposal_token_id: int
    raw_model_nll: float
    raw_rank: int
    policy_rank: int
    decoder_probability: float
    proposal_agreement: bool
    is_eog: bool
    realized_visible: bool


@dataclass(frozen=True)
class ActionOutcome:
    action: PolicyAction
    boundary_before: int
    boundary_after: int
    resolved_text: str
    resolved_token_ids: tuple[int, ...]
    visible_token_ids: tuple[int, ...]
    terminal_token_id: int | None
    stop_reason: str
    evidence: tuple[TokenEvidence, ...]
    status: str = "completed"
    divergence: Divergence | None = None
    replay_eog_token_id: int | None = None
    diagnostics: Mapping[str, Any] | None = None

    def expectation(self) -> ReplayExpectation:
        return ReplayExpectation(
            self.visible_token_ids,
            self.terminal_token_id,
            self.stop_reason,
        )

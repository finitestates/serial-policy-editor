"""Thin Python adapter for the experimental Rust episode-history kernel.

Each public operation serializes its typed inputs once, makes one extension
call, then rebuilds the existing Python-facing value objects once from the
result. The released episode engine continues to use its Python kernel.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from . import _native

from trajectory_editor.core.actions import (
    PolicyAction,
    SetSampler,
    action_from_dict as python_action_from_dict,
)
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.results import (
    ActionOutcome,
    Divergence,
    ReplayExpectation,
    TokenEvidence,
)
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_history import (
    EpisodeHistory,
    HistoryTruncation,
    RecordedAttempt,
)


def _record_expectation(expectation: ReplayExpectation | None) -> dict[str, Any] | None:
    if expectation is None:
        return None
    return {
        "token_ids": list(expectation.token_ids),
        "terminal_token_id": expectation.terminal_token_id,
        "stop_reason": expectation.stop_reason,
    }


def _record_outcome(outcome: ActionOutcome) -> dict[str, Any]:
    return {
        "action": outcome.action.to_dict(),
        "boundary_before": outcome.boundary_before,
        "boundary_after": outcome.boundary_after,
        "resolved_text": outcome.resolved_text,
        "resolved_token_ids": list(outcome.resolved_token_ids),
        "visible_token_ids": list(outcome.visible_token_ids),
        "terminal_token_id": outcome.terminal_token_id,
        "stop_reason": outcome.stop_reason,
        "evidence": [
            {field.name: getattr(item, field.name) for field in fields(TokenEvidence)}
            for item in outcome.evidence
        ],
        "status": outcome.status,
        "divergence": (
            None
            if outcome.divergence is None
            else {
                field.name: getattr(outcome.divergence, field.name)
                for field in fields(Divergence)
            }
        ),
        "replay_eog_token_id": outcome.replay_eog_token_id,
        "diagnostics": (
            dict(outcome.diagnostics)
            if isinstance(outcome.diagnostics, Mapping)
            else outcome.diagnostics
        ),
    }


def _record_attempt(attempt: RecordedAttempt) -> dict[str, Any]:
    return {
        "ordinal": attempt.ordinal,
        "action": attempt.action.to_dict(),
        "outcome": _record_outcome(attempt.outcome),
        "expectation": _record_expectation(attempt.expectation),
    }


def _record_history(history: EpisodeHistory) -> dict[str, Any]:
    if not isinstance(history, EpisodeHistory):
        raise TypeError("history must be an EpisodeHistory")
    return {"attempts": [_record_attempt(attempt) for attempt in history.attempts]}


def _raise_error(error: Mapping[str, Any]) -> None:
    message = str(error.get("message", "Rust history operation failed"))
    if error.get("kind") == "unsupported-action-kind":
        from trajectory_editor.core.actions import UnsupportedPolicyActionKind

        raise UnsupportedPolicyActionKind(message)
    if error.get("kind") == "invalid-action":
        raise EditorError(message)
    raise ValueError(message)


def _execute(request: dict[str, Any]) -> Any:
    """Cross the extension boundary once for a complete logical operation."""

    try:
        wire = json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"history operation contains non-JSON data: {exc}") from exc
    response = json.loads(_native.execute_json(wire))
    if not response.get("ok"):
        _raise_error(response.get("error", {}))
    return response["result"]


def _action(raw: Mapping[str, Any]):
    return python_action_from_dict(raw)


def _expectation(raw: Mapping[str, Any] | None) -> ReplayExpectation | None:
    return None if raw is None else ReplayExpectation.from_mapping(raw)


def _from_attempt(raw: Mapping[str, Any]) -> RecordedAttempt:
    action = _action(raw["action"])
    outcome_raw = raw["outcome"]
    outcome_action = _action(outcome_raw["action"])
    evidence = tuple(TokenEvidence(**item) for item in outcome_raw["evidence"])
    divergence_raw = outcome_raw.get("divergence")
    divergence = None if divergence_raw is None else Divergence(**divergence_raw)
    outcome = ActionOutcome(
        action=outcome_action,
        boundary_before=outcome_raw["boundary_before"],
        boundary_after=outcome_raw["boundary_after"],
        resolved_text=outcome_raw["resolved_text"],
        resolved_token_ids=tuple(outcome_raw["resolved_token_ids"]),
        visible_token_ids=tuple(outcome_raw["visible_token_ids"]),
        terminal_token_id=outcome_raw["terminal_token_id"],
        stop_reason=outcome_raw["stop_reason"],
        evidence=evidence,
        status=outcome_raw["status"],
        divergence=divergence,
        replay_eog_token_id=outcome_raw["replay_eog_token_id"],
        diagnostics=outcome_raw["diagnostics"],
    )
    return RecordedAttempt(
        ordinal=raw["ordinal"],
        action=action,
        outcome=outcome,
        expectation=_expectation(raw.get("expectation")),
    )


def parse_action(raw: Mapping[str, Any]):
    """Parse an action with Python's sampler validator and Rust action model."""

    if not isinstance(raw, Mapping):
        raise TypeError("policy action must be a mapping")
    action_record = dict(raw)
    # SamplerConfig has compatibility checks that intentionally remain Python's
    # responsibility. Send its validated canonical object across with the action.
    if action_record.get("kind") == "set-sampler":
        action_record = python_action_from_dict(action_record).to_dict()
    return _action(_execute({"operation": "action", "action": action_record}))


def visible(history: EpisodeHistory) -> dict[str, Any]:
    """Return the validated visible projections from one complete history."""

    result = _execute({"operation": "visible", "history": _record_history(history)})
    result["visible_token_ids"] = tuple(result["visible_token_ids"])
    result["visible_token_evidence"] = tuple(
        TokenEvidence(**item) for item in result["visible_token_evidence"]
    )
    return result


def truncate(
    history: EpisodeHistory,
    boundary: int,
    *,
    include_boundary_events: bool = False,
) -> HistoryTruncation:
    """Apply one root-relative truncation through the typed Rust kernel."""

    raw = _execute(
        {
            "operation": "truncate",
            "history": _record_history(history),
            "boundary": boundary,
            # Python's reference operation uses normal truth-value testing for
            # this optional flag, so normalize that once at the adapter edge.
            "include_boundary_events": bool(include_boundary_events),
        }
    )
    return HistoryTruncation(
        requested_boundary=raw["requested_boundary"],
        retained=EpisodeHistory(tuple(_from_attempt(item) for item in raw["retained"]["attempts"])),
        discarded=tuple(_from_attempt(item) for item in raw["discarded"]),
        partial=None if raw["partial"] is None else _from_attempt(raw["partial"]),
    )


def sampler_after_action(sampling: SamplerConfig, action: PolicyAction) -> SamplerConfig:
    """Apply reroll or whole-config replacement, then validate in Python."""

    if not isinstance(sampling, SamplerConfig):
        raise TypeError("sampling must be a SamplerConfig")
    if not hasattr(action, "to_dict"):
        raise TypeError("action must be a policy action")
    result = _execute(
        {
            "operation": "sampler-after-action",
            "sampling": sampling.to_dict(),
            "action": action.to_dict(),
        }
    )
    return SamplerConfig.from_record(result)


__all__ = ["parse_action", "sampler_after_action", "truncate", "visible"]

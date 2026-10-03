#!/usr/bin/env python3
"""Generate shared cases and expected results from the Python reference."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "core" / "src"))

from trajectory_editor.core.actions import (  # noqa: E402
    Accept,
    EndGeneration,
    Hold,
    Phrase,
    Reroll,
    SelectRawRank,
    SetSampler,
    UnsupportedPolicyActionKind,
    Write,
    action_from_dict,
    sampler_after_action,
)
from trajectory_editor.core.errors import EditorError  # noqa: E402
from trajectory_editor.core.results import (  # noqa: E402
    ActionOutcome,
    Divergence,
    ReplayExpectation,
    TokenEvidence,
)
from trajectory_editor.core.sampler_config import SamplerConfig  # noqa: E402
from trajectory_editor.episode_history import (  # noqa: E402
    EpisodeHistory,
    HistoryTruncation,
    RecordedAttempt,
)


def evidence(
    boundary: int,
    token_id: int,
    text: str,
    *,
    visible: bool = True,
    is_eog: bool = False,
) -> TokenEvidence:
    return TokenEvidence(
        boundary=boundary,
        sampling_boundary=boundary,
        token_id=token_id,
        text=text,
        proposal_token_id=token_id,
        raw_model_nll=0.125,
        raw_rank=2,
        policy_rank=1,
        decoder_probability=0.75,
        proposal_agreement=True,
        is_eog=is_eog,
        realized_visible=visible,
    )


def make_attempt(
    ordinal: int,
    action,
    before: int,
    pieces: tuple[tuple[int, str], ...] = (),
    *,
    status: str = "completed",
    terminal_token_id: int | None = None,
    divergence: Divergence | None = None,
    replay_eog_token_id: int | None = None,
    diagnostics: dict[str, Any] | None = None,
    expectation: ReplayExpectation | None = None,
):
    visible_ids = tuple(token_id for token_id, _ in pieces)
    visible_text = "".join(text for _, text in pieces)
    after = before + len(pieces)
    items = [evidence(before + index, token_id, text) for index, (token_id, text) in enumerate(pieces)]
    if terminal_token_id is not None:
        items.append(
            evidence(after, terminal_token_id, "<eog>", visible=False, is_eog=True)
        )
    outcome = ActionOutcome(
        action=action,
        boundary_before=before,
        boundary_after=after,
        resolved_text=visible_text,
        resolved_token_ids=(
            visible_ids
            if terminal_token_id is None
            else (*visible_ids, terminal_token_id)
        ),
        visible_token_ids=visible_ids,
        terminal_token_id=terminal_token_id,
        stop_reason="eog" if terminal_token_id is not None else "completed",
        evidence=tuple(items),
        status=status,
        divergence=divergence,
        replay_eog_token_id=replay_eog_token_id,
        diagnostics=diagnostics,
    )
    return RecordedAttempt(ordinal, action, outcome, expectation)


def record_expectation(value: ReplayExpectation | None):
    if value is None:
        return None
    return {
        "token_ids": list(value.token_ids),
        "terminal_token_id": value.terminal_token_id,
        "stop_reason": value.stop_reason,
    }


def record_outcome(value: ActionOutcome):
    return {
        "action": value.action.to_dict(),
        "boundary_before": value.boundary_before,
        "boundary_after": value.boundary_after,
        "resolved_text": value.resolved_text,
        "resolved_token_ids": list(value.resolved_token_ids),
        "visible_token_ids": list(value.visible_token_ids),
        "terminal_token_id": value.terminal_token_id,
        "stop_reason": value.stop_reason,
        "evidence": [
            {field.name: getattr(item, field.name) for field in fields(TokenEvidence)}
            for item in value.evidence
        ],
        "status": value.status,
        "divergence": (
            None
            if value.divergence is None
            else {
                field.name: getattr(value.divergence, field.name)
                for field in fields(Divergence)
            }
        ),
        "replay_eog_token_id": value.replay_eog_token_id,
        "diagnostics": value.diagnostics,
    }


def record_attempt(value: RecordedAttempt):
    return {
        "ordinal": value.ordinal,
        "action": value.action.to_dict(),
        "outcome": record_outcome(value.outcome),
        "expectation": record_expectation(value.expectation),
    }


def record_history(value: EpisodeHistory):
    return {"attempts": [record_attempt(item) for item in value.attempts]}


def record_truncation(value: HistoryTruncation):
    return {
        "requested_boundary": value.requested_boundary,
        "retained": record_history(value.retained),
        "discarded": [record_attempt(item) for item in value.discarded],
        "partial": None if value.partial is None else record_attempt(value.partial),
    }


def history_from_record(raw: dict[str, Any]) -> EpisodeHistory:
    attempts = []
    for item in raw["attempts"]:
        action = action_from_dict(item["action"])
        outcome_raw = item["outcome"]
        outcome_action = action_from_dict(outcome_raw["action"])
        outcome = ActionOutcome(
            action=outcome_action,
            boundary_before=outcome_raw["boundary_before"],
            boundary_after=outcome_raw["boundary_after"],
            resolved_text=outcome_raw["resolved_text"],
            resolved_token_ids=tuple(outcome_raw["resolved_token_ids"]),
            visible_token_ids=tuple(outcome_raw["visible_token_ids"]),
            terminal_token_id=outcome_raw["terminal_token_id"],
            stop_reason=outcome_raw["stop_reason"],
            evidence=tuple(TokenEvidence(**evidence_raw) for evidence_raw in outcome_raw["evidence"]),
            status=outcome_raw["status"],
            divergence=(
                None
                if outcome_raw["divergence"] is None
                else Divergence(**outcome_raw["divergence"])
            ),
            replay_eog_token_id=outcome_raw["replay_eog_token_id"],
            diagnostics=outcome_raw["diagnostics"],
        )
        expectation_raw = item.get("expectation")
        expectation = (
            None
            if expectation_raw is None
            else ReplayExpectation.from_mapping(expectation_raw)
        )
        attempts.append(
            RecordedAttempt(item["ordinal"], action, outcome, expectation)
        )
    return EpisodeHistory(tuple(attempts))


def _error(kind: str, message: str):
    return {"ok": False, "error": {"kind": kind, "message": message}}


def python_result(request: dict[str, Any]):
    """Execute a fixture using the current Python implementation."""

    operation = request.get("operation")
    try:
        if operation == "action":
            result = action_from_dict(request["action"]).to_dict()
        elif operation == "sampler-after-action":
            config = SamplerConfig.from_record(request["sampling"])
            action = action_from_dict(request["action"])
            result = sampler_after_action(config, action).to_dict()
        else:
            history = history_from_record(request["history"])
            if operation == "visible":
                result = {
                    "current_boundary": history.current_boundary,
                    "visible_token_ids": list(history.visible_token_ids),
                    "visible_token_evidence": [
                        {field.name: getattr(item, field.name) for field in fields(TokenEvidence)}
                        for item in history.visible_token_evidence
                    ],
                    "visible_text": history.visible_text,
                }
            elif operation == "truncate":
                result = record_truncation(
                    history.truncate(
                        request["boundary"],
                        include_boundary_events=request.get(
                            "include_boundary_events", False
                        ),
                    )
                )
            else:
                return _error(
                    "invalid-request", f"unsupported history operation {operation!r}"
                )
        return {"ok": True, "result": result}
    except UnsupportedPolicyActionKind as exc:
        return _error("unsupported-action-kind", str(exc))
    except EditorError as exc:
        return _error("invalid-action", str(exc))
    except (TypeError, ValueError) as exc:
        if operation == "truncate":
            boundary = request.get("boundary")
            if type(boundary) is not int or boundary < 0:
                return _error("invalid-boundary", str(exc))
            if isinstance(exc, ValueError) and "retained boundary" in str(exc):
                return _error("invalid-boundary", str(exc))
        return _error("invalid-history", str(exc))


def _case(name: str, request: dict[str, Any]):
    return {"name": name, "request": request, "expected": python_result(request)}


def build_cases():
    cases = []

    cases.append(_case("empty-history-at-zero", {
        "operation": "truncate",
        "history": {"attempts": []},
        "boundary": 0,
    }))
    cases.append(_case("empty-visible-projection", {
        "operation": "visible",
        "history": {"attempts": []},
    }))

    zero_history = EpisodeHistory((
        make_attempt(0, Accept(), 0),
        make_attempt(1, Hold(2), 0, ((11, "A"), (12, "B"))),
        make_attempt(
            2,
            Phrase("handoff", mode="exact"),
            2,
            status="handed-off",
            diagnostics={"operation": "check-phrase", "note": "retained raw"},
        ),
        make_attempt(3, Write("C", mode="exact"), 2, ((13, "C"),)),
        make_attempt(4, Accept(), 3),
    ))
    zero_record = record_history(zero_history)
    for name, boundary, include in (
        ("zero-width-before-boundary", 2, False),
        ("zero-width-at-boundary-excluded", 2, False),
        ("zero-width-at-boundary-included", 2, True),
        ("boundary-zero-excludes-root-events", 0, False),
        ("boundary-zero-includes-root-events", 0, True),
        ("current-boundary-excludes-events", 3, False),
        ("current-boundary-includes-events", 3, True),
    ):
        # The same request includes events both before and at the selected cut;
        # these names expose each distinction to the parity report.
        cases.append(_case(name, {
            "operation": "truncate",
            "history": zero_record,
            "boundary": boundary,
            "include_boundary_events": include,
        }))
    cases.append(_case("visible-evidence-ignores-handoff", {
        "operation": "visible",
        "history": zero_record,
    }))

    phrase = Phrase("original", mode="continuation")
    divergence = Divergence(
        boundary=2,
        action_kind=phrase.kind,
        reason="replay changed",
        expected_token_id=8,
        actual_token_id=80,
        expected_stop_reason="completed",
        actual_stop_reason="eog",
    )
    partial_phrase = EpisodeHistory((
        make_attempt(0, Hold(1), 0, ((7, "P"),)),
        make_attempt(
            1,
            phrase,
            1,
            ((8, "one"), (9, " "), (10, "two")),
            status="completed-with-divergence",
            terminal_token_id=99,
            divergence=divergence,
            replay_eog_token_id=99,
            diagnostics={"discarded_suffix": ["two", "<eog>"]},
            expectation=ReplayExpectation((8, 9, 10), 99, "eog"),
        ),
    ))
    cases.append(_case("partial-phrase-clears-suffix-metadata", {
        "operation": "truncate",
        "history": record_history(partial_phrase),
        "boundary": 2,
    }))

    partial_write_action = Write("old wording", mode="continuation")
    partial_write = EpisodeHistory((
        make_attempt(
            0,
            partial_write_action,
            0,
            ((14, "new"), (15, " "), (16, "wording")),
        ),
    ))
    cases.append(_case("partial-write-becomes-exact-write", {
        "operation": "truncate",
        "history": record_history(partial_write),
        "boundary": 2,
    }))

    partial_hold_action = Hold(5, boundary="sentence")
    partial_hold = EpisodeHistory((
        make_attempt(
            0,
            partial_hold_action,
            0,
            ((17, "a"), (18, "b"), (19, "c")),
        ),
    ))
    cases.append(_case("partial-hold-drops-original-limit-and-boundary", {
        "operation": "truncate",
        "history": record_history(partial_hold),
        "boundary": 2,
    }))

    selected = SelectRawRank(4)
    partial_other = EpisodeHistory((
        make_attempt(
            5,
            selected,
            0,
            ((21, "x"), (22, "y"), (23, "z")),
            status="completed-with-divergence",
            terminal_token_id=98,
            replay_eog_token_id=98,
            diagnostics={"suffix": "discard this"},
        ),
    ))
    cases.append(_case("partial-token-action-becomes-hold", {
        "operation": "truncate",
        "history": record_history(partial_other),
        "boundary": 2,
    }))

    invalid_ordinal = json.loads(json.dumps(record_history(EpisodeHistory((
        make_attempt(0, Hold(1), 0, ((31, "x"),)),
    )))))
    invalid_ordinal["attempts"][0]["ordinal"] = True
    cases.append(_case("invalid-boolean-ordinal", {
        "operation": "visible", "history": invalid_ordinal,
    }))
    invalid_gap = record_history(EpisodeHistory((
        make_attempt(0, Hold(1), 0, ((31, "x"),)),
        make_attempt(1, Hold(1), 1, ((32, "y"),)),
    )))
    invalid_gap["attempts"][1]["outcome"]["boundary_before"] = 2
    invalid_gap["attempts"][1]["outcome"]["boundary_after"] = 3
    invalid_gap["attempts"][1]["outcome"]["evidence"][0]["boundary"] = 2
    invalid_gap["attempts"][1]["outcome"]["evidence"][0]["sampling_boundary"] = 2
    cases.append(_case("invalid-root-relative-gap", {
        "operation": "visible", "history": invalid_gap,
    }))
    invalid_visible = record_history(EpisodeHistory((
        make_attempt(0, Hold(1), 0, ((31, "x"),)),
    )))
    invalid_visible["attempts"][0]["outcome"]["evidence"][0]["token_id"] = 999
    cases.append(_case("invalid-visible-token-evidence", {
        "operation": "visible", "history": invalid_visible,
    }))
    invalid_ordered_evidence = record_history(EpisodeHistory((
        make_attempt(0, Hold(2), 0, ((31, "x"), (32, "y"))),
    )))
    invalid_ordered_evidence["attempts"][0]["outcome"]["evidence"][1]["boundary"] = 0
    cases.append(_case("invalid-visible-evidence-order", {
        "operation": "visible", "history": invalid_ordered_evidence,
    }))
    invalid_ordinal_order = record_history(EpisodeHistory((
        make_attempt(2, Hold(1), 0, ((31, "x"),)),
        make_attempt(4, Hold(1), 1, ((32, "y"),)),
    )))
    invalid_ordinal_order["attempts"][1]["ordinal"] = 2
    cases.append(_case("invalid-duplicate-ordinal", {
        "operation": "visible", "history": invalid_ordinal_order,
    }))
    invalid_negative_ordinal = json.loads(json.dumps(invalid_ordinal_order))
    invalid_negative_ordinal["attempts"][0]["ordinal"] = -1
    cases.append(_case("invalid-negative-ordinal", {
        "operation": "visible", "history": invalid_negative_ordinal,
    }))
    invalid_boolean_boundary = record_history(EpisodeHistory((
        make_attempt(0, Hold(1), 0, ((31, "x"),)),
    )))
    invalid_boolean_boundary["attempts"][0]["outcome"]["boundary_before"] = True
    cases.append(_case("invalid-boolean-history-boundary", {
        "operation": "visible", "history": invalid_boolean_boundary,
    }))
    malformed_action_history = record_history(EpisodeHistory((
        make_attempt(0, Hold(1), 0, ((31, "x"),)),
    )))
    malformed_action = {"kind": "select", "rank": True}
    malformed_action_history["attempts"][0]["action"] = malformed_action
    malformed_action_history["attempts"][0]["outcome"]["action"] = malformed_action
    cases.append(_case("malformed-known-action-in-history", {
        "operation": "visible", "history": malformed_action_history,
    }))
    unknown_action_history = record_history(EpisodeHistory((
        make_attempt(0, Hold(1), 0, ((31, "x"),)),
    )))
    unknown_action = {"kind": "finish"}
    unknown_action_history["attempts"][0]["action"] = unknown_action
    unknown_action_history["attempts"][0]["outcome"]["action"] = unknown_action
    cases.append(_case("unknown-action-in-history", {
        "operation": "visible", "history": unknown_action_history,
    }))

    # Canonical values and aliases accepted by action_from_dict().
    action_cases = [
        ("action-accept", {"kind": "accept"}),
        ("action-select-alias", {"kind": "select", "selected_rank": 3}),
        ("action-insert-alias", {"kind": "insert", "supplied_text": " hi", "insert_mode": "exact"}),
        ("action-write", {"kind": "write", "text": " there", "mode": "continuation"}),
        ("action-check-phrase", {"kind": "check-phrase", "text": "needle", "mode": "exact", "max_tokens": 7, "max_shift": 2.25}),
        ("action-force-phrase", {"kind": "force-phrase", "text": "needle", "mode": "continuation"}),
        ("action-hold-alias", {"kind": "hold", "requested_visible_tokens": 2, "boundary": "newline"}),
        ("action-teacher-eog-alias", {"kind": "teacher-eog"}),
        ("action-end-generation", {"kind": "end-generation"}),
        ("action-reroll-min-seed", {"kind": "reroll", "seed": -(1 << 63)}),
        ("action-reroll-max-seed", {"kind": "reroll", "seed": (1 << 63) - 1}),
        ("action-set-sampler", {"kind": "set-sampler", "sampling": SamplerConfig(temperature=0.4, top_k=13, seed=77).to_dict()}),
        ("action-invalid-bool-rank", {"kind": "select", "rank": True}),
        ("action-invalid-reroll-bool", {"kind": "reroll", "seed": True}),
        ("action-invalid-hold-boundary", {"kind": "hold", "limit": 1, "boundary": "word"}),
        ("action-invalid-phrase-mode", {"kind": "check-phrase", "text": "x", "mode": "loose"}),
        ("action-unknown-kind", {"kind": "finish"}),
        ("action-malformed-write-mode", {"kind": "write", "text": "x", "mode": "loose"}),
    ]
    cases.extend(
        _case(name, {"operation": "action", "action": action})
        for name, action in action_cases
    )

    initial = SamplerConfig(
        temperature=0.35,
        top_k=17,
        seed=12,
        presence_penalty=0.25,
        frequency_penalty=0.5,
        repeat_last_n=23,
    )
    replacement = SamplerConfig(
        temperature=0.0,
        top_k=5,
        seed=91,
        cfg_unconditional_prompt="separate context",
        cfg_scale=1.75,
        presence_penalty=-0.125,
    )
    cases.append(_case("sampler-reroll-changes-only-seed", {
        "operation": "sampler-after-action",
        "sampling": initial.to_dict(),
        "action": Reroll(-(1 << 63)).to_dict(),
    }))
    cases.append(_case("sampler-set-replaces-complete-config", {
        "operation": "sampler-after-action",
        "sampling": initial.to_dict(),
        "action": SetSampler(replacement).to_dict(),
    }))

    for name, boundary in (
        ("truncate-boolean-boundary", True),
        ("truncate-fractional-boundary", 1.5),
        ("truncate-negative-boundary", -1),
        ("truncate-past-current-boundary", 4),
    ):
        cases.append(_case(name, {
            "operation": "truncate",
            "history": zero_record,
            "boundary": boundary,
        }))

    return cases


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "fixtures" / "history-cases.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"cases": build_cases()}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

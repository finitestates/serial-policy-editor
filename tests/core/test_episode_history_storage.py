from __future__ import annotations

import pytest

from trajectory_editor.episode_history import materialize_stored_prefix

pytestmark = pytest.mark.invariant

def test_stored_prefix_keeps_absolute_boundaries_when_cutting_a_hold():
    actions = [
        {
            "ordinal": 0,
            "boundary_before": 0,
            "boundary_after": 2,
            "kind": "hold",
            "arguments": {"kind": "hold", "limit": 2, "boundary": None},
            "resolved_text": " A B",
            "status": "completed",
            "stop_reason": "requested-length",
            "mismatch": None,
        }
    ]
    tokens = [
        {"action_ordinal": 0, "boundary": 0, "text": " A", "realized_visible": 1},
        {"action_ordinal": 0, "boundary": 1, "text": " B", "realized_visible": 1},
    ]

    prefix = materialize_stored_prefix(
        actions,
        tokens,
        [{"start_boundary": 0, "sampling_json": "{}"}],
        [{"start_boundary": 0, "max_tokens": None, "checkpoint_boundary": None}],
        1,
    )

    action = prefix.actions[0]
    assert action.boundary_before == 0
    assert action.boundary_after == 1
    assert action.arguments == {"kind": "hold", "limit": 1, "boundary": None}
    assert [token["boundary"] for token in action.tokens] == [0]
    assert prefix.source_boundary == 1


def test_stored_prefix_turns_a_partial_phrase_into_an_exact_write():
    source_action = {
        "ordinal": 0,
        "boundary_before": 0,
        "boundary_after": 3,
        "kind": "check-phrase",
        "arguments": {
            "kind": "check-phrase",
            "text": " A B C",
            "mode": "continuation",
            "max_tokens": 16,
            "max_shift": 6.0,
            "diagnostics": {"retained": True},
        },
        "resolved_text": " A B C",
        "status": "completed",
        "stop_reason": "completed",
        "mismatch": {"reason": "old divergence"},
    }
    tokens = [
        {
            "action_ordinal": 0,
            "boundary": index,
            "token_id": index + 1,
            "text": text,
            "realized_visible": 1,
        }
        for index, text in enumerate((" A", " B", " C"))
    ]

    prefix = materialize_stored_prefix(
        [source_action],
        tokens,
        (),
        (),
        2,
    )

    action = prefix.actions[0]
    assert action.kind == "write"
    assert action.arguments["text"] == " A B"
    assert action.arguments["mode"] == "exact"
    assert action.arguments["original_action"] == source_action["arguments"]
    assert action.mismatch is None
    assert [token["token_id"] for token in action.tokens] == [1, 2]

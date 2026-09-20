from __future__ import annotations

from trajectory_editor.fork_materializer import materialize_stored_prefix


def test_stored_prefix_materializer_keeps_absolute_boundaries_when_cutting_an_action():
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

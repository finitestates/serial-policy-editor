from __future__ import annotations

import pytest

from trajectory_editor.domain import EditorError
from trajectory_editor.tui import CommandKind, parse_command


def parse(raw: str):
    return parse_command(raw, menu_size=12, default_hold_tokens=24, vocabulary_size=1000)


@pytest.mark.parametrize("target", ("nautical", "@nautical", "scene.motion"))
@pytest.mark.parametrize("state", ("on", "off"))
def test_research_group_learning_toggle_is_parsed_by_research_surface(target, state):
    command = parse(f"b {target} learn {state}")
    assert command.kind == CommandKind.BIAS
    assert command.bias_group_name == target
    assert command.bias_learnable is (state == "on")
    assert command.bias_group_members is None


@pytest.mark.parametrize(
    "raw",
    (
        "b nautical learn",
        "b nautical learn yes",
        "b nautical learn on after dragon",
        "b nautical learn off after dragon",
        "b nautical learn off 2",
    ),
)
def test_research_group_learning_toggle_rejects_malformed_forms(raw):
    with pytest.raises(EditorError):
        parse(raw)

"""Pure geometry contracts for the small terminal split allocator."""

from __future__ import annotations

import pytest
from trajectory_editor.term.layout import Rect, SplitItem, allocate_vertical_split


def _assert_vertical_partition(parent: Rect, regions: dict[str, Rect]) -> None:
    cursor = parent.y
    for region in regions.values():
        assert region.x == parent.x
        assert region.width == parent.width
        assert region.y == cursor
        cursor += region.height
    assert cursor == parent.y + parent.height


def test_vertical_split_accounts_for_content_gutter_and_flexible_candidate_area():
    parent = Rect(3, 4, 40, 12)
    items = [
        SplitItem("heading", 1, 1, priority=0),
        SplitItem("context_pane", 2, 3, priority=1),
        SplitItem("preview", 1, 1, priority=2),
        SplitItem("table", 2, flex=True, priority=3),
        SplitItem("command", 1, 1, priority=0),
        SplitItem("hint", 1, 1, priority=0),
    ]

    first = allocate_vertical_split(parent, items)
    second = allocate_vertical_split(parent, items)

    assert first == second
    _assert_vertical_partition(parent, first)
    assert first["heading"] == Rect(3, 4, 40, 1)
    assert first["context_pane"] == Rect(3, 5, 40, 3)
    assert first["table"].height == 5
    assert first["command"].y == 14
    assert first["hint"].y == 15

    context = allocate_vertical_split(first["context_pane"], [
        SplitItem("context", 1, 2, priority=0),
        SplitItem("divider", 1, 1, priority=1),
    ])
    _assert_vertical_partition(first["context_pane"], context)
    assert context["context"] == Rect(3, 5, 40, 2)
    assert context["divider"] == Rect(3, 7, 40, 1)


def test_compact_split_drops_the_context_and_gutter_together_before_controls():
    parent = Rect(0, 0, 40, 7)
    regions = allocate_vertical_split(parent, [
        SplitItem("heading", 1, 1, priority=4),
        SplitItem("context_pane", 2, 5, priority=4),
        SplitItem("preview", 1, 1, priority=2),
        SplitItem("table", 2, flex=True, priority=1),
        SplitItem("command", 1, 1, priority=0),
        SplitItem("hint", 1, 1, priority=3),
    ])

    _assert_vertical_partition(parent, regions)
    assert regions["command"].height == 1
    assert regions["table"].height == 3
    assert regions["hint"].height == 1
    assert regions["heading"].height == 1
    assert regions["context_pane"].height == 0


def test_split_rejects_gaps_and_duplicate_names():
    with pytest.raises(ValueError, match="vertical split leaves 2 rows"):
        allocate_vertical_split(Rect(0, 0, 10, 4), [
            SplitItem("top", 1, 1),
            SplitItem("bottom", 1, 1),
        ])
    with pytest.raises(ValueError, match="names must be unique"):
        allocate_vertical_split(Rect(0, 0, 10, 2), [
            SplitItem("same", 1, 1),
            SplitItem("same", 1, 1),
        ])


def test_split_rejects_invalid_rectangles_and_child_sizing():
    with pytest.raises(ValueError, match="dimensions must be nonnegative"):
        Rect(0, 0, 1, -1)
    with pytest.raises(ValueError, match="preferred must be at least minimum"):
        allocate_vertical_split(Rect(0, 0, 10, 2), [
            SplitItem("bad", 2, 1),
        ])

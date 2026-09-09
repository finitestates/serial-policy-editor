"""Compare cached wrapping with the original character-by-character renderer."""
from unittest.mock import patch

import pytest
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.utils import get_cwidth
from trajectory_editor import live_tui


def reference_rows(context: str, proposal: str, width: int) -> list[StyleAndTextTuples]:
    """Wrap styled context into terminal rows, preserving the proposal highlight."""
    rows: list[StyleAndTextTuples] = [[]]
    column = 0
    for style, text in (("", context), ("class:proposal", proposal)):
        for char in text:
            if char == "\n":
                rows.append([])
                column = 0
                continue
            rendered = " " * (8 - column % 8) if char == "\t" else char
            for glyph in rendered:
                cells = max(0, get_cwidth(glyph))
                if column + cells > width:
                    rows.append([])
                    column = 0
                if rows[-1] and rows[-1][-1][0] == style:
                    previous_style, previous_text = rows[-1][-1]
                    rows[-1][-1] = (previous_style, previous_text + glyph)
                else:
                    rows[-1].append((style, glyph))
                column += cells
    return rows


@pytest.mark.parametrize("context", ["", "abc", "abcd", "a\n", "\n\n", "界é\txyz", "x" * 1000])
@pytest.mark.parametrize("proposal", ["", "!", "界\tA\nB", "é", "\n"])
@pytest.mark.parametrize("width", [1, 4, 17])
def test_cached_rows_match_original(context, proposal, width):
    assert live_tui._context_rows(context, proposal, width) == reference_rows(context, proposal, width)


def test_preview_changes_and_scrolling_reuse_context():
    live_tui._wrapped_context.cache_clear()
    context = "history\n" * 100
    with patch.object(live_tui, "_append_wrapped_text", wraps=live_tui._append_wrapped_text) as wrap:
        first = live_tui._context_rows(context, "draft", 39)
        first[0].append(("", "corruption"))
        second = live_tui._context_rows(context, "new draft", 39)
        assert second == reference_rows(context, "new draft", 39)
        live_tui._context_view(context, "third draft", 40, 30, 8)
        live_tui._context_view(context, "third draft", 40, 60, 0)
        assert sum(call.args[2] == context for call in wrap.call_args_list) == 1
    assert live_tui._wrapped_context.cache_info().currsize == 1


def test_context_and_width_changes_rewrap():
    live_tui._wrapped_context.cache_clear()
    for context, width in [("abc", 4), ("abc", 5), ("changed", 5)]:
        assert live_tui._context_rows(context, "!", width) == reference_rows(context, "!", width)
    assert live_tui._wrapped_context.cache_info().misses == 3
    assert live_tui._wrapped_context.cache_info().currsize == 1


def test_context_sanitization_is_cached_separately_from_preview():
    live_tui._safe_context_text.cache_clear()
    with patch.object(live_tui, "_safe_rendered_text", wraps=live_tui._safe_rendered_text) as sanitize:
        expected = live_tui._safe_context_text("a\t\x00界")
        live_tui._safe_rendered_text("draft")
        assert live_tui._safe_context_text("a\t\x00界") == expected
        assert sanitize.call_count == 2

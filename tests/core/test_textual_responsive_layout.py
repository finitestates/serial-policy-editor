"""Responsive geometry checks for the Textual request screens."""

from __future__ import annotations

from dataclasses import replace

import pytest
from rich.text import Text
from textual.widgets import DataTable, Input, Static, TextArea

from trajectory_editor.terminal_contracts import BeamInput
from trajectory_editor.textual_tui import PolicyEditorApp
from trajectory_editor.ui_themes import resolve_live_theme
from tests.core.textual_support import (
    beam_state,
    choice_state,
    edge_state,
    install_request,
    prompt_state,
    run_pilot,
)

pytestmark = pytest.mark.current_workflow


def _static_text(widget: Static) -> str:
    content = widget.content
    return content.plain if isinstance(content, Text) else str(content)


@pytest.mark.parametrize(
    "size",
    [(40, 12), (44, 14), (60, 18), (80, 24), (120, 40), (160, 50), (240, 80)],
)
def test_ordinary_prompt_is_centered_capped_and_has_a_visible_input(size):
    async def scenario():
        width, _height = size
        app = PolicyEditorApp()
        async with app.run_test(size=size) as pilot:
            await install_request(app, pilot, prompt_state("What next?"))
            screen = app._active_screen
            group = screen.query_one("#prompt-group")
            prompt_input = screen.query_one("#prompt-input", Input)

            assert group.region.width <= min(width, 96)
            assert group.region.x >= max(0, (width - 96) // 2)
            assert prompt_input.region.height >= 3
            assert prompt_input.region.width >= 8
            assert prompt_input.region.x + prompt_input.region.width <= width
            assert app.focused is prompt_input
            assert screen.query_one("#hint", Static).region.bottom <= size[1]

    run_pilot(scenario)


def test_choice_context_uses_a_fixed_scroll_viewport_as_history_grows():
    async def scenario():
        state = choice_state()
        state = replace(
            state,
            choice=replace(
                state.choice,
                context_text_tail="\n".join(f"context line {index}" for index in range(80)),
            ),
        )
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, state)
            screen = app._active_screen
            context = screen.query_one("#context-scroll")
            table = screen.query_one("#choice-table", DataTable)

            assert context.region.height == 8
            assert context.max_scroll_y > 0
            assert table.region.height < 20
            assert table.region.bottom <= screen.query_one("#choice-input").region.y

    run_pilot(scenario)


@pytest.mark.parametrize("size", [(40, 12), (120, 40), (240, 80)])
def test_multiline_prompt_field_and_wrapping_hint_remain_visible(size):
    async def scenario():
        _width, height = size
        app = PolicyEditorApp()
        async with app.run_test(size=size) as pilot:
            await install_request(
                app,
                pilot,
                prompt_state(
                    "New episode > ",
                    multiline=True,
                    body="A short context line for the prompt.",
                ),
            )
            screen = app._active_screen
            group = screen.query_one("#prompt-group")
            field = screen.query_one("#multiline-input", TextArea)
            hint = screen.query_one("#hint", Static)

            assert field.region.height >= 3
            assert field.region.y + field.region.height <= hint.region.y
            assert hint.region.bottom == height
            assert app.focused is field
            assert group.region.width <= min(size[0], 96)

    run_pilot(scenario)


@pytest.mark.parametrize(
    ("height", "breakpoint_class"),
    [(17, "-short"), (18, "-regular"), (19, "-regular")],
    ids=("below", "at", "above"),
)
def test_prompt_vertical_breakpoint_has_one_height_threshold(height, breakpoint_class):
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(80, height)) as pilot:
            await install_request(app, pilot, prompt_state("Next step?"))
            screen = app._active_screen
            group = screen.query_one("#prompt-group")
            prompt_input = screen.query_one("#prompt-input", Input)
            hint = screen.query_one("#hint", Static)

            assert screen.has_class(breakpoint_class)
            assert group.region.right <= 80
            assert prompt_input.region.bottom <= hint.region.y
            assert hint.region.bottom == height
            assert app.focused is prompt_input

    run_pilot(scenario)


def test_edge_commands_fit_their_content_and_descriptions_wrap_on_narrow_terminals():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(40, 12)) as pilot:
            await install_request(app, pilot, edge_state())
            screen = app._active_screen
            table = screen.query_one("#edge-commands", DataTable)
            command_width = table.columns["command"].width

            assert command_width < 26
            assert table.columns["description"].width > 0
            assert any(row.height > 1 for row in table.rows.values())
            assert screen.query_one("#hint", Static).region.bottom == 12

    run_pilot(scenario)


def test_beam_panes_stack_at_narrow_width_and_reflow_without_stealing_focus():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, beam_state())
            screen = app._active_screen
            body = screen.query_one("#beam-body")
            table = screen.query_one("#beam-table", DataTable)
            detail = screen.query_one("#beam-detail-pane")
            beam_input = screen.query_one("#beam-input", TextArea)

            assert screen.has_class("-side-by-side")
            assert not screen.has_class("-stacked")
            assert table.region.width >= 48
            assert detail.region.width >= 36
            assert app.focused is beam_input

            await pilot.resize_terminal(80, 24)
            assert screen.has_class("-stacked")
            assert not screen.has_class("-side-by-side")
            assert table.region.width == body.region.width
            assert detail.region.width == body.region.width
            assert app.focused is beam_input

            await pilot.resize_terminal(160, 50)
            assert screen.has_class("-side-by-side")
            assert not screen.has_class("-stacked")
            assert detail.region.width >= 36
            assert app.focused is beam_input

    run_pilot(scenario)


@pytest.mark.parametrize("size", [(40, 12), (44, 14), (60, 18), (80, 24)])
def test_beam_compact_layout_keeps_the_table_input_and_help_inside_the_viewport(size):
    async def scenario():
        width, height = size
        app = PolicyEditorApp()
        async with app.run_test(size=size) as pilot:
            await install_request(app, pilot, beam_state())
            screen = app._active_screen
            body = screen.query_one("#beam-body")
            table = screen.query_one("#beam-table", DataTable)
            beam_input = screen.query_one("#beam-input", TextArea)
            hint = screen.query_one("#hint", Static)
            hint_text = _static_text(hint)

            assert screen.has_class("-stacked")
            assert table.region.width <= width
            assert table.virtual_size.width <= table.scrollable_content_region.width
            assert beam_input.region.y + beam_input.region.height <= height
            assert hint.region.bottom == height
            assert "PgUp/Dn details" in hint_text
            assert "Ctrl+K commands" in hint_text
            assert "F1 help" in hint_text
            assert app.focused is beam_input

    run_pilot(scenario)


@pytest.mark.parametrize(
    ("theme", "environment"),
    [
        ("amber-cyan", {"COLORTERM": "truecolor"}),
        ("monochrome", {"COLORTERM": "truecolor"}),
        ("high-contrast", {"COLORTERM": "truecolor"}),
        (None, {"NO_COLOR": ""}),
    ],
    ids=("amber-cyan", "monochrome", "high-contrast", "no-color"),
)
@pytest.mark.parametrize(
    "size",
    [(40, 12), (44, 14), (60, 18), (80, 24), (120, 40), (160, 50), (240, 80)],
)
def test_request_screens_keep_controls_and_submission_visible_by_theme_and_size(
    theme, environment, size
):
    async def scenario():
        width, height = size
        live_theme = theme or resolve_live_theme(None, environment=environment)
        app = PolicyEditorApp(theme=live_theme, environment=environment)
        async with app.run_test(size=size) as pilot:
            choice = await install_request(app, pilot, choice_state())
            screen = app._active_screen
            assert screen.query_one("#choice-table", DataTable).row_count == 2
            assert screen.query_one("#choice-input", TextArea).region.bottom <= height
            assert screen.query_one("#hint", Static).region.bottom == height
            assert app.focused is screen.query_one("#choice-input", TextArea)
            await pilot.press("enter")
            await pilot.pause()
            assert choice.response.result() == ""

            edge = await install_request(app, pilot, edge_state())
            screen = app._active_screen
            edge_input = screen.query_one("#edge-input", TextArea)
            edge_table = screen.query_one("#edge-commands", DataTable)
            assert edge_input.region.bottom <= height
            assert edge_table.scrollable_content_region.width > 0
            assert screen.query_one("#hint", Static).region.bottom == height
            assert app.focused is edge_input
            await pilot.press("ctrl+d")
            await pilot.pause()
            assert edge.response.result() is None

            beam = await install_request(app, pilot, beam_state())
            screen = app._active_screen
            beam_input = screen.query_one("#beam-input", TextArea)
            beam_table = screen.query_one("#beam-table", DataTable)
            assert beam_input.region.bottom <= height
            assert beam_table.scrollable_content_region.width > 0
            assert screen.query_one("#hint", Static).region.bottom == height
            assert app.focused is beam_input
            await pilot.press("right")
            await pilot.pause()
            assert beam.response.result() == BeamInput("advance 1", "b1")

            prompt = await install_request(app, pilot, prompt_state("Next? "))
            screen = app._active_screen
            prompt_input = screen.query_one("#prompt-input", Input)
            assert 0 < prompt_input.region.width <= min(width, 96)
            assert prompt_input.region.bottom <= height
            assert screen.query_one("#hint", Static).region.bottom == height
            assert app.focused is prompt_input
            await pilot.press("a", "enter")
            await pilot.pause()
            assert prompt.response.result() == "a"

    run_pilot(scenario)

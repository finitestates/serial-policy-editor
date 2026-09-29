"""Human-style interaction coverage for the Textual terminal migration."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
from textual.command import CommandList
from textual.events import MouseScrollDown
from textual.widgets import DataTable, Static, TextArea

from trajectory_editor.edge_commands import NewCommand, SaveCommand, parse_edge_command
from trajectory_editor.edge_help import edge_help
from trajectory_editor.terminal_contracts import BeamInput, BeamViewRow, PromptRequest
from trajectory_editor.textual_tui import BeamScreen, PolicyEditorApp
from tests.core.textual_support import (
    beam_state,
    choice_state,
    edge_state,
    install_request,
    prompt_state,
    run_pilot,
    submitted,
)

pytestmark = pytest.mark.current_workflow

_SNAPSHOT_DIR = Path(__file__).with_name("textual_migration_snapshots")
_SNAPSHOT_SIZES = ((40, 12), (80, 24), (160, 50))


def _save_or_compare_svg(app: PolicyEditorApp, name: str) -> None:
    actual = app.export_screenshot(simplify=True)
    path = _SNAPSHOT_DIR / name
    if os.environ.get("UPDATE_TEXTUAL_SNAPSHOTS") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
    assert path.exists(), f"missing Textual SVG snapshot: {path}"
    assert actual == path.read_text(encoding="utf-8")


@pytest.mark.parametrize("size", _SNAPSHOT_SIZES, ids=("short", "ordinary", "wide"))
def test_help_and_command_palette_fit_and_restore_editor_state(size):
    async def scenario():
        width, height = size
        app = PolicyEditorApp()
        async with app.run_test(size=size) as pilot:
            request = await install_request(app, pilot, choice_state())
            screen = app._active_screen
            editor = screen.query_one("#choice-input", TextArea)
            await pilot.press("2")
            await pilot.pause()
            contents = editor.text
            assert contents == "2"
            assert app.focused is editor

            await pilot.press("ctrl+k", *tuple("x text"))
            await pilot.pause(0.25)
            palette = app.screen
            commands = palette.query_one(CommandList)
            assert commands.option_count > 0
            assert 0 <= commands.region.x < width
            assert commands.region.right <= width
            assert 0 <= commands.region.y < height
            assert commands.region.bottom <= height
            _save_or_compare_svg(app, f"palette-{width}x{height}.svg")

            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is screen
            assert app.focused is editor
            assert editor.text == contents
            assert not request.response.done()

            await pilot.press("f1")
            await pilot.pause()
            help_screen = app.screen
            dialog = help_screen.query_one("#help-dialog")
            help_scroll = help_screen.query_one("#help-scroll")
            close_hint = help_screen.query_one("#hint", Static)
            assert abs(2 * dialog.region.x + dialog.region.width - width) <= 1
            assert abs(2 * dialog.region.y + dialog.region.height - height) <= 1
            assert dialog.region.right <= width
            assert dialog.region.bottom <= height
            assert help_scroll.max_scroll_y > 0
            assert close_hint.region.y >= dialog.region.y
            assert close_hint.region.bottom <= dialog.region.bottom
            _save_or_compare_svg(app, f"help-{width}x{height}.svg")

            await pilot.press("pagedown")
            await pilot.pause()
            assert help_scroll.scroll_y > 0
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is screen
            assert app.focused is editor
            assert editor.text == contents
            assert not request.response.done()

    run_pilot(scenario)


class DeterministicFakeEngine:
    """Small stateful oracle for the submitted values in the Pilot journey."""

    def __init__(self) -> None:
        self.text = ""
        self.actions: list[object] = []

    def choice(self, raw: str, state) -> None:
        self.actions.append(("choice", raw))
        if raw.startswith("x "):
            self.text += raw[2:]
        elif raw.isdigit():
            self.text += state.resolve_candidate(int(raw)).text

    def edge(self, raw: str) -> object:
        parsed = parse_edge_command(raw)
        self.actions.append(("edge", parsed))
        return parsed

    def beam(self, value: BeamInput) -> None:
        self.actions.append(("beam", value))

    def prompt(self, label: str, value: str) -> None:
        self.actions.append((label, value))


def test_one_app_carries_a_multi_request_pilot_journey():
    async def scenario():
        engine = DeterministicFakeEngine()
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            first_state = choice_state()
            first = await install_request(app, pilot, first_state)
            first_screen = app._active_screen
            first_editor = first_screen.query_one("#choice-input", TextArea)

            # Incomplete preview feedback remains editable; the editor can
            # expand before correction, then a palette template is a direct
            # continuation of the same command.
            await pilot.press(*tuple("x "), "ctrl+e", "enter")
            await pilot.pause()
            assert first_editor.has_class("expanded")
            assert first_screen._preview.state == "incomplete"
            feedback = str(first_screen.query_one("#choice-feedback", Static).render()).upper()
            assert "TYPE TEXT AFTER" in feedback
            assert not first.response.done()
            await pilot.press("backspace")
            assert first_editor.text == "x"

            await pilot.press("ctrl+k", *tuple("x text"))
            await pilot.pause(0.25)
            keyboard_commands = app.screen.query_one(CommandList)
            assert keyboard_commands.option_count > 0
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen is first_screen
            assert app.focused is first_editor
            assert first_editor.text == "x "
            await pilot.press(*tuple("alpha"))
            await pilot.pause()
            assert first_screen._preview.valid
            assert first_screen._preview.appended_text == "alpha"
            await pilot.press("enter")
            await pilot.pause()
            first_result = submitted(first)
            assert first_result == "x alpha"
            engine.choice(first_result, first_state)
            assert engine.text == "alpha"

            second_state = choice_state()
            second = await install_request(app, pilot, second_state, generation=2)
            second_screen = app._active_screen
            second_table = second_screen.query_one("#choice-table", DataTable)
            second_editor = second_screen.query_one("#choice-input", TextArea)

            await pilot.click(
                second_table,
                offset=(5, second_table.header_height + 1),
            )
            await pilot.pause()
            assert second_editor.text == "2", (
                f"cursor={second_table.cursor_coordinate}; "
                f"accepting={second_screen.accepting_input}; "
                f"cutoff={app._input_event_cutoff}; focus={app.focused!r}"
            )
            assert app.focused is second_editor
            await pilot.press("backspace", "2")
            await pilot.pause()
            assert second_editor.text == "2"
            assert second_screen._preview.candidate_rank == 2

            # Cancel one search, then choose the raw-rank template from the palette.
            await pilot.press("ctrl+k", *tuple("unused search"), "escape")
            await pilot.pause()
            assert app.screen is second_screen
            assert app.focused is second_editor
            assert second_editor.text == "2"

            await pilot.press("ctrl+k", *tuple("1..N"))
            await pilot.pause(0.25)
            mouse_commands = app.screen.query_one(CommandList)
            assert mouse_commands.option_count == 1
            await pilot.click(mouse_commands, offset=(3, 1))
            await pilot.pause()
            assert app.screen is second_screen
            assert app.focused is second_editor
            assert second_editor.text == "1"
            await pilot.press("backspace", "2")
            await pilot.pause()
            assert second_screen._preview.candidate_rank == 2

            await pilot.press("f1")
            await pilot.pause()
            help_scroll = app.screen.query_one("#help-scroll")
            assert help_scroll.max_scroll_y > 0
            await pilot.press("pagedown")
            await pilot.pause()
            assert help_scroll.scroll_y > 0
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is second_screen
            assert app.focused is second_editor
            assert second_editor.text == "2"
            assert second_screen.query_one("#choice-preview", Static)

            await pilot.press("enter")
            await pilot.pause()
            second_result = submitted(second)
            assert second_result == "2"
            assert second.submitted_target == (2, 3)
            engine.choice(second_result, second_state)
            assert engine.text == "alpha beta"

            edge = await install_request(app, pilot, edge_state(mode="session"), generation=3)
            edge_screen = app._active_screen
            edge_table = edge_screen.query_one("#edge-commands", DataTable)
            edge_row = next(
                i for i, item in enumerate(edge_help(edge_screen.state.mode))
                if item.command == "new TEXT"
            )
            edge_table.focus()
            await pilot.press("pagedown")
            await pilot.pause()
            assert edge_table.scroll_y > 0
            edge_table.scroll_to(y=0, animate=False, immediate=True)
            await pilot.pause()
            await pilot.click(edge_table, offset=(4, edge_table.header_height + edge_row))
            await pilot.pause()
            edge_editor = edge_screen.query_one("#edge-input", TextArea)
            assert app.focused is edge_editor
            assert edge_editor.text == "new "
            await pilot.press(*tuple("branch"), "backspace", "h")
            assert edge_editor.text == "new branch"
            await pilot.press("enter")
            await pilot.pause()
            new_root = engine.edge(submitted(edge))
            assert new_root == NewCommand("branch")

            save_edge = await install_request(
                app, pilot, edge_state(mode="session"), generation=4,
            )
            save_screen = app._active_screen
            save_table = save_screen.query_one("#edge-commands", DataTable)
            save_row = next(
                i for i, item in enumerate(edge_help(save_screen.state.mode))
                if item.command.startswith("save [")
            )
            save_table.move_cursor(row=save_row, column=0, animate=False)
            await pilot.pause()
            save_editor = save_screen.query_one("#edge-input", TextArea)
            assert save_editor.text == "save "
            await pilot.press("enter")
            await pilot.pause()
            save_action = engine.edge(submitted(save_edge))
            assert save_action == SaveCommand(None, None)

            path_request = await install_request(
                app, pilot, prompt_state("Workspace name? "), generation=5,
            )
            await pilot.press(*tuple("journey"), "enter")
            await pilot.pause()
            workspace_name = submitted(path_request)
            engine.prompt("workspace", workspace_name)
            assert workspace_name == "journey"

            beam_rows = list(beam_state(row_count=3).rows)
            beam_rows[1] = replace(
                beam_rows[1],
                recent_steps=tuple(f"generated step {i}" for i in range(80)),
            )
            beam_state_value = replace(
                beam_state(row_count=3), rows=tuple(beam_rows),
            )
            beam_request = await install_request(app, pilot, beam_state_value, generation=6)
            beam_screen = app._active_screen
            assert isinstance(beam_screen, BeamScreen)
            beam_table = beam_screen.query_one("#beam-table", DataTable)
            beam_editor = beam_screen.query_one("#beam-input", TextArea)
            detail_pane = beam_screen.query_one("#beam-detail-pane")

            details_before = app.stats["beam_detail_renders"]
            await pilot.click(beam_table, offset=(5, beam_table.header_height + 1))
            await pilot.pause()
            assert beam_screen.selected_label == "b2"
            assert app.focused is beam_editor
            assert "SELECTED: b2" in str(beam_screen.query_one("#beam-detail", Static).render())
            assert app.stats["beam_detail_renders"] == details_before + 1
            assert detail_pane.max_scroll_y > 0
            await pilot.press("pagedown")
            await pilot.pause()
            assert detail_pane.scroll_y > 0

            # Resizing preserves the editor focus, then a table-focused pass
            # crosses immediately below/at/above the CSS breakpoint.
            await pilot.resize_terminal(40, 12)
            await pilot.pause()
            assert app.focused is beam_editor
            assert beam_screen.has_class("-stacked")
            await pilot.resize_terminal(160, 50)
            await pilot.pause()
            assert app.focused is beam_editor
            assert beam_screen.has_class("-side-by-side")
            await pilot.resize_terminal(40, 12)
            await pilot.pause()
            assert app.focused is beam_editor
            beam_table.focus()
            for width, breakpoint_class in (
                (119, "-stacked"), (120, "-side-by-side"), (121, "-side-by-side"),
            ):
                await pilot.resize_terminal(width, 24)
                await pilot.pause()
                assert app.focused is beam_table
                assert beam_screen.has_class(breakpoint_class)
                assert beam_screen.selected_label == "b2"
            beam_editor.focus()
            await pilot.press(*tuple("advance 2"), "enter")
            await pilot.pause()
            beam_value = submitted(beam_request)
            assert beam_value == BeamInput("advance 2", "b2")
            engine.beam(beam_value)

            multiline = await install_request(
                app,
                pilot,
                prompt_state("New prompt > ", multiline=True, body="Type two lines."),
                generation=7,
            )
            await pilot.press(*tuple("new"), "enter", *tuple("prompt"), "escape", "enter")
            await pilot.pause()
            multiline_value = submitted(multiline)
            assert multiline_value == "new\nprompt"
            engine.prompt("multiline", multiline_value)

            # The same app proceeds immediately to another request; stale
            # input is covered separately at the App event boundary and PTY.
            last = await install_request(app, pilot, prompt_state("Finish? "), generation=8)
            await pilot.press(*tuple("done"), "enter")
            await pilot.pause()
            engine.prompt("finish", submitted(last))

            assert engine.text == "alpha beta"
            assert engine.actions == [
                ("choice", "x alpha"),
                ("choice", "2"),
                ("edge", NewCommand("branch")),
                ("edge", SaveCommand(None, None)),
                ("workspace", "journey"),
                ("beam", BeamInput("advance 2", "b2")),
                ("multiline", "new\nprompt"),
                ("finish", "done"),
            ]

    run_pilot(scenario)


def test_stale_mouse_input_is_filtered_at_request_turnover():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(80, 24)) as pilot:
            first = await install_request(app, pilot, prompt_state("First? "))
            await pilot.press("enter")
            await pilot.pause()
            assert submitted(first) == ""
            cutoff = app._input_event_cutoff
            assert cutoff is not None

            rows = list(beam_state(row_count=1).rows)
            rows[0] = replace(
                rows[0], recent_steps=tuple(f"visible step {index}" for index in range(40)),
            )
            beam = await install_request(
                app, pilot, replace(beam_state(row_count=1), rows=tuple(rows)),
                generation=2,
            )
            pane = app._active_screen.query_one("#beam-detail-pane")
            assert pane.max_scroll_y > 0

            def scroll_event():
                x = max(0, pane.size.width // 2)
                y = max(0, pane.size.height // 2)
                return MouseScrollDown(
                    widget=pane,
                    x=x,
                    y=y,
                    delta_x=0,
                    delta_y=1,
                    button=0,
                    shift=False,
                    meta=False,
                    ctrl=False,
                    screen_x=pane.region.x + x,
                    screen_y=pane.region.y + y,
                )

            stale = scroll_event()
            stale.time = cutoff - .01
            stale.set_sender(app)
            before_filtered = app.stats["stale_input_events"]
            app.post_message(stale)
            await pilot.pause()
            assert app.stats["stale_input_events"] == before_filtered + 1
            assert app._input_event_cutoff == cutoff
            assert not beam.response.done()

            current = scroll_event()
            current.time = cutoff + .01
            current.set_sender(app)
            app.post_message(current)
            await pilot.pause()
            assert app._input_event_cutoff is None
            assert not beam.response.done()

    run_pilot(scenario)

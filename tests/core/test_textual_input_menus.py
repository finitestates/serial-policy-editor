from __future__ import annotations

from dataclasses import replace

import pytest
from textual.command import CommandList
from textual.widgets import DataTable, Input, Static, TextArea

from trajectory_editor.edge_commands import NewCommand, SaveCommand, parse_edge_command
from trajectory_editor.edge_help import edge_help
from trajectory_editor.terminal_contracts import BeamInput
from trajectory_editor.textual_tui import PolicyEditorApp
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


@pytest.mark.parametrize(
    ("state_factory", "selector", "widget_type"),
    [
        (choice_state, "#choice-input", TextArea),
        (edge_state, "#edge-input", TextArea),
        (beam_state, "#beam-input", TextArea),
        (prompt_state, "#prompt-input", Input),
    ],
    ids=("choice", "edge", "beam", "prompt"),
)
def test_question_mark_is_text_and_f1_opens_help(state_factory, selector, widget_type):
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, state_factory())
            screen = app._active_screen

            await pilot.press("?")
            await pilot.pause()
            editor = screen.query_one(selector, widget_type)
            value = editor.value if isinstance(editor, Input) else editor.text
            assert value == "?"
            assert app.screen is screen
            assert not request.response.done()

            await pilot.press("f1")
            await pilot.pause()
            assert type(app.screen).__name__ == "HelpScreen"
            assert "commands" in str(app.screen.query_one("#help-body", Static).render()).lower()
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is screen
            assert not request.response.done()

    run_pilot(scenario)


def test_single_key_prompt_does_not_open_the_command_palette():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, prompt_state(single_key=True))
            screen = app._active_screen

            await pilot.press("f1")
            await pilot.pause()
            assert type(app.screen).__name__ == "HelpScreen"
            assert not request.response.done()
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is screen

            await pilot.press("ctrl+k")
            await pilot.pause()
            assert app.screen is screen
            assert screen.accepting_input
            assert not request.response.done()

            await pilot.press("?")
            await pilot.pause()
            assert submitted(request) == "?"

    run_pilot(scenario)


def test_empty_command_palette_query_shows_app_and_system_commands():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, choice_state())
            await pilot.press("ctrl+k")
            await pilot.pause(0.3)

            commands = app.screen.query_one(CommandList)
            prompts = [str(commands.get_option_at_index(i).prompt) for i in range(commands.option_count)]
            assert any(prompt.startswith("accept\n") for prompt in prompts)
            assert any(prompt.startswith("Quit\n") for prompt in prompts)

    run_pilot(scenario)


@pytest.mark.parametrize("state_factory", (choice_state, edge_state, beam_state), ids=("choice", "edge", "beam"))
def test_help_can_be_opened_from_the_command_palette(state_factory):
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, state_factory())
            await pilot.press("ctrl+k")
            await pilot.pause(0.2)
            await pilot.press(*tuple("help"))
            await pilot.pause(0.2)
            await pilot.press("enter")
            await pilot.pause()

            assert type(app.screen).__name__ == "HelpScreen"
            assert not request.response.done()

    run_pilot(scenario)


def test_clicking_a_choice_candidate_returns_focus_and_keeps_typing_in_the_editor():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, choice_state())
            screen = app._active_screen
            table = screen.query_one("#choice-table", DataTable)
            input_widget = screen.query_one("#choice-input", TextArea)

            await pilot.click(table, offset=(5, table.header_height + 1))
            await pilot.pause(0.2)
            assert input_widget.text == "2"
            assert app.focused is input_widget

            await pilot.press("backspace", "2")
            await pilot.pause()
            assert input_widget.text == "2"
            assert screen._preview.candidate_rank == 2
            assert screen._preview.state == "ready"
            await pilot.press("enter")
            await pilot.pause()
            assert submitted(request) == "2"
            assert request.submitted_target == (2, 3)

    run_pilot(scenario)


def test_edge_template_click_returns_focus_for_argument_editing_and_enter_submits():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, edge_state(mode="session"))
            screen = app._active_screen
            table = screen.query_one("#edge-commands", DataTable)
            input_widget = screen.query_one("#edge-input", TextArea)

            row_index = next(
                index for index, item in enumerate(edge_help(screen.state.mode))
                if item.command == "new TEXT"
            )
            await pilot.click(table, offset=(4, table.header_height + row_index))
            await pilot.pause()
            assert app.focused is input_widget
            assert input_widget.text == "new "
            await pilot.press(*tuple("branch"), "backspace", "h")
            await pilot.pause()
            command = input_widget.text
            assert command == "new branch"

            await pilot.press("enter")
            await pilot.pause()
            assert submitted(request) == command
            assert parse_edge_command(command) == NewCommand("branch")

    run_pilot(scenario)


def test_blank_save_template_submits_the_documented_follow_on_command():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(100, 32)) as pilot:
            request = await install_request(app, pilot, edge_state(mode="session"))
            screen = app._active_screen
            table = screen.query_one("#edge-commands", DataTable)
            row_index = next(
                index for index, item in enumerate(edge_help(screen.state.mode))
                if item.command.startswith("save [")
            )

            table.move_cursor(row=row_index, column=0, animate=False)
            await pilot.pause()
            editor = screen.query_one("#edge-input", TextArea)
            assert app.focused is editor
            assert editor.text == "save "
            await pilot.press("enter")
            await pilot.pause()

            assert submitted(request) == "save "
            assert parse_edge_command("save ") == SaveCommand(None, None)

    run_pilot(scenario)


def test_beam_detail_page_keys_scroll_without_changing_branch():
    async def scenario():
        state = beam_state(row_count=1)
        long_row = replace(
            state.rows[0],
            recent_steps=tuple(f"generated step {index}" for index in range(80)),
        )
        state = replace(state, rows=(long_row,))
        app = PolicyEditorApp()
        async with app.run_test(size=(100, 30)) as pilot:
            await install_request(app, pilot, state)
            screen = app._active_screen
            pane = screen.query_one("#beam-detail-pane")
            assert pane.max_scroll_y > 0
            selected = screen.selected_label

            await pilot.click(pane, offset=(2, 2))
            await pilot.press("pagedown")
            await pilot.pause()
            assert pane.scroll_y > 0
            assert screen.selected_label == selected

    run_pilot(scenario)


@pytest.mark.parametrize(
    ("at_edge", "expected"),
    ((False, "advance 1"), (True, "resume")),
)
def test_beam_right_keeps_its_step_and_resume_meaning(at_edge, expected):
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(100, 30)) as pilot:
            request = await install_request(app, pilot, beam_state(at_edge=at_edge))
            await pilot.press("right")
            await pilot.pause()
            assert submitted(request) == BeamInput(expected, "b1")

    run_pilot(scenario)

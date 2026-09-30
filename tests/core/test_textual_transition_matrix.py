"""Same-kind request and overlay transitions for the Textual terminal UI.

These Pilot checks validate mounted widget state and request values. They do
not assert terminal-driver paint behavior or physical display frames.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Input, Static, TextArea
from trajectory_editor.terminal_contracts import BoundaryReview, PromptRequest
from trajectory_editor.textual_tui import (
    BeamScreen,
    ChoiceScreen,
    EdgeScreen,
    OutputScreen,
    PolicyEditorApp,
    PromptScreen,
)

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


def _plain(value: object) -> str:
    return value.plain if isinstance(value, Text) else str(value)


def test_repeated_edge_requests_reuse_the_mounted_command_surface():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            first = await install_request(app, pilot, edge_state(), generation=1)
            screen = app._active_screen
            assert isinstance(screen, EdgeScreen)
            table = screen.query_one("#edge-commands", DataTable)
            editor = screen.query_one("#edge-input", TextArea)
            await pilot.press(*"s top_k=2", "enter")
            await pilot.pause()
            assert submitted(first) == "s top_k=2"

            second_state = edge_state(mode="session")
            second_state = replace(
                second_state, episode_id="episode-2", boundary=4,
                sampler_summary="temperature=0.3 · top_k=5",
            )
            second = await install_request(app, pilot, second_state, generation=2)
            assert app._active_screen is screen
            assert screen.is_mounted
            assert screen.lifecycle is second
            assert screen.query_one("#edge-commands", DataTable) is table
            assert screen.query_one("#edge-input", TextArea) is editor
            assert editor.text == ""
            assert app.focused is editor
            header = _plain(screen.query_one("#edge-header", Static).content)
            assert "LIVE SESSION" in header
            assert "episode-2" in header
            assert "boundary 4" in header
            assert "top_k=5" in header

            await pilot.press(*"new top_k=7", "enter")
            await pilot.pause()
            assert submitted(second) == "new top_k=7"

    run_pilot(scenario)


def test_compatible_prompt_requests_reuse_editor_and_refresh_prompt_content():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            first = await install_request(
                app, pilot, prompt_state("Name > ", body="First prompt body"),
                generation=1,
            )
            screen = app._active_screen
            assert isinstance(screen, PromptScreen)
            editor = screen.query_one("#prompt-input", Input)
            label = screen.query_one("#prompt-label", Static)
            body = screen.query_one("#prompt-body", VerticalScroll)
            body_text = body.query_one(Static)
            await pilot.press(*"Ada Lovelace", "enter")
            await pilot.pause()
            assert submitted(first) == "Ada Lovelace"

            second = await install_request(
                app, pilot, prompt_state("Number > ", body="Second prompt body"),
                generation=2,
            )
            assert app._active_screen is screen
            assert screen.is_mounted
            assert screen.lifecycle is second
            assert screen.query_one("#prompt-input", Input) is editor
            assert screen.query_one("#prompt-label", Static) is label
            assert screen.query_one("#prompt-body", VerticalScroll) is body
            assert label.content == "Number > "
            assert _plain(body_text.content) == "Second prompt body"
            assert editor.value == ""
            assert app.focused is editor

            await pilot.press(*"42", "enter")
            await pilot.pause()
            assert submitted(second) == "42"

    run_pilot(scenario)


def test_repeated_page_requests_reuse_the_page_return_target_and_refresh_content():
    async def scenario():
        app = PolicyEditorApp()
        first_body = "First page heading\n" + "\n".join(
            f"First page line {index}" for index in range(24)
        )
        second_body = "Second page heading\n" + "\n".join(
            f"Second page line {index}" for index in range(24)
        )
        async with app.run_test(size=(100, 30)) as pilot:
            first = await install_request(
                app, pilot, PromptRequest("First page", body=first_body, page=True),
                generation=1,
            )
            screen = app._active_screen
            assert isinstance(screen, PromptScreen)
            return_target = screen.query_one("#page-return", Input)
            page_body = screen.query_one("#page-scroll").query_one(Static)
            assert _plain(page_body.content) == first_body
            assert app.focused is return_target

            await pilot.press("q")
            await pilot.pause()
            assert submitted(first) == ""

            second = await install_request(
                app, pilot, PromptRequest("Second page", body=second_body, page=True),
                generation=2,
            )
            assert app._active_screen is screen
            assert screen.is_mounted
            assert screen.lifecycle is second
            assert screen.query_one("#page-return", Input) is return_target
            assert screen.query_one("#page-scroll").query_one(Static) is page_body
            assert _plain(page_body.content) == second_body
            assert return_target.value == ""
            assert app.focused is return_target

            await pilot.press("enter")
            await pilot.pause()
            assert submitted(second) == ""

    run_pilot(scenario)


def test_repeated_single_key_requests_reuse_the_key_target_and_refresh_prompt_content():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(100, 30)) as pilot:
            first = await install_request(
                app,
                pilot,
                PromptRequest("First key", body="First key instructions", single_key=True),
                generation=1,
            )
            screen = app._active_screen
            assert isinstance(screen, PromptScreen)
            key_target = screen.query_one("#single-key-hint", Input)
            label = screen.query_one("#prompt-label", Static)
            body = screen.query_one("#prompt-body").query_one(Static)
            await pilot.press("x")
            await pilot.pause()
            assert submitted(first) == "x"

            second = await install_request(
                app,
                pilot,
                PromptRequest("Second key", body="Second key instructions", single_key=True),
                generation=2,
            )
            assert app._active_screen is screen
            assert screen.is_mounted
            assert screen.lifecycle is second
            assert screen.query_one("#single-key-hint", Input) is key_target
            assert screen.query_one("#prompt-label", Static) is label
            assert screen.query_one("#prompt-body").query_one(Static) is body
            assert _plain(label.content) == "Second key"
            assert _plain(body.content) == "Second key instructions"
            assert key_target.value == ""
            assert app.focused is key_target

            await pilot.press("z")
            await pilot.pause()
            assert submitted(second) == "z"

    run_pilot(scenario)


def test_repeated_multiline_requests_reuse_the_editor_and_refresh_prompt_content():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(100, 30)) as pilot:
            first = await install_request(
                app,
                pilot,
                PromptRequest("First prompt > ", body="First multiline context", multiline=True),
                generation=1,
            )
            screen = app._active_screen
            assert isinstance(screen, PromptScreen)
            editor = screen.query_one("#multiline-input", TextArea)
            label = screen.query_one("#prompt-label", Static)
            instructions = screen.query_one("#prompt-instructions", Static)
            body = screen.query_one("#prompt-body").query_one(Static)
            await pilot.press(*tuple("first line"), "enter", *tuple("second line"))
            await pilot.press("escape", "enter")
            await pilot.pause()
            assert submitted(first) == "first line\nsecond line"

            second = await install_request(
                app,
                pilot,
                PromptRequest("Second prompt > ", body="Second multiline context", multiline=True),
                generation=2,
            )
            assert app._active_screen is screen
            assert screen.is_mounted
            assert screen.lifecycle is second
            assert screen.query_one("#multiline-input", TextArea) is editor
            assert screen.query_one("#prompt-label", Static) is label
            assert screen.query_one("#prompt-instructions", Static) is instructions
            assert screen.query_one("#prompt-body").query_one(Static) is body
            assert _plain(label.content) == "Second prompt > "
            assert _plain(body.content) == "Second multiline context"
            assert _plain(instructions.content) == (
                "Write the new prompt. Enter adds a line; Esc then Enter submits."
            )
            assert editor.text == ""
            assert app.focused is editor

            await pilot.press(*tuple("replacement"), "escape", "enter")
            await pilot.pause()
            assert submitted(second) == "replacement"

    run_pilot(scenario)


def test_repeated_isolated_chord_requests_reuse_the_editor_and_refresh_choices():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(100, 30)) as pilot:
            first = await install_request(
                app,
                pilot,
                PromptRequest("First chord > ", body="a (1) | b (2)", isolated=True),
                generation=1,
            )
            screen = app._active_screen
            assert isinstance(screen, PromptScreen)
            editor = screen.query_one("#prompt-input", Input)
            label = screen.query_one("#prompt-label", Static)
            body = screen.query_one("#prompt-body").query_one(Static)
            await pilot.press("a", "enter")
            await pilot.pause()
            assert submitted(first) == "a"

            second = await install_request(
                app,
                pilot,
                PromptRequest("Second chord > ", body="x (1) | y (2)", isolated=True),
                generation=2,
            )
            assert app._active_screen is screen
            assert screen.is_mounted
            assert screen.lifecycle is second
            assert screen.query_one("#prompt-input", Input) is editor
            assert screen.query_one("#prompt-label", Static) is label
            assert screen.query_one("#prompt-body").query_one(Static) is body
            assert _plain(label.content) == "Second chord > "
            assert _plain(body.content) == "x (1) | y (2)"
            assert editor.value == ""
            assert app.focused is editor

            await pilot.press("y", "enter")
            await pilot.pause()
            assert submitted(second) == "y"

    run_pilot(scenario)


@pytest.mark.parametrize(
    ("destination", "selector", "expected_focus", "visible_content"),
    [
        (
            PromptRequest("History", body="Complete page destination", page=True),
            "#page-return",
            "page",
            "Complete page destination",
        ),
        (
            PromptRequest(
                "Write > ", body="Complete multiline destination", multiline=True,
            ),
            "#multiline-input",
            "multiline",
            "Write the new prompt. Enter adds a line; Esc then Enter submits.",
        ),
    ],
    ids=("ordinary-to-page", "ordinary-to-multiline"),
)
def test_structurally_incompatible_prompt_requests_mount_the_complete_destination(
    destination, selector, expected_focus, visible_content,
):
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(100, 30)) as pilot:
            first = await install_request(
                app, pilot, PromptRequest("Old ordinary prompt > ", body="Old body"),
                generation=1,
            )
            previous_screen = app._active_screen
            await pilot.press(*tuple("old answer"), "enter")
            await pilot.pause()
            assert submitted(first) == "old answer"

            second = await install_request(app, pilot, destination, generation=2)
            destination_screen = app._active_screen
            assert isinstance(destination_screen, PromptScreen)
            assert destination_screen is not previous_screen
            assert destination_screen.is_mounted
            assert app.screen is destination_screen
            assert destination_screen.lifecycle is second
            target = destination_screen.query_one(selector)
            assert app.focused is target
            if expected_focus == "page":
                page_text = destination_screen.query_one("#page-scroll").query_one(Static)
                assert _plain(page_text.content) == visible_content
                assert _plain(destination_screen.query_one("#hint", Static).content).startswith(
                    "PgUp/PgDn scroll"
                )
            else:
                assert _plain(
                    destination_screen.query_one("#prompt-instructions", Static).content
                ) == visible_content
                assert _plain(
                    destination_screen.query_one("#prompt-body").query_one(Static).content
                ) == "Complete multiline destination"
                assert _plain(
                    destination_screen.query_one("#prompt-label", Static).content
                ) == "Write > "
            assert not second.response.done()

            if expected_focus == "page":
                await pilot.press("q")
                await pilot.pause()
                assert submitted(second) == ""
            else:
                await pilot.press(*tuple("new content"), "escape", "enter")
                await pilot.pause()
                assert submitted(second) == "new content"

    run_pilot(scenario)


def test_live_choice_boundary_update_keeps_the_mounted_table_and_editor():
    async def scenario():
        app = PolicyEditorApp()
        first_state = choice_state()
        candidates = first_state.candidates
        next_choice = replace(
            first_state.choice,
            aligned_step=1,
            sampling_boundary=1,
            context_text_tail="next boundary context",
            proposal_text=" beta",
            candidates=candidates,
        )
        second_state = replace(first_state, choice=next_choice, initial_command="2")

        async with app.run_test(size=(120, 40)) as pilot:
            first = await install_request(app, pilot, first_state, generation=1)
            screen = app._active_screen
            assert isinstance(screen, ChoiceScreen)
            table = screen.query_one("#choice-table", DataTable)
            editor = screen.query_one("#choice-input", TextArea)
            await pilot.press(*"x draft", "enter")
            await pilot.pause()
            assert submitted(first) == "x draft"

            second = await install_request(app, pilot, second_state, generation=2)
            assert app._active_screen is screen
            assert screen.is_mounted
            assert screen.lifecycle is second
            assert screen.query_one("#choice-table", DataTable) is table
            assert screen.query_one("#choice-input", TextArea) is editor
            assert table.row_count == 2
            assert "Step 1" in _plain(screen.query_one("#choice-heading", Static).content)
            assert "next boundary context" in _plain(screen.query_one("#context", Static).content)
            assert editor.text == "2"
            assert app.focused is editor

            await pilot.press("enter")
            await pilot.pause()
            assert submitted(second) == "2"

    run_pilot(scenario)


def test_historical_review_to_live_choice_is_a_full_kind_change():
    async def scenario():
        app = PolicyEditorApp()
        review_state = replace(
            choice_state(),
            review=BoundaryReview(2, 1, "historical context", {"kind": "token-boundary"}),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            review = await install_request(app, pilot, review_state, generation=1)
            review_screen = app._active_screen
            assert isinstance(review_screen, ChoiceScreen)
            assert review_screen.state.review is not None
            assert list(review_screen.query("#review-header"))
            await pilot.press("escape")
            await pilot.pause()
            assert submitted(review) == "\x1b"

            live = await install_request(app, pilot, choice_state(), generation=2)
            live_screen = app._active_screen
            assert isinstance(live_screen, ChoiceScreen)
            assert live_screen is not review_screen
            assert live_screen.is_mounted
            assert live_screen.lifecycle is live
            assert live_screen.state.review is None
            assert not list(live_screen.query("#review-header"))
            assert live_screen.query_one("#choice-table", DataTable).row_count == 2
            assert app.focused is live_screen.query_one("#choice-input", TextArea)

            await pilot.press("1", "enter")
            await pilot.pause()
            assert submitted(live) == "1"

    run_pilot(scenario)


def test_help_output_and_palette_dismissal_preserve_all_beam_panes_and_focus():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, beam_state(row_count=3))
            screen = app._active_screen
            assert isinstance(screen, BeamScreen)
            heading = screen.query_one("#beam-heading", Static)
            context = screen.query_one("#beam-context", Static)
            table = screen.query_one("#beam-table", DataTable)
            detail = screen.query_one("#beam-detail", Static)
            notice = screen.query_one("#beam-notice", Static)
            editor = screen.query_one("#beam-input", TextArea)
            hint = screen.query_one("#hint", Static)
            panes = (heading, context, table, detail, notice, editor, hint)

            def snapshot():
                row_keys = tuple(str(key.value) for key in table.rows)
                table_cells = tuple(
                    tuple(_plain(table.get_cell(row_key, column_key)) for column_key in table.columns)
                    for row_key in row_keys
                )
                return (
                    _plain(heading.content),
                    _plain(context.content),
                    table_cells,
                    _plain(detail.content),
                    _plain(notice.content),
                    editor.text,
                    _plain(hint.content),
                )

            initial = snapshot()
            assert app.focused is editor
            assert not request.response.done()

            await pilot.press("f1")
            await pilot.pause()
            assert type(app.screen).__name__ == "HelpScreen"
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is screen
            assert app._active_screen is screen
            assert tuple(
                screen.query_one(selector)
                for selector in (
                    "#beam-heading", "#beam-context", "#beam-table", "#beam-detail",
                    "#beam-notice", "#beam-input", "#hint",
                )
            ) == panes
            assert snapshot() == initial
            assert app.focused is editor

            app.write_output("captured diagnostic\n")
            await pilot.press("ctrl+l")
            await pilot.pause()
            assert isinstance(app.screen, OutputScreen)
            output_screen = app.screen
            output = output_screen.query_one("#output-body")
            assert "captured diagnostic" in "\n".join(map(str, output.lines))
            # Crossing the bounded history limit takes the refresh path while
            # the output overlay is mounted; the underlying Beam stays intact.
            app.write_output("x" * (app.OUTPUT_HISTORY_LIMIT + 1))
            await pilot.pause()
            assert app.screen is output_screen
            assert snapshot() == initial
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is screen
            assert snapshot() == initial
            assert app.focused is editor

            await pilot.press("ctrl+k")
            await pilot.pause()
            assert type(app.screen).__name__ == "CommandPalette"
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is screen
            assert app._active_screen is screen
            assert snapshot() == initial
            assert app.focused is editor
            assert not request.response.done()

            await pilot.press("enter")
            await pilot.pause()
            assert submitted(request) is not None

    run_pilot(scenario)


@pytest.mark.parametrize(
    ("state", "selector", "screen_type"),
    [
        (beam_state(), "#beam-input", BeamScreen),
        (edge_state(), "#edge-input", EdgeScreen),
        (choice_state(), "#choice-input", ChoiceScreen),
        (prompt_state(), "#prompt-input", PromptScreen),
    ],
    ids=("beam", "edge", "choice", "prompt"),
)
def test_help_and_output_close_return_focus_to_each_request_editor(state, selector, screen_type):
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, state)
            screen = app._active_screen
            assert isinstance(screen, screen_type)
            editor = screen.query_one(selector)
            assert app.focused is editor

            await pilot.press("f1")
            await pilot.pause()
            assert type(app.screen).__name__ == "HelpScreen"
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is screen
            assert app.focused is editor

            await pilot.press("ctrl+l")
            await pilot.pause()
            assert isinstance(app.screen, OutputScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is screen
            assert app.focused is editor
            assert not request.response.done()

            await pilot.press("ctrl+d")
            await pilot.pause()

    run_pilot(scenario)

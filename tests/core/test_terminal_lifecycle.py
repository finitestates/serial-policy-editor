"""Textual Pilot coverage for terminal request and screen lifecycles."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace

import pytest
from rich.text import Text
from textual import events
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Input, Static, TextArea
from trajectory_editor.core.candidates import Candidate
from trajectory_editor.edge_help import edge_help
from trajectory_editor.terminal_contracts import (
    SEAMLESS_REACTIVATE,
    BeamInput,
    BoundaryReview,
    ChoiceFeedback,
)
from trajectory_editor.textual_tui import (
    BeamScreen,
    ChoiceScreen,
    EdgeScreen,
    OutputScreen,
    PolicyEditorApp,
    PromptScreen,
    _RequestLifecycle,
)
from trajectory_editor.ui_themes import LIVE_THEME_NAMES

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


def _text(widget: Static) -> str:
    renderable = widget.content
    return renderable.plain if isinstance(renderable, Text) else str(renderable)


def _cell_text(table: DataTable, row_key: str, column_key: str) -> str:
    value = table.get_cell(row_key, column_key)
    return value.plain if isinstance(value, Text) else str(value)


def test_linux_driver_builds_in_the_ui_thread_without_registering_signals(monkeypatch):
    if os.name != "posix":
        pytest.skip("the threaded Linux driver is POSIX-specific")

    class NonTerminalStream:
        def fileno(self):
            return 0

        def isatty(self):
            return False

    monkeypatch.setattr(sys, "__stdin__", NonTerminalStream())
    monkeypatch.setattr(sys, "__stderr__", NonTerminalStream())
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("LINES", "25")
    monkeypatch.setattr(
        os, "get_terminal_size", lambda _fd: os.terminal_size((147, 52))
    )
    app = PolicyEditorApp()
    drivers = []
    errors = []

    def build_driver_in_ui_thread():
        async def build_driver():
            drivers.append(
                app._build_driver(headless=False, inline=False, mouse=False, size=None)
            )

        try:
            asyncio.run(build_driver())
        except BaseException as error:  # noqa: BLE001 - assert worker failures in the test thread.
            errors.append(error)

    worker = threading.Thread(target=build_driver_in_ui_thread, name="test-textual-ui")
    worker.start()
    worker.join(timeout=5)
    app.close_executors()

    assert not worker.is_alive()
    assert not errors, repr(errors)
    assert len(drivers) == 1
    assert type(drivers[0]) is app.driver_class
    assert drivers[0]._get_terminal_size() == (147, 52)


def test_pilot_drives_choice_edge_beam_and_prompt_requests():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            choice = await install_request(app, pilot, choice_state())
            assert isinstance(app._active_screen, ChoiceScreen)
            assert _text(app._active_screen.query_one("#choice-heading", Static)).startswith("Step 0")
            assert app._active_screen.query_one("#choice-table", DataTable).row_count == 2
            await pilot.press("1", "enter", "x")
            await pilot.pause()
            assert submitted(choice) == "1"

            edge = await install_request(app, pilot, edge_state(), generation=2)
            assert isinstance(app._active_screen, EdgeScreen)
            assert app._active_screen.query_one("#edge-input", TextArea).text == ""
            assert app._active_screen.query_one("#edge-input", TextArea).size.width > 0
            assert "episode-1" in _text(app._active_screen.query_one("#edge-header", Static))
            await pilot.press(*"s top_k=2", "enter")
            await pilot.pause()
            assert submitted(edge) == "s top_k=2"

            beam = await install_request(app, pilot, beam_state(), generation=3)
            assert isinstance(app._active_screen, BeamScreen)
            assert app._active_screen.query_one("#beam-input", TextArea).size.width > 0
            assert "shared context" in _text(app._active_screen.query_one("#beam-context", Static))
            await pilot.press("enter")
            await pilot.pause()
            assert submitted(beam) == BeamInput("select b1", "b1")

            prompt = await install_request(app, pilot, prompt_state(), generation=4)
            assert isinstance(app._active_screen, PromptScreen)
            await pilot.press("f", "o", "o", "enter")
            await pilot.pause()
            assert submitted(prompt) == "foo"

    run_pilot(scenario)


@pytest.mark.parametrize(
    ("command", "initial_command"),
    [("m20", None), ("beam", None), ("m20", "2"), ("beam", "2")],
)
def test_choice_command_typing_keeps_focus_after_first_character(command, initial_command):
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            state = choice_state(initial_command=initial_command)
            request = await install_request(app, pilot, state)
            screen = app._active_screen
            command_input = screen.query_one("#choice-input", TextArea)
            refresh_preview = screen._refresh_preview

            def refresh_preview_moves_focus():
                refresh_preview()
                screen.query_one("#choice-table", DataTable).focus()

            screen._refresh_preview = refresh_preview_moves_focus

            await pilot.press(command[0])
            await pilot.pause()
            assert command_input.text == command[0]
            assert app.focused is command_input

            await pilot.press(*command[1:])
            await pilot.pause()
            assert command_input.text == command
            assert app.focused is command_input
            assert not request.response.done()

            await pilot.press("ctrl+c")
            await pilot.pause()

    run_pilot(scenario)


@pytest.mark.parametrize("size", [(120, 40), (160, 50)])
def test_choice_candidate_table_uses_available_terminal_space(size):
    async def scenario():
        state = choice_state()
        candidates = tuple(
            Candidate(rank, rank + 1, f" token-{rank}", 1 / rank, False, 1 / rank)
            for rank in range(1, 13)
        )
        state = replace(
            state,
            candidates=candidates,
            choice=replace(state.choice, candidates=candidates),
        )
        app = PolicyEditorApp()
        async with app.run_test(size=size) as pilot:
            await install_request(app, pilot, state)
            screen = app._active_screen
            table = screen.query_one("#choice-table", DataTable)
            assert table.row_count == 12
            assert table.size.height <= size[1] * 0.3 + 1
            assert table.size.height >= 3
            assert table.virtual_size.height <= size[1] * 0.3 + 3
            assert table.virtual_size.width <= table.scrollable_content_region.width
            assert app._output_history_chars == 0

    run_pilot(scenario)


def test_choice_layout_reflows_between_regular_terminal_sizes():
    async def scenario():
        state = choice_state()
        candidates = tuple(
            Candidate(rank, rank + 1, f" token-{rank}", 1 / rank, False, 1 / rank)
            for rank in range(1, 13)
        )
        state = replace(
            state,
            candidates=candidates,
            choice=replace(state.choice, candidates=candidates),
        )
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, state)
            table = app._active_screen.query_one("#choice-table", DataTable)
            regular_height = table.size.height
            await pilot.resize_terminal(140, 50)
            assert regular_height <= 40 * 0.3 + 1
            assert table.size.height <= 50 * 0.3 + 1
            assert table.virtual_size.width <= table.scrollable_content_region.width
            await pilot.resize_terminal(120, 40)
            assert table.size.height == regular_height
            assert table.virtual_size.height <= 40 * 0.3 + 3

    run_pilot(scenario)


def test_output_modal_matches_the_bounded_history_while_open_and_after_reopen():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, prompt_state())
            limit = app.OUTPUT_HISTORY_LIMIT
            app.write_output("discarded" + "x" * (limit + 7))
            assert app._output_history_chars == limit
            assert "".join(app._output_chunks) == "x" * limit
            assert app.stats["output_history_high_water_chars"] == limit

            app.start_output()
            await pilot.pause()
            assert isinstance(app.screen, OutputScreen)
            log = app.screen.query_one("#output-body")
            assert "".join(segment.text for line in log.lines for segment in line) == "x" * limit

            app.write_output("ab")
            await pilot.pause()
            expected = "x" * (limit - 2) + "ab"
            assert "".join(app._output_chunks) == expected
            assert app._output_history_chars == limit
            assert app.stats["output_history_high_water_chars"] == limit
            assert "".join(segment.text for line in log.lines for segment in line) == expected

            log.scroll_to(y=0, animate=False, immediate=True)
            await pilot.pause()
            assert log.scroll_y < log.max_scroll_y
            for character in "cdefghijklmnop":
                app.write_output(character)
            await pilot.pause()
            expected = "x" * (limit - 16) + "abcdefghijklmnop"
            assert "".join(app._output_chunks) == expected
            assert app._output_history_chars == limit
            assert "".join(segment.text for line in log.lines for segment in line) == expected
            assert log.scroll_y < log.max_scroll_y

            await pilot.press("q")
            await pilot.pause()
            app.start_output()
            await pilot.pause()
            reopened_log = app.screen.query_one("#output-body")
            assert "".join(
                segment.text for line in reopened_log.lines for segment in line
            ) == expected
            assert len(reopened_log.lines) <= limit + 1

    run_pilot(scenario)


def test_edge_mouse_selection_inserts_command_template():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            lifecycle = await install_request(app, pilot, edge_state())
            screen = app._active_screen
            table = screen.query_one("#edge-commands", DataTable)
            row_index = next(
                index for index, item in enumerate(edge_help(screen.state.mode))
                if item.command == "new TEXT"
            )
            await pilot.click(table, offset=(4, table.header_height + row_index))
            await pilot.pause()
            command_input = screen.query_one("#edge-input", TextArea)
            assert command_input.text == "new "
            assert app.focused is command_input
            await pilot.press("enter")
            await pilot.pause()
            assert submitted(lifecycle) == "new "

    run_pilot(scenario)


@pytest.mark.parametrize("size", [(120, 40), (140, 45)])
def test_edge_beam_and_prompt_inputs_keep_a_useful_capped_width(size):
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=size) as pilot:
            edge = await install_request(app, pilot, edge_state())
            edge_screen = app._active_screen
            edge_input = edge_screen.query_one("#edge-input", TextArea)
            assert 8 <= edge_input.size.width <= 120
            edge_table = edge_screen.query_one("#edge-commands", DataTable)
            assert edge_table.virtual_size.width <= edge_table.scrollable_content_region.width
            await pilot.press("ctrl+d")
            await pilot.pause()
            assert submitted(edge) is None

            beam = await install_request(app, pilot, beam_state(), generation=2)
            beam_screen = app._active_screen
            assert 8 <= beam_screen.query_one("#beam-input", TextArea).size.width <= 120
            await pilot.press("ctrl+d")
            await pilot.pause()
            assert submitted(beam) == BeamInput("return", "b1")

            await install_request(app, pilot, prompt_state(), generation=3)
            prompt_input = app._active_screen.query_one("#prompt-input", Input)
            assert 8 <= prompt_input.size.width <= 96

    run_pilot(scenario)


def test_submitted_screen_stays_visible_until_the_next_request_replaces_it():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            choice = await install_request(app, pilot, choice_state())
            submitted_screen = app.screen
            await pilot.press("1", "enter")
            await pilot.pause()

            assert submitted(choice) == "1"
            assert app.screen is submitted_screen
            assert not app._active_screen

            await install_request(app, pilot, edge_state(), generation=2)
            assert isinstance(app._active_screen, EdgeScreen)
            assert app.screen is app._active_screen

    run_pilot(scenario)


def test_queued_keystrokes_after_submit_do_not_reach_the_next_request():
    async def scenario():
        app = PolicyEditorApp()
        first = _RequestLifecycle(1, choice_state(), Future(), None)
        second = _RequestLifecycle(2, prompt_state(), Future(), None)
        with ThreadPoolExecutor(max_workers=1) as engine_thread:
            def run_requests():
                app.call_from_thread(app.show_request, first)
                choice = first.response.result(timeout=5)
                app.call_from_thread(app.show_request, second)
                prompt = second.response.result(timeout=5)
                return choice, prompt

            async with app.run_test(size=(120, 40)) as pilot:
                operation = engine_thread.submit(run_requests)
                for _ in range(50):
                    await pilot.pause(.02)
                    if isinstance(app._active_screen, ChoiceScreen) and app._active_screen.accepting_input:
                        break
                assert isinstance(app._active_screen, ChoiceScreen)
                assert app._active_screen.accepting_input
                await pilot.press("1")
                await pilot.pause()
                assert app._active_screen.query_one("#choice-input", TextArea).text == "1"
                queued_enter = events.Key("enter", None)
                queued_character = events.Key("x", "x")
                queued_enter.set_sender(app)
                queued_character.set_sender(app)
                app.post_message(queued_enter)
                app.post_message(queued_character)
                await pilot.pause()
                for _ in range(50):
                    await pilot.pause(.02)
                    if isinstance(app._active_screen, PromptScreen) and app._active_screen.accepting_input:
                        break
                assert isinstance(app._active_screen, PromptScreen)
                screen = app._active_screen
                assert screen.query_one("#prompt-input", Input).value == ""
                assert not operation.done()
                await pilot.press("y", "enter")
                await pilot.pause()
                assert operation.result(timeout=2) == ("1", "y")

    run_pilot(scenario)


@pytest.mark.parametrize("theme", LIVE_THEME_NAMES)
def test_choice_submission_is_a_single_result_and_keeps_invalid_feedback_editable(theme):
    async def scenario():
        app = PolicyEditorApp(theme=theme, environment={"COLORTERM": "truecolor"})
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, choice_state())
            await pilot.press("9", "enter")
            await pilot.pause()
            screen = app._active_screen
            assert isinstance(screen, ChoiceScreen)
            assert not request.response.done()
            feedback = _text(screen.query_one("#choice-feedback", Static))
            assert "COMMAND" in feedback.upper()
            assert "rank must be 1..5" in feedback.lower()
            editor = screen.query_one("#choice-input", TextArea)
            assert editor.text == "9"
            text_before_backspace = editor.text
            cursor_before_backspace = editor.cursor_location
            await pilot.press("backspace")
            await pilot.pause()
            assert editor.text == "", (
                f"before={text_before_backspace!r}/{cursor_before_backspace}; "
                f"after={editor.text!r}/{editor.cursor_location}; "
                f"internal={screen._command_text!r}; focused={app.focused!r}"
            )
            await pilot.press("2", "enter")
            await pilot.pause()
            assert submitted(request) == "2"

    run_pilot(scenario)


def test_choice_displays_feedback_title_category_and_detail_lines():
    async def scenario():
        state = choice_state(
            feedback=ChoiceFeedback("error", "INVALID BIAS", ("unknown group", "rank 2 is unchanged")),
        )
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, state)
            feedback = app._active_screen.query_one("#choice-feedback", Static).content
            assert isinstance(feedback, Text)
            assert feedback.plain == "INVALID BIAS\n  unknown group\n  rank 2 is unchanged\n"
            assert feedback.spans

    run_pilot(scenario)


def test_choice_candidate_highlight_follows_the_live_command_preview():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, choice_state())
            screen = app._active_screen
            assert isinstance(screen, ChoiceScreen)
            table = screen.query_one("#choice-table", DataTable)
            assert table.get_cell("1", "marker").plain == "▶"
            assert table.cursor_coordinate.row == 0

            await pilot.press("2")
            await pilot.pause()

            assert screen._preview is not None
            assert screen._preview.candidate_rank == 2
            assert table.get_cell("1", "marker").plain == " "
            assert table.get_cell("2", "marker").plain == "▶"
            assert table.cursor_coordinate.row == 1

    run_pilot(scenario)


def test_python_and_native_output_is_captured_and_standard_fds_are_restored():
    script = r"""
import os
import sys
from trajectory_editor.tui import _ProcessOutputCapture, _SessionOutput

stdout = _SessionOutput()
stderr = _SessionOutput()
capture = _ProcessOutputCapture(stdout, stderr)
capture.start()
try:
    os.write(1, b"native stdout")
    os.write(2, b"native stderr")
    sys.stdout.write(" python stdout")
    sys.stderr.write(" python stderr")
finally:
    capture.stop()
assert stdout.getvalue() == "native stdout python stdout"
assert stderr.getvalue() == "native stderr python stderr"
print("standard descriptors restored")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == "standard descriptors restored\n"
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("feedback", "key", "expected"),
    [
        (ChoiceFeedback("info", "suggestions", completion_commands=("t alpha", "x beta")), "tab", "t alpha"),
        (ChoiceFeedback("info", "suggestions", completion_commands=("t alpha", "x beta")), "shift+tab", "x beta"),
        (ChoiceFeedback("search", "matches", initial_tab_command="2"), "tab", "2"),
    ],
)
def test_choice_tab_uses_feedback_completion_and_search_lens_commands(feedback, key, expected):
    async def scenario():
        state = choice_state(
            feedback=feedback,
            search_lens_active=feedback.category == "search",
            target_token_id=3 if feedback.category == "search" else None,
            display_candidates=(choice_state().candidates[1],) if feedback.category == "search" else None,
        )
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, state)
            screen = app._active_screen
            await pilot.press(key)
            await pilot.pause()
            assert screen.query_one("#choice-input", TextArea).text == expected
            await pilot.press("enter")
            await pilot.pause()
            assert submitted(request) == expected

    run_pilot(scenario)


def test_choice_search_lens_falls_back_to_target_token_rank():
    async def scenario():
        state = choice_state(
            search_lens_active=True,
            target_token_id=3,
            display_candidates=(choice_state().candidates[1],),
        )
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, state)
            screen = app._active_screen
            await pilot.press("shift+tab")
            await pilot.pause()
            assert screen.query_one("#choice-input", TextArea).text == "2"
            await pilot.press("enter")
            await pilot.pause()
            assert submitted(request) == "2"

    run_pilot(scenario)


@pytest.mark.parametrize("prefix", ("t ", "x "))
def test_choice_expanded_authored_editor_and_alt_enter_newline(prefix):
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, choice_state())
            screen = app._active_screen
            await pilot.press("ctrl+e")
            await pilot.pause()
            assert not screen.query_one("#choice-input", TextArea).has_class("expanded")
            await pilot.press(*prefix, "a", "ctrl+e", "alt+enter", "b")
            await pilot.pause()
            widget = screen.query_one("#choice-input", TextArea)
            assert widget.text == f"{prefix}a\nb"
            assert widget.has_class("expanded")
            assert widget.styles.height.value == 8
            assert widget.styles.min_height.value == 3
            await pilot.press("ctrl+e")
            await pilot.pause()
            assert not widget.has_class("expanded")
            await pilot.press("ctrl+e")
            await pilot.pause()
            assert widget.has_class("expanded")
            await pilot.press("enter")
            await pilot.pause()
            assert submitted(request) == f"{prefix}a\nb"

        review_state = replace(
            choice_state(),
            review=BoundaryReview(1, 0, "context", {"kind": "token-boundary"}),
        )
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, review_state)
            screen = app._active_screen
            await pilot.press("ctrl+e")
            await pilot.pause()
            widget = screen.query_one("#choice-input", TextArea)
            assert not widget.has_class("expanded")

    run_pilot(scenario)


def test_choice_ctrl_g_submits_numeric_rank_exploration():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, choice_state())
            await pilot.press("2", "ctrl+g")
            await pilot.pause()
            assert submitted(request) == "ms 2"

    run_pilot(scenario)


@pytest.mark.parametrize(
    ("state", "query", "expected"),
    [
        (choice_state(), "x TEXT", "x "),
        (edge_state(mode="session"), "new TEXT", "new "),
        (beam_state(), "advance N", "advance 1"),
    ],
)
def test_ctrl_k_fuzzy_palette_inserts_a_template_in_the_active_screen(state, query, expected):
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, state)
            await pilot.press("ctrl+k")
            await pilot.pause()
            assert type(app.screen).__name__ == "CommandPalette"
            await pilot.press(*query)
            await pilot.press("enter")
            await pilot.pause()
            assert app._active_screen.accepting_input
            selector = {
                "ChoiceScreen": "#choice-input",
                "EdgeScreen": "#edge-input",
                "BeamScreen": "#beam-input",
            }[type(app._active_screen).__name__]
            assert app._active_screen.query_one(selector, TextArea).text == expected

    run_pilot(scenario)


def test_choice_context_paging_temporarily_disables_tail_follow():
    async def scenario():
        state = choice_state()
        state = replace(
            state,
            choice=replace(state.choice, context_text_tail="\n".join(f"line {i}" for i in range(80))),
        )
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, state)
            screen = app._active_screen
            scroll = screen.query_one("#context-scroll")
            await pilot.press("pageup")
            await pilot.pause()
            assert screen._follow_tail is False
            assert scroll.scroll_y < scroll.max_scroll_y
            await pilot.press("pagedown")
            await pilot.pause()
            assert scroll.scroll_y == scroll.max_scroll_y
            assert screen._follow_tail is True

    run_pilot(scenario)


def test_choice_context_growth_keeps_all_prepared_history_and_respects_paging():
    async def scenario():
        initial_tail = "\n".join(f"earlier boundary {index}" for index in range(80))
        state = choice_state()
        state = replace(
            state,
            choice=replace(state.choice, context_text_tail=initial_tail),
        )
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, state)
            screen = app._active_screen
            scroll = screen.query_one("#context-scroll", VerticalScroll)
            rendered_chars = 0
            render_work = len(f"DECISION BOUNDARY\n{initial_tail}")
            full_tail = initial_tail

            for index in range(40):
                addition = f"\nlive boundary {index}"
                full_tail += addition
                screen.state = replace(
                    screen.state,
                    choice=replace(screen.state.choice, context_text_tail=full_tail),
                )
                screen._render_boundary_context()
                rendered_chars += len(addition)
                render_work += len(f"DECISION BOUNDARY\n{full_tail}")

            assert screen._rendered_context.plain == f"DECISION BOUNDARY\n{full_tail}"
            assert app.stats["context_append_characters"] == rendered_chars
            assert app.stats["context_high_water_characters"] == len(full_tail)
            assert app.stats["context_rendered_characters"] == render_work
            assert len(screen._rendered_context.plain) > 1_000

            await pilot.press("pageup")
            await pilot.pause()
            assert screen._follow_tail is False
            previous_scroll_y = scroll.scroll_y
            full_tail += "\nwhile paging"
            screen.state = replace(
                screen.state,
                choice=replace(screen.state.choice, context_text_tail=full_tail),
            )
            screen._render_boundary_context()
            render_work += len(f"DECISION BOUNDARY\n{full_tail}")
            await pilot.pause()
            assert app.stats["context_rendered_characters"] == render_work
            assert scroll.scroll_y == previous_scroll_y
            assert screen._rendered_context.plain.endswith(full_tail)

            await pilot.press("pagedown")
            await pilot.pause()
            assert screen._follow_tail is True
            full_tail += "\nfollow the newest boundary"
            screen.state = replace(
                screen.state,
                choice=replace(screen.state.choice, context_text_tail=full_tail),
            )
            screen._render_boundary_context()
            render_work += len(f"DECISION BOUNDARY\n{full_tail}")
            await pilot.pause()
            assert app.stats["context_rendered_characters"] == render_work
            assert scroll.scroll_y == scroll.max_scroll_y

    run_pilot(scenario)


def test_choice_preview_cache_tracks_command_and_request_generation(monkeypatch):
    from trajectory_editor import textual_tui

    original = textual_tui.action_preview
    calls = []

    def counted(*args, **kwargs):
        calls.append((args[1],))
        return original(*args, **kwargs)

    monkeypatch.setattr(textual_tui, "action_preview", counted)

    async def scenario():
        state = choice_state()
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, state, generation=11)
            screen = app._active_screen
            screen._refresh_preview()
            screen._refresh_preview()
            assert len(calls) == 1
            screen.query_one("#choice-input", TextArea).text = "t alpha"
            await pilot.pause()
            assert len(calls) == 2
        app.close_executors()

        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, state, generation=12)
            screen = app._active_screen
            screen._refresh_preview()
            assert len(calls) == 3
            assert screen.generation == 12

    run_pilot(scenario)


def test_choice_boundary_stays_committed_until_next_step():
    async def scenario():
        state = choice_state()
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            lifecycle = await install_request(app, pilot, state)
            screen = app._active_screen
            context = screen.query_one("#context", Static)
            original_context = context.content
            assert "context" in _text(context)
            assert "alpha" not in _text(context)

            await pilot.press("t", " ", "x")
            await pilot.pause()
            assert screen._command_text == "t x"
            assert context.content is original_context

            await pilot.press("enter")
            await pilot.pause()
            assert submitted(lifecycle) == "t x"

            next_choice = replace(
                state.choice,
                aligned_step=state.choice.aligned_step + 1,
                context_text_tail="context x",
            )
            await install_request(
                app,
                pilot,
                choice_state(choice=next_choice),
                generation=2,
            )
            next_screen = app._active_screen
            assert next_screen is screen
            assert "Step 1" in _text(next_screen.query_one("#choice-heading", Static))
            next_context = _text(next_screen.query_one("#context", Static))
            assert "context x" in next_context
            assert "alpha" not in next_context

    run_pilot(scenario)


@pytest.mark.parametrize(
    ("seamless", "reactivate", "key", "expected"),
    [
        (True, True, "enter", SEAMLESS_REACTIVATE),
        (False, False, "enter", "\x1b"),
        (True, True, "escape", "\x1b"),
    ],
)
def test_historical_review_enter_and_escape_return_the_review_contract(
    seamless, reactivate, key, expected
):
    async def scenario():
        state = replace(
            choice_state(),
            review=BoundaryReview(2, 1, "historical text", {"kind": "token-boundary"}),
            seamless=seamless,
            reactivate_on_review_enter=reactivate,
        )
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, state)
            assert "HISTORICAL CONTEXT" in _text(app._active_screen.query_one("#review-context", Static))
            await pilot.press(key)
            await pilot.pause()
            assert submitted(request) == expected

    run_pilot(scenario)


def test_edge_ctrl_d_returns_none():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, edge_state())
            await pilot.press("ctrl+d")
            await pilot.pause()
            assert submitted(request) is None

    run_pilot(scenario)


def test_edge_blank_enter_is_empty_and_ctrl_c_is_keyboard_interrupt():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, edge_state())
            await pilot.press("enter")
            await pilot.pause()
            assert submitted(request) == ""

        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, edge_state())
            await pilot.press("ctrl+c")
            await pilot.pause()
            with pytest.raises(KeyboardInterrupt):
                submitted(request)

    run_pilot(scenario)


@pytest.mark.parametrize(
    ("key", "initial", "selected"),
    [
        ("up", "b2", "b1"),
        ("down", "b1", "b2"),
    ],
)
def test_beam_arrows_move_the_selected_row_and_enter_commits_it(key, initial, selected):
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(
                app, pilot, replace(beam_state(), selected_label=initial),
            )
            screen = app._active_screen
            await pilot.press(key)
            await pilot.pause()
            assert f"SELECTED: {selected}" in _text(screen.query_one("#beam-detail", Static))
            await pilot.press("enter")
            await pilot.pause()
            assert submitted(request) == BeamInput(f"select {selected}", selected)

    run_pilot(scenario)


def test_beam_empty_enter_selects_the_current_branch():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, beam_state())
            await pilot.press("enter")
            await pilot.pause()
            assert submitted(request) == BeamInput("select b1", "b1")

    run_pilot(scenario)


def test_beam_empty_enter_resumes_at_the_edge():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, beam_state(at_edge=True))
            await pilot.press("enter")
            await pilot.pause()
            assert submitted(request) == BeamInput("resume", "b1")

    run_pilot(scenario)


def test_beam_renders_survivors_and_details_on_a_regular_terminal():
    async def scenario():
        state = beam_state(row_count=40)
        long_continuation = (
            "The wind howls outside, and the shutters rattle against the old stone "
            "walls while rain streams down."
        )
        rows = tuple(
            replace(
                row,
                label="branch-40" if index == 39 else row.label,
                continuation=long_continuation,
                recent_steps=(
                    tuple(f"detail step {step}" for step in range(80))
                    if index == 0 else row.recent_steps
                ),
            )
            for index, row in enumerate(state.rows)
        )
        state = replace(
            state,
            title="BEAM · width 12 · depth 1",
            rows=rows,
            selected_label=None,
        )
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, state)
            screen = app._active_screen
            body = screen.query_one("#beam-body")
            table = screen.query_one("#beam-table", DataTable)
            detail_pane = screen.query_one("#beam-detail-pane", VerticalScroll)
            details = _text(screen.query_one("#beam-detail", Static))

            assert body.region.width == 120
            assert body.region.height > 40 * 0.3
            assert table.region.width >= 75
            assert detail_pane.region.width >= 35
            assert 3 <= detail_pane.region.height <= body.region.height
            assert screen.query_one("#hint", Static).region.bottom == 40
            assert table.row_count == 40
            assert table.max_scroll_y > 0
            assert detail_pane.max_scroll_y > 0
            assert table.virtual_size.width <= table.scrollable_content_region.width
            assert _cell_text(table, "branch-40", "label") == "branch-40"
            assert _cell_text(table, "branch-40", "continuation") == long_continuation
            assert table.get_row_height("branch-40") >= 2
            assert screen.selected_label == "b1"
            assert long_continuation in details
            assert "family A" in details
            assert "Model rank: 2 · Step log-p: −0.250" in details
            assert "Model log-p: −0.800" in details

    run_pilot(scenario)


@pytest.mark.parametrize(
    ("key", "state", "expected", "needs_enter"),
    [
        ("backspace", beam_state(), BeamInput("kill b1", "b1"), False),
        ("p", beam_state(), BeamInput("protect", "b1"), False),
        ("p", beam_state(stochastic=True), BeamInput("p", "b1"), True),
        ("p", beam_state(at_edge=True), BeamInput("p", "b1"), True),
        ("f", beam_state(at_edge=True), BeamInput("f", "b1"), True),
        ("f", beam_state(), BeamInput("families", "b1"), False),
        ("right", beam_state(), BeamInput("advance 1", "b1"), False),
        ("left", beam_state(), BeamInput("rewind", "b1"), False),
        ("right", beam_state(at_edge=True), BeamInput("resume", "b1"), False),
        ("left", beam_state(at_edge=True), BeamInput("resume", "b1"), False),
        ("escape", beam_state(), BeamInput("return", "b1"), False),
        ("ctrl+d", beam_state(), BeamInput("return", "b1"), False),
    ],
)
def test_beam_shortcuts_submit_contract_commands(key, state, expected, needs_enter):
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, state)
            screen = app._active_screen
            await pilot.press(key)
            await pilot.pause()
            if needs_enter:
                assert not request.response.done()
                assert screen.query_one("#beam-input", TextArea).text == expected.command
                await pilot.press("enter")
                await pilot.pause()
                assert submitted(request) == expected
            else:
                assert submitted(request) == expected

    run_pilot(scenario)


def test_beam_stochastic_score_format_preserves_unicode_minus_and_notice():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, beam_state(stochastic=True))
            table = app._active_screen.query_one("#beam-table", DataTable)
            formatted_score = "G −0.45 · log-p -0.800"
            assert _cell_text(table, "b1", "score") == formatted_score
            assert table.columns["score"].content_width >= len(formatted_score)
            assert table.virtual_size.width <= table.scrollable_content_region.width
            assert "beam notice" in _text(app._active_screen.query_one("#beam-notice", Static))

    run_pilot(scenario)


def test_beam_shortcut_letters_are_command_text_when_the_buffer_is_not_empty():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, beam_state())
            await pilot.press("x", "p", "f", "enter")
            await pilot.pause()
            assert submitted(request) == BeamInput("xpf", "b1")

    run_pilot(scenario)


@pytest.mark.parametrize(
    ("prompt_req", "keys", "expected"),
    [
        (prompt_state(), ("a", "enter"), "a"),
        (prompt_state(), ("ctrl+d",), None),
        (prompt_state(), ("escape",), None),
        (prompt_state(single_key=True), ("z",), "z"),
        (prompt_state(single_key=True), ("backspace",), "\x7f"),
        (prompt_state(single_key=True), ("escape",), "\x1b"),
        (prompt_state(single_key=True), ("ctrl+d",), None),
        (prompt_state(multiline=True), ("ctrl+d",), None),
        (prompt_state(page=True, body="line one\nline two"), ("enter",), ""),
        (prompt_state(page=True, body="line one\nline two"), ("escape",), ""),
        (prompt_state(page=True, body="line one\nline two"), ("q",), ""),
    ],
)
def test_prompt_single_line_single_key_and_page_results(prompt_req, keys, expected):
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            lifecycle = await install_request(app, pilot, prompt_req)
            await pilot.press(*keys)
            await pilot.pause()
            assert submitted(lifecycle) == expected

    run_pilot(scenario)


@pytest.mark.parametrize(
    ("prompt_req", "selector", "multiline", "submit_keys"),
    [
        (prompt_state(), "#prompt-input", False, ("enter",)),
        (prompt_state(multiline=True), "#multiline-input", True, ("escape", "enter")),
    ],
)
def test_prompt_backspace_edits_typed_text(prompt_req, selector, multiline, submit_keys):
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            lifecycle = await install_request(app, pilot, prompt_req)
            await pilot.press("a", "b", "c", "backspace")
            await pilot.pause()
            if multiline:
                assert app._active_screen.query_one(selector, TextArea).text == "ab"
            else:
                assert app._active_screen.query_one(selector, Input).value == "ab"
            await pilot.press(*submit_keys)
            await pilot.pause()
            assert submitted(lifecycle) == "ab"

    run_pilot(scenario)


def test_page_prompt_scrolls_by_ten_and_multiline_requires_escape_then_enter():
    async def scenario():
        app = PolicyEditorApp()
        body = "\n".join(f"page line {index}" for index in range(100))
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, prompt_state(page=True, body=body))
            screen = app._active_screen
            page = screen.query_one("#page-scroll")
            await pilot.press("pagedown")
            await pilot.pause()
            assert page.scroll_y == 10
            await pilot.press("pageup")
            await pilot.pause()
            assert page.scroll_y == 0

        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(
                app,
                pilot,
                prompt_state("New prompt > ", multiline=True, isolated=True),
            )
            screen = app._active_screen
            instructions = _text(screen.query_one("#prompt-instructions", Static))
            assert "Write the new prompt" in instructions
            assert "Esc then Enter submits" in instructions
            await pilot.press("enter")
            await pilot.pause()
            assert not request.response.done()
            assert screen.query_one("#multiline-input", TextArea).text == "\n"
            await pilot.press("escape")
            await pilot.pause()
            status = screen.query_one("#prompt-status", Static)
            assert "press Enter to submit" in _text(status)
            assert status.has_class("feedback-info")
            assert not status.has_class("feedback-error")
            await pilot.press("enter")
            await pilot.pause()
            assert "Write at least one character" in _text(status)
            assert status.has_class("feedback-error")
            screen.query_one("#multiline-input", TextArea).text = ""
            await pilot.press("a", "escape", "enter")
            await pilot.pause()
            assert submitted(request) == "a"

        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, prompt_state(multiline=True))
            await pilot.press("a", "enter", "b", "escape", "enter")
            await pilot.pause()
            assert submitted(request) == "a\nb"

    run_pilot(scenario)


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (choice_state(), ""),
        (edge_state(), ""),
        (beam_state(), BeamInput("select b1", "b1")),
        (prompt_state(), ""),
        (
            replace(
                choice_state(),
                review=BoundaryReview(2, 1, "historical", {"kind": "token-boundary"}),
            ),
            "\x1b",
        ),
    ],
)
def test_help_modal_is_available_over_each_request_screen(state, expected):
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            request = await install_request(app, pilot, state)
            await pilot.press("f1")
            await pilot.pause()
            assert "commands" in _text(app.screen.query_one("#help-body", Static)).lower()
            assert not request.response.done()
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert submitted(request) == expected

    run_pilot(scenario)


def test_help_command_list_scrolls_inside_its_modal():
    async def scenario():
        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await install_request(app, pilot, choice_state())
            await pilot.press("f1")
            await pilot.pause()
            scroll = app.screen.query_one("#help-scroll", VerticalScroll)
            assert scroll.max_scroll_y > 0
            await pilot.press("pagedown")
            await pilot.pause()
            assert scroll.scroll_y == 10
            await pilot.press("pageup")
            await pilot.pause()
            assert scroll.scroll_y == 0

    run_pilot(scenario)

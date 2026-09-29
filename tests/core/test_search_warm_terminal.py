"""Search rank warming runs off the Textual thread and returns by generation."""

from __future__ import annotations

import threading

import pytest
from rich.text import Text
from textual.widgets import Static
from trajectory_editor.terminal_contracts import ChoiceFeedback
from trajectory_editor.textual_tui import PolicyEditorApp, TextualTerminalSession

from tests.core.textual_support import (
    choice_state,
    install_request,
    run_pilot,
    submitted,
)

pytestmark = pytest.mark.current_workflow


def _plain(widget: Static) -> str:
    value = widget.content
    return value.plain if isinstance(value, Text) else str(value)


def _search_state(warm, cancel, *, initial_tab_command="2"):
    base = choice_state()
    return choice_state(
        feedback=ChoiceFeedback(
            "search", "SEARCH RESULTS", initial_tab_command=initial_tab_command,
        ),
        search_lens_active=True,
        target_token_id=3,
        display_candidates=(base.candidates[1],),
        warm_search_token=warm,
        cancel_search_warm=cancel,
        search_warm_target=(2, 3),
        search_warm_commands=("/needle",),
    )


def test_search_warm_runs_on_worker_shows_pending_rank_and_delivers_to_ui_thread():
    async def scenario():
        ui_thread = threading.get_ident()
        entered = threading.Event()
        release = threading.Event()
        calls = []
        completions = []

        def warm(raw_rank, token_id, generation, cancelled):
            calls.append((raw_rank, token_id, generation, threading.get_ident()))
            entered.set()
            release.wait(2)
            return not cancelled()

        app = PolicyEditorApp()
        original = app._warm_completed

        def on_ui_completion(*args):
            completions.append(threading.get_ident())
            return original(*args)

        app._warm_completed = on_ui_completion
        async with app.run_test(size=(120, 40)) as pilot:
            state = _search_state(warm, lambda: None)
            lifecycle = await install_request(app, pilot, state, generation=31)
            screen = app._active_screen
            assert entered.wait(1)
            pending_line = screen.query_one("#hint", Static)
            assert "Resolving raw rank 2…" in _plain(pending_line)
            assert pending_line.size.height == 1
            before_resolution = tuple(
                (selector, screen.query_one(selector).region.y, screen.query_one(selector).region.height)
                for selector in ("#context-scroll", "#choice-table", "#choice-input", "#hint")
            )
            release.set()
            for _ in range(20):
                await pilot.pause(.02)
                if screen._warm_pending_target is None:
                    break
            assert screen._warm_pending_target is None
            assert pending_line.size.height == 1
            assert "Resolving raw rank" not in _plain(pending_line)
            after_resolution = tuple(
                (selector, screen.query_one(selector).region.y, screen.query_one(selector).region.height)
                for selector in ("#context-scroll", "#choice-table", "#choice-input", "#hint")
            )
            assert after_resolution == before_resolution
            assert calls == [(2, 3, 31, calls[0][3])]
            assert calls[0][3] != ui_thread
            assert completions == [ui_thread]
            assert app.stats["warm_dispatches"] == 1
            await pilot.press("tab", "enter")
            await pilot.pause()
            assert submitted(lifecycle) == "2"
            assert lifecycle.submitted_target == (2, 3)

            terminal = TextualTerminalSession()
            terminal.application = app
            terminal._finish_warm(lifecycle)
            assert app.stats["promotions"] == 1

    run_pilot(scenario)


def test_selecting_a_different_rank_cancels_the_search_warm():
    async def scenario():
        ui_thread = threading.get_ident()
        cancelled_threads = []
        entered = threading.Event()

        def warm(raw_rank, token_id, generation, cancelled):
            entered.set()
            return True

        def cancel():
            cancelled_threads.append(threading.get_ident())

        app = PolicyEditorApp()
        async with app.run_test(size=(120, 40)) as pilot:
            state = _search_state(warm, cancel)
            lifecycle = await install_request(app, pilot, state, generation=32)
            assert entered.wait(1)
            for _ in range(10):
                await pilot.pause(.02)
                if lifecycle.warm_future.done():
                    break
            await pilot.press("1", "enter")
            await pilot.pause()
            assert submitted(lifecycle) == "1"
            assert lifecycle.submitted_target == (1, 2)
            assert lifecycle.warm_cancelled.is_set()
            terminal = TextualTerminalSession()
            terminal.application = app
            terminal._finish_warm(lifecycle)
            assert cancelled_threads and cancelled_threads[0] != ui_thread
            assert app.stats["promotions"] == 0

    run_pilot(scenario)

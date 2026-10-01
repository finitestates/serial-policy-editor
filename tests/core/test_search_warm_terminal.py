"""Search rank warming runs off the UI thread and returns by generation."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from trajectory_editor.terminal_contracts import ChoiceFeedback

from tests.core.term_support import LiveLoop, choice_state

pytestmark = pytest.mark.current_workflow


def _search_state(warm, cancel, *, initial_tab_command="2"):
    base = choice_state()
    return choice_state(
        feedback=ChoiceFeedback("search", "SEARCH RESULTS", initial_tab_command=initial_tab_command),
        search_lens_active=True,
        target_token_id=3,
        display_candidates=(base.candidates[1],),
        warm_search_token=warm,
        cancel_search_warm=cancel,
        search_warm_target=(2, 3),
        search_warm_commands=("/needle",),
    )


def _engine(loop: LiveLoop, state):
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-engine") as engine:
        entered = threading.Event()

        def run():
            with loop:
                entered.set()
                return loop.session.read_choice(state), threading.get_ident()

        future = engine.submit(run)
        assert entered.wait(5)
        yield future


def test_search_warm_runs_on_worker_without_screen_chatter_and_delivers_to_ui_thread():
    entered = threading.Event()
    release = threading.Event()
    calls = []
    completions = []

    def warm(raw_rank, token_id, generation, cancelled):
        calls.append((raw_rank, token_id, generation, threading.get_ident()))
        entered.set()
        release.wait(2)
        return not cancelled()

    loop = LiveLoop()
    for future in _engine(loop, _search_state(warm, lambda: None)):
        app = loop.app
        original = app._warm_completed
        app._warm_completed = lambda *args, original=original: (
            completions.append(threading.current_thread().name), original(*args),
        )
        assert entered.wait(2)
        pending = loop.wait_for(lambda text: "Command >" in text)
        assert "Resolving raw rank" not in pending
        release.set()
        settled = loop.wait_for(lambda _text: bool(completions))
        # Background cache work does not repaint the user's view with progress text.
        assert pending == settled
        assert calls[0][:2] == (2, 3)
        assert completions == ["spe-terminal-ui"]
        assert calls[0][3] != threading.get_ident()
        loop.keys("\t\r")
        result, _engine_thread = future.result(timeout=5)
        assert result == "2"
        assert app.stats["warm_dispatches"] == 1
        assert app.stats["promotions"] == 1


def test_search_warm_failure_is_available_in_output_without_painting_the_view():
    entered = threading.Event()
    release = threading.Event()

    def warm(_raw_rank, _token_id, _generation, _cancelled):
        entered.set()
        release.wait(2)
        raise RuntimeError("backend cache diagnostic")

    loop = LiveLoop()
    for future in _engine(loop, _search_state(warm, lambda: None)):
        app = loop.app
        assert entered.wait(2)
        screen = loop.frame()
        assert "backend cache diagnostic" not in screen
        assert "Resolving raw rank" not in screen
        release.set()
        loop.wait_for(
            lambda _text: "backend cache diagnostic" in loop.on_ui(app.output_history),
        )
        assert "backend cache diagnostic" not in loop.frame()
        loop.keys("\x0c")
        output = loop.wait_for(lambda text: "[diagnostic]" in text)
        assert "search warm-up failed" in output
        assert "RuntimeError: backend cache diagnostic" in output
        loop.keys("\x0c")
        loop.wait_for(lambda text: "Captured output" not in text)
        loop.keys("2\r")
        result, _engine_thread = future.result(timeout=5)
        assert result == "2"


def test_selecting_a_different_rank_cancels_the_search_warm():
    cancelled_threads = []
    entered = threading.Event()

    def warm(raw_rank, token_id, generation, cancelled):
        entered.set()
        return True

    loop = LiveLoop()
    for future in _engine(loop, _search_state(warm, lambda: cancelled_threads.append(threading.current_thread().name))):
        assert entered.wait(2)
        loop.wait_for(lambda text: "Command >" in text)
        assert "Resolving raw rank" not in loop.frame()
        loop.keys("1\r")
        result, _engine_thread = future.result(timeout=5)
        assert result == "1"
        assert cancelled_threads and cancelled_threads[0].startswith("spe-search-warm")
        assert loop.app.stats["promotions"] == 0

"""Ordinary rank browsing leaves speculative engine state untouched."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

import pytest
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_ui import InteractivePolicy
from trajectory_editor.textual_tui import PolicyEditorApp, _RequestLifecycle

from tests.core.textual_support import run_pilot
from tests.fakes import SpeculativeFakeBackend

pytestmark = pytest.mark.current_workflow


class _PilotTerminal:
    """Blocking request adapter used only to exercise an engine thread in Pilot."""

    def __init__(self, app: PolicyEditorApp) -> None:
        self.app = app
        self.states = []
        self._counter = 0
        self._lock = Lock()

    def read_choice(self, state):
        with self._lock:
            self._counter += 1
            generation = self._counter
        lifecycle = _RequestLifecycle(generation, state, Future(), None)
        self.states.append(state)
        self.app.call_from_thread(self.app.show_request, lifecycle)
        return lifecycle.response.result(timeout=5)


def test_rank_navigation_does_not_speculate_before_engine_commit():
    async def scenario():
        backend = SpeculativeFakeBackend()
        engine = EpisodeEngine(
            backend, initial_token_ids=[7], sampling=SamplerConfig(temperature=0.0),
        )
        app = PolicyEditorApp()
        terminal = _PilotTerminal(app)
        policy = InteractivePolicy(io=terminal, menu_size=3)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-engine-owner")
        try:
            async with app.run_test(size=(120, 40)) as pilot:
                def choose_and_commit():
                    action = policy.choose(engine, engine.observe())
                    before_commit = tuple(backend.tokens)
                    evals_before_commit = tuple(backend.eval_calls)
                    outcome = engine.apply(action)
                    return action, before_commit, evals_before_commit, outcome

                operation = executor.submit(choose_and_commit)
                for _ in range(50):
                    await pilot.pause(.02)
                    if (
                        app._active_screen is not None
                        and app._active_screen.accepting_input
                    ):
                        break
                screen = app._active_screen
                assert screen is not None
                assert screen.accepting_input
                assert terminal.states[0].warm_search_token is None
                assert backend.eval_calls == []
                # The proposal rank is prefilled; two tabs advance from rank 1 to rank 3.
                for _ in range(2):
                    await pilot.press("tab")
                    await pilot.pause()
                assert screen._command_text == "3"
                assert backend.eval_calls == []
                assert backend.tokens == [7]
                await pilot.press("enter")
                await pilot.pause()
                action, before_commit, evals_before_commit, outcome = operation.result(timeout=2)
                assert action.kind == "select-raw-rank"
                assert action.rank == 3
                assert before_commit == (7,)
                assert evals_before_commit == ()
                assert outcome.visible_token_ids == (3,)
                assert backend.eval_calls == [(3,)]
                # The real commit happened on the same engine owner thread.
                assert backend.tokens == [7, 3]
                assert app.stats["warm_dispatches"] == 0
                assert app.stats["promotions"] == 0
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    run_pilot(scenario)

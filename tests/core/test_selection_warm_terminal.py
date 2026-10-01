"""Ordinary rank browsing leaves speculative engine state untouched."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_ui import InteractivePolicy

from tests.core.term_support import LiveLoop
from tests.fakes import SpeculativeFakeBackend

pytestmark = pytest.mark.current_workflow


def test_rank_navigation_does_not_speculate_before_engine_commit():
    backend = SpeculativeFakeBackend()
    engine = EpisodeEngine(backend, initial_token_ids=[7], sampling=SamplerConfig(temperature=0.0))
    loop = LiveLoop()
    states = []
    ready = threading.Event()

    class Recording:
        def read_choice(self, state):
            states.append(state)
            return loop.session.read_choice(state)

    policy = InteractivePolicy(io=Recording(), menu_size=3)

    def choose_and_commit():
        with loop:
            ready.set()
            action = policy.choose(engine, engine.observe())
            before_commit = tuple(backend.tokens)
            evals_before_commit = tuple(backend.eval_calls)
            outcome = engine.apply(action)
            return action, before_commit, evals_before_commit, outcome

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-engine-owner") as executor:
        operation = executor.submit(choose_and_commit)
        assert ready.wait(5)
        loop.wait_for(lambda text: "Command > 1" in text)
        assert states[0].warm_search_token is None
        assert backend.eval_calls == []
        # The proposal rank is prefilled; two tabs advance from rank 1 to rank 3.
        loop.keys("\t")
        loop.keys("\t")
        loop.wait_for(lambda text: "Command > 3" in text)
        assert backend.eval_calls == []
        assert backend.tokens == [7]
        loop.keys("\r")
        action, before_commit, evals_before_commit, outcome = operation.result(timeout=5)
    assert action.kind == "select-raw-rank"
    assert action.rank == 3
    assert before_commit == (7,)
    assert evals_before_commit == ()
    assert outcome.visible_token_ids == (3,)
    assert backend.eval_calls == [(3,)]
    assert backend.tokens == [7, 3]

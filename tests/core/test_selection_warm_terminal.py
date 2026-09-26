"""Ordinary choice navigation does not speculate on selected tokens."""

from __future__ import annotations

from io import StringIO
from threading import Thread
from time import monotonic, sleep

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output

from tests.fakes import SpeculativeFakeBackend
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.run_loop import run_plan
from trajectory_editor.episode_session import LiveSession
from trajectory_editor.episode_ui import InteractivePolicy
from trajectory_editor.persistent_tui import PersistentTerminalSession
from trajectory_editor.terminal_contracts import ChoiceViewState


def _until(predicate, timeout: float = 3.0) -> bool:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return True
        sleep(0.005)
    return False


def _terminal():
    return Vt100_Output(
        StringIO(), lambda: Size(rows=24, columns=80), term="xterm",
    )


@pytest.mark.invariant
def test_run_plan_rank_navigation_does_not_speculate_before_commit():
    backend = SpeculativeFakeBackend()
    engine = EpisodeEngine(
        backend, initial_token_ids=[7], sampling=SamplerConfig(temperature=0.0),
    )
    live_session = LiveSession(engine)
    errors = []

    class RecordingTerminalSession(PersistentTerminalSession):
        def __init__(self, **kwargs):
            self.choice_states = []
            super().__init__(**kwargs)

        def read_choice(self, state):
            self.choice_states.append(state)
            return super().read_choice(state)

    with create_pipe_input() as pipe:
        with RecordingTerminalSession(
            input_device=pipe, output_device=_terminal(),
        ) as terminal:
            policy = InteractivePolicy(io=terminal, menu_size=3)

            def feed():
                try:
                    assert _until(lambda: terminal._current is not None
                                  and terminal.accepting_input)
                    first_state = terminal._current.state
                    assert isinstance(first_state, ChoiceViewState)
                    assert first_state.warm_search_token is None

                    # Tab twice changes the ordinary rank selection from the
                    # prefilled proposal to rank 3. Let the old debounce window pass.
                    pipe.send_text("\t\t")
                    assert _until(lambda: terminal.choice_view.command_buffer.text == "3")
                    sleep(0.35)
                    assert backend.eval_calls == []
                    assert backend.tokens == [7]
                    pipe.send_text("\r")

                    assert _until(lambda: len(terminal.choice_states) >= 2
                                  and terminal.accepting_input)
                    assert terminal._current.state.warm_search_token is None
                    pipe.send_text("e!\r")
                except BaseException as exc:
                    errors.append(exc)
                    pipe.close()

            feeder = Thread(target=feed)
            feeder.start()
            result = run_plan(live_session, divergence_policy="handoff",
                live_policy=policy, max_live_actions=2,
            )
            feeder.join(timeout=3)
            assert not feeder.is_alive() and not errors

    assert len(result.outcomes) == 2
    assert result.outcomes[0].visible_token_ids == (3,)
    assert live_session.history_visible_token_ids == (3,)
    assert backend.eval_calls == [(3,)]
    assert backend.tokens == [7, 3]
    assert terminal.stats["warm_dispatches"] == 0
    assert terminal.stats["promotions"] == 0

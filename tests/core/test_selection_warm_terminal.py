"""The live choice session debounces selected-token warm-up on the owner thread."""

from __future__ import annotations

from dataclasses import replace
from io import StringIO
from threading import Event, Thread, get_ident
from time import monotonic, sleep

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output

from tests.fakes import SpeculativeFakeBackend
from trajectory_editor.core.actions import SelectRawRank
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_ui import _choice_from_observation
from trajectory_editor.episode_hash import token_prefix_sha256
from trajectory_editor.persistent_tui import PersistentTerminalSession
from trajectory_editor.terminal_contracts import ChoiceViewState

TEST_WARM_DELAY = 0.25
TIMER_TOLERANCE = 0.03


def _until(predicate, timeout: float = 3.0) -> bool:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return True
        sleep(0.005)
    return False


def _terminal():
    output = Vt100_Output(
        StringIO(), lambda: Size(rows=24, columns=80), term="xterm",
    )
    return output


def _choice(engine: EpisodeEngine, warm):
    observation = engine.observe()
    candidates = engine.candidates(observation, count=3)
    choice = _choice_from_observation(
        engine, observation, candidates,
        context_text_tail=observation.context_text,
        context_token_sha256=token_prefix_sha256(list(observation.prefix_token_ids)),
        serial=1,
    )
    state = ChoiceViewState(
        choice, engine.remaining, candidates, lambda text, mode: text,
        resolve_candidate=lambda rank: engine.candidates(
            observation, start_rank=rank, count=1,
        )[0],
        initial_command=str(observation.proposal_raw_rank),
        warm_selection=warm,
        cancel_warm_selection=engine.discard_speculative_accept,
    )
    return observation, state


@pytest.mark.invariant
def test_fast_tab_sequence_warms_only_final_rank_and_enter_promotes_it():
    backend = SpeculativeFakeBackend()
    engine = EpisodeEngine(
        backend, initial_token_ids=[7], sampling=SamplerConfig(temperature=0.0),
    )
    completed = Event()
    calls = []
    owner = get_ident()

    def warm(rank, token_id, generation, cancelled):
        calls.append((rank, token_id, generation, monotonic(), get_ident()))
        result = engine.speculate_accept(
            observation, raw_rank=rank, token_id=token_id,
            generation=generation, cancelled=cancelled,
        )
        completed.set()
        return result

    observation, state = _choice(engine, warm)
    selected_at = []
    errors = []
    with create_pipe_input() as pipe:
        with PersistentTerminalSession(
            input_device=pipe,
            output_device=_terminal(),
            warm_debounce_mode="fixed",
            fixed_delay=TEST_WARM_DELAY,
        ) as session:
            def feed():
                try:
                    assert _until(lambda: session._current is not None
                                  and session._current.state is state
                                  and session.accepting_input)
                    pipe.send_text("\t\t")
                    assert _until(lambda: session.choice_view.command_buffer.text == "3")
                    selected_at.append(monotonic())
                    assert completed.wait(3)
                    pipe.send_text("\r")
                except BaseException as exc:
                    errors.append(exc)
                    pipe.close()

            feeder = Thread(target=feed)
            feeder.start()
            raw = session.read_choice(state)
            feeder.join(timeout=3)
            assert not feeder.is_alive() and not errors
    assert raw == "3"
    assert [(rank, token) for rank, token, *_ in calls] == [(3, 3)]
    assert calls[0][3] - selected_at[0] >= TEST_WARM_DELAY - TIMER_TOLERANCE
    assert calls[0][4] == owner
    assert backend.eval_calls == [(3,)]
    engine.apply(SelectRawRank(3))
    assert backend.eval_calls == [(3,)]
    assert backend.tokens == [7, 3]


@pytest.mark.invariant
def test_typed_rank_warms_after_pause_but_changed_enter_misses():
    backend = SpeculativeFakeBackend()
    engine = EpisodeEngine(
        backend, initial_token_ids=[7], sampling=SamplerConfig(temperature=0.0),
    )
    completed = Event()
    calls = []

    def warm(rank, token_id, generation, cancelled):
        calls.append((rank, token_id, monotonic()))
        result = engine.speculate_accept(
            observation, raw_rank=rank, token_id=token_id,
            generation=generation, cancelled=cancelled,
        )
        completed.set()
        return result

    observation, state = _choice(engine, warm)
    selected_at = []
    errors = []
    with create_pipe_input() as pipe:
        with PersistentTerminalSession(
            input_device=pipe,
            output_device=_terminal(),
            warm_debounce_mode="fixed",
            fixed_delay=TEST_WARM_DELAY,
        ) as session:
            def feed():
                try:
                    assert _until(lambda: session._current is not None
                                  and session._current.state is state
                                  and session.accepting_input)
                    pipe.send_text("3")
                    assert _until(lambda: session.choice_view.command_buffer.text == "3")
                    selected_at.append(monotonic())
                    assert completed.wait(3)
                    pipe.send_text("\x7f2\r")
                except BaseException as exc:
                    errors.append(exc)
                    pipe.close()

            feeder = Thread(target=feed)
            feeder.start()
            raw = session.read_choice(state)
            feeder.join(timeout=3)
            assert not feeder.is_alive() and not errors
    assert raw == "2"
    assert [(rank, token) for rank, token, _ in calls] == [(3, 3)]
    assert calls[0][2] - selected_at[0] >= TEST_WARM_DELAY - TIMER_TOLERANCE
    assert engine._prepared_accept is None
    engine.apply(SelectRawRank(2))
    assert backend.eval_calls == [(3,), (2,)]


@pytest.mark.invariant
def test_selection_change_during_uninterruptible_warm_cancels_old_result():
    started = Event()
    release = Event()

    def block_after_eval():
        started.set()
        assert release.wait(3)

    backend = SpeculativeFakeBackend(on_eval=block_after_eval)
    engine = EpisodeEngine(
        backend, initial_token_ids=[7], sampling=SamplerConfig(temperature=0.0),
    )

    def warm(rank, token_id, generation, cancelled):
        return engine.speculate_accept(
            observation, raw_rank=rank, token_id=token_id,
            generation=generation, cancelled=cancelled,
        )

    observation, state = _choice(engine, warm)
    errors = []
    with create_pipe_input() as pipe:
        with PersistentTerminalSession(
            input_device=pipe,
            output_device=_terminal(),
            warm_debounce_mode="fixed",
            fixed_delay=TEST_WARM_DELAY,
        ) as session:
            def feed():
                try:
                    assert _until(lambda: session._current is not None
                                  and session._current.state is state
                                  and session.accepting_input)
                    assert started.wait(3)
                    pipe.send_text("\t\r")
                    assert _until(lambda: session._current.response.done())
                except BaseException as exc:
                    errors.append(exc)
                    pipe.close()
                finally:
                    release.set()

            feeder = Thread(target=feed)
            feeder.start()
            raw = session.read_choice(state)
            feeder.join(timeout=3)
            assert not feeder.is_alive() and not errors
    assert raw == "2"
    assert engine._prepared_accept is None
    assert backend.tokens == [7]
    backend.on_eval = None
    engine.apply(SelectRawRank(2))
    assert backend.eval_calls == [(1,), (2,)]

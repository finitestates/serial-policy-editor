"""Search-priority speculative warms carry only into matching selections."""

from __future__ import annotations

from io import StringIO
from threading import Event, Thread
from time import monotonic, sleep

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output

from tests.fakes import SnapshotFakeBackend
from trajectory_editor.core.actions import Accept, SelectRawRank
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_ui import InteractivePolicy, _choice_from_observation
from trajectory_editor.persistent_tui import PersistentTerminalSession
from trajectory_editor.terminal_contracts import ChoiceFeedback, ChoiceViewState


TEST_WARM_DELAY = 0.22
TIMER_TOLERANCE = 0.04


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


class SearchBackend(SnapshotFakeBackend):
    def __init__(self, *, on_eval=None):
        super().__init__(on_eval=on_eval)
        self.pieces = {**self.pieces, 3: "TERM", 4: "SECOND"}

    def tokenize(self, text: str, *, add_bos: bool = False, special: bool = False):
        if not add_bos and text == "TERM":
            return [3]
        if not add_bos and text == "MULTI":
            return [3, 4]
        if not add_bos and text == "SECOND":
            return [4]
        return super().tokenize(text, add_bos=add_bos, special=special)


def _engine(backend=None):
    return EpisodeEngine(
        backend or SearchBackend(),
        initial_token_ids=[7],
        sampling=SamplerConfig(temperature=0.0),
    )


def _search_state(
    engine: EpisodeEngine,
    observation,
    *,
    token_id: int,
    search_lens_active: bool = True,
    suggestions: tuple[str, ...] = (),
    calls: list | None = None,
    completed: dict[int, Event] | None = None,
):
    rank = observation.statistics.raw_rank(token_id)
    target = (rank, token_id)
    menu = tuple(engine.candidates(observation, count=3))
    start = max(1, rank - 2) if search_lens_active else 1
    stop = min(len(observation.logits), rank + 2) if search_lens_active else len(menu)
    rows = tuple(engine.candidates(observation, start_rank=start, count=stop - start + 1))
    by_rank = {candidate.rank: candidate for candidate in (*menu, *rows)}
    choice = _choice_from_observation(
        engine, observation, menu, context_characters=0, serial=1,
    )

    def warm(raw_rank, selected_token_id, generation, cancelled):
        if calls is not None:
            calls.append((raw_rank, selected_token_id, monotonic()))
        result = engine.speculate_accept(
            observation, raw_rank=raw_rank, token_id=selected_token_id,
            generation=generation, cancelled=cancelled,
        )
        if completed is not None:
            completed.setdefault(selected_token_id, Event()).set()
        return result

    feedback = (
        ChoiceFeedback(
            "search", "SEARCH RESULTS",
            completion_commands=suggestions,
            initial_tab_command=str(rank) if search_lens_active else None,
        )
        if search_lens_active or suggestions else None
    )
    state = ChoiceViewState(
        choice=choice,
        remaining_tokens=engine.remaining,
        candidates=tuple(by_rank[key] for key in sorted(by_rank)),
        display_candidates=rows if search_lens_active else menu,
        resolve_insertion=lambda text, mode: text,
        resolve_candidate=lambda raw_rank: engine.candidates(
            observation, start_rank=raw_rank, count=1,
        )[0],
        target_token_id=token_id if search_lens_active else None,
        feedback=feedback,
        search_lens_active=search_lens_active,
        warm_selection=warm,
        cancel_warm_selection=engine.discard_speculative_accept,
        search_warm_target=target,
        search_warm_commands=suggestions,
        search_warm_prepared=engine.has_prepared_accept(observation, rank, token_id),
    )
    return rank, state


def _request_warm_done(session, state) -> bool:
    request = session._current
    return bool(
        request is not None
        and request.state is state
        and request.warm_future is not None
        and request.warm_future.done()
    )


@pytest.mark.invariant
def test_exact_search_warms_before_tab_and_rank_commit_promotes_it():
    backend = SearchBackend()
    engine = _engine(backend)
    observation = engine.observe()
    rank = observation.statistics.raw_rank(3)
    errors = []

    with create_pipe_input() as pipe:
        with PersistentTerminalSession(
            input_device=pipe, output_device=_terminal(),
            warm_debounce_mode="fixed", fixed_delay=TEST_WARM_DELAY,
        ) as session:
            policy = InteractivePolicy(io=session, search_radius=1)

            def feed():
                try:
                    assert _until(lambda: session._current is not None and session.accepting_input)
                    pipe.send_text("/TERM\r")
                    assert _until(lambda: (
                        session._current is not None
                        and isinstance(session._current.state, ChoiceViewState)
                        and session._current.state.search_warm_target == (rank, 3)
                        and _request_warm_done(session, session._current.state)
                    ))
                    request = session._current
                    assert request.warm_future.result() is True
                    assert request.state.search_lens_active
                    assert session.choice_view.command_buffer.text == ""
                    # Speculation restored the committed prefix before selection.
                    assert backend.tokens == [7]
                    pipe.send_text("\t")
                    assert _until(lambda: session.choice_view.command_buffer.text == str(rank))
                    pipe.send_text("\r")
                except BaseException as exc:
                    errors.append(exc)
                    pipe.close()

            feeder = Thread(target=feed)
            feeder.start()
            action = policy.choose(engine, observation)
            feeder.join(timeout=3)
            assert not feeder.is_alive() and not errors

    assert isinstance(action, SelectRawRank)
    assert action.rank == rank
    engine.apply(action)
    assert backend.eval_calls == [(3,)]
    assert backend.tokens == [7, 3]
    assert session.stats["promotions"] == 1


@pytest.mark.invariant
def test_multi_token_search_warms_first_result_and_preserves_it_through_suggestion():
    backend = SearchBackend()
    engine = _engine(backend)
    observation = engine.observe()
    first_rank = observation.statistics.raw_rank(3)
    first_suggestion = '/"TERM"'
    errors = []

    with create_pipe_input() as pipe:
        with PersistentTerminalSession(
            input_device=pipe, output_device=_terminal(),
            warm_debounce_mode="fixed", fixed_delay=TEST_WARM_DELAY,
        ) as session:
            policy = InteractivePolicy(io=session, search_radius=1)

            def feed():
                try:
                    assert _until(lambda: session._current is not None and session.accepting_input)
                    pipe.send_text("/MULTI\r")
                    assert _until(lambda: (
                        session._current is not None
                        and isinstance(session._current.state, ChoiceViewState)
                        and session._current.state.feedback is not None
                        and len(session._current.state.feedback.completion_commands) == 2
                        and session._current.state.search_warm_target == (first_rank, 3)
                        and _request_warm_done(session, session._current.state)
                    ))
                    request = session._current
                    assert request.warm_future.result() is True
                    assert backend.eval_calls == [(3,)]
                    pipe.send_text("\t")
                    assert _until(lambda: session.choice_view.command_buffer.text == first_suggestion)
                    pipe.send_text("\r")
                    assert _until(lambda: (
                        session._current is not None
                        and isinstance(session._current.state, ChoiceViewState)
                        and session._current.state.search_lens_active
                        and session._current.state.search_warm_target == (first_rank, 3)
                        and _request_warm_done(session, session._current.state)
                    ))
                    # The next view recognizes the carried prepared state; it does not eval again.
                    assert backend.eval_calls == [(3,)]
                    pipe.send_text("\t")
                    assert _until(lambda: session.choice_view.command_buffer.text == str(first_rank))
                    pipe.send_text("\r")
                except BaseException as exc:
                    errors.append(exc)
                    pipe.close()

            feeder = Thread(target=feed)
            feeder.start()
            action = policy.choose(engine, observation)
            feeder.join(timeout=3)
            assert not feeder.is_alive() and not errors

    assert isinstance(action, SelectRawRank)
    assert action.rank == first_rank
    engine.apply(action)
    assert backend.eval_calls == [(3,)]
    assert backend.tokens == [7, 3]
    assert session.stats["promotions"] == 1


@pytest.mark.invariant
def test_moving_off_blocked_search_warm_debounces_only_final_neighbor():
    started = Event()
    release = Event()
    first_eval = True

    def block_first_eval():
        nonlocal first_eval
        if first_eval:
            first_eval = False
            started.set()
            assert release.wait(3)

    backend = SearchBackend(on_eval=block_first_eval)
    engine = _engine(backend)
    observation = engine.observe()
    calls = []
    completed = {}
    _, state = _search_state(
        engine, observation, token_id=1, calls=calls, completed=completed,
    )
    errors = []
    selected_final_at = []

    with create_pipe_input() as pipe:
        with PersistentTerminalSession(
            input_device=pipe, output_device=_terminal(),
            warm_debounce_mode="fixed", fixed_delay=TEST_WARM_DELAY,
        ) as session:
            def feed():
                try:
                    assert _until(lambda: session._current is not None
                                  and session._current.state is state
                                  and session.accepting_input)
                    assert started.wait(3)
                    pipe.send_text("\t")
                    assert _until(lambda: session.choice_view.command_buffer.text == "1")
                    pipe.send_text("\t")
                    assert _until(lambda: session.choice_view.command_buffer.text == "2")
                    pipe.send_text("\t")
                    assert _until(lambda: session.choice_view.command_buffer.text == "3")
                    selected_final_at.append(monotonic())
                    release.set()
                    assert completed.setdefault(3, Event()).wait(3)
                    assert _until(lambda: engine.has_prepared_accept(observation, 3, 3))
                    assert calls[-1][:2] == (3, 3)
                    assert calls[-1][2] - selected_final_at[0] >= TEST_WARM_DELAY - TIMER_TOLERANCE
                    assert not any(call[:2] == (3, 2) for call in calls[1:])
                    pipe.send_text("\r")
                except BaseException as exc:
                    errors.append(exc)
                    release.set()
                    pipe.close()

            feeder = Thread(target=feed)
            feeder.start()
            raw = session.read_choice(state)
            feeder.join(timeout=3)
            assert not feeder.is_alive() and not errors

    assert raw == "3"
    assert calls[0][:2] == (1, 1)
    assert calls[-1][:2] == (3, 3)
    assert not any(call[:2] == (2, 2) for call in calls)
    engine.apply(SelectRawRank(3))
    assert backend.eval_calls == [(1,), (3,)]
    assert backend.tokens == [7, 3]


@pytest.mark.invariant
def test_blank_enter_keeps_accept_semantics_without_false_search_promotion():
    backend = SearchBackend()
    engine = _engine(backend)
    observation = engine.observe()
    _, state = _search_state(engine, observation, token_id=3)
    warmed = Event()
    errors = []
    original_warm = state.warm_selection

    def warm(rank, token_id, generation, cancelled):
        result = original_warm(rank, token_id, generation, cancelled)
        warmed.set()
        return result

    object.__setattr__(state, "warm_selection", warm)

    with create_pipe_input() as pipe:
        with PersistentTerminalSession(
            input_device=pipe, output_device=_terminal(),
            warm_debounce_mode="fixed", fixed_delay=TEST_WARM_DELAY,
        ) as session:
            def feed():
                try:
                    assert _until(lambda: session._current is not None
                                  and session._current.state is state
                                  and session.accepting_input)
                    assert warmed.wait(3)
                    assert _request_warm_done(session, state)
                    pipe.send_text("\r")
                except BaseException as exc:
                    errors.append(exc)
                    pipe.close()

            feeder = Thread(target=feed)
            feeder.start()
            raw = session.read_choice(state)
            feeder.join(timeout=3)
            assert not feeder.is_alive() and not errors

    assert raw == ""
    assert session.stats["promotions"] == 0
    assert engine._prepared_accept is None
    engine.apply(Accept())
    assert backend.eval_calls == [(3,), (1,)]
    assert backend.tokens == [7, 1]

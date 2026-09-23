"""Active, model-free gates for the persistent terminal application."""

from concurrent.futures import Future
from dataclasses import replace
from io import StringIO
from threading import Event, Thread, get_ident
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output

import trajectory_editor.persistent_tui as persistent_tui
import trajectory_editor.tui as tui
from trajectory_editor.core.candidates import Candidate
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.ui import ChoiceSet
from trajectory_editor.live_tui import PreviewPending, action_preview
from trajectory_editor.persistent_tui import PersistentTerminalSession, _Request
from trajectory_editor.terminal_contracts import (
    BoundaryReview, ChoiceViewState, EdgeViewState, PromptRequest,
)
from trajectory_editor.tui import TerminalIO


def _choice_state(resolve=lambda text, mode: text):
    candidate = Candidate(1, 2, " x", .8, False, .8)
    choice = ChoiceSet(
        "choice", "prompt", 0, 0, "0" * 64, "P", 2, " x", .8, .8,
        False, (candidate,), vocabulary_size=5, proposal_raw_rank=1,
    )
    return ChoiceViewState(choice, None, (candidate,), resolve)


def _terminal():
    stream = StringIO()
    output = Vt100_Output(stream, lambda: Size(rows=24, columns=80), term="xterm")
    return stream, output


def _send_when_ready(session, pipe, state, text):
    _wait_until_ready(session, pipe, state)
    pipe.send_text(text)


def _wait_until_ready(session, pipe, state):
    deadline = monotonic() + 3
    while monotonic() < deadline:
        if session._current is not None and session._current.state is state and session.accepting_input:
            return
        sleep(.005)
    pipe.close()
    raise AssertionError(f"terminal never accepted {type(state).__name__}")


def test_one_application_transitions_across_choice_review_edge_page_prompt_choice():
    stream, output = _terminal()
    prompt = PromptRequest("Name> ")
    page = PromptRequest("", body="Long page\nsecond line", page=True)
    edge = EdgeViewState("episode", 0, 3, 3, "temperature 1")
    choice = _choice_state()
    review = replace(choice, review=BoundaryReview(
        0, 0, "P", "0" * 64, {"kind": "token-boundary"},
    ))
    next_choice = replace(choice, choice=replace(choice.choice, choice_set_id="next"))
    with create_pipe_input() as pipe:
        with PersistentTerminalSession(input_device=pipe, output_device=output) as session:
            def complete(read, state, sent):
                feeder = Thread(target=_send_when_ready, args=(session, pipe, state, sent))
                feeder.start()
                value = read(state)
                feeder.join(timeout=3)
                assert not feeder.is_alive()
                return value

            application = session.application
            assert complete(session.read_choice, choice, "1\r") == "1"
            choice_view = session.choice_view
            assert complete(session.read_choice, review, "\r") == "\x1b"
            assert complete(session.read_edge, edge, "c\r") == "c"
            assert complete(session.prompt, page, "\r") == ""
            assert complete(session.prompt, prompt, "name\r") == "name"
            assert session._prompt_view.body.text == ""
            assert complete(session.read_choice, next_choice, "1\r") == "1"
            assert session.application is application
            assert session.choice_view is choice_view
            assert session.choice_view is not None and session.edge_view is not None
    assert "\x1b[?1049h" in stream.getvalue()
    assert "\x1b[?1049l" in stream.getvalue()


def test_live_composer_rejects_empty_then_returns_to_edge_after_cancel():
    stream, output = _terminal()
    prompt = PromptRequest("New prompt > ", multiline=True, isolated=True)
    edge = EdgeViewState("episode", 0, 3, 3, "sampler")
    with create_pipe_input() as pipe:
        with PersistentTerminalSession(input_device=pipe, output_device=output) as session:
            def feed_prompt():
                _wait_until_ready(session, pipe, prompt)
                pipe.send_text("\x1b\r")
                deadline = monotonic() + 3
                while monotonic() < deadline and session._prompt_view._error == "":
                    sleep(.005)
                assert session._prompt_view._error == "Write at least one character."
                assert not session._current.response.done()
                pipe.send_text("\x04")

            feeder = Thread(target=feed_prompt)
            feeder.start()
            assert session.prompt(prompt) is None
            feeder.join(timeout=3)
            assert not feeder.is_alive()
            edge_feeder = Thread(target=_send_when_ready, args=(session, pipe, edge, "q\r"))
            edge_feeder.start()
            assert session.read_edge(edge) == "q"
            edge_feeder.join(timeout=3)
            assert not edge_feeder.is_alive()
    assert "\x1b[?1049l" in stream.getvalue()


def test_live_composer_preserves_multiline_text_on_submit():
    stream, output = _terminal()
    prompt = PromptRequest("New prompt > ", multiline=True, isolated=True)
    with create_pipe_input() as pipe:
        with PersistentTerminalSession(input_device=pipe, output_device=output) as session:
            feeder = Thread(target=_send_when_ready, args=(
                session, pipe, prompt, "first\rsecond\x1b\r",
            ))
            feeder.start()
            assert session.prompt(prompt) == "first\nsecond"
            feeder.join(timeout=3)
            assert not feeder.is_alive()
    assert "\x1b[?1049l" in stream.getvalue()


def test_captured_output_replays_once_after_live_terminal_restoration(monkeypatch, capsys):
    stream, output = _terminal()

    def create_session(*, theme):
        return PersistentTerminalSession(input_device=pipe, output_device=output, theme=theme)

    terminal = TerminalIO(live_choices=False)
    terminal._live_choices = True
    monkeypatch.setattr(persistent_tui, "PersistentTerminalSession", create_session)
    with create_pipe_input() as pipe:
        with pytest.raises(EditorError, match="startup failed"):
            with terminal.session():
                print("loading message")
                print("diagnostic message", file=tui.sys.stderr)
                raise EditorError("startup failed")
    assert "\x1b[?1049l" in stream.getvalue()
    captured = capsys.readouterr()
    assert captured.out == "loading message\n"
    assert captured.err == "diagnostic message\n"


def test_preview_callback_runs_on_episode_owner_thread():
    stream, output = _terminal()
    resolved = Event()
    callback_threads = []

    def resolve(text, mode):
        callback_threads.append(get_ident())
        resolved.set()
        return text

    state = _choice_state(resolve)
    with create_pipe_input() as pipe:
        with PersistentTerminalSession(input_device=pipe, output_device=output) as session:
            def feed():
                _send_when_ready(session, pipe, state, "x test")
                if not resolved.wait(3):
                    pipe.close()
                    return
                pipe.send_text("\r")

            feeder = Thread(target=feed)
            feeder.start()
            owner = get_ident()
            assert session.read_choice(state) == "x test"
            feeder.join(timeout=3)
            assert not feeder.is_alive()
    assert callback_threads == [owner]


def test_unexpected_preview_failure_wakes_owner_and_restores_terminal():
    stream, output = _terminal()
    called = Event()

    def fail(text, mode):
        called.set()
        raise RuntimeError("backend preview failed")

    state = _choice_state(fail)
    with pytest.raises(RuntimeError, match="backend preview failed"):
        with create_pipe_input() as pipe:
            with PersistentTerminalSession(input_device=pipe, output_device=output) as session:
                feeder = Thread(target=_send_when_ready, args=(session, pipe, state, "x fail"))
                feeder.start()
                try:
                    session.read_choice(state)
                finally:
                    feeder.join(timeout=3)
                    assert not feeder.is_alive()
    assert called.is_set()
    assert "\x1b[?1049l" in stream.getvalue()


def test_expected_preview_rejection_remains_an_editable_choice():
    stream, output = _terminal()
    rejected = Event()

    def reject(text, mode):
        rejected.set()
        raise EditorError("write has no tokens")

    state = _choice_state(reject)
    with create_pipe_input() as pipe:
        with PersistentTerminalSession(input_device=pipe, output_device=output) as session:
            def feed():
                _send_when_ready(session, pipe, state, "x invalid")
                assert rejected.wait(3)
                pipe.send_text("\r")

            feeder = Thread(target=feed)
            feeder.start()
            assert session.read_choice(state) == "x invalid"
            feeder.join(timeout=3)
            assert not feeder.is_alive()
            assert session._failure is None
    assert "\x1b[?1049l" in stream.getvalue()


def test_candidate_preview_validation_is_rendered_as_feedback():
    state = _choice_state()

    def reject(rank):
        raise EditorError(f"rank {rank} is unavailable")

    preview = action_preview(
        state.choice, "2", state.candidates, state.resolve_insertion,
        resolve_candidate=reject,
    )
    assert not preview.valid
    assert "rank 2 is unavailable" in preview.detail


def test_superseded_and_abandoned_previews_are_cancelled():
    session = PersistentTerminalSession()
    request = _Request(_choice_state())
    with pytest.raises(PreviewPending):
        session._preview(request, ("insertion", "first", "exact"), lambda: "first")
    first = request.previews[("insertion", "first", "exact")]
    with pytest.raises(PreviewPending):
        session._preview(request, ("insertion", "second", "exact"), lambda: "second")
    assert first.cancelled()

    captured = []
    session._owner = get_ident()
    session._thread = SimpleNamespace(is_alive=lambda: True)

    def complete_immediately(callback, active):
        pending = Future()
        active.previews["pending"] = pending
        captured.append(pending)
        active.response.set_result("done")

    session._call = complete_immediately
    assert session.prompt(PromptRequest("Next> ")) == "done"
    assert captured[0].cancelled()


def test_submitted_or_unrendered_request_suppresses_stale_input():
    session = PersistentTerminalSession()
    old = _Request(PromptRequest("Old> "))
    request = _Request(PromptRequest("Next> "))
    session._current = old
    session._view_ready = old
    session._events = SimpleNamespace(put=lambda event: None)
    session._submit(result="old result")
    assert old.response.result() == "old result"

    session._current = request
    session._view_ready = None
    assert not session.accepting_input
    session._submit(result="stale")
    assert not request.response.done()

    session._before_render(None)
    assert session.accepting_input
    session._submit(result="fresh")
    assert request.response.result() == "fresh"
    session._submit(result="later")
    assert request.response.result() == "fresh"


def test_typeahead_from_old_choice_does_not_submit_new_choice():
    stream, output = _terminal()
    old = _choice_state()
    new = replace(old, choice=replace(old.choice, choice_set_id="new"))
    with create_pipe_input() as pipe:
        with PersistentTerminalSession(input_device=pipe, output_device=output) as session:
            first = Thread(target=_send_when_ready, args=(session, pipe, old, "1\r9\r"))
            first.start()
            assert session.read_choice(old) == "1"
            first.join(timeout=3)
            assert not first.is_alive()

            second = Thread(target=_send_when_ready, args=(session, pipe, new, "2\r"))
            second.start()
            assert session.read_choice(new) == "2"
            second.join(timeout=3)
            assert not second.is_alive()
    assert "\x1b[?1049l" in stream.getvalue()


def test_terminal_restores_screen_when_episode_raises():
    stream, output = _terminal()
    try:
        with create_pipe_input() as pipe:
            with PersistentTerminalSession(input_device=pipe, output_device=output):
                raise RuntimeError("episode failed")
    except RuntimeError as exc:
        assert str(exc) == "episode failed"
    else:
        raise AssertionError("episode error was swallowed")
    rendered = stream.getvalue()
    assert rendered.index("\x1b[?1049h") < rendered.index("\x1b[?1049l")


def test_input_eof_releases_waiting_owner_and_restores_terminal():
    stream, output = _terminal()
    state = PromptRequest("Input> ")
    with pytest.raises(EOFError):
        with create_pipe_input() as pipe:
            with PersistentTerminalSession(input_device=pipe, output_device=output) as session:
                def close_input():
                    _wait_until_ready(session, pipe, state)
                    pipe.close()

                feeder = Thread(target=close_input)
                feeder.start()
                try:
                    session.prompt(state)
                finally:
                    feeder.join(timeout=3)
                    assert not feeder.is_alive()
    assert "\x1b[?1049l" in stream.getvalue()


def test_interrupt_releases_waiting_owner_and_restores_terminal(monkeypatch):
    stream, output = _terminal()
    state = PromptRequest("Input> ")
    monkeypatch.setattr(persistent_tui, "interrupt_main", lambda: None)
    with pytest.raises(KeyboardInterrupt):
        with create_pipe_input() as pipe:
            with PersistentTerminalSession(input_device=pipe, output_device=output) as session:
                feeder = Thread(target=_send_when_ready, args=(session, pipe, state, "\x03"))
                feeder.start()
                try:
                    session.prompt(state)
                finally:
                    feeder.join(timeout=3)
                    assert not feeder.is_alive()
    assert "\x1b[?1049l" in stream.getvalue()


def test_plain_terminal_consumes_the_same_choice_edge_and_prompt_requests(monkeypatch, capsys):
    replies = iter(("1", "c", "yes", "a"))
    monkeypatch.setattr("builtins.input", lambda prompt: next(replies))
    terminal = TerminalIO(live_choices=False)
    assert not terminal.capabilities.live_views
    with terminal.session() as session:
        assert session is None
        assert terminal.read_choice(_choice_state()) == "1"
        assert terminal.read_edge(EdgeViewState("episode", 0, 3, 3, "sampler")) == "c"
        assert terminal.prompt(PromptRequest("Confirm> ")) == "yes"
        assert terminal.prompt(PromptRequest("Chord> ", body="a (1)", isolated=True)) == "a"
    output = capsys.readouterr().out
    assert "Actions:" in output and "chord RANK RANK" in output
    assert "Live edge @ boundary 0" in output
    assert "a (1)" in output


def test_terminal_io_creates_one_live_application_for_its_session(monkeypatch):
    stream, output = _terminal()
    created = []
    implementation = PersistentTerminalSession

    def create_session(*, theme):
        session = implementation(input_device=pipe, output_device=output, theme=theme)
        created.append(session)
        return session

    terminal = TerminalIO(live_choices=False)
    terminal._live_choices = True
    monkeypatch.setattr(persistent_tui, "PersistentTerminalSession", create_session)
    state = _choice_state()
    with create_pipe_input() as pipe:
        with terminal.session() as session:
            app = session.application
            feeder = Thread(target=_send_when_ready, args=(session, pipe, state, "1\r"))
            feeder.start()
            assert terminal.read_choice(state) == "1"
            feeder.join(timeout=3)
            assert not feeder.is_alive()
            assert session.application is app
            assert created == [session]
    assert "\x1b[?1049l" in stream.getvalue()


def test_terminal_backend_fallback_is_selected_at_construction(monkeypatch):
    with monkeypatch.context() as patcher:
        patcher.setattr(tui.sys, "stdin", SimpleNamespace(isatty=lambda: True, fileno=lambda: 0))
        patcher.setattr(tui.sys, "stdout", SimpleNamespace(isatty=lambda: True, fileno=lambda: 1))
        patcher.setattr(tui.importlib.util, "find_spec", lambda name: object())
        assert not TerminalIO(live_choices=False).capabilities.live_views
        assert TerminalIO().capabilities.live_views
        assert TerminalIO().capabilities.seamless_review

        patcher.setattr(tui.sys, "stdout", SimpleNamespace(isatty=lambda: False, fileno=lambda: 1))
        assert not TerminalIO().capabilities.live_views

        patcher.setattr(tui.sys, "stdout", SimpleNamespace(isatty=lambda: True))
        assert not TerminalIO().capabilities.live_views

        patcher.setattr(tui.sys, "stdout", SimpleNamespace(isatty=lambda: True, fileno=lambda: 1))
        patcher.setattr(tui.importlib.util, "find_spec", lambda name: None)
        assert not TerminalIO().capabilities.live_views


def test_live_requests_require_the_session_context():
    terminal = TerminalIO(live_choices=False)
    terminal._live_choices = True
    with pytest.raises(RuntimeError, match="TerminalIO.session"):
        terminal.read_choice(_choice_state())
    with pytest.raises(RuntimeError, match="TerminalIO.session"):
        terminal.read_edge(EdgeViewState("episode", 0, 3, 3, "sampler"))
    with pytest.raises(RuntimeError, match="TerminalIO.session"):
        terminal.prompt(PromptRequest("Input> "))


def test_text_helpers_share_the_prompt_request(monkeypatch):
    terminal = TerminalIO(live_choices=False)
    requests = []

    def respond(request):
        requests.append(request)
        return "answer"

    monkeypatch.setattr(terminal, "prompt", respond)
    assert terminal.read("Input> ") == "answer"
    assert terminal.read_key("Confirm> ") == "answer"
    assert terminal.read_multiline_prompt() == "answer"
    terminal.page("details")
    assert requests[0] == PromptRequest("Input> ")
    assert requests[1].single_key
    assert requests[2].multiline
    assert requests[3].page and requests[3].body == "details"


def test_long_status_is_not_inferred_to_be_a_page():
    class LiveSink:
        def __init__(self):
            self.writes = []
            self.prompts = []

        def write(self, text, *, end):
            self.writes.append((text, end))

        def prompt(self, request):
            self.prompts.append(request)
            return ""

    terminal = TerminalIO(live_choices=False)
    sink = LiveSink()
    terminal._live_session = sink
    text = "\n".join(str(index) for index in range(10))
    terminal.write(text)
    assert sink.writes == [(text, "\n")]
    assert sink.prompts == []
    terminal.page(text)
    assert sink.prompts == [PromptRequest("", body=text, page=True)]

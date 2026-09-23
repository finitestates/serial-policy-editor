"""Active, model-free gates for the persistent terminal application."""

from io import StringIO
from threading import Event, Thread, get_ident
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output

from trajectory_editor.core.candidates import Candidate
from trajectory_editor.core.ui import ChoiceSet
from trajectory_editor.persistent_tui import PersistentTerminalSession, _Request
from trajectory_editor.terminal_contracts import ChoiceViewState, EdgeViewState, PromptRequest
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
    deadline = monotonic() + 3
    while monotonic() < deadline:
        if session._current is not None and session._current.state is state and session.accepting_input:
            pipe.send_text(text)
            return
        sleep(.005)
    pipe.close()
    raise AssertionError(f"terminal never accepted {type(state).__name__}")


def test_one_application_transitions_across_prompt_edge_and_choice():
    stream, output = _terminal()
    prompt = PromptRequest("Name> ")
    edge = EdgeViewState("episode", 0, 3, 3, "temperature 1")
    choice = _choice_state()
    chord = PromptRequest("Chord> ", body="a (1) | b (2)", isolated=True)
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
            assert complete(session.prompt, prompt, "name\r") == "name"
            assert complete(session.read_edge, edge, "c\r") == "c"
            assert complete(session.read_choice, choice, "1\r") == "1"
            assert complete(session.prompt, chord, "a\r") == "a"
            assert session.application is application
            assert session.choice_view is not None and session.edge_view is not None
    assert "\x1b[?1049h" in stream.getvalue()
    assert "\x1b[?1049l" in stream.getvalue()


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


def test_submitted_or_unrendered_request_suppresses_stale_input():
    session = PersistentTerminalSession()
    request = _Request(PromptRequest("Next> "))
    session._current = request
    session._view_ready = None
    session._events = SimpleNamespace(put=lambda event: None)
    assert not session.accepting_input
    session._submit(result="stale")
    assert not request.response.done()

    session._before_render(None)
    assert session.accepting_input
    session._submit(result="fresh")
    assert request.response.result() == "fresh"
    session._submit(result="later")
    assert request.response.result() == "fresh"


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


def test_live_requests_require_the_session_context():
    terminal = TerminalIO(live_choices=False)
    terminal._live_choices = True
    with pytest.raises(RuntimeError, match="TerminalIO.session"):
        terminal.read_choice(_choice_state())
    with pytest.raises(RuntimeError, match="TerminalIO.session"):
        terminal.read_edge(EdgeViewState("episode", 0, 3, 3, "sampler"))


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

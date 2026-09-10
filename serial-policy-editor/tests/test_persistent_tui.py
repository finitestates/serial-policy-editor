"""Exercise the real application loop, terminal output and episode-thread bridge."""

import threading
from dataclasses import replace
from io import StringIO
from unittest.mock import patch

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output

from tests.test_episode_runtime import engine, NoEogBackend
from trajectory_editor.edge_tui import EdgeViewState
from trajectory_editor.episode_ui import _choice_from_observation
from trajectory_editor.live_tui import ChoiceViewState
from trajectory_editor.persistent_tui import PersistentTerminalSession
from trajectory_editor.tui import TerminalIO


class RecordingOutput(Vt100_Output):
    def __init__(self):
        self.stream = StringIO()
        self.size = Size(rows=24, columns=80)
        self.events = []
        super().__init__(self.stream, lambda: self.size, term="xterm-256color", enable_cpr=False)

    def enter_alternate_screen(self):
        self.events.append("enter")
        super().enter_alternate_screen()

    def quit_alternate_screen(self):
        self.events.append("quit")
        super().quit_alternate_screen()

    def erase_down(self):
        self.events.append("erase")
        super().erase_down()


class DrivenSession(PersistentTerminalSession):
    """Send input only after each new view is painted, like a terminal user."""

    def __init__(self, pipe, output, commands):
        super().__init__(input_device=pipe, output_device=output)
        self.commands = iter(commands)
        self.seen = None
        self.views = []
        self.snapshots = []

    def _rendered(self, app):
        super()._rendered(app)
        if not app.is_done and self.accepting_input and self._current is not self.seen:
            self.seen = self._current
            self.views.append(self._current.state)
            self.snapshots.append(tuple(self.output_device.events))
            self.input_device.send_text(next(self.commands))


def choice_state(runtime=None):
    runtime = runtime or engine(max_tokens=10)
    observation = runtime.observe()
    candidates = runtime.candidates(observation, count=3)
    choice = _choice_from_observation(runtime, observation, candidates, context_characters=0, serial=1)
    return ChoiceViewState(choice, runtime.remaining, candidates,
                           resolve_insertion=lambda text, mode: text)


def test_choices_edge_prompts_and_pager_share_one_renderer():
    output = RecordingOutput()
    state = choice_state()
    with create_pipe_input() as pipe:
        with DrivenSession(pipe, output, ['[', 'm10\r', 'c\r', 'note\r', 'e', 'q', '\r']) as session:
            app = session.application
            renderer = app.renderer
            assert session.read_choice(state) == '['
            choice_view = session.choice_view
            buffer = choice_view.command_buffer
            assert session.read_choice(replace(state, feedback=None)) == 'm10'
            assert session.choice_view is choice_view
            assert session.choice_view.command_buffer is buffer
            assert session.read_edge(EdgeViewState('test', 1, 10, 9, 'seed=1')) == 'c'
            assert session.read('Note> ') == 'note'
            assert session.read('Confirm> ', single_key=True) == 'e'
            session.page('\n'.join(f'line {i}' for i in range(100)))
            assert session.read_choice(state) == ''
            assert session.application is app
            assert app.renderer is renderer
            assert all(events == ('enter', 'erase') for events in session.snapshots)
        assert output.events == ['enter', 'erase', 'erase', 'quit']


def test_submission_consumes_stale_typeahead():
    output = RecordingOutput()
    with create_pipe_input() as pipe:
        with DrivenSession(pipe, output, ['1\r\r2\r', '3\r']) as session:
            assert session.read_choice(choice_state()) == '1'
            assert session.read_choice(choice_state()) == '3'


def test_preview_callbacks_execute_on_episode_thread():
    output = RecordingOutput()
    runtime = engine(max_tokens=10)
    state = choice_state(runtime)
    observation = runtime.observe()
    owner = threading.get_ident()
    callbacks = []

    def resolve(rank):
        callbacks.append(threading.get_ident())
        return runtime.candidates(observation, start_rank=rank, count=1)[0]

    class PreviewSession(DrivenSession):
        def _rendered(self, app):
            super()._rendered(app)
            if self.accepting_input:
                future = self._current.previews.get(('candidate', 7))
                if future is not None and future.done():
                    assert future.result().rank == 7
                    self.input_device.send_text('\r')

    with create_pipe_input() as pipe:
        with PreviewSession(pipe, output, ['7']) as session:
            assert session.read_choice(replace(state, resolve_candidate=resolve)) == '7'
    assert callbacks == [owner]


@pytest.mark.parametrize('keys, expected', [('hello\r', 'hello'), ('\x04', None)])
def test_prompt_input(keys, expected):
    output = RecordingOutput()
    with create_pipe_input() as pipe:
        with DrivenSession(pipe, output, [keys]) as session:
            assert session.read('Input> ') == expected


def test_interrupt_restores_terminal():
    output = RecordingOutput()
    with create_pipe_input() as pipe:
        with pytest.raises(KeyboardInterrupt):
            with DrivenSession(pipe, output, ['\x03']) as session:
                session.read_choice(choice_state())
        assert output.events[-1] == 'quit'
        assert not session._thread.is_alive()


def test_owner_exception_restores_terminal():
    output = RecordingOutput()
    with create_pipe_input() as pipe:
        with pytest.raises(ValueError, match='engine failed'):
            with DrivenSession(pipe, output, ['\r']) as session:
                session.read_choice(choice_state())
                raise ValueError('engine failed')
    assert output.events[-1] == 'quit'


def test_cli_navigation_preserves_engine_ownership_and_prints_after_exit(tmp_path, capsys):
    from trajectory_editor.episode_cli import main
    from trajectory_editor.episode_store import EpisodeStore
    output = RecordingOutput()
    io = TerminalIO(live_choices=False)
    io._live_choices = True
    owner = threading.get_ident()

    class Backend(NoEogBackend):
        def last_logits(self):
            assert threading.get_ident() == owner
            return super().last_logits()

        def reset(self, tokens):
            assert threading.get_ident() == owner
            return super().reset(tokens)

    with create_pipe_input() as pipe:
        session = DrivenSession(pipe, output, [
            'm2\r', 'ms 7\r', '\x1b', 't A B\r', '[', '\r',
            'f0\r', 'q\r', 'p\r', 'q', 'q\r',
        ])
        with patch('trajectory_editor.episode_cli.TerminalIO', return_value=io), \
             patch('trajectory_editor.episode_cli._backend', return_value=Backend()), \
             patch('trajectory_editor.persistent_tui.PersistentTerminalSession', return_value=session):
            status = main(['--workspace', str(tmp_path/'episodes.db'), '--new-prompt', 'P',
                           '--model', 'fake.gguf', '--max-tokens', '10', '--seed', '1'])
    assert status == 0
    assert 'remains unsealed' in capsys.readouterr().out
    assert output.events.count('enter') == output.events.count('quit') == 1
    with EpisodeStore(tmp_path/'episodes.db') as store:
        assert len(store.connection.execute('SELECT * FROM episodes').fetchall()) == 2
        kinds = [row[0] for row in store.connection.execute('SELECT kind FROM interactions')]
        assert 'seamless-rewind' in kinds
        assert 'fork-requested' in kinds


def test_resize_and_input_continue_while_owner_is_busy():
    output = RecordingOutput()
    rendered = threading.Event()
    with create_pipe_input() as pipe:
        with DrivenSession(pipe, output, ['1\r', '3\r']) as session:
            assert session.read_choice(choice_state()) == '1'
            original_buffer = session.choice_view.command_buffer

            def resized(app):
                if not session.accepting_input and app.renderer._last_size == output.size:
                    rendered.set()

            def resize_and_type():
                session.application.after_render += resized
                output.size = Size(rows=45, columns=100)
                pipe.send_text('\x0c\r2\r')  # Ctrl-L and commits while work is pending.
                session.application.invalidate()

            session._call(resize_and_type)
            assert rendered.wait(3), 'UI stopped rendering while episode thread waited'
            assert original_buffer.text == '1'
            assert session.read_choice(choice_state()) == '3'


def test_insertion_preview_errors_are_visible_and_do_not_stop_session():
    output = RecordingOutput()
    owner = threading.get_ident()
    calls = []

    def resolve(text, mode):
        assert threading.get_ident() == owner
        calls.append(text)
        raise ValueError('cannot tokenize this text')

    class PreviewSession(DrivenSession):
        def _rendered(self, app):
            super()._rendered(app)
            if self.accepting_input:
                ready = [f for f in self._current.previews.values() if f.done()]
                if ready:
                    assert 'cannot tokenize this text' in ''.join(
                        text for _, text in self.choice_view._render())
                    self.input_device.send_text('\r')

    with create_pipe_input() as pipe:
        with PreviewSession(pipe, output, ['t hello']) as session:
            assert session.read_choice(replace(choice_state(), resolve_insertion=resolve)) == 't hello'
    assert calls == ['hello']


def test_terminal_render_failure_propagates_and_restores_screen():
    output = RecordingOutput()
    with create_pipe_input() as pipe:
        with pytest.raises(RuntimeError, match='render failed'):
            with DrivenSession(pipe, output, ['\r']) as session:
                session.read_choice(choice_state())
                with patch('trajectory_editor.live_tui._render_choice', side_effect=RuntimeError('render failed')):
                    session.read_choice(choice_state())
    assert output.events[-1] == 'quit'


def test_cli_error_is_printed_after_terminal_restoration(tmp_path):
    from trajectory_editor.episode_cli import main
    output = RecordingOutput()
    io = TerminalIO(live_choices=False)
    io._live_choices = True

    class ErrorStream(StringIO):
        def write(self, text):
            if 'engine failed' in text:
                assert output.events[-1] == 'quit'
            return super().write(text)

    errors = ErrorStream()
    with create_pipe_input() as pipe:
        session = DrivenSession(pipe, output, [])
        with patch('trajectory_editor.episode_cli.TerminalIO', return_value=io), \
             patch('trajectory_editor.episode_cli._backend', return_value=NoEogBackend()), \
             patch('trajectory_editor.episode_cli.EpisodeRunner.run', side_effect=RuntimeError('engine failed')), \
             patch('trajectory_editor.episode_cli.sys.stderr', errors), \
             patch('trajectory_editor.persistent_tui.PersistentTerminalSession', return_value=session):
            result = main(['--workspace', str(tmp_path/'episodes.db'), '--new-prompt', 'P',
                           '--model', 'fake.gguf', '--max-tokens', '10'])
    assert result == 2
    assert 'error: engine failed' in errors.getvalue()

"""Check terminal-cell layout, rather than just rendered text fragments."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.output import DummyOutput

from tests.test_episode_runtime import engine
from trajectory_editor.episode_ui import _choice_from_observation
from trajectory_editor.live_tui import PersistentFullscreenSession, read_live_choice
from trajectory_editor.tui import ChoiceFeedback


class ResizableOutput(DummyOutput):
    size = Size(rows=24, columns=80)

    def get_size(self):
        return self.size


class AlternateScreenOutput(DummyOutput):
    def __init__(self):
        super().__init__()
        self.alternate_screen_events = []

    def enter_alternate_screen(self):
        self.alternate_screen_events.append("enter")

    def quit_alternate_screen(self):
        self.alternate_screen_events.append("quit")


def paint(app, output):
    app.render_counter += 1
    app.layout.container.reset()
    screen = Screen()
    size = output.get_size()
    app.layout.container.write_to_screen(
        screen, MouseHandlers(), WritePosition(0, 0, size.columns, size.rows),
        '', True, None,
    )
    screen.draw_all_floats()
    return [''.join(screen.data_buffer[y][x].char for x in range(size.columns)).rstrip()
            for y in range(size.rows)]


@pytest.mark.parametrize('width', [80, 120])
@pytest.mark.parametrize('writing', [False, True])
def test_fullscreen_resize_preserves_controls_and_uses_extra_height(writing, width):
    runtime = engine(max_tokens=3)
    observation = runtime.observe()
    candidates = runtime.candidates(observation, count=8)
    choice = _choice_from_observation(runtime, observation, candidates,
                                      context_characters=0, serial=1)
    choice = replace(choice, context_text_tail='\n'.join(f'history {i}' for i in range(200)))
    output = ResizableOutput()

    async def exercise(app):
        assert app.full_screen
        assert not app.erase_when_done
        with set_app(app):
            if writing:
                app.current_buffer.text = 't ' + 'draft\n' * 100
                toggle = next(b for b in app.key_bindings.bindings if b.keys == (Keys.ControlE,))
                toggle.handler(SimpleNamespace(app=app))
            counts = []
            for height in (24, 60, 24):
                output.size = Size(rows=height, columns=width)
                lines = paint(app, output)
                text = '\n'.join(lines)
                assert ('Candidates' if writing else 'decode-p') in text
                assert any(line.startswith('›') for line in lines[-(height // 3 + 2):])
                assert 'Enter commits' in '\n'.join(lines[-2:])
                if not writing:
                    assert 'INVALID COMMAND' in text
                    assert 'Choose a rank from 1 through 8.' in text
                counts.append(sum('history ' in line or 'draft' in line for line in lines))
            assert counts[1] > counts[0] + 12
            assert counts[2] == counts[0]
        return ''

    with create_pipe_input() as pipe, patch.object(Application, 'run', lambda app: asyncio.run(exercise(app))):
        read_live_choice(choice, remaining_tokens=3, candidates=candidates,
                         resolve_insertion=lambda text, mode: text,
                         feedback=None if writing else ChoiceFeedback(
                             'error', 'INVALID COMMAND', ('Choose a rank from 1 through 8.',)),
                         input_device=pipe, output_device=output)


def test_edge_uses_full_screen_with_bottom_input():
    from trajectory_editor.edge_tui import read_live_edge_command
    output = ResizableOutput()
    output.size = Size(rows=60, columns=100)

    async def exercise(app):
        assert app.full_screen
        assert not app.erase_when_done
        with set_app(app):
            lines = paint(app, output)
            assert any('Command ›' in line for line in lines[-3:])
            assert 'Enter submits' in lines[-1]
        return 'c'

    with create_pipe_input() as pipe, patch.object(Application, 'run', lambda app: asyncio.run(exercise(app))):
        assert read_live_edge_command(
            episode_id='test', boundary=1, current_budget=100, remaining_tokens=99,
            sampler_summary='seed=1', input_device=pipe, output_device=output,
        ) == 'c'


def test_persistent_fullscreen_session_keeps_alternate_screen_between_choices():
    runtime = engine(max_tokens=3)
    observation = runtime.observe()
    candidates = runtime.candidates(observation, count=3)
    choice = _choice_from_observation(
        runtime, observation, candidates, context_characters=0, serial=1
    )
    output = AlternateScreenOutput()

    with create_pipe_input() as pipe:
        pipe.send_text("[[")
        with PersistentFullscreenSession(input_device=pipe, output_device=output) as session:
            assert output.alternate_screen_events == ["enter"]
            for _ in range(2):
                assert read_live_choice(
                    choice,
                    remaining_tokens=3,
                    candidates=candidates,
                    resolve_insertion=lambda text, mode: text,
                    input_device=session.input_device,
                    output_device=session.output_device,
                ) == "["
                assert output.alternate_screen_events == ["enter"]

        assert output.alternate_screen_events == ["enter", "quit"]


def test_persistent_fullscreen_session_restores_terminal_on_exception():
    output = AlternateScreenOutput()

    with create_pipe_input() as pipe:
        with pytest.raises(KeyboardInterrupt):
            with PersistentFullscreenSession(input_device=pipe, output_device=output):
                raise KeyboardInterrupt

    assert output.alternate_screen_events == ["enter", "quit"]

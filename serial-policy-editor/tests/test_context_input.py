import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from tests.test_episode_runtime import engine
from trajectory_editor.episode_ui import InteractivePolicy, _choice_from_observation
from trajectory_editor.live_tui import _context_view, _context_rows, action_preview, read_live_choice
from trajectory_editor.tui import parse_command

@pytest.mark.parametrize('mode', ['t', 'x'])
@pytest.mark.parametrize('prefill', [None, '1'])
@pytest.mark.parametrize('paste', [False, 'command', 'payload'])
def test_multiline_raw_input(mode, prefill, paste):
    runtime = engine(max_tokens=3)
    observation = runtime.observe()
    candidates = runtime.candidates(observation, count=3)
    choice = _choice_from_observation(runtime, observation, candidates, context_characters=0, serial=1)
    payload = '\n  hello\t世界 "quotes" \\n\n\nend  \n'
    command = mode + ' ' + payload
    if paste == 'command':
        keys = '\x1b[200~' + command + '\x1b[201~'
    elif paste == 'payload':
        keys = mode + ' ' + '\x1b[200~' + payload + '\x1b[201~'
    else:
        keys = command.replace('\n', '\x1b\r')
    with create_pipe_input() as pipe:
        pipe.send_text(keys + '\r')
        result = read_live_choice(choice, remaining_tokens=runtime.remaining,
                                 candidates=candidates, resolve_insertion=lambda text, mode: text,
                                 initial_command=prefill, input_device=pipe, output_device=DummyOutput())
    assert result == command
    parsed = parse_command(result, menu_size=3, default_hold_tokens=10)
    assert parsed.action.supplied_text == payload
    assert action_preview(choice, result, candidates, lambda text, mode: text).appended_text == payload
    assert runtime.boundary == 0

def test_context_scroll_and_highlight():
    context = '\n'.join(f'line {i}' for i in range(30))
    latest, count = _context_view(context, 'PROPOSAL', 80, 30)
    text = ''.join(value for _, value in latest)
    assert 'line 0\n' not in text
    assert 'line 29PROPOSAL' in text
    assert count == 11
    assert any(style == 'class:proposal' for style, _ in latest)
    oldest, _ = _context_view(context, 'PROPOSAL', 80, 30, 10000)
    text = ''.join(value for _, value in oldest)
    assert 'line 0\n' in text
    assert 'PROPOSAL' not in text

def test_context_wraps_wide_characters_and_tabs():
    from prompt_toolkit.utils import get_cwidth
    rows = _context_rows('界界\tx', '', 4)
    assert all(sum(get_cwidth(text) for _, text in row) <= 4 for row in rows)

def test_default_keeps_full_context():
    from dataclasses import replace
    runtime = engine(max_tokens=3)
    observation = replace(runtime.observe(), context_text='older ' * 200)
    choice = _choice_from_observation(runtime, observation, (), context_characters=InteractivePolicy().context_characters, serial=1)
    assert choice.context_text_tail == observation.context_text

@pytest.mark.parametrize('height', [24, 30, 50])
def test_writing_layout_stays_fixed_as_draft_grows(height):
    from unittest.mock import patch
    from trajectory_editor.live_tui import _render_choice, _writing_sizes
    runtime = engine(max_tokens=3)
    observation = runtime.observe()
    candidates = runtime.candidates(observation, count=3)
    choice = _choice_from_observation(runtime, observation, candidates, context_characters=0, serial=1)
    with patch('trajectory_editor.live_tui._terminal_size', return_value=(80, height)):
        outputs = [''.join(text for _, text in _render_choice(
            choice, candidates, command, None, lambda text, mode: text, None))
            for command in ['t ', 't short', 'x ' + ('long line\n' * 100)]]
    assert len({output.count('\n') for output in outputs}) == 1
    assert all('Writing' in output for output in outputs)
    assert outputs[0].count('\n') + _writing_sizes(height)[0] + 2 <= height
    assert '101 lines' in outputs[-1]


def test_editing_prefix_restores_layout_without_losing_draft():
    from trajectory_editor.live_tui import _is_writing
    runtime = engine(max_tokens=3)
    observation = runtime.observe()
    candidates = runtime.candidates(observation, count=3)
    choice = _choice_from_observation(runtime, observation, candidates, context_characters=0, serial=1)
    with create_pipe_input() as pipe:
        pipe.send_text('t draft\x01\x1b[3~\x1b[3~x \r')
        result = read_live_choice(choice, remaining_tokens=3, candidates=candidates,
                                 resolve_insertion=lambda text, mode: text,
                                 input_device=pipe, output_device=DummyOutput())
    assert result == 'x draft'
    assert _is_writing(result)
    assert not _is_writing('draft')

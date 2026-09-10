"""Opt-in real-model navigation through the persistent terminal application."""
from contextlib import ExitStack
import threading
from unittest.mock import patch

import pytest
from prompt_toolkit.input import create_pipe_input

from tests.test_llama_sampler_smoke import model  # Shared opt-in GGUF fixture.
from tests.test_persistent_tui import DrivenSession, RecordingOutput
from trajectory_editor.decoder import LlamaCppDecoder
from trajectory_editor.episode_cli import main
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.live_tui import ChoiceViewState
from trajectory_editor.tui import TerminalIO

pytestmark = pytest.mark.llama_smoke


def test_real_model_persistent_navigation(tmp_path, model, capsys):
    owner = threading.get_ident()
    calls = []
    output = RecordingOutput()
    io = TerminalIO(live_choices=False)
    io._live_choices = True
    workspace = tmp_path / 'live.sqlite3'

    def checked(name):
        original = getattr(LlamaCppDecoder, name)
        def call(self, *args, **kwargs):
            assert threading.get_ident() == owner, f'{name} escaped the episode thread'
            calls.append(name)
            return original(self, *args, **kwargs)
        return call

    with create_pipe_input() as pipe, ExitStack() as patches:
        session = DrivenSession(pipe, output, [
            'm20\r', 'ms 100\r', '\x1b', 't hello world\r', '[', '\r',
            'f0\r', 'h 2\r', 'q\r', 's temperature=0.7\r', 'c\r', 'q\r', 'q\r',
        ])
        patches.enter_context(patch('trajectory_editor.episode_cli.TerminalIO', return_value=io))
        patches.enter_context(patch('trajectory_editor.persistent_tui.PersistentTerminalSession', return_value=session))
        for name in ('reset', 'eval', 'last_logits', 'tokenize', 'render', 'branch_to_prefix'):
            patches.enter_context(patch.object(LlamaCppDecoder, name, checked(name)))
        status = main([
            '--workspace', str(workspace), '--model', str(model), '--seed', '71',
            '--new-prompt', 'Continue this numbered list of animals: 1. cat 2. dog 3.',
            '--n-gpu-layers', '0', '--n-threads', '2', '--n-threads-batch', '2',
            '--n-ctx', '256', '--n-batch', '64', '--max-tokens', '20',
        ])
    assert status == 0
    assert 'remains unsealed' in capsys.readouterr().out
    assert 'branch_to_prefix' in calls and 'eval' in calls
    assert output.events == ['enter', 'erase', 'erase', 'quit']
    choices = [state for state in session.views if isinstance(state, ChoiceViewState)]
    assert all(state.choice.vocabulary_size > 10000 for state in choices)
    with EpisodeStore(workspace) as store:
        assert store.connection.execute('SELECT COUNT(*) FROM episodes').fetchone()[0] == 2
        kinds = {row[0] for row in store.connection.execute('SELECT kind FROM interactions')}
        assert {'seamless-rewind', 'fork-requested', 'menu-expanded'} <= kinds

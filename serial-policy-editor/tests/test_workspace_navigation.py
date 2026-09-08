from unittest.mock import patch

import pytest

from tests.fakes import ScriptedIO
from tests.test_episode_runtime import NoEogBackend, LiveScriptedIO, engine
from tests.test_replay_eog import create
from trajectory_editor.domain import EditorError
from trajectory_editor.episode_actions import Hold, Write
from trajectory_editor.episode_cli import (
    _rewind_episode, _live_edge_menu, _load_episode_backend,
    _model_continuation, _interactive_policy, build_parser, main,
)
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_ui import InteractivePolicy


def test_rewind_crosses_checkpoint_and_trims_write_with_original_evidence(tmp_path):
    runtime = engine(NoEogBackend(), max_tokens=20)
    with EpisodeStore(tmp_path / 'episodes.sqlite3') as store:
        identifier = create(store, runtime)
        original = runtime.apply(Write(' A B', 'exact'))
        store.record_action(identifier, 0, original)
        store.record_interaction(identifier, 2, 'checkpoint-resume', {})
        store.record_action(identifier, 1, runtime.apply(Hold(2)))
        policy = InteractivePolicy(io=ScriptedIO([]), store=store, episode_id=identifier, seamless=True)
        assert policy._seamless_targets(runtime, runtime.boundary) == (0, 1, 2, 3, 4)
        _rewind_episode(store, identifier, runtime, 1)
        action, expectation = store.replay_tape(identifier)[0]
        assert action == Write(' A', 'exact')
        assert expectation.token_ids == (1,)
        assert store.actions(identifier)[0]['arguments']['original_write'] == original.action.to_dict()
        reference = engine(NoEogBackend(), max_tokens=20)
        assert reference.apply(action, expectation=expectation).divergence is None
        assert store.interactions(identifier) == []


def test_numbers_titles_and_filter_survive_reopen(tmp_path):
    path = tmp_path / 'episodes.sqlite3'
    with EpisodeStore(path) as store:
        first = create(store, engine(), 'a')
        second = create(store, engine(), 'b')
        store.rename(first, 'A title')
        store.finish_episode(second, visible_text='', terminal_token_id=None, terminal_reason='menu-end')
        assert '#2' not in store.workspace_list()
        assert '#2' in store.workspace_list(include_finished=True)
    with EpisodeStore(path) as store:
        assert store.resolve_id('#1') == first
        assert store.label(first) == '#1  A title'
        assert store.resolve_id('#2') == second
        assert store.resolve_id(first) == first
        create(store, engine(), 'c')
        assert store.resolve_id('#3') == 'c'
        with pytest.raises(EditorError):
            store.resolve_id('#99')


@pytest.mark.parametrize('commands', [['#2'], ['ls', '#2'], ['ls all', '#2']])
def test_edge_switch_by_number(tmp_path, commands):
    with EpisodeStore(tmp_path / 'episodes.sqlite3') as store:
        runtime = engine()
        first = create(store, runtime, 'a')
        create(store, engine(), 'b')
        assert _live_edge_menu(ScriptedIO(commands), store, first, runtime) == ('switch', 'b')


def test_live_rewind_default_and_plain_edge_command(tmp_path):
    with EpisodeStore(tmp_path / 'episodes.sqlite3') as store:
        runtime = engine()
        identifier = create(store, runtime)
        args = build_parser().parse_args([])
        assert _interactive_policy(args, store, identifier, LiveScriptedIO([])).seamless
        store.record_action(identifier, 0, runtime.apply(Write(' A B', 'exact')))
        assert _live_edge_menu(ScriptedIO(['rewind 1', 'q']), store, identifier, runtime) == ('quit', None)
        assert runtime.visible_token_ids == [1]


def test_fork_inherits_current_unlimited_budget(tmp_path):
    io = ScriptedIO(['q', 'n off', 'q', 'f 0', 'q', 'quit'])
    path = tmp_path / 'episodes.sqlite3'
    with patch('trajectory_editor.episode_cli._backend', return_value=NoEogBackend()), patch('trajectory_editor.episode_cli.TerminalIO', return_value=io):
        assert main(['--workspace', str(path), '--model', 'fake.gguf', '--new-prompt', 'P', '--max-tokens', '20', '--plain-ui']) == 0
    with EpisodeStore(path) as store:
        assert store.get_episode(store.resolve_id('#2'))['max_tokens'] is None


def test_load_saved_model_and_options_without_cli_model():
    args = build_parser().parse_args([])
    saved = {'backend': {'model_path': '/old/model', 'backend': 'transformers', 'load_options': {'transformers_device': 'cpu'}}}
    with patch('trajectory_editor.episode_cli._backend', return_value=NoEogBackend()) as load:
        _, _, changed = _load_episode_backend(args, saved, ScriptedIO([]))
    assert not changed
    assert str(load.call_args.args[0].model) == '/old/model'
    assert load.call_args.args[0].backend == 'transformers'
    assert load.call_args.args[0].transformers_device == 'cpu'


@pytest.mark.parametrize('answer', ['n', 'y'])
def test_changed_model_requires_confirmation(answer):
    args = build_parser().parse_args(['--model', '/new/model'])
    saved = {'backend': {'model_path': '/old/model', 'backend': 'llama.cpp'}}
    with patch('trajectory_editor.episode_cli._backend', return_value=NoEogBackend()) as load:
        if answer == 'n':
            with pytest.raises(EditorError, match='cancelled'):
                _load_episode_backend(args, saved, ScriptedIO([answer]))
            load.assert_not_called()
        else:
            assert _load_episode_backend(args, saved, ScriptedIO([answer]))[2]


def test_missing_saved_model_can_be_replaced():
    args = build_parser().parse_args([])
    saved = {'backend': {'model_path': '/missing/model', 'backend': 'llama.cpp'}}
    io = ScriptedIO(['/replacement/model', 'transformers', 'y'])
    with patch('trajectory_editor.episode_cli._backend', side_effect=[OSError('missing'), NoEogBackend()]):
        _, _, changed = _load_episode_backend(args, saved, io)
    assert changed


def test_model_change_retokenizes_and_preserves_source(tmp_path):
    class NewTokenizer(NoEogBackend):
        def tokenize(self, text, **kwargs):
            self.received_text = text
            return [7, 4]
    with EpisodeStore(tmp_path / 'episodes.sqlite3') as store:
        runtime = engine()
        identifier = create(store, runtime)
        store.record_action(identifier, 0, runtime.apply(Write(' A', 'exact')))
        store.update_episode(identifier, visible_text=' A', max_tokens=None)
        tokens = store.tokens(identifier)
        backend = NewTokenizer()
        new, child = _model_continuation(store, identifier, backend, {})
        assert backend.received_text == 'P A'
        assert new.initial_token_ids == (7, 4)
        assert child != identifier
        assert store.tokens(identifier) == tokens
        assert store.get_episode(child)['metadata']['model_change_from'] == identifier


def test_switch_restores_destination_without_generating(tmp_path):
    path = tmp_path / 'episodes.sqlite3'
    with EpisodeStore(path) as store:
        create(store, engine(), 'a')
        create(store, engine(), 'b')
    io = ScriptedIO(['q', '#2', '#1', 'quit'])
    with patch('trajectory_editor.episode_cli._backend', side_effect=lambda args: NoEogBackend()), patch('trajectory_editor.episode_cli.TerminalIO', return_value=io):
        # Old records without paths can use an explicitly supplied model on
        # initial resume; switching asks for the missing model location.
        io.responses = ['q', '#2', '/fake/model', '#1', '/fake/model', 'quit']
        assert main(['--workspace', str(path), '--model', '/fake/model', '--resume', '#1', '--plain-ui']) == 0
    with EpisodeStore(path) as store:
        assert store.tokens('a') == store.tokens('b') == []
        assert len(store.list_episodes()) == 2


def test_same_model_switch_reuses_loaded_backend():
    args = build_parser().parse_args([])
    backend = NoEogBackend()
    provenance = {'backend': 'llama.cpp', 'model_path': '/saved/model', 'load_options': {'n_ctx': 2048}}
    with patch('trajectory_editor.episode_cli._backend') as load:
        result, _, changed = _load_episode_backend(
            args, {'backend': provenance}, ScriptedIO([]), use_saved=True,
            current_backend=backend, current_provenance=provenance,
        )
    assert result is backend
    assert not changed
    load.assert_not_called()


def test_switch_cancel_keeps_current_episode_and_context(tmp_path):
    path = tmp_path / 'episodes.sqlite3'
    with EpisodeStore(path) as store:
        create(store, engine(), 'a')
        create(store, engine(), 'b')
    io = ScriptedIO(['q', '#2', '', 'continue', 't hello', 'q', 'quit'])
    with patch('trajectory_editor.episode_cli._backend', return_value=NoEogBackend()), patch('trajectory_editor.episode_cli.TerminalIO', return_value=io):
        assert main(['--workspace', str(path), '--model', '/fake/model', '--resume', '#1', '--plain-ui']) == 0
    with EpisodeStore(path) as store:
        assert store.get_episode('a')['visible_text'] == ' hello'
        assert store.tokens('b') == []


def test_finished_episode_is_inspected_before_fork(tmp_path):
    path = tmp_path / 'episodes.sqlite3'
    with EpisodeStore(path) as store:
        create(store, engine(), 'a')
        create(store, engine(), 'b')
        store.finish_episode('b', visible_text='', terminal_token_id=None, terminal_reason='menu-end')
    io = ScriptedIO(['q', '#2', 'n', '#2', 'y', '/fake/model', 'quit'])
    with patch('trajectory_editor.episode_cli._backend', side_effect=lambda args: NoEogBackend()), patch('trajectory_editor.episode_cli.TerminalIO', return_value=io):
        assert main(['--workspace', str(path), '--model', '/fake/model', '--resume', '#1', '--plain-ui']) == 0
    with EpisodeStore(path) as store:
        child = store.get_episode(store.resolve_id('#3'))
        assert child['parent_episode_id'] == 'b'
        assert store.get_episode('b')['status'] == 'completed'
        assert store.tokens(child['episode_id']) == []


def test_resume_uses_saved_model_without_model_argument(tmp_path):
    path = tmp_path / 'episodes.sqlite3'
    first_io = ScriptedIO(['q', 'quit'])
    with patch('trajectory_editor.episode_cli._backend', return_value=NoEogBackend()), patch('trajectory_editor.episode_cli.TerminalIO', return_value=first_io):
        assert main(['--workspace', str(path), '--model', '/saved/model', '--new-prompt', 'P', '--plain-ui']) == 0
    with patch('trajectory_editor.episode_cli._backend', return_value=NoEogBackend()) as load, patch('trajectory_editor.episode_cli.TerminalIO', return_value=ScriptedIO(['q', 'quit'])):
        assert main(['--workspace', str(path), '--resume', '#1', '--plain-ui']) == 0
    assert str(load.call_args.args[0].model) == '/saved/model'


def test_loading_options_explicit_override_and_legacy_configuration():
    args = build_parser().parse_args(['--n-ctx', '4096'])
    args._explicit_options = {'n_ctx'}
    source = {'backend': {'backend': 'llama.cpp', 'model_path': '/saved/model',
                          'runtime_configuration': {'n_ctx': 8192, 'n_batch': 32, 'flash_attn': False}}}
    with patch('trajectory_editor.episode_cli._backend', return_value=NoEogBackend()) as load:
        _load_episode_backend(args, source, ScriptedIO([]))
    selected = load.call_args.args[0]
    assert selected.n_ctx == 4096
    assert selected.n_batch == 32
    assert selected.no_flash_attn

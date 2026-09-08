from unittest.mock import patch

import pytest

from tests.fakes import ScriptedIO
from tests.test_episode_runtime import NoEogBackend
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.episode_actions import EndGeneration, Hold, SelectRawRank, Write
from trajectory_editor.episode_cli import main
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_projector import project_fork_map
from trajectory_editor.episode_store import EpisodeStore


@pytest.fixture
def source_workspace(tmp_path):
    path = tmp_path / 'episodes.sqlite3'
    with EpisodeStore(path) as store:
        source = EpisodeEngine(NoEogBackend(), sampling=SamplingConfig(temperature=0, seed=999),
                               initial_text='P', initial_token_ids=[7], coordinate_offset=40)
        store.create_episode(episode_id='source', initial_text=source.initial_text,
            initial_token_ids=source.initial_token_ids, sampling=source.sampling,
            stream_fingerprint=source.stream_fingerprint, coordinate_offset=40,
            max_tokens=None, backend={})
        for ordinal, action in enumerate([Write(' A B', 'exact'), Hold(2)]):
            store.record_action('source', ordinal, source.apply(action))
        store.update_episode('source', visible_text=source.backend.render(source.visible_token_ids), max_tokens=None)
    return path


def run_cli(path, commands, *flags):
    io = ScriptedIO(commands)
    with patch('trajectory_editor.episode_cli._backend', return_value=NoEogBackend()), patch(
        'trajectory_editor.episode_cli.TerminalIO', return_value=io
    ):
        status = main(['--workspace', str(path), '--model', 'fake', '--plain-ui', *flags])
    assert status == 0
    return io


def start(path, commands, *flags):
    return run_cli(path, commands, '--new-prompt', 'P', '--episode-id', 'destination',
                   '--seed', '77', '--temperature', '0', '--top-k', '4', *flags)


def test_edge_replay_appends_without_new_prompt_or_episode(source_workspace):
    path = source_workspace
    with EpisodeStore(path) as store:
        source_tokens = store.tokens('source')
    start(path, ['t hello', 'q', 'spr #1', 'quit'])
    with EpisodeStore(path) as store:
        destination = store.get_episode('destination')
        assert len(store.list_episodes()) == 2
        assert destination['initial_text'] == 'P'
        assert destination['initial_token_ids'] == [7]
        assert destination['visible_text'] == ' helloP A B!!'
        assert destination['parent_episode_id'] is None
        actions = store.actions('destination')
        assert [(a['boundary_before'], a['boundary_after']) for a in actions] == [(0, 1), (1, 2), (2, 4), (4, 6)]
        assert 'replay_origin' not in actions[0]['arguments']
        assert [a['arguments']['replay_origin'] for a in actions[1:]] == [
            {'episode_id': 'source', 'boundary': 0, 'part': 'prompt'},
            {'episode_id': 'source', 'boundary': 0}, {'episode_id': 'source', 'boundary': 2}]
        assert [row['sampling_coordinate'] for row in store.tokens('destination')] == list(range(6))
        assert store.final_sampling('destination').seed == 77
        assert store.final_sampling('destination').top_k == 4
        assert store.sampling_segment('destination')['coordinate_offset'] == 0
        assert store.tokens('source') == source_tokens
        assert project_fork_map(store, 'destination') == 'P|0| hello|1|P|2| A|3| B|4|!|5|!|6|'


@pytest.mark.parametrize('boundary,expected', [(0, []), (1, [4]), (3, [4, 7, 1]), (5, [4, 7, 1, 2, 5])])
def test_rewind_across_entire_splice_and_inside_actions(source_workspace, boundary, expected):
    path = source_workspace
    start(path, ['t hello', 'q', 'spr #1', f'rewind {boundary}', 'quit'])
    with EpisodeStore(path) as store:
        assert store.get_episode('destination')['initial_token_ids'] == [7]
        assert [row['token_id'] for row in store.tokens('destination')] == expected
        assert len(store.list_episodes()) == 2
        if boundary > 1:
            origin = store.actions('destination')[2]['arguments']['replay_origin']
            assert origin == {'episode_id': 'source', 'boundary': 0}
        if boundary == 3:
            assert store.replay_tape('destination')[2][0] == Write(' A', 'exact')
        if boundary == 5:
            assert store.replay_tape('destination')[3][0] == Hold(1)
    # A fresh CLI run restores the same episode and numbering after the cut.
    run_cli(path, ['q', 'quit'], '--resume', '#2')
    with EpisodeStore(path) as store:
        assert [row['token_id'] for row in store.tokens('destination')] == expected


@pytest.mark.parametrize('budget,count', [(None, 6), (1, 1), (4, 4), (5, 4), (10, 6)])
def test_edge_replay_preserves_remaining_budget(source_workspace, budget, count):
    path = source_workspace
    flags = [] if budget is None else ['--max-tokens', str(budget)]
    commands = ['t hello'] + ([] if budget == 1 else ['q']) + ['spr #1', 'quit']
    start(path, commands, *flags)
    with EpisodeStore(path) as store:
        destination = store.get_episode('destination')
        assert destination['checkpoint_boundary'] == budget
        assert destination['max_tokens'] == budget
        assert len(store.tokens('destination')) == count
        assert len(store.list_episodes()) == 2


def test_self_replay_snapshots_finite_procedure(tmp_path):
    path = tmp_path / 'episodes.sqlite3'
    start(path, ['t hello', 'q', 'spr #1', 'spr #1', 'quit'])
    with EpisodeStore(path) as store:
        assert len(store.list_episodes()) == 1
        assert store.get_episode('destination')['visible_text'] == ' helloP helloP helloP hello'
        assert len(store.actions('destination')) == 7
        starts = [event for event in store.interactions('destination') if event['kind'] == 'replay-start']
        assert [event['payload']['action_count'] for event in starts] == [2, 4]


def test_composite_can_be_replayed_as_a_complete_procedure(source_workspace):
    path = source_workspace
    start(path, ['t hello', 'q', 'spr #1', 'quit'])
    run_cli(path, ['quit'], '--replay', '#2', '--episode-id', 'replayed')
    with EpisodeStore(path) as store:
        assert store.get_episode('replayed')['initial_text'] == 'P'
        assert store.get_episode('replayed')['visible_text'] == store.get_episode('destination')['visible_text']
        assert [row['token_id'] for row in store.tokens('replayed')] == [4, 7, 1, 2, 5, 5]


@pytest.mark.parametrize('action', [SelectRawRank(1), EndGeneration()])
def test_handoff_and_eog_keep_destination_open(tmp_path, action):
    path = tmp_path / 'episodes.sqlite3'
    with EpisodeStore(path) as store:
        source = EpisodeEngine(NoEogBackend(), sampling=SamplingConfig(temperature=0), initial_token_ids=[7])
        store.create_episode(episode_id='source', initial_text=' hello', initial_token_ids=[7],
            sampling=source.sampling, stream_fingerprint=source.stream_fingerprint,
            coordinate_offset=0, max_tokens=None, backend={})
        store.record_action('source', 0, source.apply(action))
    start(path, ['t hello', 'q', 'spr #1', 'quit'])
    with EpisodeStore(path) as store:
        destination = store.get_episode('destination')
        assert destination['visible_text'] == ' hello hello'
        assert destination['status'] == 'open'
        assert destination['terminal_token_id'] is None
        assert store.actions('destination')[-1]['arguments']['replay_origin']['episode_id'] == 'source'


def test_rewind_before_splice_restores_historical_settings(source_workspace):
    path = source_workspace
    start(path, ['t hello', 'q', 's seed=88 top_k=2', 'spr #1', 'rewind 0', 'quit'])
    with EpisodeStore(path) as store:
        restored = store.final_sampling('destination')
        assert restored.seed == 77
        assert restored.top_k == 4
        assert store.tokens('destination') == []
        assert all(event['kind'] != 'replay-start' for event in store.interactions('destination'))


def test_live_history_navigation_crosses_splice_to_boundary_zero(source_workspace):
    from tests.test_episode_runtime import LiveScriptedIO
    from trajectory_editor.tui import SEAMLESS_REACTIVATE

    path = source_workspace
    io = LiveScriptedIO(['t hello', 'q', 'spr #1', 'c', *(['['] * 6), SEAMLESS_REACTIVATE, 'q', 'quit'])
    with patch('trajectory_editor.episode_cli._backend', return_value=NoEogBackend()), patch(
        'trajectory_editor.episode_cli.TerminalIO', return_value=io
    ):
        assert main(['--workspace', str(path), '--model', 'fake', '--new-prompt', 'P',
                     '--episode-id', 'destination', '--temperature', '0']) == 0
    with EpisodeStore(path) as store:
        assert store.get_episode('destination')['initial_text'] == 'P'
        assert store.tokens('destination') == []
        assert len(store.list_episodes()) == 2


def test_prompt_only_edge_replay_inserts_exact_text(tmp_path):
    path = tmp_path / 'episodes.sqlite3'
    with EpisodeStore(path) as store:
        source = EpisodeEngine(NoEogBackend(), sampling=SamplingConfig(), initial_token_ids=[7])
        store.create_episode(episode_id='source', initial_text='P', initial_token_ids=[7],
            sampling=source.sampling, stream_fingerprint=source.stream_fingerprint,
            coordinate_offset=0, max_tokens=None, backend={})
    start(path, ['t hello', 'q', 'spr #1', 'quit'], '--max-tokens', '10')
    with EpisodeStore(path) as store:
        assert store.get_episode('destination')['visible_text'] == ' helloP'
        assert store.get_episode('destination')['checkpoint_boundary'] == 10
        assert len(store.actions('destination')) == 2
        assert len(store.list_episodes()) == 2
        assert store.interactions('destination')[-1]['payload']['action_count'] == 1


@pytest.mark.parametrize('rewind', [None, 2])
def test_prompt_write_uses_destination_tokenizer_and_is_rewindable(tmp_path, rewind):
    path = tmp_path / 'episodes.sqlite3'
    with EpisodeStore(path) as store:
        source = EpisodeEngine(NoEogBackend(), sampling=SamplingConfig(), initial_token_ids=[7])
        # Deliberately different source token ledger: importing IDs would give P.
        store.create_episode(episode_id='source', initial_text=' A B', initial_token_ids=[7],
            sampling=source.sampling, stream_fingerprint=source.stream_fingerprint,
            coordinate_offset=0, max_tokens=None, backend={})
    commands = ['t hello', 'q', 'spr #1']
    if rewind is not None:
        commands.append(f'rewind {rewind}')
    commands.append('quit')
    backend = NoEogBackend()
    with patch('trajectory_editor.episode_cli._backend', return_value=backend), patch(
        'trajectory_editor.episode_cli.TerminalIO', return_value=ScriptedIO(commands)
    ), patch.object(backend, 'tokenize', wraps=backend.tokenize) as tokenize:
        assert main(['--workspace', str(path), '--model', 'fake', '--plain-ui',
                     '--new-prompt', 'P', '--episode-id', 'destination']) == 0
    prompt_calls = [call for call in tokenize.call_args_list if call.args[0] == ' A B']
    assert len(prompt_calls) == 1
    assert prompt_calls[0].kwargs == {'add_bos': False, 'special': False}
    with EpisodeStore(path) as store:
        destination = store.get_episode('destination')
        assert destination['initial_token_ids'] == [7]
        assert destination['visible_text'] == (' hello A B' if rewind is None else ' hello A')
        prompt_action = store.actions('destination')[1]
        assert prompt_action['kind'] == 'write'
        assert prompt_action['arguments']['mode'] == 'exact'
        assert prompt_action['arguments']['replay_origin'] == {
            'episode_id': 'source', 'boundary': 0, 'part': 'prompt'}
        assert store.get_episode('source')['initial_token_ids'] == [7]
        assert store.tokens('source') == []


def test_prompt_write_respects_remaining_allowance_and_stops_tape(tmp_path):
    path = tmp_path / 'episodes.sqlite3'
    with EpisodeStore(path) as store:
        source = EpisodeEngine(NoEogBackend(), sampling=SamplingConfig(), initial_token_ids=[7])
        store.create_episode(episode_id='source', initial_text=' A B', initial_token_ids=[7],
            sampling=source.sampling, stream_fingerprint=source.stream_fingerprint,
            coordinate_offset=0, max_tokens=None, backend={})
        store.record_action('source', 0, source.apply(Write('hello')))
    start(path, ['t hello', 'q', 'spr #1', 'quit'], '--max-tokens', '2')
    with EpisodeStore(path) as store:
        assert store.get_episode('destination')['visible_text'] == ' hello'
        assert len(store.actions('destination')) == 1
        assert store.get_episode('destination')['checkpoint_boundary'] == 2
        rejected = [event for event in store.interactions('destination') if event['kind'] == 'instruction-rejected']
        assert rejected[-1]['payload']['action'] == Write(' A B', 'exact').to_dict()


def test_cli_prompt_only_replay_does_not_insert_prompt_as_action(tmp_path):
    path = tmp_path / 'episodes.sqlite3'
    with EpisodeStore(path) as store:
        source = EpisodeEngine(NoEogBackend(), sampling=SamplingConfig(), initial_token_ids=[7])
        store.create_episode(episode_id='source', initial_text='P', initial_token_ids=[7],
            sampling=source.sampling, stream_fingerprint=source.stream_fingerprint,
            coordinate_offset=0, max_tokens=None, backend={})
    run_cli(path, ['quit'], '--replay', '#1', '--episode-id', 'target')
    with EpisodeStore(path) as store:
        target = store.get_episode('target')
        assert target['initial_text'] == 'P'
        assert target['initial_token_ids'] == [7]
        assert target['visible_text'] == ''
        assert store.actions('target') == []

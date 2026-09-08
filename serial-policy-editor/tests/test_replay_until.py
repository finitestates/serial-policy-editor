import pytest

from tests.test_edge_replay_continuation import source_workspace, start, run_cli
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_projector import project_fork_map


@pytest.mark.parametrize('until,text', [(0, 'P'), (1, 'P A'), (2, 'P A B'), (3, 'P A B!'), (4, 'P A B!!')])
def test_edge_cutoff_uses_source_boundaries(source_workspace, until, text):
    start(source_workspace, ['q', f'spr #1 --until {until}', 'quit'])
    with EpisodeStore(source_workspace) as store:
        assert store.get_episode('destination')['visible_text'] == text
        assert store.get_episode('destination')['initial_text'] == 'P'
        assert len(store.tokens('destination')) == until + 1
        assert len(store.tokens('source')) == 4


@pytest.mark.parametrize('until,text', [(0, ''), (1, ' A'), (2, ' A B'), (3, ' A B!'), (4, ' A B!!')])
def test_cli_cutoff_returns_to_edge(source_workspace, until, text):
    run_cli(source_workspace, ['quit'], '--replay', '#1', '--until', str(until), '--episode-id', 'destination')
    with EpisodeStore(source_workspace) as store:
        assert store.get_episode('destination')['visible_text'] == text
        assert store.get_episode('destination')['initial_text'] == 'P'
        assert len(store.tokens('destination')) == until
        assert not any(row['mismatch'] for row in store.actions('destination'))


def test_map_selects_source_prefix(source_workspace):
    io = start(source_workspace, ['q', 'spr #1 m', '3', 'quit'])
    with EpisodeStore(source_workspace) as store:
        assert store.get_episode('destination')['visible_text'] == 'P A B!'


@pytest.mark.parametrize('commands', [ ['spr #1 m', '', 'quit'], ['spr #1 --until 5', 'quit'],
    ['spr #1 --until -1', 'quit'], ['spr #1 --until nope', 'quit'] ])
def test_invalid_or_cancelled_selection_does_not_insert(source_workspace, commands):
    start(source_workspace, ['q', *commands])
    with EpisodeStore(source_workspace) as store:
        assert store.tokens('destination') == []


def test_prompt_tokens_remain_individually_indexed(source_workspace):
    with EpisodeStore(source_workspace) as store:
        with store.transaction() as db:
            db.execute("UPDATE episodes SET initial_text = ' A B' WHERE episode_id = 'source'")
    start(source_workspace, ['q', 'spr #1 --until 0', 'quit'])
    with EpisodeStore(source_workspace) as store:
        assert project_fork_map(store, 'destination') == 'P|0| A|1| B|2|'
        assert len(store.actions('destination')) == 1


@pytest.mark.parametrize('until,seed', [(0, 999), (1, 999), (2, 123), (3, 123)])
def test_cutoff_does_not_import_future_sampler(source_workspace, until, seed):
    from dataclasses import replace
    from trajectory_editor.domain import SamplingConfig
    with EpisodeStore(source_workspace) as store:
        segment = store.sampling_segment('source', 0)
        for boundary, next_seed in [(2, 123), (4, 456)]:
            store.record_sampling_segment('source', start_boundary=boundary,
                sampling=replace(SamplingConfig.from_record(segment['sampling']), seed=next_seed),
                stream_fingerprint=segment['stream_fingerprint'], coordinate_offset=40)
    run_cli(source_workspace, ['quit'], '--replay', '#1', '--until', str(until), '--episode-id', 'destination')
    with EpisodeStore(source_workspace) as store:
        assert store.final_sampling('destination').seed == seed


def test_budget_can_stop_before_selected_boundary(source_workspace):
    start(source_workspace, ['q', 'spr #1 --until 4', 'quit'], '--max-tokens', '1')
    with EpisodeStore(source_workspace) as store:
        assert store.get_episode('destination')['visible_text'] == 'P'
        assert store.get_episode('destination')['checkpoint_boundary'] == 1


def test_invalid_map_choice_then_cancel_does_not_insert(source_workspace):
    start(source_workspace, ['q', 'spr #1 m', '99', '', 'quit'])
    with EpisodeStore(source_workspace) as store:
        assert store.tokens('destination') == []

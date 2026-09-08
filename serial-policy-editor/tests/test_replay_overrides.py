from dataclasses import replace
from unittest.mock import patch

import pytest

from tests.fakes import ScriptedIO
from tests.test_episode_runtime import NoEogBackend
from tests.test_replay_eog import create
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.episode_actions import Hold
from trajectory_editor.episode_cli import _sampler_summary, _spr_engine_from_source, main
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_policy import EpisodeRunner
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.sampling import draw_token


@pytest.fixture
def replay_source(tmp_path):
    path = tmp_path / 'episodes.sqlite3'
    initial = SamplingConfig(temperature=0, top_k=8, top_p=1, min_p=0, seed=11)
    middle = replace(initial, temperature=0.5, top_k=3, seed=22)
    final = replace(middle, temperature=0.7, top_k=4, seed=33)
    with EpisodeStore(path) as store:
        source = EpisodeEngine(NoEogBackend(), sampling=initial, initial_token_ids=[7])
        create(store, source, 'source')
        for ordinal, config in enumerate((initial, middle)):
            source.sampling = config
            store.record_sampling_segment('source', start_boundary=source.boundary,
                sampling=config, stream_fingerprint=source.stream_fingerprint, coordinate_offset=0)
            store.record_action('source', ordinal, source.apply(Hold(1)))
        store.record_sampling_segment('source', start_boundary=2, sampling=final,
            stream_fingerprint=source.stream_fingerprint, coordinate_offset=0)
        store.update_episode('source', visible_text=source.backend.render(source.visible_token_ids), max_tokens=None)
    return path, initial, middle, final


@pytest.mark.parametrize('flags,fixed,override,budget', [
    (['--seed', '77'], False, {'seed': 77}, None),
    (['--fixed-config'], True, {}, None),
    (['--seed', '77', '--fixed-config'], True, {'seed': 77}, None),
    (['--max-tokens', '100'], False, {}, 100),
    (['--max-tokens', '100', '--seed', '77', '--fixed-config'], True, {'seed': 77}, 100),
])
def test_cli_replay_configuration_and_edge_ownership(replay_source, flags, fixed, override, budget):
    path, initial, middle, final = replay_source
    expected_steps = [replace(initial, **override), replace(initial if fixed else middle, **override)]
    expected_edge = replace(initial if fixed else final, **override)
    io = ScriptedIO(['s seed=99 temperature=0.2 top_k=2', 'c', 'h 1', 'q', 'quit'])
    replay_settings = []
    seeds = []
    original_apply = EpisodeEngine.apply

    def observe_apply(runtime, action, **kwargs):
        if kwargs.get('replay'):
            replay_settings.append(runtime.sampling)
        return original_apply(runtime, action, **kwargs)

    def observe_draw(distribution, **kwargs):
        seeds.append(kwargs['seed'])
        return draw_token(distribution, **kwargs)

    with patch('trajectory_editor.episode_cli._backend', return_value=NoEogBackend()), patch(
        'trajectory_editor.episode_cli.TerminalIO', return_value=io
    ), patch.object(EpisodeEngine, 'apply', observe_apply), patch(
        'trajectory_editor.episode_engine.draw_token', side_effect=observe_draw
    ):
        assert main(['--workspace', str(path), '--model', 'fake', '--replay', 'source',
                     '--episode-id', 'target', '--plain-ui', '--divergence-policy', 'ballistic', *flags]) == 0
    assert replay_settings == expected_steps
    assert seeds[:2] == [config.seed for config in expected_steps]
    # Live editing also draws the next proposal before the user opens EDGE.
    assert seeds[2:] and all(seed == 99 for seed in seeds[2:])
    headers = [line for line in io.output if 'Live edge @ boundary' in line]
    assert _sampler_summary(expected_edge) in headers[0]
    teacher = replace(expected_edge, seed=99, temperature=0.2, top_k=2)
    assert _sampler_summary(teacher) in headers[-1]
    with EpisodeStore(path) as store:
        assert store.final_sampling('target') == teacher
        assert store.get_episode('target')['max_tokens'] == budget
        assert len(store.tokens('target')) == 3
        assert store.sampling_segment('target', 1)['sampling'] == expected_steps[1].to_dict()


@pytest.mark.parametrize('field,value', [
    ('seed', 77), ('temperature', 0.2), ('top_k', 2), ('top_p', 0.8),
    ('min_p', 0.1), ('repeat_penalty', 1.2), ('repeat_last_n', 5),
    ('presence_penalty', 0.3), ('frequency_penalty', 0.4),
])
def test_each_override_preserves_other_source_transitions(replay_source, field, value):
    path, initial, middle, final = replay_source
    with EpisodeStore(path) as store:
        runtime, plan = _spr_engine_from_source(store, 'source', NoEogBackend(),
            sampling=initial, max_tokens=None, sampling_overrides={field: value})
        assert runtime.sampling == replace(initial, **{field: value})
        assert [step.sampling for step in plan] == [replace(c, **{field: value}) for c in (initial, middle)]
        assert plan.final_sampling == replace(final, **{field: value})


def test_partial_override_early_exit_discards_future_transitions(replay_source):
    path, initial, _, _ = replay_source
    with EpisodeStore(path) as store:
        runtime, plan = _spr_engine_from_source(store, 'source', NoEogBackend(),
            sampling=initial, max_tokens=1, sampling_overrides={'seed': 77})
        create(store, runtime, 'target')
        runner = EpisodeRunner(runtime, store, 'target', divergence_policy='ballistic')
        result = runner.run(tape=plan)
        assert not result.replay_exhausted
        assert runtime.sampling == replace(initial, seed=77)
        teacher = replace(runtime.sampling, seed=99)
        runtime.resume(sampling=teacher)
        runner.run()
        assert runtime.sampling == teacher


def test_fixed_config_requires_cli_replay(tmp_path, capsys):
    with patch('trajectory_editor.episode_cli._backend') as load:
        assert main(['--workspace', str(tmp_path / 'unused.sqlite3'), '--new-prompt', 'P', '--fixed-config']) == 2
        load.assert_not_called()
    assert '--fixed-config requires --replay' in capsys.readouterr().err


def test_random_seed_is_a_per_field_override(replay_source):
    path, initial, middle, final = replay_source
    io = ScriptedIO(['quit'])
    with patch('trajectory_editor.episode_cli._random_seed', return_value=77), patch(
        'trajectory_editor.episode_cli._backend', return_value=NoEogBackend()
    ), patch('trajectory_editor.episode_cli.TerminalIO', return_value=io):
        assert main(['--workspace', str(path), '--model', 'fake', '--replay', 'source',
                     '--episode-id', 'target', '--random-seed', '--divergence-policy', 'ballistic', '--plain-ui']) == 0
    with EpisodeStore(path) as store:
        assert store.sampling_segment('target', 1)['sampling'] == replace(middle, seed=77).to_dict()
        assert store.final_sampling('target') == replace(final, seed=77)

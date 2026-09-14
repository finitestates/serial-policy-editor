"""Experimental controls, including the identity of persisted latent coordinates."""

import math
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from tests.fakes import ScriptedIO
from tests.test_latent_preference import FEATURES, LatentBackend, _observation
from trajectory_editor.bias_presets import load_bias_preset, project_biases
from trajectory_editor.domain import EditorError, SamplingConfig
from trajectory_editor.episode_actions import SelectRawRank, Write
from trajectory_editor.episode_cli import (
    _apply_latent_seed, _latent_config_from_args, _sampler_override,
    _sampling_from_args, build_parser, main,
)
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_lifecycle import (
    _create_episode, _fork_engine, _restore_engine, _rewind_episode,
    _spr_engine_from_source,
)
from trajectory_editor.episode_policy import EpisodeRunner, _WriteLearningAccumulator
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.latent_features import DEFAULT_PROJECTION_SEED, project_token_embeddings
from trajectory_editor.latent_preference import LatentPreferenceConfig, LatentPreferenceLearner


def learner(**kwargs):
    return LatentPreferenceLearner(FEATURES, enabled=True, dimension=2, **kwargs)


def test_default_update_matches_local_main_fixture():
    # Captured from b1e696f, before the experimental controls were added.
    sampling = SamplingConfig(latent_preference_z=(0.12, -0.08))
    result = learner().update(_observation(sampling), 3, sampling)
    assert result.old_policy_rank == 4
    assert result.old_policy_probability == pytest.approx(0.025440951760371408)
    assert result.severity == pytest.approx(0.20065763012322416, abs=1e-15)
    assert result.new_z == pytest.approx((0.10645405167970969, -0.08012752774251529), abs=1e-15)
    assert _observation(result.sampling).statistics.adjusted == pytest.approx([
        2.0, 1.1064540520310402, 0.5971249714493752, -1.1064540520310402,
        -2.080127529799938, -2.919872470200062, -3.978709189221263, -5.0,
    ], abs=1e-12)


@pytest.mark.parametrize('decay', [0.0, 0.25, 1.0])
def test_decay_and_learning_are_separate(decay):
    sampling = SamplingConfig(latent_preference_z=(0.3, -0.2))
    observation = _observation(sampling)
    base = learner().update(observation, 3, sampling)
    result = learner(decay=decay).update(observation, 3, sampling)
    expected = (1 - decay) * np.array(sampling.latent_preference_z) + base.learning_delta
    assert result.new_z == pytest.approx(expected)
    assert result.delta == pytest.approx(expected - sampling.latent_preference_z)
    assert result.decay_norm == pytest.approx(decay * np.linalg.norm(sampling.latent_preference_z))
    assert result.learning_step_norm == pytest.approx(base.learning_step_norm)


@pytest.mark.parametrize('cap', [1, 10, 1000])
def test_shifted_severity_and_dead_zone_gate_rejection(cap):
    sampling = SamplingConfig(latent_preference_z=(0.3, -0.2))
    observation = _observation(sampling, proposal_token_id=5)
    model = learner(decay=0.25, dead_zone_rank=3, severity_cap=cap, rejection_strength=10)
    inside = model.update(observation, 1, sampling)
    assert inside.severity == 0
    assert inside.learning_step_norm == 0
    assert inside.new_z == pytest.approx(np.array(sampling.latent_preference_z) * 0.75)
    outside = model.update(observation, 3, sampling)
    assert outside.old_policy_rank == 4
    assert outside.severity == pytest.approx(min(1, math.log1p(1) / math.log1p(cap)))


@pytest.mark.parametrize('strength', [0.0, 1.0, 3.0])
@pytest.mark.parametrize('proposal', [1, 4, 0])
def test_rejection_uses_captured_proposal_and_acceptance_is_expectation_based(strength, proposal):
    sampling = SamplingConfig()
    observation = _observation(sampling, proposal_token_id=proposal)
    result = learner(rejection_strength=strength, max_step=100).update(observation, 1, sampling)
    mean = np.asarray(result.policy_weighted_mean_features)
    direction = FEATURES[1] - mean
    if proposal != 1:
        direction += strength * (mean - FEATURES[proposal])
    assert result.learning_delta == pytest.approx(0.05 * result.severity * direction)
    if strength == 1 and proposal != 1:
        assert result.learning_delta == pytest.approx(0.05 * result.severity * (FEATURES[1] - FEATURES[proposal]))
    assert result.proposal_token_id == proposal
    assert result.proposal_rejected is (proposal != 1)
    assert result.to_dict()['rejection_strength'] == strength


def test_fast_slow_independent_dynamics_and_clipping():
    sampling = SamplingConfig()
    result = learner(fast_slow=True, learning_rate=0.1, fast_learning_rate=0.4,
                     max_step=10, fast_max_step=10).update(_observation(sampling), 3, sampling)
    assert result.new_fast_z == pytest.approx(4 * np.asarray(result.new_z))
    assert result.fast_strength == 0.5
    observation = _observation(result.sampling)
    decayed = learner(fast_slow=True, dead_zone_rank=8, decay=0.1, fast_decay=0.5).update(
        observation, 3, result.sampling)
    assert decayed.new_z == pytest.approx(0.9 * np.asarray(result.new_z))
    assert decayed.new_fast_z == pytest.approx(0.5 * np.asarray(result.new_fast_z))
    bounded = learner(fast_slow=True, learning_rate=100, fast_learning_rate=100,
                      rejection_strength=20, max_step=0.15, max_norm=0.1,
                      fast_max_step=0.2, fast_max_norm=0.12).update(observation, 3, result.sampling)
    assert bounded.learning_step_norm <= 0.15 + 1e-12
    assert bounded.fast_learning_step_norm <= 0.2 + 1e-12
    assert bounded.z_norm <= 0.1 + 1e-12
    assert bounded.fast_z_norm <= 0.12 + 1e-12


def test_disabled_neither_learns_nor_decays_and_fast_off_preserves_saved_fast():
    sampling = SamplingConfig(latent_preference_z=(0.3, -0.2),
                              latent_preference_fast_z=(0.4, 0.2), latent_fast_strength=0.7)
    disabled = LatentPreferenceLearner(FEATURES, dimension=2, decay=1, fast_slow=True, fast_decay=1)
    result = disabled.update(_observation(sampling), 3, sampling)
    assert result.sampling == sampling
    assert result.update_norm == result.fast_update_norm == result.decay_norm == 0
    result = learner().update(_observation(sampling), 3, sampling)
    assert result.sampling.latent_preference_fast_z == sampling.latent_preference_fast_z
    assert result.sampling.latent_fast_strength == sampling.latent_fast_strength


@pytest.mark.parametrize('kwargs', [
    {'decay': -0.01}, {'decay': 1.01}, {'decay': float('nan')},
    {'severity_cap': 0}, {'severity_cap': 1.5}, {'dead_zone_rank': 0},
    {'rejection_strength': -1}, {'rejection_strength': float('inf')},
    {'fast_decay': 1.01}, {'fast_learning_rate': -1}, {'fast_max_step': -1},
    {'fast_max_norm': float('nan')}, {'fast_strength': -1},
    {'projection_seed': -(1 << 63) - 1},
])
def test_invalid_learner_controls(kwargs):
    with pytest.raises(EditorError):
        LatentPreferenceConfig(**kwargs)


@pytest.mark.parametrize('kwargs', [
    {'latent_preference_z': (1, 2), 'latent_preference_fast_z': (1,)},
    {'latent_preference_fast_z': (float('inf'),)},
    {'latent_preference_fast_z': '12'}, {'latent_fast_strength': -1},
    {'latent_projection_seed': 1 << 63}, {'latent_projection_seed': True},
])
def test_invalid_saved_state(kwargs):
    with pytest.raises(EditorError):
        SamplingConfig(**kwargs)


def test_fast_only_sampling_and_state_copies():
    sampling = SamplingConfig(latent_preference_fast_z=(0.5, -0.3), latent_fast_strength=0.8,
                              latent_projection_seed=-17)
    observation = _observation(sampling)
    assert observation.statistics.latent_logit_adjustments == pytest.approx(0.8 * (FEATURES @ np.array([0.5, -0.3])))
    assert sampling.policy_active
    assert SamplingConfig.from_record(sampling.to_dict()) == sampling
    assert _sampling_from_args(build_parser().parse_args([]), sampling) == sampling
    assert _sampler_override(sampling, 'seed=42') == replace(sampling, seed=42)
    assert learner().update(observation, 3, sampling).new_z


def test_seed_projection_signed_determinism_and_old_record_migration():
    embeddings = np.arange(40, dtype=np.float32).reshape(8, 5)
    negative = project_token_embeddings(embeddings, feature_dimension=3, projection_seed=-17)
    same = project_token_embeddings(embeddings, feature_dimension=3, projection_seed=-17)
    positive = project_token_embeddings(embeddings, feature_dimension=3, projection_seed=17)
    assert np.array_equal(negative, same)
    assert not np.array_equal(negative, positive)
    old = SamplingConfig(latent_preference_z=(0.1, 0.2)).to_dict()
    for key in ('latent_projection_seed', 'latent_preference_fast_z', 'latent_fast_strength'):
        old.pop(key)
    assert SamplingConfig.from_record(old).latent_projection_seed == DEFAULT_PROJECTION_SEED


def test_preset_round_trip_includes_both_memories_and_seed(tmp_path):
    sampling = SamplingConfig(latent_preference_z=(0.5, -0.1), latent_strength=0.8,
                              latent_preference_fast_z=(0.2, 0.1), latent_fast_strength=0.4,
                              latent_projection_seed=-17)
    backend = LatentBackend()
    engine = EpisodeEngine(backend, initial_token_ids=[7], sampling=sampling)
    with EpisodeStore(tmp_path / 'episodes.sqlite3') as store:
        episode = _create_episode(store, engine, backend_provenance=backend.provenance())
        path = tmp_path / 'preset.json'
        path.write_text(project_biases(store, episode))
        assert load_bias_preset(path, backend, backend.provenance()) == sampling


def test_cli_profile_resolves_all_controls_and_seed_exclusion():
    parser = build_parser()
    config = _latent_config_from_args(parser.parse_args(['--latent-fast-slow', '--latent-learning-rate', '.03',
                                                         '--latent-strength', '.8', '--latent-max-norm', '.7']))
    assert config.fast_learning_rate == 0.12
    assert config.fast_strength == 0.4
    assert config.fast_decay == 0.1
    assert config.fast_max_step == config.max_step
    assert config.fast_max_norm == 0.7
    args = parser.parse_args('''--latent-preference --latent-dimension 64 --latent-learning-rate .03
        --latent-strength .8 --latent-max-step .15 --latent-max-norm 3 --latent-decay .002
        --latent-severity-cap 300 --latent-dead-zone-rank 2 --latent-rejection-strength 1
        --latent-fast-slow --latent-fast-learning-rate .15 --latent-fast-decay .12
        --latent-fast-strength .4 --latent-fast-max-step .20 --latent-fast-max-norm .8
        --latent-random-seed'''.split())
    config = _latent_config_from_args(args)
    assert (config.fast_learning_rate, config.fast_decay, config.fast_max_norm) == (.15, .12, .8)
    with pytest.raises(SystemExit):
        parser.parse_args(['--latent-seed', '1', '--latent-random-seed'])


def test_seed_override_clears_both_coordinates_and_replay_rejects_override():
    sampling = SamplingConfig(latent_preference_z=(0.2, 0.3), latent_preference_fast_z=(0.1, 0.2),
                              latent_projection_seed=-17)
    io = ScriptedIO([])
    assert _apply_latent_seed(sampling, -17, io) is sampling
    changed = _apply_latent_seed(sampling, 18, io)
    assert changed.latent_projection_seed == 18
    assert changed.latent_preference_z == changed.latent_preference_fast_z == ()
    assert any('memory reset' in text for text in io.output)
    with pytest.raises(EditorError, match='replay seed'):
        _apply_latent_seed(sampling, 18, io, replay=True)


def test_write_sums_evidence_before_clipping_and_decays_once():
    sampling = SamplingConfig(latent_preference_z=(0.5, 0.2), latent_preference_fast_z=(0.3, 0.1))
    model = learner(decay=0.2, fast_slow=True, fast_decay=0.5,
                    learning_rate=1, fast_learning_rate=2, max_step=0.03, fast_max_step=0.04)
    observations = [_observation(sampling, proposal_token_id=4), _observation(sampling, proposal_token_id=5)]
    individual = [model.update(o, token, sampling) for o, token in zip(observations, [3, 1])]
    accumulator = _WriteLearningAccumulator(LatentBackend(), sampling, None, model)
    for o, token in zip(observations, [3, 1]):
        accumulator.add(o, token)
    result = accumulator.finish(2).latent_result
    for prefix, decay, bound, old in [('', .2, .03, sampling.latent_preference_z),
                                      ('fast_', .5, .04, sampling.latent_preference_fast_z)]:
        evidence = np.sum([getattr(r, prefix + 'learning_evidence') for r in individual], axis=0)
        step = evidence * min(1, bound / np.linalg.norm(evidence))
        actual = result.new_z if not prefix else result.new_fast_z
        assert actual == pytest.approx((1 - decay) * np.array(old) + step)
        assert getattr(result, prefix + 'learning_step_norm') <= bound + 1e-12
    assert len(accumulator.finish(2).to_dict()['latent_token_observations']) == 2


def test_write_dead_zone_only_forgets_once():
    sampling = SamplingConfig(latent_preference_z=(0.5, 0.2), latent_preference_fast_z=(0.3, 0.1))
    model = learner(decay=.2, dead_zone_rank=8, fast_slow=True, fast_decay=.5, rejection_strength=3)
    accumulator = _WriteLearningAccumulator(LatentBackend(), sampling, None, model)
    for token in (1, 3, 5):
        accumulator.add(_observation(sampling), token)
    result = accumulator.finish(3).latent_result
    assert result.new_z == pytest.approx(np.array(sampling.latent_preference_z) * .8)
    assert result.new_fast_z == pytest.approx(np.array(sampling.latent_preference_fast_z) * .5)


class SeedBackend(LatentBackend):
    def __init__(self):
        super().__init__()
        self.projections = []

    def latent_token_features(self, *, feature_dimension, projection_seed, **kwargs):
        self.projections.append((feature_dimension, projection_seed))
        return project_token_embeddings(np.arange(40, dtype=np.float32).reshape(8, 5),
                                        feature_dimension=feature_dimension, projection_seed=projection_seed)


@pytest.fixture
def saved_latent_history(tmp_path):
    path = tmp_path / 'history.sqlite3'
    initial = SamplingConfig(latent_preference_z=(.2, -.1), latent_strength=.8,
                             latent_preference_fast_z=(.1, .05), latent_fast_strength=.4,
                             latent_projection_seed=-17)
    middle = replace(initial, latent_preference_z=(), latent_preference_fast_z=(.3, -.2),
                     latent_projection_seed=29)
    final = replace(initial, latent_preference_z=(.4, .1), latent_preference_fast_z=(.1, -.2),
                    latent_projection_seed=-31)
    backend = SeedBackend()
    engine = EpisodeEngine(backend, initial_token_ids=[7], sampling=initial)
    with EpisodeStore(path) as store:
        _create_episode(store, engine, backend_provenance=backend.provenance(), requested_id='source')
        for index, state in enumerate((initial, middle)):
            engine.sampling = state
            store.record_sampling_segment('source', start_boundary=index, sampling=state,
                                          stream_fingerprint=engine.stream_fingerprint, coordinate_offset=0)
            store.record_action('source', index, engine.apply(Write(' A', 'exact')))
        store.record_sampling_segment('source', start_boundary=2, sampling=final,
                                      stream_fingerprint=engine.stream_fingerprint, coordinate_offset=0)
    return path, initial, middle, final


@pytest.mark.parametrize('boundary', [0, 1, 2])
def test_restore_rewind_fork_use_historical_coordinates(saved_latent_history, boundary):
    path, *states = saved_latent_history
    backend = SeedBackend()
    with EpisodeStore(path) as store:
        restored = _restore_engine(store, 'source', backend, max_tokens=None, sampling_override=None)
        assert restored.sampling == states[-1]
        restored.observe()
        assert backend.projections[-1] == (2, -31)
        fork_backend = SeedBackend()
        fork = _fork_engine(store, 'source', restored, boundary, backend=fork_backend, max_tokens=None)
        assert fork.sampling == states[boundary]
        _rewind_episode(store, 'source', restored, boundary)
        assert restored.sampling == states[boundary]
        assert restored.observe().statistics.adjusted == pytest.approx(fork.observe().statistics.adjusted)
        assert fork_backend.projections[-1] == (2, states[boundary].latent_projection_seed)
        assert backend.projections[-1] == (2, states[boundary].latent_projection_seed)


def test_replay_imports_each_saved_seed_and_never_learns(saved_latent_history):
    path, initial, middle, final = saved_latent_history
    backend = SeedBackend()
    with EpisodeStore(path) as store:
        engine, plan = _spr_engine_from_source(store, 'source', backend, sampling=initial, max_tokens=None)
        assert [step.sampling for step in plan] == [initial, middle]
        assert plan.final_sampling == final
        _create_episode(store, engine, backend_provenance=backend.provenance(), requested_id='replay')
        model = learner(fast_slow=True, decay=1, fast_decay=1)
        with patch.object(model, 'update', side_effect=AssertionError('replay must not learn')):
            EpisodeRunner(engine, store, 'replay', latent_learner=model, learn_from_write=True).run(tape=plan)
        assert engine.sampling == final
        assert store.final_sampling('replay') == final
        assert backend.projections == [(2, -17), (2, 29)]
        assert not any(i['kind'] in ('latent-preference-update', 'write-learning-update')
                       for i in store.interactions('replay'))


def test_live_learner_uses_current_saved_seed_even_after_empty_state_restore():
    backend = SeedBackend()
    model = LatentPreferenceLearner(feature_provider=backend.latent_token_features, enabled=True, dimension=2)
    for seed in (-17, 29):
        sampling = SamplingConfig(latent_projection_seed=seed)
        model.update(_observation(sampling), 3, sampling)
        assert backend.projections[-1] == (2, seed)


def test_random_latent_seed_printed_once_persisted_and_replay_does_not_draw(tmp_path, capsys):
    path = tmp_path / 'random.sqlite3'
    with patch('trajectory_editor.episode_cli._random_seed', return_value=-123456) as randomize, patch(
        'trajectory_editor.episode_cli._backend', return_value=SeedBackend()
    ), patch('trajectory_editor.episode_cli.TerminalIO', return_value=ScriptedIO(['h 3'])):
        assert main(['--workspace', str(path), '--model', 'fake', '--new-prompt', 'P',
                     '--episode-id', 'random', '--latent-random-seed', '--plain-ui']) == 0
        randomize.assert_called_once_with()
    assert capsys.readouterr().out.count('Random latent seed: -123456') == 1
    with EpisodeStore(path) as store:
        assert store.final_sampling('random').latent_projection_seed == -123456
    with patch('trajectory_editor.episode_cli._random_seed', side_effect=AssertionError('no reroll')), patch(
        'trajectory_editor.episode_cli._backend', return_value=SeedBackend()
    ), patch('trajectory_editor.episode_cli.TerminalIO', return_value=ScriptedIO(['quit'])):
        assert main(['--workspace', str(path), '--model', 'fake', '--replay', 'random',
                     '--episode-id', 'replayed', '--plain-ui']) == 0
    with EpisodeStore(path) as store:
        assert store.final_sampling('replayed').latent_projection_seed == -123456


@pytest.mark.parametrize('mode', ['--resume', '--fork-from'])
@pytest.mark.parametrize('override', [None, -31, 42])
def test_cli_resume_and_fork_preserve_or_reset_latent_identity(saved_latent_history, mode, override):
    path, _, _, final = saved_latent_history
    io = ScriptedIO(['q', 'quit'])
    flags = [] if override is None else ['--latent-seed', str(override)]
    if mode == '--fork-from':
        flags += ['--episode-id', 'fork']
    with patch('trajectory_editor.episode_cli._backend', return_value=SeedBackend()), patch(
        'trajectory_editor.episode_cli.TerminalIO', return_value=io
    ):
        assert main(['--workspace', str(path), '--model', 'fake', mode, 'source', '--plain-ui', *flags]) == 0
    with EpisodeStore(path) as store:
        saved = store.final_sampling('source' if mode == '--resume' else 'fork')
    if override == 42:
        assert saved.latent_projection_seed == 42
        assert saved.latent_preference_z == saved.latent_preference_fast_z == ()
        assert any('memory reset' in message for message in io.output)
    else:
        assert saved == final


@pytest.mark.parametrize('flags', [['--latent-seed', '42'], ['--latent-seed', '-17'], ['--latent-random-seed']])
def test_cli_replay_rejects_incompatible_seed_at_any_segment(saved_latent_history, flags, capsys):
    path, *_ = saved_latent_history
    with patch('trajectory_editor.episode_cli._backend', return_value=SeedBackend()), patch(
        'trajectory_editor.episode_cli._random_seed', side_effect=AssertionError('no reroll')):
        assert main(['--workspace', str(path), '--model', 'fake', '--replay', 'source',
                     '--episode-id', 'rejected', '--plain-ui', *flags]) == 2
    assert 'seed' in capsys.readouterr().err
    with EpisodeStore(path) as store:
        assert 'rejected' not in [episode['episode_id'] for episode in store.list_episodes()]


def test_live_eog_choice_does_not_learn_or_decay(tmp_path):
    backend = SeedBackend()
    sampling = SamplingConfig(latent_preference_z=(.3, .2), latent_preference_fast_z=(.2, .1))
    runtime = EpisodeEngine(backend, initial_token_ids=[7], sampling=sampling)
    with EpisodeStore(tmp_path / 'eog.sqlite3') as store:
        episode = _create_episode(store, runtime, backend_provenance=backend.provenance())
        policy = type('ChooseEog', (), {'choose': lambda self, engine, obs: SelectRawRank(obs.statistics.raw_rank(0))})()
        EpisodeRunner(runtime, store, episode, latent_learner=learner(decay=1, fast_slow=True, fast_decay=1)).run(
            live_policy=policy, max_live_actions=1)
        assert runtime.sampling == sampling
        assert not store.interactions(episode)

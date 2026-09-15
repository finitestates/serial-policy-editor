"""Teacher evidence gated by actual decoder eligibility, independently of rank."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from tests.fakes import ScriptedIO
from tests.test_latent_preference import FEATURES, LatentBackend
from trajectory_editor.bias_rules import BiasGroup, BiasRule
from trajectory_editor.domain import EditorError, SamplingConfig
from trajectory_editor.episode_cli import (
    _latent_config_from_args, _latent_preference_notice, _online_learning_notice,
    _write_learning_notice, build_parser, main,
)
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_actions import Write
from trajectory_editor.episode_policy import _WriteLearningAccumulator
from trajectory_editor.latent_preference import LatentPreferenceConfig, LatentPreferenceLearner
from trajectory_editor.online_learning import OnlineLearningConfig, OnlineLearner


class GateBackend(LatentBackend):
    def last_logits(self):
        return np.asarray([-20., 5., 4., 3., -1., -2., -3., -4.])


def engine(**settings):
    values = dict(temperature=1., top_k=8, top_p=1., min_p=0.,
                  bias_groups=(BiasGroup('target', (BiasRule(routes=((3,),), bias=0.),)),))
    values.update(settings)
    return EpisodeEngine(GateBackend(), initial_token_ids=[7], sampling=SamplingConfig(**values))


def learners(**settings):
    values = dict(enabled=True, learning_gate='sampler', learning_rate=.1)
    values.update(settings)
    return (OnlineLearner(**values), LatentPreferenceLearner(FEATURES, dimension=2, **values))


@pytest.mark.parametrize('settings,eligible', [
    ({}, True),
    ({'top_k': 2}, False),
    ({'top_p': .9}, False),
    ({'min_p': .2}, False),
    ({'min_p': .2, 'temperature': 2.}, True),
    ({'min_p': .2, 'temperature': .5}, False),
    ({'temperature': 0.}, False),
])
def test_gate_uses_actual_filters_and_replaces_rank_severity(settings, eligible):
    runtime = engine(**settings)
    observation = runtime.observe()
    assert observation.statistics.policy_rank(3) == 3
    # Even a rank dead zone covering the entire vocabulary must not override
    # sampler mode; excluded choices get severity 1 without an attenuation flag.
    for learner in learners(dead_zone_rank=8, severity_cap=100000, rejection_strength=1.):
        result = learner.update(observation, 3, runtime.sampling)
        assert result.sampler_eligible is eligible
        assert result.severity == float(not eligible)
        assert (result.update_norm == 0) is eligible
        assert result.to_dict()['learning_gate'] == 'sampler'
        assert result.to_dict()['sampler_probability'] == observation.statistics.distribution.probability(3)
        assert result.old_policy_probability > 0  # Learning sees untruncated scores.


def test_gate_uses_steered_policy_and_rechecks_each_observation():
    runtime = engine(top_k=2)
    for learner in learners():
        assert learner.update(runtime.observe(), 3, runtime.sampling).severity == 1
    runtime.sampling = replace(runtime.sampling, bias_groups=(replace(runtime.sampling.bias_groups[0], bias=3.),))
    for learner in learners():
        assert learner.update(runtime.observe(), 3, runtime.sampling).severity == 0


def test_membership_does_not_mistake_probability_underflow_for_filtering():
    runtime = engine()
    runtime.backend.last_logits = lambda: np.asarray([-20., 5., 4., -1000., -1., -2., -3., -4.])
    observation = runtime.observe()
    assert observation.statistics.distribution.probability(3) == 0
    assert 3 in observation.statistics.distribution.ids
    for learner in learners():
        result = learner.update(observation, 3, runtime.sampling)
        assert result.sampler_eligible and result.severity == 0


def test_sampler_gate_leaves_decay_and_fast_slow_limits_independent():
    runtime = engine(latent_preference_z=(.2, .1), latent_preference_fast_z=(.1, .2), latent_fast_strength=.5)
    model = LatentPreferenceLearner(FEATURES, enabled=True, learning_gate='sampler',
                                    fast_slow=True, decay=.2, fast_decay=.5)
    result = model.update(runtime.observe(), 3, runtime.sampling)
    assert result.learning_step_norm == result.fast_learning_step_norm == 0
    assert result.new_z == pytest.approx((.16, .08))
    assert result.new_fast_z == pytest.approx((.05, .1))
    runtime.sampling = replace(runtime.sampling, top_k=1)
    model = LatentPreferenceLearner(FEATURES, enabled=True, learning_gate='sampler', fast_slow=True,
                                    learning_rate=100, fast_learning_rate=100,
                                    max_step=.03, fast_max_step=.02, max_norm=.1, fast_max_norm=.1)
    result = model.update(runtime.observe(), 3, runtime.sampling)
    assert result.severity == 1
    assert result.learning_step_norm <= .03 + 1e-12 and result.fast_learning_step_norm <= .02 + 1e-12
    assert result.z_norm <= .1 + 1e-12 and result.fast_z_norm <= .1 + 1e-12


@pytest.mark.parametrize('tokens', ([1, 3, 2], [1, 2]))
def test_write_gates_each_token_and_aggregates_decay_once(tokens):
    runtime = engine(top_k=2, latent_preference_z=(.2, .1),
                     latent_preference_fast_z=(.1, .2), latent_fast_strength=.5)
    runtime.sampling = replace(runtime.sampling,
        bias_groups=(replace(runtime.sampling.bias_groups[0], bias=.2),))
    group_model = OnlineLearner(enabled=True, learning_gate='sampler', decay=.2, learning_rate=.1)
    latent_model = LatentPreferenceLearner(FEATURES, enabled=True, learning_gate='sampler',
                                           fast_slow=True, decay=.2, fast_decay=.5)
    accumulator = _WriteLearningAccumulator(runtime.backend, runtime.sampling, group_model, latent_model)
    for token in tokens:
        accumulator.add(runtime.observe(), token)
    result = accumulator.finish(len(tokens))
    assert [t.sampler_eligible for t in result.tokens] == [t != 3 for t in tokens]
    for r in accumulator.latent_results:
        assert r.severity == float(r.chosen_token_id == 3)
    raw_group = sum(r.evidence['target'] for r in accumulator.group_results)
    assert result.group_result.new_group_weights['target'] == pytest.approx(.8 * .2 + raw_group)
    raw_slow = np.sum([r.learning_evidence for r in accumulator.latent_results], axis=0)
    raw_fast = np.sum([r.fast_learning_evidence for r in accumulator.latent_results], axis=0)
    fast_norm = np.linalg.norm(raw_fast)
    if fast_norm > latent_model.config.fast_max_step:
        raw_fast *= latent_model.config.fast_max_step / fast_norm
    assert result.latent_result.new_z == pytest.approx(.8 * np.array((.2, .1)) + raw_slow)
    assert result.latent_result.new_fast_z == pytest.approx(.5 * np.array((.1, .2)) + raw_fast)
    payload = result.to_dict()
    assert payload['group_update']['sampler_eligible'] is None  # No misleading first-token summary.
    assert payload['latent_update']['sampler_probability'] is None
    assert [t['sampler_eligible'] for t in payload['tokens']] == [t != 3 for t in tokens]
    io = ScriptedIO([])
    _write_learning_notice(io, result)
    assert f'{tokens.count(3)}/{len(tokens)} tokens excluded' in io.output[-1]


def test_real_write_gate_observes_preceding_written_tokens():
    class ContextBackend(GateBackend):
        def last_logits(self):
            logits = super().last_logits()
            if self.tokens[-1] == 1:
                logits[2] = 6.  # B becomes eligible after teacher writes A.
            return logits

    runtime = EpisodeEngine(ContextBackend(), initial_token_ids=[7], sampling=SamplingConfig(
        temperature=1, top_k=1, top_p=1, min_p=0))
    assert 2 not in runtime.observe().statistics.distribution.ids
    accumulator = _WriteLearningAccumulator(runtime.backend, runtime.sampling, None,
        LatentPreferenceLearner(FEATURES, dimension=2, enabled=True, learning_gate='sampler'))
    runtime.apply(Write(' A B', 'exact'), on_precommit_observation=accumulator.add)
    result = accumulator.finish(runtime.boundary)
    assert [t.token_id for t in result.tokens] == [1, 2]
    assert all(t.sampler_eligible for t in result.tokens)
    assert result.latent_result.learning_step_norm == 0


def test_default_rank_behavior_and_disabled_learners_are_preserved():
    runtime = engine()
    for learner in (OnlineLearner(enabled=True), LatentPreferenceLearner(FEATURES, enabled=True, dimension=2)):
        result = learner.update(runtime.observe(), 3, runtime.sampling)
        assert result.learning_gate == 'rank' and result.sampler_eligible
        assert result.severity > 0 and result.update_norm > 0
    for learner in learners(enabled=False, decay=1):
        assert learner.update(runtime.observe(), 3, runtime.sampling).sampling == runtime.sampling


def test_cli_flags_validate_and_reach_both_learners(tmp_path):
    parser = build_parser()
    defaults = parser.parse_args([])
    assert defaults.learning_gate == defaults.latent_learning_gate == 'rank'
    args = parser.parse_args(['--learning-gate', 'sampler', '--latent-learning-gate', 'sampler'])
    assert _latent_config_from_args(args).learning_gate == 'sampler'
    for factory in (OnlineLearningConfig, LatentPreferenceConfig):
        with pytest.raises(EditorError, match='gate'):
            factory(learning_gate='unknown')
    io = ScriptedIO(['q', 'quit'])
    backend = GateBackend()
    with patch('trajectory_editor.episode_cli.TerminalIO', return_value=io), \
         patch('trajectory_editor.episode_cli._backend', return_value=backend), \
         patch('trajectory_editor.episode_cli.OnlineLearner', wraps=OnlineLearner) as group_spy, \
         patch('trajectory_editor.episode_cli.LatentPreferenceLearner', wraps=LatentPreferenceLearner) as latent_spy:
        assert main(['--model', str(tmp_path / 'model.gguf'), '--workspace', str(tmp_path / 'run.db'),
                     '--new-prompt', 'P', '--online-learning', '--latent-preference', '--latent-dimension', '2',
                     '--learning-gate', 'sampler', '--latent-learning-gate', 'sampler']) == 0
    assert group_spy.call_args.kwargs['learning_gate'] == 'sampler'
    assert latent_spy.call_args.kwargs['config'].learning_gate == 'sampler'


def test_single_update_notices_explain_gate():
    for settings, message in (({}, 'already eligible'), ({'top_k': 2}, 'excluded')):
        runtime = engine(**settings)
        io = ScriptedIO([])
        group_model, latent_model = learners()
        _online_learning_notice(io, group_model.update(runtime.observe(), 3, runtime.sampling))
        _latent_preference_notice(io, latent_model.update(runtime.observe(), 3, runtime.sampling))
        assert all(message in line for line in io.output)

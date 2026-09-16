"""Behavioral checks for opt-in decay, write sizing, and rejection experiments."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from tests.fakes import ScriptedIO
from tests.test_token_preference import FEATURES
from tests.test_sampler_learning_gate import GateBackend, engine, learners
from trajectory_editor.bias_rules import BiasGroup, BiasRule
from trajectory_editor.domain import EditorError
from trajectory_editor.episode_cli import build_parser, main, _write_learning_notice
from trajectory_editor.episode_policy import _WriteLearningAccumulator
from trajectory_editor.token_preference import TokenPreferenceConfig, TokenPreferenceLearner
from trajectory_editor.online_learning import OnlineLearningConfig, OnlineLearner
from trajectory_editor.learning_readout import show_learning_details


def memory_engine():
    runtime = engine(top_k=2, token_preference_vector=(.2, .1),
                     token_preference_fast_vector=(.1, .2), token_preference_fast_strength=.5)
    runtime.sampling = replace(runtime.sampling,
        bias_groups=(replace(runtime.sampling.bias_groups[0], bias=.2),))
    return runtime


def memory_learners(mode, **kwargs):
    settings = dict(enabled=True, learning_gate='sampler', decay=.2,
                    decay_on=mode, learning_rate=.1, max_step=10)
    settings.update(kwargs)
    return (OnlineLearner(**settings), TokenPreferenceLearner(FEATURES, dimension=2,
        fast_slow=True, fast_decay=.5, fast_max_step=10, fast_max_norm=10, **settings))


def assert_memory(result, applied):
    assert result.effective_decay == (.2 if applied else 0)
    if hasattr(result, 'new_z'):
        assert result.effective_fast_decay == (.5 if applied else 0)
        assert result.new_z == pytest.approx(np.array((.2, .1)) * (1-result.effective_decay)
                                             + result.learning_delta)
        assert result.new_fast_z == pytest.approx(np.array((.1, .2)) * (1-result.effective_fast_decay)
                                                  + result.fast_learning_delta)
    else:
        assert result.new_group_weights['target'] == pytest.approx(
            .2 * (1-result.effective_decay) + result.evidence['target'])


@pytest.mark.parametrize('mode', ['update', 'rejection', 'evidence'])
@pytest.mark.parametrize('chosen', [1, 2, 3])
def test_conditional_decay_distinguishes_accept_rejection_and_admitted_evidence(mode, chosen):
    runtime = memory_engine()
    observation = replace(runtime.observe(), proposal_token_id=1)
    for learner in memory_learners(mode):
        result = learner.update(observation, chosen, runtime.sampling)
        assert result.severity == (chosen == 3)
        assert_memory(result, mode == 'update' or chosen == 3 or (mode == 'rejection' and chosen == 2))


@pytest.mark.parametrize('mode', ['update', 'rejection', 'evidence'])
@pytest.mark.parametrize('tokens', [[1, 1], [1, 2], [1, 3], [3, 1]])
def test_write_decay_triggers_on_any_token_but_only_once(mode, tokens):
    runtime = memory_engine()
    group, preference = memory_learners(mode)
    acc = _WriteLearningAccumulator(runtime.backend, runtime.sampling, group, preference)
    observation = replace(runtime.observe(), proposal_token_id=1)
    for token in tokens:
        acc.add(observation, token)
    result = acc.finish(len(tokens))
    applies = mode == 'update' or 3 in tokens or (mode == 'rejection' and 2 in tokens)
    for r in (result.group_result, result.token_preference_result):
        assert_memory(r, applies)
        assert r.proposal_rejected == any(t != 1 for t in tokens)
        assert r.to_dict()['decay_on'] == mode
        assert r.to_dict()['effective_decay'] == (.2 if applies else 0)
    payload = result.to_dict()
    assert all(t['decay_on'] == mode for t in payload['token_preference_token_observations'])


@pytest.mark.parametrize('reduction,factor', [('sum', 2), ('mean', 1), ('sqrt', np.sqrt(2))])
def test_write_reduction_counts_only_admitted_evidence_and_scales_both_channels(reduction, factor):
    runtime = memory_engine()
    group, preference = memory_learners('evidence', write_reduction=reduction)
    observation = replace(runtime.observe(), proposal_token_id=1)
    acc = _WriteLearningAccumulator(runtime.backend, runtime.sampling, group, preference)
    for token in [3, 1, 3, 2]:
        acc.add(observation, token)
    result = acc.finish(4)
    single_group = group.update(observation, 3, runtime.sampling)
    single_preference = preference.update(observation, 3, runtime.sampling)
    assert result.group_result.evidence['target'] == pytest.approx(factor * single_group.evidence['target'])
    assert result.token_preference_result.learning_evidence == pytest.approx(factor * np.array(single_preference.learning_evidence))
    assert result.token_preference_result.fast_learning_evidence == pytest.approx(factor * np.array(single_preference.fast_learning_evidence))
    for r in (result.group_result, result.token_preference_result):
        assert r.write_evidence_tokens == 2
        assert r.write_evidence_scale == pytest.approx(factor / 2)
        assert_memory(r, True)
    io = ScriptedIO([])
    _write_learning_notice(io, result)
    if reduction != 'sum':
        show_learning_details(io)
        assert reduction in io.output[-1] and '2 evidence tokens' in io.output[-1]


@pytest.mark.parametrize('reduction', ['mean', 'sqrt'])
def test_all_skipped_write_has_no_division_by_zero(reduction):
    runtime = memory_engine()
    for learner in memory_learners('evidence', write_reduction=reduction):
        observation = replace(runtime.observe(), proposal_token_id=1)
        results = [learner.update(observation, 1, runtime.sampling)] * 3
        result = learner.aggregate(results, runtime.sampling)
        assert result.write_evidence_tokens == 0 and result.write_evidence_scale == 1
        assert_memory(result, False)


def test_evidence_decay_depends_on_gate_even_when_learning_rate_is_zero():
    runtime = memory_engine()
    for learner in memory_learners('evidence', learning_rate=0):
        result = learner.update(replace(runtime.observe(), proposal_token_id=1), 3, runtime.sampling)
        assert_memory(result, True)


@pytest.mark.parametrize('rho', [0., .5, 1.])
def test_sampler_rejection_uses_actual_probability_weighted_features(rho):
    # A group spanning both a likely proposal and the excluded teacher choice
    # exposes the effect of which proposal happened to be sampled.
    runtime = engine(top_k=2, temperature=.7, bias_groups=(BiasGroup('target',
        (BiasRule(routes=((1,), (3,)), bias=0.),)),))
    obs = runtime.observe()
    distribution = obs.statistics.distribution
    mean = obs.statistics.policy_probabilities @ FEATURES
    target = distribution.probabilities @ FEATURES[distribution.ids]
    expected = .1 * (FEATURES[3] - mean + rho * (mean - target))
    group_mean = sum(obs.statistics.policy_probabilities[[1, 3]])
    group_target = distribution.probability(1)
    for proposal in [1, 2]:
        observation = replace(obs, proposal_token_id=proposal)
        group, preference = learners(rejection_strength=rho, rejection_target='sampler', max_step=10)
        assert preference.update(observation, 3, runtime.sampling).learning_evidence == pytest.approx(expected)
        assert group.update(observation, 3, runtime.sampling).evidence['target'] == pytest.approx(
            .1 * (1 - group_mean + rho * (group_mean - group_target)))
    if rho:
        group, preference = learners(rejection_strength=rho, max_step=10)
        a = preference.update(replace(obs, proposal_token_id=1), 3, runtime.sampling)
        b = preference.update(replace(obs, proposal_token_id=2), 3, runtime.sampling)
        assert a.learning_evidence != pytest.approx(b.learning_evidence)


def test_coincidental_acceptance_and_zero_rejection_strength_preserve_old_direction():
    runtime = engine(top_k=2)
    for chosen, proposal, rho in [(3, 3, 1), (3, 1, 0)]:
        observation = replace(runtime.observe(), proposal_token_id=proposal)
        for original, experiment in zip(learners(rejection_strength=rho),
                                       learners(rejection_strength=rho, rejection_target='sampler')):
            a = original.update(observation, chosen, runtime.sampling)
            b = experiment.update(observation, chosen, runtime.sampling)
            assert a.sampling == b.sampling


@pytest.mark.parametrize('field', ['decay_on', 'write_reduction', 'rejection_target'])
def test_invalid_controls_rejected(field):
    for factory in [OnlineLearningConfig, TokenPreferenceConfig]:
        with pytest.raises(EditorError, match=field):
            factory(**{field: 'unknown'})


def test_cli_defaults_and_all_six_flags_reach_learners(tmp_path):
    defaults = build_parser().parse_args([])
    for prefix in ['learning', 'token_preference']:
        assert getattr(defaults, prefix + '_decay_on') == 'update'
        assert getattr(defaults, prefix + '_write_reduction') == 'sum'
        assert getattr(defaults, prefix + '_rejection_target') == 'proposal'
    io = ScriptedIO(['q', 'quit'])
    flags = [arg for prefix in ['learning', 'token-preference'] for arg in
             [f'--{prefix}-decay-on', 'evidence', f'--{prefix}-write-reduction', 'sqrt',
              f'--{prefix}-rejection-target', 'sampler']]
    with patch('trajectory_editor.episode_cli.TerminalIO', return_value=io), \
         patch('trajectory_editor.episode_cli._backend', return_value=GateBackend()), \
         patch('trajectory_editor.episode_cli.OnlineLearner', wraps=OnlineLearner) as group, \
         patch('trajectory_editor.episode_cli.TokenPreferenceLearner', wraps=TokenPreferenceLearner) as preference:
        assert main(['--model', str(tmp_path / 'model.gguf'), '--workspace', str(tmp_path / 'run.db'),
                     '--new-prompt', 'P', '--online-learning', '--token-preference',
                     '--token-preference-dimension', '2', *flags]) == 0
    for field, expected in [('decay_on', 'evidence'), ('write_reduction', 'sqrt'), ('rejection_target', 'sampler')]:
        assert group.call_args.kwargs[field] == expected
        assert getattr(preference.call_args.kwargs['config'], field) == expected


def test_nonlinear_fallback_freezes_original_sampler_target():
    from trajectory_editor.group_control import GroupControl
    runtime = engine(top_k=2, bias_groups=(
        BiasGroup('target', (BiasRule(routes=((1,), (3,)), bias=0.),)),
        BiasGroup('other', (BiasRule(routes=((2,),), bias=0.),))),
        group_controls=(GroupControl('other', 'promote', .1, max_bias=.1),))
    observation = replace(runtime.observe(), proposal_token_id=1)
    model = OnlineLearner(enabled=True, learning_gate='sampler', rejection_strength=1,
                          rejection_target='sampler', epsilon=.001)
    result = model.update(observation, 3, runtime.sampling)
    # At strength 1, chosen-vs-negative log-odds cancel the normalization.
    # The other group's controller is independent of the target's magnitude.
    expected_gradient = observation.statistics.distribution.probability(1) - 1
    assert result.gradients['target'] == pytest.approx(expected_gradient, abs=1e-5)
    assert result.skipped['other'] == 'controlled by appearance objective'


def test_write_reduction_precedes_clipping_and_frozen_groups_stay_frozen():
    runtime = memory_engine()
    runtime.sampling = replace(runtime.sampling, bias_groups=(
        *runtime.sampling.bias_groups,
        BiasGroup('frozen', (BiasRule(routes=((3,),), bias=0.),), bias=.3, learnable=False)))
    observation = replace(runtime.observe(), proposal_token_id=1)
    for learner in memory_learners('evidence', write_reduction='mean', learning_rate=10, max_step=.01):
        results = [learner.update(observation, 3, runtime.sampling)] * 4
        result = learner.aggregate(results, runtime.sampling)
        if hasattr(result, 'new_z'):
            assert result.learning_step_norm == pytest.approx(.01)
            assert result.learning_evidence == pytest.approx(results[0].learning_evidence)
        else:
            assert result.new_group_weights['target'] == pytest.approx(.16 + .01)
            assert result.new_group_weights['frozen'] == .3
            assert result.evidence == pytest.approx(results[0].evidence)

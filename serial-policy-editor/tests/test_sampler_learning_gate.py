"""Teacher evidence gated by actual decoder eligibility, independently of rank."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from tests.fakes import ScriptedIO
from tests.test_token_preference import FEATURES, TokenPreferenceBackend
from trajectory_editor.bias_rules import BiasGroup, BiasRule
from trajectory_editor.domain import EditorError, SamplingConfig
from trajectory_editor.episode_cli import (
    _token_preference_config_from_args, _token_preference_notice, _online_learning_notice,
    _write_learning_notice, build_parser, main,
)
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_actions import Write
from trajectory_editor.episode_policy import _WriteLearningAccumulator
from trajectory_editor.token_preference import TokenPreferenceConfig, TokenPreferenceLearner
from trajectory_editor.online_learning import OnlineLearningConfig, OnlineLearner


class GateBackend(TokenPreferenceBackend):
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
    return (OnlineLearner(**values), TokenPreferenceLearner(FEATURES, dimension=2, **values))


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
    runtime = engine(token_preference_vector=(.2, .1), token_preference_fast_vector=(.1, .2), token_preference_fast_strength=.5)
    model = TokenPreferenceLearner(FEATURES, enabled=True, learning_gate='sampler',
                                    fast_slow=True, decay=.2, fast_decay=.5)
    result = model.update(runtime.observe(), 3, runtime.sampling)
    assert result.learning_step_norm == result.fast_learning_step_norm == 0
    assert result.new_z == pytest.approx((.16, .08))
    assert result.new_fast_z == pytest.approx((.05, .1))
    runtime.sampling = replace(runtime.sampling, top_k=1)
    model = TokenPreferenceLearner(FEATURES, enabled=True, learning_gate='sampler', fast_slow=True,
                                    learning_rate=100, fast_learning_rate=100,
                                    max_step=.03, fast_max_step=.02, max_norm=.1, fast_max_norm=.1)
    result = model.update(runtime.observe(), 3, runtime.sampling)
    assert result.severity == 1
    assert result.learning_step_norm <= .03 + 1e-12 and result.fast_learning_step_norm <= .02 + 1e-12
    assert result.z_norm <= .1 + 1e-12 and result.fast_z_norm <= .1 + 1e-12


@pytest.mark.parametrize('tokens', ([1, 3, 2], [1, 2]))
def test_write_gates_each_token_and_updates_groups_only_for_positive_members(tokens):
    runtime = engine(top_k=2, token_preference_vector=(.2, .1),
                     token_preference_fast_vector=(.1, .2), token_preference_fast_strength=.5)
    runtime.sampling = replace(runtime.sampling,
        bias_groups=(replace(runtime.sampling.bias_groups[0], bias=.2),))
    group_model = OnlineLearner(enabled=True, learning_gate='sampler', decay=.2, learning_rate=.1)
    token_preference_model = TokenPreferenceLearner(FEATURES, enabled=True, learning_gate='sampler',
                                           fast_slow=True, decay=.2, fast_decay=.5)
    accumulator = _WriteLearningAccumulator(runtime.backend, runtime.sampling, group_model, token_preference_model)
    for token in tokens:
        accumulator.add(runtime.observe(), token)
    result = accumulator.finish(len(tokens))
    assert [t.sampler_eligible for t in result.tokens] == [t != 3 for t in tokens]
    for r in accumulator.token_preference_results:
        assert r.severity == float(r.chosen_token_id == 3)
    group_events = accumulator.group_results
    assert len(group_events) == len(tokens)
    assert [bool(event.evidence) for event in group_events] == [token == 3 for token in tokens]
    assert result.group_result.new_group_weights['target'] == accumulator.sampling.bias_groups[0].bias
    assert len(accumulator.token_preference_results) == len(tokens)
    assert result.token_preference_result.new_z == accumulator.sampling.token_preference_vector
    assert result.token_preference_result.new_fast_z == accumulator.sampling.token_preference_fast_vector
    payload = result.to_dict()
    assert len(payload['group_token_updates']) == len(tokens)
    assert len(payload['token_preference_token_updates']) == len(tokens)
    assert [t['sampler_eligible'] for t in payload['tokens']] == [t != 3 for t in tokens]
    io = ScriptedIO([])
    _write_learning_notice(io, result)
    assert f'{tokens.count(3)}/{len(tokens)} evidence' in io.output[-1]


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
        TokenPreferenceLearner(FEATURES, dimension=2, enabled=True, learning_gate='sampler'))
    runtime.apply(Write(' A B', 'exact'), on_precommit_observation=accumulator.add)
    result = accumulator.finish(runtime.boundary)
    assert [t.token_id for t in result.tokens] == [1, 2]
    assert all(t.sampler_eligible for t in result.tokens)
    assert result.token_preference_result.learning_step_norm == 0


def test_default_rank_behavior_and_disabled_learners_are_preserved():
    runtime = engine()
    for learner in (OnlineLearner(enabled=True), TokenPreferenceLearner(FEATURES, enabled=True, dimension=2)):
        result = learner.update(runtime.observe(), 3, runtime.sampling)
        assert result.learning_gate == 'rank' and result.sampler_eligible
        assert result.severity > 0 and result.update_norm > 0
    for learner in learners(enabled=False, decay=1):
        assert learner.update(runtime.observe(), 3, runtime.sampling).sampling == runtime.sampling


def test_cli_flags_validate_and_reach_both_learners(tmp_path):
    parser = build_parser()
    defaults = parser.parse_args([])
    assert defaults.learning_gate == defaults.token_preference_learning_gate == 'rank'
    assert defaults.learning_dead_zone_rank == 0
    assert defaults.token_preference_dead_zone_rank == 0
    args = parser.parse_args(['--learning-gate', 'sampler', '--token-preference-learning-gate', 'sampler'])
    assert _token_preference_config_from_args(args).learning_gate == 'sampler'
    zero_rank = parser.parse_args([
        '--learning-dead-zone-rank', '0',
        '--token-preference-dead-zone-rank', '0',
    ])
    assert zero_rank.learning_dead_zone_rank == 0
    assert zero_rank.token_preference_dead_zone_rank == 0
    for factory in (OnlineLearningConfig, TokenPreferenceConfig):
        with pytest.raises(EditorError, match='gate'):
            factory(learning_gate='unknown')
    io = ScriptedIO(['q', 'quit'])
    backend = GateBackend()
    with patch('trajectory_editor.episode_cli.TerminalIO', return_value=io), \
         patch('trajectory_editor.episode_cli._backend', return_value=backend), \
         patch('trajectory_editor.episode_cli.OnlineLearner', wraps=OnlineLearner) as group_spy, \
         patch('trajectory_editor.episode_cli.TokenPreferenceLearner', wraps=TokenPreferenceLearner) as token_preference_spy:
        assert main(['--model', str(tmp_path / 'model.gguf'), '--workspace', str(tmp_path / 'run.db'),
                     '--new-prompt', 'P', '--online-learning', '--token-preference', '--token-preference-dimension', '2',
                     '--learning-gate', 'sampler', '--token-preference-learning-gate', 'sampler']) == 0
    assert group_spy.call_args.kwargs['learning_gate'] == 'sampler'
    assert token_preference_spy.call_args.kwargs['config'].learning_gate == 'sampler'


def test_single_update_notices_explain_gate():
    for settings, message in (({}, 'already eligible'), ({'top_k': 2}, 'excluded')):
        runtime = engine(**settings)
        io = ScriptedIO([])
        group_model, token_preference_model = learners()
        _online_learning_notice(io, group_model.update(runtime.observe(), 3, runtime.sampling))
        _token_preference_notice(io, token_preference_model.update(runtime.observe(), 3, runtime.sampling))
        assert all(message in line for line in io.output)

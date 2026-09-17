"""Behavior-preserving checks for the ephemeral learning observation cache."""

from tests.test_sampler_learning_gate import engine
from tests.test_token_preference import FEATURES
from trajectory_editor.learning_observation import CompiledLearningObservation
from trajectory_editor.online_learning import OnlineLearner
from trajectory_editor.token_preference import TokenPreferenceLearner


def test_compiled_observation_reuses_policy_inputs_for_both_learners():
    runtime = engine(token_preference_vector=(0.2, 0.1))
    observation = runtime.observe()
    compiled = CompiledLearningObservation.from_observation(observation)

    group = OnlineLearner(enabled=True, learning_rate=0.2)
    preference = TokenPreferenceLearner(FEATURES, enabled=True, dimension=2)
    group_result = group.update(
        observation, 3, runtime.sampling, compiled=compiled
    )
    preference_result = preference.update(
        observation, 3, runtime.sampling, compiled=compiled
    )

    assert compiled.statistics is observation.statistics
    assert compiled.policy_probabilities is observation.statistics.policy_probabilities
    assert compiled.sampler_eligible(3) is True
    assert compiled.sampler_probability(3) == observation.statistics.distribution.probability(3)
    assert group_result.chosen_token_id == preference_result.chosen_token_id == 3
    assert group_result.severity == preference_result.severity == 1.0


def test_compiled_and_direct_updates_are_identical():
    runtime = engine(token_preference_vector=(0.12, -0.08))
    sampling = runtime.sampling
    observation = runtime.observe()
    compiled = CompiledLearningObservation.from_observation(observation)

    group = OnlineLearner(enabled=True, learning_rate=0.2)
    direct_group = group.update(observation, 3, sampling)
    compiled_group = group.update(observation, 3, sampling, compiled=compiled)
    assert compiled_group == direct_group

    preference = TokenPreferenceLearner(FEATURES, enabled=True, dimension=2)
    direct_preference = preference.update(observation, 3, sampling)
    compiled_preference = preference.update(
        observation, 3, sampling, compiled=compiled
    )
    assert compiled_preference == direct_preference


def test_group_matchers_are_normalized_once_per_route_template():
    runtime = engine()
    learner = OnlineLearner(enabled=True)
    learner.update(runtime.observe(), 3, runtime.sampling)
    learner.update(runtime.observe(), 3, runtime.sampling)

    assert len(learner._group_features._matchers) == 1

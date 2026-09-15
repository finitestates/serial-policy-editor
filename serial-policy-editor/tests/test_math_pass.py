"""Numerical coverage for the versioned v2 policy mathematics."""

from types import SimpleNamespace

import numpy as np
import pytest

from trajectory_editor.bias_rules import BiasGroup, BiasRule
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.group_control import (
    GroupControl,
    GroupRouteTrie,
    control_adjustments,
    gamma_poisson_rate,
)
from trajectory_editor.latent_features import project_token_embeddings
from trajectory_editor.latent_preference import (
    LatentPreferenceLearner,
    choice_gradient,
    fisher_matrix,
    pairwise_logistic_gradient,
)
from trajectory_editor.sampling import (
    ObservationStatistics,
    calibrated_latent_gain,
    policy_kl,
)


def _group(name="phrase", routes=((1, 2), (2, 1))):
    return BiasGroup(
        name,
        (BiasRule(routes=routes, bias=0.0),),
    )


def _latent_observation(sampling, features, logits=None, proposal=0):
    values = np.asarray(
        logits if logits is not None else np.linspace(1.2, -0.2, len(features)),
        dtype=np.float64,
    )
    statistics = ObservationStatistics(
        values, sampling, [], latent_features=features,
    )
    return SimpleNamespace(
        boundary=0,
        proposal_token_id=proposal,
        logits=values,
        statistics=statistics,
    )


def test_whitened_projection_is_centered_isotropic_and_deterministic():
    embeddings = np.asarray(
        [[0.0, 1.0, 3.0], [1.0, 2.0, 0.0], [4.0, -1.0, 2.0],
         [2.0, 5.0, 1.0], [-2.0, 0.5, 4.0], [3.0, 1.0, -1.0]],
        dtype=np.float32,
    )
    first = project_token_embeddings(
        embeddings, feature_dimension=3, projection_seed=17,
        feature_scheme="whitened-projection-v2",
    )
    second = project_token_embeddings(
        embeddings, feature_dimension=3, projection_seed=17,
        feature_scheme="whitened-projection-v2",
    )
    covariance = np.cov(first.astype(np.float64), rowvar=False, bias=True)
    assert np.array_equal(first, second)
    np.testing.assert_allclose(np.mean(first, axis=0), 0.0, atol=2.0e-6)
    assert np.mean(np.sum(first * first, axis=1)) == pytest.approx(1.0, abs=2.0e-6)
    assert np.linalg.cond(covariance + 1.0e-9 * np.eye(3)) < 1.2
    assert not first.flags.writeable


def test_choice_and_pairwise_gradients_match_centered_finite_differences():
    features = np.asarray(
        [[-1.0, 0.2], [0.5, 1.1], [1.4, -0.4], [-0.3, -0.8]],
        dtype=np.float64,
    )
    base = np.asarray([0.2, -0.1, 0.4, -0.7])
    z = np.asarray([0.3, -0.2])
    chosen, rejected = 1, 2

    def log_probability(vector):
        logits = base + features @ vector
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities)
        return np.log(probabilities[chosen])

    logits = base + features @ z
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= np.sum(probabilities)
    analytical = choice_gradient(features, probabilities, chosen)
    finite_difference = np.asarray([
        (log_probability(z + np.eye(2)[index] * 1.0e-6)
         - log_probability(z - np.eye(2)[index] * 1.0e-6)) / 2.0e-6
        for index in range(2)
    ])
    np.testing.assert_allclose(analytical, finite_difference, rtol=1.0e-7, atol=1.0e-8)

    delta = features[chosen] - features[rejected]

    def pair_log_probability(vector):
        return -np.logaddexp(0.0, -float(vector @ delta))

    pair_analytical = pairwise_logistic_gradient(z, features[chosen], features[rejected])
    pair_finite_difference = np.asarray([
        (pair_log_probability(z + np.eye(2)[index] * 1.0e-6)
         - pair_log_probability(z - np.eye(2)[index] * 1.0e-6)) / 2.0e-6
        for index in range(2)
    ])
    np.testing.assert_allclose(
        pair_analytical, pair_finite_difference, rtol=1.0e-7, atol=1.0e-8
    )


def test_fisher_is_symmetric_psd_and_damping_is_positive_definite():
    features = np.asarray(
        [[-1.0, 0.2], [0.5, 1.1], [1.4, -0.4], [-0.3, -0.8]],
        dtype=np.float64,
    )
    probabilities = np.asarray([0.1, 0.2, 0.3, 0.4])
    fisher = fisher_matrix(features, probabilities)
    np.testing.assert_allclose(fisher, fisher.T)
    assert np.min(np.linalg.eigvalsh(fisher)) >= -1.0e-12
    assert np.min(np.linalg.eigvalsh(fisher + 1.0e-3 * np.eye(2))) > 0.0


def test_v2_learning_surface_does_not_depend_on_deployment_strength():
    features = np.asarray(
        [[-1.0, 0.0], [1.0, 0.2], [0.7, 1.0], [-0.2, -1.0], [0.0, 0.4]],
        dtype=np.float32,
    )
    common = dict(
        latent_preference_z=(0.4, -0.3),
        latent_learning_scheme="fisher-kl-v2",
        latent_feature_scheme="whitened-projection-v2",
    )
    low = SamplingConfig(**common, latent_strength=0.5)
    high = SamplingConfig(**common, latent_strength=8.0)
    low_stats = _latent_observation(low, features).statistics
    high_stats = _latent_observation(high, features).statistics
    np.testing.assert_allclose(
        low_stats.pre_latent_probabilities,
        high_stats.pre_latent_probabilities,
    )
    np.testing.assert_allclose(
        low_stats.learning_probabilities,
        high_stats.learning_probabilities,
    )
    assert policy_kl(
        low_stats.baseline_probabilities, low_stats.pre_latent_probabilities
    ) < policy_kl(
        high_stats.baseline_probabilities, high_stats.pre_latent_probabilities
    )


def test_kl_calibrated_gain_hits_exact_tilt_budget():
    probabilities = np.asarray([0.05, 0.15, 0.3, 0.5], dtype=np.float64)
    scores = np.asarray([-1.2, 0.1, 0.7, 1.6], dtype=np.float64)
    target = 0.08
    gain = calibrated_latent_gain(
        probabilities, scores, target, min_gain=0.0, max_gain=8.0
    )
    centered = scores - np.dot(probabilities, scores)
    tilted = probabilities * np.exp(gain * centered)
    tilted /= np.sum(tilted)
    assert policy_kl(tilted, probabilities) == pytest.approx(target, abs=1.0e-10)


def test_fisher_kl_update_is_finite_and_within_safety_budget():
    features = np.asarray(
        [[-1.0, 0.0], [1.0, 0.2], [0.7, 1.0], [-0.2, -1.0], [0.0, 0.4]],
        dtype=np.float32,
    )
    sampling = SamplingConfig(
        latent_learning_scheme="fisher-kl-v2",
        latent_feature_scheme="whitened-projection-v2",
    )
    observation = _latent_observation(sampling, features, proposal=0)
    learner = LatentPreferenceLearner(
        features, enabled=True, dimension=2, learning_scheme="fisher-kl-v2",
        learning_metric="fisher", learning_kl=0.02, fisher_mode="diagonal",
        fisher_ridge=1.0e-3, max_step=10.0, max_norm=10.0,
        no_severity_attenuation=True,
    )
    result = learner.update(observation, 2, sampling)
    assert result.learning_scheme == "fisher-kl-v2"
    assert result.fisher_condition_estimate >= 1.0
    assert result.learning_step_kl <= result.requested_learning_kl * 1.05
    assert np.all(np.isfinite(result.new_z))


def test_pairwise_rejection_gradient_fades_when_margin_is_correct():
    features = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    base = SamplingConfig(
        latent_learning_scheme="fisher-kl-v2",
        latent_preference_z=(0.0, 0.0),
    )
    observation = _latent_observation(base, features, proposal=2)
    learner = LatentPreferenceLearner(
        features, enabled=True, dimension=2, learning_scheme="fisher-kl-v2",
        learning_metric="fisher", learning_kl=0.02, rejection_strength=1.0,
        no_severity_attenuation=True, max_step=10.0, max_norm=10.0,
    )
    zero_margin = learner.update(observation, 1, base)
    positive = SamplingConfig(
        latent_learning_scheme="fisher-kl-v2",
        latent_preference_z=(4.0, 0.0),
    )
    positive_result = learner.update(
        _latent_observation(positive, features, proposal=2), 1, positive
    )
    assert zero_margin.loss > positive_result.loss
    assert np.linalg.norm(zero_margin.learning_evidence) > 0.0
    assert np.linalg.norm(positive_result.learning_evidence) < np.linalg.norm(
        zero_margin.learning_evidence
    )


def test_gamma_poisson_posterior_mean_and_variance():
    mean, variance = gamma_poisson_rate(3, 20, 0.1, prior_exposure=16)
    assert mean == pytest.approx((3 + 16 * 0.1) / 36)
    assert variance == pytest.approx((3 + 16 * 0.1) / (36 * 36))
    _, lower_variance = gamma_poisson_rate(3, 100, 0.1, prior_exposure=16)
    assert lower_variance < variance


def test_v2_controller_reports_explicit_terms_without_hidden_gain():
    group = _group(routes=((1,),))
    control = GroupControl(
        "phrase", "promote", 0.1, scheme="appearance-rate-v2",
        feedforward_gain=0.5, proportional_gain=0.25, integral_gain=0.0,
        max_bias=10.0,
    )
    biases, diagnostics = control_adjustments(
        (control,), (group,), [0, 0, 0], np.zeros(4),
        scheme="appearance-rate-v2",
    )
    row = diagnostics[0]
    assert row["feedforward"] == pytest.approx(0.5 * np.log(2.0))
    assert row["integral"] == 0.0
    assert row["raw_pressure"] == pytest.approx(
        row["feedforward"] + row["proportional"] + row["integral"]
    )
    assert biases[1] == pytest.approx(row["pressure"])


def test_group_route_trie_preserves_overlapping_suffix_prefixes():
    trie = GroupRouteTrie(((1, 2), (2, 1, 3)))
    state = trie.state_for_history((1, 2, 1))
    assert trie.nodes[state].prefix == (2, 1)
    assert trie.outgoing(state) == (3,)


def test_v2_sampling_state_round_trip_keeps_math_schemes():
    state = SamplingConfig(
        latent_feature_scheme="whitened-projection-v2",
        latent_learning_scheme="fisher-kl-v2",
        latent_influence_mode="kl",
        latent_influence_kl=0.03,
        group_control_scheme="appearance-rate-v2",
        group_controls=(GroupControl(
            "phrase", "promote", 0.1, scheme="appearance-rate-v2",
        ),),
        bias_groups=(_group(),),
    )
    assert SamplingConfig.from_record(state.to_dict()) == state

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rust_sampler as sampling
from trajectory_editor.bias_groups import BiasGroup, BiasMember, BiasRoute, BiasToken
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.policy_calculations import PolicyCalculations
from trajectory_editor.core.sampler_config import SamplerConfig


ROOT = Path(__file__).resolve().parents[1]


def distribution(ids=(4, 1, 7), probabilities=(0.2, 0.5, 0.3), scores=(0.1, 0.9, 0.4)):
    return sampling.SparseDistribution(
        np.asarray(ids, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        None if scores is None else np.asarray(scores, dtype=np.float64),
    )


def test_numpy_vectors_are_rejected_instead_of_flattened_or_truncated():
    with pytest.raises(ValueError, match="one-dimensional"):
        sampling._validated_logits(np.ones((2, 2)))
    with pytest.raises(ValueError, match="candidate probabilities"):
        sampling.draw_token(
            sampling.SparseDistribution(
                np.asarray([1, 2], dtype=np.int64),
                np.asarray([1.0], dtype=np.float64),
            ),
            seed=1,
            stream_fingerprint="a" * 64,
            aligned_step=0,
        )
    with pytest.raises(ValueError, match="candidate scores do not match"):
        sampling.gaussian_ranking_scores(
            sampling.SparseDistribution(
                np.asarray([1, 2], dtype=np.int64),
                np.asarray([0.5, 0.5]),
                np.asarray([1.0]),
            ),
            seed=1,
            stream_fingerprint="a" * 64,
            aligned_step=0,
        )


def test_address_and_noise_validation_uses_editor_errors():
    with pytest.raises(EditorError, match="signed-64-bit"):
        sampling.position_uniform(1 << 63, "a" * 64, 0)
    with pytest.raises(EditorError, match="lowercase SHA-256"):
        sampling.position_uniform(1, "A" * 64, 0)
    with pytest.raises(EditorError, match="nonnegative integer"):
        sampling.position_uniform(1, "a" * 64, True)
    with pytest.raises(EditorError, match="finite and nonnegative"):
        sampling.gaussian_ranking_scores(
            distribution(),
            seed=1,
            stream_fingerprint="a" * 64,
            aligned_step=0,
            noise_std=float("inf"),
        )
    with pytest.raises(EditorError, match="finite and nonnegative"):
        sampling.perturbation_ranking_scores(
            distribution(),
            seed=1,
            stream_fingerprint="a" * 64,
            aligned_step=0,
            kernel="laplace-max",
            noise_std=-0.1,
        )
    for invalid in (-0.1, float("nan")):
        with pytest.raises(EditorError, match="finite and nonnegative"):
            sampling.gumbel_ranking_scores(
                distribution(),
                seed=1,
                stream_fingerprint="a" * 64,
                aligned_step=0,
                gumbel_noise_scale=invalid,
            )
    with pytest.raises(EditorError, match="finite and nonnegative"):
        sampling.perturbation_ranking_scores(
            distribution(),
            seed=1,
            stream_fingerprint="a" * 64,
            aligned_step=0,
            kernel="laplace-max",
            noise_std=float("inf"),
        )


def test_gumbel_model_rank_checks_and_historical_zero_scale_shortcut():
    value = distribution()
    with pytest.raises(ValueError, match="requires candidate model ranks"):
        sampling.gumbel_ranking_scores(
            value,
            seed=3,
            stream_fingerprint="a" * 64,
            aligned_step=0,
            noise_address="model-rank",
        )
    with pytest.raises(ValueError, match="distinct positive integers"):
        sampling.gumbel_ranking_scores(
            value,
            seed=3,
            stream_fingerprint="a" * 64,
            aligned_step=0,
            noise_address="model-rank",
            candidate_model_ranks=np.asarray([1, 1, 2], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="distinct positive integers"):
        sampling.gumbel_ranking_scores(
            value,
            seed=3,
            stream_fingerprint="a" * 64,
            aligned_step=0,
            noise_address="model-rank",
            candidate_model_ranks=np.asarray([1.0, 2.0, 3.0]),
        )

    raw = np.asarray(value.scores, dtype=np.float64)
    zero_noise = sampling.gumbel_ranking_scores(
        value,
        seed="unused because scale is zero",
        stream_fingerprint="not a fingerprint",
        aligned_step=-7,
        candidate_model_ranks=np.ones((2, 2)),
        gumbel_noise_scale=0.0,
    )
    np.testing.assert_array_equal(zero_noise, raw)
    assert sampling.draw_token(
        value,
        seed="unused because scale is zero",
        stream_fingerprint="not a fingerprint",
        aligned_step=-7,
        kernel="gumbel-max",
        candidate_model_ranks=np.ones((2, 2)),
        gumbel_noise_scale=0.0,
    ) == 1


def test_exact_ties_and_duplicate_candidate_ids_keep_token_id_tie_break():
    value = distribution(ids=(9, 2, 2), probabilities=(0.3, 0.4, 0.3), scores=(1.0, 1.0, 1.0))
    assert sampling.gumbel_winner(value, np.asarray([4.0, 4.0, 4.0])) == 2
    assert sampling.gaussian_winner(value, np.asarray([0.0, 0.0, 0.0])) == 2
    assert sampling.perturbation_winner(value, np.asarray([0.0, 0.0, 0.0])) == 2


def test_empty_vectors_student_t_edges_and_conditional_prefix_validation():
    with pytest.raises(ValueError, match="finite nonempty"):
        sampling._softmax(np.asarray([], dtype=np.float64))
    with pytest.raises(EditorError, match="student_t_df"):
        sampling.perturbation_ranking_scores(
            distribution(),
            seed=1,
            stream_fingerprint="a" * 64,
            aligned_step=0,
            kernel="student-t-max",
            student_t_df=0.0,
        )
    with pytest.raises(EditorError, match="nonnegative token IDs"):
        sampling.conditional_gumbel_top_k(
            np.asarray([-0.1, -0.5]),
            count=2,
            parent_score=0.0,
            parent_log_probability=-0.1,
            seed=1,
            stream_fingerprint="a" * 64,
            aligned_step=0,
            prefix_token_ids=[1, -1],
        )


def test_history_penalty_fixtures_match_production_policy_calculations():
    fixture = json.loads(
        (ROOT / "fixtures" / "sampling-cases.json").read_text(encoding="utf-8")
    )
    tolerance = fixture["float_tolerance"]
    for case in fixture["history_penalty_cases"]:
        logits = np.asarray(case["logits"], dtype=np.float64)
        original_logits = logits.copy()
        history = case["history_token_ids"]
        config = SamplerConfig(**case["config"])
        reference = PolicyCalculations(logits, config, history)
        actual = sampling.apply_history_penalties(logits, history, config)

        np.testing.assert_array_equal(logits, original_logits)
        np.testing.assert_allclose(
            actual,
            reference.adjusted,
            rtol=0.0,
            atol=tolerance,
            err_msg=case["name"],
        )
        np.testing.assert_allclose(
            actual,
            case["expected"]["adjusted_logits"],
            rtol=0.0,
            atol=tolerance,
            err_msg=case["name"],
        )
        assert [reference.raw_rank(token_id) for token_id in range(len(logits))] == case[
            "expected"
        ]["raw_ranks"]
        assert [
            reference.policy_rank(token_id) for token_id in range(len(logits))
        ] == case["expected"]["policy_ranks"]


def test_history_penalty_validation_matches_python_error_categories():
    logits = np.asarray([2.0, 0.0, -1.0], dtype=np.float64)
    active = SamplerConfig(repeat_penalty=1.1)
    inactive = SamplerConfig(repeat_last_n=0, presence_penalty=0.5)

    with pytest.raises(ValueError, match="one-dimensional"):
        sampling.apply_history_penalties(logits, np.asarray([[0, 1]]), active)
    with pytest.raises(ValueError, match="decoder vocabulary"):
        sampling.apply_history_penalties(logits, [-1], active)
    with pytest.raises(ValueError, match="decoder vocabulary"):
        sampling.apply_history_penalties(logits, [3], inactive)
    with pytest.raises(ValueError, match="finite nonempty"):
        sampling.apply_history_penalties(np.asarray([np.nan]), [0], active)
    with pytest.raises(ValueError, match="exact prefix token ids"):
        sampling.apply_history_penalties(logits, None, active)
    np.testing.assert_array_equal(
        sampling.apply_history_penalties(logits, None, inactive), logits
    )

    wide_window = SamplerConfig(repeat_penalty=1.2, repeat_last_n=1 << 100)
    wide_history = [0, 1, 1]
    expected = PolicyCalculations(logits, wide_window, wide_history).adjusted
    actual = sampling.apply_history_penalties(logits, wide_history, wide_window)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)

    for kwargs in (
        {"repeat_penalty": 0.0},
        {"repeat_penalty": float("inf")},
        {"repeat_last_n": -2},
        {"presence_penalty": float("nan")},
        {"frequency_penalty": float("inf")},
    ):
        with pytest.raises(EditorError):
            SamplerConfig(**kwargs)

    overflowing = SamplerConfig(repeat_penalty=1.0e308)
    with pytest.raises(ValueError, match="history penalties produced non-finite"):
        sampling.apply_history_penalties(
            np.asarray([-1.0e308]), [0], overflowing
        )


def _bias_fixture_config(case):
    return SamplerConfig(
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        min_p=0.0,
        typical_p=1.0,
        tail_free_z=1.0,
        repeat_last_n=0,
        bias_groups=case["bias_groups"],
        token_biases=case["token_biases"],
    )


def _bias_contribution_record(item):
    return {
        "token_id": item.token_id,
        "amount": item.amount,
        "active": item.active,
        "source": item.source,
        "group_name": item.group_name,
        "member_routes": [
            {
                "member_text": route.member_text,
                "member_literal": route.member_literal,
                "route": route.route.to_dict(),
                "active": route.active,
            }
            for route in item.member_routes
        ],
    }


def test_bias_fixtures_match_python_contributions_and_policy_calculations():
    fixture = json.loads(
        (ROOT / "fixtures" / "sampling-cases.json").read_text(encoding="utf-8")
    )
    tolerance = fixture["float_tolerance"]
    for case in fixture["bias_cases"]:
        config = _bias_fixture_config(case)
        logits = np.asarray(case["logits"], dtype=np.float64)
        original_logits = logits.copy()
        history = case["history_token_ids"]
        reference = PolicyCalculations(logits, config, history)
        expected_contributions = config.bias_contributions(
            history, include_inactive=case["include_inactive"]
        )
        actual_contributions = sampling.bias_contributions(
            history, config, include_inactive=case["include_inactive"]
        )
        assert actual_contributions == expected_contributions, case["name"]
        assert [
            _bias_contribution_record(item) for item in actual_contributions
        ] == case["expected"]["contributions"], case["name"]

        expected_active = config.active_biases(history)
        actual_active = sampling.active_biases(history, config)
        assert actual_active == expected_active
        assert [list(item) for item in sorted(actual_active.items())] == case[
            "expected"
        ]["active_biases"]

        adjusted = sampling.apply_biases(logits, history, config)
        np.testing.assert_allclose(
            adjusted,
            reference.adjusted,
            rtol=0.0,
            atol=tolerance,
            err_msg=case["name"],
        )
        np.testing.assert_allclose(
            adjusted,
            case["expected"]["adjusted_logits"],
            rtol=0.0,
            atol=tolerance,
            err_msg=f"{case['name']} fixture",
        )
        np.testing.assert_array_equal(logits, original_logits)
        assert [reference.raw_rank(token_id) for token_id in range(len(logits))] == case[
            "expected"
        ]["raw_ranks"]
        assert [
            reference.policy_rank(token_id) for token_id in range(len(logits))
        ] == case["expected"]["policy_ranks"]


def test_bias_adapter_validation_rejects_invalid_ids_and_nonfinite_results():
    logits = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    direct_out_of_range = SamplerConfig(token_biases=(BiasToken(3, 1.0),))
    with pytest.raises(ValueError, match="bias token id is outside"):
        sampling.apply_biases(logits, [], direct_out_of_range)

    route_out_of_range = SamplerConfig(
        bias_groups=(
            BiasGroup(
                "invalid",
                (BiasMember("phrase", (BiasRoute((0, 3), ("phrase",)),)),),
                1.0,
            ),
        )
    )
    with pytest.raises(ValueError, match="bias token id is outside"):
        sampling.apply_biases(logits, [0], route_out_of_range)
    with pytest.raises(ValueError, match="history token ids"):
        sampling.apply_biases(logits, [3], SamplerConfig())

    overflowing = SamplerConfig(token_biases=(BiasToken(0, 1.0e308),))
    with pytest.raises(ValueError, match="non-finite policy logits"):
        sampling.apply_biases(np.asarray([1.0e308]), [], overflowing)
    with pytest.raises(ValueError, match="finite nonempty"):
        sampling.apply_biases(np.asarray([np.nan]), [], SamplerConfig())

    no_history = SamplerConfig(
        bias_groups=(
            BiasGroup(
                "phrase",
                (BiasMember("phrase", (BiasRoute((0, 1), ("phrase",)),)),),
                1.0,
            ),
        )
    )
    with pytest.raises(EditorError, match="exact context token IDs"):
        sampling.active_biases(None, no_history)


def test_bias_kernel_preserves_policy_adjustment_order_when_composed():
    config = SamplerConfig(
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        min_p=0.0,
        repeat_penalty=2.0,
        repeat_last_n=-1,
        presence_penalty=0.25,
        frequency_penalty=0.1,
        activation_vector=(1.0,),
        activation_vector_strength=0.5,
        bias_groups=(
            BiasGroup(
                "phrase",
                (BiasMember("phrase", (BiasRoute((0, 1, 2), ("phrase",)),)),),
                0.75,
            ),
        ),
        token_biases=(BiasToken(2, 0.25),),
    )
    history = [0, 1]
    logits = np.asarray([-2.0, 1.0, 0.0, 0.5, 0.0], dtype=np.float64)
    original_logits = logits.copy()
    activation_adjustments = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0])
    ephemeral = {2: 0.125}
    reference = PolicyCalculations(
        logits,
        config,
        history,
        activation_logit_adjustments=activation_adjustments,
        ephemeral_logit_biases=ephemeral,
    )

    after_history = sampling.apply_history_penalties(logits, history, config)
    after_activation = sampling.apply_activation_adjustments(
        after_history,
        activation_adjustments,
        config.activation_vector_strength,
    )
    after_grouped_and_direct = sampling.apply_biases(
        after_activation, history, config
    )
    composed = sampling.apply_ephemeral_biases(after_grouped_and_direct, ephemeral)

    np.testing.assert_allclose(composed, reference.adjusted, rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(logits, original_logits)


def test_adjustment_cfg_and_metric_fixtures_match_policy_calculations():
    fixture = json.loads(
        (ROOT / "fixtures" / "sampling-cases.json").read_text(encoding="utf-8")
    )
    tolerance = fixture["float_tolerance"]
    for case in fixture["adjustment_cases"]:
        logits = np.asarray(case["logits"], dtype=np.float64)
        original = logits.copy()
        config = SamplerConfig.from_record(case["config"])
        history = case["history_token_ids"]
        activation = case["activation_adjustments"]
        ephemeral = {int(token): amount for token, amount in case["ephemeral_biases"].items()}
        policy = PolicyCalculations(
            logits,
            config,
            history,
            activation_logit_adjustments=activation,
            ephemeral_logit_biases=ephemeral,
        )
        adjusted = sampling.apply_history_penalties(logits, history, config)
        if activation is not None:
            adjusted = sampling.apply_activation_adjustments(
                adjusted,
                np.asarray(activation, dtype=np.float64),
                config.activation_vector_strength,
            )
        adjusted = sampling.apply_biases(adjusted, history, config)
        adjusted = sampling.apply_ephemeral_biases(adjusted, ephemeral)
        np.testing.assert_allclose(adjusted, policy.adjusted, rtol=0.0, atol=tolerance)
        np.testing.assert_allclose(
            adjusted, case["expected"]["adjusted_logits"], rtol=0.0, atol=tolerance
        )
        np.testing.assert_array_equal(logits, original)
        assert [policy.raw_rank(token) for token in range(len(logits))] == case[
            "expected"
        ]["raw_ranks"]
        assert [policy.policy_rank(token) for token in range(len(logits))] == case[
            "expected"
        ]["policy_ranks"]

    for case in fixture["activation_kernel_cases"]:
        logits = np.asarray(case["logits"], dtype=np.float64)
        original = logits.copy()
        actual = sampling.apply_activation_adjustments(
            logits,
            np.asarray(case["adjustments"], dtype=np.float64),
            case["strength"],
        )
        np.testing.assert_allclose(
            actual, case["expected"], rtol=0.0, atol=tolerance
        )
        np.testing.assert_array_equal(logits, original)

    for case in fixture["cfg_cases"]:
        conditional = np.asarray(case["conditional_logits"], dtype=np.float64)
        unconditional = np.asarray(case["unconditional_logits"], dtype=np.float64)
        actual = sampling.cfg_combine_logits(conditional, unconditional, case["scale"])
        np.testing.assert_allclose(
            actual, case["expected"], rtol=0.0, atol=tolerance
        )

    for case in fixture["metric_cases"]:
        logits = np.asarray(case["logits"], dtype=np.float64)
        config = SamplerConfig.from_record(case["config"])
        policy = PolicyCalculations(logits, config, case["history_token_ids"])
        policy_logits = None if policy._policy_shares_raw else policy.adjusted
        metrics = sampling.PolicyMetrics(logits, policy_logits)
        expected = case["expected"]
        assert metrics.raw_logsumexp_ready is False
        assert metrics.logit_mean_std_ready is False
        assert metrics.maximum == expected["maximum"]
        assert metrics.top_raw_ids(len(expected["top_raw_ids"])) == expected["top_raw_ids"]
        assert metrics.top_policy_ids(len(expected["top_policy_ids"])) == expected["top_policy_ids"]
        assert [metrics.raw_rank(token) for token in range(len(logits))] == expected["raw_ranks"]
        assert [metrics.policy_rank(token) for token in range(len(logits))] == expected["policy_ranks"]
        assert metrics.raw_logsumexp_ready is False
        selected = case["selected_ids"]
        actual_z = metrics.logit_z_scores(selected)
        for actual, golden in zip(actual_z, expected["logit_z_scores"]):
            if golden is None:
                assert actual is None
            else:
                assert actual == pytest.approx(golden, abs=tolerance)
        assert metrics.logit_mean_std_ready is True
        assert metrics.raw_logsumexp_ready is False
        np.testing.assert_allclose(
            metrics.raw_probabilities(selected),
            expected["raw_probabilities"],
            rtol=0.0,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            metrics.policy_probabilities_at(selected),
            expected["policy_probabilities"],
            rtol=0.0,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            [metrics.raw_nll(token) for token in range(len(logits))],
            expected["raw_nll"],
            rtol=0.0,
            atol=tolerance,
        )
        assert metrics.log_z == pytest.approx(expected["log_z"], abs=tolerance)
        assert metrics.denominator == pytest.approx(
            expected["denominator"], abs=tolerance
        )
        assert metrics.raw_logsumexp_ready is True


def test_adjustment_and_cfg_validation_reject_shape_ids_nonfinite_and_overflow():
    logits = np.asarray([1.0, -2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="match the policy vocabulary"):
        sampling.apply_activation_adjustments(logits, np.asarray([1.0]), 1.0)
    with pytest.raises(ValueError, match="adjustments must be finite"):
        sampling.apply_activation_adjustments(
            logits, np.asarray([0.0, np.nan, 0.0]), 1.0
        )
    with pytest.raises(ValueError, match="strength must be finite"):
        sampling.apply_activation_adjustments(logits, np.zeros(3), float("inf"))
    assert sampling.apply_activation_adjustments(logits, np.ones(3), -0.5).tolist() == [
        0.5,
        -2.5,
        2.5,
    ]
    with pytest.raises(ValueError, match="non-finite policy logits"):
        sampling.apply_activation_adjustments(
            np.asarray([1.0e308]), np.asarray([1.0e308]), 2.0
        )

    for invalid in ({3: 1.0}, {-1: 1.0}, {1: float("nan")}):
        with pytest.raises(ValueError, match="ephemeral logit biases are invalid"):
            sampling.apply_ephemeral_biases(logits, invalid)
    with pytest.raises(ValueError, match="produced non-finite policy logits"):
        sampling.apply_ephemeral_biases(np.asarray([1.0e308]), {0: 1.0e308})
    identity = sampling.apply_ephemeral_biases(logits, {})
    np.testing.assert_array_equal(identity, logits)
    assert identity is not logits

    with pytest.raises(ValueError, match="one-dimensional array"):
        sampling.cfg_combine_logits(np.ones((1, 3)), logits, 1.0)
    with pytest.raises(EditorError, match="cfg_scale must be finite and nonnegative"):
        sampling.cfg_combine_logits(logits, logits, -1.0)
    with pytest.raises(ValueError, match="matching vocabulary"):
        sampling.cfg_combine_logits(logits, np.asarray([1.0]), 1.0)
    with pytest.raises(ValueError, match="non-finite logits"):
        sampling.cfg_combine_logits(
            np.asarray([1.0e308]), np.asarray([-1.0e308]), 2.0
        )


def test_policy_metrics_keep_softmax_lazy_and_validate_ids():
    logits = np.asarray([2.0, 1.0, 0.0, -1.0], dtype=np.float64)
    adjusted = np.asarray([0.0, 0.5, 1.0, -1.0], dtype=np.float64)
    metrics = sampling.PolicyMetrics(logits, adjusted)
    assert metrics.raw_logsumexp_ready is False
    assert metrics.logit_mean_std_ready is False
    assert metrics.maximum == 2.0
    assert metrics.raw_rank(1) == 2
    assert metrics.top_raw_ids(3) == [0, 1, 2]
    assert metrics.top_raw_ids(0) == []
    assert metrics.top_raw_ids(-1) == [0, 1]
    assert metrics.top_policy_ids(3) == [2, 1, 0]
    assert metrics.logit_z(0) == pytest.approx((2.0 - np.mean(logits)) / np.std(logits))
    assert metrics.raw_logsumexp_ready is False
    assert metrics.logit_mean_std_ready is True
    with pytest.raises(ValueError, match="outside the decoder vocabulary"):
        metrics.raw_probabilities([-1])
    with pytest.raises(ValueError, match="outside the decoder vocabulary"):
        metrics.policy_rank(4)
    flat = sampling.PolicyMetrics(np.asarray([4.0, 4.0]))
    assert flat.logit_z_scores([0, 1]) == [None, None]
    assert flat.raw_logsumexp_ready is False

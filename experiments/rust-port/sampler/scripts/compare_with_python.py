#!/usr/bin/env python3
"""Run shared golden cases through Python reference and Rust adapter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO / "core" / "src"))

from trajectory_editor.core import sampling as reference  # noqa: E402
from trajectory_editor.core.policy_calculations import PolicyCalculations  # noqa: E402
from trajectory_editor.core.sampler_config import SamplerConfig  # noqa: E402
import rust_sampler as rust  # noqa: E402


def close(actual, expected, tolerance, context):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=0.0,
        atol=tolerance,
        err_msg=context,
    )


def distribution(case, module):
    return module.SparseDistribution(
        np.asarray(case["ids"], dtype=np.int64),
        np.asarray(case["probabilities"], dtype=np.float64),
        None if case["scores"] is None else np.asarray(case["scores"], dtype=np.float64),
    )


def rank_options(case):
    value = case["options"].get("candidate_model_ranks")
    return None if value is None else np.asarray(value, dtype=np.int64)


def bias_config(case):
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


def bias_contribution_record(item):
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


def reference_draw(case, value):
    options = dict(case["options"])
    options["candidate_model_ranks"] = rank_options(case)
    return value.draw_token(
        distribution(case, value),
        seed=case["seed"],
        stream_fingerprint=case["fingerprint"],
        aligned_step=int(case["boundary"]),
        kernel=case["kernel"],
        **options,
    )


def ranking(case, module):
    value = distribution(case, module)
    options = case["options"]
    if case["kernel"] == "gumbel-max":
        return module.gumbel_ranking_scores(
            value,
            seed=case["seed"],
            stream_fingerprint=case["fingerprint"],
            aligned_step=int(case["boundary"]),
            noise_address=options.get("gumbel_noise_address", "token-id"),
            candidate_model_ranks=rank_options(case),
            gumbel_noise_scale=options.get("gumbel_noise_scale", 1.0),
        )
    if case["kernel"] == "gaussian-max":
        return module.gaussian_ranking_scores(
            value,
            seed=case["seed"],
            stream_fingerprint=case["fingerprint"],
            aligned_step=int(case["boundary"]),
            noise_std=options.get("gaussian_noise_std", 1.0),
        )
    if case["kernel"] in module.PERTURB_MAX_KERNELS:
        return module.perturbation_ranking_scores(
            value,
            seed=case["seed"],
            stream_fingerprint=case["fingerprint"],
            aligned_step=int(case["boundary"]),
            kernel=case["kernel"],
            noise_std=options.get("perturb_noise_std", 1.0),
            student_t_df=options.get("student_t_df", 3.0),
        )
    return None


def main() -> int:
    fixture_path = ROOT / "fixtures" / "sampling-cases.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    tolerance = fixture["float_tolerance"]
    checks = 0

    assert rust.RNG_SCHEME == reference.RNG_SCHEME == fixture["rng_scheme"]
    for case in fixture["rng_cases"]:
        py_value = reference.position_uniform(
            case["seed"], case["fingerprint"], int(case["boundary"])
        )
        rs_value = rust.position_uniform(
            case["seed"], case["fingerprint"], int(case["boundary"])
        )
        assert py_value == case["expected"]
        assert rs_value == case["expected"]
        checks += 1
    for case in fixture["token_uniform_cases"]:
        py_value = reference.position_uniform_token(
            case["seed"], case["fingerprint"], int(case["boundary"]), int(case["token_id"])
        )
        rs_value = rust.position_uniform_token(
            case["seed"], case["fingerprint"], int(case["boundary"]), int(case["token_id"])
        )
        assert py_value == case["expected"] == rs_value
        checks += 1

    rank_case = fixture["rank_case"]
    logits = np.asarray(rank_case["logits"], dtype=np.float64)
    for token_id, expected in enumerate(rank_case["raw_ranks"]):
        assert reference.raw_rank(logits, token_id) == expected
        assert rust.raw_rank(logits, token_id) == expected
        checks += 1
    assert reference.top_raw_ids(logits, 3) == rank_case["top_ids_3"]
    assert rust.top_raw_ids(logits, 3) == rank_case["top_ids_3"]
    checks += 1

    for case in fixture["history_penalty_cases"]:
        logits = np.asarray(case["logits"], dtype=np.float64)
        original_logits = logits.copy()
        history = case["history_token_ids"]
        config = SamplerConfig(**case["config"])
        py_policy = PolicyCalculations(logits, config, history)
        rs_adjusted = rust.apply_history_penalties(logits, history, config)
        close(rs_adjusted, py_policy.adjusted, tolerance, case["name"])
        close(
            rs_adjusted,
            case["expected"]["adjusted_logits"],
            tolerance,
            f"{case['name']} fixture",
        )
        np.testing.assert_array_equal(logits, original_logits)
        for token_id in range(len(logits)):
            raw_rank = case["expected"]["raw_ranks"][token_id]
            policy_rank = case["expected"]["policy_ranks"][token_id]
            assert py_policy.raw_rank(token_id) == rust.raw_rank(logits, token_id) == raw_rank
            assert py_policy.policy_rank(token_id) == rust.raw_rank(rs_adjusted, token_id) == policy_rank
            checks += 2
        checks += 2

    for case in fixture["bias_cases"]:
        config = bias_config(case)
        logits = np.asarray(case["logits"], dtype=np.float64)
        original_logits = logits.copy()
        history = case["history_token_ids"]
        py_policy = PolicyCalculations(logits, config, history)
        py_contributions = config.bias_contributions(
            history, include_inactive=case["include_inactive"]
        )
        rs_contributions = rust.bias_contributions(
            history, config, include_inactive=case["include_inactive"]
        )
        assert rs_contributions == py_contributions, case["name"]
        assert [bias_contribution_record(item) for item in rs_contributions] == case[
            "expected"
        ]["contributions"], case["name"]
        checks += len(rs_contributions)

        py_active = config.active_biases(history)
        rs_active = rust.active_biases(history, config)
        assert rs_active == py_active
        assert sorted(rs_active.items()) == [
            (item[0], item[1]) for item in case["expected"]["active_biases"]
        ]
        checks += len(rs_active)

        rs_adjusted = rust.apply_biases(logits, history, config)
        close(rs_adjusted, py_policy.adjusted, tolerance, case["name"])
        close(
            rs_adjusted,
            case["expected"]["adjusted_logits"],
            tolerance,
            f"{case['name']} fixture",
        )
        np.testing.assert_array_equal(logits, original_logits)
        for token_id in range(len(logits)):
            raw_rank = case["expected"]["raw_ranks"][token_id]
            policy_rank = case["expected"]["policy_ranks"][token_id]
            assert py_policy.raw_rank(token_id) == rust.raw_rank(logits, token_id) == raw_rank
            assert py_policy.policy_rank(token_id) == rust.raw_rank(rs_adjusted, token_id) == policy_rank
            checks += 2
        checks += 2

    for case in fixture["adjustment_cases"]:
        logits = np.asarray(case["logits"], dtype=np.float64)
        original_logits = logits.copy()
        history = case["history_token_ids"]
        config = SamplerConfig.from_record(case["config"])
        activation = case["activation_adjustments"]
        ephemeral = {int(token): amount for token, amount in case["ephemeral_biases"].items()}
        py_policy = PolicyCalculations(
            logits,
            config,
            history,
            activation_logit_adjustments=activation,
            ephemeral_logit_biases=ephemeral,
        )
        rs_adjusted = rust.apply_history_penalties(logits, history, config)
        if activation is not None:
            rs_adjusted = rust.apply_activation_adjustments(
                rs_adjusted,
                np.asarray(activation, dtype=np.float64),
                config.activation_vector_strength,
            )
        rs_adjusted = rust.apply_biases(rs_adjusted, history, config)
        rs_adjusted = rust.apply_ephemeral_biases(rs_adjusted, ephemeral)
        close(rs_adjusted, py_policy.adjusted, tolerance, case["name"])
        close(rs_adjusted, case["expected"]["adjusted_logits"], tolerance, case["name"])
        np.testing.assert_array_equal(logits, original_logits)
        for token_id in range(len(logits)):
            assert py_policy.raw_rank(token_id) == case["expected"]["raw_ranks"][token_id]
            assert py_policy.raw_rank(token_id) == rust.raw_rank(logits, token_id)
            assert py_policy.policy_rank(token_id) == case["expected"]["policy_ranks"][token_id]
            assert py_policy.policy_rank(token_id) == rust.raw_rank(rs_adjusted, token_id)
            checks += 4
        checks += 2

    for case in fixture["activation_kernel_cases"]:
        logits = np.asarray(case["logits"], dtype=np.float64)
        original_logits = logits.copy()
        actual = rust.apply_activation_adjustments(
            logits,
            np.asarray(case["adjustments"], dtype=np.float64),
            case["strength"],
        )
        close(actual, case["expected"], tolerance, case["name"])
        np.testing.assert_array_equal(logits, original_logits)
        checks += len(logits) + 1

    for case in fixture["cfg_cases"]:
        conditional = np.asarray(case["conditional_logits"], dtype=np.float64)
        unconditional = np.asarray(case["unconditional_logits"], dtype=np.float64)
        expected = unconditional + case["scale"] * (conditional - unconditional)
        assert expected.tolist() == case["expected"]
        actual = rust.cfg_combine_logits(conditional, unconditional, case["scale"])
        close(actual, expected, tolerance, case["name"])
        checks += len(actual)

    for case in fixture["metric_cases"]:
        logits = np.asarray(case["logits"], dtype=np.float64)
        config = SamplerConfig.from_record(case["config"])
        py_policy = PolicyCalculations(logits, config, case["history_token_ids"])
        policy_logits = None if py_policy._policy_shares_raw else py_policy.adjusted
        rs_policy = rust.PolicyMetrics(logits, policy_logits)
        assert not rs_policy.raw_logsumexp_ready
        assert not rs_policy.logit_mean_std_ready
        close(rs_policy.maximum, case["expected"]["maximum"], tolerance, case["name"])
        assert rs_policy.top_raw_ids(len(case["expected"]["top_raw_ids"])) == case["expected"]["top_raw_ids"]
        assert rs_policy.top_policy_ids(len(case["expected"]["top_policy_ids"])) == case["expected"]["top_policy_ids"]
        assert not rs_policy.raw_logsumexp_ready
        assert not rs_policy.logit_mean_std_ready
        token_ids = list(range(len(logits)))
        assert [rs_policy.raw_rank(token_id) for token_id in token_ids] == case["expected"]["raw_ranks"]
        assert [rs_policy.policy_rank(token_id) for token_id in token_ids] == case["expected"]["policy_ranks"]
        selected = case["selected_ids"]
        rs_z = rs_policy.logit_z_scores(selected)
        py_z = py_policy.logit_z_scores(selected)
        for index, (actual, expected) in enumerate(zip(rs_z, py_z)):
            if expected is None:
                assert actual is None
            else:
                close(actual, expected, tolerance, f"{case['name']} z-score {index}")
        assert not rs_policy.raw_logsumexp_ready
        assert rs_policy.logit_mean_std_ready
        for index, (actual, expected) in enumerate(
            zip(rs_z, case["expected"]["logit_z_scores"])
        ):
            if expected is None:
                assert actual is None
            else:
                close(actual, expected, tolerance, f"{case['name']} golden z-score {index}")
        close(rs_policy.raw_probabilities(selected), py_policy.raw_probabilities(selected), tolerance, case["name"])
        close(rs_policy.raw_probabilities(selected), case["expected"]["raw_probabilities"], tolerance, case["name"])
        close(rs_policy.policy_probabilities_at(selected), py_policy.policy_probabilities_at(selected), tolerance, case["name"])
        close(rs_policy.policy_probabilities_at(selected), case["expected"]["policy_probabilities"], tolerance, case["name"])
        close([rs_policy.raw_nll(token_id) for token_id in token_ids], [py_policy.raw_nll(token_id) for token_id in token_ids], tolerance, case["name"])
        close(rs_policy.log_z, py_policy.log_z, tolerance, case["name"])
        close(rs_policy.log_z, case["expected"]["log_z"], tolerance, case["name"])
        close(rs_policy.denominator, py_policy.denominator, tolerance, case["name"])
        close(rs_policy.denominator, case["expected"]["denominator"], tolerance, case["name"])
        assert rs_policy.raw_logsumexp_ready
        checks += (
            4 * len(token_ids)
            + 2 * len(case["expected"]["top_raw_ids"])
            + 4 * len(selected)
            + 5
        )

    for case in fixture["filter_cases"]:
        config = type("Config", (), case["config"])()
        py_result = reference.apply_candidate_filter(np.asarray(case["logits"]), config)
        rs_result = rust.apply_candidate_filter(np.asarray(case["logits"]), config)
        close(rs_result.scaled_logits, py_result.scaled_logits, tolerance, case["name"])
        close(rs_result.scaled_logits, case["expected"]["scaled_logits"], tolerance, case["name"])
        for name, py_ids in py_result.stages.items():
            rs_ids = rs_result.stages[name]
            golden_ids = case["expected"]["stages"][name]
            assert (rs_ids is None) == (py_ids is None), (case["name"], name)
            if py_ids is not None:
                assert rs_ids.tolist() == py_ids.tolist() == golden_ids, (case["name"], name)
                checks += 1
        assert rs_result.diagnostics == py_result.diagnostics == case["expected"]["diagnostics"]

    order_scores = []
    for case in fixture["draw_cases"]:
        py_dist = distribution(case, reference)
        rs_dist = distribution(case, rust)
        py_ranking = ranking(case, reference)
        rs_ranking = ranking(case, rust)
        expected_ranking = case["expected"]["ranking_scores"]
        if py_ranking is None:
            assert expected_ranking is None
        else:
            close(py_ranking, rs_ranking, tolerance, case["name"])
            close(py_ranking, expected_ranking, tolerance, f"{case['name']} Python fixture")
            if case["kernel"] == "gumbel-max":
                py_winner = reference.gumbel_winner(py_dist, py_ranking)
                rs_winner = rust.gumbel_winner(rs_dist, rs_ranking)
                py_ranked = reference.gumbel_ranked_ids(
                    py_dist,
                    seed=case["seed"],
                    stream_fingerprint=case["fingerprint"],
                    aligned_step=int(case["boundary"]),
                    noise_address=case["options"].get("gumbel_noise_address", "token-id"),
                    candidate_model_ranks=rank_options(case),
                    gumbel_noise_scale=case["options"].get("gumbel_noise_scale", 1.0),
                )
                rs_ranked = rust.gumbel_ranked_ids(
                    rs_dist,
                    seed=case["seed"],
                    stream_fingerprint=case["fingerprint"],
                    aligned_step=int(case["boundary"]),
                    noise_address=case["options"].get("gumbel_noise_address", "token-id"),
                    candidate_model_ranks=rank_options(case),
                    gumbel_noise_scale=case["options"].get("gumbel_noise_scale", 1.0),
                )
                assert py_winner == rs_winner == case["expected"]["winner"], case["name"]
                assert py_ranked.tolist() == rs_ranked.tolist() == case["expected"]["ranked_ids"]
                checks += len(py_ranked) + 1
            elif case["kernel"] == "gaussian-max":
                py_winner = reference.gaussian_winner(py_dist, py_ranking)
                rs_winner = rust.gaussian_winner(rs_dist, rs_ranking)
                assert py_winner == rs_winner == case["expected"]["winner"], case["name"]
                checks += 1
            else:
                py_winner = reference.perturbation_winner(py_dist, py_ranking)
                rs_winner = rust.perturbation_winner(rs_dist, rs_ranking)
                assert py_winner == rs_winner == case["expected"]["winner"], case["name"]
                checks += 1
            if case["kernel"] == "gumbel-max" and case["name"].startswith("gumbel-token-order-"):
                order_scores.append(dict(zip(case["ids"], rs_ranking)))
            checks += len(py_ranking)
        py_draw = reference_draw(case, reference)
        rs_options = dict(case["options"])
        rs_options["candidate_model_ranks"] = rank_options(case)
        rs_draw = rust.draw_token(
            rs_dist,
            seed=case["seed"],
            stream_fingerprint=case["fingerprint"],
            aligned_step=int(case["boundary"]),
            kernel=case["kernel"],
            **rs_options,
        )
        assert py_draw == rs_draw == case["expected"]["draw"], case["name"]
        checks += 1
    assert len(order_scores) == 2
    for token_id in order_scores[0]:
        assert abs(order_scores[0][token_id] - order_scores[1][token_id]) <= tolerance

    case = fixture["conditional_case"]
    common = dict(
        count=case["count"],
        parent_score=case["parent_score"],
        parent_log_probability=case["parent_log_probability"],
        seed=case["seed"],
        stream_fingerprint=case["fingerprint"],
        aligned_step=int(case["boundary"]),
        prefix_token_ids=case["prefix_token_ids"],
    )
    py_children = reference.conditional_gumbel_top_k(
        np.asarray(case["log_probabilities"], dtype=np.float64), **common
    )
    rs_children = rust.conditional_gumbel_top_k(
        np.asarray(case["log_probabilities"], dtype=np.float64), **common
    )
    assert [child for child, _ in rs_children] == [child for child, _ in py_children]
    assert [child for child, _ in rs_children] == [child for child, _ in case["expected"]]
    close([score for _, score in rs_children], [score for _, score in py_children], tolerance, "conditional Gumbel")
    checks += len(rs_children)

    case = fixture["seed_search_case"]
    py_dist = reference.SparseDistribution(
        np.asarray(case["ids"], dtype=np.int64),
        np.asarray(case["probabilities"], dtype=np.float64),
        np.asarray(case["scores"], dtype=np.float64),
    )
    rs_dist = rust.SparseDistribution(
        np.asarray(case["ids"], dtype=np.int64),
        np.asarray(case["probabilities"], dtype=np.float64),
        np.asarray(case["scores"], dtype=np.float64),
    )
    py_iter = iter(case["candidate_seeds"])
    rs_iter = iter(case["candidate_seeds"])
    py_result = reference.find_seed_for_token(
        py_dist,
        case["target"],
        current_seed=case["current_seed"],
        stream_fingerprint=case["fingerprint"],
        aligned_step=int(case["boundary"]),
        kernel=case["kernel"],
        next_seed=lambda: next(py_iter),
    )
    rs_result = rust.find_seed_for_token(
        rs_dist,
        case["target"],
        current_seed=case["current_seed"],
        stream_fingerprint=case["fingerprint"],
        aligned_step=int(case["boundary"]),
        kernel=case["kernel"],
        next_seed=lambda: next(rs_iter),
    )
    expected = (case["expected"]["seed"], case["expected"]["checked"])
    assert py_result == rs_result == expected
    checks += 1

    print(
        f"Rust/Python parity passed: {checks} value checks across "
        f"{len(fixture['draw_cases'])} draw cases and "
        f"{len(fixture['history_penalty_cases'])} history-penalty cases and "
        f"{len(fixture['bias_cases'])} bias cases, "
        f"{len(fixture['adjustment_cases'])} ordered policy-adjustment cases, "
        f"{len(fixture['activation_kernel_cases'])} activation-kernel cases, "
        f"{len(fixture['cfg_cases'])} CFG cases, and "
        f"{len(fixture['metric_cases'])} lazy-metric cases"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

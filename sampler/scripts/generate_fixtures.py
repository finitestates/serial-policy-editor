#!/usr/bin/env python3
"""Refresh checked-in expected values from the authoritative Python sampler."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from trajectory_editor.bias_groups import BiasGroup, BiasMember, BiasRoute, BiasToken
from trajectory_editor.core import sampling
from trajectory_editor.core.policy_calculations import PolicyCalculations
from trajectory_editor.core.sampler_config import SamplerConfig


ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64


def as_list(values):
    return None if values is None else np.asarray(values).tolist()


def filter_case(name, logits, **config):
    defaults = dict(
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        min_p=0.0,
        typical_p=1.0,
        tail_free_z=1.0,
    )
    defaults.update(config)
    result = sampling.apply_candidate_filter(np.asarray(logits, dtype=np.float64), SimpleNamespace(**defaults))
    return {
        "name": name,
        "logits": logits,
        "config": defaults,
        "expected": {
            "scaled_logits": as_list(result.scaled_logits),
            "stages": {key: as_list(value) for key, value in result.stages.items()},
            "diagnostics": result.diagnostics,
        },
    }


def draw_case(name, ids, probabilities, scores, *, seed=17, fingerprint=FINGERPRINT_A,
              boundary=3, kernel="categorical", **options):
    distribution = sampling.SparseDistribution(
        np.asarray(ids, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        None if scores is None else np.asarray(scores, dtype=np.float64),
    )
    args = {
        "seed": seed,
        "stream_fingerprint": fingerprint,
        "aligned_step": boundary,
        "kernel": kernel,
        **options,
    }
    ranking = None
    if kernel == "gumbel-max":
        ranking = sampling.gumbel_ranking_scores(
            distribution,
            seed=seed,
            stream_fingerprint=fingerprint,
            aligned_step=boundary,
            noise_address=options.get("gumbel_noise_address", "token-id"),
            candidate_model_ranks=options.get("candidate_model_ranks"),
            gumbel_noise_scale=options.get("gumbel_noise_scale", 1.0),
        )
    elif kernel == "gaussian-max":
        ranking = sampling.gaussian_ranking_scores(
            distribution,
            seed=seed,
            stream_fingerprint=fingerprint,
            aligned_step=boundary,
            noise_std=options.get("gaussian_noise_std", 1.0),
        )
    elif kernel in sampling.PERTURB_MAX_KERNELS:
        ranking = sampling.perturbation_ranking_scores(
            distribution,
            seed=seed,
            stream_fingerprint=fingerprint,
            aligned_step=boundary,
            kernel=kernel,
            noise_std=options.get("perturb_noise_std", 1.0),
            student_t_df=options.get("student_t_df", 3.0),
        )
    result = {
        "name": name,
        "ids": ids,
        "probabilities": probabilities,
        "scores": scores,
        "seed": seed,
        "fingerprint": fingerprint,
        "boundary": str(boundary),
        "kernel": kernel,
        "options": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in options.items()
        },
        "expected": {
            "ranking_scores": as_list(ranking),
            "draw": sampling.draw_token(distribution, **args),
        },
    }
    if ranking is not None:
        if kernel == "gumbel-max":
            result["expected"]["winner"] = sampling.gumbel_winner(distribution, ranking)
            result["expected"]["ranked_ids"] = sampling.gumbel_ranked_ids(
                distribution,
                seed=seed,
                stream_fingerprint=fingerprint,
                aligned_step=boundary,
                noise_address=options.get("gumbel_noise_address", "token-id"),
                candidate_model_ranks=options.get("candidate_model_ranks"),
                gumbel_noise_scale=options.get("gumbel_noise_scale", 1.0),
            ).tolist()
        elif kernel == "gaussian-max":
            result["expected"]["winner"] = sampling.gaussian_winner(distribution, ranking)
        else:
            result["expected"]["winner"] = sampling.perturbation_winner(distribution, ranking)
    return result


def history_penalty_case(name, logits, history_token_ids, **penalties):
    config = {
        "temperature": 1.0,
        "top_k": None,
        "top_p": 1.0,
        "min_p": 0.0,
        "typical_p": 1.0,
        "tail_free_z": 1.0,
        "repeat_penalty": 1.0,
        "repeat_last_n": -1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    config.update(penalties)
    values = np.asarray(logits, dtype=np.float64)
    history = list(history_token_ids)
    policy = PolicyCalculations(values, SamplerConfig(**config), history)
    return {
        "name": name,
        "logits": logits,
        "history_token_ids": history,
        "config": config,
        "expected": {
            "adjusted_logits": as_list(policy.adjusted),
            "raw_ranks": [policy.raw_rank(token_id) for token_id in range(len(values))],
            "policy_ranks": [
                policy.policy_rank(token_id) for token_id in range(len(values))
            ],
        },
    }


def bias_contribution_record(item):
    return {
        "token_id": item.token_id,
        "amount": item.amount,
        "active": item.active,
        "source": item.source,
        "group_name": item.group_name,
        "member_routes": [
            {
                "member_text": member_route.member_text,
                "member_literal": member_route.member_literal,
                "route": member_route.route.to_dict(),
                "active": member_route.active,
            }
            for member_route in item.member_routes
        ],
    }


def bias_case(name, logits, history_token_ids, bias_groups, token_biases,
              *, include_inactive=True):
    config = SamplerConfig(
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        min_p=0.0,
        typical_p=1.0,
        tail_free_z=1.0,
        repeat_last_n=0,
        bias_groups=bias_groups,
        token_biases=token_biases,
    )
    values = np.asarray(logits, dtype=np.float64)
    history = list(history_token_ids)
    policy = PolicyCalculations(values, config, history)
    return {
        "name": name,
        "logits": logits,
        "history_token_ids": history,
        "include_inactive": include_inactive,
        "bias_groups": [group.to_dict() for group in config.bias_groups],
        "token_biases": [token.to_dict() for token in config.token_biases],
        "expected": {
            "active_biases": [
                [int(token_id), float(amount)]
                for token_id, amount in sorted(config.active_biases(history).items())
            ],
            "contributions": [
                bias_contribution_record(item)
                for item in config.bias_contributions(
                    history, include_inactive=include_inactive
                )
            ],
            "adjusted_logits": as_list(policy.adjusted),
            "raw_ranks": [policy.raw_rank(token_id) for token_id in range(len(values))],
            "policy_ranks": [
                policy.policy_rank(token_id) for token_id in range(len(values))
            ],
        },
    }


def adjustment_case(name, logits, history_token_ids, config, activation_adjustments,
                    ephemeral_biases):
    values = np.asarray(logits, dtype=np.float64)
    history = list(history_token_ids)
    policy = PolicyCalculations(
        values,
        config,
        history,
        activation_logit_adjustments=activation_adjustments,
        ephemeral_logit_biases=ephemeral_biases,
    )
    return {
        "name": name,
        "logits": logits,
        "history_token_ids": history,
        "activation_adjustments": activation_adjustments,
        "ephemeral_biases": {str(token): amount for token, amount in ephemeral_biases.items()},
        "config": config.to_dict(),
        "expected": {
            "adjusted_logits": as_list(policy.adjusted),
            "raw_ranks": [policy.raw_rank(token_id) for token_id in range(len(values))],
            "policy_ranks": [
                policy.policy_rank(token_id) for token_id in range(len(values))
            ],
        },
    }


def metrics_case(name, logits, config=None, history_token_ids=None):
    values = np.asarray(logits, dtype=np.float64)
    config = config or SamplerConfig(
        temperature=1.0, top_k=None, top_p=1.0, min_p=0.0
    )
    history = [] if history_token_ids is None else list(history_token_ids)
    policy = PolicyCalculations(
        values,
        config,
        history,
    )
    policy_values = None if policy._policy_shares_raw else as_list(policy.adjusted)
    token_ids = list(range(len(values)))
    selected = token_ids[::2]
    result = {
        "name": name,
        "logits": logits,
        "policy_logits": policy_values,
        "history_token_ids": history,
        "config": config.to_dict(),
        "selected_ids": selected,
        "expected": {
            "maximum": policy.maximum,
            "raw_probabilities": as_list(policy.raw_probabilities(selected)),
            "raw_nll": [policy.raw_nll(token_id) for token_id in token_ids],
            "denominator": policy.denominator,
            "raw_ranks": [policy.raw_rank(token_id) for token_id in token_ids],
            "policy_ranks": [policy.policy_rank(token_id) for token_id in token_ids],
            "top_raw_ids": policy.top_raw_ids(min(3, len(values))),
            "top_policy_ids": policy.top_policy_ids(min(3, len(values))),
            "policy_probabilities": as_list(policy.policy_probabilities_at(selected)),
            "logit_z_scores": policy.logit_z_scores(selected),
            "log_z": policy.log_z,
        },
    }
    return result


def cfg_case(name, conditional_logits, unconditional_logits, scale):
    conditional = np.asarray(conditional_logits, dtype=np.float64)
    unconditional = np.asarray(unconditional_logits, dtype=np.float64)
    expected = unconditional + scale * (conditional - unconditional)
    return {
        "name": name,
        "conditional_logits": conditional_logits,
        "unconditional_logits": unconditional_logits,
        "scale": scale,
        "expected": as_list(expected),
    }


class BiasFixtureBackend:
    """Small tokenizer stand-in used only while Python compiles test routes."""

    _surfaces = {
        "item": (1, 10),
        " item": (2, 10),
        "Item": (3, 11),
        " Item": (4, 11),
        "ITEM": (5, 12),
        " ITEM": (6, 12),
    }

    def tokenize(self, text, *, add_bos, special):
        assert add_bos is False and special is False
        return list(self._surfaces[text])

    def vocabulary_size(self):
        return 32

    def is_eog(self, token_id):
        return False


def main():
    rng_cases = [
        {"seed": sampling.MIN_SEED, "fingerprint": FINGERPRINT_A, "boundary": "0"},
        {"seed": sampling.MAX_SEED, "fingerprint": FINGERPRINT_B, "boundary": "18446744073709551616"},
        {"seed": 0, "fingerprint": "0123456789abcdef" * 4, "boundary": "9223372036854775808"},
    ]
    for case in rng_cases:
        case["expected"] = sampling.position_uniform(
            case["seed"], case["fingerprint"], int(case["boundary"])
        )

    token_uniform_cases = [
        {"seed": 17, "fingerprint": FINGERPRINT_A, "boundary": "3", "token_id": 0},
        {"seed": sampling.MIN_SEED, "fingerprint": FINGERPRINT_B, "boundary": "18446744073709551616", "token_id": 1208925819614629174706176},
    ]
    for case in token_uniform_cases:
        case["expected"] = sampling.position_uniform_token(
            case["seed"], case["fingerprint"], int(case["boundary"]), case["token_id"]
        )
        case["token_id"] = str(case["token_id"])

    logits = [2.0, 2.0, 1.0, -0.5, -0.5, -4.0]
    filters = [
        filter_case("greedy", logits, temperature=0.0, top_k=4, top_p=0.3, min_p=0.8),
        filter_case("neutral", logits),
        filter_case(
            "all-stages",
            [6.0, 4.0, 3.0, 2.0, 0.0, -1.0],
            temperature=0.7,
            top_k=6,
            typical_p=0.72,
            tail_free_z=0.85,
            top_p=0.8,
            min_p=0.03,
        ),
        filter_case("typical-only", [5.0, 3.0, 1.0, 0.0, -2.0], typical_p=0.55),
        filter_case("tail-free-only", [6.0, 2.5, 1.0, 0.2, -0.3, -1.0], tail_free_z=0.4),
        filter_case("top-p-min-p", [4.0, 3.0, 2.0, 1.0, 0.0], top_p=0.88, min_p=0.09),
    ]

    draws = [
        draw_case("categorical", [4, 1, 7], [0.2, 0.5, 0.3], [0.1, 0.9, 0.4]),
        draw_case("gumbel-token-order-a", [4, 1, 7], [0.2, 0.5, 0.3], [0.1, 0.9, 0.4], kernel="gumbel-max"),
        draw_case("gumbel-token-order-b", [7, 4, 1], [0.3, 0.2, 0.5], [0.4, 0.1, 0.9], kernel="gumbel-max"),
        draw_case("gumbel-model-rank", [4, 1, 7], [0.2, 0.5, 0.3], [0.1, 0.9, 0.4], kernel="gumbel-max", gumbel_noise_address="model-rank", candidate_model_ranks=np.asarray([8, 2, 5], dtype=np.int64)),
        draw_case("gumbel-zero-scale", [4, 1, 7], [0.2, 0.5, 0.3], [0.1, 0.9, 0.4], kernel="gumbel-max", gumbel_noise_scale=0.0),
        draw_case("gaussian-zero-noise", [4, 1, 7], [0.2, 0.5, 0.3], [0.1, 0.9, 0.4], kernel="gaussian-max", gaussian_noise_std=0.0),
        draw_case("gaussian-noise", [4, 1, 7], [0.2, 0.5, 0.3], [0.1, 0.9, 0.4], kernel="gaussian-max", gaussian_noise_std=0.75),
        draw_case("logistic", [4, 1, 7], [0.2, 0.5, 0.3], [0.1, 0.9, 0.4], kernel="logistic-max"),
        draw_case("student-t-df3", [4, 1, 7], [0.2, 0.5, 0.3], [1.0, 0.0, -2.0], seed=19, boundary=4, kernel="student-t-max"),
        draw_case("student-t-df1", [4, 1, 7], [0.2, 0.5, 0.3], [1.0, 0.0, -2.0], seed=19, boundary=4, kernel="student-t-max", student_t_df=1.0),
        draw_case("student-t-df-half", [4, 1, 7], [0.2, 0.5, 0.3], [1.0, 0.0, -2.0], seed=19, boundary=4, kernel="student-t-max", student_t_df=0.5),
        draw_case("laplace", [4, 1, 7], [0.2, 0.5, 0.3], [0.1, 0.9, 0.4], kernel="laplace-max"),
        draw_case("uniform", [4, 1, 7], [0.2, 0.5, 0.3], [0.1, 0.9, 0.4], kernel="uniform-max"),
        draw_case("duplicate-token-ids", [3, 1, 3], [0.25, 0.25, 0.5], [0.5, 0.5, 0.7], kernel="gumbel-max"),
    ]

    history_penalty_cases = [
        history_penalty_case(
            "empty-history",
            [2.0, 0.0, -2.0, 1.0],
            [],
            repeat_penalty=2.0,
            presence_penalty=0.5,
            frequency_penalty=0.25,
        ),
        history_penalty_case(
            "all-history-repeat-presence-frequency",
            [2.0, 0.0, -2.0, 1.0],
            [0, 0, 1, 2],
            repeat_penalty=2.0,
            repeat_last_n=-1,
            presence_penalty=0.5,
            frequency_penalty=0.25,
        ),
        history_penalty_case(
            "zero-window-is-inactive",
            [2.0, 0.0, -2.0, 1.0],
            [0, 1, 1],
            repeat_penalty=0.5,
            repeat_last_n=0,
            presence_penalty=1.0,
            frequency_penalty=0.75,
        ),
        history_penalty_case(
            "short-positive-tail-repeated-id-and-outside-tail",
            [2.0, 0.0, -2.0, -1.0],
            [0, 1, 1, 2],
            repeat_penalty=1.5,
            repeat_last_n=3,
            presence_penalty=0.4,
            frequency_penalty=0.3,
        ),
        history_penalty_case(
            "positive-tail-longer-than-history-negative-penalties",
            [2.0, 0.0, -2.0, -1.0],
            [0, 0, 1],
            repeat_penalty=0.5,
            repeat_last_n=9,
            presence_penalty=-0.5,
            frequency_penalty=-0.25,
        ),
        history_penalty_case(
            "repeat-off-presence-and-frequency-on",
            [2.0, 0.0, -2.0],
            [0, 1, 2, 2],
            repeat_penalty=1.0,
            repeat_last_n=-1,
            presence_penalty=0.25,
            frequency_penalty=0.5,
        ),
    ]

    bias_fixture_backend = BiasFixtureBackend()
    compiled_surface_member = BiasMember.compile("item", bias_fixture_backend)
    bias_cases = [
        bias_case(
            "overlapping-direct-and-grouped-routes",
            [4.0, 3.0, 2.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1, 2],
            [
                BiasGroup(
                    "nautical",
                    (
                        BiasMember(
                            "steamship",
                            (BiasRoute((1, 2, 3), ("steamship",)),),
                        ),
                        BiasMember(
                            "steamship suffix",
                            (BiasRoute((2, 3), ("steamship suffix",)),),
                        ),
                        BiasMember(
                            "inactive route",
                            (BiasRoute((9, 3), ("inactive route",)),),
                        ),
                    ),
                    2.5,
                ),
                BiasGroup(
                    "ships",
                    (BiasMember("steamship", (BiasRoute((1, 2, 3), ("steamship",)),)),),
                    2.25,
                ),
                BiasGroup(
                    "spare",
                    (
                        BiasMember("quiet phrase", (BiasRoute((1, 4), ("quiet phrase",)),)),
                        BiasMember("single token", (BiasRoute((5,), ("single token",)),)),
                    ),
                    0.5,
                ),
            ],
            [BiasToken(3, 1.25)],
        ),
        bias_case(
            "exact-phrase-prefix-and-inactive-routes",
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [1],
            [
                BiasGroup(
                    "phrases",
                    (
                        BiasMember("full phrase", (BiasRoute((1, 2, 4), ("full phrase",)),)),
                        BiasMember("wrong final prefix", (BiasRoute((2, 5), ("wrong final prefix",)),)),
                        BiasMember("one-token phrase", (BiasRoute((1, 6), ("one-token phrase",)),)),
                        BiasMember("unconditional token", (BiasRoute((7,), ("unconditional token",)),)),
                    ),
                    0.75,
                ),
            ],
            [BiasToken(0, 0.25)],
        ),
        bias_case(
            "python-compiled-case-and-spacing-variants",
            [0.0] * 16,
            [2],
            [BiasGroup("surface-forms", (compiled_surface_member,), 0.75)],
            [],
        ),
    ]

    adjustment_config = SamplerConfig(
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        min_p=0.0,
        repeat_penalty=2.0,
        repeat_last_n=-1,
        presence_penalty=0.5,
        frequency_penalty=0.25,
        activation_vector=(1.0,),
        activation_vector_strength=0.5,
        bias_groups=(
            BiasGroup(
                "phrase",
                (BiasMember("target", (BiasRoute((0, 1, 2), ("target",)),)),),
                0.75,
            ),
        ),
        token_biases=(BiasToken(3, 0.25),),
    )
    adjustment_cases = [
        adjustment_case(
            "ordered-history-activation-grouped-direct-ephemeral",
            [2.0, 1.0, 0.5, -1.0, 3.0],
            [0, 0, 1],
            adjustment_config,
            [0.0, 0.5, -0.5, 1.0, -1.0],
            {2: -0.75, 4: 0.5},
        ),
        adjustment_case(
            "inactive-activation-and-empty-ephemeral",
            [1.0, 0.0, -1.0],
            [1],
            SamplerConfig(
                temperature=1.0,
                top_k=None,
                top_p=1.0,
                min_p=0.0,
                activation_vector=(),
                activation_vector_strength=0.0,
            ),
            None,
            {},
        ),
    ]
    activation_kernel_cases = [
        {
            "name": "positive-strength",
            "logits": [1.0, -2.0, 3.0],
            "adjustments": [0.5, -1.0, 2.0],
            "strength": 0.25,
        },
        {
            "name": "negative-strength",
            "logits": [1.0, -2.0, 3.0],
            "adjustments": [0.5, -1.0, 2.0],
            "strength": -0.5,
        },
        {
            "name": "zero-strength",
            "logits": [1.0, -2.0, 3.0],
            "adjustments": [0.5, -1.0, 2.0],
            "strength": 0.0,
        },
    ]
    for case in activation_kernel_cases:
        case["expected"] = as_list(
            np.asarray(case["logits"], dtype=np.float64)
            + case["strength"]
            * np.asarray(case["adjustments"], dtype=np.float64)
        )

    cfg_cases = [
        cfg_case("cfg-scale-zero", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 0.0),
        cfg_case("cfg-scale-one", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 1.0),
        cfg_case("cfg-amplifies-conditional", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 2.0),
    ]

    metric_cases = [
        metrics_case("identity-lazy-metrics", [2.0, 1.0, 0.0, -1.0]),
        metrics_case(
            "history-adjusted-policy-metrics",
            [2.0, 1.0, 0.0, -1.0],
            SamplerConfig(
                temperature=1.0,
                top_k=None,
                top_p=1.0,
                min_p=0.0,
                repeat_penalty=2.0,
                repeat_last_n=-1,
                presence_penalty=0.5,
                frequency_penalty=0.25,
            ),
            [0, 0, 1],
        ),
        metrics_case("flat-logits-undefined-z", [5.0, 5.0, 5.0]),
    ]

    conditional = {
        "log_probabilities": [-0.1, -0.5, -1.3, -2.2],
        "count": 4,
        "parent_score": 0.25,
        "parent_log_probability": -0.7,
        "seed": sampling.MAX_SEED,
        "fingerprint": FINGERPRINT_B,
        "boundary": "1208925819614629174706176",
        "prefix_token_ids": [9, 2, 9],
    }
    conditional_result = sampling.conditional_gumbel_top_k(
        np.asarray(conditional["log_probabilities"], dtype=np.float64),
        count=conditional["count"],
        parent_score=conditional["parent_score"],
        parent_log_probability=conditional["parent_log_probability"],
        seed=conditional["seed"],
        stream_fingerprint=conditional["fingerprint"],
        aligned_step=int(conditional["boundary"]),
        prefix_token_ids=conditional["prefix_token_ids"],
    )
    conditional["expected"] = [[token_id, score] for token_id, score in conditional_result]

    search_distribution = sampling.SparseDistribution(
        np.asarray([1, 4, 7], dtype=np.int64),
        np.asarray([0.2, 0.5, 0.3], dtype=np.float64),
        np.asarray([0.1, 0.9, 0.4], dtype=np.float64),
    )
    search_draw = lambda seed: sampling.draw_token(
        search_distribution,
        seed=seed,
        stream_fingerprint=FINGERPRINT_A,
        aligned_step=3,
        kernel="categorical",
    )
    target = 4
    nonmatching = next(seed for seed in range(1000) if search_draw(seed) != target)
    matching = next(seed for seed in range(1000) if search_draw(seed) == target)
    seed_search = {
        "ids": [1, 4, 7],
        "probabilities": [0.2, 0.5, 0.3],
        "scores": [0.1, 0.9, 0.4],
        "target": target,
        "current_seed": 12345,
        "fingerprint": FINGERPRINT_A,
        "boundary": "3",
        "kernel": "categorical",
        "candidate_seeds": [12345, nonmatching, matching],
        "expected": {"seed": matching, "checked": 3},
    }

    fixture = {
        "rng_scheme": sampling.RNG_SCHEME,
        "rng_cases": rng_cases,
        "token_uniform_cases": token_uniform_cases,
        "rank_case": {
            "logits": [2.0, 2.0, -1.0, 2.0, 0.0],
            "raw_ranks": [1, 2, 5, 3, 4],
            "top_ids_3": [0, 1, 3],
        },
        "filter_cases": filters,
        "draw_cases": draws,
        "history_penalty_cases": history_penalty_cases,
        "bias_cases": bias_cases,
        "adjustment_cases": adjustment_cases,
        "activation_kernel_cases": activation_kernel_cases,
        "cfg_cases": cfg_cases,
        "metric_cases": metric_cases,
        "conditional_case": conditional,
        "seed_search_case": seed_search,
        "float_tolerance": 1e-14,
    }
    (ROOT / "fixtures" / "sampling-cases.json").write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

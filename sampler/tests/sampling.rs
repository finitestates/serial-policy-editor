use serde_json::Value;

use rust_sampler_native::{
    BiasGroup, BiasMember, BiasRoute, BiasToken, DrawOptions, PolicyMetrics, SparseDistribution,
    active_biases, apply_activation_adjustments, apply_biases, apply_ephemeral_biases,
    apply_filter, apply_history_penalties, bias_contributions, cfg_combine_logits,
    conditional_gumbel_top_k, draw_token, gaussian_ranking_scores, gumbel_ranking_scores,
    perturbation_ranking_scores, position_uniform, position_uniform_token, rank, ranking_ids,
    top_ids,
};

fn fixtures() -> Value {
    serde_json::from_str(include_str!("../fixtures/sampling-cases.json"))
        .expect("shared sampler fixtures should be valid JSON")
}

fn f64s(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("expected array")
        .iter()
        .map(|item| item.as_f64().expect("expected float"))
        .collect()
}

fn i64s(value: &Value) -> Vec<i64> {
    value
        .as_array()
        .expect("expected array")
        .iter()
        .map(|item| item.as_i64().expect("expected integer"))
        .collect()
}

fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .expect("expected array")
        .iter()
        .map(|item| item.as_str().expect("expected string").to_owned())
        .collect()
}

fn bias_inputs_from_parts(
    groups_value: &Value,
    tokens_value: &Value,
) -> (Vec<BiasGroup>, Vec<BiasToken>) {
    let groups = groups_value
        .as_array()
        .unwrap()
        .iter()
        .map(|group| BiasGroup {
            name: group["name"].as_str().unwrap().to_owned(),
            bias: group["bias"].as_f64().unwrap(),
            members: group["members"]
                .as_array()
                .unwrap()
                .iter()
                .map(|member| BiasMember {
                    text: member["text"].as_str().unwrap().to_owned(),
                    literal: member["literal"].as_bool().unwrap(),
                    routes: member["routes"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|route| BiasRoute {
                            token_ids: i64s(&route["token_ids"]),
                            surfaces: strings(&route["surfaces"]),
                        })
                        .collect(),
                })
                .collect(),
        })
        .collect();
    let token_biases = tokens_value
        .as_array()
        .unwrap()
        .iter()
        .map(|token| BiasToken {
            token_id: token["token_id"].as_i64().unwrap(),
            bias: token["bias"].as_f64().unwrap(),
        })
        .collect();
    (groups, token_biases)
}

fn bias_inputs(case: &Value) -> (Vec<BiasGroup>, Vec<BiasToken>, Vec<i64>) {
    let (groups, token_biases) =
        bias_inputs_from_parts(&case["bias_groups"], &case["token_biases"]);
    (groups, token_biases, i64s(&case["history_token_ids"]))
}

fn assert_float_slices(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "float at {index}: actual {actual:.17e}, expected {expected:.17e}, tolerance {tolerance:.1e}"
        );
    }
}

fn assert_bias_contributions(
    actual: &[rust_sampler_native::BiasContribution],
    expected: &Value,
    tolerance: f64,
) {
    let expected = expected.as_array().expect("expected contribution array");
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.token_id, expected["token_id"].as_i64().unwrap());
        assert!(
            (actual.amount - expected["amount"].as_f64().unwrap()).abs() <= tolerance,
            "contribution amount mismatch for {}",
            actual.source
        );
        assert_eq!(actual.active, expected["active"].as_bool().unwrap());
        assert_eq!(actual.source, expected["source"].as_str().unwrap());
        assert_eq!(
            actual.group_name.as_deref(),
            expected["group_name"].as_str()
        );
        let expected_routes = expected["member_routes"].as_array().unwrap();
        assert_eq!(actual.member_routes.len(), expected_routes.len());
        for (actual_route, expected_route) in actual.member_routes.iter().zip(expected_routes) {
            assert_eq!(
                actual_route.member_text,
                expected_route["member_text"].as_str().unwrap()
            );
            assert_eq!(
                actual_route.member_literal,
                expected_route["member_literal"].as_bool().unwrap()
            );
            assert_eq!(
                actual_route.active,
                expected_route["active"].as_bool().unwrap()
            );
            assert_eq!(
                actual_route.route.token_ids,
                i64s(&expected_route["route"]["token_ids"])
            );
            assert_eq!(
                actual_route.route.surfaces,
                strings(&expected_route["route"]["surfaces"])
            );
        }
    }
}

fn distribution(case: &Value) -> SparseDistribution {
    SparseDistribution {
        ids: i64s(&case["ids"]),
        probabilities: f64s(&case["probabilities"]),
        scores: case["scores"].as_array().map(|_| f64s(&case["scores"])),
    }
}

fn draw_options<'a>(case: &'a Value, ranks: Option<&'a [String]>) -> DrawOptions<'a> {
    let options = &case["options"];
    DrawOptions {
        seed: case["seed"].as_i64().unwrap(),
        fingerprint: case["fingerprint"].as_str().unwrap(),
        boundary: case["boundary"].as_str().unwrap(),
        kernel: case["kernel"].as_str().unwrap(),
        gaussian_noise_std: options
            .get("gaussian_noise_std")
            .and_then(Value::as_f64)
            .unwrap_or(1.0),
        perturb_noise_std: options
            .get("perturb_noise_std")
            .and_then(Value::as_f64)
            .unwrap_or(1.0),
        student_t_df: options
            .get("student_t_df")
            .and_then(Value::as_f64)
            .unwrap_or(3.0),
        gumbel_noise_address: options
            .get("gumbel_noise_address")
            .and_then(Value::as_str)
            .unwrap_or("token-id"),
        candidate_model_ranks: ranks,
        gumbel_noise_scale: options
            .get("gumbel_noise_scale")
            .and_then(Value::as_f64)
            .unwrap_or(1.0),
    }
}

#[test]
fn shared_rng_and_rank_fixtures_match_python() {
    let fixture = fixtures();
    assert_eq!(
        fixture["rng_scheme"].as_str(),
        Some(rust_sampler_native::RNG_SCHEME)
    );
    let tolerance = fixture["float_tolerance"].as_f64().unwrap();
    for case in fixture["rng_cases"].as_array().unwrap() {
        let actual = position_uniform(
            case["seed"].as_i64().unwrap(),
            case["fingerprint"].as_str().unwrap(),
            case["boundary"].as_str().unwrap(),
        )
        .unwrap();
        assert!((actual - case["expected"].as_f64().unwrap()).abs() <= tolerance);
    }
    for case in fixture["token_uniform_cases"].as_array().unwrap() {
        let actual = position_uniform_token(
            case["seed"].as_i64().unwrap(),
            case["fingerprint"].as_str().unwrap(),
            case["boundary"].as_str().unwrap(),
            case["token_id"].as_str().unwrap(),
        )
        .unwrap();
        assert!((actual - case["expected"].as_f64().unwrap()).abs() <= tolerance);
    }
    let case = &fixture["rank_case"];
    let logits = f64s(&case["logits"]);
    for (token_id, expected) in i64s(&case["raw_ranks"]).iter().enumerate() {
        assert_eq!(rank(&logits, token_id as i64).unwrap() as i64, *expected);
    }
    assert_eq!(top_ids(&logits, 3).unwrap(), i64s(&case["top_ids_3"]));
}

#[test]
fn shared_filter_fixtures_match_python_stage_ids() {
    let fixture = fixtures();
    let tolerance = fixture["float_tolerance"].as_f64().unwrap();
    for case in fixture["filter_cases"].as_array().unwrap() {
        let config = &case["config"];
        let top_k = config["top_k"].as_i64();
        let actual = apply_filter(
            &f64s(&case["logits"]),
            config["temperature"].as_f64().unwrap(),
            top_k,
            config["top_p"].as_f64().unwrap(),
            config["min_p"].as_f64().unwrap(),
            config["typical_p"].as_f64().unwrap(),
            config["tail_free_z"].as_f64().unwrap(),
        )
        .unwrap();
        assert_float_slices(
            &actual.scaled_logits,
            &f64s(&case["expected"]["scaled_logits"]),
            tolerance,
        );
        for (index, name) in [
            "after_temperature",
            "after_top_k",
            "after_typical",
            "after_tail_free",
            "after_top_p",
            "after_min_p",
        ]
        .iter()
        .enumerate()
        {
            let expected = &case["expected"]["stages"][name];
            match expected {
                Value::Null => assert!(actual.stages[index].is_none(), "{} {name}", case["name"]),
                _ => assert_eq!(
                    actual.stages[index].as_ref().unwrap(),
                    &i64s(expected),
                    "{} {name}",
                    case["name"]
                ),
            }
        }
    }
}

#[test]
fn shared_draw_fixtures_match_python_kernels_and_winners() {
    let fixture = fixtures();
    let tolerance = fixture["float_tolerance"].as_f64().unwrap();
    for case in fixture["draw_cases"].as_array().unwrap() {
        let distribution = distribution(case);
        let ranks = case["options"]["candidate_model_ranks"]
            .as_array()
            .map(|values| {
                values
                    .iter()
                    .map(|value| value.as_i64().unwrap().to_string())
                    .collect::<Vec<_>>()
            });
        let options = draw_options(case, ranks.as_deref());
        let actual_draw = draw_token(&distribution, options).unwrap();
        assert_eq!(
            actual_draw,
            case["expected"]["draw"].as_i64().unwrap(),
            "{}",
            case["name"]
        );
        if !case["expected"]["ranking_scores"].is_null() {
            let expected_scores = f64s(&case["expected"]["ranking_scores"]);
            let actual_scores = match case["kernel"].as_str().unwrap() {
                "gumbel-max" => {
                    let scores = gumbel_ranking_scores(
                        &distribution,
                        options.seed,
                        options.fingerprint,
                        options.boundary,
                        options.gumbel_noise_address,
                        options.candidate_model_ranks,
                        options.gumbel_noise_scale,
                    )
                    .unwrap();
                    assert_eq!(
                        ranking_ids(&distribution, &scores).unwrap(),
                        i64s(&case["expected"]["ranked_ids"]),
                        "{} ranked IDs",
                        case["name"]
                    );
                    scores
                }
                "gaussian-max" => gaussian_ranking_scores(
                    &distribution,
                    options.seed,
                    options.fingerprint,
                    options.boundary,
                    options.gaussian_noise_std,
                )
                .unwrap(),
                kernel => perturbation_ranking_scores(
                    &distribution,
                    options.seed,
                    options.fingerprint,
                    options.boundary,
                    kernel,
                    options.perturb_noise_std,
                    options.student_t_df,
                )
                .unwrap(),
            };
            assert_float_slices(&actual_scores, &expected_scores, tolerance);
        }
    }
}

#[test]
fn shared_conditional_gumbel_fixture_matches_python() {
    let fixture = fixtures();
    let case = &fixture["conditional_case"];
    let actual = conditional_gumbel_top_k(
        &f64s(&case["log_probabilities"]),
        case["count"].as_u64().unwrap() as usize,
        case["parent_score"].as_f64().unwrap(),
        case["parent_log_probability"].as_f64().unwrap(),
        case["seed"].as_i64().unwrap(),
        case["fingerprint"].as_str().unwrap(),
        case["boundary"].as_str().unwrap(),
        &case["prefix_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let expected = case["expected"].as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.0 as u64, expected[0].as_u64().unwrap());
        assert!(
            (actual.1 - expected[1].as_f64().unwrap()).abs()
                <= fixture["float_tolerance"].as_f64().unwrap()
        );
    }
}

#[test]
fn fixture_order_variants_keep_token_addressing_attached_to_ids() {
    let fixture = fixtures();
    let mut rankings = Vec::new();
    for case in fixture["draw_cases"].as_array().unwrap() {
        if case["name"]
            .as_str()
            .unwrap()
            .starts_with("gumbel-token-order-")
        {
            let distribution = distribution(case);
            let options = draw_options(case, None);
            let scores = gumbel_ranking_scores(
                &distribution,
                options.seed,
                options.fingerprint,
                options.boundary,
                options.gumbel_noise_address,
                options.candidate_model_ranks,
                options.gumbel_noise_scale,
            )
            .unwrap();
            let mut paired: Vec<(i64, f64)> = distribution.ids.into_iter().zip(scores).collect();
            paired.sort_by_key(|(id, _)| *id);
            rankings.push(paired);
        }
    }
    assert_eq!(rankings.len(), 2);
    for (left, right) in rankings[0].iter().zip(&rankings[1]) {
        assert_eq!(left.0, right.0);
        assert!((left.1 - right.1).abs() <= fixture["float_tolerance"].as_f64().unwrap());
    }
}

#[test]
fn shared_history_penalty_fixtures_match_python_policy_calculations() {
    let fixture = fixtures();
    let tolerance = fixture["float_tolerance"].as_f64().unwrap();
    for case in fixture["history_penalty_cases"].as_array().unwrap() {
        let logits = f64s(&case["logits"]);
        let original_logits = logits.clone();
        let history = i64s(&case["history_token_ids"]);
        let config = &case["config"];
        let actual = apply_history_penalties(
            &logits,
            &history,
            config["repeat_penalty"].as_f64().unwrap(),
            config["repeat_last_n"].as_i64().unwrap(),
            config["presence_penalty"].as_f64().unwrap(),
            config["frequency_penalty"].as_f64().unwrap(),
        )
        .unwrap();
        assert_eq!(
            logits, original_logits,
            "{} mutated its input",
            case["name"]
        );
        assert_float_slices(
            &actual,
            &f64s(&case["expected"]["adjusted_logits"]),
            tolerance,
        );
        for token_id in 0..logits.len() {
            assert_eq!(
                rank(&logits, token_id as i64).unwrap(),
                case["expected"]["raw_ranks"][token_id].as_u64().unwrap() as usize,
                "{} raw rank {token_id}",
                case["name"]
            );
            assert_eq!(
                rank(&actual, token_id as i64).unwrap(),
                case["expected"]["policy_ranks"][token_id].as_u64().unwrap() as usize,
                "{} policy rank {token_id}",
                case["name"]
            );
        }
    }
}

#[test]
fn history_penalties_reject_invalid_inputs_and_overflow() {
    assert!(apply_history_penalties(&[], &[], 1.0, -1, 0.0, 0.0).is_err());
    assert!(apply_history_penalties(&[f64::NAN], &[], 1.0, -1, 0.0, 0.0).is_err());
    assert!(apply_history_penalties(&[f64::INFINITY], &[], 1.0, -1, 0.0, 0.0).is_err());
    assert!(apply_history_penalties(&[1.0], &[-1], 1.0, -1, 0.0, 0.0).is_err());
    assert!(apply_history_penalties(&[1.0], &[1], 1.0, 0, 0.0, 0.0).is_err());
    assert!(apply_history_penalties(&[1.0], &[0], 0.0, -1, 0.0, 0.0).is_err());
    assert!(apply_history_penalties(&[1.0], &[0], f64::INFINITY, -1, 0.0, 0.0).is_err());
    assert!(apply_history_penalties(&[1.0], &[0], 1.0, -2, 0.0, 0.0).is_err());
    assert!(apply_history_penalties(&[1.0], &[0], 1.0, -1, f64::NAN, 0.0).is_err());
    assert!(apply_history_penalties(&[1.0], &[0], 1.0, -1, 0.0, f64::INFINITY).is_err());
    assert!(apply_history_penalties(&[-1.0e308], &[0], 1.0e308, -1, 0.0, 0.0).is_err());
}

#[test]
fn shared_bias_fixtures_match_python_contributions_and_policy_logits() {
    let fixture = fixtures();
    let tolerance = fixture["float_tolerance"].as_f64().unwrap();
    for case in fixture["bias_cases"].as_array().unwrap() {
        let (groups, token_biases, history) = bias_inputs(case);
        let logits = f64s(&case["logits"]);
        let original_logits = logits.clone();
        let contributions = bias_contributions(
            &groups,
            &token_biases,
            Some(&history),
            case["include_inactive"].as_bool().unwrap(),
        )
        .unwrap();
        assert_bias_contributions(
            &contributions,
            &case["expected"]["contributions"],
            tolerance,
        );

        let active = active_biases(&groups, &token_biases, Some(&history)).unwrap();
        let expected_active = case["expected"]["active_biases"].as_array().unwrap();
        assert_eq!(active.len(), expected_active.len());
        for pair in expected_active {
            let token_id = pair[0].as_i64().unwrap();
            let amount = pair[1].as_f64().unwrap();
            assert!((active[&token_id] - amount).abs() <= tolerance);
        }

        let adjusted = apply_biases(&logits, &groups, &token_biases, Some(&history)).unwrap();
        assert_eq!(
            logits, original_logits,
            "{} mutated its input",
            case["name"]
        );
        assert_float_slices(
            &adjusted,
            &f64s(&case["expected"]["adjusted_logits"]),
            tolerance,
        );
        for token_id in 0..logits.len() {
            assert_eq!(
                rank(&logits, token_id as i64).unwrap(),
                case["expected"]["raw_ranks"][token_id].as_u64().unwrap() as usize,
                "{} raw rank {token_id}",
                case["name"]
            );
            assert_eq!(
                rank(&adjusted, token_id as i64).unwrap(),
                case["expected"]["policy_ranks"][token_id].as_u64().unwrap() as usize,
                "{} policy rank {token_id}",
                case["name"]
            );
        }
    }
}

#[test]
fn bias_kernels_reject_missing_history_invalid_ids_and_nonfinite_results() {
    let multi_token_group = BiasGroup {
        name: "phrase".to_owned(),
        members: vec![BiasMember {
            text: "two tokens".to_owned(),
            literal: false,
            routes: vec![BiasRoute {
                token_ids: vec![0, 1],
                surfaces: vec!["two tokens".to_owned()],
            }],
        }],
        bias: 1.0,
    };
    assert!(
        bias_contributions(std::slice::from_ref(&multi_token_group), &[], None, false).is_err()
    );
    assert!(apply_biases(&[1.0], &[multi_token_group], &[], Some(&[])).is_err());

    assert!(
        apply_biases(
            &[1.0],
            &[],
            &[BiasToken {
                token_id: 1,
                bias: 1.0
            }],
            Some(&[])
        )
        .is_err()
    );
    assert!(apply_biases(&[1.0], &[], &[], Some(&[1])).is_err());
    assert!(apply_biases(&[f64::NAN], &[], &[], Some(&[])).is_err());
    assert!(
        apply_biases(
            &[f64::MAX],
            &[],
            &[BiasToken {
                token_id: 0,
                bias: f64::MAX,
            }],
            Some(&[]),
        )
        .is_err()
    );
}

#[test]
fn shared_policy_adjustments_cfg_and_lazy_metric_fixtures_match_python() {
    let fixture = fixtures();
    let tolerance = fixture["float_tolerance"].as_f64().unwrap();

    for case in fixture["activation_kernel_cases"].as_array().unwrap() {
        let logits = f64s(&case["logits"]);
        let adjustments = f64s(&case["adjustments"]);
        let actual =
            apply_activation_adjustments(&logits, &adjustments, case["strength"].as_f64().unwrap())
                .unwrap();
        assert_float_slices(&actual, &f64s(&case["expected"]), tolerance);
    }

    for case in fixture["adjustment_cases"].as_array().unwrap() {
        let logits = f64s(&case["logits"]);
        let history = i64s(&case["history_token_ids"]);
        let config = &case["config"];
        let mut adjusted = apply_history_penalties(
            &logits,
            &history,
            config["repeat_penalty"].as_f64().unwrap(),
            config["repeat_last_n"].as_i64().unwrap(),
            config["presence_penalty"].as_f64().unwrap(),
            config["frequency_penalty"].as_f64().unwrap(),
        )
        .unwrap();
        if let Some(adjustments) = case["activation_adjustments"].as_array() {
            let adjustments: Vec<f64> = adjustments
                .iter()
                .map(|value| value.as_f64().unwrap())
                .collect();
            adjusted = apply_activation_adjustments(
                &adjusted,
                &adjustments,
                config["steering_strength"].as_f64().unwrap(),
            )
            .unwrap();
        }
        let empty = Value::Array(Vec::new());
        let groups_value = config.get("bias_groups").unwrap_or(&empty);
        let tokens_value = config.get("token_biases").unwrap_or(&empty);
        let (groups, token_biases) = bias_inputs_from_parts(groups_value, tokens_value);
        adjusted = apply_biases(&adjusted, &groups, &token_biases, Some(&history)).unwrap();
        let ephemeral: Vec<(i64, f64)> = case["ephemeral_biases"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(token, amount)| (token.parse().unwrap(), amount.as_f64().unwrap()))
            .collect();
        adjusted = apply_ephemeral_biases(&adjusted, &ephemeral).unwrap();
        assert_float_slices(
            &adjusted,
            &f64s(&case["expected"]["adjusted_logits"]),
            tolerance,
        );
        for (token_id, expected) in i64s(&case["expected"]["raw_ranks"]).iter().enumerate() {
            assert_eq!(rank(&logits, token_id as i64).unwrap() as i64, *expected);
        }
        for (token_id, expected) in i64s(&case["expected"]["policy_ranks"]).iter().enumerate() {
            assert_eq!(rank(&adjusted, token_id as i64).unwrap() as i64, *expected);
        }
    }

    for case in fixture["cfg_cases"].as_array().unwrap() {
        let actual = cfg_combine_logits(
            &f64s(&case["conditional_logits"]),
            &f64s(&case["unconditional_logits"]),
            case["scale"].as_f64().unwrap(),
        )
        .unwrap();
        assert_float_slices(&actual, &f64s(&case["expected"]), tolerance);
    }

    for case in fixture["metric_cases"].as_array().unwrap() {
        let raw = f64s(&case["logits"]);
        let policy_values = case["policy_logits"]
            .as_array()
            .map(|_| f64s(&case["policy_logits"]));
        let mut metrics = PolicyMetrics::new(&raw, policy_values.as_deref()).unwrap();
        assert!(!metrics.raw_logsumexp_ready());
        assert!(!metrics.logit_mean_std_ready());
        assert!(
            (metrics.maximum() - case["expected"]["maximum"].as_f64().unwrap()).abs() <= tolerance
        );
        assert_eq!(
            metrics
                .top_raw_ids(case["expected"]["top_raw_ids"].as_array().unwrap().len() as i64)
                .unwrap(),
            i64s(&case["expected"]["top_raw_ids"]),
        );
        assert_eq!(
            metrics
                .top_policy_ids(
                    case["expected"]["top_policy_ids"].as_array().unwrap().len() as i64,
                )
                .unwrap(),
            i64s(&case["expected"]["top_policy_ids"]),
        );
        let token_ids = (0..raw.len() as i64).collect::<Vec<_>>();
        assert_eq!(
            token_ids
                .iter()
                .map(|token_id| metrics.raw_rank(*token_id).unwrap() as i64)
                .collect::<Vec<_>>(),
            i64s(&case["expected"]["raw_ranks"]),
        );
        assert_eq!(
            token_ids
                .iter()
                .map(|token_id| metrics.policy_rank(*token_id).unwrap() as i64)
                .collect::<Vec<_>>(),
            i64s(&case["expected"]["policy_ranks"]),
        );
        let selected = i64s(&case["selected_ids"]);
        let z_scores = metrics.logit_z_scores(&selected).unwrap();
        assert!(!metrics.raw_logsumexp_ready());
        assert!(metrics.logit_mean_std_ready());
        let expected_z = case["expected"]["logit_z_scores"].as_array().unwrap();
        for (actual, expected) in z_scores.iter().zip(expected_z) {
            match (actual, expected.as_f64()) {
                (Some(actual), Some(expected)) => assert!((actual - expected).abs() <= tolerance),
                (None, None) => {}
                _ => panic!("z-score definedness differs in {}", case["name"]),
            }
        }
        assert_float_slices(
            &metrics.raw_probabilities(&selected).unwrap(),
            &f64s(&case["expected"]["raw_probabilities"]),
            tolerance,
        );
        assert_float_slices(
            &metrics.policy_probabilities(&selected).unwrap(),
            &f64s(&case["expected"]["policy_probabilities"]),
            tolerance,
        );
        assert_float_slices(
            &token_ids
                .iter()
                .map(|token_id| metrics.raw_nll(*token_id).unwrap())
                .collect::<Vec<_>>(),
            &f64s(&case["expected"]["raw_nll"]),
            tolerance,
        );
        assert!(
            (metrics.log_z().unwrap() - case["expected"]["log_z"].as_f64().unwrap()).abs()
                <= tolerance
        );
        assert!(
            (metrics.denominator().unwrap() - case["expected"]["denominator"].as_f64().unwrap())
                .abs()
                <= tolerance
        );
        assert!(metrics.raw_logsumexp_ready());
    }
}

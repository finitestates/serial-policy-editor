use serde_json::Value;

use rust_sampler_native::{
    DrawOptions, SparseDistribution, apply_filter, apply_history_penalties,
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

fn assert_float_slices(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "float at {index}: actual {actual:.17e}, expected {expected:.17e}, tolerance {tolerance:.1e}"
        );
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

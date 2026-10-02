//! Numeric sampling kernels for the policy editor.
//!
//! The Python package remains the reference implementation.  This crate owns
//! the arithmetic and addressable random draws; `python/rust_sampler` only
//! checks/converts NumPy values at the extension boundary.

use std::collections::HashSet;

use blake2::digest::consts::{U8, U16};
use blake2::{Blake2b, Digest};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

pub const RNG_SCHEME: &str = "blake2b64-token-prefix-quantile-v2";
pub const MIN_SEED: i64 = i64::MIN;
pub const MAX_SEED: i64 = i64::MAX;
pub const DRAW_KERNELS: &[&str] = &[
    "categorical",
    "gumbel-max",
    "gaussian-max",
    "logistic-max",
    "student-t-max",
    "laplace-max",
    "uniform-max",
];
pub const PERTURB_MAX_KERNELS: &[&str] = &[
    "logistic-max",
    "student-t-max",
    "laplace-max",
    "uniform-max",
];

create_exception!(_native, NativeEditorError, PyValueError);

#[derive(Debug)]
pub enum SamplingError {
    Editor(String),
    Value(String),
}

pub type SamplingResult<T> = Result<T, SamplingError>;

impl SamplingError {
    fn editor(message: impl Into<String>) -> Self {
        Self::Editor(message.into())
    }

    fn value(message: impl Into<String>) -> Self {
        Self::Value(message.into())
    }
}

fn into_pyerr(error: SamplingError) -> PyErr {
    match error {
        SamplingError::Editor(message) => NativeEditorError::new_err(message),
        SamplingError::Value(message) => PyValueError::new_err(message),
    }
}

fn blake2b8(payload: &[u8]) -> [u8; 8] {
    let digest = Blake2b::<U8>::digest(payload);
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest);
    bytes
}

fn blake2b16(payload: &[u8]) -> [u8; 16] {
    let digest = Blake2b::<U16>::digest(payload);
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest);
    bytes
}

fn midpoint_uniform(payload: &[u8]) -> f64 {
    let integer = u64::from_be_bytes(blake2b8(payload));
    (integer as f64 + 0.5) / 18_446_744_073_709_551_616.0
}

fn validate_seed(seed: i64) -> SamplingResult<()> {
    if !(MIN_SEED..=MAX_SEED).contains(&seed) {
        return Err(SamplingError::editor(
            "seed must be a signed-64-bit integer",
        ));
    }
    Ok(())
}

fn validate_fingerprint(value: &str) -> SamplingResult<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(SamplingError::editor(
            "stream_fingerprint must be a lowercase SHA-256 hex digest",
        ));
    }
    Ok(())
}

fn validate_boundary(value: &str, name: &str) -> SamplingResult<()> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(SamplingError::editor(format!(
            "{name} must be a nonnegative integer"
        )));
    }
    Ok(())
}

fn scalar_address(seed: i64, fingerprint: &str, boundary: &str) -> String {
    format!("{RNG_SCHEME}:{seed}:{fingerprint}:{boundary}")
}

pub fn position_uniform(seed: i64, fingerprint: &str, boundary: &str) -> SamplingResult<f64> {
    validate_seed(seed)?;
    validate_fingerprint(fingerprint)?;
    validate_boundary(boundary, "sampling boundary")?;
    Ok(midpoint_uniform(
        scalar_address(seed, fingerprint, boundary).as_bytes(),
    ))
}

pub fn position_uniform_token(
    seed: i64,
    fingerprint: &str,
    boundary: &str,
    token_id: &str,
) -> SamplingResult<f64> {
    if token_id.is_empty() || !token_id.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(SamplingError::editor(
            "token id must be a nonnegative integer",
        ));
    }
    validate_seed(seed)?;
    validate_fingerprint(fingerprint)?;
    validate_boundary(boundary, "sampling boundary")?;
    let payload = format!("{RNG_SCHEME}:gumbel-max:{seed}:{fingerprint}:{boundary}:{token_id}");
    Ok(midpoint_uniform(payload.as_bytes()))
}

pub fn position_uniform_model_rank(
    seed: i64,
    fingerprint: &str,
    boundary: &str,
    model_rank: &str,
) -> SamplingResult<f64> {
    if model_rank.is_empty()
        || !model_rank.bytes().all(|byte| byte.is_ascii_digit())
        || model_rank.bytes().all(|byte| byte == b'0')
    {
        return Err(SamplingError::editor(
            "model rank must be a positive integer",
        ));
    }
    validate_seed(seed)?;
    validate_fingerprint(fingerprint)?;
    validate_boundary(boundary, "sampling boundary")?;
    let payload =
        format!("{RNG_SCHEME}:gumbel-max:model-rank:{seed}:{fingerprint}:{boundary}:{model_rank}");
    Ok(midpoint_uniform(payload.as_bytes()))
}

pub fn validated_logits(values: &[f64]) -> SamplingResult<Vec<f64>> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(SamplingError::value(
            "decoder logits must be a finite nonempty one-dimensional array",
        ));
    }
    Ok(values.to_vec())
}

pub fn softmax(values: &[f64]) -> SamplingResult<Vec<f64>> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(SamplingError::value(
            "softmax values must be a finite nonempty vector",
        ));
    }
    let max_value = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = values
        .iter()
        .map(|value| (value - max_value).exp())
        .collect();
    let denominator: f64 = exponentials.iter().sum();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(SamplingError::value("softmax normalization failed"));
    }
    Ok(exponentials
        .into_iter()
        .map(|value| value / denominator)
        .collect())
}

fn sorted_ids(ids: &[i64], scores: &[f64]) -> Vec<i64> {
    let mut selected = ids.to_vec();
    selected.sort_by(|left, right| {
        scores[*right as usize]
            .total_cmp(&scores[*left as usize])
            .then_with(|| left.cmp(right))
    });
    selected
}

pub fn top_ids(values: &[f64], count: i64) -> SamplingResult<Vec<i64>> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(SamplingError::value(
            "decoder logits must be a finite nonempty vector",
        ));
    }
    let count = count.clamp(1, values.len() as i64) as usize;
    let mut ids: Vec<i64> = (0..values.len()).map(|value| value as i64).collect();
    ids.sort_by(|left, right| {
        values[*right as usize]
            .total_cmp(&values[*left as usize])
            .then_with(|| left.cmp(right))
    });
    ids.truncate(count);
    Ok(ids)
}

pub fn rank(values: &[f64], token_id: i64) -> SamplingResult<usize> {
    if token_id < 0 || token_id as usize >= values.len() {
        return Err(SamplingError::value(
            "token id is outside the decoder vocabulary",
        ));
    }
    let target = values[token_id as usize];
    let higher = values.iter().filter(|value| **value > target).count();
    let tied_before = values[..token_id as usize]
        .iter()
        .filter(|value| **value == target)
        .count();
    Ok(1 + higher + tied_before)
}

#[derive(Clone, Debug, PartialEq)]
pub struct FilterResult {
    pub scaled_logits: Vec<f64>,
    /// Stage order: temperature, top-k, typical-p, tail-free, top-p, min-p.
    pub stages: [Option<Vec<i64>>; 6],
    pub greedy: bool,
    pub unfiltered: bool,
}

fn threshold_count_left(values: &[f64], threshold: f64) -> usize {
    let mut cumulative = 0.0;
    for (index, value) in values.iter().enumerate() {
        cumulative += value;
        if cumulative >= threshold {
            return index + 1;
        }
    }
    values.len()
}

fn typical(ids: &[i64], scores: &[f64], typical_p: f64) -> SamplingResult<Vec<i64>> {
    let ids = sorted_ids(ids, scores);
    if typical_p >= 1.0 || ids.len() <= 1 {
        return Ok(ids);
    }
    let selected_scores: Vec<f64> = ids.iter().map(|id| scores[*id as usize]).collect();
    let probabilities = softmax(&selected_scores)?;
    let entropy: f64 = probabilities
        .iter()
        .map(|probability| {
            let safe = probability.max(f64::MIN_POSITIVE);
            -safe * safe.ln()
        })
        .sum();
    let mut indices: Vec<usize> = (0..ids.len()).collect();
    indices.sort_by(|left, right| {
        let left_probability = probabilities[*left].max(f64::MIN_POSITIVE);
        let right_probability = probabilities[*right].max(f64::MIN_POSITIVE);
        let left_value = (-left_probability.ln() - entropy).abs();
        let right_value = (-right_probability.ln() - entropy).abs();
        left_value
            .total_cmp(&right_value)
            .then_with(|| ids[*left].cmp(&ids[*right]))
    });
    let ordered_probabilities: Vec<f64> =
        indices.iter().map(|index| probabilities[*index]).collect();
    let keep = threshold_count_left(&ordered_probabilities, typical_p).max(1);
    Ok(indices
        .into_iter()
        .take(keep)
        .map(|index| ids[index])
        .collect())
}

fn tail_free(ids: &[i64], scores: &[f64], tail_free_z: f64) -> SamplingResult<Vec<i64>> {
    let ids = sorted_ids(ids, scores);
    if tail_free_z >= 1.0 || ids.len() < 3 {
        return Ok(ids);
    }
    let selected_scores: Vec<f64> = ids.iter().map(|id| scores[*id as usize]).collect();
    let probabilities = softmax(&selected_scores)?;
    let first: Vec<f64> = probabilities
        .windows(2)
        .map(|window| (window[1] - window[0]).abs())
        .collect();
    let second: Vec<f64> = first
        .windows(2)
        .map(|window| (window[1] - window[0]).abs())
        .collect();
    let total: f64 = second.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return Ok(ids);
    }
    let mass: Vec<f64> = second.iter().map(|value| value / total).collect();
    let keep = (threshold_count_left(&mass, tail_free_z) + 1).clamp(1, ids.len());
    Ok(ids.into_iter().take(keep).collect())
}

pub fn apply_filter(
    adjusted: &[f64],
    temperature: f64,
    top_k: Option<i64>,
    top_p: f64,
    min_p: f64,
    typical_p: f64,
    tail_free_z: f64,
) -> SamplingResult<FilterResult> {
    if temperature == 0.0 {
        let greedy = top_ids(adjusted, 1)?;
        return Ok(FilterResult {
            scaled_logits: adjusted.to_vec(),
            stages: [
                Some(greedy.clone()),
                Some(greedy.clone()),
                Some(greedy.clone()),
                Some(greedy.clone()),
                Some(greedy.clone()),
                Some(greedy),
            ],
            greedy: true,
            unfiltered: false,
        });
    }
    let scaled: Vec<f64> = if temperature == 1.0 {
        adjusted.to_vec()
    } else {
        adjusted.iter().map(|value| value / temperature).collect()
    };
    if scaled.iter().any(|value| !value.is_finite()) {
        return Err(SamplingError::value(
            "temperature produced non-finite scaled logits",
        ));
    }
    let unfiltered = temperature == 1.0
        && top_k.is_none()
        && typical_p == 1.0
        && tail_free_z == 1.0
        && top_p == 1.0
        && min_p == 0.0;
    if unfiltered {
        let ids: Vec<i64> = (0..scaled.len()).map(|value| value as i64).collect();
        return Ok(FilterResult {
            scaled_logits: scaled,
            stages: [
                None,
                Some(ids.clone()),
                Some(ids.clone()),
                Some(ids.clone()),
                Some(ids.clone()),
                Some(ids),
            ],
            greedy: false,
            unfiltered: true,
        });
    }
    let all_ids: Vec<i64> = (0..scaled.len()).map(|value| value as i64).collect();
    let after_top_k = if let Some(top_k) = top_k {
        top_ids(&scaled, top_k.min(scaled.len() as i64))?
    } else {
        all_ids
    };
    let after_typical = if typical_p >= 1.0 {
        after_top_k.clone()
    } else {
        typical(&after_top_k, &scaled, typical_p)?
    };
    let after_tail_free = if typical_p >= 1.0 && tail_free_z >= 1.0 {
        after_top_k.clone()
    } else {
        tail_free(&after_typical, &scaled, tail_free_z)?
    };
    let mut after_top_p = after_tail_free.clone();
    if top_p < 1.0 {
        let values: Vec<f64> = after_top_p.iter().map(|id| scaled[*id as usize]).collect();
        let probabilities = softmax(&values)?;
        let keep = threshold_count_left(&probabilities, top_p).max(1);
        after_top_p.truncate(keep);
    }
    let mut after_min_p = after_top_p.clone();
    if min_p > 0.0 {
        let values: Vec<f64> = after_top_p.iter().map(|id| scaled[*id as usize]).collect();
        let probabilities = softmax(&values)?;
        if let Some(max_probability) = probabilities.iter().copied().reduce(f64::max) {
            let filtered: Vec<i64> = after_top_p
                .iter()
                .zip(probabilities.iter())
                .filter_map(|(id, probability)| {
                    (*probability >= min_p * max_probability).then_some(*id)
                })
                .collect();
            if !filtered.is_empty() {
                after_min_p = filtered;
            }
        }
    }
    Ok(FilterResult {
        scaled_logits: scaled,
        stages: [
            None,
            Some(after_top_k),
            Some(after_typical),
            Some(after_tail_free),
            Some(after_top_p),
            Some(after_min_p),
        ],
        greedy: false,
        unfiltered: false,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparseDistribution {
    pub ids: Vec<i64>,
    pub probabilities: Vec<f64>,
    pub scores: Option<Vec<f64>>,
}

impl SparseDistribution {
    pub fn probability(&self, token_id: i64) -> f64 {
        self.ids
            .iter()
            .position(|id| *id == token_id)
            .and_then(|index| self.probabilities.get(index).copied())
            .unwrap_or(0.0)
    }
}

fn validate_score_ids(distribution: &SparseDistribution, kernel: &str) -> SamplingResult<Vec<f64>> {
    let Some(scores) = distribution.scores.as_ref() else {
        return Err(SamplingError::value(format!(
            "{kernel} requires candidate scores"
        )));
    };
    if scores.len() != distribution.ids.len() {
        return Err(SamplingError::value(
            "candidate scores do not match candidate IDs",
        ));
    }
    if distribution.ids.is_empty() {
        return Err(SamplingError::value(format!(
            "{kernel} requires at least one candidate"
        )));
    }
    Ok(scores.clone())
}

fn validate_noise(noise: f64, name: &str) -> SamplingResult<()> {
    if !noise.is_finite() || noise < 0.0 {
        return Err(SamplingError::editor(format!(
            "{name} must be finite and nonnegative"
        )));
    }
    Ok(())
}

pub fn gaussian_ranking_scores(
    distribution: &SparseDistribution,
    seed: i64,
    fingerprint: &str,
    boundary: &str,
    noise_std: f64,
) -> SamplingResult<Vec<f64>> {
    let scores = validate_score_ids(distribution, "gaussian-max")?;
    validate_noise(noise_std, "gaussian_noise_std")?;
    validate_seed(seed)?;
    validate_fingerprint(fingerprint)?;
    validate_boundary(boundary, "sampling boundary")?;
    if noise_std == 0.0 {
        return Ok(scores);
    }
    let mut ranking = Vec::with_capacity(scores.len());
    for (score, token_id) in scores.iter().zip(&distribution.ids) {
        if *token_id < 0 {
            return Err(SamplingError::editor(
                "token id must be a nonnegative integer",
            ));
        }
        let prefix =
            format!("{RNG_SCHEME}:gaussian-max:{seed}:{fingerprint}:{boundary}:{token_id}:");
        let first = midpoint_uniform(format!("{prefix}0").as_bytes());
        let second = midpoint_uniform(format!("{prefix}1").as_bytes());
        let radius = (-2.0 * first.ln()).sqrt();
        let perturbation = radius * (2.0 * std::f64::consts::PI * second).cos();
        ranking.push(score + noise_std * perturbation);
    }
    if ranking.iter().any(|value| !value.is_finite()) {
        return Err(SamplingError::value(
            "gaussian noise produced non-finite ranking scores",
        ));
    }
    Ok(ranking)
}

fn perturbation_uniform(
    seed: i64,
    fingerprint: &str,
    boundary: &str,
    token_id: i64,
    lane: u64,
) -> f64 {
    let payload =
        format!("{RNG_SCHEME}:perturb-max:{seed}:{fingerprint}:{boundary}:{token_id}:{lane}");
    let integer = u64::from_be_bytes(blake2b8(payload.as_bytes())) >> 12;
    (integer as f64 + 0.5) / 4_503_599_627_370_496.0
}

fn normal_from_uniforms(first: f64, second: f64) -> f64 {
    (-2.0 * first.ln()).sqrt() * (2.0 * std::f64::consts::PI * second).cos()
}

struct PerturbationDraw<'a> {
    seed: i64,
    fingerprint: &'a str,
    boundary: &'a str,
    token_id: i64,
    lane: u64,
}

impl PerturbationDraw<'_> {
    fn uniform(&mut self) -> f64 {
        let result = perturbation_uniform(
            self.seed,
            self.fingerprint,
            self.boundary,
            self.token_id,
            self.lane,
        );
        self.lane += 1;
        result
    }

    fn normal(&mut self) -> f64 {
        let first = self.uniform();
        let second = self.uniform();
        normal_from_uniforms(first, second)
    }
}

fn log_gamma_ge_one(shape: f64, random: &mut PerturbationDraw<'_>) -> SamplingResult<f64> {
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    for _ in 0..128 {
        let value = random.normal();
        let base = 1.0 + c * value;
        if base <= 0.0 {
            continue;
        }
        let cube = base * base * base;
        let draw = random.uniform();
        if draw < 1.0 - 0.0331 * value.powi(4)
            || draw.ln() < 0.5 * value * value + d * (1.0 - cube + cube.ln())
        {
            return Ok(d.ln() + cube.ln());
        }
    }
    Err(SamplingError::value(
        "Student-t gamma draw did not converge",
    ))
}

fn log_gamma(shape: f64, random: &mut PerturbationDraw<'_>) -> SamplingResult<f64> {
    if shape >= 1.0 {
        return log_gamma_ge_one(shape, random);
    }
    if shape <= 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    let boosted = log_gamma_ge_one(shape + 1.0, random)?;
    Ok(boosted + random.uniform().ln() / shape)
}

fn student_t_unit_noise(
    seed: i64,
    fingerprint: &str,
    boundary: &str,
    token_id: i64,
    degrees_of_freedom: f64,
) -> SamplingResult<f64> {
    let uniform_at = |lane| perturbation_uniform(seed, fingerprint, boundary, token_id, lane);
    if degrees_of_freedom == 3.0 {
        let mut normals = Vec::with_capacity(4);
        for offset in [0_u64, 2] {
            let radius = (-2.0 * uniform_at(offset).ln()).sqrt();
            let angle = 2.0 * std::f64::consts::PI * uniform_at(offset + 1);
            normals.push(radius * angle.cos());
            normals.push(radius * angle.sin());
        }
        return Ok(normals[0]
            / (normals[1] * normals[1] + normals[2] * normals[2] + normals[3] * normals[3])
                .sqrt());
    }
    let mut random = PerturbationDraw {
        seed,
        fingerprint,
        boundary,
        token_id,
        lane: 0,
    };
    let normal = random.normal();
    let gamma = log_gamma(degrees_of_freedom / 2.0, &mut random)?;
    if normal == 0.0 {
        return Ok(0.0);
    }
    let log_magnitude = normal.abs().ln()
        - 0.5 * (2.0_f64.ln() + gamma - degrees_of_freedom.ln())
        - 0.5 * 3.0_f64.ln();
    if log_magnitude > f64::MAX.ln() {
        return Ok(normal.signum() * f64::INFINITY);
    }
    if log_magnitude < f64::from_bits(1).ln() {
        return Ok(normal.signum() * 0.0);
    }
    Ok(normal.signum() * log_magnitude.exp())
}

fn validate_student_t_df(value: f64) -> SamplingResult<f64> {
    if !value.is_finite() || value <= 0.0 {
        return Err(SamplingError::editor(
            "student_t_df must be finite and greater than 0",
        ));
    }
    Ok(value)
}

pub fn perturbation_ranking_scores(
    distribution: &SparseDistribution,
    seed: i64,
    fingerprint: &str,
    boundary: &str,
    kernel: &str,
    noise_std: f64,
    student_t_df: f64,
) -> SamplingResult<Vec<f64>> {
    if !PERTURB_MAX_KERNELS.contains(&kernel) {
        return Err(SamplingError::editor(
            "unsupported perturb-and-argmax kernel",
        ));
    }
    let student_t_df = if kernel == "student-t-max" {
        validate_student_t_df(student_t_df)?
    } else {
        student_t_df
    };
    let scores = validate_score_ids(distribution, kernel)?;
    validate_noise(noise_std, "perturb_noise_std")?;
    validate_seed(seed)?;
    validate_fingerprint(fingerprint)?;
    validate_boundary(boundary, "sampling boundary")?;
    if noise_std == 0.0 {
        return Ok(scores);
    }
    let mut ranking = Vec::with_capacity(scores.len());
    for (score, token_id) in scores.iter().zip(&distribution.ids) {
        if *token_id < 0 {
            return Err(SamplingError::editor(
                "token id must be a nonnegative integer",
            ));
        }
        let unit_noise = if kernel == "student-t-max" {
            student_t_unit_noise(seed, fingerprint, boundary, *token_id, student_t_df)?
        } else {
            let uniform = perturbation_uniform(seed, fingerprint, boundary, *token_id, 0);
            match kernel {
                "logistic-max" => {
                    (3.0_f64.sqrt() / std::f64::consts::PI) * (uniform.ln() - (-uniform).ln_1p())
                }
                "laplace-max" => {
                    if uniform < 0.5 {
                        (2.0 * uniform).ln() / 2.0_f64.sqrt()
                    } else {
                        -(2.0 * (1.0 - uniform)).ln() / 2.0_f64.sqrt()
                    }
                }
                _ => 3.0_f64.sqrt() * (2.0 * uniform - 1.0),
            }
        };
        ranking.push(score + noise_std * unit_noise);
    }
    if ranking.iter().any(|value| !value.is_finite()) {
        if kernel == "student-t-max" {
            return Err(SamplingError::value(format!(
                "student-t-max at df={} produced non-finite ranking scores",
                student_t_df
            )));
        }
        return Err(SamplingError::value(format!(
            "{kernel} noise produced non-finite ranking scores"
        )));
    }
    Ok(ranking)
}

fn logaddexp(left: f64, right: f64) -> f64 {
    let high = left.max(right);
    let low = left.min(right);
    if high == f64::INFINITY {
        high
    } else {
        high + (low - high).exp().ln_1p()
    }
}

fn choose_weighted(remaining: &[f64], remaining_mass: f64, quantile: f64) -> usize {
    let target = quantile * remaining_mass;
    let mut cumulative = 0.0;
    let mut last_positive = 0;
    for (index, weight) in remaining.iter().enumerate() {
        cumulative += weight;
        if *weight > 0.0 {
            last_positive = index;
        }
        if cumulative > target {
            return index;
        }
    }
    last_positive
}

#[allow(clippy::too_many_arguments)]
pub fn conditional_gumbel_top_k(
    log_probabilities: &[f64],
    count: usize,
    parent_score: f64,
    parent_log_probability: f64,
    seed: i64,
    fingerprint: &str,
    boundary: &str,
    prefix_token_ids: &[u64],
) -> SamplingResult<Vec<(usize, f64)>> {
    if log_probabilities.is_empty() || log_probabilities.iter().any(|value| !value.is_finite()) {
        return Err(SamplingError::value(
            "conditional Gumbel sampling requires finite log probabilities",
        ));
    }
    if count == 0 {
        return Err(SamplingError::editor(
            "conditional Gumbel sample count must be a positive integer",
        ));
    }
    if !parent_score.is_finite() || !parent_log_probability.is_finite() {
        return Err(SamplingError::editor(
            "conditional Gumbel parent scores must be finite",
        ));
    }
    validate_seed(seed)?;
    validate_fingerprint(fingerprint)?;
    validate_boundary(boundary, "sampling boundary")?;

    let max_value = log_probabilities
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = log_probabilities
        .iter()
        .map(|value| (value - max_value).exp())
        .collect();
    let denominator: f64 = exponentials.iter().sum();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(SamplingError::value(
            "conditional Gumbel child mass is invalid",
        ));
    }
    let mut remaining: Vec<f64> = exponentials
        .iter()
        .map(|value| value / denominator)
        .collect();

    let mut prefix_bytes = Vec::with_capacity(8 + prefix_token_ids.len() * 8);
    prefix_bytes.extend_from_slice(&(prefix_token_ids.len() as u64).to_be_bytes());
    for token_id in prefix_token_ids {
        prefix_bytes.extend_from_slice(&token_id.to_be_bytes());
    }
    let address =
        format!("{RNG_SCHEME}:stochastic-beam-gumbel-top-k-v1:{seed}:{fingerprint}:{boundary}:");
    let mut payload = address.into_bytes();
    payload.extend_from_slice(&prefix_bytes);
    let parent_key = blake2b16(&payload);

    let uniform = |lane: &[u8], ordinal: u64| {
        let mut child_payload = Vec::with_capacity(24 + lane.len());
        child_payload.extend_from_slice(b"stochastic-beam-child-v1:");
        child_payload.extend_from_slice(&parent_key);
        child_payload.push(b':');
        child_payload.extend_from_slice(lane);
        child_payload.push(b':');
        child_payload.extend_from_slice(&ordinal.to_be_bytes());
        midpoint_uniform(&child_payload)
    };

    let mut remaining_mass: f64 = remaining.iter().sum();
    let winner = choose_weighted(&remaining, remaining_mass, uniform(b"winner", 0));
    let mut result = vec![(winner, parent_score)];
    remaining[winner] = 0.0;
    for ordinal in 1..count.min(log_probabilities.len()) {
        remaining_mass = remaining.iter().sum();
        if remaining_mass <= 0.0 {
            break;
        }
        let log_rate = parent_log_probability + remaining_mass.ln();
        let exponential_wait = -uniform(b"wait", ordinal as u64).ln();
        let log_wait = exponential_wait.ln() - log_rate;
        let child_score = -logaddexp(-parent_score, log_wait);
        let token_id = choose_weighted(
            &remaining,
            remaining_mass,
            uniform(b"winner", ordinal as u64),
        );
        result.push((token_id, child_score));
        remaining[token_id] = 0.0;
    }
    Ok(result)
}

fn winner(
    distribution: &SparseDistribution,
    ranking_scores: &[f64],
    kernel: &str,
) -> SamplingResult<i64> {
    if ranking_scores.len() != distribution.ids.len() {
        let message = match kernel {
            "gumbel" => "Gumbel scores do not match candidate IDs",
            "gaussian" => "Gaussian scores do not match candidate IDs",
            _ => "perturbation scores do not match candidate IDs",
        };
        return Err(SamplingError::value(message));
    }
    if distribution.ids.is_empty() {
        let message = match kernel {
            "gumbel" => "gumbel-max requires at least one candidate",
            "gaussian" => "gaussian-max requires at least one candidate",
            _ => "perturb-and-argmax requires at least one candidate",
        };
        return Err(SamplingError::value(message));
    }
    let max_score = ranking_scores
        .iter()
        .copied()
        .reduce(f64::max)
        .unwrap_or(f64::NEG_INFINITY);
    distribution
        .ids
        .iter()
        .zip(ranking_scores)
        .filter_map(|(id, score)| (*score == max_score).then_some(*id))
        .min()
        .ok_or_else(|| SamplingError::value("no maximum candidate score"))
}

pub fn gumbel_ranking_scores(
    distribution: &SparseDistribution,
    seed: i64,
    fingerprint: &str,
    boundary: &str,
    noise_address: &str,
    candidate_model_ranks: Option<&[String]>,
    noise_scale: f64,
) -> SamplingResult<Vec<f64>> {
    let scores = validate_score_ids(distribution, "gumbel-max")?;
    validate_noise(noise_scale, "gumbel_noise_scale")?;
    if noise_address != "token-id" && noise_address != "model-rank" {
        return Err(SamplingError::editor(
            "gumbel_noise_address must be token-id or model-rank",
        ));
    }
    // This historical fast path intentionally skips address and rank checks.
    if noise_scale == 0.0 {
        return Ok(scores);
    }
    let uniforms = if noise_address == "model-rank" {
        let ranks = candidate_model_ranks.ok_or_else(|| {
            SamplingError::value("model-rank Gumbel noise requires candidate model ranks")
        })?;
        if ranks.len() != distribution.ids.len() {
            return Err(SamplingError::value(
                "candidate model ranks must be distinct positive integers",
            ));
        }
        let mut unique = HashSet::new();
        for rank in ranks {
            if rank.is_empty()
                || !rank.bytes().all(|byte| byte.is_ascii_digit())
                || rank.bytes().all(|byte| byte == b'0')
                || !unique.insert(rank.clone())
            {
                return Err(SamplingError::value(
                    "candidate model ranks must be distinct positive integers",
                ));
            }
        }
        ranks
            .iter()
            .map(|rank| position_uniform_model_rank(seed, fingerprint, boundary, rank))
            .collect::<SamplingResult<Vec<_>>>()?
    } else {
        distribution
            .ids
            .iter()
            .map(|id| position_uniform_token(seed, fingerprint, boundary, &id.to_string()))
            .collect::<SamplingResult<Vec<_>>>()?
    };
    let ranking: Vec<f64> = if noise_scale == 1.0 {
        scores
            .iter()
            .zip(uniforms)
            .map(|(score, uniform)| score - (-uniform.ln()).ln())
            .collect()
    } else {
        scores
            .iter()
            .zip(uniforms)
            .map(|(score, uniform)| score - noise_scale * (-uniform.ln()).ln())
            .collect()
    };
    if noise_scale != 1.0 && ranking.iter().any(|value| !value.is_finite()) {
        return Err(SamplingError::value(
            "Gumbel noise produced non-finite ranking scores",
        ));
    }
    Ok(ranking)
}

#[derive(Clone, Copy)]
pub struct DrawOptions<'a> {
    pub seed: i64,
    pub fingerprint: &'a str,
    pub boundary: &'a str,
    pub kernel: &'a str,
    pub gaussian_noise_std: f64,
    pub perturb_noise_std: f64,
    pub student_t_df: f64,
    pub gumbel_noise_address: &'a str,
    pub candidate_model_ranks: Option<&'a [String]>,
    pub gumbel_noise_scale: f64,
}

pub fn draw_token(
    distribution: &SparseDistribution,
    options: DrawOptions<'_>,
) -> SamplingResult<i64> {
    if !DRAW_KERNELS.contains(&options.kernel) {
        return Err(SamplingError::editor("unsupported draw kernel"));
    }
    match options.kernel {
        "gumbel-max" => {
            let scores = gumbel_ranking_scores(
                distribution,
                options.seed,
                options.fingerprint,
                options.boundary,
                options.gumbel_noise_address,
                options.candidate_model_ranks,
                options.gumbel_noise_scale,
            )?;
            winner(distribution, &scores, "gumbel")
        }
        "gaussian-max" => {
            let scores = gaussian_ranking_scores(
                distribution,
                options.seed,
                options.fingerprint,
                options.boundary,
                options.gaussian_noise_std,
            )?;
            winner(distribution, &scores, "gaussian")
        }
        "logistic-max" | "student-t-max" | "laplace-max" | "uniform-max" => {
            let scores = perturbation_ranking_scores(
                distribution,
                options.seed,
                options.fingerprint,
                options.boundary,
                options.kernel,
                options.perturb_noise_std,
                options.student_t_df,
            )?;
            winner(distribution, &scores, "perturb")
        }
        _ => {
            let draw = position_uniform(options.seed, options.fingerprint, options.boundary)?;
            let mut cumulative = 0.0;
            let mut index = distribution.probabilities.len();
            for (position, probability) in distribution.probabilities.iter().enumerate() {
                cumulative += probability;
                if cumulative > draw {
                    index = position;
                    break;
                }
            }
            if distribution.ids.is_empty() {
                return Err(SamplingError::value(
                    "cannot draw from an empty candidate set",
                ));
            }
            Ok(distribution.ids[index.min(distribution.ids.len() - 1)])
        }
    }
}

pub fn ranking_ids(
    distribution: &SparseDistribution,
    ranking_scores: &[f64],
) -> SamplingResult<Vec<i64>> {
    if ranking_scores.len() != distribution.ids.len() {
        return Err(SamplingError::value(
            "Gumbel scores do not match candidate IDs",
        ));
    }
    let mut indices: Vec<usize> = (0..distribution.ids.len()).collect();
    indices.sort_by(|left, right| {
        ranking_scores[*right]
            .total_cmp(&ranking_scores[*left])
            .then_with(|| distribution.ids[*left].cmp(&distribution.ids[*right]))
    });
    Ok(indices
        .into_iter()
        .map(|index| distribution.ids[index])
        .collect())
}

pub fn validate_seed_search_target(
    distribution: &SparseDistribution,
    token_id: i64,
    options: DrawOptions<'_>,
) -> SamplingResult<()> {
    if token_id < 0 {
        return Err(SamplingError::editor(
            "draw token id must be a nonnegative integer",
        ));
    }
    if !distribution.ids.contains(&token_id) {
        return Err(SamplingError::editor(format!(
            "token {token_id} is outside the active truncated candidate set"
        )));
    }
    if options.kernel == "categorical" && distribution.probability(token_id) <= 0.0 {
        return Err(SamplingError::editor(format!(
            "token {token_id} has no selectable categorical mass"
        )));
    }
    if PERTURB_MAX_KERNELS.contains(&options.kernel) {
        validate_noise(options.perturb_noise_std, "perturb_noise_std")?;
    }
    if options.kernel == "student-t-max" {
        validate_student_t_df(options.student_t_df)?;
    }
    if options.kernel == "gaussian-max" && options.gaussian_noise_std == 0.0 {
        let mut current_options = options;
        current_options.gaussian_noise_std = 0.0;
        if draw_token(distribution, current_options)? != token_id {
            return Err(SamplingError::editor(
                "gaussian_noise_std=0 makes this token impossible to select",
            ));
        }
    }
    if PERTURB_MAX_KERNELS.contains(&options.kernel) && options.perturb_noise_std == 0.0 {
        let mut current_options = options;
        current_options.perturb_noise_std = 0.0;
        if draw_token(distribution, current_options)? != token_id {
            return Err(SamplingError::editor(
                "perturb_noise_std=0 makes this token impossible to select",
            ));
        }
    }
    if options.kernel == "uniform-max" && options.perturb_noise_std > 0.0 {
        let scores = validate_score_ids(distribution, "uniform-max")?;
        let other_scores: Vec<f64> = scores
            .iter()
            .zip(&distribution.ids)
            .filter_map(|(score, id)| (*id != token_id).then_some(*score))
            .collect();
        if let Some(max_other) = other_scores.iter().copied().reduce(f64::max) {
            let target_score = distribution
                .ids
                .iter()
                .position(|id| *id == token_id)
                .map(|index| scores[index])
                .unwrap_or(f64::NEG_INFINITY);
            let half_width = 3.0_f64.sqrt() * options.perturb_noise_std;
            if target_score + half_width <= max_other - half_width {
                return Err(SamplingError::editor(
                    "uniform-max bounded noise makes this token impossible to select",
                ));
            }
        }
    }
    if options.kernel == "gumbel-max" && options.gumbel_noise_scale == 0.0 {
        let mut current_options = options;
        current_options.gumbel_noise_scale = 0.0;
        if draw_token(distribution, current_options)? != token_id {
            return Err(SamplingError::editor(
                "gumbel_noise_scale=0 makes this token impossible to select",
            ));
        }
    }
    Ok(())
}

pub fn seed_search_candidate(
    distribution: &SparseDistribution,
    token_id: i64,
    seed: i64,
    options: DrawOptions<'_>,
) -> SamplingResult<bool> {
    validate_seed(seed)?;
    let mut candidate_options = options;
    candidate_options.seed = seed;
    Ok(draw_token(distribution, candidate_options)? == token_id)
}

fn make_distribution(
    ids: Vec<i64>,
    probabilities: Vec<f64>,
    scores: Option<Vec<f64>>,
) -> SparseDistribution {
    SparseDistribution {
        ids,
        probabilities,
        scores,
    }
}

#[pyfunction(name = "softmax")]
fn py_softmax(values: Vec<f64>) -> PyResult<Vec<f64>> {
    softmax(&values).map_err(into_pyerr)
}

#[pyfunction(name = "top_ids")]
fn py_top_ids(values: Vec<f64>, count: i64) -> PyResult<Vec<i64>> {
    top_ids(&values, count).map_err(into_pyerr)
}

#[pyfunction(name = "validated_logits")]
fn py_validated_logits(values: Vec<f64>) -> PyResult<Vec<f64>> {
    validated_logits(&values).map_err(into_pyerr)
}

#[pyfunction(name = "rank")]
fn py_rank(values: Vec<f64>, token_id: i64) -> PyResult<usize> {
    rank(&values, token_id).map_err(into_pyerr)
}

#[pyfunction(name = "apply_filter")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn py_apply_filter(
    adjusted: Vec<f64>,
    temperature: f64,
    top_k: Option<i64>,
    top_p: f64,
    min_p: f64,
    typical_p: f64,
    tail_free_z: f64,
) -> PyResult<(Vec<f64>, Vec<Option<Vec<i64>>>, bool, bool)> {
    apply_filter(
        &adjusted,
        temperature,
        top_k,
        top_p,
        min_p,
        typical_p,
        tail_free_z,
    )
    .map(|result| {
        (
            result.scaled_logits,
            result.stages.into_iter().collect(),
            result.greedy,
            result.unfiltered,
        )
    })
    .map_err(into_pyerr)
}

#[pyfunction(name = "probability")]
fn py_probability(ids: Vec<i64>, probabilities: Vec<f64>, token_id: i64) -> f64 {
    make_distribution(ids, probabilities, None).probability(token_id)
}

#[pyfunction(name = "position_uniform")]
fn py_position_uniform(seed: i64, fingerprint: &str, boundary: &str) -> PyResult<f64> {
    position_uniform(seed, fingerprint, boundary).map_err(into_pyerr)
}

#[pyfunction(name = "position_uniform_token")]
fn py_position_uniform_token(
    seed: i64,
    fingerprint: &str,
    boundary: &str,
    token_id: &str,
) -> PyResult<f64> {
    position_uniform_token(seed, fingerprint, boundary, token_id).map_err(into_pyerr)
}

#[pyfunction(name = "position_uniform_model_rank")]
fn py_position_uniform_model_rank(
    seed: i64,
    fingerprint: &str,
    boundary: &str,
    model_rank: &str,
) -> PyResult<f64> {
    position_uniform_model_rank(seed, fingerprint, boundary, model_rank).map_err(into_pyerr)
}

#[pyfunction(name = "conditional_gumbel_top_k")]
#[allow(clippy::too_many_arguments)]
fn py_conditional_gumbel_top_k(
    log_probabilities: Vec<f64>,
    count: usize,
    parent_score: f64,
    parent_log_probability: f64,
    seed: i64,
    fingerprint: &str,
    boundary: &str,
    prefix_token_ids: Vec<u64>,
) -> PyResult<Vec<(usize, f64)>> {
    conditional_gumbel_top_k(
        &log_probabilities,
        count,
        parent_score,
        parent_log_probability,
        seed,
        fingerprint,
        boundary,
        &prefix_token_ids,
    )
    .map_err(into_pyerr)
}

#[pyfunction(name = "gaussian_ranking_scores")]
fn py_gaussian_ranking_scores(
    ids: Vec<i64>,
    probabilities: Vec<f64>,
    scores: Option<Vec<f64>>,
    seed: i64,
    fingerprint: &str,
    boundary: &str,
    noise_std: f64,
) -> PyResult<Vec<f64>> {
    gaussian_ranking_scores(
        &make_distribution(ids, probabilities, scores),
        seed,
        fingerprint,
        boundary,
        noise_std,
    )
    .map_err(into_pyerr)
}

#[pyfunction(name = "perturbation_ranking_scores")]
#[allow(clippy::too_many_arguments)]
fn py_perturbation_ranking_scores(
    ids: Vec<i64>,
    probabilities: Vec<f64>,
    scores: Option<Vec<f64>>,
    seed: i64,
    fingerprint: &str,
    boundary: &str,
    kernel: &str,
    noise_std: f64,
    student_t_df: f64,
) -> PyResult<Vec<f64>> {
    perturbation_ranking_scores(
        &make_distribution(ids, probabilities, scores),
        seed,
        fingerprint,
        boundary,
        kernel,
        noise_std,
        student_t_df,
    )
    .map_err(into_pyerr)
}

#[pyfunction(name = "gumbel_ranking_scores")]
#[allow(clippy::too_many_arguments)]
fn py_gumbel_ranking_scores(
    ids: Vec<i64>,
    probabilities: Vec<f64>,
    scores: Option<Vec<f64>>,
    seed: i64,
    fingerprint: &str,
    boundary: &str,
    noise_address: &str,
    candidate_model_ranks: Option<Vec<String>>,
    noise_scale: f64,
) -> PyResult<Vec<f64>> {
    gumbel_ranking_scores(
        &make_distribution(ids, probabilities, scores),
        seed,
        fingerprint,
        boundary,
        noise_address,
        candidate_model_ranks.as_deref(),
        noise_scale,
    )
    .map_err(into_pyerr)
}

#[pyfunction(name = "draw_token")]
#[allow(clippy::too_many_arguments)]
fn py_draw_token(
    ids: Vec<i64>,
    probabilities: Vec<f64>,
    scores: Option<Vec<f64>>,
    seed: i64,
    fingerprint: &str,
    boundary: &str,
    kernel: &str,
    gaussian_noise_std: f64,
    perturb_noise_std: f64,
    student_t_df: f64,
    gumbel_noise_address: &str,
    candidate_model_ranks: Option<Vec<String>>,
    gumbel_noise_scale: f64,
) -> PyResult<i64> {
    draw_token(
        &make_distribution(ids, probabilities, scores),
        DrawOptions {
            seed,
            fingerprint,
            boundary,
            kernel,
            gaussian_noise_std,
            perturb_noise_std,
            student_t_df,
            gumbel_noise_address,
            candidate_model_ranks: candidate_model_ranks.as_deref(),
            gumbel_noise_scale,
        },
    )
    .map_err(into_pyerr)
}

#[pyfunction(name = "gumbel_winner")]
fn py_gumbel_winner(ids: Vec<i64>, ranking_scores: Vec<f64>) -> PyResult<i64> {
    winner(
        &make_distribution(ids, Vec::new(), None),
        &ranking_scores,
        "gumbel",
    )
    .map_err(into_pyerr)
}

#[pyfunction(name = "gaussian_winner")]
fn py_gaussian_winner(ids: Vec<i64>, ranking_scores: Vec<f64>) -> PyResult<i64> {
    winner(
        &make_distribution(ids, Vec::new(), None),
        &ranking_scores,
        "gaussian",
    )
    .map_err(into_pyerr)
}

#[pyfunction(name = "perturbation_winner")]
fn py_perturbation_winner(ids: Vec<i64>, ranking_scores: Vec<f64>) -> PyResult<i64> {
    winner(
        &make_distribution(ids, Vec::new(), None),
        &ranking_scores,
        "perturb",
    )
    .map_err(into_pyerr)
}

#[pyfunction(name = "gumbel_ranked_ids")]
#[allow(clippy::too_many_arguments)]
fn py_gumbel_ranked_ids(
    ids: Vec<i64>,
    probabilities: Vec<f64>,
    scores: Option<Vec<f64>>,
    seed: i64,
    fingerprint: &str,
    boundary: &str,
    noise_address: &str,
    candidate_model_ranks: Option<Vec<String>>,
    noise_scale: f64,
) -> PyResult<Vec<i64>> {
    let distribution = make_distribution(ids, probabilities, scores);
    let ranking = gumbel_ranking_scores(
        &distribution,
        seed,
        fingerprint,
        boundary,
        noise_address,
        candidate_model_ranks.as_deref(),
        noise_scale,
    )
    .map_err(into_pyerr)?;
    ranking_ids(&distribution, &ranking).map_err(into_pyerr)
}

#[pyfunction(name = "validate_seed_search_target")]
#[allow(clippy::too_many_arguments)]
fn py_validate_seed_search_target(
    ids: Vec<i64>,
    probabilities: Vec<f64>,
    scores: Option<Vec<f64>>,
    token_id: i64,
    current_seed: i64,
    fingerprint: &str,
    boundary: &str,
    kernel: &str,
    gaussian_noise_std: f64,
    perturb_noise_std: f64,
    student_t_df: f64,
    gumbel_noise_address: &str,
    candidate_model_ranks: Option<Vec<String>>,
    gumbel_noise_scale: f64,
) -> PyResult<()> {
    validate_seed_search_target(
        &make_distribution(ids, probabilities, scores),
        token_id,
        DrawOptions {
            seed: current_seed,
            fingerprint,
            boundary,
            kernel,
            gaussian_noise_std,
            perturb_noise_std,
            student_t_df,
            gumbel_noise_address,
            candidate_model_ranks: candidate_model_ranks.as_deref(),
            gumbel_noise_scale,
        },
    )
    .map_err(into_pyerr)
}

#[pyfunction(name = "seed_search_candidate")]
#[allow(clippy::too_many_arguments)]
fn py_seed_search_candidate(
    ids: Vec<i64>,
    probabilities: Vec<f64>,
    scores: Option<Vec<f64>>,
    token_id: i64,
    seed: i64,
    fingerprint: &str,
    boundary: &str,
    kernel: &str,
    gaussian_noise_std: f64,
    perturb_noise_std: f64,
    student_t_df: f64,
    gumbel_noise_address: &str,
    candidate_model_ranks: Option<Vec<String>>,
    gumbel_noise_scale: f64,
) -> PyResult<bool> {
    seed_search_candidate(
        &make_distribution(ids, probabilities, scores),
        token_id,
        seed,
        DrawOptions {
            seed,
            fingerprint,
            boundary,
            kernel,
            gaussian_noise_std,
            perturb_noise_std,
            student_t_df,
            gumbel_noise_address,
            candidate_model_ranks: candidate_model_ranks.as_deref(),
            gumbel_noise_scale,
        },
    )
    .map_err(into_pyerr)
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add(
        "NativeEditorError",
        module.py().get_type::<NativeEditorError>(),
    )?;
    module.add("RNG_SCHEME", RNG_SCHEME)?;
    module.add("MIN_SEED", MIN_SEED)?;
    module.add("MAX_SEED", MAX_SEED)?;
    module.add("DRAW_KERNELS", DRAW_KERNELS)?;
    module.add("PERTURB_MAX_KERNELS", PERTURB_MAX_KERNELS)?;
    module.add_function(wrap_pyfunction!(py_softmax, module)?)?;
    module.add_function(wrap_pyfunction!(py_top_ids, module)?)?;
    module.add_function(wrap_pyfunction!(py_validated_logits, module)?)?;
    module.add_function(wrap_pyfunction!(py_rank, module)?)?;
    module.add_function(wrap_pyfunction!(py_apply_filter, module)?)?;
    module.add_function(wrap_pyfunction!(py_probability, module)?)?;
    module.add_function(wrap_pyfunction!(py_position_uniform, module)?)?;
    module.add_function(wrap_pyfunction!(py_position_uniform_token, module)?)?;
    module.add_function(wrap_pyfunction!(py_position_uniform_model_rank, module)?)?;
    module.add_function(wrap_pyfunction!(py_conditional_gumbel_top_k, module)?)?;
    module.add_function(wrap_pyfunction!(py_gaussian_ranking_scores, module)?)?;
    module.add_function(wrap_pyfunction!(py_perturbation_ranking_scores, module)?)?;
    module.add_function(wrap_pyfunction!(py_gumbel_ranking_scores, module)?)?;
    module.add_function(wrap_pyfunction!(py_draw_token, module)?)?;
    module.add_function(wrap_pyfunction!(py_gumbel_winner, module)?)?;
    module.add_function(wrap_pyfunction!(py_gaussian_winner, module)?)?;
    module.add_function(wrap_pyfunction!(py_perturbation_winner, module)?)?;
    module.add_function(wrap_pyfunction!(py_gumbel_ranked_ids, module)?)?;
    module.add_function(wrap_pyfunction!(py_validate_seed_search_target, module)?)?;
    module.add_function(wrap_pyfunction!(py_seed_search_candidate, module)?)?;
    Ok(())
}

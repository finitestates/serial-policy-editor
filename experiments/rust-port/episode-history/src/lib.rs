//! Typed, storage-neutral episode history operations.
//!
//! Python values cross the extension boundary as one JSON document per
//! operation. The kernel below works with ordinary Rust enums and structs;
//! it does not know about SQLite, model backends, or the episode engine.

#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value, json};

const PHRASE_DEFAULT_MAX_TOKENS: u64 = 16;
const PHRASE_DEFAULT_MAX_SHIFT: f64 = 6.0;
const ACTION_ERROR_MARKER: &str = "__episode_history_action_error__";

#[derive(Clone, Debug, PartialEq)]
pub enum PolicyAction {
    Accept,
    SelectRawRank {
        rank: u64,
    },
    Write {
        text: String,
        mode: String,
    },
    Phrase {
        text: String,
        mode: String,
        force: bool,
        max_tokens: u64,
        max_shift: f64,
    },
    Hold {
        limit: u64,
        boundary: Option<String>,
    },
    EndGeneration,
    Reroll {
        seed: i64,
    },
    SetSampler {
        sampling: Value,
    },
}

impl PolicyAction {
    /// Parse the Python action record, including its documented legacy aliases.
    pub fn from_value(raw: &Value) -> Result<Self, KernelError> {
        let object = raw
            .as_object()
            .ok_or_else(|| KernelError::invalid_action("policy action must be a mapping"))?;
        let kind = object
            .get("kind")
            .and_then(Value::as_str)
            .filter(|kind| !kind.trim().is_empty())
            .ok_or_else(|| {
                KernelError::invalid_action("policy action kind must be a nonempty string")
            })?;

        match kind {
            "accept" => Ok(Self::Accept),
            "select" | "select-raw-rank" => {
                let raw_rank = selected(object, "rank", "selected_rank")
                    .and_then(exact_i128)
                    .ok_or_else(|| {
                        KernelError::invalid_action("select action has no valid raw rank")
                    })?;
                if raw_rank < 1 {
                    return Err(KernelError::invalid_action(
                        "raw rank must be a positive integer",
                    ));
                }
                let rank = u64::try_from(raw_rank).map_err(|_| {
                    KernelError::invalid_action("raw rank must be a positive integer")
                })?;
                Ok(Self::SelectRawRank { rank })
            }
            "insert" | "write" => {
                let text = selected(object, "text", "supplied_text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| KernelError::invalid_action("write action is malformed"))?;
                let mode = match selected(object, "mode", "insert_mode") {
                    None => "continuation",
                    Some(value) => value
                        .as_str()
                        .ok_or_else(|| KernelError::invalid_action("write action is malformed"))?,
                };
                validate_nonempty_text(text, "write text must be nonempty")?;
                validate_mode(mode, false)?;
                Ok(Self::Write {
                    text: text.to_owned(),
                    mode: mode.to_owned(),
                })
            }
            "check-phrase" | "force-phrase" => {
                let text = selected(object, "text", "supplied_text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| KernelError::invalid_action("phrase action is malformed"))?;
                let mode = match object.get("mode") {
                    None => "continuation",
                    Some(value) => value
                        .as_str()
                        .ok_or_else(|| KernelError::invalid_action("phrase action is malformed"))?,
                };
                let max_tokens = match object.get("max_tokens") {
                    None => PHRASE_DEFAULT_MAX_TOKENS,
                    Some(value) => {
                        let count = exact_i128(value).ok_or_else(|| {
                            KernelError::invalid_action(
                                "phrase max_tokens must be a positive integer",
                            )
                        })?;
                        if count < 1 {
                            return Err(KernelError::invalid_action(
                                "phrase max_tokens must be a positive integer",
                            ));
                        }
                        u64::try_from(count).map_err(|_| {
                            KernelError::invalid_action(
                                "phrase max_tokens must be a positive integer",
                            )
                        })?
                    }
                };
                let max_shift = match object.get("max_shift") {
                    None => PHRASE_DEFAULT_MAX_SHIFT,
                    Some(value) => value.as_f64().ok_or_else(|| {
                        KernelError::invalid_action("phrase max_shift must be a finite number")
                    })?,
                };
                validate_nonempty_text(text, "phrase text must be nonempty")?;
                validate_mode(mode, true)?;
                if !max_shift.is_finite() {
                    return Err(KernelError::invalid_action(
                        "phrase max_shift must be a finite number",
                    ));
                }
                if max_shift < 0.0 {
                    return Err(KernelError::invalid_action(
                        "phrase max_shift must be nonnegative",
                    ));
                }
                Ok(Self::Phrase {
                    text: text.to_owned(),
                    mode: mode.to_owned(),
                    force: kind == "force-phrase",
                    max_tokens,
                    max_shift,
                })
            }
            "hold" => {
                let raw_limit = selected(object, "limit", "requested_visible_tokens")
                    .and_then(exact_i128)
                    .ok_or_else(|| KernelError::invalid_action("hold action has no valid limit"))?;
                if raw_limit < 1 {
                    return Err(KernelError::invalid_action(
                        "hold limit must be a positive integer",
                    ));
                }
                let limit = u64::try_from(raw_limit).map_err(|_| {
                    KernelError::invalid_action("hold limit must be a positive integer")
                })?;
                let boundary = match object.get("boundary") {
                    None | Some(Value::Null) => None,
                    Some(Value::String(value)) => Some(value.clone()),
                    Some(_) => {
                        return Err(KernelError::invalid_action(
                            "hold boundary must be sentence, newline, or null",
                        ));
                    }
                };
                if boundary
                    .as_deref()
                    .is_some_and(|value| value != "sentence" && value != "newline")
                {
                    return Err(KernelError::invalid_action(
                        "hold boundary must be sentence, newline, or null",
                    ));
                }
                Ok(Self::Hold { limit, boundary })
            }
            "teacher-eog" | "end-generation" => Ok(Self::EndGeneration),
            "reroll" => {
                let seed = object.get("seed").and_then(Value::as_i64).ok_or_else(|| {
                    KernelError::invalid_action("reroll action has no valid seed")
                })?;
                Ok(Self::Reroll { seed })
            }
            "set-sampler" => {
                let sampling = object
                    .get("sampling")
                    .filter(|value| value.is_object())
                    .ok_or_else(|| {
                        KernelError::invalid_action(
                            "sampler change has no valid sampling configuration",
                        )
                    })?;
                Ok(Self::SetSampler {
                    sampling: sampling.clone(),
                })
            }
            _ => Err(KernelError::unsupported_action(kind)),
        }
    }

    /// Return the canonical Python `to_dict()` representation.
    pub fn to_value(&self) -> Value {
        match self {
            Self::Accept => json!({"kind": "accept"}),
            Self::SelectRawRank { rank } => json!({"kind": "select-raw-rank", "rank": rank}),
            Self::Write { text, mode } => json!({
                "kind": "write",
                "text": text,
                "mode": mode,
            }),
            Self::Phrase {
                text,
                mode,
                force,
                max_tokens,
                max_shift,
            } => json!({
                "kind": if *force { "force-phrase" } else { "check-phrase" },
                "text": text,
                "mode": mode,
                "max_tokens": max_tokens,
                "max_shift": max_shift,
            }),
            Self::Hold { limit, boundary } => json!({
                "kind": "hold",
                "limit": limit,
                "boundary": boundary,
            }),
            Self::EndGeneration => json!({"kind": "end-generation"}),
            Self::Reroll { seed } => json!({"kind": "reroll", "seed": seed}),
            Self::SetSampler { sampling } => json!({
                "kind": "set-sampler",
                "sampling": sampling,
            }),
        }
    }
}

impl Serialize for PolicyAction {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_value().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PolicyAction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        Self::from_value(&value).map_err(|error| {
            D::Error::custom(format!(
                "{ACTION_ERROR_MARKER}{}|{}",
                error.kind.wire_name(),
                error.message
            ))
        })
    }
}

fn selected<'a>(object: &'a Map<String, Value>, primary: &str, alias: &str) -> Option<&'a Value> {
    object.get(primary).or_else(|| object.get(alias))
}

fn exact_u64(value: &Value) -> Option<u64> {
    value.as_u64()
}

fn exact_i128(value: &Value) -> Option<i128> {
    let number = value.as_number()?;
    if let Some(value) = number.as_i64() {
        Some(i128::from(value))
    } else {
        number.as_u64().map(i128::from)
    }
}

fn validate_nonempty_text(text: &str, message: &str) -> Result<(), KernelError> {
    if text.is_empty() {
        Err(KernelError::invalid_action(message))
    } else {
        Ok(())
    }
}

fn validate_mode(mode: &str, phrase: bool) -> Result<(), KernelError> {
    if mode == "continuation" || mode == "exact" {
        Ok(())
    } else {
        Err(KernelError::invalid_action(if phrase {
            "phrase mode must be continuation or exact"
        } else {
            "write mode must be continuation or exact"
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplayExpectation {
    #[serde(default)]
    pub token_ids: Vec<i64>,
    #[serde(default)]
    pub terminal_token_id: Option<i64>,
    #[serde(default)]
    pub stop_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Divergence {
    pub boundary: u64,
    pub action_kind: String,
    pub reason: String,
    pub expected_token_id: Option<i64>,
    pub actual_token_id: Option<i64>,
    #[serde(default)]
    pub expected_stop_reason: Option<String>,
    #[serde(default)]
    pub actual_stop_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenEvidence {
    pub boundary: u64,
    pub sampling_boundary: i64,
    pub token_id: i64,
    pub text: String,
    pub proposal_token_id: i64,
    pub raw_model_nll: Option<f64>,
    pub raw_rank: Option<u64>,
    pub policy_rank: Option<u64>,
    pub decoder_probability: f64,
    pub proposal_agreement: bool,
    pub is_eog: bool,
    pub realized_visible: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionOutcome {
    pub action: PolicyAction,
    pub boundary_before: u64,
    pub boundary_after: u64,
    pub resolved_text: String,
    pub resolved_token_ids: Vec<i64>,
    pub visible_token_ids: Vec<i64>,
    pub terminal_token_id: Option<i64>,
    pub stop_reason: String,
    pub evidence: Vec<TokenEvidence>,
    #[serde(default = "completed_status")]
    pub status: String,
    #[serde(default)]
    pub divergence: Option<Divergence>,
    #[serde(default)]
    pub replay_eog_token_id: Option<i64>,
    #[serde(default)]
    pub diagnostics: Option<Value>,
}

fn completed_status() -> String {
    "completed".to_owned()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordedAttempt {
    pub ordinal: u64,
    pub action: PolicyAction,
    pub outcome: ActionOutcome,
    #[serde(default)]
    pub expectation: Option<ReplayExpectation>,
}

impl RecordedAttempt {
    pub fn requested_action(&self) -> &PolicyAction {
        &self.action
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EpisodeHistory {
    #[serde(default)]
    pub attempts: Vec<RecordedAttempt>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HistoryTruncation {
    pub requested_boundary: u64,
    pub retained: EpisodeHistory,
    pub discarded: Vec<RecordedAttempt>,
    pub partial: Option<RecordedAttempt>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidAction,
    UnsupportedActionKind,
    InvalidHistory,
    InvalidBoundary,
    InvalidRequest,
}

impl ErrorKind {
    fn wire_name(&self) -> &'static str {
        match self {
            Self::InvalidAction => "invalid-action",
            Self::UnsupportedActionKind => "unsupported-action-kind",
            Self::InvalidHistory => "invalid-history",
            Self::InvalidBoundary => "invalid-boundary",
            Self::InvalidRequest => "invalid-request",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelError {
    pub kind: ErrorKind,
    pub message: String,
}

impl KernelError {
    fn invalid_action(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidAction,
            message: message.into(),
        }
    }

    fn unsupported_action(kind: &str) -> Self {
        Self {
            kind: ErrorKind::UnsupportedActionKind,
            message: format!("unsupported policy action kind '{kind}'"),
        }
    }

    fn invalid_history(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidHistory,
            message: message.into(),
        }
    }

    fn invalid_boundary(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidBoundary,
            message: message.into(),
        }
    }

    fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidRequest,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for KernelError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for KernelError {}

impl EpisodeHistory {
    pub fn new(attempts: Vec<RecordedAttempt>) -> Result<Self, KernelError> {
        let history = Self { attempts };
        history.validate()?;
        Ok(history)
    }

    pub fn validate(&self) -> Result<(), KernelError> {
        let mut previous_ordinal = None;
        let mut previous_boundary = 0_u64;

        for (index, attempt) in self.attempts.iter().enumerate() {
            if attempt.action != attempt.outcome.action {
                return Err(KernelError::invalid_history(
                    "attempt action must match outcome action",
                ));
            }
            if previous_ordinal.is_some_and(|ordinal| attempt.ordinal <= ordinal) {
                return Err(KernelError::invalid_history(
                    "attempt ordinals must be strictly increasing",
                ));
            }

            let outcome = &attempt.outcome;
            let before = outcome.boundary_before;
            let after = outcome.boundary_after;
            if after < before {
                return Err(KernelError::invalid_history(
                    "attempt boundaries must be nonnegative and ordered",
                ));
            }
            if index == 0 && before != 0 {
                return Err(KernelError::invalid_history(
                    "history must begin at root-visible boundary zero",
                ));
            }
            if before != previous_boundary {
                return Err(KernelError::invalid_history(
                    "attempt boundaries must be contiguous",
                ));
            }
            let width = after - before;
            if usize::try_from(width).ok() != Some(outcome.visible_token_ids.len()) {
                return Err(KernelError::invalid_history(
                    "boundary width must equal the number of visible token ids",
                ));
            }

            let mut visible_ids = Vec::new();
            let mut visible_boundaries = Vec::new();
            for item in &outcome.evidence {
                if item.boundary < before {
                    return Err(KernelError::invalid_history(
                        "evidence boundaries must be root-relative",
                    ));
                }
                if item.realized_visible {
                    if item.boundary >= after {
                        return Err(KernelError::invalid_history(
                            "visible evidence must lie inside its visible boundary span",
                        ));
                    }
                    visible_ids.push(item.token_id);
                    visible_boundaries.push(item.boundary);
                } else if item.boundary > after {
                    return Err(KernelError::invalid_history(
                        "non-visible evidence must not pass the outcome boundary",
                    ));
                }
            }
            if visible_ids != outcome.visible_token_ids {
                return Err(KernelError::invalid_history(
                    "visible evidence token ids must match the outcome visible ids",
                ));
            }
            if visible_boundaries.len() != outcome.visible_token_ids.len()
                || visible_boundaries
                    .iter()
                    .enumerate()
                    .any(|(offset, boundary)| {
                        u64::try_from(offset)
                            .ok()
                            .and_then(|offset| before.checked_add(offset))
                            != Some(*boundary)
                    })
            {
                return Err(KernelError::invalid_history(
                    "visible evidence boundaries must be contiguous and ordered",
                ));
            }

            previous_ordinal = Some(attempt.ordinal);
            previous_boundary = after;
        }
        Ok(())
    }

    pub fn current_boundary(&self) -> u64 {
        self.attempts
            .last()
            .map(|attempt| attempt.outcome.boundary_after)
            .unwrap_or(0)
    }

    pub fn visible_token_ids(&self) -> Vec<i64> {
        self.attempts
            .iter()
            .flat_map(|attempt| attempt.outcome.visible_token_ids.iter().copied())
            .collect()
    }

    pub fn visible_token_evidence(&self) -> Vec<TokenEvidence> {
        self.attempts
            .iter()
            .flat_map(|attempt| {
                attempt
                    .outcome
                    .evidence
                    .iter()
                    .filter(|item| item.realized_visible)
                    .cloned()
            })
            .collect()
    }

    pub fn visible_text(&self) -> String {
        self.attempts
            .iter()
            .flat_map(|attempt| attempt.outcome.evidence.iter())
            .filter(|item| item.realized_visible)
            .map(|item| item.text.as_str())
            .collect()
    }

    pub fn truncate(
        &self,
        boundary: u64,
        include_boundary_events: bool,
    ) -> Result<HistoryTruncation, KernelError> {
        self.validate()?;
        let current = self.current_boundary();
        if boundary > current {
            return Err(KernelError::invalid_boundary(format!(
                "retained boundary must be between 0 and {current}"
            )));
        }

        let mut retained = Vec::new();
        let mut discarded = Vec::new();
        let mut partial = None;

        for attempt in &self.attempts {
            let before = attempt.outcome.boundary_before;
            let after = attempt.outcome.boundary_after;

            if after < boundary || (after == boundary && before < boundary) {
                retained.push(attempt.clone());
                continue;
            }
            if include_boundary_events && before == boundary && after == boundary {
                retained.push(attempt.clone());
                continue;
            }
            if before < boundary && boundary < after {
                partial = Some(attempt.clone());
                if let Some(transformed) = partial_attempt(attempt, boundary)? {
                    retained.push(transformed);
                } else {
                    discarded.push(attempt.clone());
                }
                continue;
            }
            discarded.push(attempt.clone());
        }

        Ok(HistoryTruncation {
            requested_boundary: boundary,
            retained: EpisodeHistory::new(retained)?,
            discarded,
            partial,
        })
    }
}

fn partial_attempt(
    attempt: &RecordedAttempt,
    boundary: u64,
) -> Result<Option<RecordedAttempt>, KernelError> {
    let original = &attempt.outcome;
    let count = usize::try_from(boundary - original.boundary_before)
        .map_err(|_| KernelError::invalid_history("partial visible count is too large"))?;
    let retained_evidence: Vec<_> = original
        .evidence
        .iter()
        .filter(|item| item.realized_visible)
        .take(count)
        .cloned()
        .collect();
    let visible_ids: Vec<_> = retained_evidence.iter().map(|item| item.token_id).collect();
    if visible_ids.is_empty() {
        return Ok(None);
    }
    let retained_text: String = retained_evidence
        .iter()
        .map(|item| item.text.as_str())
        .collect();
    let (action, stop_reason) = match &attempt.action {
        PolicyAction::Write { .. } | PolicyAction::Phrase { .. } => (
            PolicyAction::Write {
                text: retained_text.clone(),
                mode: "exact".to_owned(),
            },
            "completed".to_owned(),
        ),
        _ => (
            PolicyAction::Hold {
                limit: u64::try_from(visible_ids.len())
                    .map_err(|_| KernelError::invalid_history("partial hold is too large"))?,
                boundary: None,
            },
            "requested-length".to_owned(),
        ),
    };

    let outcome = ActionOutcome {
        action: action.clone(),
        boundary_before: original.boundary_before,
        boundary_after: boundary,
        resolved_text: retained_text,
        resolved_token_ids: visible_ids.clone(),
        visible_token_ids: visible_ids.clone(),
        terminal_token_id: None,
        stop_reason: stop_reason.clone(),
        evidence: retained_evidence,
        status: "completed".to_owned(),
        divergence: None,
        replay_eog_token_id: None,
        diagnostics: None,
    };
    let expectation = ReplayExpectation {
        token_ids: visible_ids,
        terminal_token_id: None,
        stop_reason: Some(stop_reason),
    };
    Ok(Some(RecordedAttempt {
        ordinal: attempt.ordinal,
        action,
        outcome,
        expectation: Some(expectation),
    }))
}

/// Apply the sampler-state effect of one action while retaining all other
/// fields. Sampler configuration validation belongs to Python.
pub fn sampler_after_action(sampling: &Value, action: &PolicyAction) -> Result<Value, KernelError> {
    match action {
        PolicyAction::SetSampler { sampling } => Ok(sampling.clone()),
        PolicyAction::Reroll { seed } => {
            let mut next = sampling.as_object().cloned().ok_or_else(|| {
                KernelError::invalid_request("sampling configuration must be a mapping")
            })?;
            next.insert("seed".to_owned(), Value::Number((*seed).into()));
            Ok(Value::Object(next))
        }
        _ => Ok(sampling.clone()),
    }
}

/// Run one whole operation and return a JSON protocol envelope.
pub fn execute_request(request: &Value) -> Value {
    match execute_request_inner(request) {
        Ok(result) => json!({"ok": true, "result": result}),
        Err(error) => json!({
            "ok": false,
            "error": {
                "kind": error.kind.wire_name(),
                "message": error.message,
            },
        }),
    }
}

fn execute_request_inner(request: &Value) -> Result<Value, KernelError> {
    let object = request
        .as_object()
        .ok_or_else(|| KernelError::invalid_request("operation request must be a mapping"))?;
    let operation = object
        .get("operation")
        .and_then(Value::as_str)
        .ok_or_else(|| KernelError::invalid_request("operation must be a string"))?;

    match operation {
        "action" => {
            let raw = object
                .get("action")
                .ok_or_else(|| KernelError::invalid_request("action is required"))?;
            Ok(PolicyAction::from_value(raw)?.to_value())
        }
        "sampler-after-action" => {
            let sampling = object
                .get("sampling")
                .filter(|value| value.is_object())
                .ok_or_else(|| {
                    KernelError::invalid_request("sampling configuration must be a mapping")
                })?;
            let action = object
                .get("action")
                .ok_or_else(|| KernelError::invalid_request("action is required"))?;
            let action = PolicyAction::from_value(action)?;
            sampler_after_action(sampling, &action)
        }
        "visible" => {
            let history = parse_history(object.get("history"))?;
            history.validate()?;
            Ok(json!({
                "current_boundary": history.current_boundary(),
                "visible_token_ids": history.visible_token_ids(),
                "visible_token_evidence": history.visible_token_evidence(),
                "visible_text": history.visible_text(),
            }))
        }
        "truncate" => {
            let history = parse_history(object.get("history"))?;
            let boundary = object.get("boundary").and_then(exact_u64).ok_or_else(|| {
                KernelError::invalid_boundary("retained boundary must be a nonnegative integer")
            })?;
            let include_boundary_events = match object.get("include_boundary_events") {
                None => false,
                Some(Value::Bool(value)) => *value,
                Some(_) => {
                    return Err(KernelError::invalid_request(
                        "include_boundary_events must be a boolean",
                    ));
                }
            };
            let truncation = history.truncate(boundary, include_boundary_events)?;
            serde_json::to_value(truncation)
                .map_err(|error| KernelError::invalid_history(error.to_string()))
        }
        _ => Err(KernelError::invalid_request(format!(
            "unsupported history operation '{operation}'"
        ))),
    }
}

fn parse_history(raw: Option<&Value>) -> Result<EpisodeHistory, KernelError> {
    let value = raw.ok_or_else(|| KernelError::invalid_request("history is required"))?;
    let history: EpisodeHistory = serde_json::from_value(value.clone()).map_err(|error| {
        let message = error.to_string();
        if let Some(action_error) = message.strip_prefix(ACTION_ERROR_MARKER)
            && let Some((kind, message)) = action_error.split_once('|')
        {
            let kind = match kind {
                "invalid-action" => ErrorKind::InvalidAction,
                "unsupported-action-kind" => ErrorKind::UnsupportedActionKind,
                _ => ErrorKind::InvalidHistory,
            };
            return KernelError {
                kind,
                message: message.to_owned(),
            };
        }
        let kind = if message.contains("unsupported policy action kind") {
            ErrorKind::UnsupportedActionKind
        } else {
            ErrorKind::InvalidHistory
        };
        KernelError { kind, message }
    })?;
    Ok(history)
}

/// PyO3 entry point. It accepts and returns one JSON string so history records
/// cross the boundary together rather than one field at a time.
#[cfg(feature = "python")]
#[pyfunction]
fn execute_json(input: &str) -> String {
    let response = match serde_json::from_str::<Value>(input) {
        Ok(request) => execute_request(&request),
        Err(error) => json!({
            "ok": false,
            "error": {
                "kind": "invalid-request",
                "message": error.to_string(),
            },
        }),
    };
    serde_json::to_string(&response).unwrap_or_else(|error| {
        format!(
            "{{\"ok\":false,\"error\":{{\"kind\":\"invalid-request\",\"message\":{}}}}}",
            serde_json::to_string(&error.to_string())
                .unwrap_or_else(|_| "\"serialization error\"".to_owned())
        )
    })
}

#[cfg(feature = "python")]
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(execute_json, module)?)?;
    Ok(())
}

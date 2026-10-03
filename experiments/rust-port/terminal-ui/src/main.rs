//! A small Rust terminal process for the existing Python PTY/pyte harness.
//!
//! The default mode remains the terminal smoke process. `--choice` runs one
//! fixture-backed Choice turn through the sampler and episode-history crates.

use std::collections::HashMap;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

mod worker;
use worker::BackendWorker;

use episode_history_native::{
    ActionOutcome, EpisodeHistory, PolicyAction, RecordedAttempt, TokenEvidence,
};
use rust_sampler_native::{
    DrawOptions, SparseDistribution, apply_filter, draw_token, rank, softmax,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use unicode_width::UnicodeWidthChar;

const STARTUP: &[u8] = b"\x1b[?1049h\x1b[2J\x1b[?25h";
const FRAME_START: &[u8] = b"\x1b[?2026h";
const FRAME_END: &[u8] = b"\x1b[?2026l";
const INPUT_PREFIX: &str = "Input > ";
const CHOICE_PREFIX: &str = "Choice > ";
// The root prompt is the pre-0 model context; its token-prefix fingerprint
// identifies the stream while generated-token history starts at boundary 0.
const REAL_PROMPT: &str = "A short list of everyday objects:";
const REAL_TOP_K: i64 = 5;
const REAL_SEED: i64 = 17;

static RESIZE_PENDING: AtomicBool = AtomicBool::new(false);

extern "C" fn on_resize(_signal: libc::c_int) {
    RESIZE_PENDING.store(true, Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Size {
    columns: usize,
    rows: usize,
}

#[derive(Serialize)]
struct FrameRecord {
    sequence: u64,
    checkpoint: String,
    decision_index: Option<usize>,
    size: [usize; 2],
    lines: Vec<String>,
    cursor: Option<[usize; 2]>,
    end_offset: usize,
}

struct FrameContent {
    checkpoint: String,
    decision_index: Option<usize>,
    lines: Vec<String>,
    cursor: Option<[usize; 2]>,
}

struct TerminalGuard {
    original: libc::termios,
    mode_changed: bool,
    old_winch_handler: Option<libc::sighandler_t>,
    screen_entered: bool,
}

impl TerminalGuard {
    fn enter() -> io::Result<(Self, usize)> {
        // SAFETY: `original` is written by tcgetattr before it is read below.
        let mut original = unsafe { std::mem::zeroed::<libc::termios>() };
        // SAFETY: the pointer is valid and points to writable termios storage.
        if unsafe { libc::tcgetattr(libc::STDIN_FILENO, &mut original) } != 0 {
            return Err(io::Error::last_os_error());
        }

        let mut guard = Self {
            original,
            mode_changed: false,
            old_winch_handler: None,
            screen_entered: false,
        };

        let mut raw = original;
        raw.c_iflag &= !(libc::BRKINT | libc::ICRNL | libc::INPCK | libc::ISTRIP | libc::IXON);
        raw.c_oflag &= !libc::OPOST;
        raw.c_cflag |= libc::CS8;
        raw.c_lflag &= !(libc::ECHO | libc::ICANON | libc::IEXTEN | libc::ISIG);
        raw.c_cc[libc::VMIN] = 1;
        raw.c_cc[libc::VTIME] = 0;

        // SAFETY: `raw` is a valid termios value for this controlling PTY.
        if unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &raw) } != 0 {
            return Err(io::Error::last_os_error());
        }
        guard.mode_changed = true;

        RESIZE_PENDING.store(false, Ordering::Relaxed);
        // SAFETY: the handler only performs a lock-free atomic store.
        let old_handler =
            unsafe { libc::signal(libc::SIGWINCH, on_resize as *const () as libc::sighandler_t) };
        if old_handler == libc::SIG_ERR {
            return Err(io::Error::last_os_error());
        }
        guard.old_winch_handler = Some(old_handler);

        guard.screen_entered = true;
        let mut stdout = io::stdout().lock();
        stdout.write_all(STARTUP)?;
        stdout.flush()?;
        Ok((guard, STARTUP.len()))
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if let Some(handler) = self.old_winch_handler {
            // SAFETY: restoring the process's previous SIGWINCH handler.
            unsafe {
                libc::signal(libc::SIGWINCH, handler);
            }
        }
        if self.mode_changed {
            // SAFETY: `original` was read from this same terminal with tcgetattr.
            unsafe {
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &self.original);
            }
        }
        if self.screen_entered {
            let mut stdout = io::stdout().lock();
            let _ = stdout.write_all(b"\x1b[?25h\x1b[?1049l");
            let _ = stdout.flush();
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct FixtureContext {
    boundary: u64,
    token_ids: Vec<i64>,
    token_text: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct FixtureSampler {
    temperature: f64,
    top_k: Option<i64>,
    top_p: f64,
    min_p: f64,
    typical_p: f64,
    tail_free_z: f64,
    draw_kernel: String,
    #[serde(default = "default_one")]
    gaussian_noise_std: f64,
    #[serde(default = "default_one")]
    perturb_noise_std: f64,
    #[serde(default = "default_student_t_df")]
    student_t_df: f64,
    seed: i64,
    stream_fingerprint: String,
    boundary: i64,
    #[serde(default = "default_token_id_address")]
    gumbel_noise_address: String,
    #[serde(default = "default_one")]
    gumbel_noise_scale: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct FixtureInput {
    context: FixtureContext,
    candidate_token_ids: Vec<i64>,
    token_text: HashMap<String, String>,
    logits: Vec<f64>,
    sampler: FixtureSampler,
}

#[derive(Debug, Deserialize)]
struct FixtureFile {
    fixture_id: String,
    input: FixtureInput,
}

#[derive(Clone, Debug)]
struct ChoiceCandidate {
    rank: usize,
    token_id: i64,
    text: String,
    probability: f64,
    is_eog: bool,
}

#[derive(Clone, Debug)]
struct AcceptedTurn {
    action: PolicyAction,
    token_id: i64,
    evidence: TokenEvidence,
    boundary_after: u64,
}

struct ChoiceState {
    fixture_id: String,
    profile_name: String,
    is_real_model: bool,
    turn_index: usize,
    root_token_ids: Vec<i64>,
    stream_fingerprint: String,
    tokenizer_id: String,
    context_text: String,
    context_boundary: u64,
    sampler: FixtureSampler,
    candidates: Vec<ChoiceCandidate>,
    proposal_token_id: i64,
    history: EpisodeHistory,
    accepted: Option<AcceptedTurn>,
    input: String,
    status: Option<String>,
}

struct SmokeState {
    input: String,
    last_submitted: String,
}

enum AppState {
    Smoke(SmokeState),
    Choice(Box<ChoiceState>),
}

enum SubmitResult {
    Exit,
    Redraw(Option<Value>),
    Advance(Option<Value>),
}

impl ChoiceState {
    fn load() -> io::Result<Self> {
        let fixture: FixtureFile =
            serde_json::from_str(include_str!("../fixtures/choice-turn.json"))
                .map_err(invalid_data)?;
        let fixture_id = fixture.fixture_id;
        let input = fixture.input;
        if input.context.token_ids.len() != input.context.token_text.len()
            || usize::try_from(input.context.boundary).ok() != Some(input.context.token_ids.len())
            || input.sampler.boundary < 0
            || u64::try_from(input.sampler.boundary).ok() != Some(input.context.boundary)
        {
            return Err(invalid_data(
                "fixture context and sampling boundary disagree",
            ));
        }

        let filtered = apply_filter(
            &input.logits,
            input.sampler.temperature,
            input.sampler.top_k,
            input.sampler.top_p,
            input.sampler.min_p,
            input.sampler.typical_p,
            input.sampler.tail_free_z,
        )
        .map_err(invalid_data)?;
        let candidate_ids = filtered.stages[5]
            .as_ref()
            .ok_or_else(|| invalid_data("candidate filter did not return a final set"))?;
        if candidate_ids != &input.candidate_token_ids {
            return Err(invalid_data(
                "fixture candidate IDs do not match the Rust filtered candidate view",
            ));
        }
        let candidate_scores = candidate_ids
            .iter()
            .map(|token_id| {
                let index = usize::try_from(*token_id)
                    .map_err(|_| invalid_data("fixture candidate ID must be nonnegative"))?;
                filtered
                    .scaled_logits
                    .get(index)
                    .copied()
                    .ok_or_else(|| invalid_data("fixture candidate ID is outside logits"))
            })
            .collect::<io::Result<Vec<_>>>()?;
        let probabilities = softmax(&candidate_scores).map_err(invalid_data)?;
        let distribution = SparseDistribution {
            ids: candidate_ids.clone(),
            probabilities: probabilities.clone(),
            scores: Some(candidate_scores),
        };
        let boundary = input.sampler.boundary.to_string();
        let proposal_token_id = draw_token(
            &distribution,
            DrawOptions {
                seed: input.sampler.seed,
                fingerprint: &input.sampler.stream_fingerprint,
                boundary: &boundary,
                kernel: &input.sampler.draw_kernel,
                gaussian_noise_std: input.sampler.gaussian_noise_std,
                perturb_noise_std: input.sampler.perturb_noise_std,
                student_t_df: input.sampler.student_t_df,
                gumbel_noise_address: &input.sampler.gumbel_noise_address,
                candidate_model_ranks: None,
                gumbel_noise_scale: input.sampler.gumbel_noise_scale,
            },
        )
        .map_err(invalid_data)?;

        let candidates = candidate_ids
            .iter()
            .zip(probabilities)
            .map(|(token_id, probability)| {
                let text = input
                    .token_text
                    .get(&token_id.to_string())
                    .cloned()
                    .ok_or_else(|| invalid_data("fixture is missing candidate token text"))?;
                let candidate_rank = rank(&input.logits, *token_id).map_err(invalid_data)?;
                Ok(ChoiceCandidate {
                    rank: candidate_rank,
                    token_id: *token_id,
                    text,
                    probability,
                    is_eog: false,
                })
            })
            .collect::<io::Result<Vec<_>>>()?;

        let context_text = input.context.token_text.concat();
        let history = context_history(&input.context, &context_text)?;
        let root_token_ids = input.context.token_ids.clone();
        let stream_fingerprint = input.sampler.stream_fingerprint.clone();
        Ok(Self {
            fixture_id,
            profile_name: "fixture".to_owned(),
            is_real_model: false,
            turn_index: 0,
            root_token_ids,
            stream_fingerprint,
            tokenizer_id: "fixture".to_owned(),
            context_text,
            context_boundary: input.context.boundary,
            sampler: input.sampler,
            candidates,
            proposal_token_id,
            history,
            accepted: None,
            input: String::new(),
            status: None,
        })
    }

    fn load_real(worker: &mut BackendWorker) -> io::Result<Self> {
        let profile = worker
            .hello()
            .get("profile")
            .ok_or_else(|| invalid_data("worker startup omitted profile metadata"))?;
        let profile_name = profile
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_data("worker profile omitted its name"))?
            .to_owned();
        let tokenizer_id = worker
            .hello()
            .get("tokenizer_id")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_data("worker startup omitted tokenizer identity"))?
            .to_owned();
        let tokenized = worker.tokenize(REAL_PROMPT)?;
        let root_token_ids = tokenized
            .get("token_ids")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_data("worker tokenize response omitted token IDs"))?
            .iter()
            .map(|value| {
                value
                    .as_i64()
                    .ok_or_else(|| invalid_data("worker returned an invalid token ID"))
            })
            .collect::<io::Result<Vec<_>>>()?;
        if root_token_ids.is_empty() {
            return Err(invalid_data(
                "backend tokenizer returned an empty root prefix",
            ));
        }
        let response_tokenizer = tokenized
            .get("tokenizer_id")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_data("worker tokenize response omitted tokenizer identity"))?;
        let fingerprint = tokenized
            .get("stream_fingerprint")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_data("worker tokenize response omitted stream fingerprint"))?
            .to_owned();
        if response_tokenizer != tokenizer_id {
            return Err(invalid_data(
                "tokenizer identity changed during tokenization",
            ));
        }
        let sampler = real_sampler(0, fingerprint.clone());
        worker.reset(&root_token_ids)?;
        let logits = worker.logits(0)?;
        let context_text = worker.render(&root_token_ids)?;
        let history = EpisodeHistory::default();
        let (candidates, proposal_token_id) = real_candidates(worker, &logits, &sampler)?;
        Ok(Self {
            fixture_id: "real-model-choice-v1".to_owned(),
            profile_name,
            is_real_model: true,
            turn_index: 0,
            root_token_ids,
            stream_fingerprint: fingerprint,
            tokenizer_id,
            context_text,
            context_boundary: 0,
            sampler,
            candidates,
            proposal_token_id,
            history,
            accepted: None,
            input: String::new(),
            status: None,
        })
    }

    fn advance_real(&mut self, worker: &mut BackendWorker) -> io::Result<()> {
        let accepted = self
            .accepted
            .as_ref()
            .ok_or_else(|| invalid_data("cannot advance before accepting a Choice"))?;
        if self.turn_index != 0 || accepted.evidence.is_eog {
            return Err(invalid_data(
                "real-model Choice cannot advance after this outcome",
            ));
        }
        let prefix = self.model_prefix_token_ids();
        let boundary = self.history.current_boundary();
        let next_sampling_boundary =
            i64::try_from(boundary).map_err(|_| invalid_data("visible boundary is too large"))?;
        let logits = worker.logits(boundary)?;
        self.context_text = worker.render(&prefix)?;
        self.context_boundary = boundary;
        self.sampler.boundary = next_sampling_boundary;
        let (candidates, proposal_token_id) = real_candidates(worker, &logits, &self.sampler)?;
        self.candidates = candidates;
        self.proposal_token_id = proposal_token_id;
        self.turn_index = 1;
        self.accepted = None;
        self.input.clear();
        self.status = None;
        Ok(())
    }

    fn model_prefix_token_ids(&self) -> Vec<i64> {
        if self.is_real_model {
            let mut prefix = self.root_token_ids.clone();
            prefix.extend(self.history.visible_token_ids());
            prefix
        } else {
            self.history.visible_token_ids()
        }
    }

    fn handle_character(&mut self, byte: u8) -> bool {
        if byte == b'q' && self.input.is_empty() {
            return true;
        }
        if self.accepted.is_some() {
            return false;
        }
        if (0x20..=0x7e).contains(&byte) {
            self.input.push(char::from(byte));
            self.status = None;
        }
        false
    }

    fn backspace(&mut self) {
        if self.accepted.is_none() {
            self.input.pop();
            self.status = None;
        }
    }

    fn submit(&mut self, worker: Option<&mut BackendWorker>) -> io::Result<SubmitResult> {
        let command = self.input.trim().to_owned();
        if self.accepted.is_some() {
            return if matches!(command.as_str(), "q" | "quit" | "exit") {
                Ok(SubmitResult::Exit)
            } else {
                self.input.clear();
                self.status = Some("Choice is already committed".to_owned());
                Ok(SubmitResult::Redraw(None))
            };
        }
        if matches!(command.as_str(), "q" | "quit" | "exit") {
            return Ok(SubmitResult::Exit);
        }

        let action = if command.is_empty() || command == "accept" {
            PolicyAction::from_value(&json!({"kind": "accept"})).map_err(invalid_data)?
        } else if let Ok(selected_rank) = command.parse::<u64>() {
            PolicyAction::from_value(&json!({"kind": "select", "rank": selected_rank}))
                .map_err(invalid_data)?
        } else {
            self.input.clear();
            self.status = Some("Use accept, a candidate rank, or q".to_owned());
            return Ok(SubmitResult::Redraw(None));
        };

        let selected_token_id = match &action {
            PolicyAction::Accept => self.proposal_token_id,
            PolicyAction::SelectRawRank { rank } => self
                .candidates
                .iter()
                .find(|candidate| u64::try_from(candidate.rank).ok() == Some(*rank))
                .map(|candidate| candidate.token_id)
                .ok_or_else(|| invalid_data("selected raw rank is outside the candidate view"))?,
            _ => return Err(invalid_data("Choice submit created an unexpected action")),
        };
        let candidate = self
            .candidates
            .iter()
            .find(|candidate| candidate.token_id == selected_token_id)
            .ok_or_else(|| invalid_data("selected token is outside the candidate view"))?;
        let prefix_token_ids = self.model_prefix_token_ids();
        let decision_context_text = self.context_text.clone();
        let before = self.history.current_boundary();
        let after = before
            .checked_add(1)
            .ok_or_else(|| invalid_data("visible history boundary overflow"))?;
        let evidence = TokenEvidence {
            boundary: before,
            sampling_boundary: self.sampler.boundary,
            token_id: selected_token_id,
            text: candidate.text.clone(),
            proposal_token_id: self.proposal_token_id,
            raw_model_nll: None,
            raw_rank: None,
            policy_rank: None,
            decoder_probability: candidate.probability,
            proposal_agreement: selected_token_id == self.proposal_token_id,
            is_eog: candidate.is_eog,
            realized_visible: true,
        };
        let outcome = ActionOutcome {
            action: action.clone(),
            boundary_before: before,
            boundary_after: after,
            resolved_text: candidate.text.clone(),
            resolved_token_ids: vec![selected_token_id],
            visible_token_ids: vec![selected_token_id],
            terminal_token_id: candidate.is_eog.then_some(selected_token_id),
            stop_reason: if candidate.is_eog {
                "eog".to_owned()
            } else {
                "completed".to_owned()
            },
            evidence: vec![evidence.clone()],
            status: if candidate.is_eog {
                "eog".to_owned()
            } else {
                "completed".to_owned()
            },
            divergence: None,
            replay_eog_token_id: None,
            diagnostics: None,
        };
        let mut attempts = self.history.attempts.clone();
        let ordinal = u64::try_from(attempts.len())
            .map_err(|_| invalid_data("too many fixture history attempts"))?;
        attempts.push(RecordedAttempt {
            ordinal,
            action: action.clone(),
            outcome,
            expectation: None,
        });
        self.history = EpisodeHistory::new(attempts).map_err(invalid_data)?;
        if self.is_real_model {
            let worker =
                worker.ok_or_else(|| invalid_data("real-model Choice has no backend worker"))?;
            if !candidate.is_eog {
                worker.eval(&[selected_token_id])?;
            }
            self.context_text = worker.render(&self.model_prefix_token_ids())?;
        }
        self.accepted = Some(AcceptedTurn {
            action: action.clone(),
            token_id: selected_token_id,
            evidence: evidence.clone(),
            boundary_after: after,
        });
        self.input.clear();
        self.status = None;

        let history_record = serde_json::to_value(&self.history).map_err(invalid_data)?;
        let mut semantic_record = json!({
            "schema_version": 1,
            "fixture_id": self.fixture_id,
            "checkpoint": "choice.accepted",
            "command": command,
            "action": action.to_value(),
            "proposal_token_id": self.proposal_token_id,
            "selected_token_id": selected_token_id,
            "boundary_before": before,
            "boundary_after": after,
            "evidence": evidence,
            "visible_token_ids": self.history.visible_token_ids(),
            "visible_text": self.history.visible_text(),
            "history": history_record,
        });
        if self.is_real_model {
            semantic_record["checkpoint"] = json!("choice.real.accepted");
            semantic_record["profile_name"] = json!(self.profile_name);
            semantic_record["decision_index"] = json!(self.turn_index);
            semantic_record["root_token_ids"] = json!(self.root_token_ids);
            semantic_record["prefix_token_ids"] = json!(prefix_token_ids);
            semantic_record["stream_fingerprint"] = json!(self.stream_fingerprint);
            semantic_record["tokenizer_id"] = json!(self.tokenizer_id);
            semantic_record["sampling_boundary"] = json!(self.sampler.boundary);
            semantic_record["candidate_token_ids"] = json!(
                self.candidates
                    .iter()
                    .map(|item| item.token_id)
                    .collect::<Vec<_>>()
            );
            semantic_record["candidate_raw_ranks"] = json!(
                self.candidates
                    .iter()
                    .map(|item| item.rank)
                    .collect::<Vec<_>>()
            );
            semantic_record["candidate_texts"] = json!(
                self.candidates
                    .iter()
                    .map(|item| item.text.clone())
                    .collect::<Vec<_>>()
            );
            semantic_record["candidate_probabilities"] = json!(
                self.candidates
                    .iter()
                    .map(|item| item.probability)
                    .collect::<Vec<_>>()
            );
            semantic_record["candidate_is_eog"] = json!(
                self.candidates
                    .iter()
                    .map(|item| item.is_eog)
                    .collect::<Vec<_>>()
            );
            semantic_record["context_text"] = json!(self.context_text);
            semantic_record["decision_context_text"] = json!(decision_context_text);
            semantic_record["sampler"] = json!({
                "temperature": self.sampler.temperature,
                "top_k": self.sampler.top_k,
                "top_p": self.sampler.top_p,
                "min_p": self.sampler.min_p,
                "typical_p": self.sampler.typical_p,
                "tail_free_z": self.sampler.tail_free_z,
                "draw_kernel": self.sampler.draw_kernel,
                "gaussian_noise_std": self.sampler.gaussian_noise_std,
                "perturb_noise_std": self.sampler.perturb_noise_std,
                "student_t_df": self.sampler.student_t_df,
                "seed": self.sampler.seed,
                "gumbel_noise_address": self.sampler.gumbel_noise_address,
                "gumbel_noise_scale": self.sampler.gumbel_noise_scale,
            });
            if self.turn_index == 0 && !candidate.is_eog {
                return Ok(SubmitResult::Advance(Some(semantic_record)));
            }
        }
        Ok(SubmitResult::Redraw(Some(semantic_record)))
    }

    fn render(&self, size: Size) -> FrameContent {
        if self.is_real_model {
            return self.render_real_model(size);
        }
        let mut lines = if let Some(accepted) = &self.accepted {
            vec![
                "Rust Choice / fixture turn".to_owned(),
                format!("Context @ boundary {}", accepted.boundary_after),
                self.history.visible_text(),
                format!("Action: {}", action_name(&accepted.action)),
                format!(
                    "Accepted token: {} / \"{}\"",
                    accepted.token_id, accepted.evidence.text
                ),
                format!(
                    "Evidence: boundary {} / sampling {} / agreement {}",
                    accepted.evidence.boundary,
                    accepted.evidence.sampling_boundary,
                    if accepted.evidence.proposal_agreement {
                        "yes"
                    } else {
                        "no"
                    },
                ),
                format!(
                    "Visible token IDs: {}",
                    self.history
                        .visible_token_ids()
                        .iter()
                        .map(i64::to_string)
                        .collect::<Vec<_>>()
                        .join(", "),
                ),
                String::new(),
                "Choice committed".to_owned(),
                "Press q to exit".to_owned(),
            ]
        } else {
            let mut rows = vec![
                "Rust Choice / fixture turn".to_owned(),
                format!("Context @ boundary {}", self.context_boundary),
                self.context_text.clone(),
                String::new(),
                "Rank  Token  Text".to_owned(),
            ];
            for candidate in &self.candidates {
                let marker = if candidate.token_id == self.proposal_token_id {
                    '>'
                } else {
                    ' '
                };
                rows.push(format!(
                    "{marker} {:>1}     {}   {}",
                    candidate.rank,
                    candidate.token_id,
                    candidate.text.trim_start(),
                ));
            }
            let proposal = self
                .candidates
                .iter()
                .find(|candidate| candidate.token_id == self.proposal_token_id);
            rows.push(String::new());
            if let Some(candidate) = proposal {
                rows.push(format!(
                    "Proposal: rank {} / token {} / \"{}\"",
                    candidate.rank, candidate.token_id, candidate.text
                ));
            }
            rows.push(String::new());
            rows.push(format!("{CHOICE_PREFIX}{}", self.input));
            rows.push("Enter submits · 1..N selects · q exits".to_owned());
            if let Some(status) = &self.status {
                rows.push(status.clone());
            }
            rows
        };

        let cursor = if self.accepted.is_some() {
            None
        } else {
            let row = 11.min(size.rows.saturating_sub(1));
            Some([
                (CHOICE_PREFIX.chars().count() + self.input.chars().count())
                    .min(size.columns.saturating_sub(1)),
                row,
            ])
        };
        normalize_lines(&mut lines, size);
        FrameContent {
            checkpoint: if self.accepted.is_some() {
                "choice.accepted".to_owned()
            } else {
                "choice.ready".to_owned()
            },
            decision_index: None,
            lines,
            cursor,
        }
    }

    fn render_real_model(&self, size: Size) -> FrameContent {
        let turn_label = format!("Rust Choice / real model / turn {}/2", self.turn_index + 1);
        let mut lines = if let Some(accepted) = &self.accepted {
            vec![
                turn_label,
                format!("Profile: {}", safe_terminal_text(&self.profile_name)),
                format!("Context @ boundary {}", accepted.boundary_after),
                format!("Context: {}", safe_terminal_text(&self.context_text)),
                format!("Action: {}", action_name(&accepted.action)),
                format!(
                    "Accepted token: {} / \"{}\"",
                    accepted.token_id,
                    safe_terminal_text(&accepted.evidence.text),
                ),
                format!(
                    "Evidence: boundary {} / sampling {} / agreement {}",
                    accepted.evidence.boundary,
                    accepted.evidence.sampling_boundary,
                    if accepted.evidence.proposal_agreement {
                        "yes"
                    } else {
                        "no"
                    },
                ),
                format!(
                    "Visible token IDs: {}",
                    self.history
                        .visible_token_ids()
                        .iter()
                        .map(i64::to_string)
                        .collect::<Vec<_>>()
                        .join(", "),
                ),
                String::new(),
                "Choice committed".to_owned(),
                "Press q to exit".to_owned(),
            ]
        } else {
            let mut rows = vec![
                turn_label,
                format!("Profile: {}", safe_terminal_text(&self.profile_name)),
                format!("Context @ boundary {}", self.context_boundary),
                format!("Context: {}", safe_terminal_text(&self.context_text)),
                String::new(),
                "Rank  Token  Text".to_owned(),
            ];
            for candidate in &self.candidates {
                let marker = if candidate.token_id == self.proposal_token_id {
                    '>'
                } else {
                    ' '
                };
                let suffix = if candidate.is_eog { " [EOG]" } else { "" };
                rows.push(format!(
                    "{marker} {:>5} {:>6}  \"{}\"{suffix}",
                    candidate.rank,
                    candidate.token_id,
                    safe_terminal_text(candidate.text.trim_start()),
                ));
            }
            if let Some(candidate) = self
                .candidates
                .iter()
                .find(|candidate| candidate.token_id == self.proposal_token_id)
            {
                rows.push(format!(
                    "Proposal: rank {} / token {} / \"{}\"",
                    candidate.rank,
                    candidate.token_id,
                    safe_terminal_text(&candidate.text),
                ));
            } else {
                rows.push("Proposal: unavailable".to_owned());
            }
            rows.push(String::new());
            rows.push(format!("{CHOICE_PREFIX}{}", self.input));
            rows.push("Enter accepts · q exits".to_owned());
            rows
        };
        let cursor = if self.accepted.is_some() {
            None
        } else {
            Some([
                (CHOICE_PREFIX.chars().count() + self.input.chars().count())
                    .min(size.columns.saturating_sub(1)),
                13.min(size.rows.saturating_sub(1)),
            ])
        };
        normalize_lines(&mut lines, size);
        FrameContent {
            checkpoint: if self.accepted.is_some() {
                "choice.real.accepted".to_owned()
            } else {
                "choice.real.ready".to_owned()
            },
            decision_index: Some(self.turn_index),
            lines,
            cursor,
        }
    }
}

impl AppState {
    fn render(&self, size: Size) -> FrameContent {
        match self {
            Self::Smoke(state) => render_smoke(state, size),
            Self::Choice(state) => state.render(size),
        }
    }

    fn handle_character(&mut self, byte: u8) -> bool {
        match self {
            Self::Smoke(state) => {
                if (0x20..=0x7e).contains(&byte) {
                    state.input.push(char::from(byte));
                }
                false
            }
            Self::Choice(state) => state.handle_character(byte),
        }
    }

    fn backspace(&mut self) {
        match self {
            Self::Smoke(state) => {
                state.input.pop();
            }
            Self::Choice(state) => state.backspace(),
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Rust terminal PTY process failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> io::Result<()> {
    let mode = match env::args().nth(1).as_deref() {
        None => "smoke",
        Some("--choice") => "fixture-choice",
        Some("--real-model") => "real-model-choice",
        Some(other) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported terminal mode {other:?}"),
            ));
        }
    };
    let choice_mode = mode != "smoke";
    let real_model_mode = mode == "real-model-choice";
    let frame_log_path = env::var_os("SPE_TERMINAL_FRAME_LOG")
        .map(PathBuf::from)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "SPE_TERMINAL_FRAME_LOG is required",
            )
        })?;
    let mut frame_log = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(frame_log_path)?;
    let mut semantic_log = if choice_mode {
        let path = env::var_os("SPE_TERMINAL_SEMANTIC_LOG")
            .map(PathBuf::from)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "SPE_TERMINAL_SEMANTIC_LOG is required for Choice mode",
                )
            })?;
        Some(
            OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(path)?,
        )
    } else {
        None
    };

    let mut worker = if real_model_mode {
        Some(BackendWorker::spawn_from_env()?)
    } else {
        None
    };
    let mut app = if mode == "fixture-choice" {
        AppState::Choice(Box::new(ChoiceState::load()?))
    } else if real_model_mode {
        AppState::Choice(Box::new(ChoiceState::load_real(
            worker
                .as_mut()
                .ok_or_else(|| invalid_data("real-model worker did not start"))?,
        )?))
    } else {
        AppState::Smoke(SmokeState {
            input: String::new(),
            last_submitted: "(none)".to_owned(),
        })
    };
    let (_terminal, startup_bytes) = TerminalGuard::enter()?;
    let mut size = terminal_size()?;
    let mut output_offset = startup_bytes;
    let mut sequence = 0_u64;
    present(
        &mut frame_log,
        &mut output_offset,
        &mut sequence,
        size,
        app.render(size),
    )?;

    loop {
        let mut descriptor = libc::pollfd {
            fd: libc::STDIN_FILENO,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: `descriptor` points to one initialized pollfd for the duration
        // of the call. A finite timeout lets the loop observe SIGWINCH promptly.
        let polled = unsafe { libc::poll(&mut descriptor, 1, 100) };
        if polled < 0 && io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
            return Err(io::Error::last_os_error());
        }

        if RESIZE_PENDING.swap(false, Ordering::Relaxed) {
            let current = terminal_size()?;
            if current != size {
                size = current;
                present(
                    &mut frame_log,
                    &mut output_offset,
                    &mut sequence,
                    size,
                    app.render(size),
                )?;
            }
        }

        if polled <= 0 || descriptor.revents & (libc::POLLIN | libc::POLLHUP) == 0 {
            continue;
        }

        let mut byte = [0_u8; 1];
        // SAFETY: `byte` is writable storage for one byte, and stdin is the
        // controlling PTY descriptor polled above.
        let read = unsafe { libc::read(libc::STDIN_FILENO, byte.as_mut_ptr().cast(), byte.len()) };
        if read < 0 {
            if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(io::Error::last_os_error());
        }
        if read == 0 {
            break;
        }

        match byte[0] {
            0x03 | 0x04 => break,
            b'\r' | b'\n' => {
                let mut advance_real_choice = false;
                let submit = match &mut app {
                    AppState::Smoke(state) => {
                        if matches!(state.input.as_str(), "quit" | "exit") {
                            SubmitResult::Exit
                        } else {
                            state.last_submitted = if state.input.is_empty() {
                                "(empty)".to_owned()
                            } else {
                                state.input.clone()
                            };
                            state.input.clear();
                            SubmitResult::Redraw(None)
                        }
                    }
                    AppState::Choice(state) => state.submit(worker.as_mut())?,
                };
                match submit {
                    SubmitResult::Exit => break,
                    SubmitResult::Redraw(semantic) => {
                        if let (Some(log), Some(record)) = (&mut semantic_log, semantic) {
                            serde_json::to_writer(&mut *log, &record).map_err(invalid_data)?;
                            log.write_all(b"\n")?;
                            log.flush()?;
                        }
                    }
                    SubmitResult::Advance(semantic) => {
                        if let (Some(log), Some(record)) = (&mut semantic_log, semantic) {
                            serde_json::to_writer(&mut *log, &record).map_err(invalid_data)?;
                            log.write_all(b"\n")?;
                            log.flush()?;
                        }
                        advance_real_choice = true;
                    }
                }
                present(
                    &mut frame_log,
                    &mut output_offset,
                    &mut sequence,
                    size,
                    app.render(size),
                )?;
                if advance_real_choice {
                    let AppState::Choice(state) = &mut app else {
                        return Err(invalid_data("real-model advance lost Choice state"));
                    };
                    state.advance_real(
                        worker
                            .as_mut()
                            .ok_or_else(|| invalid_data("real-model worker was lost"))?,
                    )?;
                    present(
                        &mut frame_log,
                        &mut output_offset,
                        &mut sequence,
                        size,
                        app.render(size),
                    )?;
                }
                continue;
            }
            0x08 | 0x7f => app.backspace(),
            byte @ 0x20..=0x7e => {
                if app.handle_character(byte) {
                    break;
                }
            }
            _ => continue,
        }
        present(
            &mut frame_log,
            &mut output_offset,
            &mut sequence,
            size,
            app.render(size),
        )?;
    }

    if let Some(worker) = worker.as_mut() {
        worker.shutdown()?;
    }

    Ok(())
}

fn terminal_size() -> io::Result<Size> {
    // SAFETY: zero is a valid initial value for winsize; ioctl fills the struct.
    let mut actual = unsafe { std::mem::zeroed::<libc::winsize>() };
    // SAFETY: `actual` points to writable winsize storage for this ioctl.
    if unsafe { libc::ioctl(libc::STDIN_FILENO, libc::TIOCGWINSZ, &mut actual) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(Size {
        columns: match usize::from(actual.ws_col) {
            0 => 80,
            columns => columns,
        },
        rows: match usize::from(actual.ws_row) {
            0 => 24,
            rows => rows,
        },
    })
}

fn present(
    frame_log: &mut File,
    output_offset: &mut usize,
    sequence: &mut u64,
    size: Size,
    mut content: FrameContent,
) -> io::Result<()> {
    *sequence += 1;
    normalize_lines(&mut content.lines, size);
    let mut transaction = Vec::new();
    transaction.extend_from_slice(FRAME_START);
    for (row, line) in content.lines.iter().enumerate() {
        transaction.extend_from_slice(format!("\x1b[{};1H", row + 1).as_bytes());
        transaction.extend_from_slice(line.as_bytes());
    }
    match content.cursor {
        Some([x, y]) => {
            transaction.extend_from_slice(format!("\x1b[{};{}H\x1b[?25h", y + 1, x + 1).as_bytes())
        }
        None => transaction.extend_from_slice(b"\x1b[?25l"),
    }
    transaction.extend_from_slice(FRAME_END);

    {
        let mut stdout = io::stdout().lock();
        stdout.write_all(&transaction)?;
        stdout.flush()?;
    }
    *output_offset += transaction.len();

    let record = FrameRecord {
        sequence: *sequence,
        checkpoint: content.checkpoint,
        decision_index: content.decision_index,
        size: [size.columns, size.rows],
        lines: content.lines,
        cursor: content.cursor,
        end_offset: *output_offset,
    };
    serde_json::to_writer(&mut *frame_log, &record).map_err(invalid_data)?;
    frame_log.write_all(b"\n")?;
    frame_log.flush()
}

fn render_smoke(state: &SmokeState, size: Size) -> FrameContent {
    let mut lines = vec![String::new(); size.rows];
    if size.rows > 0 {
        lines[0] = "Rust terminal PTY smoke".to_owned();
    }
    if size.rows > 1 {
        lines[1] = format!("Geometry: {}x{}", size.columns, size.rows);
    }
    if size.rows > 2 {
        lines[2] = "Type text and press Enter; type quit to exit.".to_owned();
    }
    if size.rows > 3 {
        lines[3] = format!("Submitted: {}", state.last_submitted);
    }
    let input_row = 4.min(size.rows.saturating_sub(1));
    lines[input_row] = format!("{INPUT_PREFIX}{}", state.input);
    for line in &mut lines {
        fit_line(line, size.columns);
    }
    let cursor = [
        (INPUT_PREFIX.len() + state.input.len()).min(size.columns.saturating_sub(1)),
        input_row,
    ];
    FrameContent {
        checkpoint: "smoke".to_owned(),
        decision_index: None,
        lines,
        cursor: Some(cursor),
    }
}

fn real_sampler(boundary: i64, stream_fingerprint: String) -> FixtureSampler {
    FixtureSampler {
        temperature: 0.8,
        top_k: Some(REAL_TOP_K),
        top_p: 1.0,
        min_p: 0.0,
        typical_p: 1.0,
        tail_free_z: 1.0,
        draw_kernel: "categorical".to_owned(),
        gaussian_noise_std: 1.0,
        perturb_noise_std: 1.0,
        student_t_df: 3.0,
        seed: REAL_SEED,
        stream_fingerprint,
        boundary,
        gumbel_noise_address: "token-id".to_owned(),
        gumbel_noise_scale: 1.0,
    }
}

fn real_candidates(
    worker: &mut BackendWorker,
    logits: &[f64],
    sampler: &FixtureSampler,
) -> io::Result<(Vec<ChoiceCandidate>, i64)> {
    if logits.len() != worker.vocabulary_size() || logits.iter().any(|value| !value.is_finite()) {
        return Err(invalid_data(
            "backend logits do not match the finite vocabulary vector",
        ));
    }
    let filtered = apply_filter(
        logits,
        sampler.temperature,
        sampler.top_k,
        sampler.top_p,
        sampler.min_p,
        sampler.typical_p,
        sampler.tail_free_z,
    )
    .map_err(invalid_data)?;
    let candidate_ids = filtered.stages[5]
        .as_ref()
        .ok_or_else(|| invalid_data("candidate filter did not return a final set"))?;
    if candidate_ids.len() != REAL_TOP_K as usize {
        return Err(invalid_data(
            "top-k filter returned an unexpected candidate count",
        ));
    }
    let candidate_scores = candidate_ids
        .iter()
        .map(|token_id| {
            let index = usize::try_from(*token_id)
                .map_err(|_| invalid_data("candidate token ID must be nonnegative"))?;
            filtered
                .scaled_logits
                .get(index)
                .copied()
                .ok_or_else(|| invalid_data("candidate token ID is outside logits"))
        })
        .collect::<io::Result<Vec<_>>>()?;
    let probabilities = softmax(&candidate_scores).map_err(invalid_data)?;
    let distribution = SparseDistribution {
        ids: candidate_ids.clone(),
        probabilities: probabilities.clone(),
        scores: Some(candidate_scores),
    };
    let boundary = sampler.boundary.to_string();
    let proposal = draw_token(
        &distribution,
        DrawOptions {
            seed: sampler.seed,
            fingerprint: &sampler.stream_fingerprint,
            boundary: &boundary,
            kernel: &sampler.draw_kernel,
            gaussian_noise_std: sampler.gaussian_noise_std,
            perturb_noise_std: sampler.perturb_noise_std,
            student_t_df: sampler.student_t_df,
            gumbel_noise_address: &sampler.gumbel_noise_address,
            candidate_model_ranks: None,
            gumbel_noise_scale: sampler.gumbel_noise_scale,
        },
    )
    .map_err(invalid_data)?;
    let eog_token_ids = worker.eog_token_ids()?;
    let candidates = candidate_ids
        .iter()
        .zip(probabilities)
        .map(|(token_id, probability)| {
            let is_eog = worker.is_eog(*token_id)?;
            let reported_eog = eog_token_ids.contains(token_id);
            if is_eog != reported_eog {
                return Err(invalid_data(
                    "backend EOG query disagrees with its EOG token set",
                ));
            }
            Ok(ChoiceCandidate {
                rank: rank(logits, *token_id).map_err(invalid_data)?,
                token_id: *token_id,
                text: worker.token_text(*token_id)?,
                probability,
                is_eog,
            })
        })
        .collect::<io::Result<Vec<_>>>()?;
    Ok((candidates, proposal))
}

fn safe_terminal_text(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_control() {
                format!("<U+{:04X}>", u32::from(character))
            } else {
                character.to_string()
            }
        })
        .collect()
}

fn context_history(context: &FixtureContext, context_text: &str) -> io::Result<EpisodeHistory> {
    if context.token_ids.is_empty() {
        return Ok(EpisodeHistory::default());
    }
    let action = PolicyAction::Write {
        text: context_text.to_owned(),
        mode: "exact".to_owned(),
    };
    let evidence: Vec<_> = context
        .token_ids
        .iter()
        .zip(&context.token_text)
        .enumerate()
        .map(|(offset, (token_id, text))| TokenEvidence {
            boundary: u64::try_from(offset).unwrap_or(u64::MAX),
            sampling_boundary: 0,
            token_id: *token_id,
            text: text.clone(),
            proposal_token_id: *token_id,
            raw_model_nll: None,
            raw_rank: None,
            policy_rank: None,
            decoder_probability: 1.0,
            proposal_agreement: true,
            is_eog: false,
            realized_visible: true,
        })
        .collect();
    let outcome = ActionOutcome {
        action: action.clone(),
        boundary_before: 0,
        boundary_after: context.boundary,
        resolved_text: context_text.to_owned(),
        resolved_token_ids: context.token_ids.clone(),
        visible_token_ids: context.token_ids.clone(),
        terminal_token_id: None,
        stop_reason: "completed".to_owned(),
        evidence,
        status: "completed".to_owned(),
        divergence: None,
        replay_eog_token_id: None,
        diagnostics: None,
    };
    EpisodeHistory::new(vec![RecordedAttempt {
        ordinal: 0,
        action,
        outcome,
        expectation: None,
    }])
    .map_err(invalid_data)
}

fn action_name(action: &PolicyAction) -> &'static str {
    match action {
        PolicyAction::Accept => "accept",
        PolicyAction::SelectRawRank { .. } => "select",
        _ => "action",
    }
}

fn normalize_lines(lines: &mut Vec<String>, size: Size) {
    lines.truncate(size.rows);
    while lines.len() < size.rows {
        lines.push(String::new());
    }
    for line in lines {
        fit_line(line, size.columns);
    }
}

fn fit_line(line: &mut String, width: usize) {
    let mut fitted = String::new();
    let mut used = 0_usize;
    for character in line.chars() {
        let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
        if used.saturating_add(character_width) > width {
            break;
        }
        fitted.push(character);
        used += character_width;
    }
    line.clear();
    line.push_str(&fitted);
    line.push_str(&" ".repeat(width.saturating_sub(used)));
}

fn invalid_data(error: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
}

fn default_one() -> f64 {
    1.0
}

fn default_student_t_df() -> f64 {
    3.0
}

fn default_token_id_address() -> String {
    "token-id".to_owned()
}

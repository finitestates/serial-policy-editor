# Serial Policy Editor — scope notes

This build deliberately reduces SPE to the parts that serve the editor itself.
It is not intended to preserve any historical obligations.

## Kept

- The interactive decision UI, including full-vocabulary `/TERM` search.
- Plain fallback and live terminal interface.
- llama.cpp and Hugging Face Transformers backends.
- The sampler pipeline: temperature, top-k eligibility, min-p logit gap, repetition, presence,
  and frequency penalties, deterministic seed/boundary behavior.
- Per-token editorial evidence including proposal token, proposal agreement,
  raw rank and policy rank; raw-model NLL and eligible softmax are optional diagnostics.
- A compact episode workspace used for persistence, resumption, evidence,
  and Serial Policy Replay.
- Serial Policy Replay over current-format stored episodes, with `handoff` and
  `ballistic` divergence modes.
- A projector that can be used to view saved episodes or export data in various formats.

## Lifecycle semantics

Episodes have no global token budget. `h N` delegates a finite span, while
`q` opens the live-edge menu immediately without generating tokens. Sampler
changes are ordinary ordered actions, so rewind and fork derive their settings
from the retained action prefix.

True termination is limited to:

1. explicit End at the live-edge menu;
2. teacher EOG (`e` with confirmation or `e!` without confirmation);
3. model EOG during autonomous `h` delegation.

SPR route exhaustion or normal divergence yields at a live edge. In ballistic
mode a divergence is recorded and replay continues where structurally possible.
A behavioral divergence is not treated as replay failure.

## Intentionally removed

- categorical CDF draws, top-p, typical-p and tail-free filtering;
- Gumbel beam search without replacement and stochastic beam controls;
- neighbor-margin/z-score overlays and column cycling;
- mandatory probability evidence and duplicate beam score/probability rows;
- legacy verifier and schema-compatibility machinery;
- execution replay and cold rebuild;
- legacy prefix-reconstruction planning;
- numerical-surface fingerprinting and surface-equality claims;
- backend conformance report subsystem;
- legacy report navigator/replay command;
- recovery/salvage machinery tied to old computational-state claims;
- old session/launcher/derived-prompt architecture;
- legacy Navigator and report forest.

Backend evaluation state, including any cache supplied by the backend library,
is an implementation detail rather than durable episode state. Backends use
incremental evaluation where supported and complete-prefix evaluation as the
fallback; replay remains the equivalence check for recorded behavior.

## Maintenance rule

New code should serve one of these purposes directly:

- run the live editor;
- provide model inference;
- implement sampling / token evidence;
- persist or resume a live episode;
- replay a policy episode;
- present/export an episode.

Historical auditing or implementation-path identity alone is not a reason to
reintroduce a subsystem.

Defaults are plain argmax, temperature 1, unrestricted eligibility and min-p 0.
Selective noise limits perturbation, not eligibility. Search retains full model
logits. Schema 3 requires a fresh workspace. Documentation describes this
experiment; inherited tests and the independent oracle still need migration.

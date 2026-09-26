# Serial Policy Editor — scope notes

This build deliberately reduces SPE to the parts that serve the editor itself.
It is not intended to preserve any historical obligations.

## Kept

- The interactive decision UI, including full-vocabulary `/TERM` search.
- Plain and prompt-toolkit terminal interfaces.
- llama.cpp and Hugging Face Transformers backends.
- The sampler pipeline: temperature, top-k/top-p/min-p, repetition, presence,
  and frequency penalties, deterministic seed/boundary behavior.
- Per-token editorial evidence including proposal token, proposal agreement,
  raw-model NLL, raw rank, policy rank, and decoder probability.
- A compact episode workspace used for persistence, resumption, evidence,
  and Serial Policy Replay.
- Serial Policy Replay over current-format stored episodes, with `handoff` and
  `ballistic` divergence modes.
- A projector that can be used to view saved episodes or export data in various formats.

## Lifecycle semantics

Token budgets are optional; the default is unlimited. An explicit budget is a checkpoint, not termination. Reaching it yields to the live
edge menu while keeping the same episode alive. The user may continue with a new
budget or sampler settings, fork, invoke SPR, project, quit while leaving the
episode resumable, or explicitly end/seal it.

`q` opens the live-edge menu immediately without generating tokens.

True termination is limited to:

1. explicit End at the live-edge menu;
2. teacher EOG (`e` with confirmation or `e!` without confirmation);
3. model EOG during autonomous `h` delegation.

SPR route exhaustion or normal divergence yields at a live edge. In ballistic
mode a divergence is recorded and replay continues where structurally possible.
A behavioral divergence is not treated as replay failure.

## Intentionally removed

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

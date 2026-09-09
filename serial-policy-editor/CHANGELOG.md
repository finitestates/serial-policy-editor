# Changelog

## 0.3.6.dev0 — development snapshot

- Use the full terminal for the live editor, historical review, and EDGE menu.
  Anchor input at the bottom and reserve room for candidate rows and feedback.
- Retain full context by default, with Page Up/Page Down scrolling and an optional
  `--context-chars` limit. Support multiline raw text, pasted whitespace, and
  explicit Ctrl+E expansion of the writing area.
- Reuse sampler statistics when history penalties are inactive, defer context
  decoding until display, cache context wrapping, and skip evidence loading for
  plain text projection.
- Keep experimental history replacement, fork-edit variations, token repair,
  and the proposed 100-token hold default on `codex/history-edit`. The main
  branch retains the 24-token hold default and existing replay behavior.
- This is an unreleased development snapshot; the 0.3.5 release is unchanged.

## 0.3.5

- Documentation-focused release with unchanged 0.3.4 runtime behavior, CLI
  options, dependencies, and workspace format.
- Replace the stale workspace README with a public project introduction and
  installation entry point.
- Add a first-session walkthrough, operation comparison, troubleshooting, and
  backup/export guidance. Reorganize replay and budget details outside tests
  and historical scope notes. Label prior smoke results as 0.3.4 evidence.
- Update package and CLI version metadata to 0.3.5.

## 0.3.4

- Add `--replay EPISODE --until Y`, EDGE `spr #N --until Y`, and `spr #N m`
  for choosing a cutoff from the source token map. Cutoffs can split writes and
  holds. EDGE prompt insertion retains ordinary per-token destination boundaries.

- Store budget edits and renewals as history. Rewind and forks restore the
  selected boundary's allowance and remainder; model changes preserve remaining
  tokens. Missing history falls back to unlimited, with a notice only for records
  listing a finite budget. Sampler recovery copies preserve budget history.

- EDGE SPR inserts the source prompt as an exact, rewindable write before its
  recorded actions. Prompt-only episodes now insert text; the destination
  tokenizer and remaining budget apply. CLI replay still uses the prompt only
  as initial context.

- EDGE SPR now appends to the current episode, preserving boundary 0, editable
  history, sampler stream, and remaining allowance. Replayed actions record
  source provenance and remain rewindable, including across the splice.
  Self-replay snapshots a finite procedure; CLI replay still creates an episode.

- CLI replay sampler flags now override only the specified fields while other
  fields follow recorded source transitions. Added `--fixed-config` to freeze
  the source-initial configuration plus explicit overrides during replay.
  User changes at EDGE remain authoritative after replay yields.

- Validate saved sampler configurations and stream coordinates without silent
  defaults or coercion. Recoverable records require a preview and confirmation;
  recovery creates a separate copy with provenance, leaving the source intact.
  Unsupported explicit RNG/policy schemes are rejected.
- Removed the unused llama.cpp sampler seed setting; SPE's sampler owns the seed.

- Made live rewind the default, including across checkpoints and inside writes;
  added `rewind N` at EDGE for both interfaces.
- Added stable workspace episode numbers, titles, `ls`, `ls all`, and direct
  `#N` switching at EDGE. Switching saves the current episode and pauses at the
  destination; finished episodes offer inspection and an explicit fork.
- Resume loads saved model/backend options, offers recovery for unavailable
  models, and confirms model replacements. Replacement resumes retokenize text
  into a linked episode, retaining the source evidence.
- In-session forks inherit current allowances rather than stale CLI budgets.

- Seamless rewind restores the sampler settings, seed, and stream coordinates
  at the destination boundary, so subsequent holds agree with a fork there.
  Both rewind entry points share restoration logic; token budgets are unchanged.

- Rewind now reuses the backend's existing fork-positioning path, falling back
  to one complete-prefix reset. Resume batches saved continuation tokens instead
  of evaluating them one at a time. Both show a context-restoration message.
- Removed obsolete code from a prehistoric version of the program that no longer
  does anything.
- Removed unused engine snapshots, counters, serializers, progress output, and
  historical choice-surface rendering. Moved sampling reference calculations into
  test support and removed the unused decoder diagnostics API.
- Replay now loads sampler segments once per procedure instead of querying for
  every action, while preserving boundary-specific sampler settings.
- Holds skip intermediate span decoding. Sentence and newline holds now stop
  immediately after the first token containing `.`, `!`, `?`, or a newline,
  respectively. Compound tokens stay whole; separate closing quotes and whitespace
  are no longer consumed. Boundary flags are cached from recorded token text,
  without extra decoding or sentence lookahead. Older sentence-hold replays may
  report divergence under the new stopping rule.
- Standardized on pytest as the complete-suite runner, added a test dependency
  extra, and converted the sampling tests to pytest style.

## 0.3.3

- Rewinding inside a conditional hold now removes its sentence/newline stop
  condition: the retained portion becomes a plain finite hold for replay and
  procedure projection. Fully retained holds keep their original condition.

- Removed the menu cap. Expansion is limited only by vocabulary size and counts
  main-menu rows independently of previously explored search neighborhoods.

- Live numeric entry previews undisplayed tokens. `ms N` and Ctrl+G on a numeric
  input open its neighborhood without committing; Enter remains selection.

- Added `--project EPISODE_ID --procedure`: a model-free source-boundary command
  listing sharing replay's derived moves, with result comments, initial metadata,
  and explicit sampler menu transitions.

- Continuation writes always use automatic spacing; exact writes never add it.
  Removed the CLI and runtime spacing toggle.

- Teacher proposal selections, blank Enter, and explicit acceptance now record
  concrete raw ranks. Proposal agreement remains evidence rather than changing
  the replay instruction; holds remain sampler-driven.

- Allow direct numeric selection across the full model vocabulary without
  requiring prior menu exposure or search. Out-of-range ranks remain invalid.

- Explicit replay plans distinguish empty replay from live editing and apply
  trailing source sampler settings only on normal completion. Early handoffs
  discard pending settings; fixed-sampler replay leaves live menu edits free.

- Replay holds now follow their derived limits and current stopping conditions;
  recorded expectations observe execution instead of controlling span length
  or stop reasons. Sentence lookahead is compared only if selected.

- Known target instruction rejections yield to the live edge and record a
  reason without sealing the episode; backend errors retain normal handling.

- Reject holds whose requested maximum exceeds the remaining allowance before
  generating any tokens; open the edge menu without consuming the allowance.

- Replay EOG now stops the tape and opens a live edge in both handoff and
  ballistic modes. The terminal token is not committed; preceding visible
  tokens remain, and expected/unexpected EOG is recorded separately.

- Token budgets are optional (unlimited by default). `q` opens the live edge
  immediately; `n off` removes an allowance. Early menu visits preserve the
  remaining allowance across sampler changes and session restoration.
- Replay respects explicit target budgets without enlarging them; reaching
  the budget hands control back to the live edge.

## 0.3.2

- Backend evaluation now uses incremental state by default where supported;
  `--cache off` / `--no-cache` selects complete-prefix evaluation. Forks can
  reposition the active backend without rebuilding an already shared prefix.

- Forks support current and backward/absolute boundary commands. Added
  `--list --lineage EPISODE_ID` and `--with-lineage` projector metadata for
  typed fork families and separately reported replays.

- The sampled proposal's raw-rank number is now prefilled at each teacher
  decision; Enter remains the explicit commit action. `--manual-acceptance`
  restores a blank command prompt.

- Added a live-edge `fm` visual fork map: `|N|` markers are exact absolute fork boundaries, including `|0|` for restarting from the original prompt/context.

- Reuse observation statistics across candidate display and action commits.
- Preserve recorded sampler transitions during source replay and inherit
  unspecified sampler fields when applying command-line overrides.

## 0.3.1

- Reduced the package to the live editor, sampler, llama.cpp/Transformers adapters, compact episode storage, minimal projector, and Serial Policy Replay.
- Token budgets are checkpoints rather than terminal events.
- Continuing after a checkpoint stays in the same episode.
- Restored distinct `e` confirmation and `e!` immediate EOG behavior.
- Added live-edge sampler changes and SPR invocation.
- SPR divergence is a counterfactual observation; route exhaustion yields to a live edge.
- Transformers evaluation state is private to the adapter and is not part of
  episode evidence or replay semantics.
- Removed legacy verifier, cold rebuild, execution replay, lineage/reconstruction/recovery machinery, Navigator, and legacy replay commands.
- Restored a compact `--full-evidence` Projector view for teacher-selected tokens using the evidence already stored in the episode ledger.
- Added opt-in `--with-model-probs` / `--with-model` Projector display for raw-model and decoder probabilities; rejected proposal text remains out of scope.

# Changelog

## 0.6.0 — 2026-09-21

- Add teacher replay plans and storage-independent replay recipes, including
  explicit source boundaries and destination placement semantics.
- Share live and durable execution paths while supporting ephemeral runs and
  representation-independent surviving-procedure export.
- Make episode history, controls, lineage, and EDGE command parsing explicit
  semantic seams, with persistence remaining an adapter concern.
- Refactor episode runtime setup, storage boundaries, and root-relative episode
  semantics while preserving live/durable behavior parity.

## 0.5.0 — 2026-09-18

- Establish `core/` as the standalone runtime package with replayable actions,
  results, `SamplerConfig`, persistence, menus, and backend contracts.
- Move research learners, instrumentation, compatibility surfaces, and retired
  integrations under `archive/` so core does not import research modules.
- Make `vector/` an optional singular `policy-editor-vector` distribution for
  conventional hidden-state steering vectors and cvector import/production.
- Keep arbitrary external vector loading in core without requiring vector
  production or analysis dependencies.
- Reduce the active suite to contract-focused core and optional-vector tests,
  while retaining historical research and integration tests in the archive.

## 0.4.6 — 2026-09-17

- Establish a repository-root pytest entry point restricted to the package test
  tree, and make the Transformers smoke harness patch the executable
  `ControllerPipeline` statistics seam shared by both backends.
- Persist cached SHA-256 model identity for new steering artifacts and runtime
  provenance. Version the artifact schema's compatibility declaration while
  continuing to read legacy metadata-only artifacts with an explicit weak
  compatibility policy and useful mismatch diagnostics.
- Harden the optional native worker boundary with timeout, output and prompt
  limits, response metadata validation, distinct startup/model/protocol errors,
  and build-time checks for its llama.cpp staging-header and library contract.
- Add bounded real llama.cpp release smoke coverage for CFG, hidden-state
  control, phrase recovery, cache navigation, and advanced sampling kernels;
  optional suites remain local-model-only and skip cleanly when unavailable.
- Add direct llama.cpp residual-stream capture for selected layers and token
  positions, including efficient range capture and prompt-pair hidden-state
  vector creation. The existing `cvector-generator` export/import workflow
  remains available for PCA and mean-based vector training.
- Complete the first portable hidden-state control path across llama.cpp and
  Transformers with explicit one-based decoder-block residual coordinates,
  model metadata, replay-safe artifacts, and real llama.cpp smoke coverage.
- Simplify teacher learning semantics: every explicit teacher-selected token
  is authoritative supervision, learnable groups use selection-gated positive
  updates, and each token in a typed write is learned as its own sequential
  event.
- Share compiled observation data and normalized group matchers across the
  sequential learner update path without changing the bounded update rules.

## 0.4.4 — 2026-09-16

- Replace the ambiguous public `activation` vector surface with explicit
  `output-head` and `hidden-state` workbench commands.
- Type portable steering artifacts as `output-head-steering-vector` or
  `hidden-state-vector` under `spe-steering-vector-v1`; ambiguous legacy
  activation-vector artifacts are rejected instead of being guessed.
- Rename launcher and setup-menu controls to `--steering-vector`,
  `--steering-strength`, and `steering PATH`, and expose `vector` as the
  reusable profile field.
- Clarify controller-stack and trace labels so output-head logit steering is
  not presented as an internal hidden-state intervention.

## 0.4.3 — 2026-09-16

- Add episode-pair activation tooling: derive portable output-layer vectors
  from positive/negative replay or fork episodes, or export escaped paired
  prompt files plus provenance for llama.cpp's `llama-cvector-generator`.
- Add an interactive pre-runtime setup menu for selecting episode sources,
  model/backend, steering artifacts, common sampler settings, budgets, seeds,
  and learner toggles before the existing launch path begins. `--setup-menu`
  opens it explicitly; headless and flag-driven workflows remain available.
- Begin the typed `RuntimePlan` boundary behind the setup menu, preserving
  explicit-versus-inherited launch settings while projecting the finalized plan
  into the existing replay-aware launcher.
- Let setup switch workspaces, list and inspect episodes by `#N`/`N`, and show
  read-only fork maps with `fm [#N]`; a bare `sampler` now explains all
  available sampler settings and their planned values.
- Make `learning`/`group` and `preference` learner panels discoverable from
  setup: bare commands show every associated control, `key=value` edits are
  accepted, and the main plan redraws after each change so enabled states are
  visible.
- Add the first descriptive controller stack: `controllers`/`stack` shows the
  ordered policy surfaces and feedback learners in setup and final preflight,
  without changing their established intervention mathematics.
- Correct the controller-stack order to distinguish backend model preparation
  from logit-space control, expose group control after token preference, and
  reconcile reused backend control-vector state before every new model surface.
- Add the first executable `ControllerPipeline` seam with opt-in immutable
  intermediate-surface traces; existing observation arithmetic remains the
  replay authority.
- Add strict reusable controller profiles in user-facing YAML. Setup now
  supports `profile print`, `profile save PATH`, and transactional
  `profile load PATH`; profiles use canonical JSON for SHA-256 identity and
  exclude episode/workspace/model launch context so replay selection remains
  untouched.

## 0.4.2 — 2026-09-16

- Rename the current projected token-feature learner and controls from
  “latent” to “token preference”; reserve “activation vector” for future
  residual-stream interventions. This is a breaking CLI, Python API, and
  persisted-preset terminology change. Projection seed flags now use their
  explicit `--token-preference-projection-seed` names, and presets use v4.

- Add `policy-editor-vector` for standalone token-preference vector artifacts:
  extract, inspect, validate, explain, blend, and apply model-matched vectors
  without confusing token-feature preferences with residual-stream activations.

- Add first-generation output activation vectors to `policy-editor-vector`:
  create a unit-normalized Prompt A minus Prompt B direction with llama.cpp or
  Transformers, inspect/validate/explain/blend the artifact, and load it into
  `policy-editor`. Persist activation state, model identity, artifact digest,
  and output-head actuation through sampler segments and replay.

- Import llama.cpp `llama-cvector-generator` GGUFs as layerwise activation
  artifacts, preserving their `direction.N` hidden-state directions and layer
  range. Install them through llama.cpp's native control-vector runtime,
  including prefix rebuilds when the active vector changes; retain the
  output-layer path for static token-level explanations.

- Replace opaque learning notices with compact explanations of gate skips,
  learning, decay, and unchanged memory. Add the read-only `learning` command at
  token choices and the live edge for the latest teaching event's slow/fast
  learning, decay, clipping, memory magnitudes, and per-token Write evidence.
  Keep detailed reports out of the one-line notice and label their historical scope.

- Add opt-in teacher-learning controls for both manual groups and token preference memory:
  `--{learning,token-preference}-decay-on {update,rejection,evidence}` preserves memory on
  agreement or gate skips; `--{learning,token-preference}-write-reduction {sum,mean,sqrt}`
  scales admitted typed evidence before clipping; and
  `--{learning,token-preference}-rejection-target {proposal,sampler}` can contrast corrections
  with the frozen sampler expectation. Preserve existing defaults and once-per-write
  decay. Include effective controls in records/readouts and paired trial commands.

- Add experimental `--learning-gate sampler` and `--token-preference-learning-gate sampler`
  to learn at full severity only from teacher tokens excluded by the actual
  sampler filters. Preserve rank-based defaults and independent decay. Include
  per-token eligibility in write evidence and explain gating in learning readouts.

- Add `b GROUP learn on` / `off` to opt manual groups into online teacher
  learning or freeze their current amounts without editing a preset. Resolve
  groups directly from loaded YAML/catalogs and show eligibility in `b` status.
  Preserve the learning choice across membership and numeric edits. Appearance
  objectives remain separate and must be cleared before enabling group learning.

## 0.4.0 — 2026-09-15

- Load standalone relative lexical weights with `--reference`, without catalogs,
  groups, or learning. Normal reference behavior is global and independent of
  group direction, with one optional overall strength.
- Load group YAML directly with `--groups`. Bare semantic `b target +`, `-`, and
  `=` now activate promote, suppress, and maintain appearance objectives.
  `b target off` disables the selected scope; explicit amounts remain manual.
- Control completed appearances during autonomous generation, using canonical
  entry/continuation routes and bounded history feedback. Scopes retain their
  own activation and baseline. `b` shows controller diagnostics.
- Separate canonical runtime routes from optional compiler exploration
  (`--explore`). Shared command resolution replaces the UI's duplicated bias
  assembly logic.
- Export/import full steering in `spe-bias-rules-v3`, including reference state,
  group objectives, and both token preference vectors and metadata. Read v2
  presets and minimal preference-array files. Preserve exact episode history
  origins for replay while rebasing portable presets to the destination.
- Fix stale learner observations after interactive bias edits and accidental
  modification of excluded/out-of-range manual weights. Sum typed evidence
  before clipping/decay in both teacher learners. Add severity, dead-zone,
  rejection, and decay controls to the manual group fitter and use analytical
  sparse gradients for its normal fixed-feature policy.
- Carry forward optional fast/slow preference memory, persisted projection seeds,
  decay, severity/dead-zone/rejection controls, and full-strength severity mode.
  Add policy diagnostics and chunked feature construction to reduce memory spikes.
- Apply explicit reference and preset imports across every replay segment while
  preserving other source transitions. Clear token-based steering on model change.
- Support references/presets and shared steering commands in the HTTP adapter.

See the archived [Steering guide](archive/research/docs/STEERING.md) for
historical interfaces, limitations, and migration details.

## 0.3.8 — 2026-09-14

This release carries the model-aware biasing work forward into a broader,
opt-in learning layer. The default editor behavior remains conservative: the
new reference prior, named-group learner, token preference learner, and typed-write
learning are all explicitly enabled features.

- Add a documented `spe-bias-catalog-v1` compiler format. YAML catalog inputs
  accept simple term lists, explicit `terms`/`groups` mappings, per-term
  options, generated case/spacing/plural/suffix forms, and model-specific
  deterministic token routes.
- Add route modes for `tail`, telescoping `path`, and `beheaded` lexical
  matching, with direct, word-aligned, cohesive, and fragmented route
  classification. Add cohesive route filtering and exact tail fallbacks when
  no safe decomposition remains.
- Add legacy, full, equal, information, amplified-information, and
  naive-chaining edge allocation strategies, including optional weighted
  reference YAML input and allocation diagnostics in compiled catalogs.
- Add one logical runtime matcher for direct rules, catalog terms, catalog
  groups, runtime groups, multi-token targets, shared prefixes, conditional
  triggers, and exact sentence/newline/stop-token lifetimes.
- Add durable named bias groups with append-only membership, shared amounts,
  sampler-state persistence, replay/fork/rewind behavior, and model-matched
  JSON preset export. Add `--rules-only` flattening and
  `--editor-friendly` YAML group export.
- Add a stateful weighted lexical reference prior backed by one
  history-reconstructed token trie. Support active/global scope, contrastive
  and ballistic modes, optional EXIT-vs-CONTINUE gates, and configurable
  attraction and strength.
- Add an opt-in online learner for named bias-group strengths. Updates are
  bounded, finite-difference based, restricted to live raw-rank selections by
  default, persisted as sampler segments, and recorded as diagnostic
  interactions.
- Add an independent opt-in token preference learner. It projects the model's
  output embedding into a deterministic fixed feature space and learns only a
  bounded anonymous preference vector, which is persisted and replayed without
  rerunning the learner.
- Add `--learn-from-write` so enabled learners may consume a live typed write.
  Each token is observed before commit, updates are averaged, and one update is
  installed after the write remains atomic.
- Improve the Transformers backend's final-position logits path and expose
  cached model features needed by token preference learning.
- Remove obsolete intermediate beheaded-route patch artifacts; their behavior
  is represented by the refined compiler and runtime implementation.
- Expand tests and user documentation for catalogs, references, route
  allocation, online learning, token preference learning, and typed-write learning.

## 0.3.7 — 2026-09-12

- Add model-specific YAML bias catalogs with deterministic term decomposition,
  bounded route selection, catalog groups, and sentence-case expansion for
  multi-word terms.
- Add per-term `auto`/`tail`/`path` mode selection with optional head and
  continuation scales for path rules.
- Add automatic `beheaded` path handling for bare boundary and short
  one- or two-letter route heads, while preserving explicit ordinary path mode.
- Add opt-in `cohesive` route selection for retaining useful whole-word-like
  decompositions while filtering tiny alternate subword fragments.
- Rank direct, word-aligned, and cohesive routes ahead of fragmented tokenizer
  routes instead of automatically reserving the tokenizer's default path.
- Add tail-only warning fallbacks when cohesive compilation finds no safe route.
- Replace the experimental bias representations with unified logical rules,
  including telescoping path semantics for lexical multi-token terms.
- Add durable runtime groups with append-only membership, shared bias amounts,
  and sampler-state rewind/replay/fork behavior.
- Add full named-group bias presets and `--rules-only` flattened export.
- Add `--editor-friendly` YAML export for recompilable named-group definitions.

## 0.3.6 — 2026-09-10

- Use the full terminal for the live editor, historical review, and EDGE menu.
  Anchor input at the bottom and reserve room for candidate rows and feedback.
- Retain full context by default, with Page Up/Page Down scrolling and an optional
  `--context-chars` limit. Support multiline raw text, pasted whitespace, and
  explicit Ctrl+E expansion of the writing area.
- Reuse sampler statistics when history penalties are inactive, defer context
  decoding until display, cache context wrapping, and skip evidence loading for
  plain text projection.
- Keep one terminal application alive across interactive commands, including
  editing, review, menus, confirmations, and episode navigation. Preserve raw
  mode between actions and restore the terminal when the session ends.
- Remove transient busy feedback to prevent editor layout shifts.
- Increase the default hold from 24 to 100 tokens; `--hold-default` remains
  available for customization.
- Exclude experimental history replacement and fork-edit variations.

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

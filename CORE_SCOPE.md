# Core runtime scope

This project is an interactive episode runtime: a menu-driven environment
for selecting tokens sequentially. This environment also has the capacity for rewinding, forking, speculative decoding, as well as replaying episodes.

The engine's semantic state includes the token ledger, selection configuration,
model identity, and replay stream identity. Selection starts with model/guidance
logits, applies policy adjustments and temperature, establishes eligibility,
adds optional noise, and chooses the highest score (ties use lowest token ID).
There is no token CDF. The default is plain argmax over the full vocabulary.

Noise coordinates retain the seed, root-prefix fingerprint, visible-token
boundary, and token-ID or model-rank address. The scheme literal remains
`blake2b64-token-prefix-quantile-v2`; changing it would change replay. Replaying
also requires matching model evaluation, logits and settings. A seed alone is
not sufficient. Editorial intervention, rewind and fork retain their existing
history and stream-identity semantics.

**The core test of program correctness is that replay must always terminate at a live edge:**
- *What this means*: a sequence of teacher actions represented as a replay tape can be executed automatically by the program itself without program failure, leaving the running program at an operational runtime menu (called `the EDGE menu`) within an active episode.
- *What this does not mean*: a given replay tape will emit the exact same sequence of tokens as the episode from which the tape was derived (although it often does mean that). Whether divergence from the original token sequence is desirable or undesirable depends on what a particular replay episode is attempting to demonstrate or accomplish.

Replay has two supported divergence modes:
- `handoff`: At the first sign that the replay is about to commit a token that diverges from what is observed in the source, the replay terminates and yields control.
- `ballistic`: The replay continues regardless of divergence and only yields when the tape is exhausted.

By design, a replay tape is not a 1-to-1 reconstruction of every action taken during an episode. The tapes themselves are storage-agnostic, transient runtime artifacts. In theory, they can be constructed from basically any data storage medium that exists.

## Main-program capabilities

New, replayed, and restored episodes use the same in-memory runtime. A normal
run does not create a database; an SQLite workspace is optional and is used
when the user explicitly opens one or saves a live branch. The runtime
currently supports:

- full-vocabulary search, token inspection, rank-addressed selection, rewind,
  fork, and fork maps;
- llama.cpp and Transformers inference backends, with backend dependencies
  installed as optional extras;
- logit-gap/top-k eligibility and deterministic or perturb-and-argmax selection, including
  classifier-free guidance and Gumbel-Max;
- grouped phrase biases and direct token biases, with bounded surface variants
  and exact-prefix matching for multi-token terms;
- repeat, presence, and frequency penalties over recent token history;
- checked and forced text actions, chord previews, and deterministic beam exploration;
- candidate-order and logit diagnostics, including the raw-rank-1 model-gap
  view;
- externally produced steering-vector artifacts and episode projection or
  portable teacher-tape export.

On a usable TTY, the live interface renders choice, EDGE, beam, and prompt requests in
separate screens through one app running on a dedicated UI thread. The
synchronous episode thread keeps engine and storage ownership. Piped and
noninteractive runs use the plain terminal fallback.

A database or other storage medium can be part of the workspace implementation, but it is not the definition
of an episode. The runtime owns episode/action meaning; persistence adapts that
meaning to the workspace.

### Candidate rank and menu order

The rank shown beside a candidate is its **model rank**: its one-based position
under the model's unadjusted logits for the current decision. It is the stable
numeric address used to select that candidate, including with `draw N`.

Menu views can change the order in which candidates appear. Policy sorting uses
the policy-adjusted scores, which can include vectors and other adjustments;
Gumbel sorting uses the Gumbel scores over the eligible set. These views
change row order but keep each candidate's model-rank address unchanged. The
policy and Gumbel positions are separate ranks, not replacements for the model
rank.

## Replay contract

An executable replay plan is an ordered sequence of teacher actions. Each step contains an action and, optionally, an expected result: visible token IDs, an optional terminal token ID, and a stop reason. The step number is its position in the plan; it is not a separate in-memory identity field.
The runner receives the divergence policy for the replay run (handoff or ballistic); the policy is not stored on each step. Ballistic replay needs no extra per-step field. When a step has no expected result, replay can still execute it, but cannot compare its outcome with the recorded one.

Replay executes supported teacher actions in order. At the first action it cannot interpret, it stops before applying that action, leaves later actions unexecuted, and yields to the EDGE menu with a warning that identifies the step and reason. It does not skip or reinterpret the action. A step with no recorded expectation may still execute, but its result cannot be checked against the source.

Source-derived plans follow sampler commands in the source action sequence by default. A caller can preserve the destination sampler or override the source command values; fixed or counterfactual plans can omit the source commands.

### Serial policy replay
`spr SOURCE [until]` appends the source episode's surviving teacher procedure to the currently selected episode, leaving the destination root intact. The source prompt, if present, is replayed as an exact write, followed by the selected source actions and their recorded expectations. until is measured in the source's root-relative visible-token boundaries; the resulting history is recorded at the destination's boundaries. SPR preserves the destination's sampler context instead of applying source sampler commands, while the selected divergence policy controls whether replay hands off on divergence or continues.

## Rewind contract

A rewind boundary is a count of visible tokens after the episode's initial
text. Boundary `0` is between the initial text and the first teacher-produced
token. The valid range is `0` through the current visible boundary; non-visible
evidence such as EOG does not advance it.

Rewind truncates the selected open episode or branch in place through that
boundary. It removes later history, repositions the runtime to the retained
prefix, clears terminal state, derives sampler settings from the retained
action sequence, and preserves the root stream identity. Commands at the
selected boundary are removed, so rewinding to boundary `N` returns to the
state before any command recorded at `N`. Rewind does not create a child
branch; fork first when both paths should be kept.

Completed or failed durable
episodes stay sealed and must be forked to continue. If the boundary cuts through an action, retain only its visible prefix. Represent a partial text or phrase write as an exact write of the retained
text; represent partial token generation as a finite hold for the retained
token count. The unretained remainder of the original action does not become part of a replay plan derived from an episode.

## Forking contract

An ordinary fork creates a new open episode from the source's history through a
root-relative visible-token boundary. The source remains unchanged. The new episode
inherits the initial text and retained action prefix. Its next local action
begins at the selected boundary, before commands recorded there.

Nested ordinary forks keep the same root-relative history frame. Parent identity and `fork_boundary` record provenance; they do not restrict where the
child may rewind or fork. A child may rewind before its original fork boundary. If the fork boundary cuts through an action, materialize the retained prefix
using the same partial-action rules as rewind. 

A model-change fork is a separate case: select the prefix at the source boundary, then materialize its text under the destination tokenizer. The
child's runtime boundaries follow that destination representation, while
`fork_boundary` remains provenance for the source boundary. Forking actions do not become part of a replay plan derived from an episode.

## Core-only acceptance gate

The core distribution must be independently buildable and runnable without
vector-production or research-extension files. A clean core-only environment must be
able to import the package, run `policy-editor -h`, create and replay a minimal
episode with a supported backend, and export that episode through `projector`.
Optional entry points may be unavailable without their extensions; their
absence must not prevent the core command or core package from starting.

## Argmax experiment compatibility

This branch is an unvalidated experiment, not the previously released 1.0.0
contract. `EligibleScores` replaces `SparseDistribution`; eligibility is set
membership, independent of optional softmax values. `Candidate.eligible_softmax`
replaces `decoder_probability`; neighbor-margin and vocabulary-z-score fields
are removed. Normal token evidence leaves eligible softmax unset. Explicit
projector probability requests reconstruct missing diagnostics through replay.

Workspace schema 3 stores nullable `eligible_softmax`. Earlier schemas are
rejected before schema creation; use a fresh workspace and leave old files
untouched. Removed categorical/filter/stochastic-beam settings are rejected.
See [core API notes](core/README.md) and the revised
[contracts](tests/CORE_CONTRACTS.md). Those contracts specify required behavior;
the inherited harnesses have not yet been migrated or run against this cut.

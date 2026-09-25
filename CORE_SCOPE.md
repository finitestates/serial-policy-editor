# Core runtime scope

This project is an interactive episode runtime: a menu-driven environment
for selecting tokens sequentially. This environment also has the capacity for rewinding, forking, speculative decoding, as well as replaying episodes.

The engine's complete semantic state is the token ledger plus the sampler configuration and coordinate. The kernel of the program deterministically samples the probability distribution of a language model by constructing sampler coordinates out of the SHA256 hash of prefix, the current offset, and a seed number.

The deterministic sampler state makes forking, chording, rewinding, and replaying comparatively easy to do. Assuming you know the step, seed number, and prefix, you can calculate the sampler coordinates at a given step exactly.

Even though the sampler state is deterministic, it doesn't feel that way unless you do a lot of episodes with the exact same prefix, model, and teacher decisions. The editor gives the user the freedom to intervene basically whenever, so no trajectory is "set in stone," unless the user wants it to be.

**The core test of program correctness is that replay must always terminate at a live edge:**
- *What this means*: a sequence of teacher actions represented as a replay tape can be executed automatically by the program itself without program failure, leaving the running program at an operational runtime menu (called `the EDGE menu`) within an active episode.
- *What this does not mean*: a given replay tape will emit the exact same sequence of tokens as the episode from which the tape was derived (although it often does mean that). Whether divergence from the original token sequence is desirable or undesirable depends on what a particular replay episode is attempting to demonstrate or accomplish.

Replay has two supported divergence modes:
- `handoff`: At the first sign that the replay is about to commit a token that diverges from what is observed in the source, the replay terminates and yields control.
- `ballistic`: The replay continues regardless of divergence and only yields when the tape is exhausted.

By design, a replay tape is not a 1-to-1 reconstruction of every action taken during an episode. The tapes themselves are storage-agnostic, transient runtime artifacts. In theory, they can be constructed from basically any data storage medium that exists.

## Main-program capabilities

The regular installation should support:

- episode workspaces, replay/resume/fork, rewind, and fork maps;
- full-vocabulary search and token inspection;
- llama.cpp and Transformers backends;
- sampler filtering and draw methods, including CFG and Gumbel draws;
- naive and conditional bias rules;
- history penalties;
- speculative decoding via check/force actions & chording;
- default menu view and the ability to cycle through displayed columns;
- raw/model-logit display, including the raw-rank-1 model-gap view;
- optional loading and application of externally produced vectors;
- episode export through `projector`.

A database or other storage medium can be part of the workspace implementation, but it is not the definition
of an episode. The runtime owns episode/action meaning; persistence adapts that
meaning to the workspace.

## Replay contract

An executable replay plan is an ordered sequence of teacher actions. Each step contains an action and, optionally, an expected result: visible token IDs, an optional terminal token ID, and a stop reason. The step number is its position in the plan; it is not a separate in-memory identity field.
The runner receives the divergence policy for the replay run (handoff or ballistic); the policy is not stored on each step. Ballistic replay needs no extra per-step field. When a step has no expected result, replay can still execute it, but cannot compare its outcome with the recorded one.

Replay executes supported teacher actions in order. At the first action it cannot interpret, it stops before applying that action, leaves later actions unexecuted, and yields to the EDGE menu with a warning that identifies the step and reason. It does not skip or reinterpret the action. A step with no recorded expectation may still execute, but its result cannot be checked against the source.

Source-derived plans follow the source sampler schedule by default. A caller can instead preserve the destination sampler or supply a different schedule; fixed or counterfactual plans can omit the source schedule.

### Serial policy replay
`spr SOURCE [until]` appends the source episode's surviving teacher procedure to the currently selected episode, leaving the destination root intact. The source prompt, if present, is replayed as an exact write, followed by the selected source actions and their recorded expectations. until is measured in the source's root-relative visible-token coordinates; the resulting history is recorded at the destination's coordinates. SPR preserves the destination's sampler context instead of applying the source sampler schedule, while the selected divergence policy controls whether replay hands off on divergence or continues.

## Rewind contract

A rewind boundary is a count of visible tokens after the episode's initial
text. Boundary `0` is between the initial text and the first teacher-produced
token. The valid range is `0` through the current visible boundary; non-visible
evidence such as EOG does not advance it.

Rewind truncates the selected open episode or branch in place through that
boundary. It removes later history, repositions the runtime to the retained
prefix, clears terminal state, and restores the sampler, stream coordinates,
and budget state for the selected boundary. Rewind does not create a child
branch; fork first when both paths should be kept.

Completed or failed durable
episodes stay sealed and must be forked to continue. If the boundary cuts through an action, retain only its visible prefix. Represent a partial text or phrase write as an exact write of the retained
text; represent partial token generation as a finite hold for the retained
token count. The unretained remainder of the original action does not become part of a replay plan derived from an episode.

## Forking contract

An ordinary fork creates a new open episode from the source's history through a
root-relative visible-token boundary. The source remains unchanged. The new episode
inherits the initial text, retained visible prefix, and the control state needed
to continue from that boundary. Its next local action begins there.

Nested ordinary forks keep the same root-relative history frame. Parent identity and `fork_boundary` record provenance; they do not restrict where the
child may rewind or fork. A child may rewind before its original fork boundary. If the fork boundary cuts through an action, materialize the retained prefix
using the same partial-action rules as rewind. 

A model-change fork is a separate case: select the prefix in the source's coordinates, then materialize its text under the destination tokenizer. The
child's runtime boundaries follow that destination representation, while
`fork_boundary` remains provenance for the source boundary. Forking actions do not become part of a replay plan derived from an episode.

## Core-only acceptance gate

The core distribution must be independently buildable and runnable without
vector-production or archived extension files. A clean core-only environment must be
able to import the package, run `policy-editor -h`, create and replay a minimal
episode with a supported backend, and export that episode through `projector`.
Optional entry points may be unavailable without their extensions; their
absence must not prevent the core command or core package from starting.

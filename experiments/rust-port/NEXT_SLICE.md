# Completed slice specification: two real-model Choice turns

**Status:** completed and validated on 2026-10-02. See the past-tense record
in [REAL_MODEL_CHOICE.md](REAL_MODEL_CHOICE.md). This file preserves the
original scope and acceptance gates for that completed slice.

## Objective

Connect the standalone Rust terminal process to one real SPE inference backend
for a short, opt-in Choice interaction. Rust should own candidate calculation,
the visible Choice state, accepted actions, and episode history. A persistent
Python worker should own the existing model adapter and its inference state.

This is a bounded process-boundary probe, not a commitment to the final
packaging or runtime architecture. It should answer whether the existing
backend contract and a long-lived worker can support a model-backed Rust turn
without routing dense vocabulary logits through JSON or mixing worker traffic
with terminal output.

The current fixture path remains useful and must keep working. Its limits and
evidence are recorded in the [completed fixture Choice slice](PRIOR_SLICE.md).

## Read these first

- [Rust experiment status and commands](README.md)
- [Rust migration roadmap](ROADMAP.md)
- [Rust terminal process and PTY contract](terminal-ui/README.md)
- [Python inference backend contract](../../core/src/trajectory_editor/core/backend.py)
- [Production token-prefix fingerprint](../../core/src/trajectory_editor/episode_hash.py)
- [Real-model harness and model-root rules](../../benchmarks/README.md)
- [Transformers CPU smoke profile](../../benchmarks/profiles/gpt2_cpu_smoke.yaml)
- [llama.cpp CPU smoke profile](../../benchmarks/profiles/llama_1b_smoke.yaml)
- [Decision-level fresh-prefix oracle](../../benchmarks/real_model_scenarios.py)
- [Completed deterministic fixture Choice slice](PRIOR_SLICE.md)
- [Completed Rust history-penalty slice](PREVIOUS_AGENT.md)

The backend contract is the boundary to preserve: vocabulary size, reset and
incremental evaluation, full-vocabulary logits, tokenization and rendering,
token text, EOG behavior, tokenizer identity, and provenance. Optional
batching, snapshots, cache-position operations, and speculative capabilities
are outside this slice.

## Scenario

Keep the existing fixture-driven process mode unchanged. Add an explicitly
selected real-model mode that performs two consecutive Choice turns:

1. Tokenize the fixed prompt "A short list of everyday objects:" with
   add_bos=True and special=True. Record the exact root token IDs and tokenizer
   identity. Compute the stream fingerprint with the production
   token_prefix_sha256 helper over those IDs; do not choose an arbitrary
   fingerprint or duplicate its encoding.
2. Prefill the backend and request its full-vocabulary logits.
3. Have the Rust sampler calculate the candidate view and proposal. Display
   the model-rendered context, candidate token IDs and text, proposal, and
   current boundary.
4. Accept the proposal through the same submit path used by the fixture mode.
   Record the action, selected token, evidence, and updated history.
5. Advance the same backend incrementally with the accepted token. Request the
   next logits and show a second Choice at the next boundary.
6. Accept the second proposal and exit cleanly.

Use a fixed, documented sampler configuration and seed for both model
profiles. It may be supplied by the runner or the profiles, but record the
effective values. Preserve candidate ordering, token IDs, sampler coordinates,
and the existing Rust/Python sampler contract.

The two backends intentionally exercise different adapters: the small
Transformers GPT-2 directory and the llama.cpp Q4_K_M GGUF in the linked smoke
profiles. Run them as separate invocations so model resources are released
between runs.

## Process boundary

Use one persistent Python worker for the full interaction. It should load a
backend through the existing SPE backend factory and use the selected smoke
profile's launch settings. Reuse the profile parsing and provenance behavior
from the [real-model harness](../../benchmarks/real_model.py) where practical;
do not create a second model-loading implementation.

Keep the worker on dedicated child-process pipes. Its stdout must never share
the PTY, and diagnostics belong on stderr or in an artifact file. Define a
small, versioned request/response protocol for the operations this scenario
needs: startup and provenance, tokenize, reset, incremental eval, logits,
render or token text, EOG queries, and orderly shutdown. Errors must return a
clear failure to the Rust process; a missing model or dependency must not turn
into a skipped check.

Transfer dense logits as a framed binary numeric payload with an explicit
dtype and element count. Do not encode a vocabulary-sized vector as a JSON
number list. Validate the payload length, vocabulary size, one-dimensional
shape, and finiteness at the receiving boundary. Document any conversion
between the backend's returned dtype and the Rust sampler's numeric type.

Keep the checked-in fixture mode as the default and route it and the worker
through the same Rust-side decision interface. Add model-free protocol
coverage so ordinary tests do not need model weights or optional inference
packages.

## Correctness and evidence

Compare the Rust decisions with the production Python sampler using the same
backend logits, exact root and visible token IDs, sampler settings, seed,
stream fingerprint, and boundary. The Python oracle must independently
calculate its candidate order and proposal, and derive expected display text
from the backend tokenizer/renderer; never derive expected values from the
Rust frame log.

After the two live turns finish, replay each captured decision from its exact
full prefix and compare the fresh Python decision to the Rust-selected token.
Follow the decision-level approach in
[real_model_scenarios.py](../../benchmarks/real_model_scenarios.py): sampled
token disagreement fails; logit deltas and top-token changes are diagnostic
data. The oracle runs after the live interaction so it cannot change the
worker's incremental path before the second Choice.

Record a versioned sidecar with at least:

- profile name, backend, selected model identity and checksum when available;
- tokenizer identity, vocabulary size, backend provenance, and launch settings;
- exact root, visible-prefix, and captured candidate token IDs;
- sampler configuration, seed, fingerprint, and both sampling boundaries;
- backend operation timings, transferred logit byte counts, and total bytes;
- protocol version, semantic outcomes, fresh-prefix comparison, and failures.

Retain raw PTY bytes, ordered frame offsets, and the existing semantic sidecar.
Author the screen layout and cursor checkpoints before changing renderer code.
For model-dependent fields, derive expected cells from the Python oracle and
backend transcript, not from Rust's frame report. Replay every frame with
pyte, check complete rows and cursor state, exercise resize, and keep a
negative control that makes the screen comparison fail. Ensure model token
text cannot inject terminal control sequences into the display.

Timing is diagnostic for this boundary probe. Report backend service time and
serialization/transfer time separately; do not claim a speedup from these
measurements.

## Scope boundaries

- Keep changes under experiments/rust-port, its focused tests, and the
  existing PTY test support needed to launch the real-model process.
- Leave core runtime behavior, production CLI, package dependencies, and
  release configuration unchanged.
- Do not add persistence, branches, rewind, batching, snapshots, speculative
  decoding, CFG, direct or grouped biases, or the full production Choice view.
- Keep the normal test suite model-free. Real-model runs are explicit and
  fail when the selected profile, model, or optional dependency is missing.
- Do not make a packaging recommendation from this probe alone. Record the
  boundary evidence and remaining tradeoffs for a later roadmap decision.

This probes the backend edge ahead of the full runtime migration in
[roadmap stage 3](ROADMAP.md#3-move-episode-execution-and-replay). It does not
complete stage 2: the remaining policy calculations still need their own
bounded slices.

## Required validation

Keep the existing checks and add focused model-free tests for worker protocol
framing, error handling, logits validation, and the fixture-mode regression.
Run the compiled-Rust PTY selection and the full PTY module. Then run the
real-model PTY path once for each smoke profile; do not silently skip either
profile.

Install the optional backend extras in the same interpreter used by the
runner. From the repository root, the profile-specific installs are:

    core/.venv/bin/python -m pip install -e './core[transformers-accelerate]'
    core/.venv/bin/python -m pip install -e './core[llama]'

The real-model runner should expose and document an invocation equivalent to
the following from the repository root. /path/to/models is a placeholder for
the operator's model directory; it is not a default or a machine-specific
path.

    cargo build --manifest-path experiments/rust-port/terminal-ui/Cargo.toml --locked
    core/.venv/bin/python experiments/rust-port/terminal-ui/scripts/real_model_choice.py --model-root /path/to/models --profile benchmarks/profiles/gpt2_cpu_smoke.yaml
    core/.venv/bin/python experiments/rust-port/terminal-ui/scripts/real_model_choice.py --model-root /path/to/models --profile benchmarks/profiles/llama_1b_smoke.yaml

If the runner uses a different script path or flags, update this brief and
[terminal-ui/README.md](terminal-ui/README.md) together. For profile behavior
and model-root resolution, follow [benchmarks/README.md](../../benchmarks/README.md).
Also run the shared Rust workspace formatting, locked tests, and Clippy
commands from [rust-port/README.md](README.md).

The implementation, artifacts, and exact command outcomes are recorded in
[REAL_MODEL_CHOICE.md](REAL_MODEL_CHOICE.md). Do not treat this preserved
specification as a claim about checks beyond that record.

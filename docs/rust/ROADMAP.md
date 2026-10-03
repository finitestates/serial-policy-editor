# Rust migration roadmap

## Target

Rust should eventually own the episode runtime and terminal UI: policy state,
action execution, replay, branch handling, rendering, and terminal I/O. Model
inference remains a replaceable backend. Python may continue to host
Transformers/PyTorch and the existing llama.cpp binding; the process boundary
(PyO3, an embedded Python host, or a Python backend worker) is a later design
decision.

The Rust UI must be tested as a real process under a PTY. The test runner and
screen oracle can remain Python: a PTY carries bytes and resize signals, not
language-specific objects.

## Sequence

### 0. Prove the Rust terminal test seam — implemented

Build a tiny Rust terminal process, launch it from the existing Python PTY
harness, and verify that the existing `pyte` replay can reconstruct each
intended frame. See [`TERMINAL_UI.md`](TERMINAL_UI.md) for scope
and the frame-log contract.

**Gate:** preserve the raw PTY capture; replay every logged frame boundary;
check complete cell rows and cursor state; test at least one key interaction
and resize in both directions; retain a negative control that makes the oracle
fail. This establishes the cross-language testing path before full UI work.

The smoke binary is in `terminal-ui/src/main.rs`; the PTY test builds it once
per pytest session and runs it through the existing `Session` launcher. On
2026-10-02, the full `tests/core/test_live_terminal_pty.py` module passed
(13 tests), and the Rust crate passed formatting and Clippy with warnings
denied. This establishes the test seam only; the production terminal UI is
still Python.

### 1. Integrate the proven kernels — implemented experiment

The first integration is one deterministic candidate-choice turn in the
compiled Rust terminal process. It uses fixture logits, the existing Rust
sampler and episode-history kernels, and the current PTY/pyte harness. The
fixture, semantic record, screen contract, and validation evidence are in
[`PRIOR_SLICE.md`](PRIOR_SLICE.md).

Keep Python facades around the kernels for parity checks. Avoid converting
full-vocabulary NumPy arrays to Python lists at every decision; keep this slice
fixture-driven and leave the production inference boundary undecided. The
sampler's Python binding is optional so its arithmetic can be called as a
normal Rust library.

**Gate passed for this slice:** the shared Python/Rust fixture agrees on exact
candidate order, proposal token, action, token IDs, and root-relative
boundaries. Probability comparisons use a relative tolerance of `1e-12` and an
absolute tolerance of `1e-14`. The PTY test replays every frame from raw bytes
and compares the authored `choice.ready`, `choice.accepted`, and both
`choice.resized` geometries. It also retains a negative control that removes
the proposal row.

### Real-model backend boundary probe — revalidated 2026-10-03

Both local smoke profiles passed with generated-token boundaries 0 and 1.
The fresh backend rebuilds the exact full prefix by prefilling the prompt and
then evaluating generated tokens one at a time, matching the live operation
boundaries. The reports and retained PTY/model evidence are in
[`REAL_MODEL_CHOICE.md`](REAL_MODEL_CHOICE.md); the original probe scope is
archived in [`REAL_MODEL_CHOICE_BRIEF.md`](REAL_MODEL_CHOICE_BRIEF.md).

### 2. Move policy calculations — implemented experiment

The isolated sampler crate implements the numeric policy pipeline around
backend-produced logits: history penalties, output-head activation adjustments,
direct and grouped biases, ephemeral biases, candidate-filter stages, sparse
draw probabilities, CFG logit blending, ranks, and lazy evidence metrics. Python
continues to produce model-dependent output-head adjustments and owns model
inference, CFG branch positioning, and backend caches.

**Gate passed for the numeric policy surface:** shared fixtures are generated
from `PolicyCalculations` and compared against plain Rust and PyO3 adapter
paths. Integer IDs, ranks, and filter stages compare exactly; finite numeric
values use absolute tolerance `1e-14`. The sampler, CFG, and
vector/backend core contract tests pass. Real-model checks pass for
GPT-2/Transformers and Llama 3.2 1B/llama.cpp, including numeric CFG with two
real backend branches and output-head projections. Detailed evidence is in
[`POLICY_CALCULATIONS_SLICE.md`](POLICY_CALCULATIONS_SLICE.md).

`SamplerConfig` validation/serialization and action/replay serialization remain
Python-owned compatibility boundaries. The Stage 2 Rust surface consumes
already-resolved configuration values; it does not introduce Rust config or
action records. Backend CFG lifecycle remains part of the later episode-runtime
port. Python production behavior, dependencies, CLI, and release packaging are
unchanged.

### 3. Move episode execution and replay

Port action resolution, observation construction, outcome/evidence creation,
replay-plan execution, divergence policies, and root-relative rewind/fork
operations. Preserve the backend as a capability boundary: basic tokenization,
evaluation, logits, rendering, and EOG behavior, with optional interfaces for
batching, snapshots, position-aware caches, and speculative operations.

**Gate:** cover engine, replay, lifecycle, and property slots. Run equivalent
journeys through both existing model adapters; compare token IDs, sampler
coordinates, action outcomes, divergence status, and the resulting live edge.

### 4. Move branch sessions, persistence, and projection

After the in-memory engine is stable, port live-session/roster state, lineage,
SQLite access, materialization, and projector/export logic. Keep persistence as
an adapter for the episode model. Preserve the existing workspace schema and
teacher-plan/JSONL formats during this migration.

**Gate:** save, close, resume, rewind, fork, branch switch, family save, and
export match the existing lifecycle and persistence contracts. Keep prior
workspace rejection behavior unchanged unless a separately reviewed schema
migration is planned.

### 5. Port the terminal UI to Rust

Once the Rust runtime exposes stable request/view and response/action values,
port the input parser, cell canvas, views, frame driver, and terminal lifecycle.
Start with the PTY smoke process from stage 0, then add views and user journeys
incrementally. Keep a Python launcher or backend host as needed; Rust owns the
terminal event loop and rendering path.

**Gate:** run production journeys through the compiled Rust process under the
same PTY oracle. Assert every intermediate screen against the intended grid,
test resize and input transitions, retain negative controls, and inspect raw
captures for synchronized frame transactions and unexpected erases. Add
input-to-frame timing measurements for performance comparisons. PTY evidence
establishes emitted screen behavior under those conditions; it does not make
a monitor-scanout claim.

### 6. Package the chosen process shape

Choose whether the supported command is a Rust executable with an optional
Python inference worker, a Python executable with a Rust UI/core extension, or
both. Keep core-only installation model-free and preserve the supported Python
API/CLI until a major-version change is intended. Put production Rust sources
where source distributions and the documented `core` sparse checkout can
build them, and publish/test wheels for supported Python versions if PyO3 is
part of the product.

**Gate:** clean checkout build, core-only startup/help, no-backend startup,
optional backend installation, PTY integration, and the existing release test
matrix. Report measured startup and input-to-frame latency separately from
model inference latency.

## Cross-cutting rules

- Keep the Python implementation as the behavior oracle until each Rust slice
  passes its contract gate; do not switch the runtime wholesale.
- Preserve the draw coordinates (seed, root stream fingerprint, and boundary),
  candidate ordering, token-ID semantics, and action/tape formats.
- Run real backend comparisons through both Transformers and llama.cpp when a
  slice affects inference or sampler decisions.
- Keep the raw PTY stream and ordered frame records as complementary evidence.
  `pyte` is the independent emulator/oracle; the Rust app only reports its
  intended frame and the byte offset at which it was written.
- Treat the vector-production package as a separate Python/Transformers
  component unless a later scope decision includes it.
- Do not promote a slice or claim it is faster based on architecture alone.

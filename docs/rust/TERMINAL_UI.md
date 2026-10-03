# Rust terminal PTY experiment

**Status:** the standalone process has a passing PTY smoke path and
fixture-backed Choice turn. The real-model worker path passed both corrected
smoke profiles on 2026-10-03 at generated-token boundaries 0 and 1. The
sampler's Stage 2 policy work is complete as an isolated experiment; its
evidence is in [`POLICY_CALCULATIONS_SLICE.md`](POLICY_CALCULATIONS_SLICE.md).

The historical real-model results and artifact locations are in
[`REAL_MODEL_CHOICE.md`](REAL_MODEL_CHOICE.md). The original probe scope
is archived in [`REAL_MODEL_CHOICE_BRIEF.md`](REAL_MODEL_CHOICE_BRIEF.md).

## Process modes

The default mode keeps the original editable-input smoke journey: type a value,
submit it, resize, then submit `quit`. The `--choice` mode uses the Rust sampler
and episode-history libraries through the same process loop. It loads
  [`terminal-ui/fixtures/choice-turn.json`](../../terminal-ui/fixtures/choice-turn.json), calculates the
filtered candidate view and proposal, displays the proposal and raw ranks, and
accepts the existing `accept` command on Enter. Numeric commands select a
displayed raw rank.

The fixture's expected semantic result is data in the fixture. The complete
authored cell grids and cursor positions are kept separately in
  [`terminal-ui/fixtures/choice-screen-contract.json`](../../terminal-ui/fixtures/choice-screen-contract.json);
the test does not derive expected grids from the process frame log. The
initial context is represented as an exact-write seed attempt in the typed
history, so the accepted proposal extends boundary 3 to boundary 4.

After acceptance, Choice mode remains active for capture and resize. Press
`q`, Ctrl-C, or Ctrl-D to exit cleanly.

## Capture and semantic sidecars

The process writes intended terminal frames and cumulative PTY byte offsets to
the JSONL path in `SPE_TERMINAL_FRAME_LOG`. In Choice mode it also requires
`SPE_TERMINAL_SEMANTIC_LOG` and writes the canonical action, selected token,
evidence, before/after boundary, and typed history there. The semantic file is
separate from stdout/stderr and the frame log.

The Python test uses the same `Session` PTY driver as the production journeys.
It preserves raw PTY bytes, replays each ordered frame through `pyte`, checks
complete rows and cursor state, checks the authored checkpoints, and uses a
deliberately removed proposal row as a negative control.

## Run

From the repository root:

```sh
cargo build --manifest-path terminal-ui/Cargo.toml --locked
core/.venv/bin/python -m pytest -q tests/core/test_live_terminal_pty.py -k compiled_rust
core/.venv/bin/python -m pytest -q tests/core/test_live_terminal_pty.py
```

For workspace-wide Rust formatting, tests, and Clippy, use the shared commands
in [`../../RUST.md`](../../RUST.md).

The test contract checks terminal cells and cursor state through `pyte`; it
does not compare styles.

## Real-model mode

`--real-model` starts one persistent Python worker using the selected profile
through `benchmarks.real_model.load_profile` and the production backend
factory. Worker stdout is a dedicated framed RPC pipe; diagnostics go to the
artifact stderr log. JSON request and response headers use a four-byte
little-endian length. Full-vocabulary logits follow as an explicitly sized
`f64le` binary payload. The worker records the backend's source dtype and
converts logits to little-endian float64 for the Rust sampler.

The fixed sampler for both smoke profiles uses temperature `0.8`, `top_k: 5`,
`top_p: 1.0`, `min_p: 0.0`, `typical_p: 1.0`, `tail_free_z: 1.0`, categorical
drawing, token-ID addressing, and seed `17`. The profile still supplies model
loading and backend launch settings. The runner accepts each proposal through
the same Choice submit path as fixture mode, advances the live backend
incrementally, and captures decisions at visible-history boundaries `0` and
`1` before doing fresh-prefix replays. The fixed prompt is pre-0 context: its
token-prefix SHA-256 supplies the sampler's 256-bit stream identity, while the
prompt tokens remain in the model prefix and outside generated-token history.

From the repository root, build the standalone process, install the optional
extras in the same interpreter used by the runner, and run each profile in a
separate invocation:

```sh
cargo build --manifest-path terminal-ui/Cargo.toml --locked
core/.venv/bin/python -m pip install -e './core[transformers-accelerate]'
core/.venv/bin/python -m pip install -e './core[llama]'
core/.venv/bin/python terminal-ui/scripts/real_model_choice.py --model-root /path/to/models --profile benchmarks/profiles/gpt2_cpu_smoke.yaml
core/.venv/bin/python terminal-ui/scripts/real_model_choice.py --model-root /path/to/models --profile benchmarks/profiles/llama_1b_smoke.yaml
```

The runner fails when a model, optional dependency, worker operation, decision,
or screen comparison fails. It writes a versioned `report.json`, per-turn
semantic JSONL, worker metadata and timings, raw PTY bytes, frame offsets, and
the two transferred logit arrays under `/tmp/spe-rust-real-model-choice` by
default. The report compares Rust with the Python sampler using the exact
captured live logits. A fresh backend rebuilds each exact full prefix by
prefilling the root prompt and evaluating generated tokens one at a time,
matching the live operation boundaries. This matters for quantized llama.cpp
CPU inference: folding a generated token into one larger reset batch changed
its logits and could change the sampled token. The passing replays match both
decisions; logit deltas and top-token changes remain diagnostic values. The
screen oracle builds expected cells from the Python decision and backend
transcript, replays each frame with `pyte`, checks resize in both directions,
and retains a negative control that removes the proposal row.

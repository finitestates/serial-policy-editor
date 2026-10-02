# Rust port experiments

This directory holds isolated Rust experiments for the Serial Policy Editor.
The released `core` package remains Python; nothing in this workspace is a
runtime or build dependency of the Python distribution. Each experiment
documents its behavioral boundary, Python adapter (if any), parity evidence,
and build commands.

The crates share this experiment-only Cargo workspace so cross-crate work has
one lockfile and one set of Rust commands. From the repository root:

```sh
cargo fmt --manifest-path experiments/rust-port/Cargo.toml --all -- --check
cargo test --manifest-path experiments/rust-port/Cargo.toml --workspace --locked
cargo clippy --manifest-path experiments/rust-port/Cargo.toml \
  --workspace --all-targets --all-features --locked -- -D warnings
```

## Current layout

| Path | Status | Scope |
| --- | --- | --- |
| [`sampler/`](sampler/README.md) | Implemented experiment | Numeric filtering, deterministic draws, and sampler helpers from `core/src/trajectory_editor/core/sampling.py`, plus the history-penalty transform from `core/src/trajectory_editor/core/policy_calculations.py`. |
| [`episode-history/`](episode-history/README.md) | Implemented experiment | Typed actions, outcomes, evidence, in-memory history validation/truncation, and sampler updates. |
| [`terminal-ui/`](terminal-ui/README.md) | PTY smoke and fixture Choice turn implemented | Exercise the sampler and history crates through one compiled Rust terminal process and the existing PTY/pyte oracle. No production Rust UI yet. |

The completed integration slice is a deterministic candidate-choice turn
joining these pieces. Its fixture, semantic result, visual contract, validation
commands, and evidence are in [`NEXT_SLICE.md`](NEXT_SLICE.md). The Python
bindings remain available for parity checks; the terminal binary calls the
sampler and episode-history kernels as Rust libraries. The sampler's completed
history-penalty kernel and its evidence are recorded in
[`NEXT_AGENT.md`](NEXT_AGENT.md).

## Direction

The intended destination is a Rust-owned episode runtime and terminal UI.
Model-specific inference can remain behind Python adapters for Transformers
and existing llama.cpp bindings while that provides value. The PTY smoke and
fixture-backed Choice turn are implemented. Current kernel work continues
roadmap stage 2 by moving additional policy calculations into Rust.

See [`ROADMAP.md`](ROADMAP.md) for migration order, boundaries, and gates. The
PTY smoke contract is in [`terminal-ui/README.md`](terminal-ui/README.md).

Experiment-wide promotion rules are in [`../README.md`](../README.md). Rust
migration order and its behavioral gates are in [`ROADMAP.md`](ROADMAP.md).

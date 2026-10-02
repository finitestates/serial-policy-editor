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
| [`sampler/`](sampler/README.md) | Implemented experiment | Numeric filtering, deterministic draws, and sampler helpers from `core/src/trajectory_editor/core/sampling.py`. |
| [`episode-history/`](episode-history/README.md) | Implemented experiment | Typed actions, in-memory history validation/projection/truncation, and sampler changes. |
| [`terminal-ui/`](terminal-ui/README.md) | PTY smoke and fixture Choice turn implemented | Exercise the sampler and history crates through one compiled Rust terminal process and the existing PTY/pyte oracle. No production Rust UI yet. |

The completed integration slice is a deterministic candidate-choice turn
joining these pieces. Its fixture, semantic result, visual contract, validation
commands, and evidence are in [`NEXT_SLICE.md`](NEXT_SLICE.md). The Python
bindings remain available for parity checks; the terminal binary calls the
sampler and episode-history kernels as Rust libraries.

## Direction

The intended destination is a Rust-owned episode runtime and terminal UI.
Model-specific inference can remain behind Python adapters for Transformers
and existing llama.cpp bindings while that provides value. The PTY smoke and
the bounded Choice interaction are implemented; the next migration stage is
moving additional policy calculations into Rust.

See [`ROADMAP.md`](ROADMAP.md) for migration order, boundaries, and gates. The
PTY smoke contract is in [`terminal-ui/README.md`](terminal-ui/README.md).

## Promotion rule

These experiments are not dependencies of the released Python packages.
Promote functionality only after behavioral parity, real-process integration,
packaging, and maintenance costs are reviewed. Preserve the existing Python
API and CLI during the 1.x line. Do not claim a speedup without repeatable
measurements against the same inputs and machine.

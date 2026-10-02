# Rust port experiments

This directory holds isolated Rust experiments for the Serial Policy Editor.
The released `core` package remains Python. Each experiment documents its
behavioral boundary, Python adapter (if any), parity evidence, and build
commands.

## Current layout

| Path | Status | Scope |
| --- | --- | --- |
| [`sampler/`](sampler/README.md) | Implemented experiment | Numeric filtering, deterministic draws, and sampler helpers from `core/src/trajectory_editor/core/sampling.py`. |
| [`episode-history/`](episode-history/README.md) | Implemented experiment | Typed actions, in-memory history validation/projection/truncation, and sampler changes. |
| [`terminal-ui/`](terminal-ui/README.md) | PTY smoke implemented | Prove a compiled Rust terminal process can use the existing PTY/pyte screen oracle. No production Rust UI yet. |

The sampler and episode-history experiments are separate crates with separate
lockfiles and Python bindings. Keep them independently buildable while they
remain behavior experiments. Add a Cargo workspace when shared Rust types or
coordinated release/build commands make one useful; do not introduce a
workspace just to make the directory look unified.

## Direction

The intended destination is a Rust-owned episode runtime and terminal UI.
Model-specific inference can remain behind Python adapters for Transformers
and existing llama.cpp bindings while that provides value. The first Rust UI
work is a small PTY smoke slice, before porting the full terminal application.

See [`ROADMAP.md`](ROADMAP.md) for migration order, boundaries, and gates. The
terminal slice has its own [brief](terminal-ui/README.md).

## Promotion rule

These experiments are not dependencies of the released Python packages.
Promote functionality only after behavioral parity, real-process integration,
packaging, and maintenance costs are reviewed. Preserve the existing Python
API and CLI during the 1.x line. Do not claim a speedup without repeatable
measurements against the same inputs and machine.

# Rust workspace

This repository has a top-level Cargo workspace for the Rust implementation
work. The crates build and test independently of the Python runtime. The
released `core` package remains the supported application runtime while the
Rust migration is evaluated; this workspace does not change Python package
dependencies or command behavior.

## Workspace layout

| Path | Responsibility | Current state |
| --- | --- | --- |
| `sampler/` | Numeric policy transforms, filtering, deterministic draws, and lazy metrics | Stage 2 policy surface has parity and real-model evidence. |
| `episode-history/` | Typed actions, outcomes, token evidence, history validation, and truncation | In-memory history kernel is implemented; the full episode engine and replay remain Python. |
| `terminal-ui/` | Standalone terminal process, fixture Choice turn, and Python inference worker | PTY and real-model probes are implemented; this is not yet the production UI. |

The Python adapters and their fixtures live beside each crate. The shared
Cargo lockfile is at the repository root. Rust build output goes to the root
`target/`, which is ignored by Git.

## Build and validate

Run Rust workspace commands from the repository root:

```sh
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo build -p rust-terminal-ui-smoke --locked
```

The PTY integration uses the existing Python screen oracle:

```sh
core/.venv/bin/python -m pytest -q tests/core/test_live_terminal_pty.py -k compiled_rust
core/.venv/bin/python -m pytest -q tests/core/test_rust_real_model_protocol.py
```

Python adapter parity checks require the `core` development environment and a
Maturin-built extension. For example, to build the sampler extension:

```sh
core/.venv/bin/maturin build \
  --manifest-path sampler/Cargo.toml \
  --interpreter core/.venv/bin/python \
  --out /tmp/rust-sampler-wheel
```

Each component guide documents its adapter installation, focused checks,
fixtures, and parity script: [sampler](docs/rust/SAMPLER.md),
[episode history](docs/rust/EPISODE_HISTORY.md), and
[terminal process](docs/rust/TERMINAL_UI.md).

## Model boundary

Model loading, device/cache ownership, and Transformers/PyTorch operations
remain Python-owned. The Rust real-model probe starts one persistent Python
worker and receives framed, full-vocabulary logits; it exercises both the
Transformers and existing llama.cpp adapters. Direct Rust-to-llama.cpp
interop remains a separate backend implementation choice.

The base Rust crates can build without Python. Python is required for parity
adapters and model-backed probes, not for the sampler and history libraries
themselves. Keep model latency, transfer costs, and input-to-frame latency as
separate measurements.

## Migration status

The PTY test seam, fixture Choice turn, and numeric policy surface are in
place. The next planned slice is episode execution and replay: map
`EpisodeEngine` observation/action/outcome transitions onto the Rust history
types, then preserve rewind, fork, divergence, and backend-position contracts.
Session persistence, production UI, and release packaging follow as separate
stages. See the [migration roadmap](docs/rust/ROADMAP.md) and the retained
[validation and boundary records](docs/rust/).

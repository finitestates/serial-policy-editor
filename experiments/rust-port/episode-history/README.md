# Rust episode history and action transitions

This is an isolated Rust crate for the in-memory policy action and history
kernel in `core/src/trajectory_editor/episode_history.py` and
`core/src/trajectory_editor/core/actions.py`. The Python runtime remains the
released implementation and behavior oracle. This experiment is not wired
into episode execution, replay, storage, SQLite, or the terminal UI.

The Rust crate owns typed action, evidence, outcome, attempt, history, and
truncation values. Its PyO3 entry point accepts one JSON document for a full
operation and returns one JSON document. The Python adapter converts existing
Python records at that boundary, so a history does not cross as a sequence of
per-token or per-field calls.

## Contract map

| Python contract | Rust operation and values | Shared parity coverage |
| --- | --- | --- |
| `action_from_dict()` and action `to_dict()` | `PolicyAction::from_value()` / canonical `to_value()` | Every action variant, canonical forms, aliases, malformed known actions, and unknown kinds |
| `EpisodeHistory.__post_init__()` | `EpisodeHistory::new()` / `validate()` | Ordered ordinals, root boundary zero, contiguous spans, evidence type/range/order/ID consistency |
| Visible ID, evidence, text, and boundary properties | `EpisodeHistory::visible_*()` | Empty histories, visible tokens, handoffs, and non-visible EOG evidence |
| `EpisodeHistory.truncate()` | `EpisodeHistory::truncate()` | Boundary zero/current/action edges, boundary events, partial writes/phrases/other actions, terminal and divergence cleanup, and invalid boundaries |
| `sampler_after_action()` | `sampler_after_action()` | Reroll changes only `seed`; `SetSampler` replaces the complete serialized config |

Fixture expectations in `fixtures/history-cases.json` are generated from the
Python implementation. Rust unit tests consume those same cases, and
`scripts/compare_with_python.py` reruns the current Python reference and the
built extension against them.

## Behavior and data boundary

- Boundaries are root-relative visible-token counts. Rust never rebases a
  retained prefix. A child/fork relationship is provenance and is not stored
  in this kernel; an input history already describes its root-relative frame,
  so it can be truncated before any external fork point.
- Non-visible evidence, including EOG, does not advance the boundary. A normal
  truncate excludes zero-width events exactly at the requested boundary;
  `include_boundary_events=True` includes them. Earlier zero-width attempts
  remain in raw history, including `handed-off` attempts.
- A cut inside a `Write` or `Phrase` becomes an exact `Write` containing the
  retained evidence text. A cut inside any other token-producing action
  becomes a finite `Hold`. Partial outcomes keep only visible evidence and
  clear terminal, divergence, replay-EOG, diagnostics, and other suffix state.
  The new replay expectation is built from the retained IDs and stop reason.
- Actions use their canonical `to_dict()` spellings. Accepted aliases match
  the current parser: `select`, `insert`, `check-phrase`, `force-phrase`,
  `teacher-eog`, and the legacy field names described by the fixtures.
  Unsupported kinds and malformed supported kinds remain separate error
  categories; the Python adapter maps unknown kinds to
  `UnsupportedPolicyActionKind`.
- Sampler configuration is opaque JSON inside Rust. Python validates and
  canonicalizes it with `SamplerConfig`; Rust preserves every field for
  `Reroll` and replaces the full object for `SetSampler`. Adapter results pass
  back through `SamplerConfig.from_record()`.
- `ActionOutcome.diagnostics` is preserved as a JSON value. The adapter
  requires diagnostics and any future extra payload to be JSON-serializable.
  SQLite-shaped records and durable-prefix projection stay in Python.
- The current Rust data model uses unsigned 64-bit ordinals/boundaries/ranks,
  signed 64-bit token IDs and sampling boundaries, and signed 64-bit reroll
  seeds. This matches the runtime's token/seed domain but is narrower than
  Python's arbitrary-size integers for ordinal, boundary, and rank fields.

## Build and compare

From the repository root, the reference tests are:

```sh
core/.venv/bin/python -m pytest -q \
  tests/core/test_episode_history.py \
  tests/core/test_episode_live_history.py \
  tests/core/test_sampler_contracts.py
```

Generate fixtures after an intentional contract change, then run the
independent Rust library tests:

```sh
core/.venv/bin/python experiments/rust-port/episode-history/scripts/generate_fixtures.py
cargo test --manifest-path experiments/rust-port/episode-history/Cargo.toml
```

Build/install the thin Python extension into the core test environment and
run the cross-language and adapter checks:

```sh
core/.venv/bin/maturin build \
  --manifest-path experiments/rust-port/episode-history/Cargo.toml \
  --interpreter core/.venv/bin/python \
  --out /tmp/rust-episode-history-wheel
# Run this only if the core virtualenv was created without pip.
core/.venv/bin/python -m ensurepip --upgrade
core/.venv/bin/python -m pip install --no-deps \
  /tmp/rust-episode-history-wheel/rust_episode_history_experiment-0.1.0-*.whl
core/.venv/bin/python experiments/rust-port/episode-history/scripts/compare_with_python.py
core/.venv/bin/python -m pytest -q \
  experiments/rust-port/episode-history/tests/test_python_adapter.py
```

Run Rust formatting and lint checks with:

```sh
cargo fmt --manifest-path experiments/rust-port/episode-history/Cargo.toml -- --check
cargo clippy --manifest-path experiments/rust-port/episode-history/Cargo.toml --features python --all-targets -- -D warnings
```

The wheel is an experiment-only package named `rust_episode_history`. Do not
add it as a dependency of `core`; production promotion would require a
separate review of parity, packaging, and maintenance costs.

## Rust notes for Python maintainers

`PolicyAction` is an enum: each value is exactly one action variant. History
records are structs containing their fields. Operations return `Result<T,
KernelError>`, which makes invalid input explicit at the call site. The
library uses owned values so records do not borrow memory from Python. No
`unsafe` code is used. PyO3 is an optional `python` Cargo feature used only
for the extension build; plain `cargo test` compiles the library without it.

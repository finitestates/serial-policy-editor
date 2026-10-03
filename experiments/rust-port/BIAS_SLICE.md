# Direct and grouped token-bias slice

**Status:** implemented and validated on 2026-10-03 in the isolated Rust
sampler experiment. Python production behavior remains the reference.

## Implemented behavior

The Rust library now exposes `bias_contributions`, `active_biases`, and
`apply_biases`. It accepts compiled token-ID routes and exact visible history;
it does not load a model tokenizer or expand text surfaces.

- A route matches when the token IDs before its final target token equal the
  end of visible history. A one-token route has an empty prefix and is always
  active.
- Each grouped bias contributes once per target token, even when multiple
  member routes in that group reach the target. Direct-token and overlapping
  group sources sum together.
- Contributions retain Python's fields, source names, group identity, route
  token IDs and surfaces, per-route active flags, and stable ordering.
  `include_inactive=True` returns unmatched group sources as well.
- Applying biases returns an owned vector. It checks finite input logits and
  results, history and route vocabulary IDs, and preserves its input. Raw
  logits and raw ranks remain unchanged; adjusted logits and policy ranks
  reflect the bias.
- The Python adapter converts `BiasGroup`, `BiasMember`, `BiasRoute`, and
  `BiasToken` records to Rust inputs and recreates the production
  `BiasContribution` and `BiasMemberRoute` dataclasses on return. The same
  kernels remain callable from Rust without initializing Python.

The existing `PolicyCalculations` order remains history penalties, optional
activation adjustments, direct/grouped biases, ephemeral biases, then
candidate filtering. This slice implements direct and grouped biases only; it
does not wire the experiment into the released runtime.

## Parity fixtures

`sampler/scripts/generate_fixtures.py` generates three bias cases from
production `BiasGroup`, `BiasMember`, `BiasRoute`, `BiasToken`,
`SamplerConfig.bias_contributions()`, `active_biases()`, and
`PolicyCalculations` behavior. A small tokenizer stand-in is used only by
Python's `BiasMember.compile()` to create case and leading-space route variants.
Rust and the Python comparison check source records and route identity exactly,
and finite adjusted values at absolute tolerance `1e-14`.

The cases cover overlapping direct/group sources, several matching routes in
one group, inactive phrase routes, exact and mismatching token prefixes,
unconditional one-token routes, compiled case/spacing variants, immutable
inputs, and raw/policy-rank changes. Additional Rust and adapter checks reject
missing multi-token history, invalid vocabulary IDs, non-finite logits, and
non-finite adjusted results. A Python adapter check composes history penalties,
activation, direct/grouped bias, and ephemeral bias in the production order
and matches `PolicyCalculations`.

## Validation run

The sampler extension was built for CPython 3.11 with Maturin and force
reinstalled into `core/.venv`. The following checks passed:

```text
cargo fmt --manifest-path experiments/rust-port/Cargo.toml --all -- --check
passed
cargo test --manifest-path experiments/rust-port/Cargo.toml --workspace --locked
passed: 13 Rust tests across episode history, sampler, and terminal worker
cargo clippy --manifest-path experiments/rust-port/Cargo.toml --workspace --all-targets --all-features --locked -- -D warnings
passed
core/.venv/bin/python -m pytest -q tests/core/test_sampler_contracts.py
passed: 24
core/.venv/bin/python -m pytest -q tests/core/test_live_terminal_pty.py -k compiled_rust
passed: 2, 12 deselected
core/.venv/bin/python -m pytest -q experiments/rust-port/sampler/tests/test_python_adapter.py
passed: 10
PYTHONPATH=core/src core/.venv/bin/python experiments/rust-port/sampler/scripts/compare_with_python.py
passed: 279 value checks, including all 3 bias cases
```

No benchmark was run, so this slice makes no performance claim. The remaining
Stage 2 policy work is recorded in
[`POLICY_CALCULATIONS_SLICE.md`](POLICY_CALCULATIONS_SLICE.md).

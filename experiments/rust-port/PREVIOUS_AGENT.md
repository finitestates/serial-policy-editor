# Completed slice: Rust history-penalty kernel

**Status:** implemented and validated on 2026-10-02. This file records the
completed slice, its scope, parity evidence, and known input-boundary
difference.

## Summary

The Rust sampler experiment now includes the history-penalty transform from
`core/src/trajectory_editor/core/policy_calculations.py`. The Python runtime
remains the released implementation and behavior oracle. The change is limited
to `experiments/rust-port/sampler/`; it does not port `PolicyCalculations` as a
whole or change production behavior.

The library API is `rust_sampler_native::apply_history_penalties()`. The
optional PyO3 adapter exposes it as
`rust_sampler.apply_history_penalties(logits, history_token_ids, config)`.
It accepts NumPy logits, an integer history sequence, and a config object with
the four history settings. It returns a new `float64` vector and leaves input
logits unchanged. Ordinary Rust builds continue to work without Python.

## Implemented behavior

For finite base logits and an exact history-token sequence, the transform:

1. Uses the full history when `repeat_last_n == -1`, no history when it is
   `0`, and the final N IDs when it is positive.
2. Validates every history ID against the full vocabulary before selecting the
   tail, matching the production policy-calculation boundary.
3. Applies the repeat penalty once per distinct token to its original logit:
   multiplies a negative value and divides a zero-or-positive value.
4. Subtracts presence penalty once per distinct token, then subtracts frequency
   penalty multiplied by that token's count in the selected tail.
5. Rejects invalid windows, invalid IDs, non-finite inputs/settings, and
   non-finite output.

Finite negative presence and frequency penalties are supported. Repeat penalty
must be finite and positive. A zero-length window leaves logits unchanged,
even when the other penalty values are nonzero. `history_token_ids=None` is
accepted by the Python adapter only when the settings make history penalties
inactive, matching `PolicyCalculations`.

The transform changes policy logits and policy rank. It leaves raw logits and
raw rank unchanged. Direct or grouped token biases, activation adjustments,
ephemeral biases, filtering, normalization, and sampling remain outside this
function.

## Parity evidence

Six shared cases in
`experiments/rust-port/sampler/fixtures/sampling-cases.json` are generated from
production `PolicyCalculations`. They cover empty and all-history inputs,
zero, short, and long positive windows, repeated IDs, IDs outside the selected
tail, positive/zero/negative logits, positive and negative presence/frequency
penalties, repeat penalty on/off, raw and policy ranks, and unchanged inputs.
Rust unit tests and the Python comparison/adapter checks consume these cases.

Invalid IDs, invalid windows, non-finite inputs/settings, and finite-input
overflow are also checked in Rust and adapter tests. Discrete ranks compare
exactly; adjusted logits use absolute tolerance `1e-14`.

### Known input-boundary difference

The Python adapter requires a one-dimensional integer ID sequence. The
production private helper casts general array-like history values to `int64`;
non-integer coercions are outside this exact-token slice and were not included
in parity cases. The adapter clamps positive `repeat_last_n` values above
signed-64-bit max before calling Rust. This selects the same tail for any
realizable in-memory history. No numerical differences were found in the
checked cases.

## Validation recorded for this slice

The following commands were run from the repository root:

```text
cargo fmt --manifest-path experiments/rust-port/Cargo.toml --all -- --check
  passed
cargo test --manifest-path experiments/rust-port/Cargo.toml --workspace --locked
  passed (7 sampler tests, 1 episode-history test)
cargo clippy --manifest-path experiments/rust-port/Cargo.toml \
  --workspace --all-targets --all-features --locked -- -D warnings
  passed
core/.venv/bin/python -m pytest -q tests/core/test_sampler_contracts.py
  passed (24 tests)
core/.venv/bin/maturin build --manifest-path experiments/rust-port/sampler/Cargo.toml \
  --interpreter core/.venv/bin/python --out /tmp/rust-sampler-wheel
  passed (CPython 3.11 wheel)
core/.venv/bin/python -m pip install --force-reinstall --no-deps \
  /tmp/rust-sampler-wheel/rust_sampler_experiment-0.1.0-cp311-cp311-manylinux_2_34_x86_64.whl
  passed
PYTHONPATH=core/src core/.venv/bin/python \
  experiments/rust-port/sampler/scripts/compare_with_python.py
  passed (186 checks, 14 draw cases, 6 history-penalty cases)
PYTHONPATH=core/src core/.venv/bin/python -m pytest -q \
  experiments/rust-port/sampler/tests/test_python_adapter.py
  passed (7 tests)
```

`uv` was unavailable, so the built wheel was installed with the pip fallback
documented in `experiments/rust-port/sampler/README.md`. No speed claim was
measured. The changes remain uncommitted for maintainer review.

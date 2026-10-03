# Stage 2: Rust policy calculations

**Status:** implemented and validated on 2026-10-03 in the isolated Rust
sampler experiment. Python remains the released runtime and behavior oracle.

## Implemented surface

The Rust sampler now covers the numeric policy pipeline around one logits
snapshot:

1. History penalties.
2. Output-head activation-logit adjustments supplied by the model backend.
3. Direct-token and exact-prefix grouped biases.
4. Ephemeral sparse token biases for the current decision.
5. Temperature, top-k, typical-p, tail-free, top-p, and min-p filtering.
6. Sparse draw probabilities and deterministic sampler draws.

`cfg_combine_logits` implements the numeric CFG blend
`U + scale * (C - U)`. Model loading, unconditional-prompt tokenization,
continuation positioning, cache ownership, and CFG lifecycle stay in Python.
Output-head projection also remains a model operation; Rust receives its
full-vocabulary adjustment vector.

The Rust `PolicyMetrics` type keeps raw and policy logits separate and computes
maximum, raw/policy probabilities, raw NLL and log-sum-exp, raw/policy ranks,
top IDs, and full-vocabulary raw-logit z-scores on demand. Ranking and z-scores
do not force dense probability normalization. An unchanged policy can share the
raw normalization surface; an explicit adjusted vector gets its own cache.

The PyO3 adapter accepts NumPy arrays and existing `SamplerConfig` values at
the boundary. Plain Rust callers can use the same kernels without initializing
Python. The experiment is not wired into `core`; package dependencies, CLI
behavior, and release configuration are unchanged. `SamplerConfig` and action
serialization remain Python-owned compatibility boundaries.

## Parity evidence

Shared fixtures are generated from production `PolicyCalculations` and
`sampling.py`. They cover positive, negative, and zero activation strengths;
inactive activation; sparse multiple-token updates; unchanged input arrays;
out-of-range IDs; non-finite inputs and overflow; raw-rank preservation;
policy-rank changes; and an ordered combination of history, activation,
grouped/direct bias, and ephemeral adjustments. CFG fixtures cover scales 0, 1,
and 2. Metric fixtures cover identity and adjusted policy surfaces, deferred
normalization, selected probabilities, NLL, full-vocabulary population
z-scores, and undefined z-scores for flat logits.

The Rust tests and the Python adapter comparison consume the same fixtures.
Integer IDs, ranks, and filter stages compare exactly; finite float comparisons
use absolute tolerance `1e-14`. Error-category, non-mutation, and lazy-cache
checks also run at the adapter boundary.

## Real-model evidence

`sampler/scripts/real_model_policy.py` loads both conditional and unconditional
branches for each profile. It checks two successive shared-continuation
boundaries, projects a deterministic activation direction through each real
model's output head, then compares Rust CFG, adjustment order, candidate
filtering, ranks, probabilities, NLL, and z-scores with Python.

The 2026-10-03 run passed for GPT-2 through Transformers/CPU and Llama 3.2 1B
Q4_K_M through llama.cpp/CPU at boundaries 0 and 1. Both profiles had zero
maximum absolute delta for CFG logits and adjusted policy logits. The report is
`/tmp/spe-rust-policy-real-model.json`.

The existing production CFG lifecycle scenario also passed with both real
profiles. Reports are under
`/tmp/spe-rust-policy-stage2-cfg-lifecycle/` (GPT-2 run
`20261003T152247Z-4fd754c0.json`; Llama run
`20261003T152256Z-09fa66e3.json`). Those runs validate the current Python
backend lifecycle, while the Rust real-model probe validates the numeric
combination and policy kernels against the same backend outputs.

## Validation

Run the commands listed in [`SAMPLER.md`](SAMPLER.md). The final
recorded results are:

```text
cargo fmt --manifest-path Cargo.toml --all -- --check
passed
cargo test --manifest-path Cargo.toml --workspace --locked
passed: 14 tests across episode history, sampler, and terminal worker
cargo clippy --manifest-path Cargo.toml --workspace --all-targets --all-features --locked -- -D warnings
passed
core/.venv/bin/python -m pytest -q tests/core/test_sampler_contracts.py
passed: 24
core/.venv/bin/python -m pytest -q tests/core/test_cfg_contracts.py
passed: 28
core/.venv/bin/python -m pytest -q tests/core/test_vector_contracts.py
passed: 6
core/.venv/bin/python -m pytest -q sampler/tests/test_python_adapter.py
passed: 13
PYTHONPATH=core/src core/.venv/bin/python sampler/scripts/compare_with_python.py
passed: 437 value checks
```

The Maturin wheel was rebuilt for CPython 3.11 and force-installed with pip
because `uv` was unavailable on this checkout's PATH. No benchmark was run, so
this slice makes no performance claim.

Stage 2's numeric policy work is complete. Stage 3 moves action execution,
observation/evidence construction, and replay into Rust; see
[`ROADMAP.md`](ROADMAP.md).

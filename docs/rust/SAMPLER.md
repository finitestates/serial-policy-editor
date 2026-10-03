# Rust sampler experiment

This directory contains an experimental Rust implementation of the numeric
sampler in `core/src/trajectory_editor/core/sampling.py` and the numeric policy
surface around `core/src/trajectory_editor/core/policy_calculations.py`.

The Python package remains the released implementation and behavior oracle.
This crate is not a production dependency. Repository-wide experiment rules
are in [`../../RUST.md`](../../RUST.md); Rust migration gates are in
[`ROADMAP.md`](ROADMAP.md).

## Implemented scope

The Rust library implements the numeric sampler and is callable from Python
through a small binding. Rust owns the sampling calculations. The Python side
converts NumPy arrays and Python values at the boundary, calls Rust, and
converts results back to the existing Python-facing types. The Rust terminal
process now also calls this library without initializing Python; see the
completed fixture integration in
[`PRIOR_SLICE.md`](PRIOR_SLICE.md). The history-penalty transform is
another implemented kernel in this crate; its completed-slice record and
validation evidence are in [`PREVIOUS_AGENT.md`](PREVIOUS_AGENT.md).
Direct and grouped token biases are recorded in
[`BIAS_SLICE.md`](BIAS_SLICE.md). The full Stage 2 policy surface,
validation, and real-model evidence are recorded in
[`POLICY_CALCULATIONS_SLICE.md`](POLICY_CALCULATIONS_SLICE.md). Corrected
real-model Choice results are in [`REAL_MODEL_CHOICE.md`](REAL_MODEL_CHOICE.md).

Use `sampling.py` as the specification for sampler algorithms and
`policy_calculations.py` for policy transforms and evidence metrics. The
implemented sampler surface includes:

- candidate filtering: temperature, top-k, typical-p, tail-free, top-p, and
  min-p, with the same ordering and diagnostics;
- raw ranks and top raw token IDs;
- deterministic position-addressed random values and the existing RNG scheme;
- categorical, Gumbel-Max, Gaussian-Max, logistic-Max, Student-t-Max,
  Laplace-Max, and Uniform-Max draws;
- token-ID and model-rank Gumbel addressing;
- conditional Gumbel top-k used by stochastic beam search;
- Gumbel ranking, winner selection, ranked IDs, and seed search;
- current validation, tie-breaking, and error behavior where callers rely on
  it.

Do not change the RNG scheme, hash input strings, integer encoding, draw
coordinates, or tie-breaking rules as a way to make the port easier. A fixed
seed, stream fingerprint, boundary, sampler configuration, and candidate data
must continue to produce the same observable result as Python.

## History penalty transform

`apply_history_penalties(logits, history_token_ids, config)` exposes the
production history-penalty step without moving the rest of
`PolicyCalculations` into Rust. It returns a new NumPy `float64` vector and
does not mutate the input logits. The Rust library function accepts logit and
token-ID slices plus the four history settings and returns an owned vector in
`Result`.

The transform validates finite, nonempty logits, validates every history ID
against the vocabulary, and then selects the configured tail: `-1` uses all
history, `0` uses none, and a positive value uses that many final IDs. For
each distinct ID in the tail it applies the repeat penalty once to the
original logit (multiply negative values, divide zero or positive values),
subtracts presence penalty once, then subtracts frequency penalty times that
ID's count. Presence and frequency penalties may be negative; repeat penalty
must be finite and positive. Invalid windows, invalid IDs, non-finite settings,
and non-finite results are rejected. A zero-length window returns unchanged
logits even when the other penalty values are nonzero. `history_token_ids=None`
is accepted only when the settings make history penalties inactive, matching
the production `PolicyCalculations` boundary. The adapter expects a
one-dimensional array of integer IDs. Python permits arbitrarily large positive
window integers; the adapter clamps those to signed-64-bit max, which selects
the same tail for any realizable history.

This transform changes policy logits and policy rank only. Raw logits and raw
ranks remain available unchanged.

## Direct and grouped token biases

The adapter exposes `active_biases(history, config)`,
`bias_contributions(history, config, include_inactive=False)`, and
`apply_biases(logits, history, config)`. The first returns Rust-summed source
totals by token ID. The second returns the production `BiasContribution` and
`BiasMemberRoute` dataclasses, including route token IDs, surfaces, source
names, and per-route match state. The third returns an owned NumPy `float64`
policy-logit vector without changing its input.

Grouped routes match when all route tokens before the final target token equal
the end of visible token history. A one-token route has an empty prefix and is
always active. The kernel counts a group's amount once per target token, sums
overlapping groups and direct-token sources, and orders contributions with the
same stable source and route identity order as Python. `include_inactive`
retains unmatched grouped routes for explanation. Python compiles case and
spacing surfaces with `BiasMember.compile`; Rust receives those token-ID
routes and never loads a tokenizer.

The Rust `active_biases`, `bias_contributions`, and `apply_biases` functions
also work as a plain library without initializing Python. Applying biases
checks finite logits and results, history and route vocabulary IDs, and returns
a copy. Raw logits and raw ranks stay unchanged; adjusted logits and policy
ranks reflect the bias.

## Activation, ephemeral biases, and CFG

`apply_activation_adjustments(logits, adjustments, strength)` applies a full
vocabulary output-head adjustment vector already computed by the Python model
backend. It validates the vector shape and finite values, returns an owned
array, supports signed finite strengths, and rejects non-finite results. Vector
extraction and output-head projection remain model operations owned by Python.

`apply_ephemeral_biases(logits, mapping)` applies sparse token adjustments for
one decision. It validates token IDs, values, and the resulting logits, then
returns an owned array. `cfg_combine_logits(conditional, unconditional, scale)`
implements the production CFG formula
`unconditional + scale * (conditional - unconditional)`. The two backend
branches, tokenization, cache positioning, and lifecycle remain outside this
numeric kernel.

The composed policy order matches production:

1. History penalties.
2. Output-head activation adjustments when active.
3. Direct and grouped token biases.
4. Ephemeral per-decision biases.
5. Candidate filtering and sparse draw probabilities.

## Lazy policy metrics

The Rust `PolicyMetrics` type keeps raw and adjusted policy logits and computes
metrics only when requested. It provides the raw maximum, raw and policy
probabilities for selected IDs, raw NLL and log-sum-exp, raw and policy ranks,
top IDs, and full-vocabulary raw-logit z-scores. Ranking and z-scores do not
trigger dense soft-max normalization. The adapter exposes the same behavior as
`rust_sampler.PolicyMetrics`; passing `policy_logits=None` means the policy
shares the raw metric surface, while an explicit vector keeps an independent
policy normalization cache.

`SamplerConfig` validation/serialization, action serialization, backend model
operations, and the released runtime remain Python-owned boundaries. This
experiment does not wire Rust into the released runtime or change package
dependencies, CLI behavior, or release configuration.

## Keep the boundary understandable

The existing Python implementation uses NumPy arrays and Python exceptions.
The Rust implementation does not need to mimic those internal types. Use
ordinary Rust data such as vectors, structs, enums, and `Result` values inside
the crate. Keep the Python binding thin and translate inputs, outputs, and
errors at that edge. Avoid passing arbitrary Python objects into the Rust
algorithms.

If a Rust term is unfamiliar, explain it in comments or in this guide when it
helps a Python reader:

- A **crate** is a Rust package, roughly like a Python package.
- A **type** describes what data a value can contain. Enums are useful for
  choices such as the draw kernel.
- `Result<T, E>` means an operation either returned a value of type `T` or an
  error of type `E`; handle errors explicitly instead of silently continuing.
- **Ownership** is Rust's rule for who is responsible for a value and how long
  it can be used. Prefer straightforward owned inputs and outputs here; do not
  add clever borrowing or unsafe code for performance without evidence.
- **PyO3** is a library for connecting Rust code to Python. **maturin** builds
  and packages that connection.

## Files

```text
sampler/
├── Cargo.toml
├── pyproject.toml
├── src/
│   └── lib.rs
├── python/rust_sampler/
│   └── __init__.py
├── tests/
│   ├── sampling.rs
│   └── test_python_adapter.py
├── fixtures/
│   └── sampling-cases.json
└── scripts/
    ├── real_model_policy.py
    ├── compare_with_python.py
    └── generate_fixtures.py
```

The shared fixture file includes numeric sampler, history-penalty, direct and
grouped bias, ordered policy-adjustment, CFG, and lazy-metric cases. Rust tests
and the Python comparison script consume it; its generator derives expectations
from production `BiasGroup`, `BiasMember.compile`, `BiasRoute`, `BiasToken`, and
`PolicyCalculations` behavior.

## Useful starting points

- Implementation: `core/src/trajectory_editor/core/sampling.py`
- Sampler tests: search `tests/core` for `sampling`, `gumbel`, and
  `perturbation` test files
- Runtime boundary: `core/src/trajectory_editor/core/policy_calculations.py`
- RNG name and seed/boundary contract: `RNG_SCHEME`, `position_uniform`, and
  the Gumbel/perturbation ranking functions in `sampling.py`

## Build and try the experiment

The experiment is a separate mixed Python/Rust package. The released `core`
package does not depend on it. From the repository root, install the core test
environment and Maturin, build a wheel, then install that wheel into the core
environment:

```sh
uv sync --project core --extra test
uv pip install --python core/.venv/bin/python maturin
core/.venv/bin/maturin build \
  --manifest-path sampler/Cargo.toml \
  --interpreter core/.venv/bin/python \
  --out /tmp/rust-sampler-wheel
uv pip install --reinstall --python core/.venv/bin/python --no-deps \
  /tmp/rust-sampler-wheel/rust_sampler_experiment-0.1.0-*.whl
```

The wheel install uses `--no-deps` because `core` already installs NumPy. A
standalone environment should install the wheel's declared NumPy dependency.
Cargo downloads the Rust crates listed in `Cargo.toml` the first time it builds.
If `uv` is unavailable, install Maturin and the built wheel with pip:

```sh
core/.venv/bin/python -m pip install maturin
core/.venv/bin/python -m pip install --force-reinstall --no-deps \
  /tmp/rust-sampler-wheel/rust_sampler_experiment-0.1.0-*.whl
```

The force option also refreshes the extension when rebuilding version `0.1.0`.

The import package is named `rust_sampler`. Its sampler surface mirrors
`trajectory_editor.core.sampling`; it also exposes the policy kernels described
above. The adapter represents IDs and probability/score arrays with NumPy
arrays and maps Rust input-validation errors to `EditorError` or `ValueError` as
appropriate. For example:

```python
import numpy as np
from rust_sampler import SparseDistribution, draw_token

distribution = SparseDistribution(
    np.array([4, 1, 7], dtype=np.int64),
    np.array([0.2, 0.5, 0.3]),
    np.array([0.1, 0.9, 0.4]),
)
token_id = draw_token(
    distribution,
    seed=17,
    stream_fingerprint="a" * 64,
    aligned_step=3,
    kernel="gumbel-max",
)
```

## Parity fixtures and checks

The shared fixture covers filter stages and diagnostics, signed seed
endpoints, boundaries larger than 64 bits, token-ID and model-rank addresses,
candidate-order changes, duplicate IDs, every draw kernel, Student-t df
values 3/1/0.5, conditional Gumbel top-k, and lazy seed search. Adapter tests
also cover exact ties, invalid dimensions and settings, and model-rank
validation. Policy expectations come from production `PolicyCalculations` and
include activation scaling, ordered history/activation/group/direct/ephemeral
steps, CFG blend scales, lazy raw/policy probabilities and NLL, population
z-scores, flat-logit undefined scores, raw-rank preservation, and immutable
inputs. Rust tests and the adapter comparison consume the shared cases.

Use the workspace format/test/Clippy commands in
[`../../RUST.md`](../../RUST.md). After installing the wheel, run the Python
parity and adapter checks from the repository root:

```sh
PYTHONPATH=core/src core/.venv/bin/python \
  sampler/scripts/compare_with_python.py
PYTHONPATH=core/src core/.venv/bin/python -m pytest -q \
  sampler/tests/test_python_adapter.py
```

The comparison uses exact equality for integer IDs, ranks, filter stages, and
draws and an absolute `1e-14` tolerance for finite floating-point values,
including adjusted logits, probabilities, NLL, and z-scores. Replay-affecting
draw and winner results match exactly for the checked fixtures.
Regenerate expected data only after reviewing the Python reference results:

```sh
PYTHONPATH=core/src core/.venv/bin/python \
  sampler/scripts/generate_fixtures.py
```

Run the live numeric policy probe against local Transformers and llama.cpp
profiles with:

```sh
PYTHONPATH=core/src core/.venv/bin/python \
  sampler/scripts/real_model_policy.py \
  --model-root "$HOME/Downloads/models" \
  --profile benchmarks/profiles/gpt2_cpu_cfg.yaml \
  --profile benchmarks/profiles/llama_1b_cfg.yaml \
  --output /tmp/spe-rust-policy-real-model.json
```

`find_seed_for_token` accepts a caller-provided Python seed generator, so the
adapter asks it for one seed at a time. Rust checks the target's eligibility,
draws each candidate, and reports a match; this keeps generator calls lazy
without sending a Python callback into the Rust algorithms.

The Rust code uses no `unsafe` blocks. Per-slice validation outcomes are
recorded in the linked handoff documents.

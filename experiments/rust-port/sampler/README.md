# Rust sampler experiment

This directory contains an experimental Rust implementation of
the numeric sampler in `core/src/trajectory_editor/core/sampling.py`.

The goal is to learn Rust while seeing whether the existing sampler can be
ported faithfully behind a small Python-to-Rust boundary. The Python package
remains the released implementation and the reference for behavior during
this experiment. Do not move or replace production code as part of the first
pass.

## Implemented scope

The Rust library implements the numeric sampler and is callable from Python
through a small binding. Rust owns the sampling calculations. The Python side
converts NumPy arrays and Python values at the boundary, calls Rust, and
converts results back to the existing Python-facing types. The next integration
slice will also make this arithmetic callable from a Rust process without
initializing Python; see [`../NEXT_SLICE.md`](../NEXT_SLICE.md).

Use the current `sampling.py` as the authoritative behavioral specification.
Cover its public functions and behavior, including:

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

## Keep the boundary understandable

The existing Python implementation uses NumPy arrays and Python exceptions.
The Rust implementation does not need to mimic those internal types. Use
ordinary Rust data such as vectors, structs, enums, and `Result` values inside
the crate. Keep the Python binding thin and translate inputs, outputs, and
errors at that edge. Avoid passing arbitrary Python objects into the Rust
algorithms.

If a Rust term is unfamiliar, explain it in comments or in this README when it
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

## Suggested layout

Keep the experiment self-contained here. A reasonable starting layout is:

```text
experiments/rust-port/sampler/
├── README.md
├── Cargo.toml
├── pyproject.toml
├── src/
│   └── lib.rs
├── tests/
│   └── sampling.rs
├── fixtures/
│   └── sampling-cases.json
└── scripts/
    └── compare_with_python.py
```

The exact Rust module split is up to the implementer. Keep related logic
together, and split files when it makes the code easier to follow. The fixtures
should be readable data shared by the Rust tests and the Python comparison
script, not a second hand-written sampler.

## Implementation and review sequence

1. Treat `sampling.py` and its tests as the behavioral specification. Preserve
   the functions, validation rules, tie-breaking, and RNG details.
2. Keep the pure algorithms in the Rust library and use Rust unit tests plus
   `cargo fmt`/`cargo clippy` to keep the code legible.
3. Extend shared parity cases when a contract changes. For each case, run
   Python and Rust on identical inputs and compare outputs. Use exact
   comparison for integer IDs and discrete choices; use a documented tight
   tolerance for floating-point results where different math libraries
   produce rounding differences. Also verify that sampler draws and winners
   match exactly on the fixtures, because those affect replay. Do not hide a
   mismatch by widening the tolerance without explaining it.
4. Preserve the PyO3/maturin binding and Python adapter. Check that errors
   become useful Python exceptions and that NumPy inputs are validated rather
   than silently reshaped or truncated.
5. Run the existing focused Python sampler and replay tests against the
   unchanged Python implementation as a baseline. Then run the shared
   differential fixtures against both implementations and exercise the Rust
   binding directly from Python. Do not rewire the released runtime to Rust in
   this experiment.
6. Report what is equivalent, any known differences, build/install commands,
   validation actually run, and what a Python maintainer would need to know
   to modify the Rust code.

## Required behavior checks

Parity coverage must include at least:

- candidate arrays in more than one order, including repeated token IDs where
  the Python contract permits them;
- exact score ties and deterministic token-ID tie-breaking;
- zero and non-default noise scales, plus invalid negative/non-finite values;
- minimum and maximum supported seeds, boundaries, and malformed stream
  fingerprints;
- token-ID addressing remaining attached to token IDs when candidate order
  changes;
- model-rank addressing using the supplied model ranks, including validation
  of missing, duplicate, or invalid ranks;
- each draw kernel, including Student-t degrees-of-freedom edge cases;
- candidate-filter stage outputs and diagnostics;
- conditional Gumbel top-k and seed search;
- invalid dimensions, empty inputs, and mismatched candidate lengths.

First inspect the existing tests: extend or reuse their cases where practical
instead of inventing expectations that conflict with the Python contract.

## Boundaries and non-goals

- Do not rewrite `policy_calculations.py`, `SamplerConfig`, the episode engine,
  persistence, inference backends, terminal UI, or vector package.
- Do not change user-visible sampler semantics or the RNG scheme.
- Do not make Rust a required dependency of the released `core` package.
- Do not add `unsafe` Rust unless there is a measured need and a clear safety
  explanation.
- Do not claim the Rust port is faster without a repeatable benchmark against
  the same inputs and machine; correctness and learnability come first.
- Do not commit, tag, publish, or promote the experiment into the production
  package as part of this task.

## Useful starting points

- Implementation: `core/src/trajectory_editor/core/sampling.py`
- Sampler tests: search `tests/core` for `sampling`, `gumbel`, and
  `perturbation` test files
- Runtime boundary: `core/src/trajectory_editor/core/policy_calculations.py`
- RNG name and seed/boundary contract: `RNG_SCHEME`, `position_uniform`, and
  the Gumbel/perturbation ranking functions in `sampling.py`

The work is successful when a maintainer who does not know Rust can follow the
build instructions, see how Rust results were compared with Python, understand
the remaining differences, and decide whether the experiment should continue.

## Build and try the experiment

The experiment is a separate mixed Python/Rust package. The released `core`
package does not depend on it. From the repository root, install the core test
environment and Maturin, build a wheel, then install that wheel into the core
environment:

```sh
uv sync --project core --extra test
uv pip install --python core/.venv/bin/python maturin
core/.venv/bin/maturin build \
  --manifest-path experiments/rust-port/sampler/Cargo.toml \
  --interpreter core/.venv/bin/python \
  --out /tmp/rust-sampler-wheel
uv pip install --python core/.venv/bin/python --no-deps \
  /tmp/rust-sampler-wheel/rust_sampler_experiment-0.1.0-*.whl
```

The wheel install uses `--no-deps` because `core` already installs NumPy. A
standalone environment should install the wheel's declared NumPy dependency.
Cargo downloads the Rust crates listed in `Cargo.toml` the first time it builds.

The import package is named `rust_sampler`. Its public surface mirrors
`trajectory_editor.core.sampling` where the numeric functions are meaningful;
the adapter represents IDs and probability/score arrays with NumPy arrays and
maps Rust input-validation errors to `EditorError` or `ValueError` as
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

`fixtures/sampling-cases.json` is shared by the Rust integration tests and the
Python comparison script. The fixture generator calls the released Python
sampler and writes its expected values. It includes filter stages and
diagnostics, signed seed endpoints, boundaries larger than 64 bits, token-ID
and model-rank addresses, candidate-order changes, duplicate IDs, all draw
kernels, Student-t df values 3/1/0.5, conditional Gumbel top-k, and a lazy seed
search.

Run the checks from the repository root after installing the wheel:

```sh
cargo fmt --manifest-path experiments/rust-port/sampler/Cargo.toml -- --check
cargo test --locked --manifest-path experiments/rust-port/sampler/Cargo.toml
cargo clippy --all-targets --locked \
  --manifest-path experiments/rust-port/sampler/Cargo.toml -- -D warnings
PYTHONPATH=core/src core/.venv/bin/python \
  experiments/rust-port/sampler/scripts/compare_with_python.py
PYTHONPATH=core/src core/.venv/bin/python -m pytest -q \
  experiments/rust-port/sampler/tests/test_python_adapter.py
```

The comparison uses exact equality for integer IDs and draws and an absolute
`1e-14` tolerance for floating-point scores. That tolerance covers the checked
cases; Rust and Python may differ by a few low bits for transcendental
functions on other inputs. The replay-affecting draw and winner results match
exactly for the checked fixtures. The extension and differential suite were
also built and run under CPython 3.11 and 3.14 in the development environment.
Regenerate expected data only after reviewing the Python reference results:

```sh
PYTHONPATH=core/src core/.venv/bin/python \
  experiments/rust-port/sampler/scripts/generate_fixtures.py
```

`find_seed_for_token` accepts a caller-provided Python seed generator, so the
adapter asks it for one seed at a time. Rust checks the target's eligibility,
draws each candidate, and reports a match; this keeps generator calls lazy
without sending a Python callback into the Rust algorithms.

No speed claim has been measured. The code uses safe Rust and has no `unsafe`
blocks. Build/install or test the experiment directly; do not add it to the
released core dependencies or route runtime draws through it as part of this
experiment.

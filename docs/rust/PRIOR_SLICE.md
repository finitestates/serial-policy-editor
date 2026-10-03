# Completed slice: deterministic Rust Choice turn

**Status:** implemented and validated on 2026-10-02.

## Objective

Connect the existing Rust sampler and episode-history kernels to the Rust
terminal process for one complete, fixture-driven Choice interaction. Keep the
Python implementation as the behavior oracle and the existing Python PTY
harness as the process/screen test driver.

The slice should show one path through candidate calculation, a visible
proposal, an accepted action, updated Rust history, and the resulting terminal
frame. It is an experiment only; the released Python package remains unchanged.

## Scenario

Use a small checked-in fixture with a short context and three candidates. The
fixture supplies token IDs, decoded token text, scores/logits, sampler config,
seed, stream fingerprint, and root-relative boundary. Choose values that make
the expected proposal deterministic under the existing Python sampler.

The Rust process should:

1. Build the filtered candidate view and proposal with the Rust sampler.
2. Render a compact Choice screen with the proposal and candidate ranks.
3. Accept the existing `accept` command through the same submit path used by
   the current Choice interaction.
4. Record the accepted action, token evidence, and updated visible boundary
   with the Rust episode-history types.
5. Render the committed context and remain available long enough for the PTY
   harness to capture it before clean test exit.

Keep the fixture data and expected semantic result separate from terminal
output. The slice must not load a model or start a Python worker; it exercises
the Rust path using deterministic backend output. This leaves the eventual
inference process boundary open.

## Visual checkpoints

Before changing renderer code, author the exact expected cell grids and cursor
positions for these named checkpoints at **100x30**:

- `choice.ready`: context, three candidates, and the proposed candidate;
- `choice.accepted`: the committed text, action, and new boundary;
- `choice.resized`: the accepted state after resizing to **80x24**, then back
  to 100x30.

Use a compact ASCII wireframe as the content guide:

```text
Rust Choice / fixture turn
Context @ boundary 3
The cat sat

Rank  Token  Text
  1     17   on
> 2     23   by
  3     31   down

Proposal: rank 2 / token 23 / " by"

Choice > accept
Enter submits · 1..N selects · q exits
```

The fixture must cause that proposal; do not hard-code a proposal that bypasses
the Rust sampler. Store the complete expected grids in the test contract, not
as snapshots generated from the Rust frame log. The current pyte oracle checks
cells and cursor; keep color/style expectations to what that oracle actually
asserts.

## Implementation scope

- Extend the existing `terminal-ui` process and PTY test path. Avoid creating a
  second terminal event loop or copying the Python UI.
- Add path dependencies from the terminal crate to the sampler and
  episode-history crates through the workspace.
- Keep the sampler algorithms usable as a normal Rust library. Gate its PyO3
  imports, module, and adapter-only code behind an optional `python` feature;
  keep the maturin build enabling that feature so the existing Python adapter
  remains available for parity checks. Episode-history already has this
  feature boundary.
- Keep the semantic test record in a sidecar separate from stdout/stderr and
  the existing `SPE_TERMINAL_FRAME_LOG`. Include the canonical action,
  selected token ID, evidence, and before/after boundary so the integration
  test can compare state as well as pixels/cells.
- Preserve the existing PTY raw capture, ordered frame offsets, complete-row
  checks, erase checks, resize handling, and negative control.

## Invariants and non-goals

- Preserve the Python command meaning, sampler ordering, RNG scheme,
  seed/fingerprint/boundary coordinates, token IDs, and serialized action and
  history forms.
- Use the Python runtime to establish fixture expectations. Compare discrete
  outputs exactly; keep any floating-point tolerance narrow and documented.
- Keep Rust implementation changes in the root workspace plus the existing
  `tests/core/test_live_terminal_pty.py` harness. Do not change `core/src`,
  `core/pyproject.toml`, the released CLI, persistence/SQLite, model backends,
  or production packaging.
- Do not add `unsafe` Rust, broaden this to the full Choice view, or claim a
  speedup. If the sampler cannot be consumed without linking Python, separate
  the binding cleanly rather than introducing Python into the Rust process.

## Required checks

Run from the repository root:

```sh
cargo fmt --manifest-path Cargo.toml --all -- --check
cargo test --manifest-path Cargo.toml --workspace --locked
cargo clippy --manifest-path Cargo.toml \
  --workspace --all-targets --all-features --locked -- -D warnings
core/.venv/bin/python -m pytest -q tests/core/test_live_terminal_pty.py -k compiled_rust
core/.venv/bin/python -m pytest -q tests/core/test_live_terminal_pty.py
```

Also build the sampler's Python extension with its `python` feature, then run
its Python comparison script and adapter tests as documented in
[`SAMPLER.md`](SAMPLER.md). Run the focused episode-history
Python adapter checks from [`EPISODE_HISTORY.md`](EPISODE_HISTORY.md).

## Handoff

The shared fixture is `terminal-ui/fixtures/choice-turn.json`. Python's sampler
produces active candidate order `[17, 23, 31]`, ranks `[1, 2, 3]`, probabilities
`[0.36716540111092555, 0.3322249935333473, 0.3006096053557273]`, and proposal
token `23` for seed `17`, fingerprint `a` repeated 64 times, and boundary `3`.
Submitting `accept` records canonical action `{"kind":"accept"}`, selected
token `23`, evidence at sampling boundary `3`, and visible boundary `3 -> 4`;
the visible text becomes `The cat sat by`.

The test compares the authored `choice.ready` and `choice.accepted` grids at
100x30, then `choice.resized` at 80x24 and again after returning to 100x30. It
captures raw PTY bytes, replays each logged frame with pyte, checks complete
rows/cursor/erase-free transactions, and confirms the oracle rejects a missing
proposal row. Probability comparisons use a relative tolerance of `1e-12` with
an absolute tolerance of `1e-14` to allow small arithmetic differences across
runtimes and platforms. Token IDs, action, evidence flags, and boundaries are
compared exactly.

The initial three-token context is recorded as a typed exact-write seed attempt
because `EpisodeHistory` validates root-relative history from boundary zero;
the tested `accept` is the following attempt. This is fixture setup, not a new
production action path.

Validation run from the repository root:

```text
cargo fmt --manifest-path Cargo.toml --all -- --check  PASS
cargo test --manifest-path Cargo.toml --workspace --locked  PASS
cargo clippy --manifest-path Cargo.toml --workspace --all-targets --all-features --locked -- -D warnings  PASS
core/.venv/bin/python -m pytest -q tests/core/test_live_terminal_pty.py -k compiled_rust  2 passed
core/.venv/bin/python -m pytest -q tests/core/test_live_terminal_pty.py  14 passed
```

The sampler maturin wheel built and was installed in `core/.venv`; its Python
comparison reported 128 value checks across 14 draw cases, and its adapter
tests reported 5 passed. The episode-history wheel also built and was
installed; its comparison reported 47 shared cases and its adapter tests
reported 4 passed. Exact extension and parity commands:

```sh
core/.venv/bin/maturin build --manifest-path sampler/Cargo.toml --interpreter core/.venv/bin/python --out /tmp/rust-sampler-wheel
core/.venv/bin/python -m pip install --force-reinstall --no-deps /tmp/rust-sampler-wheel/rust_sampler_experiment-0.1.0-cp311-cp311-manylinux_2_34_x86_64.whl
PYTHONPATH=core/src core/.venv/bin/python sampler/scripts/compare_with_python.py
PYTHONPATH=core/src core/.venv/bin/python -m pytest -q sampler/tests/test_python_adapter.py
core/.venv/bin/maturin build --manifest-path episode-history/Cargo.toml --interpreter core/.venv/bin/python --out /tmp/rust-episode-history-wheel
core/.venv/bin/python -m pip install --force-reinstall --no-deps /tmp/rust-episode-history-wheel/rust_episode_history_experiment-0.1.0-cp311-cp311-manylinux_2_34_x86_64.whl
core/.venv/bin/python episode-history/scripts/compare_with_python.py
core/.venv/bin/python -m pytest -q episode-history/tests/test_python_adapter.py
```

Both native extensions are enabled by their existing `python` Cargo feature.
Their internal extension names now match their unique Rust library target
names, avoiding Cargo output collisions in the shared workspace while
retaining the `rust_sampler` and `rust_episode_history` Python packages.

No inference backend, released package, SQLite path, or production UI changed.
No speed claim was measured. This slice is committed on the experiment branch
and remains outside the released runtime.

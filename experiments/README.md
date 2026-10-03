# Experiments

This directory holds isolated implementation and design experiments for
Serial Policy Editor. The supported runtime remains in `core/`; nothing here
is part of its install or release path unless a later, explicit promotion
brings it into production.

## Start here

Start with each workspace README for its current status, boundaries, and
commands. Component READMEs describe their own APIs and validation. Slice
handoffs record implementation details and evidence when they add context that
does not belong in a component guide.

The Rust real-model Choice implementation, corrected-boundary results, and
historical runs are in [REAL_MODEL_CHOICE.md](rust-port/REAL_MODEL_CHOICE.md);
the original probe scope is archived in
[REAL_MODEL_CHOICE_BRIEF.md](rust-port/REAL_MODEL_CHOICE_BRIEF.md). The
completed direct/grouped bias slice is in
[BIAS_SLICE.md](rust-port/BIAS_SLICE.md), and the Stage 2 policy-calculation
completion record is in
[POLICY_CALCULATIONS_SLICE.md](rust-port/POLICY_CALCULATIONS_SLICE.md).
Stage 2 is complete; the next migration stage is described in
[ROADMAP.md](rust-port/ROADMAP.md).

## Inventory

| Path | Purpose |
| --- | --- |
| [`rust-port/`](rust-port/README.md) | Rust kernels, Python parity adapters, and a standalone terminal process. Its workspace README owns current status and roadmap links. |

Keep each experiment's implementation, fixtures, tests, build steps, and
evidence beside its own files.

## Working rules

- Treat the production Python behavior and its contract tests as the oracle
  until an experimental slice has passed its stated parity gate.
- Keep experiments out of production imports, dependencies, CLI behavior, and
  packaging. Promotion needs a separate review of compatibility, packaging,
  maintenance, and measured performance where performance is claimed.
- Store generated build products outside version control. Rust `target/`
  directories and Python bytecode are ignored by the experiment workspace.
- Check in readable fixtures and authored expectations. Do not derive expected
  values or terminal frames from the implementation being evaluated.
- Report commands actually run and distinguish fixture parity, process/PTY
  evidence, backend validation, and performance measurements.

## Commands and evidence

Use the commands in the relevant workspace or component README. The shared
Rust workspace commands live in [`rust-port/README.md`](rust-port/README.md);
Python extension builds and PTY journeys remain component-specific.

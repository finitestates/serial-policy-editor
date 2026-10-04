# Argmax harness migration and validation

## Checkpoints

- `7cc93c3`: implementation/documentation checkpoint; no runtime tests at that cut.
- `f1b9478`: separately committed same-depth beam pruning/backfill. This amends
  the initial beam checkpoint to retain dense observations only for current-depth
  refill, rather than for every rewind checkpoint.
- Harness migration and this validation record are subsequent uncommitted work.

## What changed

Ported the independent standard-library reference kernel to score eligibility,
logit-gap min-p, plain/perturbed argmax and selective noise. Removed its token
CDF and retired filters; preserved independent noise/address formulas. Oracle
menus rank scores directly and expose eligible-softmax only when requested.
The boundary demonstration now uses per-token Gumbel coordinates.

Migrated shared fixtures, sampler records, optional evidence, CFG diagnostics,
CLI/teacher-plan examples, projector, menu/request assertions and vector callers.
Replaced column cycling and stochastic-beam expectations. Removed neighbor-margin
and vocabulary-z-score tests because their features are gone. Generated sampler
records now cover all retained kernels and independent eligibility/noise limits.
No compatibility shim or new broad skip was introduced to conceal failures.

Added negative controls for unrequested softmax, underflow versus membership,
missing versus zero diagnostics, schema-3 nullable evidence, byte-preserving
old-schema rejection, projector reconstruction without persistence, direct overlay
state and one-time launch normalization. Documented profiles are parsed by tests.
These exposed a real profile-loader bug: explicit `none` limits were rejected
after successful CLI parsing. The loader now accepts that explicit spelling for
eligible, selective-noise and Gumbel menu limits.

Beam checks cover same-depth width, stable IDs/cursor, repeated pruning beyond
the original candidate pool, exhaustion, protected lineages, advance/rewind,
cached score reuse and bounded rewind cache retention. Fake batched and serial
frontiers agree, including CFG; selected batched paths can be promoted. Fake
backend agreement does not certify real adapter cache behavior.

The runtime PTY journey now enables l/L/~/%, clears with C, kills a live branch
with Backspace and checks same depth, row count and survivor IDs before advancing.
Existing raw-byte pyte frame comparisons, synchronized-frame checks and negative
controls remain in place. Default render fixtures omit unrequested probabilities.

## Commands and results

Executed in the isolated `/tmp/serial-policy-editor/.venv` environment with
`PYTHONPATH=core/src`. Installed pyte 0.8.2 there to enable the PTY module; the
initial headless run's pyte skip is not counted as PTY evidence.

```sh
PYTHONPATH=core/src pytest -q tests reference-kernel --disable-warnings --tb=short -rs
```

Final result: **904 passed, 2 skipped in 28.76 seconds**. Skips:

- local llama oracle: `SPE_ORACLE_LLAMA_MODEL` is unset;
- Transformers oracle: Torch is not installed in this test environment.

Focused terminal validation:

```sh
PYTHONPATH=core/src pytest -q tests/core/test_live_terminal.py \
  tests/core/test_live_terminal_fuzz.py tests/core/test_live_terminal_pty.py \
  tests/core/test_search_warm_terminal.py tests/core/test_selection_warm_terminal.py \
  tests/core/test_terminal_architecture.py --disable-warnings --tb=short -rs
```

Result before the additional runtime backfill journey assertion: **193 passed**.
The extended runtime journey passed separately and in the final full suite.

Rendered and inspected default choice/beam PNG fixtures:

```sh
PYTHONPATH=core/src python core/scripts/render_live_screens.py \
  --output /tmp/argmax-screens-final --theme chill --size 80x24 120x40 \
  --view choice beam --overlay help --png
```

Inspected the 80x24 choice and beam PNGs. This is fixture rendering review, not
physical-monitor observation. The standalone coordinate demonstration also ran.
Source syntax parsing and `git diff --check` are supplementary static checks.

## Evidence limits

No real-model parity, actual GPU/KV-cache adapter validation, cross-platform CI,
physical-display assessment or performance benchmark was run. The retained local
Python harnesses are migrated; research experiments under `experiments/` were
not ported or certified. Package version metadata is unchanged. This is an
experimental branch, not a new validated release.

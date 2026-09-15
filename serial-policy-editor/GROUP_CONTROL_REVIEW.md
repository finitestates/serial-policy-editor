# Group-control implementation review

Implemented from `70363ab` on `codex/group-control` in an independent development
checkout. The original checkout remains untouched. Package version: `0.4.0.dev0`.

## Result

Three independent inputs now have separate responsibilities:

1. **Reference YAML** supplies relative lexical weights. `--reference` works
   without a catalog, group, or learner. One optional strength controls its
   influence; the normal interface has no scope or ballistic modes.
2. **Group objectives** ask for more, less, or approximately the current rate of
   completed appearances. Bare `b target +`, `-`, and `=` activate bounded
   feedback during autonomous generation. Explicit numeric amounts remain
   manual; `off` clears the selected activation.
3. **Teacher preference learning** continues to consume teacher selections and,
   optionally, typed writes. Autonomous group feedback supplies no teacher labels.

The group monitor works on rendered surface matches, including phrases and
alternate tokenizations. Its intervention uses canonical token routes. Optional
compiler exploration retains other decompositions as inspection data; newly
compiled exploratory routes do not enter runtime steering.

## Main code boundaries

- `lexical_reference.py`: validates and compiles standalone lexical weights.
- `group_control.py`: serializable objectives, appearance measurements, scope
  reconstruction, bounded entry/continuation adjustments, and diagnostics.
- `bias_commands.py`: shared command target resolution and steering mutations.
- `bias_catalog.py`: canonical runtime routes plus optional exploratory routes.
- `sampling.py`: composes manual adjustments, lexical references, latent
  preferences, and appearance control before decoder filtering.
- `bias_presets.py`: complete `biases.json` import/export, including references,
  objectives, vectors, and relevant metadata.
- Episode lifecycle and HTTP/terminal adapters preserve the same policy state.

The controller derives its response from the saved objective and token history.
Observations do not mutate learning state. Episode snapshots keep the original
history origin; portable presets start measuring in the destination episode.
Explicit replay imports apply across all saved segments. Model changes clear
steering that belongs to the former tokenizer.

## Teacher learner corrections

The manual group fitter now exposes severity cap, dead zone, unattenuated
severity, rejection strength, and decay controls. Normal fixed group features
use analytical sparse gradients; nonlinear legacy combinations retain a
finite-difference fallback. Disabled, frozen, appearance-controlled, and
out-of-bounds manual groups remain unchanged.

Both teacher learners use the refreshed observation after an interactive policy
edit. Typed evidence is summed before applying a single clip and decay step.
Latent slow/fast vector serialization remains supported; minimal vector-array
imports are also accepted with model and projection metadata.

## Interface changes to review

- Semantic `b target =` now means **maintain**, and `b target off` clears it.
  Rank and last-token commands retain their old clear/default-step meanings.
- Canonical tokenization is the compiler default. Use `--explore` for alternate
  decompositions. Recompile an old catalog to obtain the new separation.
- Full presets use `spe-bias-rules-v3`; v2 remains readable. Enabled adaptive
  objectives cannot be flattened with `--rules-only`.
- New command-created groups are excluded from the old teacher group fitter.
  A manual JSON group can explicitly opt in with `learnable: true`.

See [STEERING.md](STEERING.md) for commands, YAML examples, and preset semantics.

## Validation

Final full test run: **889 passed, 15 skipped**. This included the local HTTP
integration test. The new controller tests cover:

- Directional changes in completed single-token and phrase appearances during
  deterministic autonomous generation, plus bounded feedback and maintain mode.
- Phrase recognition across alternate tokenization, word boundaries, and a
  phrase started at the prompt boundary.
- Scope/stop isolation, durable off, and explicit manual overrides.
- Canonical compilation without vocabulary enumeration and runtime isolation
  from exploratory routes.
- Standalone reference independence, scale invariance, and invalid input.
- Complete preset round trips, minimal learner arrays, CLI YAML loading,
  export/import, explicit replay overrides, and model-change resets.
- Identical reconstructed policies through rewind, replay, fork, and reopen.
- Teacher gradient checks, post-edit observations, excluded weights, and
  typed-evidence aggregation.

`git diff --check` passed. A wheel built successfully with all three new modules;
its CLI reports `0.4.0.dev0`.

## Practical limits

The generation checks use deterministic test backends. Real-model quality and
latency have not been benchmarked; a model backend is not installed in the test
environment. This is a development implementation ready for model trials.

The initial appearance baseline is an estimate, especially for phrases. Control
is bounded and uses no lookahead, so shared token fragments, model context, and
decoder filtering can prevent reaching a requested rate. The objective is soft;
`b` exposes observed rates and current pressure so the result can be inspected.

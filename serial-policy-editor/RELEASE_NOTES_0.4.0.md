# Serial Policy Editor 0.4.0

Released September 15, 2026.

0.4.0 brings the recent bias, reference, and preference-learning work together.
Lexical references describe relative term weights, group objectives ask for
changes in completed appearances, and teacher preferences learn from deliberate
selections. Each can be used independently.

## Adaptive group objectives

Define groups directly with `--groups groups.yaml` or at the teacher prompt:

```text
b atmosphere -> {shadow, silhouette, gathering storm}
b atmosphere +
b atmosphere -
b atmosphere =
b atmosphere off
b
```

Bare `+` promotes appearances, `-` suppresses them, and `=` captures and maintains
approximately the current rate. The controller adjusts bounded bias during
ordinary generation, including holds, without requiring teacher selections or
`--online-learning`. `--group-level` sets the requested change in baseline odds.
`b` or `groups` shows observed/target rates, scopes, and current pressure.

The monitor counts completed words and phrases in rendered text, even when they
arrive through a different tokenization. Canonical paths provide entry and
continuation support. Conditional `after … until …` activations retain separate
baselines and lifetimes. Phrase steering remains soft and cannot guarantee a
particular rate against the model's probabilities or decoder filtering.

Explicit numeric amounts such as `b atmosphere +0.5` remain manual and additive.
For semantic targets, use `off` to clear an activation: **`=` now means maintain**.
Rank and last-token commands retain their previous clear/default-step behavior.

## Standalone reference weights

```yaml
shadow: 10
silhouette: 3
outline: 1
```

```bash
policy-editor --model model.gguf --reference reference.yaml \
  --new-prompt 'Tell a story'
```

No group, catalog, or learner is needed. Weights express relative lexical
importance; scaling all weights equally leaves the result unchanged. One
optional `--reference-strength` controls overall influence. The ordinary
interface needs no active/global/ballistic mode selection. Unlisted terms remain
available, and contributions are bounded.

## Runtime routes and compiler exploration

Canonical case, spacing, plural, and explicit surface variants now form runtime
routes. The optional compiler still exposes alternate decompositions with
`policy-editor-bias --explore`; new catalogs store these separately from runtime
routes. Ordinary compilation avoids exploratory vocabulary enumeration.
Command and YAML group construction share the same semantic model.

Recompile older catalogs to obtain the new runtime/exploration separation.
See the archived [YAML reference](archive/research/docs/BIAS_CATALOG_YAML.md)
for the historical options.

## Preference learning and diagnostics

This release includes the cumulative preference-learning improvements:

- Optional slow/fast memory with independent learning rates, forgetting,
  strengths, and limits.
- Configurable severity cap, dead zone, rejection pressure, and optional full
  severity outside the dead zone.
- Persisted projection seeds and explicit random-seed selection.
- Chunked feature construction to reduce temporary memory usage.
- Policy diagnostics and refreshed pre-action observations after steering edits.
- Typed evidence summed before one clipping/decay step for the atomic write.

The older manual group fitter also gains severity, dead-zone, rejection, and
decay controls. Fixed group features use analytical sparse gradients. Frozen,
disabled, appearance-controlled, and out-of-bounds manual groups stay unchanged.
New command-created groups use appearance objectives or manual amounts; they do
not learn from teacher selections unless a manual preset explicitly opts in
with `learnable: true`.

## Presets, replay, and HTTP support

The familiar `biases.json` workflow remains:

```bash
policy-editor --workspace episodes.sqlite3 --project '#1' --biases-only > biases.json
policy-editor --model model.gguf --new-prompt 'Another story' --biases biases.json
```

The `spe-bias-rules-v3` format carries manual rules, groups, objectives,
references, and slow/fast learner vectors with their metadata. V2 presets remain
readable, and minimal preference-vector arrays can be loaded with model and
projection metadata. Membership-only YAML export remains available.
`--rules-only` refuses to flatten enabled adaptive objectives.

Episode snapshots reconstruct control across rewind, resume, fork, and replay.
Portable presets start monitoring in the destination episode. Explicit preset
and reference imports override their corresponding fields throughout replay;
unspecified settings still follow source segments. Changing models clears
steering tied to the former tokenizer.

The local HTTP server accepts reference/group YAML and presets at launch and
shares terminal steering commands through `POST /api/session/steering`.
Observation responses exposed group-control diagnostics in the archived HTTP
preview; see [the archived guide](archive/research/docs/archive/HEADLESS.md).

## Validation and documentation

- Final full regression suite: **889 passed, 15 skipped**, including the local HTTP test.
  An earlier run encountered an intermittent terminal resize/input assertion;
  that test passed in isolation and on the full rerun.
- Version metadata and built package report **0.4.0**.
- The implementation was also tested successfully by the user. Controlled
  real-model quality and latency benchmarks have not been run for this redesign.

See the archived [Steering guide](archive/research/docs/STEERING.md) for the
historical command and preset guide,
[the user guide](README.md) for learner flags and editor workflows, and
[the implementation review](archive/research/docs/GROUP_CONTROL_REVIEW.md) for
historical design and test details.

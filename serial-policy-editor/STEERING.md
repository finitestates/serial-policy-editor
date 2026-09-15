# Lexical references, group objectives, and teacher preferences

These are three independent ways to steer an episode:

| Input | Meaning |
| --- | --- |
| Reference YAML | Relative importance of terms in your lexical universe |
| Group objective | More, less, or approximately the present rate of a group |
| Teacher selections/writes | Evidence for the preference learner |

## Reference weights: load and use

```yaml
shadow: 10
silhouette: 3
outline: 1
```

```bash
policy-editor --model model.gguf --new-prompt 'Tell a story' --reference reference.yaml
```

No catalog, group, or learner is required. The same input is supported by
`policy-editor-server`. Lists of terms are accepted as equal weights. Weights
must be finite and positive; multiplying every weight by the same constant
does not change the result. A single-token lexicon with equal weights has no
relative preference to express at its root.

SPE compiles ordinary tokenizations of exact and leading-space variants. It
applies relative branch preferences and modest continuation support inside a
lexical prefix. Contributions are bounded, and unlisted tokens remain available.
The optional `--reference-strength` scales the influence (default `0.25`).
These are relative lexical weights, not final output probabilities.

Reference behavior does not depend on whether a group is enabled or positive
or negative. The former active/global/ballistic launch modes are hidden legacy
compatibility options; ordinary use needs none of them. A reference supplied
directly on the command line takes precedence over the reference in a preset.

## Define groups with YAML or commands

```yaml
groups:
  atmosphere:
    - shadow
    - silhouette
    - gathering storm
```

```bash
policy-editor --model model.gguf --new-prompt 'Tell a story' --groups groups.yaml
```

Or define a group at the teacher prompt:

```text
b atmosphere -> {shadow, silhouette, gathering storm}
```

Both paths use canonical tokenizations of case, spacing, and plural variants.
YAML supports explicit terms/forms and options; see [the YAML reference](BIAS_CATALOG_YAML.md).
Quoted command targets are exact. Bare names resolve runtime groups, then
catalog entries, then ordinary text. `@name` requires a catalog entry.

Definitions alone do not activate a steering objective. Nested runtime group
membership is flattened when assembled; it is a snapshot of those members.
Appending members retains an existing group's objective and baseline. Use `=`
to recapture the desired rate after changing the definition.

## Activate an appearance objective

```text
b atmosphere +
b atmosphere -
b atmosphere =
b atmosphere off
b
```

- `+`: promote the group relative to its captured baseline.
- `-`: suppress it relative to that baseline.
- `=`: capture and maintain approximately the current appearance rate.
- `off`: remove that activation and its manual amount.
- `b` (or `groups`): inspect manual amounts, observed/target rates, current
  controller pressure, scopes, and the loaded reference.

Bare text works too: `b gathering storm +` creates a semantic target without a
separate group-definition step. Braces apply an operation to several targets.

Objectives run during ordinary generation, including `h N`, without
`--online-learning` or teacher selections. The default level requests twice
or half the baseline odds; `--group-level` changes this level for subsequent
activations. This is a desired appearance rate, not a logit amount.

### How control works

The monitor counts completed surface matches, including full phrases, over a
recent window. Case/spacing variants are alternative matches; they do not
multiply an occurrence. Where surfaces are available, monitoring uses rendered
text and can recognize an occurrence produced by an alternate tokenization.
It checks word boundaries so `shadowing` does not count as the exact surface
`shadow`. Inflections explicitly included in a group still count.

The baseline combines observed appearances with a small estimate from the
current reference- and preference-adjusted policy. Before text has accumulated,
that estimate is necessarily approximate, especially for multi-token phrases.
Generated text supplies measurements, not new teacher labels.

The controller combines the requested direction with a smoothed appearance
deficit. It applies bounded entry/continuation adjustments, relaxing promotion
as appearances accumulate and strengthening suppression when needed. Promote
and suppress contributions do not change sign; maintain can correct in either
direction. Overlapping objectives share a total intervention bound.

Phrase steering is soft: canonical paths get gentler entry pressure and full
continuation pressure. Recognized continuations take precedence over unrelated
fresh starts within the group. There is no forced decoding or model lookahead.
The model's probabilities, decoder filtering, and shared fragments can prevent
an objective from reaching its target. Inspect observed rates rather than
assuming that a requested target is guaranteed.

### Scopes

```text
b atmosphere + after thunder until "."
b atmosphere = after {thunder, rain} until "\n"
b atmosphere off after thunder until "."
```

Any complete trigger opens the scope; repeated triggers do not multiply it.
The stop text must tokenize to one token. Use `until #N` for a token ID, or the
legacy `until .` / `until |` sentence/newline boundaries. Omitted stop text
defaults to the exact period token.

A scoped objective has its own identity and baseline and never creates an
unconditional learned bias. Off and numeric overrides affect the selected
scope. Other scoped or unscoped activations remain independent.

## Manual bias commands remain available

```text
b atmosphere +0.5
b " gathering storm" -2
12+0.5
12=
bl 3 -
```

An explicit amount on a `b` target selects manual biasing and removes the
adaptive objective for that scope. Numeric adjustments remain additive.
Rank and `bl` commands retain their former default-step behavior and `=` clear
operation. For semantic `b` targets, `=` now means maintain; use `off` to clear.
Ordinary multi-token targets use canonical path support. Exact rank-prefix and
`bl` rules retain their tail matching behavior.

## Teacher preference learning

`--latent-preference` still learns its vector from live raw-rank selections;
`--learn-from-write` additionally learns from typed spans. Slow/fast vectors,
projection seed, and strengths remain ordinary serializable learner weights.
Group objectives do not use rank severity or interpret autonomous samples as
preferences.

The `--online-learning` fitter remains available for manual groups. Launch with
it enabled, then opt in at the teacher prompt:

```bash
policy-editor --model model.gguf --groups groups.yaml --online-learning
```

```text
b atmosphere learn on
b
b atmosphere learn off
```

`learn on` enables the group and permits teacher fitting, resolving its definition
directly from loaded group YAML/catalogs if needed. `learn off` freezes its current
amount without removing its steering effect. New groups default to learning off;
membership and numeric edits preserve an existing group's learning choice.
`b` shows each group's `learnable` and enabled state. `--learnable-groups`, if
supplied, still restricts which opted-in groups can learn; it does not opt them in.

Appearance-controlled groups are always excluded from this fitter. `learn on`
rejects groups with an appearance objective; use `b atmosphere off` first (and
the matching `after … until …` for each scoped objective). Activating an appearance
objective or clearing an unscoped target switches its learning off. Group learning
is group-wide, so the toggle itself does not accept an `after` scope.

The choice persists in episodes and bias presets as `learnable: true` / `false`.
Older preset records omitting that field default to true for compatibility.
The toggle does not enable the session's learner: launch with `--online-learning`.
Disabled and frozen groups remain unchanged by the fitter.

The manual fitter supports `--learning-severity-cap`,
`--learning-dead-zone-rank`, `--learning-no-severity-attenuation`,
`--learning-rejection-strength`, and `--learning-decay`, alongside its existing
rate, step, bounds, and group-selection controls. Normal fixed group features
use a sparse analytical gradient. Both teacher learners sum typed-span evidence,
then clip and decay once. Interactive steering edits refresh the authoritative
precommit observation before either learner runs.

## Export and import `biases.json`

```bash
policy-editor --workspace episodes.sqlite3 --project '#1' --biases-only > biases.json
policy-editor --model model.gguf --new-prompt 'Another story' --biases biases.json
```

The `spe-bias-rules-v3` preset contains model identity, manual rules, group
definitions, group objectives, reference routes/settings, and latent vectors
with their metadata. No source YAML is required to reload it. Old v2 presets
remain readable. Model-specific token IDs must address the same tokenizer.

The standard weight arrays remain `latent_preference_z` and optional
`latent_preference_fast_z`, with `latent_projection_seed`, `latent_strength`,
and `latent_fast_strength`. For a minimal preference-only input, the loader also
accepts `format: spe-preference-weights-v1`, model metadata, and a `weights` array
plus the same seed/strength metadata. Empty arrays represent empty memory.

Full episode snapshots retain the original history origin, so the same prefix
reconstructs identical control during rewind, fork, resume, and replay. Portable
presets retain the objective and baseline but start monitoring generation in
the destination episode; the old episode's prompt/history is not imported.
Teacher learning is not rerun during replay. Explicit `--biases` and
`--reference` inputs override their corresponding fields throughout a replay;
other sampler settings continue to follow the saved segments. `--fixed-config`
keeps one complete policy for the entire replay. Changing models clears
token-based steering; load YAML or a preset for the new tokenizer.

`--editor-friendly` exports group membership YAML only. `--rules-only` exports
manual rules only and refuses to flatten enabled adaptive objectives. Use the
normal full preset to preserve active steering and learner weights.

The HTTP API accepts the same commands at `POST /api/session/steering` with a
`command` string and the normal revision/request ID fields. Observation responses
include group-control diagnostics. Existing action and settings endpoints remain
unchanged.

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
use a sparse analytical gradient. By default, both teacher learners sum typed-span evidence,
then clip and decay once. Interactive steering edits refresh the authoritative
precommit observation before either learner runs.

### Adaptive Dead Zones: experimental sampler eligibility gate

Use `--latent-learning-gate sampler` for the latent learner and
`--learning-gate sampler` for the manual group fitter. Both default to `rank`,
which preserves the existing rank dead zone and severity behavior.

```bash
policy-editor --model model.gguf --latent-preference \
  --latent-learning-gate sampler --latent-rejection-strength 1
```

Sampler mode replaces rank severity with a binary decision at each precommit
observation: a chosen token already surviving the actual decoder filters supplies
zero new evidence; an excluded token supplies full-severity evidence. This includes
temperature, top-k, top-p, min-p, and all current steering. Membership in the final
surviving set is authoritative, even if numerical underflow makes a surviving
token's probability zero. With greedy temperature zero, only the top token survives.

In sampler mode, the rank dead zone, severity cap, and no-attenuation flags have
no effect. Learning rate, rejection strength, clipping, group eligibility, and
memory limits retain their existing meaning. Gradients still use the untruncated
policy, so filtered-out teacher choices can teach useful corrections.

**Decay timing is separate:** by default, an already-eligible choice supplies no
evidence but configured decay still applies. Use the conditional decay experiment
below to change that.
For typed writes, each token is evaluated after the preceding written tokens;
evidence is reduced (summed by default), with clipping and any decay applied once
for the complete write.
Readouts identify eligibility skips and report how many written tokens were excluded.
Learning records include the gate, eligibility, and pre-update sampling probability.

This experiment changes which choices teach, using the existing update rule. It
does not guarantee admission in one step or calibrate an update to the exact
truncation boundary. With every token eligible, sampler mode contributes no new
evidence. Save the launch flags alongside exported weights; the gate is a learner
setting and must be supplied again when continuing learning in a later session.

### Reading learning feedback

The one-line notice explains whether the latest teaching choice supplied evidence,
was skipped by the gate, changed memory through decay, or left memory unchanged.
Type **`learning`** at a token choice or the live edge to open a scrollable report;
Enter/Esc returns to the same choice. Opening the report generates no tokens and
changes no weights.

The report includes:

- Chosen/proposed token text and IDs, pre-update policy rank, and separately labeled
  policy and sampler probabilities. Sampler probability is after decoder filtering.
- The gate's reason for admitting or skipping evidence and the rejection settings.
- New learning before/after step clipping, decay, memory-bound adjustments, and net
  movement. Slow and fast channels appear separately.
- Memory size (`z norm`) before/after the update and the channel's maximum possible
  token logit adjustment at the active strength. These are magnitudes, not confidence
  or counts of remembered preferences; vector movements do not simply add.
- Per-group changes and reasons frozen or controlled groups were skipped.
- For Writes, each learner's evidence/skip counts, proposal agreement, individual
  token ranks/gate decisions, reduction scale, and once-per-write decay.

Only the latest teaching event in this session is retained in this view. It is an
explicitly labeled historical report: Hold, replay, manual edits, and rewinds do
not refresh it. Switching episodes never shows another episode's report. The
persistent episode records remain available through the normal projection tools.

### Teacher-learning experiments

All three controls are opt-in and available for both learners. Use the `--latent-`
prefix for preference learning, or `--learning-` for fitting learnable manual groups.
They do not change appearance controllers. Existing defaults remain unchanged.

| Flag suffix | Choices (default first) | Experiment |
| --- | --- | --- |
| `decay-on` | `update`, `rejection`, `evidence` | When existing memory may decay |
| `write-reduction` | `sum`, `mean`, `sqrt` | How a typed span's admitted evidence is scaled |
| `rejection-target` | `proposal`, `sampler` | What a rejected proposal contrasts against |

**Conditional decay.** `--latent-decay-on rejection` skips decay whenever the
teacher's chosen token matches the captured proposal, including coincidental
agreement inside a write. A disagreement can still decay memory even if its token
is inside the dead zone. `--latent-decay-on evidence` requires positive gate severity:
with Adaptive Dead Zones, only an excluded choice permits decay. With rank gating,
only a choice beyond the rank dead zone permits it. Evidence here means admission
by the gate, even if a zero learning rate, clipping, or cancellation of gradients
prevents an actual learning step. Slow and fast memory share the trigger and retain
their separate decay rates. During a write, any qualifying token enables one decay
of the whole memory; the write never decays once per token. These settings have no
effect when decay rates are zero. Disabled learners and frozen groups remain frozen.

**Write evidence sizing.** With `--learn-from-write`, `sum` retains the existing
sum of token evidence. `mean` divides that sum by N; `sqrt` divides it by sqrt(N),
where N is the number of tokens with positive gate severity. Gate-skipped tokens
do not dilute the correction. With no admitted tokens the scale is 1 and evidence
is zero. Reduction happens before step clipping, for both slow and fast channels;
decay and memory bounds are applied once afterward. `mean` should reduce sensitivity
to paste length; `sqrt` leaves longer corrections more influence than shorter ones,
but less than `sum`. Opposing token signals can still cancel. This is span sizing,
not phrase recognition or tokenizer-independent group emission. Direct selections
are unaffected by `write-reduction`.

**Sampler rejection target.** With nonzero rejection strength, `sampler` replaces
the sampled proposal's negative features with the probability-weighted features of
the actual surviving sampler candidates, frozen at the precommit observation. This
includes temperature, truncation, and steering. At rejection strength 1, the latent
direction for a disagreement is `features(chosen) - E_sampler[features]`. The manual
group fitter uses the corresponding group-feature difference. This should reduce
dependence on which one of several plausible tokens happened to be sampled.
The chosen-vs-untruncated-policy term and gate remain unchanged; coincidental
acceptance retains the existing acceptance rule. At rejection strength 0 the target
flag has no effect. This does not guarantee admission after one update.

Records include the selected controls, effective decay rates, and the aggregate
write's evidence count and scale. Nondefault controls appear in learning readouts.
Like the learning gate, these are launch settings: save the flags with your weights
and supply them again when continuing learning. Exported bias presets preserve
weights, not these learner settings. See [trials 14–16](CONFIG_TRIALS.md#14--protect-memory-on-acceptance-or-on-all-gate-skips)
for paired commands that change one experiment at a time.

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

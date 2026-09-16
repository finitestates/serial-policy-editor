# Bias catalog YAML reference

Normal compilation produces canonical token routes for useful surface variants.
Alternate decomposition is an optional inspection feature and does not expand
runtime routes. Group YAML can be loaded directly into the editor:

```bash
policy-editor --model model.gguf --new-prompt 'Tell a story' --groups groups.yaml
```

## Smallest inputs

A list (or a `terms:` list) creates an automatic `global` group:

```yaml
- shadow
- silhouette
- gathering storm
```

Named groups:

```yaml
groups:
  atmosphere:
    - shadow
    - silhouette
    - gathering storm
```

A top-level mapping of names to lists is also accepted as group shorthand.
Definitions are inactive until selected with a command such as
`b atmosphere +`. See [Steering](STEERING.md) for objectives, manual overrides,
and scopes.

## Terms, forms, and groups

```yaml
defaults:
  cases: [original, lower, title, sentence]
  leading_space: true
  plural: true
  route_policy: canonical

terms:
  shadow: {}
  port_of_call:
    text: port of call
    forms: [Port of Call]
    plural: false

groups:
  nautical:
    members: [port_of_call, anchor, steamship]
  combined:
    members: [nautical, shadow]

term_options:
  shadow:
    plural: false
```

- `terms` accepts a list or mapping. A mapping key names the entry; `text`
  supplies its source spelling. A string mapping value is shorthand for `text`.
- `forms` supplies additional bases. These receive the same configured expansion
  as the primary source; they are not an exact allowlist.
- `groups` collects terms or other groups. An undeclared bare member becomes an
  implicit term. `@name` requires an already declared term or group. Cycles are
  rejected. Group names are identifier-like; term names may be phrases.
- `term_options` overrides defaults and inline term options.
- Top-level terms also belong to `global`. Terms introduced only inside another
  group do not automatically become global members. `global` is reserved as a
  group name, and an explicit `groups.global` can extend it.
- YAML lexical scalars are read as strings, so quoting a word does not change
  its meaning. Exact quoting in runtime commands is a separate interface.

The default case/spacing/plural expansion of `shadow` includes `shadow`,
` shadow`, `Shadow`, ` Shadow`, `shadows`, ` shadows`, `Shadows`, and ` Shadows`.
Plural and suffix generation is simple English spelling logic; disable it and
provide selected forms when inappropriate. Normal tokenization is authoritative;
heuristic decomposition never replaces its canonical route.

## Normal options

| Option | Meaning |
| --- | --- |
| `cases` | original, lower, title, sentence, upper; scalar or list |
| `leading_space` | Include leading-space variants (default true) |
| `plural` | Include simple plural variants (default true) |
| `suffixes` | Additional literal suffixes (default none) |
| `forms` | Additional source forms for a term |
| `mode` | auto/path/tail/beheaded; auto uses canonical path support |
| `head_scale` | Nonnegative manual first-edge scale (default 1) |
| `continuation_scale` | Nonnegative manual continuation scale (default 1) |
| `route_policy` | canonical by default; all/cohesive request exploratory routes |

Adaptive objectives use their own bounded entry/continuation controller;
manual scales and modes apply to numeric bias rules. Single-token path and tail
rules have the same one-edge effect.

## Optional compiled JSON

```bash
policy-editor-bias --model model.gguf --input groups.yaml --output catalog.json
policy-editor --model model.gguf --new-prompt 'Tell a story' --bias-catalog catalog.json
```

The JSON retains the `spe-bias-catalog-v1` envelope and now records compiler
version 2. Entries have `routes` for inspection and `runtime_routes` for canonical
runtime use. Without exploration these contain the same routes. Groups contain
their merged member routes. Model metadata validates vocabulary identity.
Existing catalog files lacking `runtime_routes` retain their historical route
set; recompile their YAML to adopt canonical runtime routing.

The complete bias/learner preset remains a different file:
`--project EPISODE --biases-only` writes a `spe-bias-rules-v4` preset, which is
loaded by `--biases`. It includes active amounts/objectives, references, and
preference vectors. Catalog JSON describes reusable definitions.

## Exploratory decomposition

```bash
policy-editor-bias --model model.gguf --input groups.yaml --explore \
  --level exhaustive --max-routes 64 --output exploration.json
```

`--explore` enables the former all-route search for inspection. YAML
`route_policy: all` or `cohesive` does likewise. Canonical runtime routes remain
separate even if an exploratory alternative ranks first.

Exploration-only options:

| Option | Meaning |
| --- | --- |
| `level` | minimal, standard, exhaustive search depth/selection |
| `max_routes` | Maximum retained exploratory routes per term across forms |
| `max_route_tokens` | Exploration depth; raised if needed to accommodate the canonical route |
| `min_route_piece_chars` | Cohesive-piece threshold (default 3) |
| `allocation` | legacy/full/equal/information/information_amplified/naive_chaining |
| `allocation_floor` | Floor for information allocation (default 0.05) |

Selecting a non-legacy allocation also requests exploration. `--diagnostics`
prints per-edge allocation statistics. These weights and route quality classes
are inspection data; they do not change canonical runtime routes. Retained-route
limits are output limits, not a guarantee of cheap exhaustive enumeration.
Normal canonical compilation performs no vocabulary-route search.

## Reference weights are independent

```yaml
shadow: 100
silhouette: 3
outline: 1
```

Load this directly with `policy-editor --reference reference.yaml`; it does not
need groups or the compiler. Lists supply equal weights. The ordinary reference
policy has one optional strength and no active/global/ballistic mode selection.

For historical allocation experiments, `policy-editor-bias --reference` still
accepts the same YAML as a compile-time lexical universe and embeds tokenized
reference routes in its catalog. Loading that catalog enables the ordinary
independent reference policy unless explicitly overridden. Compile-time prefix
statistics and runtime lexical scoring are separate consumers of those weights.

The reference YAML is not a model-weight file, group mapping, or an absolute
frequency promise. See [Steering](STEERING.md#reference-weights-load-and-use).

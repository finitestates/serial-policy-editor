# Bias catalog YAML reference

`policy-editor-bias` accepts a YAML source file containing human-readable terms
and optional named groups. The compiler loads the YAML with scalar values as
strings, so a term has the same meaning whether it is written unquoted,
single-quoted, or double-quoted. YAML quoting is not runtime bias syntax.

The output is a model-specific JSON catalog. Compile it against the same model
and tokenizer that will load it at runtime.

## Which structured file is which?

SPE uses several related but non-interchangeable files:

| File | Used by | Meaning |
| --- | --- | --- |
| Catalog YAML | `policy-editor-bias --input` | Human-readable terms, compiler options, and named groups. |
| Reference YAML | `policy-editor-bias --reference` | Optional lexical reference surfaces, either unweighted or surface-to-weighted. |
| Catalog JSON | `policy-editor --bias-catalog` | Generated model-specific token routes. It is not a runtime bias preset. |
| Runtime preset JSON | `policy-editor --biases` | Saved token rules, named runtime groups, model identity, and optional latent state. |
| Editor-friendly YAML | `--biases-only --editor-friendly` output | Recompilable `groups:` membership only; it does not contain active amounts or token routes. |

The compiler inputs are YAML because they describe semantic text. The compiler
outputs JSON because token IDs, routes, weights, and model metadata are exact
machine data. Do not copy token IDs from a catalog into a different model or
tokenizer. Do not pass a runtime preset to `--bias-catalog`, or a compiled
catalog to `--biases`.

The optional reference file is also separate from the catalog file. A reference
list gives every surface equal weight:

```yaml
- shadow
- shadowing
- silhouette
```

A reference mapping gives relative lexical weights:

```yaml
shadow: 100
shadowing: 2
silhouette: 1
```

Reference mappings use numeric values. They are not catalog group mappings and
cannot contain `members`, `defaults`, or term compiler options.

## Smallest valid inputs

A top-level YAML list is a list of terms. It becomes the automatic `global`
group:

```yaml
- sky
- cloud
- fog
- aardvark
```

An equivalent explicit form is:

```yaml
terms:
  - sky
  - cloud
  - fog
  - aardvark
```

If a mapping has no `terms`, `groups`, `defaults`, or `term_options` key, and
every value is a list, it is interpreted as a shorthand group mapping:

```yaml
nautical:
  - anchor
  - steamship
  - port of call
```

## Complete source shape

The expanded source shape is:

```yaml
defaults:
  level: standard
  mode: auto
  allocation: legacy
  allocation_floor: 0.05
  head_scale: 1.0
  continuation_scale: 1.0
  cases: [original, lower, title, sentence]
  leading_space: true
  plural: true
  suffixes:
    - ing
    - ed
  max_routes: 4096
  route_policy: all
  min_route_piece_chars: 3
  # Omit max_route_tokens to use the level-dependent default depth.

terms:
  - sky
  - shadow:
      text: shadow
      level: standard
      mode: auto
      head_scale: 1.0
      continuation_scale: 1.0
      cases: [original, lower, title, sentence]
      leading_space: true
      plural: true
      suffixes: [ing, ed, ly]
      max_routes: 32
      max_route_tokens: 3
      route_policy: cohesive
      min_route_piece_chars: 3
      forms:
        - shadow
        - shadowing

groups:
  nautical:
    members:
      - anchor
      - steamship
      - port of call
    level: exhaustive

term_options:
  aardvark:
    level: minimal
    cases: [original, lower]
    plural: false
```

The top-level keys are:

- `defaults`: compiler options inherited by every term.
- `terms`: a list or mapping of terms to compile.
- `groups`: a mapping of group names to member lists.
- `term_options`: optional per-term compiler options applied after `defaults`
  and after any inline term options.

The four keys have distinct jobs: `defaults` changes compilation defaults,
`terms` defines named surfaces, `groups` collects terms into reusable catalog
entries, and `term_options` overrides compilation for a named term. A group is
not a bias amount and does not become active until the editor applies a runtime
command such as `b nautical +1`.

`terms` may be written as a list or mapping. These forms are equivalent:

```yaml
terms:
  - velociraptor
  - shadow:
      level: exhaustive
      max_routes: 32
```

```yaml
terms:
  velociraptor: {}
  shadow:
    level: exhaustive
    max_routes: 32
```

An inline term mapping may provide `text` when the catalog name and tokenizer
source differ. A string value is shorthand for `text`:

```yaml
terms:
  display name: source text
  alias:
    text: source text
```

`forms` is an explicit list of source forms for that term. It is useful when a
term needs particular derivatives or spellings beyond the generated case,
spacing, plural, and suffix variants.

For a mapping-form term, the key is the catalog name and `text` is the source
text sent to the tokenizer. For example, this creates an entry named
`display_name` whose routes are compiled from `port of call`:

```yaml
terms:
  display_name:
    text: port of call
    mode: path
    forms:
      - port of call
      - Port of Call
```

## Compiler options

The recognized compiler options are:

| Option | Values | Meaning |
| --- | --- | --- |
| `level` | `minimal`, `standard`, `exhaustive` | Controls how deeply alternate token routes are searched. |
| `mode` | `auto`, `tail`, `path`, `beheaded` | `auto` uses beheaded path semantics for one- or two-letter route heads, then path for lexical terms and tail for phrases; the other values force the mode. |
| `allocation` | `legacy`, `full`, `equal`, `information`, `information_amplified`, `naive_chaining` | Selects how bias is distributed across route edges. `legacy` preserves the existing mode/head/continuation behavior; the other strategies store explicit per-edge weights. `information_amplified` preserves information shape but amplifies routes of length 3 or more. `naive_chaining` assigns `0`, `0.5`, `1.0`, `1.5`, ... to successive edges of multi-token routes. |
| `allocation_floor` | number from `0` to `1` | For `information`, reserves at least this fraction of a route's unit bias for every edge before renormalizing. The default `0.05` prevents a zero-weight head from dead-ending a route; set `0` for the raw information result. |
| `head_scale` | finite nonnegative number | Scales the first edge of a path rule. |
| `continuation_scale` | finite nonnegative number | Scales continuation edges after a matching path prefix. |
| `cases` | list of `original`, `lower`, `title`, `sentence`, `upper` | Case variants to search. A scalar is also accepted. `sentence` turns `my favorite chair` into `My favorite chair`. |
| `leading_space` | `true` or `false` | Include the leading-space surface variant. `both` acts like `true`; `none` acts like `false`. |
| `plural` | boolean | Search the standard pluralization variants. |
| `suffixes` | string or list of strings | Additional suffixes to search, such as `ing`, `ed`, or `ly`. |
| `max_routes` | positive integer | Maximum routes retained for each term across all generated forms. |
| `max_route_tokens` | positive integer | Maximum number of tokens in one route. Omit it for the normal level-dependent depth. |
| `route_policy` | `all`, `cohesive` | `all` makes every exact route eligible and ranks preferred routes first; `cohesive` removes fragmented routes and uses a tail-only fallback if none survive. |
| `min_route_piece_chars` | positive integer | In `cohesive` mode, minimum alphanumeric characters for a non-whole-word-like piece. |

Defaults are `standard`, `original/lower/title/sentence`, leading-space variants,
automatic mode, `legacy` allocation, an information floor of `0.05`, unit
head/continuation scales, pluralization enabled, no extra
suffixes, `max_routes: 4096`, `route_policy: all`,
`min_route_piece_chars: 3`, and the level-dependent default route depth. The
command-line options `--level`, `--allocation`, `--allocation-floor`,
`--max-routes`, `--max-route-tokens`, `--route-policy`, and
`--min-route-piece-chars`
override the YAML options for the whole compilation.

### Experimental edge allocation

The experimental allocation strategies are applied uniformly to every
accepted route; they do not privilege the tokenizer's default/canonical route.
`full` puts unit weight on every edge, `equal` divides one unit evenly across
the route, and `information` estimates prefix specificity from a compile-time
reference surface universe. `information_amplified` starts with the same
information shape, then multiplies each edge on routes of length 3 or more by
`1 + Phi(child)` without renormalizing; two-token routes are unchanged. The
`naive_chaining` experiment uses direct compositional chaining instead: a
multi-token route receives `0`, `0.5`, `1.0`, `1.5`, ... on successive edges,
while a single-token route remains at `1.0`. Its route is entered only after
the preceding token sequence matches, so a phrase such as `port of call` can
advance from `port` to `of` to `call` without applying later weights early.
default reference universe combines tokenizer vocabulary strings with
generated catalog forms. When `--reference` is supplied, it becomes the
reference universe instead; generated catalog forms are added only when
absent. A frequency-weighted YAML mapping such as
`{shadow: 100, shadowing: 2}` supplies relative lexical mass.

Information allocation is compiled into route `edge_weights`, so generation
only performs the normal prefix match and multiplies the selected weight by
the user's bias. Every complete route conserves one unit of bias after the
floor is applied. Route JSON includes `allocation_diagnostics` with each
edge's token, remaining mass, information, Phi, and final edge weight.

When `--reference` is supplied, the catalog also embeds model-tokenized
reference routes for an experimental online lexical prior. The episode editor
uses `active` scope by default, so it affects only routes represented by
currently active bias rules. `global` uses the complete reference universe.
The `--reference-prior` presets are:

* `active` / `global`: contrastive branch preference only.
* `active-exit` / `global-exit`: contrastive preference plus an
  EXIT-vs-CONTINUE gate.
* `ballistic-active` / `ballistic-global`: root entry attraction.
* `ballistic-active-exit` / `ballistic-global-exit`: root attraction plus the
  EXIT-vs-CONTINUE gate.

Use `off` to disable the prior. `--reference-prior-strength` controls relative
branch preference and defaults to `0.25`. A reference weight is a relative
lexical importance, not a literal final probability.
`--reference-prior-attraction` adds commitment pressure inside a lexical prefix
and defaults to `0`; `--reference-prior-exit-strength` defaults to `0.25` in
`*-exit` modes. The exit gate uses that value times the log ratio of
continuation mass to terminal mass and does not alter root attraction or
relative child branch scoring.

Every mode uses one history-reconstructed token trie. After a reference prefix
is entered, unrelated root routes do not restart their contribution at every
position; normal failure/suffix transitions return to them.
These are runtime presets rather than compiler YAML fields. The compiler YAML
controls terms, groups, forms, route policy, allocation, and per-term scales;
the separate `--reference` YAML supplies the weighted lexical universe.

For a phrase whose first token is common but whose continuation is distinctive,
force path mode and make the head gentler:

```yaml
terms:
  New York:
    mode: path
    head_scale: 0.25
    continuation_scale: 1.0
```

Scales affect path rules. Tail rules continue to apply their full bias only to
the matching completion edge. When alternate routes share a token at different
positions, the strongest applicable scale is used once for that logical rule.

`beheaded` is a path-like mode for routes whose first token is a bare boundary
(such as a whitespace-only token) or, after leading whitespace or a tokenizer
word-boundary marker, consists of one or two letters. The first edge receives
zero bias and continuation edges use `continuation_scale`. In `auto` mode this
behavior is selected automatically; routes without a qualifying head retain
the ordinary path-or-tail behavior.

Routes are classified as `direct`, `word_aligned`, `cohesive`, or
`fragmented`. Direct routes contain one token; word-aligned routes are
sequences of direct word pieces such as `port` + ` of` + ` call`; cohesive
routes use sizeable subword chunks; fragmented routes contain tiny internal
pieces. With `allocation: legacy`, preferred routes are selected first, then
remaining routes are chosen deterministically, round-robin across generated
forms, until `max_routes` is reached. Experimental allocations consider all
accepted routes under the same allocation strategy. The tokenizer's default
route is only one candidate and is not automatically privileged.

Set `route_policy: cohesive` to remove fragmented routes before selection.
Cohesive mode accepts pieces that align with whitespace/word boundaries or
contain at least `min_route_piece_chars` alphanumeric characters. Thus
`port` + ` of` + ` call` can remain visible, while decompositions such as
`o` + `f`, `m` + `y`, or `an` + `other` are omitted. If a term has no cohesive
route at all, the compiler retains its best exact fallback as a tail-only
route and emits a warning. The default policy remains `all`.

## Groups and references

Groups are mappings from an identifier-like name to either a member list or an
object with `members` and an optional `level`:

```yaml
groups:
  nautical:
    - anchor
    - ship
  dangerous_terms:
    members:
      - shadowing
      - velociraptor
    level: exhaustive
```

Group names must begin with a letter or underscore and may contain letters,
numbers, underscores, periods, and hyphens. Group members may refer to terms,
other groups, or new terms that are implicitly added to the catalog. A member
written as `@name` is an explicit reference and must already name a defined
term or group. Group cycles are rejected.

Terms listed under top-level `terms` are placed in the automatic `global`
group. Members introduced only through a named group stay in that named group
unless they are also listed under `terms`. `global` is reserved for the
automatic group and cannot be declared as an ordinary term. An explicit
`groups.global` definition is allowed and is combined with the top-level terms.

These two examples are equivalent ways to define a simple catalog group:

```yaml
groups:
  nautical:
    - anchor
    - steamship
    - port of call
```

```yaml
groups:
  nautical:
    members: [anchor, steamship, port of call]
```

The object form is required when the group itself needs a compiler override,
currently `level`:

```yaml
groups:
  nautical:
    level: exhaustive
    members:
      - anchor
      - steamship
      - port of call
```

Group members may be existing terms, other groups, or new implicit terms. An
unprefixed member such as `anchor` may be introduced automatically; `@anchor`
requires `anchor` to have been declared already as a term or group. Group
cycles are rejected. Catalog groups are compile-time collections; runtime
groups created with `b name -> {...}` are a separate sampler-state feature.

## Compile examples

Compile a YAML source against a GGUF model:

```bash
policy-editor-bias \
  --model /path/to/model.gguf \
  --input terms.yaml \
  --output catalog.json
```

Use a broader search for one term without changing the YAML file:

```bash
policy-editor-bias \
  --model /path/to/model.gguf \
  --input terms.yaml \
  --term velociraptor \
  --level exhaustive \
  --max-routes 64
```

The resulting catalog is loaded by the editor with:

```bash
policy-editor --model /path/to/model.gguf --bias-catalog catalog.json
```

The generated JSON has the following top-level shape:

```json
{
  "format": "spe-bias-catalog-v1",
  "model": {"backend": "llama.cpp", "vocabulary_size": 32000},
  "compiler": {"format_version": 1, "modes": ["auto", "tail", "path", "beheaded"]},
  "entries": {
    "nautical": {
      "kind": "group",
      "members": ["anchor", "steamship"],
      "routes": [
        {
          "token_ids": [101],
          "texts": [" anchor"],
          "token_texts": [" anchor"],
          "mode": "tail",
          "strategies": ["direct"],
          "route_class": "direct"
        }
      ]
    }
  }
}
```

The real `routes` arrays contain model token IDs, token text, route modes, and
optional edge weights. A reference-enabled catalog may also contain
`reference_prior_routes`. The catalog is generated data: edit the YAML and
recompile instead of hand-editing its route arrays.

At runtime, a bare name resolves to its catalog entry when present and falls
back to one-shot plain-text resolution otherwise. `@name` always requires a
catalog entry. Runtime group construction is documented in the main
[README](README.md#experimental-bias-catalog-compiler); YAML groups and runtime
groups are related concepts but are stored and edited separately.

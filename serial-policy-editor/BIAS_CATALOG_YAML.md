# Bias catalog YAML reference

`policy-editor-bias` accepts a YAML source file containing human-readable terms
and optional named groups. The compiler loads the YAML with scalar values as
strings, so a term has the same meaning whether it is written unquoted,
single-quoted, or double-quoted. YAML quoting is not runtime bias syntax.

The output is a model-specific JSON catalog. Compile it against the same model
and tokenizer that will load it at runtime.

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
  cases: [original, lower, title]
  leading_space: true
  plural: true
  suffixes:
    - ing
    - ed
  max_routes: 4096
  # Omit max_route_tokens to use the level-dependent default depth.

terms:
  - sky
  - shadow:
      text: shadow
      level: standard
      cases: [original, lower, title]
      leading_space: true
      plural: true
      suffixes: [ing, ed, ly]
      max_routes: 32
      max_route_tokens: 3
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

## Compiler options

The recognized compiler options are:

| Option | Values | Meaning |
| --- | --- | --- |
| `level` | `minimal`, `standard`, `exhaustive` | Controls how deeply alternate token routes are searched. |
| `cases` | list of `original`, `lower`, `title`, `upper` | Case variants to search. A scalar is also accepted. |
| `leading_space` | `true` or `false` | Include the leading-space surface variant. `both` acts like `true`; `none` acts like `false`. |
| `plural` | boolean | Search the standard pluralization variants. |
| `suffixes` | string or list of strings | Additional suffixes to search, such as `ing`, `ed`, or `ly`. |
| `max_routes` | positive integer | Maximum routes retained for each term across all generated forms. |
| `max_route_tokens` | positive integer | Maximum number of tokens in one route. Omit it for the normal level-dependent depth. |

Defaults are `standard`, `original/lower/title`, leading-space variants,
pluralization enabled, no extra suffixes, `max_routes: 4096`, and the
level-dependent default route depth. The command-line options `--level`,
`--max-routes`, and `--max-route-tokens` override the YAML options for the
whole compilation.

Canonical routes are retained first. Remaining routes are selected
deterministically, round-robin across generated forms, until `max_routes` is
reached. This keeps exhaustive terms bounded and reproducible.

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

At runtime, a bare name resolves to its catalog entry when present and falls
back to one-shot plain-text resolution otherwise. `@name` always requires a
catalog entry. Runtime group construction is documented in the main
[README](README.md#experimental-bias-catalog-compiler); YAML groups and runtime
groups are related concepts but are stored and edited separately.

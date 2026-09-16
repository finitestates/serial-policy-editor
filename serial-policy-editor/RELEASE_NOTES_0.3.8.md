# Serial Policy Editor 0.3.8

Released September 14, 2026.

0.3.8 extends SPE from model-aware bias editing into a small, persistent policy
layer. Human-readable YAML definitions compile against a local tokenizer into a
model-specific JSON catalog. The live editor can consume those catalogs,
maintain durable named groups, apply a stateful lexical reference prior, and
optionally learn from deliberate teacher choices or typed text.

The new learning features are opt-in. With no new flags, the ordinary editor,
replay, and sampler behavior remains the default path.

## Main additions

### Model-specific catalogs

`policy-editor-bias` compiles semantic terms into exact token routes using the
same model and tokenizer that will run the editor. Catalog compilation now
supports:

- direct, word-aligned, cohesive, and fragmented route classification;
- `tail`, `path`, `beheaded`, and automatic route modes;
- deterministic route budgets and cohesive filtering;
- legacy, equal, full, information, amplified-information, and naive-chaining
  edge allocation;
- case, leading-space, plural, suffix, explicit-form, and per-term options;
- named groups, nested group references, and weighted reference vocabularies.

The compiler's YAML input, its generated JSON catalog, the optional reference
YAML, and the editor's runtime JSON preset are different formats. The complete
schema and examples are in [BIAS_CATALOG_YAML.md](BIAS_CATALOG_YAML.md).

### Runtime rules and groups

All bias commands use one logical route matcher. Multi-token lexical targets can
advance edge by edge after each matching prefix; ordinary whitespace phrases
retain completion-only tail behavior. Conditional rules can use catalog terms,
groups, exact stop tokens, sentence/newline lifetimes, and trigger sets.

Runtime groups have one shared amount and a set of bias-free rule templates.
Their membership and amount are sampler state, so rewind, fork, resume, and
replay restore them at the relevant boundary.

### Reference priors

Compiling with `--reference reference.yaml` embeds weighted, model-tokenized
reference routes in the catalog. The editor can apply those routes through
`--reference-prior` in active or global scope, with contrastive or ballistic
entry behavior and optional EXIT-vs-CONTINUE gating. The prior reconstructs one
token-trie state from exact model-visible history, rather than independently
restarting every reference phrase.

### Opt-in learning

`--online-learning` adjusts the scalar strengths of existing named bias groups.
It is off by default, bounded by the configured bias limits, and learns from
live numeric raw-rank selections. `--learnable-groups` restricts which groups
may change.

`--token-preference` learns a bounded anonymous vector from fixed features
derived from the model output embedding. It generalizes across tokens in that
feature space, is model-specific, and is stored in sampler segments and full
JSON bias presets.

`--learn-from-write` allows either enabled learner to consume a live `Write`.
The editor observes each typed token before committing it, averages the
per-token updates, and installs the aggregate once the atomic write completes.
Replay, acceptance of the sampled proposal, EOG, and ordinary writes without
this flag do not update learned state.

## Structured files at a glance

SPE uses four intentionally separate structured artifacts:

| File | Produced/consumed by | Purpose |
| --- | --- | --- |
| `terms.yaml` | `policy-editor-bias --input` | Human-readable catalog terms, options, and groups. |
| `reference.yaml` | `policy-editor-bias --reference` | Optional lexical reference list or surface-to-weight mapping. |
| `catalog.json` | Compiler output and `--bias-catalog` | Model-specific compiled routes; do not hand-edit token IDs. |
| `biases.json` | `--biases-only` and `--biases` | Runtime logical rules, named groups, model identity, and optional token preference state. |

`--editor-friendly` emits a fifth, deliberately limited artifact: standalone
YAML `groups:` definitions for recompiling group membership. It does not carry
active bias amounts, compiled token routes, direct one-shot rules, or preference
state.

### Minimal catalog input

```yaml
terms:
  - sky
  - shadow
groups:
  nautical:
    members:
      - anchor
      - steamship
      - port of call
```

Top-level `terms` members enter the automatic `global` group. Members introduced
only through a named group remain in that named group. A group member may refer
to another term or group; prefix a member with `@` when it must already exist.
Group names are identifiers, not free-form display labels.

### Runtime preset shape

The loadable JSON preset uses `format: "spe-bias-rules-v2"` and has this shape:

```json
{
  "format": "spe-bias-rules-v2",
  "model": {
    "backend": "llama.cpp",
    "filename": "/path/to/model.gguf",
    "file_size_bytes": 123,
    "vocabulary_size": 32000
  },
  "bias_rules": [
    {
      "routes": [[101, 202]],
      "mode": "path",
      "bias": 0.5
    }
  ],
  "bias_groups": [
    {
      "name": "nautical",
      "bias": 0.75,
      "members": ["anchor", "steamship"],
      "rules": [
        {"routes": [[303]], "mode": "tail", "bias": 0}
      ]
    }
  ]
}
```

Routes and token IDs in this preset are model-specific. Loading checks the
vocabulary size and available model metadata. Group rules must have zero bias;
the group's single `bias` field is the shared runtime amount. `--rules-only`
keeps the same JSON format but flattens group rules and removes group metadata.

## Verification

- Feature-focused tests cover catalogs, route allocation, reference priors,
  online group learning, token preference learning, and typed-write learning.
- Full validation should include the ordinary pytest suite plus real-model smoke
  tests when the optional local backends and models are available.

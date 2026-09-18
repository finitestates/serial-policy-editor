# Bias system survey at `70363ab`

## Summary

The system has useful authoring, matching, and replay foundations. Its main architectural problem is that the compiler, runtime groups, scoped rules, reference prior, and learner have different notions of a target and its activation. Adding more allocation heuristics will not resolve that mismatch.

Recommended direction:

1. Preserve semantic term/group authoring and the `b … +/- … after … until …` interface.
2. Separate ordinary runtime tokenization from exploratory decomposition.
3. Give every learnable target an explicit activation, direction, scope, and strength policy.
4. Separate lexical reference weights from catalogs and group activation.
5. Share learning controls and evidence handling between the bias and latent learners.

This was an inspection, not an implementation change. HEAD matched `70363abf5f0bec6da20c25f6368dfc58568c4f20`; tracked source files had no changes. Existing untracked experiments and model artifacts were excluded from conclusions about this commit.

## 1. What exists today

| Layer | Current responsibility | Important consequence |
| --- | --- | --- |
| Catalog YAML / compiler | Generate forms, enumerate routes, classify pieces, allocate edge weights, flatten groups | Linguistic intent and experimental routing are bundled together |
| Catalog JSON | Model-specific routes, metadata, diagnostics, optional reference routes | All retained entry routes are available to runtime biasing |
| Runtime `BiasGroup` | Named, frozen rule templates plus one scalar `bias` | Definition, activation, direction, and strength are partly encoded in that scalar |
| Runtime `BiasRule` | Routes, mode, optional trigger/lifetime, amount | Standalone terms and CLI-created scoped targets bypass group learning |
| Bias learner | Finite differences over named group scalars | Learns from live selections and optional typed writes; no ongoing generation objective |
| Reference prior | Weighted token trie, relative branch scoring, optional attraction and exit gate | Runtime code is already largely independent, but loading and active scope couple it to catalogs/groups |
| Episode state | Full sampler snapshots at token boundaries | Strong foundation for reproducible replay; large static route data is repeatedly serialized |

Key sources: [catalog compiler](../src/trajectory_editor/bias_catalog.py:1350), [runtime types and matcher](../../core/src/trajectory_editor/bias_rules.py:65), [learner](../src/trajectory_editor/online_learning.py:118), [episode integration](../src/trajectory_editor/episode_policy.py:344).

### Foundations worth retaining

- YAML lists, explicit terms/forms, named groups, nested membership, and cycle rejection.
- Case and leading-space variants. The default form generator produces the eight requested `shadow` variants when the tokenizer supports them.
- Exact token-history matching for triggers and stop tokens, including replay reconstruction.
- Deduplication of overlapping edges within one logical group. Independent groups remain additive; that distinction should become documented policy.
- Fixed-logit observations, bounded learning controls, persisted strength changes, and no relearning during replay.
- The shared policy view introduced around this commit: raw and policy ranks, logit changes, and decoder probabilities already benefit both learners.
- Exploratory route inspection and allocation diagnostics, exposed as an analysis feature.

## 2. Confirmed correctness and behavioral gaps

### A. Scoped commands and learning can act on different policies

An unscoped catalog **group** becomes a runtime `BiasGroup`. A catalog **term** becomes ordinary rules. A scoped group command also becomes ordinary rules containing a snapshot of the group's routes. The learner only changes `sampling.bias_groups`.

This is more than missing coverage. A scoped command can leave a zero-strength unscoped group in the configuration. The learner can then activate that unscoped group while leaving the intended scoped amount unchanged.

Reproduction using the existing fake backend:

- `b concrete +1 after "P" until "!"` creates a scoped rule at `+1` and leaves the group's unscoped amount at zero.
- Learning a selection of its token changes the unscoped amount to approximately `+0.007104`.
- That learned amount applies even in a history where `P` has never occurred.

**Recommendation:** Represent a scoped activation as a first-class learnable object referencing the group definition. Choose explicitly whether learned magnitude is shared across activations or local to a scope; neither should leak an unconditional activation into existence.

Sources: [command resolution](../../core/src/trajectory_editor/episode_ui.py:670), [scoped group conversion](../../core/src/trajectory_editor/episode_ui.py:773), [learner target selection](../src/trajectory_editor/online_learning.py:196).

### B. Zero does not mean disabled, and sign does not mean intent

All named groups are learnable by default, including groups whose amount is zero. Their finite-difference probes temporarily assign positive and negative amounts. No field distinguishes disabled, neutral-but-learning, frozen, promote, or suppress.

Observed with default learner settings:

- A group at `0` became approximately `+0.007617` after a selection.
- A group at `+0.001` became approximately `-0.005710` when a competing token was selected.

These follow the current unconstrained preference-fitting objective. They conflict with interpreting `+` and `-` as durable user intentions.

**Recommendation:** Separate `enabled`, `learnable`, `direction`, and nonnegative `magnitude`. Keep unconstrained signed preference learning available as an explicit mode. Distinguish pause, reset, freeze, and remove in the interface.

Source: [group updates](../src/trajectory_editor/online_learning.py:196).

### C. Learning can use stale evidence after a bias edit

The runner captures an observation before entering the command UI. The UI refreshes its own observation after a bias command, but the runner later supplies the original observation to the learners.

Reproduction: promote `concrete` by `+4`, then select raw rank 3. Under the edited policy, that token is policy rank 1. The learner reports policy rank 3 and severity approximately `0.159`, and updates it anyway. For the bias learner, the gradient uses current settings while severity and baseline diagnostics use the old observation. The latent learner can also receive outdated probabilities and proposal evidence.

**Recommendation:** Capture the authoritative precommit observation after all menu edits, and give the same observation/configuration pair to both learners. The existing typed-write precommit observation mechanism is a useful model.

Sources: [UI refresh](../../core/src/trajectory_editor/episode_ui.py:903), [runner observation lifecycle](../src/trajectory_editor/episode_policy.py:538).

### D. Bounds can override both step limits and group exclusion

The selection learner clips the proposed delta, then clamps the resulting absolute bias. An existing amount outside the configured range can therefore move much more than `max_step`, even with zero gradient and zero severity.

Reproduction: an inactive group at `+8`, default maximum `+4`, and a rank-1 selection produce a `-4` change despite `max_step=0.25`.

The write aggregator has an additional defect: it clamps **every** group after averaging deltas, including groups excluded by `--learnable-groups`. A fixed `+8` group stayed unchanged in the per-token result but became `+4` in the aggregate write result.

**Recommendation:** Preserve frozen/excluded groups exactly. Establish explicit behavior for out-of-range manual values when learning is enabled; any normalization should be separate from a purportedly bounded learning step.

Sources: [selection clamp](../src/trajectory_editor/online_learning.py:234), [write clamp](../src/trajectory_editor/episode_policy.py:216).

## 3. Compiler and multi-token behavior

### Runtime routes need a separate selection policy

The compiler ranks routes by directness, boundaries, fragment size, token count, fan-out, and token IDs. It does not evaluate model likelihood or observed route frequency. Defaults allow all route classes and retain up to 4,096 routes per term. Runtime conversion does not filter exploratory routes.

Even `minimal` is not “use normal tokenizer output”: it chooses the highest-ranked route by the compiler's heuristic. In the existing `RouteRankingBackend` fixture, canonical `another` is `(1, 2)`, but minimal compilation retains `(3,)`.

Use canonical tokenizations of the intended surface variants as the baseline runtime representation, with explicitly supplied or empirically observed alternatives. Preserve exhaustive routes in a separate inspection section or artifact. Canonical tokenization is a practical baseline, not proof that a model will always generate that route; collect actual usage before adding alternatives to runtime.

Sources: [route ranking](../src/trajectory_editor/bias_catalog.py:1293), [selection and canonical handling](../src/trajectory_editor/bias_catalog.py:1371), [runtime conversion](../../core/src/trajectory_editor/bias_rules.py:529).

### The search budget does not bound the search work

`max_routes` limits retained output after enumeration, ranking, and allocation. It does not stop exploration. Minimal mode also enumerates before selecting. Enumeration depth is raised to at least the canonical route length, even when `max_route_tokens` was explicitly set lower. Route ranking additionally scans the candidate set for each candidate's head fan-out.

Introduce separate output and exploration budgets, lazy/limited enumeration, cached scoring inputs, and a tokenizer-only path for normal compilation. Report truncation and canonical exceptions clearly.

Source: [enumeration](../src/trajectory_editor/bias_catalog.py:1242).

### Existing modes cannot solve phrase reliability by themselves

- `auto` generally uses path behavior for words and tail behavior for phrases. Tail rules do nothing until the entire preceding token sequence matches; they cannot help the model start a phrase.
- Path rules promote heads, including heads shared with unrelated words. Increasing a group scalar also increases that collateral pressure.
- Each route independently falls back to its head. With paths `(1,2)` and `(3,4)`, history `(1,)` biases both continuation `2` and fresh head `3`. The existing tests explicitly preserve this behavior.
- Beheaded routes avoid the first-edge pressure but depend on the model entering the prefix unaided.
- Allocation weights that sum to one are a logit-allocation convention, not a guarantee of equal completed-phrase probability across route lengths. Modes, scales, competing continuations, and decoder filtering still matter.
- Matching recognizes token sequences, without a general completed-word boundary contract. A word token may also be the beginning of a longer spelling. This matters for distinguishing whole-word intent from stem matching.

**Recommendation:** Treat entry, continuation, completion, and abandonment as distinct measurable events. Start with a compact trie over approved routes and explicit scope gates. Make root restart behavior deliberate. Preserve overlapping legitimate matches rather than assuming a single winning route. Evaluate positive and negative steering separately: suppressing a final phrase token can leave an awkward partial phrase, while suppressing a common head affects unrelated text.

Use sequence-level feedback from typed spans to improve credit assignment. Later, test bounded model lookahead for difficult multi-token targets if its cost is justified. Increasing scalar strength alone does not provide that information.

Sources: [mode choice](../src/trajectory_editor/bias_catalog.py:1364), [path matching](../../core/src/trajectory_editor/bias_rules.py:366), [weighted matching](../../core/src/trajectory_editor/bias_rules.py:391).

### Form generation should be explicit and inspectable

Keep case and spacing expansion. The morphology is currently simple English suffix logic: pluralizing an already plural form or attaching `ing` to an `e` ending can produce unwanted surfaces. Explicit `forms` are added to the bases and then expanded again; they are not an exact allowlist.

Provide an exact-forms option, per-term exclusions, and a preview of the final surfaces. CLI-created plain-text members currently use a one-shot tokenization, so they do not receive the YAML compiler's expansions. Both authoring paths should use the same semantic resolver, while retaining an exact-token/text escape hatch.

Sources: [form generation](../src/trajectory_editor/bias_catalog.py:1017), [CLI member resolution](../../core/src/trajectory_editor/episode_ui.py:577).

## 4. Reference weights should become standalone

The requested separation is feasible. Runtime reference routes and the weighted trie already have their own representation. The coupling is primarily in CLI loading, catalog serialization, and active-route selection.

There are two distinct consumers today:

1. Compile-time character-prefix statistics for information allocation. Missing generated target forms are added with weight 1.
2. Runtime tokenized reference routes, generated from the supplied reference surfaces and leading-space variants, with each surface's weight split across variants.

Those universes are not identical and should be named separately. A reference weight is lexical mass, not a model parameter or literal output probability.

The current default `active` scope also ignores bias direction: any nonzero group amount admits matching reference routes. Reproduction with weights 100:1 gives prior adjustments about `+0.575646` and `-0.575646` for both a `+0.1` group and a `-0.1` group. Thus the supposedly suppressed high-weight token receives a net positive adjustment of about `+0.475646`. At group amount zero, the active prior disappears. This introduces a discontinuity around zero into the learner's policy family.

This is consistent with an independent prior, but the current automatic coupling obscures that independence.

**Recommendation:** A standalone reference input/artifact with tokenizer identity, weights, normalization/variant policy, and explicit scope. Let users apply it globally without constructing a dummy catalog. If tied to selected groups, define whether it expresses independent style preferences or must respect directional constraints; do not silently infer that from nonzero bias.

Sources: [reference compilation](../src/trajectory_editor/bias_catalog.py:533), [loader coupling](../../core/src/trajectory_editor/episode_cli.py:497), [active filtering and scoring](../src/trajectory_editor/sampling.py:196).

## 5. Learner feature parity

| Capability | Bias learner at this commit | Latent learner | Recommended bias behavior |
| --- | --- | --- | --- |
| Learning rate / update limit | Present | Present | Retain; distinguish per-target and total update limits |
| Severity cap | Hardcoded 1000 | `--latent-severity-cap` | Add `--learning-severity-cap` |
| Full-strength evidence | Unavailable | `--latent-no-severity-attenuation` | Add `--learning-no-severity-attenuation` |
| Dead zone | Fixed rank-1 zero severity | `--latent-dead-zone-rank` | Add `--learning-dead-zone-rank` |
| Explicit proposal rejection | Unavailable | `--latent-rejection-strength` | Add analogous control using chosen-versus-proposed group features |
| Forgetting | Unavailable | `--latent-decay` | Add decay for learned offsets, preserving explicit user intent |
| Fast/slow memory | Unavailable | Fast/slow channels and controls | Useful for local adaptation versus durable preference |
| Global policy strength | No separate multiplier | `--latent-strength` | Consider a learned-bias strength multiplier for evaluation and ablation |
| Aggregate magnitude budget | Scalar min/max only | Vector norm bound | Add a joint budget for many overlapping groups |
| Learn from typed writes | Shared opt-in flag | Shared opt-in flag | Already available to both |
| Write aggregation | Mean of already bounded token deltas | Sum raw evidence, then clip/decay once | Share explicit evidence aggregation semantics |
| Policy display | Shared | Shared | Preserve; add per-target attribution |
| Projection dimension/seed/chunking | Not applicable | Feature construction controls | No bias equivalent needed |

Sources: [bias configuration](../src/trajectory_editor/online_learning.py:32), [latent configuration](../src/trajectory_editor/token_preference.py:30), [CLI flags](../../core/src/trajectory_editor/episode_cli.py:247), [write aggregation](../src/trajectory_editor/episode_policy.py:216).

Parity should include diagnostics: settings used, proposed/rejected token, raw learning evidence, clipping, decay, before/after target probability, and reason for a skipped update. Currently the bias result has weights/gradients/deltas, but lacks much of the latent learner's richer decomposition. When both learners run, write-token summaries currently prefer the bias result's severity; differing dead-zone policies can therefore require separate channel summaries.

## 6. How automatic strength could work

The existing learner fits a selected token using the present observation. It does not update on autonomous `Hold` generation or optimize completed-word frequency. Interactive acceptance can count as a selection, so “selection evidence” is more precise than assuming every update represents a rejected proposal.

A direction-only interface is achievable in stages:

1. `+` or `-` activates a directional policy with a bounded initial strength chosen by the system. An explicit amount remains a manual override.
2. Learn a nonnegative magnitude from human selections/writes while preserving scope and sign. Use a dead zone and regularization to prevent escalation.
3. Track active target mass, route entry/completion, abandoned prefixes, actual decoder inclusion, and repeated corrections. Use that evidence for context-sensitive strength.
4. If adapting during autonomous generation, define a separate controller objective, such as a bounded odds increase or suppression ceiling. Direction alone does not specify a desired occurrence rate. A product default can supply the objective; it need not become another mandatory user tuning step.

Do not train on the model's own sampled choices as though they were fresh human preferences. Keep feedback learning distinct from generation-time control. Claims that this improves phrase success require model-based evaluation.

### The current objective and performance can be simplified

The bias learner performs two complete counterfactual policy evaluations per selected group. Each creates a new configuration, revalidates rules, and constructs observation statistics, including a decoder distribution that the loss does not use. The loss is based on the full, untempered policy softmax; temperature and top-k/top-p/min-p determine a separate decoder distribution. This smooth learning objective is useful, but improved policy probability does not guarantee the token survives decoder filtering.

For fixed activation and routes, the adjustment is linear in group strength. If `a_g(t)` is group g's deduplicated edge scale, the current negative-log-probability gradient is exactly:

`sum_t p(t) * a_g(t) - a_g(chosen)`.

Compute sparse group feature maps once and share the policy probabilities. That removes finite-difference epsilon and repeated whole-policy work for this objective. Keep finite differences as a verification method or for genuinely nonlinear experimental policies. Explicit activation also removes the reference-prior discontinuity from strength fitting.

Sources: [counterfactual loop](../src/trajectory_editor/online_learning.py:216), [policy and decoder distributions](../src/trajectory_editor/sampling.py:405).

## 7. Refactoring boundaries and persistence

Suggested concepts:

- **Term definition:** intended surfaces, exact/expanded matching policy, exclusions.
- **Group definition:** stable membership references to terms or other groups.
- **Compiled token data:** normal routes and shared matching structures, keyed by tokenizer identity.
- **Activation:** target, enabled state, direction, scope, manual/automatic strength mode, learnability.
- **Learned state:** slow/fast offsets or magnitudes, bounds, and learner configuration/version.
- **Reference lexicon:** independent weighted surfaces and compiled routes.
- **Exploration report:** alternate decompositions and experimental allocation diagnostics.

Use one resolver across command-line and YAML authoring. Move resolution out of the large UI command branch. Runtime nested groups currently copy templates, and scoped rules copy group routes; editing a source group does not dynamically update those copies. Decide and expose snapshot versus live-reference behavior.

Reuse trie/failure-transition infrastructure where appropriate, but do not mechanically replace all bias matching with the reference prior's single-state scoring. Bias groups need simultaneous overlapping targets and scope gates. Preserve deterministic rewind/replay by reconstructing state from saved history or restoring validated checkpoints.

Performance opportunities:

- Cache compiled matchers independently of mutable magnitudes.
- Cache active reference tries by activation set; active scope currently rebuilds a filtered trie on evaluation.
- Avoid rescanning full history for every route and reference evaluation.
- Store static definitions once, with lightweight activation/learner snapshots per boundary. Current sampler segments serialize full configurations.

Format changes are reasonable, especially for catalog JSON. However, `SamplingConfig` is used by episodes, replay, recovery, export, and backend policy application. Give new runtime semantics an explicit version and reject unsupported records clearly if migration is intentionally omitted.

Additional format issues to address:

- Bias preset export includes group/rule and latent state, but not reference-prior state. It does not reproduce the complete steering policy.
- Editor-friendly YAML exports membership only. Alias source text, compile options, scopes, and learned amounts are not faithfully round-tripped by that projection.
- Catalog validation has a token-text fingerprint; bias presets use weaker filename/file-size/backend/vocabulary identity. Standardize identity across artifacts. A stronger tokenizer identity should include tokenization rules/configuration, not only displayed vocabulary pieces.
- Learned values are saved, but learner launch settings are not part of `SamplingConfig`. Resuming learned state does not inherently restore the learning policy that produced it.
- The HTTP/headless runner is constructed without either learner, so future parity across interfaces requires shared configuration wiring too.

Sources: [matcher construction](../src/trajectory_editor/domain.py:245), [reference trie rebuilding](../src/trajectory_editor/sampling.py:209), [sampler persistence](../../core/src/trajectory_editor/episode_store.py:494), [preset export](../src/trajectory_editor/bias_presets.py:110), [archived headless preview](archive/HEADLESS.md).

## 8. Suggested implementation order

1. **Correctness:** authoritative precommit observations, frozen-group preservation, explicit bound handling, and scoped-learning behavior. Add regression coverage for the reproduced cases.
2. **Common learning controls:** severity/dead-zone/rejection controls, comparable write aggregation, and transparent diagnostics.
3. **Definitions and activation:** unify terms/groups/scopes as learnable targets; separate enabled state, intent, and magnitude; version runtime state.
4. **Compiler split:** normal surface tokenization for runtime, separately budgeted exploratory decomposition, shared CLI/YAML resolver.
5. **Independent references and cached matching:** separate artifact loading, explicit composition, immutable compiled data, sparse analytical group updates.
6. **Automatic magnitude and phrase experiments:** evaluate feedback-driven strength first; add ongoing control or lookahead only against explicit success criteria.

Evaluation should cover single-token words, split words, phrases, common shared heads, case/space variants, overlapping groups, and triggered scopes, with both directions. Measure completed-target rate, collateral word changes, abandoned prefixes, repetition, human correction effort, decoder inclusion, latency, and replay equality. Compare with ordinary decoding and fixed-bias baselines on fixed prompts/seeds.

## Validation performed

Ran the existing focused suites for catalog compilation, bias rules, both learners, latent controls, reference priors, typed-write learning, policy diagnostics, and commands: **252 passed**.

Also ran temporary, deterministic reproductions using the repository's fake backends for zero-group reactivation, sign reversal, inactive-group clamping, excluded-group write clamping, stale post-edit observations, scoped-learning leakage, canonical-route displacement, path restarts, and sign-independent reference scoring.

These establish code behavior. No real-model generation study or throughput benchmark was performed, so runtime quality and performance improvements above are proposals to evaluate, not measured gains. The only added repository file is this report.

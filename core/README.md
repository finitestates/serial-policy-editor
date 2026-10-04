# Policy Editor Core

This subproject contains the standalone runtime: the menu-driven episode
editor, replay/resume/fork/rewind behavior, persistence, supported model
backends, core sampler, grouped phrase biases, and episode projector.

Install it directly from this repository with:

```bash
python -m pip install ./core
```

The base install does not install an inference backend. Choose one explicitly:

```bash
python -m pip install './core[llama]'
python -m pip install './core[transformers]'
python -m pip install './core[transformers-accelerate]'
```

The ordinary Transformers extra is sufficient for basic CPU/GPU inference and
does not install Accelerate. The `transformers-gguf` and `transformers-bnb`
extras are explicit heavier paths.

Core can load externally produced steering-vector artifacts, but
vector creation and inspection are provided by the separate `vector/` package.

## In-memory episodes

For an application that does not need a workspace, construct `LiveSession`
with an `EpisodeEngine`. It records action tape, outcomes, sampler changes,
branch lineage, and rewind/fork state without importing or
requiring `EpisodeStore`. A fork is a lightweight session event: it retains a
reconstructible token prefix and reactivates on the one loaded backend. An
optional backend cache snapshot can make that faster, but is never branch
identity. Use `LiveSession.branch_handle(...)` for a branch-bound view.

The terminal editor runs new sessions in memory by default and does not create
a workspace during a normal run. Pass `--workspace` to open or create the
default `episodes.sqlite3`, or pass `--workspace PATH` to choose another file.
Saved-episode operations such as resume, replay, fork-from, and listing require
a workspace:

```bash
policy-editor --model model.gguf --new-prompt 'Tell a story'
policy-editor --model model.gguf --teacher-plan plan.yaml
policy-editor --workspace --model model.gguf --new-prompt 'Tell a story'
```

Teacher plans can be YAML or the existing JSONL tape format. A YAML plan keeps
the prompt, optional initial sampler defaults, recording metadata, and ordered
action/observation steps in one file:

```yaml
format: serial-policy-tape
version: 1
prompt: |-
  Tell a short story.
environment:
  sampler:
    draw_kernel: argmax
    temperature: 1.0
    top_k: null
    min_p: 0.0
steps:
  - step: 0
    action:
      kind: write
      mode: exact
      text: "Once upon a time."
    observation:
      token_ids: [101, 202]
      stop_reason: completed
```

Each step is one submitted action; its observation can contain multiple token
IDs. In handoff mode, every action needs an observation. Ballistic mode permits
omitting observations. Explicit CLI sampler options override the plan's initial
sampler values. `--procedure --projector ID` prints this same executable YAML,
and `--export-teacher-plan ID FILE.yaml` writes it to a file. Use a `.jsonl`
suffix to retain the existing JSONL export format.

At EDGE, `fork N` creates and selects a live branch, `rewind N` changes the
selected branch, and `branches`/`switch N` navigate retained branches.
`export FILE` writes the selected portable teacher tape without opening a
workspace. `save [WORKSPACE [ID]]` materializes just that branch as one durable
episode; without a workspace argument it uses the `--workspace` selection or
defaults to `episodes.sqlite3`. `save-family [WORKSPACE [ROOT_ID]]` materializes
all retained branches and their lineage using the same workspace selection.
`quit` and process exit discard unsaved in-memory history. The live terminal
interface is the same whether or not a workspace is open. This experimental branch requires workspace schema 3. Previous-format
workspace data is rejected, left untouched and not migrated. Package metadata
still carries the previous release version; it does not certify this branch.

### Eligibility and selective perturbation

`--eligible-k N` (also `--top-k N`) restricts proposal eligibility to the top
N adjusted logits before the min-p logit-gap cutoff. It does not discard
model logits used by search. `--eligible-k none` disables that restriction.

`--selective-noise-k N` adds noise only to the top N scores in the final eligible
set, ranked before noise with token ID breaking ties. Other eligible candidates
retain their scores and can still win. `none` (the default) perturbs every
eligible candidate. This setting requires a perturb-and-argmax draw kernel.
Temperature zero still restricts the candidate set to its single greedy winner.

For a full-vocabulary robustness probe, use:

```sh
policy-editor --unfiltered --draw-kernel gaussian-max --selective-noise-k 5
```

For competition restricted to 20 logits, with only the leading 5 perturbed:

```sh
policy-editor --eligible-k 20 --min-p 0 --draw-kernel gumbel-max --selective-noise-k 5
```

In the editor, use `s eligible_k=20 selective_noise_k=5 draw_kernel=gumbel-max`.
The eligibility setting keeps its existing `top_k` name in saved records;
`selective_noise_k` is saved independently and defaults to `none` when omitted.
Eligibility defaults to unrestricted. Plain argmax is the default kernel.
Min-p is implemented entirely as a logit gap: eligible scores must be within
`-log(min_p)` of the best score; `min_p=0` disables that cutoff.
Top-p, typical-p, tail-free filtering, categorical draws, and stochastic Gumbel
beam search have been removed. Records containing those settings are rejected.

Use `~` or `overlay noise` to show each candidate's additive perturbation. Untouched eligible candidates show zero; excluded candidates
show `--`. Noise stays hidden in the default rank / token ID / text view.

`gumbel_top_k` is a separate menu limit: it displays the leading candidates
ranked by logit plus Gumbel perturbation, with the proposal first. It does not
restrict eligibility or select which candidates receive noise. With selective
noise enabled, that ranking includes the unchanged eligible competitors.

### Direct overlay controls

| Command | Effect |
| --- | --- |
| `l` | Toggle raw model logits |
| `L` | Toggle logit diff from the raw argmax |
| `~` | Toggle additive noise |
| `%` | Toggle probability diagnostics |
| `C` | Reset to rank / token ID / text |
| `columns` | Show active overlays and shortcut guidance |
| `columns logit diff noise` | Set exactly those overlays |
| `overlay noise on` / `overlay noise off` | Explicitly enable / disable an overlay |

Shortcuts and named commands edit the same selection. All overlays are hidden
by default. The diff is `token logit - raw argmax logit`, so the raw argmax has
zero diff and lower model scores have negative diff. Noise is a separate
additive change to the eligible score. Probability is an optional diagnostic.

Neighbor margin, vocabulary z-score, and column cycling have been removed.
`c N` and `c all` remain context commands. `V` still toggles policy diagnostics;
`C` clears those columns too, without changing candidate ordering or search.

## Numeric and public API contract

`EligibleScores(ids, scores)` holds the required eligible token IDs and their
pre-noise scores. Use `is_eligible(token_id)` or `eligible_ids` for membership.
The cached `softmax` and `softmax_at(token_id)` are optional diagnostics;
selection must not require them. This replaces the public `SparseDistribution`
API. An excluded token has no eligible score; underflow of an eligible token's
softmax does not make it ineligible.

The probability overlay labels the actual calculations:

- **model-softmax**: normalized raw model logits over the vocabulary;
- **policy-softmax**: normalized adjusted policy logits over the vocabulary;
- **eligible-softmax**: normalized eligible scores after temperature and filters,
  before noise.

These are not general winner probabilities. Full independent unit-scale Gumbel
noise has categorical winner probabilities equal to eligible softmax; selective
noise and other families do not have that guarantee. Noise scales and the full
selection order are documented in the [root guide](../README.md).

`Candidate.eligible` is explicit membership. `Candidate.eligible_softmax`
replaces `decoder_probability` and may be `None`. Removed candidate fields are
`neighbor_margin` and `logit_z`. `EpisodeObservation.proposal_eligible_softmax`
is lazy; `ChoiceSet.proposal_eligible_softmax` is optional. `TokenEvidence` uses
nullable `eligible_softmax`; normal generation and writes do not request it.
An explicitly requested projector diagnostic replays the recorded model and
settings through existing identity/parity checks to reconstruct missing values.
Missing values render as `--`; calculated zero renders as zero.

Schema 3 renames the mandatory SQLite probability field to nullable
`eligible_softmax`. Old schemas are rejected rather than silently reinterpreted.
Old sampler records containing removed settings also fail validation. Portable
plans must use current settings; illustrative observation IDs in the example
above must be replaced with IDs from the chosen tokenizer.

Beam is deterministic, width-bounded sequence search. Its one cumulative score
is normalized policy log-probability, independent of proposal temperature,
eligibility and noise. It does not promise a globally optimal sequence.
`draw RAW_RANK` searches noise seeds; plain argmax rejects that search. `reroll`
is recorded but cannot change a plain argmax winner.

The [contract matrix](../tests/CORE_CONTRACTS.md) specifies the new target.
Existing harnesses and the [legacy oracle](../reference-kernel/README.md) are
not yet current validation evidence.

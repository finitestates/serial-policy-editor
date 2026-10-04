# Documentation overhaul plan

Status: documentation implemented. Runtime validation and harness/oracle
migration remain deferred to the next cut. This file records the approved scope.

## Implementation boundary at this pause

- Selection uses full model logits, explicit policy adjustments, eligibility,
  optional replay-stable noise, and final argmax. No token CDF is involved.
- Defaults: plain argmax, top-k none, min-p 0, temperature 1. Full vocabulary
  is eligible by default. Explicit min-p retains its logit-gap cutoff.
- EligibleScores carries required IDs/scores and explicit membership. Softmax
  is a lazy diagnostic, not the representation needed by the selector.
- Model/policy/eligible softmax values are opt-in UI diagnostics. The eligible
  softmax is computed over eligible pre-noise scores, not perturbed scores or
  general winner probabilities. Gumbel's categorical equivalence requires
  independent unit noise on every eligible candidate.
- Normal token evidence leaves eligible_softmax unset. An explicit projector
  probability request reconstructs missing diagnostics by replay, using the
  recorded model/settings and existing identity/parity checks.
- Workspace schema 3 renames the old mandatory decoder_probability column to
  nullable eligible_softmax. Prior workspace schemas are rejected before schema
  creation; no migration or in-place conversion is performed.
- Proposal surfaces say Argmax. Membership is independent of numerical softmax
  mass, including underflow or unrequested diagnostics.
- Beam retains its cumulative normalized policy log-probability objective. It
  has one score, branch state/text, protection/family controls, and recent token
  rank details; duplicate cumulative scores and probability detail rows are gone.
- Candidate overlays use one explicit selection. Launch view flags normalize
  once into that selection. Margin/z-score projection and legacy renderer
  preference plumbing have been removed. The default table remains three columns.

## 1. Establish the product story and algorithm

Rewrite the root README introduction around "if not the argmax, then what?".
Describe raw model argmax, adjusted argmax, and argmax after perturbation without
claiming a globally optimal sequence. Explain accept versus direct rank choice
and preserve tokenizer-transparent search.

Provide one explicit algorithm in this order: model/guidance logits, policy
adjustments, temperature scaling, optional top-k/gap eligibility, optional
selective noise, final argmax with token-ID tie breaking. Temperature zero's
single-candidate behavior needs its own sentence.

Define eligible-k and selective-noise-k with a small table and worked example.
Mention that the unchanged best outsider can win under selective noise, while
membership does not promise reachability under every noise family/strength.

Files: README.md, CORE_SCOPE.md, CUT_NOTES.md, core/README.md.

## 2. Explain noise and softmax precisely

Describe the retained families and their actual parameters. Student-t's current
scaling is t_df/sqrt(3); perturb_noise_std is not literally a finite standard
deviation at every df. Explain Gaussian versus heavy-tailed behavior without
unmeasured quality or speed claims.

Identify Gumbel as the control, and state the conditions for its categorical
winner-distribution equivalence. Do not extend that claim to selective noise.

Document model-softmax, policy-softmax, and eligible-softmax as three different
optional diagnostic surfaces. Explicitly separate eligible softmax from winner
probability. Explain that raw logits plus optional differences/noise are enough
for normal interaction.

Files: core/README.md, README.md, source module docstrings and CLI help.

## 3. Replace the UI guide and examples

Document l / L / ~ / % / C, exact `columns` selection, explicit overlay on/off,
and policy columns/order controls. Explain that C restores rank / token ID /
text without changing token search, ordering, or proposal selection.

Use examples for plain argmax, all-candidate Gumbel, eligible-k competition, and
selective-noise robustness probes. Remove invalid top-p/typical/tail-free and
stochastic beam examples. Distinguish a Gumbel-ranked menu limit from eligibility
and noise application limits. Describe seed search/reroll limitations in plain
argmax mode.

Explain deterministic beam's policy log-probability objective and bounded width;
it is a separate sequence-search objective from choosing the next argmax.

Files: README.md, core/README.md, examples/decoder_profiles.example.json,
teacher_commands.py help/palette descriptions, live screenshot fixtures.

## 4. Explain persistence and compatibility

Document schema 3, nullable evidence, explicit diagnostic replay, and the
fresh-workspace requirement. Keep old workspaces untouched. Describe the sampler
record break separately from schema changes. List renamed/removed public record
fields and EligibleScores for API users.

Preserve unchanged noise-address scheme literals and clarify that replay needs
model identity, logits/settings, boundary, seed, and token/rank addresses.

Files: core/README.md, DEVELOPMENT.md, CORE_SCOPE.md, projector/API documentation.

## 5. Align the oracle and validation narrative

Mark the independent reference kernel as legacy until it is ported in the
harness cut. Preserve previous-engine usage in LEGACY_USAGE.md.
Rewrite contract descriptions for exact winner/tie behavior, membership,
selective-noise scores, absent default softmax work, optional diagnostic
reconstruction, schema rejection, and direct overlay state.

In the next harness cut, update retired API callers and test fixtures deliberately. Do not paper over
semantic breaks with compatibility shims solely to keep old tests green.

Files: reference-kernel/, tests/CORE_CONTRACTS.md, affected test/benchmark fixtures.
Preserve CHANGELOG history; add an experiment entry instead of rewriting old
release behavior.

## 6. Review, then validate

First review the algorithm/evidence/UI contracts and the proposed wording.
When validation resumes, run focused engine/persistence/projector contracts,
noise and oracle parity, and real terminal journeys including overlays, search,
beam, focus, and resize. Model and physical-display claims need their own
appropriate evidence. Do not claim that source syntax review establishes
runtime, storage, model, or terminal correctness.

Documentation now covers the algorithm, noise, UI, API/schema compatibility,
examples, legacy oracle status and revised contract slots. No tests or runtime
checks were run during this cut. Source syntax and diff review do not establish
runtime correctness. The next work is specified separately in
[the testing harness plan](ARGMAX_TEST_HARNESS_PLAN.md).

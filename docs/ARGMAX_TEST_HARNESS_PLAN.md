# Argmax testing harness migration plan

Status: implemented for the retained Python harnesses after the documentation
checkpoint. See [results](ARGMAX_HARNESS_UPDATE.md). The following text preserves
the planned scope; real-model checks remain outside the local evidence. Static inspection found retired API/filter
references in the files below; this is a migration inventory, not an exhaustive
prediction of failures. Preserve the contract IDs in `tests/CORE_CONTRACTS.md`.

## 1. Establish independent numeric expectations

Port `reference-kernel/reference_kernel.py` before using parity as evidence.
Replace categorical/CDF and retired filters with independent adjusted-score,
temperature, eligibility, perturbation and argmax formulas. Keep replay-address
hashing, supported noise quantiles, penalties, ties and tokenizer addressing.
Do not import production helpers for expected membership, scores or winners.

Rewrite `test_reference_kernel_parity.py` and `test_reference_kernel_oracle.py`
for current records and pipeline. Replace `test_top_k_policy_remaps_same_quantile_in_production`
with independent eligible-set/winner assertions. Replace fixed categorical draw
cases with plain argmax and all-candidate Gumbel controls. Retain random-address,
reroll serialization, model-rank addressing and history checks where meaningful.
Retire the CDF-boundary demonstration or replace it with a score/noise/winner
coordinate demonstration; update its usage only after the implementation exists.

Use hand-calculated tiny vectors as well as oracle parity. Include lowest-ID
ties, empty/invalid inputs, k exceeding vocabulary, min-p boundary equality,
temperature zero, zero noise strength, and changed eligibility. Verify that an
untouched eligible outsider can win selective noise. Check all families/scales
and Student-t df behavior. A finite deterministic fixture cannot establish the
entire Gumbel categorical law; use exact construction/formula checks and any
statistical check with an explicitly bounded claim.

## 2. Migrate shared fixtures and sampler tests

Start with `tests/core/runtime_helpers.py`, `term_support.py`, and shared
Candidate/ChoiceSet/TokenEvidence constructors. Replace SparseDistribution and
mandatory decoder_probability with EligibleScores, explicit membership and
optional eligible_softmax. Do not fill missing diagnostics with invented zero.
Keep shared fixtures small so downstream failures retain their actual meaning.

Rewrite S01/S02 cases in `test_sampler_contracts.py` and generated records in
`test_property_contracts.py`. Remove top-p/typical/tail-free and categorical
success cases; replace appropriate configuration cases with rejection checks.
Remove `test_s04f_neighbor_margin_matches_consecutive_logit_gaps` and
`test_s04g_logit_z_score_uses_full_vocab_mean_std_without_softmax_wake`: their
features are gone. Replace column-cycle expectations in S04e with direct state
controls. Retain raw rank, biases, CFG, Student-t parameters and serialization.

Review `test_selective_noise.py` against the final EligibleScores interface;
its presence does not mean it passed after later cuts. Extend membership/tie
and unperturbed-competitor cases for both eligible-k and selective-noise-k.
Seed-search tests must cover plain-argmax rejection, unreachable targets, and
supported noisy targets. Reroll must be a recorded no-op for plain argmax.

## 3. Assert optional diagnostics and durable behavior

Migrate `test_lazy_candidate_projection.py`, sampler lazy-metric tests,
`test_engine_contracts.py`, `test_episode_history.py`, and projector/persistence
contracts. Add spies that fail if normal observe/accept/write/three-column
rendering materializes softmax. Explicit diagnostic requests must calculate the
correct model/policy/eligible normalization, with None distinct from zero.
Include eligible values that underflow to zero without changing membership.

Use temporary schema-3 databases to verify nullable evidence round-trip.
Construct representative old-schema files and prove rejection leaves their
bytes/schema unchanged. Test projector reconstruction with matching recorded
identity/settings and clear failure on mismatch; ensure it does not rewrite
normal evidence as a side effect. Retain lineage, tape, rewind/fork and partial
write contracts, updating only changed records.

Review `test_cfg_contracts.py`, `test_fresh_roots.py`, `test_ephemeral_cli.py`,
`test_controller_profiles.py`, and `tests/vectors/test_activation_vectors.py`
for retired settings/APIs. Preserve backend ownership and raw-rank invariants.
Add profile cases for the documented examples and explicit CLI overrides.

## 4. Replace UI and beam expectations

Migrate `test_live_terminal.py`, menu tests and render fixtures for Argmax labels,
raw/diff/noise/softmax columns, one overlay state, and absent default diagnostics.
Exercise l/L/~/%, exact columns, explicit on/off, launch normalization and C.
C must preserve search, order and selection; c N/c all must remain context
commands. Distinguish unavailable diagnostic values from computed zero.

Remove stochastic-beam and duplicate model/search-score detail assertions.
Retain deterministic cumulative policy-logp, bounded width, protection/family,
branch selection and recent-token raw-rank tests. Add independence from proposal
eligibility/temperature/noise where the beam objective requires it. Do not merely
refresh snapshots until their underlying assertions match these contracts.

Then migrate `test_live_terminal_pty.py`, `test_live_terminal_fuzz.py` and
`core/scripts/render_live_screens.py` harness consumers. Keep complete-frame,
focus, search warming, caret, resizing and terminal restoration invariants.
Follow `tests/TERMINAL_RENDERING_GUIDE.md`; retain raw captures and negative
controls. A PTY/pyte result is not physical-display or comfort evidence.

## 5. Execute validation in separate, reported stages

After the harness edit is authorized, run numeric/configuration checks first,
then fixture consumers, engine/persistence/projector, independent oracle parity,
and terminal journeys. Report exact commands and failures; do not hide semantic
breaks with compatibility shims or broad skips. Run the retained broader suite
after focused checks. Retire a test only when its feature is explicitly removed;
retain unchanged lifecycle/backend contracts even when their fixtures break.

Real-model comparisons require matching tokenizer IDs, settings, precision and
prefill/incremental boundaries. Keep them distinct from synthetic parity and
terminal evidence. Review scripts, benchmarks, CI collection and documented
commands for stale selectors before calling the harness migration complete.
Performance claims require actual benchmarks; this plan makes none.

Commit the migrated harness separately from this implementation/documentation
checkpoint so the changed contract and its later verification remain reviewable.

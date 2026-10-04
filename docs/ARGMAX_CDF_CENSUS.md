# CDF assumptions after the argmax cut

Scope: `policy-editor-argmax`, following experiment commit `c786998`.
This is a static source and documentation census. No tests, model runs, or
terminal journeys were performed. Recommendations below are not implemented.
The deterministic beam correction noted below is the sole code correction made
during this census.

## Current selection algorithm

1. Obtain the full model logits, including configured guidance/steering.
2. Apply history penalties and explicit policy biases.
3. Scale adjusted scores by positive temperature. Temperature zero selects the
   single adjusted winner before noise.
4. Restrict eligibility with optional top-k, then the min-p logit gap.
5. Add the selected noise family to all eligible candidates, or only the top
   selective-noise-k candidates ranked before perturbation.
6. Choose the maximum final score, breaking ties by the lowest token ID.

There is no categorical CDF lookup in production core/vector source. Noise
still uses deterministic uniform quantiles and transformations into named noise
families. That is noise generation, not a CDF over token probabilities.

Defaults are plain argmax, temperature 1, top-k none, selective-noise-k none,
and min-p 0.0. The full vocabulary is eligible by default. An explicit
positive min-p limits the score gap to `-log(min_p)`; min-p 0 or
`--unfiltered` removes that cutoff.

Unit Gumbel perturbation over every eligible candidate remains the categorical
control over that eligible set. Selective Gumbel perturbation leaves some scores
unchanged and no longer has that usual categorical equivalence. The other
noise families generally do not have softmax winner probabilities.

## Removed assumptions

| Former assumption | Previous surface | Result |
| --- | --- | --- |
| One uniform draw selects a cumulative probability interval | `sampling.py`: categorical branch and position_uniform | Removed |
| Cumulative token mass defines eligible support | top-p filter | Removed |
| Entropy/surprisal ordering plus cumulative mass defines support | typical-p filter | Removed |
| Probability derivative mass defines support | tail-free filter | Removed |
| Min-p needs normalized candidate probabilities | min-p filter | Replaced by `score >= maximum + log(min_p)` |
| Child selection needs probability races / sampling without replacement | conditional_gumbel_top_k and stochastic beam | Removed |
| A final softmax must be built before choosing a token | PolicyCalculations distribution construction | Softmax is lazy; downstream UI/evidence still requests it |
| Repeated cycling is necessary to reach an overlay | c column cycle and l logit cycle | Replaced by direct toggles and exact column selection |

## Surviving code and simplification opportunities

| Surface | Evidence and meaning now | Opportunity / constraint |
| --- | --- | --- |
| `core/sampling.py`: SparseDistribution | Holds IDs and scores, but accepts supplied probabilities and optional scores; probability is lazy. Kernels require scores. | Replace with an eligible-score object whose scores are required. Move diagnostic probability calculation behind a separate API. This removes a probability-bearing constructor from every noise kernel. |
| CandidateFilterResult / StandardCandidateFilter | Remaining stages are temperature, top-k, and min-p. PolicyCalculations consumes only final after_min_p IDs and scaled logits. | Replace stage dictionary with final IDs/scores plus small eligibility diagnostics, unless a future inspector needs intermediate stages. |
| `episode_engine.py`: observe vs draw_token | observe dispatches the noise kernels to preserve ranking scores; draw_token dispatches the same kernels independently for seed search. | One ranking function returning final scores, one shared argmax winner, and callers retaining or discarding scores. Avoid duplicated dispatch and three equivalent winner functions. |
| `core/episode_observation.py`: proposal_decoder_probability | Softmax over eligible pre-noise scores, not actual probability of the argmax/noise winner. | Rename to eligible-score softmax diagnostic or remove it from mandatory observation/UI contracts. Keep actual winner and final score distinct. |
| `episode_ui.py`: _choice_from_observation; `core/ui.py`: ChoiceSet | Unconditionally requests proposal_decoder_probability even when the probability overlay is off. | Make this field optional and compute only when requested. The default three-column table currently does not guarantee absence of softmax work. |
| `episode_engine.py`: _evidence; `core/results.py`: TokenEvidence | Each selected token records decoder_probability, forcing softmax on token acceptance even without visible probability columns. | Choose whether diagnostic mass belongs in durable evidence. Making it optional/deferred can remove mandatory normalization; requires a storage/export contract decision. |
| `episode_store.py`, `episode_history.py`, `projector.py` | SQLite decoder_probability is NOT NULL; readers and projection format it as decoder-p. | Rename or version this evidence coherently, including history, schema, exports, and projector. Do not just delete an engine field and leave consumers intact. |
| `core/candidates.py`: decoder_supported | Derived from decoder_probability > 0. | Use eligibility membership. Zero diagnostic mass can mean floating-point underflow rather than exclusion. Eligibility is also different from winner reachability under selective/bounded noise. |
| `core/candidates.py`, `episode_engine.py`: neighbor_margin/logit_z; `core/policy_calculations.py`: z-score helpers | Metric fields and calculations survive although their overlays were removed. | Remove the retired projection fields/calculation branches if no external API consumer needs them. This is dead overlay machinery, not CDF logic. |
| `episode_ui.py`, candidate_columns.py, rendering contracts | Legacy logit_view and show_model_probabilities inputs coexist with the canonical overlay set; commands fold them together on edits. | Normalize launch preferences once and carry only the explicit overlay set through UI state. Preserve startup flag behavior deliberately. |
| `beam.py`: score, model_log_probability, search_log_probability | Stochastic scoring and root renormalization are gone. No caller now supplies an alternate score or search-step probability. Three cumulative fields follow the same recurrence from zero. | Collapse redundant path/candidate scores and remove obsolete optional override parameters. Static inference; revalidate observable beam behavior later. |
| `beam.py`: deterministic logsumexp | Still ranks trajectories by cumulative normalized policy log-probability. Uses no CDF. | Keep normalization unless deliberately changing the search objective. Summing raw logits across different prefixes is not equivalent. Labels saying model log-p should clarify that policy-adjusted logits are used. |
| `core/sampler_config.py`, cli_config.py, edge_status.py | SamplerConfig, draw_kernel, draw labels, and three family-specific noise-strength controls survive. | Selection/noise vocabulary and a unified active noise-strength UI could simplify the surface. Renaming stored fields is a compatibility change, not just cosmetic cleanup. |
| Temperature | Positive scaling cannot change plain argmax, but changes logit gaps relative to fixed noise and the min-p cutoff. | Explain it as score scaling. Do not remove it as universally redundant while noise and eligibility use scaled scores. |
| Seed, fingerprint, boundary, token-ID/model-rank noise addressing | Still define reproducible perturbations, rerolls, and replay. Plain argmax itself ignores seed. | Retain deterministic addressing. Do not rename RNG_SCHEME by accident: its literal string participates in noise hashes. |
| draw RAW_RANK / find_seed_for_token | Searches seeds for a proposal matching a target; argmax explicitly rejects it. Selective and bounded noise can make targets impossible. | Describe as proposal/noise seed search. Disable or explain it in plain argmax mode; reroll also cannot change a plain-argmax proposal. |

## UI census

| Surface | Remaining assumption / wording | Suggested follow-up |
| --- | --- | --- |
| `plain_tui.py`: render_choice | Always prints "Sampled proposal" and a decoder percentage, even with default columns. | Say "Proposal" or "Argmax proposal"; show diagnostic probability only on request. |
| `tui_render.py`: action_preview and preview fragments | Uses "sampled proposal" labels and a decoder percentage in candidate previews. Proposal fallback obtains this probability from ChoiceSet. | Use proposal vocabulary consistently and gate probability fragments by selected diagnostics. |
| `candidate_columns.py`: probability overlay | Shows model-p, optional pol-p, and decode-p; decode-p is pre-noise eligible-score softmax. | Rename decode-p to eligible-p, with help explicitly saying it is not winner probability for general/selective noise. |
| `teacher_commands.py`: HELP_TEXT | Still says "decode-p is final", which can imply the probability after perturbation. | Explain the three diagnostic surfaces and the separate final score/noise. |
| `edge_status.py` | min_gap is already exposed, but draw= and family-specific noise controls remain. | Show selector and active noise strength; show eligibility/noise limits as different controls. |
| `beam.py`, `plain_tui.py`, `term/views.py`, `tui_render.py` | Deterministic beam still exposes cumulative log-p, model-log-p, and step-log-p. | Retain the objective but distinguish policy-adjusted scoring from raw model diagnostics. |
| `term/palette.py` / direct overlays | Direct l/L/~/% toggles, exact columns selection, and C reset are already wired through help/palette. | Keep the default rank / token ID / text table. Remove the remaining legacy preference plumbing rather than adding another navigation mechanism. |

## Documentation and examples census

| File | Finding | Classification / follow-up |
| --- | --- | --- |
| `README.md` | Describes --unfiltered as disabling removed top-p/typical/tail-free controls. Profile example uses top-k 40. Sampler prose emphasizes draw/distribution vocabulary. Claims reference-kernel checks current formulas. | Current user-facing documentation needs an experiment-specific algorithm/defaults section and corrected reference-kernel status. A nondefault top-k example is valid if labeled intentional. |
| `CORE_SCOPE.md` | Still lists stochastic beam, draw-set terminology, and coordinate descriptions centered on a single draw. | Update supported features and describe per-candidate perturbation coordinates while preserving replay semantics. |
| `CUT_NOTES.md` | Kept pipeline still includes top-p and mandatory decoder-probability evidence. | Update scope notes to the actual cut and track the evidence decision separately. |
| `core/README.md` | New experiment and overlay sections are current. Earlier teacher-plan example has top-k 40; it is an explicit restriction, not the new default. | Label example intent; add the complete algorithm and diagnostic-probability caveat. |
| `reference-kernel/README.md` | Documents categorical CDFs, removed filters, and a categorical draw-coordinate demo as production parity. | Reference implementation has not been ported; mark as prior-engine oracle or independently implement the new contract before claiming parity. |
| `examples/decoder_profiles.example.json` | Contains removed top_p fields and a profile described as the editor default with top_k 40. | Replace or clearly archive. These records are not current argmax examples. |
| `tests/CORE_CONTRACTS.md` | S02 still specifies deterministic categorical/Gumbel draws. | Rewrite the contract around adjusted/final argmax, eligibility, noise addressing, ties, and optional diagnostics when validation resumes. |
| `core/scripts/render_live_screens.py` | Output mock mentions temperature 0.7/top-k 20. | Not inherently invalid; update to representative experiment defaults and overlays if used as current screenshots. |
| `CHANGELOG.md` | Historical entries mention sampled proposals and decoder probability. | Preserve historical claims. Add a new experiment entry if desired; do not rewrite release history as if it always used argmax. |
| `docs/ARGMAX_ENGINE_CUT.md` | Records the actual removals, new defaults, compatibility break, and no-test pause. | Retain as decision record; this census extends its follow-up list. |

## Reference, tests, and compatibility

`reference-kernel/reference_kernel.py` still implements categorical draws,
probability-based filters, and the old defaults. Its demo and parity suites do
not establish parity for this experiment. The corpus under tests/core and
reference-kernel also contains removed SamplerConfig fields, stochastic beam
calls, and retired overlay APIs. The new test_selective_noise.py predates the
cut: its parametrization now includes argmax, and its engine cases still pass
top_p. These are source-observed mismatches, not test failures observed by
running a suite.

Stored sampler records with removed fields are rejected. The new branch uses
the existing policy/RNG scheme strings; noise stream identity remains sensitive
to those literals. A future record/schema version decision should distinguish
selection semantics from unchanged noise addressing. Teacher plans, workspace
resume/forks, projector reconstruction, and source replay all restore through
SamplerConfig.from_record, so that compatibility break reaches beyond the CLI.

## Correction during this census

The stochastic-beam cut removed the local search_log_probabilities array but
left three references in deterministic candidate creation (live, forced root,
and terminal candidates). Those references would raise NameError when reached.
They were removed; _candidate now uses its existing default log_probability
argument for all three paths. This is the deterministic score already supplied
by each caller. No runtime verification was performed.

## Suggested next sequence

1. Correct UI wording and make proposal probability optional in the default UI.
2. Decide whether durable evidence should retain eligible-score softmax mass;
   update storage/projector contracts together if changing it.
3. Consolidate ranking/winner dispatch and replace probability-bearing candidate
   input with required eligible scores.
4. Remove redundant beam scores and retired overlay metric projections.
5. Update current docs/examples and independently port the reference oracle.
6. Only then resume focused contracts and real terminal/model validation.

The sampling pipeline remains useful. The largest simplification is separating
score competition from optional probability diagnostics, so diagnostic mass no
longer shapes mandatory engine, UI, or storage contracts.

# Core contract matrix

This specifies required behavior for the argmax experiment. Retained harnesses
have been migrated and run locally; see
[the validation record](../docs/ARGMAX_HARNESS_UPDATE.md) for commands, skips and
evidence limits. Filenames below map contracts to harnesses. This is the source of truth for the reduced
core test suite. The
numbers below are contract slots. Every retained core test must map to one of these slots. Additional slots may be added on an as-needed basis, but only after consultation with the code owner.

Core tests assert observable state, persisted records, replay results, and may also check for patterns associated with performance regression.

## Terminal contracts

| ID | Terminal contract | Harness files |
| --- | --- | --- |
| T01 | Choice, review, EDGE, Beam, prompt, page, single-key, multiline, and isolated chord requests submit their contract values through the real key parser; the production PTY session submits the same values for typed, pasted, clicked, and multiline input | `test_live_terminal.py`, `test_chord.py`, `test_live_terminal_pty.py` |
| T02 | engine-owned insertion previews run on the request owner; search warming runs on a worker and delivers generation-tagged results on the UI thread | `test_search_warm_terminal.py`, `test_live_terminal.py` |
| T03 | invalid choice commands remain editable; a submitted view stays displayed and marked busy until replaced, and input typed meanwhile is dropped | `test_live_terminal.py`, `test_live_terminal_pty.py` |
| T04 | EDGE blank Enter, Ctrl+C, and Ctrl+D preserve their command, interrupt, and cancellation values; interrupt and exit restore the terminal; the non-TTY path remains text-only | `test_live_terminal.py`, `test_live_terminal_pty.py`, `test_terminal_scenarios.py` |
| T05 | Choice feedback navigation, search-lens rank selection, authored-text editing, context paging, and review reactivation retain their command results | `test_live_terminal.py` |
| T06 | Beam shortcuts, selection, same-depth pruning/backfill with stable survivor IDs, deterministic policy-logp formatting, and prompt-mode values are asserted headlessly and on the production PTY | `test_live_terminal.py`, `test_live_terminal_pty.py` |
| T07 | ordinary rank browsing does not mutate speculative backend state before engine commit | `test_selection_warm_terminal.py` |
| T08 | on the production PTY (scripted requests, 30 Beam advances with resizes both ways, and the real runtime), the terminal equals every intended frame, every frame is complete and from one request, and no erase is written after startup; generated input journeys keep every view complete with a visible caret at every size | `test_live_terminal_pty.py`, `test_live_terminal_fuzz.py`, `test_live_terminal.py` |
| T09 | captured output retains exactly the trailing 16,000 characters across oversized and incremental writes; the viewer follows the tail and preserves paging away from it | `test_live_terminal.py` |
| T10 | Choice renders the full prepared context tail by default and does not jump to the tail while the user pages away | `test_live_terminal.py` |

## Sampler and action contracts — 8

| ID | Contract | Harness files |
| --- | --- | --- |
| S01 | `SamplerConfig` defaults to argmax/full-vocabulary eligibility, round-trips retained fields and rejects retired settings | `test_sampler_contracts.py` |
| S02 | plain/perturbed argmax obey exact score, lowest-token-ID tie and replay-address rules; selective noise affects only the leading eligible scores | `test_sampler_contracts.py` |
| S03 | CFG evaluates `P + V[:n]` and `U + V[:n]` with exact shared continuation IDs; standalone guidance uses `add_bos=True, special=True` independently of primary representation. Only retained visible tokens count toward the cutoff; new/resumed/replayed/forked/switched states are equivalent. Guidance reuses evaluation on forward append and unchanged decisions, catches up lazily, and rebuilds once on divergence, prompt change, or ownership loss. No warming past cutoff. | `test_cfg_contracts.py`, `test_sampler_contracts.py` |
| S04 | history penalties change policy selection without changing raw rank | `test_sampler_contracts.py` |
| S05 | a group member biases only its final token after its exact token prefix matches; the rule is identical for words and phrases | `test_sampler_contracts.py` |
| S06 | finite case/spacing variants and overlapping group/direct-token sources produce an attributable sum | `test_sampler_contracts.py` |
| S07 | every core action and replay expectation serializes canonically | `test_sampler_contracts.py` |
| S08 | llama and Transformers backends consume the same core sampler contract | `test_vector_contracts.py`, sampler smoke tests |

## Engine action contracts — 9

| ID | Contract | Harness files |
| --- | --- | --- |
| E01 | accept/select commits exactly the selected token without requesting softmax diagnostics | `test_engine_contracts.py` |
| E02 | raw-rank selection addresses the full vocabulary; eligibility is set membership independent of optional softmax and underflow | `test_rank_neighborhood.py`, `test_unexposed_ranks.py` |
| E03 | exact and ordinary writes produce the expected visible span | `test_engine_contracts.py` |
| E04 | phrase/check validates a complete span atomically | `test_engine_contracts.py` |
| E05 | force phrase commits through temporary bias without persistent residue | `test_engine_contracts.py` |
| E06 | holds respect count, boundary, stop, and teacher limits | `test_engine_contracts.py` |
| E07 | check and force are durable write actions, not menu-only events | `test_engine_contracts.py` |
| E08 | model EOG, teacher EOG, finite hold, and menu termination are distinct | `test_engine_contracts.py` |
| E09 | rejected or invalid actions leave token state unchanged | `test_engine_contracts.py` |

## Replay contracts — 10

| ID | Contract | Harness files |
| --- | --- | --- |
| R01 | exact replay reproduces the recorded visible prefix | `test_replay_contracts.py` |
| R02 | replay consumes `{step-N, teacher_action, optional handoff result}` | `test_replay_contracts.py` |
| R03 | handoff stops at the first divergent action/result | `test_replay_contracts.py` |
| R04 | ballistic mode continues with teacher actions after divergence | `test_replay_contracts.py` |
| R05 | check and force replay as their recorded writes | `test_replay_contracts.py` |
| R06 | replay EOG reaches the live edge without committing a terminal token | `test_replay_contracts.py` |
| R07 | forks, rewinds, searches, and menus never become tape steps | `test_replay_contracts.py`, projector tests |
| R08 | exhausted or divergent replay yields a usable live edge | `test_replay_contracts.py`, `test_replay_until.py` |
| R09 | replay never mutates the recorded source prefix | `test_replay_contracts.py` |
| R10 | replay statuses and terminal reasons remain semantically distinct | `test_replay_contracts.py` |

## Persistence and lifecycle contracts — 7

| ID | Contract | Harness files |
| --- | --- | --- |
| L01 | resume reconstructs an open episode and continues it | `test_lifecycle_contracts.py` |
| L02 | rewind can stop at any retained token boundary | `test_lifecycle_contracts.py` |
| L03 | rewind can cut inside a checked or multi-token write | `test_lifecycle_contracts.py` |
| L04 | fork preserves exactly the requested visible prefix | `test_lifecycle_contracts.py` |
| L05 | fork/rewind preserve episode identity and parent lineage correctly | `test_lifecycle_contracts.py` |
| L06 | sampler transitions restore at the selected historical boundary | `test_lifecycle_contracts.py` |
| L07 | model/backend continuation preserves visible text and sampler results | `test_lifecycle_contracts.py` |

## Vocabulary and menu contracts — 5

| ID | Contract | Harness files |
| --- | --- | --- |
| M01 | full-vocabulary search is non-mutating | `test_menu_contracts.py` |
| M02 | absolute/relative rank navigation resolves the requested candidate | `test_menu_contracts.py` |
| M03 | menu commands distinguish editorial moves from token actions | `test_menu_contracts.py` |
| M04 | `l`, `L`, `~`, `%`, exact `columns` and explicit overlay commands share one state; `C` restores three columns without changing search/order; diagnostics are opt-in | `test_menu_contracts.py` |
| M05 | the CLI exposes only the installed core surface and supports reusable profiles | `test_menu_contracts.py`, `test_controller_profiles.py` |

## Vector and backend contracts — 5

| ID | Contract | Harness files |
| --- | --- | --- |
| V01 | core loads an external JSON vector without model metadata requirements | `test_vector_contracts.py` |
| V02 | core loads a cvector with the exact canonical layer ordering | `test_vector_contracts.py` |
| V03 | malformed or dimensionally unusable artifacts fail clearly | `test_vector_contracts.py` |
| V04 | model labels do not gate vector application | `test_vector_contracts.py` |
| V05 | output vectors use backend math and runtime capabilities | `test_vector_contracts.py` |

## SQLite and export contracts — 4

| ID | Contract | Harness files |
| --- | --- | --- |
| P01 | schema 3 persists/reloads nullable eligible_softmax; old schemas are rejected before mutation | `test_persistence_contracts.py` |
| P02 | the persisted tape contains only replayable actions and optional results | `test_persistence_contracts.py` |
| P03 | fork maps represent exact visible boundaries, including zero | `test_persistence_contracts.py` |
| P04 | episode/projector export preserves procedure and lineage; explicit diagnostics reconstruct missing softmax through replay with identity/parity checks | `test_persistence_contracts.py` |

## Property and fuzz contracts — 5

| ID | Contract | Harness files |
| --- | --- | --- |
| Q01 | generated retained sampler records round-trip exactly; retired fields are rejected | `test_property_contracts.py` |
| Q02 | replay never changes the recorded prefix | `test_property_contracts.py` |
| Q03 | rewind followed by replay reproduces the retained prefix | `test_property_contracts.py` |
| Q04 | fork at N preserves exactly the first N visible tokens | `test_property_contracts.py` |
| Q05 | generated divergence terminates at the live edge or continues ballistically | `test_property_contracts.py` |

## Numeric contract interpretation

S02 must cover eligibility after policy/temperature, top-k token-ID tie ordering,
min-p as `best + log(min_p)`, temperature zero's single winner, and perturbation
address stability. An untouched eligible competitor may win selective noise.
Full independent unit Gumbel is the categorical control; selective Gumbel is not.
The independent oracle must implement these formulas without production helpers.

E01/M04/P04 must distinguish unrequested diagnostics (`None`) from calculated
zero. Normal observation, accept, write and rendering must not materialize
softmax. Explicit diagnostics may do so. Beam retains cumulative normalized
policy log-probability and bounded search, not stochastic without-replacement
semantics. Existing slots remain stable; the migration preserves these slots.

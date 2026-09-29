# Core contract matrix

This is the source of truth for the reduced core test suite. The
numbers below are contract slots. Every retained core test must map to one of these slots. Additional slots may be added on an as-needed basis, but only after consultation with the code owner.

Core tests assert observable state, persisted records, replay results, and may also check for patterns associated with performance regression.

## Terminal contracts

| ID | Terminal contract | Active evidence |
| --- | --- | --- |
| T01 | one Textual app accepts choice, review, EDGE, beam, page, prompt, and isolated chord requests across screen transitions | `test_terminal_lifecycle.py`, `test_chord.py` |
| T02 | engine-owned insertion previews run on the request owner; search warming runs on an executor and delivers generation-tagged results on the UI thread | `test_search_warm_terminal.py` |
| T03 | invalid choice commands remain editable; a submitted screen keeps a focused read-only command bar through owner-thread handoff and rejects queued input | `test_terminal_lifecycle.py`, `test_textual_migration_journey.py`, `test_textual_driver_pty.py` |
| T04 | EDGE blank Enter, Ctrl+C, and Ctrl+D preserve their command, interrupt, and cancellation values; the non-TTY path remains text-only | `test_terminal_lifecycle.py`, `test_terminal_scenarios.py` |
| T05 | Choice feedback navigation, search-lens rank selection, authored-text editing, context paging, and review reactivation retain their command results | `test_terminal_lifecycle.py` |
| T06 | beam shortcuts, selection, stochastic score formatting, and prompt modes return the requested values | `test_terminal_lifecycle.py` |
| T07 | ordinary rank browsing does not mutate speculative backend state before engine commit | `test_selection_warm_terminal.py` |
| T08 | row selection preserves immediate command focus across repeated clicks; Beam uses remaining height; a POSIX runtime journey applies Choice, EDGE, prompt, Beam advance/select/return, and discards handoff input | `test_textual_migration_journey.py`, `test_textual_responsive_layout.py`, `test_textual_driver_pty.py` |
| T09 | captured output retains exactly the trailing 16,000 characters across oversized and incremental writes; an open or reopened viewer shows that history and preserves paging away from the tail | `test_terminal_lifecycle.py` |
| T10 | Choice renders the full prepared context tail by default, records cumulative rendered-character work as the tail grows, and does not jump to the tail while the user pages away | `test_terminal_lifecycle.py` |

## Sampler and action contracts — 8

| ID | Contract | Existing evidence to migrate |
| --- | --- | --- |
| S01 | `SamplerConfig` accepts, rejects, and round-trips core fields | `test_sampler_contracts.py` |
| S02 | deterministic categorical and Gumbel draws obey seed/tie rules | `test_sampler_contracts.py` |
| S03 | CFG evaluates `P + V[:n]` and `U + V[:n]` with exact shared continuation IDs; standalone guidance uses `add_bos=True, special=True` independently of primary representation. Only retained visible tokens count toward the cutoff; new/resumed/replayed/forked/switched states are equivalent. Guidance reuses evaluation on forward append and unchanged decisions, catches up lazily, and rebuilds once on divergence, prompt change, or ownership loss. No warming past cutoff. | `test_cfg_contracts.py`, `test_sampler_contracts.py` |
| S04 | history penalties change policy selection without changing raw rank | `test_sampler_contracts.py` |
| S05 | a group member biases only its final token after its exact token prefix matches; the rule is identical for words and phrases | `test_sampler_contracts.py` |
| S06 | finite case/spacing variants and overlapping group/direct-token sources produce an attributable sum | `test_sampler_contracts.py` |
| S07 | every core action and replay expectation serializes canonically | `test_sampler_contracts.py` |
| S08 | llama and Transformers backends consume the same core sampler contract | `test_vector_contracts.py`, sampler smoke tests |

## Engine action contracts — 9

| ID | Contract | Existing evidence to migrate |
| --- | --- | --- |
| E01 | accept/select commits exactly the selected token | `test_engine_contracts.py` |
| E02 | raw-rank selection addresses the full vocabulary | `test_rank_neighborhood.py`, `test_unexposed_ranks.py` |
| E03 | exact and ordinary writes produce the expected visible span | `test_engine_contracts.py` |
| E04 | phrase/check validates a complete span atomically | `test_engine_contracts.py` |
| E05 | force phrase commits through temporary bias without persistent residue | `test_engine_contracts.py` |
| E06 | holds respect count, boundary, stop, and teacher limits | `test_engine_contracts.py` |
| E07 | check and force are durable write actions, not menu-only events | `test_engine_contracts.py` |
| E08 | model EOG, teacher EOG, finite hold, and menu termination are distinct | `test_engine_contracts.py` |
| E09 | rejected or invalid actions leave token state unchanged | `test_engine_contracts.py` |

## Replay contracts — 10

| ID | Contract | Existing evidence to migrate |
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

| ID | Contract | Existing evidence to migrate |
| --- | --- | --- |
| L01 | resume reconstructs an open episode and continues it | `test_lifecycle_contracts.py` |
| L02 | rewind can stop at any retained token boundary | `test_lifecycle_contracts.py` |
| L03 | rewind can cut inside a checked or multi-token write | `test_lifecycle_contracts.py` |
| L04 | fork preserves exactly the requested visible prefix | `test_lifecycle_contracts.py` |
| L05 | fork/rewind preserve episode identity and parent lineage correctly | `test_lifecycle_contracts.py` |
| L06 | sampler transitions restore at the selected historical boundary | `test_lifecycle_contracts.py` |
| L07 | model/backend continuation preserves visible text and sampler results | `test_lifecycle_contracts.py` |

## Vocabulary and menu contracts — 5

| ID | Contract | Existing evidence to migrate |
| --- | --- | --- |
| M01 | full-vocabulary search is non-mutating | `test_menu_contracts.py` |
| M02 | absolute/relative rank navigation resolves the requested candidate | `test_menu_contracts.py` |
| M03 | menu commands distinguish editorial moves from token actions | `test_menu_contracts.py` |
| M04 | `l` and `L` expose sticky raw/model/gap logit views | `test_menu_contracts.py` |
| M05 | the CLI exposes only the installed core surface and supports reusable profiles | `test_menu_contracts.py`, `test_controller_profiles.py` |

## Vector and backend contracts — 5

| ID | Contract | Existing evidence to migrate |
| --- | --- | --- |
| V01 | core loads an external JSON vector without model metadata requirements | `test_vector_contracts.py` |
| V02 | core loads a cvector with the exact canonical layer ordering | `test_vector_contracts.py` |
| V03 | malformed or dimensionally unusable artifacts fail clearly | `test_vector_contracts.py` |
| V04 | model labels do not gate vector application | `test_vector_contracts.py` |
| V05 | output vectors use backend math and runtime capabilities | `test_vector_contracts.py` |

## SQLite and export contracts — 4

| ID | Contract | Existing evidence to migrate |
| --- | --- | --- |
| P01 | episode creation and action results persist and reload | `test_persistence_contracts.py` |
| P02 | the persisted tape contains only replayable actions and optional results | `test_persistence_contracts.py` |
| P03 | fork maps represent exact visible boundaries, including zero | `test_persistence_contracts.py` |
| P04 | episode/projector export preserves procedure and lineage semantics | `test_persistence_contracts.py` |

## Property and fuzz contracts — 5

| ID | Contract | Existing evidence to migrate or generate |
| --- | --- | --- |
| Q01 | generated core sampler records round-trip exactly | `test_property_contracts.py` |
| Q02 | replay never changes the recorded prefix | `test_property_contracts.py` |
| Q03 | rewind followed by replay reproduces the retained prefix | `test_property_contracts.py` |
| Q04 | fork at N preserves exactly the first N visible tokens | `test_property_contracts.py` |
| Q05 | generated divergence terminates at the live edge or continues ballistically | `test_property_contracts.py` |

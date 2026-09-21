# Core contract matrix

This is the budget and source of truth for the reduced core test suite. The
numbers below are contract slots, not a limit on parametrized examples. A
slot may cover many actions, sampler configurations, boundaries, or backend
implementations.

Every retained core test must map to one of these slots. Tests with no slot
are moved to `vectors` or the archive, deleted as obsolete, or added here only
after deliberately changing the contract budget.

Core tests assert observable state, persisted records, and replay results.
Caching is deliberately outside this contract: tests do not assert cache hits,
cache misses, `eval()`/`reset()` counts, SQL query counts, or any other
recomputation strategy. If performance work needs measurement, it belongs in
an optional benchmark or profiler check rather than a behavioral test.

## Sampler and action contracts — 8

| ID | Contract | Existing evidence to migrate |
| --- | --- | --- |
| S01 | `SamplerConfig` accepts, rejects, and round-trips core fields | `test_sampler_contracts.py` |
| S02 | deterministic categorical and Gumbel draws obey seed/tie rules | `test_sampler_contracts.py` |
| S03 | CFG applies only to its configured prefix and records core state | `test_sampler_contracts.py`, `test_llama_release_smoke.py` |
| S04 | history penalties change policy selection without changing raw rank | `test_sampler_contracts.py` |
| S05 | naive multi-token bias credits only the final entered edge | `test_sampler_contracts.py` |
| S06 | conditional bias activates after its trigger and stops at its terminator | `test_sampler_contracts.py` |
| S07 | every core action and replay expectation serializes canonically | `test_sampler_contracts.py` |
| S08 | llama and Transformers backends consume the same core sampler contract | `test_vector_contracts.py`, sampler smoke tests |

## Engine action contracts — 10

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
| E09 | budgets checkpoint without silently ending the episode | `test_engine_contracts.py` |
| E10 | rejected or invalid actions leave token state unchanged | `test_engine_contracts.py` |

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

## Persistence and lifecycle contracts — 8

| ID | Contract | Existing evidence to migrate |
| --- | --- | --- |
| L01 | resume reconstructs an open episode and continues it | `test_lifecycle_contracts.py` |
| L02 | rewind can stop at any retained token boundary | `test_lifecycle_contracts.py` |
| L03 | rewind can cut inside a checked or multi-token write | `test_lifecycle_contracts.py` |
| L04 | fork preserves exactly the requested visible prefix | `test_lifecycle_contracts.py` |
| L05 | fork/rewind preserve episode identity and parent lineage correctly | `test_lifecycle_contracts.py` |
| L06 | sampler transitions restore at the selected historical boundary | `test_lifecycle_contracts.py` |
| L07 | budget state follows the retained boundary and explicit renewal | `test_lifecycle_contracts.py` |
| L08 | model/backend continuation preserves visible text and sampler results | `test_lifecycle_contracts.py` |

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
| V01 | core loads an external JSON vector without provenance requirements | `test_vector_contracts.py` |
| V02 | core loads a cvector with the exact canonical layer ordering | `test_vector_contracts.py` |
| V03 | malformed or dimensionally unusable artifacts fail clearly | `test_vector_contracts.py` |
| V04 | core records available vector metadata without inventing provenance | `test_vector_contracts.py` |
| V05 | backend application uses the common capability boundary for llama/Transformers | `test_vector_contracts.py` |

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

## Migration rule

The existing suite is source material, not a preservation obligation. Migrate
the strongest assertions into these slots, parameterize their meaningful
variants, and delete assertions that concern removed APIs, private helpers,
cache strategy, SQL access patterns, or speculative extension behavior. New
tests are added only for a missing contract or a demonstrated mutation/branch
coverage gap.

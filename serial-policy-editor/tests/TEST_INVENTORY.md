# Test inventory and reduction plan

This is the first-pass inventory for the test reduction. It is deliberately
file-oriented so the old suite can be sorted before individual tests are
deleted. Files marked `split` must receive a function-level disposition before
the old module is removed.

Before the bucket split, the checkout had 74 test modules, 970 tests collected,
and 8 collection errors. The active split has 18 test modules: 128 core tests
and 32 vector tests. The 11-module backend/UI integration bucket and the
15-module experimental bucket are archived under `archive/tests/` and no
longer collected by default. All active test buckets collect without errors;
coverage and mutation tooling are not yet configured in the test environment.

The active package buckets now exist under `tests/core` and `tests/vectors`;
archived experimental and integration tests live under
`archive/tests/research` and `archive/tests/integration`. Shared fakes and
numerical references remain at the `tests` root. The first obsolete-test purge
removed the
compatibility modules that asserted retired sampler recovery, replay override,
replay-plan, and sampler-segment implementation details. Current
rewind/sampler state is covered by the core lifecycle bucket.

The executable core reduction is governed by
[`CORE_CONTRACTS.md`](CORE_CONTRACTS.md), which defines exactly 55 contract
slots and the allowed scope of each one.

Cache/recompute policy: core tests assert observable state and results only.
They must not require a particular `eval`, `reset`, branch, cache-hit, cache
miss, or decoding count after rewind, fork, resume, or replay. Cache behavior
is not a required correctness suite; any performance measurement belongs in an
optional benchmark or profiler check.

## Core

These tests either protect the core runtime contract or contain a small core
slice that should be rewritten into the reduced suite.

| Existing module | Disposition | Core destination |
| --- | --- | --- |
| `test_boundaries.py` | deleted | boundary cases are covered by E06 and R10 |
| `test_cli_surface.py` | deleted | clean-install surface is covered by M05 and the install smoke |
| `test_concrete_teacher_ranks.py` | deleted | rank/evidence cases are covered by E01 and replay contracts |
| `test_continuation_spacing.py` | deleted | its token-write cases are now E03; removed CLI-flag assertion was obsolete |
| `test_controller_profiles.py` | reduce | core profile round-trip |
| `test_core_actions.py` | retain/rewrite | canonical actions and `SamplerConfig` boundary |
| `test_core_install_surface.py` | retain | core-only install smoke |
| `test_edge_replay_continuation.py` | deleted | its CLI replay cases are now the compact replay-until boundary |
| `test_engine_contracts.py` | retain | ten engine-action contract slots |
| `test_episode_projector.py` | deleted | fork map and export cases are now persistence contracts |
| `test_episode_runtime.py` | deleted | engine actions, replay, and lifecycle cases are now contract tests |
| `test_phrase_actions.py` | deleted | phrase, check, force, and rewind cases are now engine/replay/lifecycle contracts |
| `test_observation_contract.py` | deleted | observation cases are now S04 and M04 |
| `test_plain_projection.py` | deleted | basic export is now P04 |
| `test_procedure_view.py` | deleted | procedure/export presentation is now P04 |
| `test_rank_neighborhood.py` | deleted | its meaningful rank cases are now M02 |
| `test_replay_contracts.py` | retain | replay divergence, handoff, ballistic, and stop semantics |
| `test_replay_until.py` | retain/rewrite | live-edge cutoff and handoff |
| `test_runtime_setup.py` | retain/rewrite | compact core setup/menu boundary |
| `test_sampler_contracts.py` | retain | eight sampler/action contract slots |
| `test_unexposed_ranks.py` | deleted | its meaningful rank cases are now M02/E02 |
| `test_menu_contracts.py` | retain | five vocabulary/menu contract slots |
| `test_lifecycle_contracts.py` | retain | eight persistence/lifecycle contract slots |
| `test_property_contracts.py` | retain | five generated/property contract slots |
| `test_persistence_contracts.py` | retain | four SQLite/export contract slots |
| `test_vector_contracts.py` | retain | five vector/backend contract slots |

## Archived backend and UI integration

These tests remain as historical evidence but are not part of the active test
suite. They may require heavy backend imports, a real model, a terminal, or a
platform-specific runtime.

| Existing module | Disposition | Destination |
| --- | --- | --- |
| `test_fullscreen_layout.py` | archived | responsive live-menu integration |
| `test_context_input.py` | archived | editor and context rendering integration |
| `test_edge_tui.py` | archived | live-edge presentation integration |
| `test_kv_quantization.py` | archived | llama loading-option configuration |
| `test_menu_expansion.py` | archived | large-menu rendering and expansion integration |
| `test_llama_sampler_smoke.py` | archived | opt-in llama sampler/backend smoke |
| `test_persistent_tui.py` | archived | persistent terminal UI integration |
| `test_persistent_tui_smoke.py` | archived | opt-in real-model UI smoke |
| `test_terminal_pty.py` | archived | PTY ownership/restoration integration |
| `test_transformers_sampler_smoke.py` | archived | opt-in Transformers sampler/backend smoke |
| `test_llama_release_smoke.py` | archived | opt-in llama worker/release smoke |

## Optional vectors

The user-facing vector package is optional and is not part of the core test
count. These tests cover conventional hidden-state steering-vector
production or the portable artifact loader/application boundary. It owns the
stronger vector correctness rules: construction details, provenance policy,
model identity, layer/site semantics, and rejection of artifacts that are not
safe for its production/analysis workflows.

| Existing module | Disposition | Destination |
| --- | --- | --- |
| `test_activation_vectors.py` | split | core permissive loading; optional-vector production/inspection and strict artifact policy |
| `test_vector_loading.py` | deleted | its cvector loader case is now V02 |
| `test_llama_hidden_state.py` | move/rewrite | `vectors` backend production tests |
| `test_llama_worker.py` | move/rewrite | `vectors` worker/production tests |
| `test_transformers_hidden_state.py` | split | core backend capability tests; `vectors` production tests |
| `test_transformers_worker_interop.py` | move/rewrite | `vectors` opt-in conformance tests |
| `test_vector_command_boundary.py` | move/rewrite | core/vector command boundary, with vector-only strictness |

## Archived experimental tests

These tests are retained as historical experimental material under
`archive/tests/research`. They are not required by the core build and are not
collected by default.

| Existing module | Disposition | Destination |
| --- | --- | --- |
| `test_activation_episode_pairs.py` | archived | research-derived vector analysis |
| `test_bias_catalog.py` | archived | research catalog/compiler tests |
| `test_bias_rules.py` | archived | research catalog/route compiler |
| `test_controller_stack_regressions.py` | archived | research control traces |
| `test_group_control.py` | archived | research group-control tests |
| `test_observation_statistics.py` | archived | research controls and priors |
| `test_online_learning.py` | archived | research learner tests |
| `test_research_command_parsing.py` | archived | research CLI parsing |
| `test_research_setup.py` | archived | research setup/configuration |
| `test_token_preference.py` | archived | research token-preference tests |
| `test_trajectory_compare.py` | archived | research comparison/analysis |
| `test_vector_impact.py` | archived | research vector-impact analysis |
| `test_write_learning.py` | archived | research write-learning tests |
| `test_research_adapter.py` | archived | historical wide-record adapter check |
| `test_vector_cli.py` | archived | token-preference vector workbench |

The removed compatibility modules were not protecting the replay contract:
they asserted retired CLI overrides, sampler-record repair prompts, SQL/cache
access patterns, and extension-wide sampler transitions. Old records remain
loadable through the core record projection where their replayable fields are
understood; unsupported extensions terminate at the live edge.

## Duplicate or obsolete material

No whole module is being declared a duplicate without a function-level check.
The following are the first known obsolete or replacement targets:

- extension CLI-flag assertions embedded in `test_commands.py` and
  `test_runtime_setup.py`;
- core construction calls for removed prompt-pair/vector helpers in
  `test_activation_vectors.py`, `test_llama_worker.py`, and
  `test_transformers_hidden_state.py`;
- root vector-CLI forwarding assumptions in `test_vector_command_boundary.py`;
- learner/controller internals that are already covered by the archived
  adapter or are no longer part of the supported runtime;
- repeated parser/round-trip cases that duplicate the sampler and replay
  contract tests.

The reduced core suite will be accepted only after the function-level pass has
assigned every deleted test to one of: retained contract, moved package,
compatibility edge, duplicate, or obsolete.

## Vector correctness boundary

The same artifact can therefore have two different test responsibilities:

| Concern | Core | Optional vectors/archive |
| --- | --- | --- |
| Recognize supported artifact format | yes | yes |
| Read vector values and available metadata | yes | yes |
| Preserve optional provenance when present | yes | yes |
| Require provenance to load | no | only where the producing/analysis workflow requires it |
| Enforce model identity or layer/site alignment before loading | no | yes, for workflows that make that guarantee |
| Reject an unknown-provenance artifact | no, not by itself | yes, where the package contract requires provenance |
| Apply a vector and report backend incompatibility | yes | yes |
| Create, compare, blend, or analyze vectors | no | vectors or archive |

Core tests should include an externally produced, provenance-light vector and
an intentionally layer-mismatched artifact to prove that core loading remains
permissive. Optional-vector tests may separately assert strict provenance or
alignment rules; those assertions must not leak back into the core install.

### Dependency-light cvector correctness test

The normal vector suite should not import Transformers, Accelerate, CUDA, or a
real model merely to test cvector layer interpretation. Build a tiny synthetic
GGUF/cvector fixture directly with the standard library:

1. encode distinct sentinel directions for each native layer;
2. load the fixture with `from_cvector_path`;
3. assert the exact flattened canonical vector, including the leading
   non-steerable slot;
4. apply the loaded artifact to a recording control backend and assert the
   exact canonical `layer_start`, `layer_end`, and vector payload;
5. parametrise malformed structural cases separately (missing layer, duplicate
   layer, width mismatch, bad tensor offset, non-finite value, and inconsistent
   layer count).

The existing hand-written cvector fixture is already close to this design: its
distinct `(1, 2, 3)` and `(4, 5, 6)` directions catch ordering mistakes. It
should be extracted from the mixed legacy vector test and extended with the
recording-backend assertion, which makes an off-by-one or reordered layer
interpretation fail directly. The real Transformers/Accelerate test remains a
slow, opt-in cross-backend conformance test rather than a prerequisite for
ordinary vector-package correctness.

## Measurement gate

Before and after the purge, record:

1. core branch coverage for the core-only install;
2. mutation score for core sampler, engine, replay, lifecycle, persistence,
   and vector-loading modules;
3. the same focused measurements for the vectors package and any future
   optional package where its contracts remain supported;
4. collection/import failures in each isolated installation.

Coverage and mutation dependencies are not currently installed, so adding the
measurement command/configuration is part of the next test-harness step rather
than silently treating the current 970-test collection as a quality baseline.

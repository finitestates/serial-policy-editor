# Core runtime scope

This project is an interactive episode projector: a menu-driven environment
for sequential token selection, replay, rewind, and branching.

## Main-program capabilities

The regular installation should support:

- episode workspaces, replay/resume/fork, rewind, and fork maps;
- full-vocabulary search and token inspection;
- llama.cpp and Transformers backends;
- sampler filtering and draw methods, including CFG and Gumbel draws;
- naive and conditional bias rules, including phrase/sequence credit rules;
- history penalties and check/force actions;
- raw/model-logit display, including the raw-rank-1 model-gap view;
- optional loading and application of externally produced vectors;
- episode export through `projector`.

SQLite is part of the workspace implementation, but it is not the definition
of an episode. The runtime owns episode/action meaning; persistence adapts that
meaning to the workspace.

## Archived extension capabilities

These do not determine the dependency graph or setup surface of the active
program:

- post-output vector creation/analysis and experimental vector work;
- online/group learning and token-preference learning;
- reference priors and YAML reference weights;
- bias catalogs and catalog-generation tools;
- experimental observers, diagnostics, and comparison/reporting paths.

Vector loading is different from vector production. Core accepts externally
produced steering artifacts and records their available metadata. The
optional vectors package adds conventional hidden-state vector production and
inspection; post-output and experimental vector work stays archived.
Producer metadata is useful but not required, and core does not infer layer
alignment from it.

## Replay contract

Core replay identity is deliberately small: a step number, the teacher action,
and an optional recorded result used to detect divergence. Divergence policy is
runner behavior, not a per-step record field; ballistic replay therefore has no
extra mode/result slot. Evidence, learner diagnostics, source provenance, and
sampler-transition metadata may support persistence or inspection, but they are
not part of the core action contract. If a historical feature cannot reconstruct
the result needed for replay, execution yields to the edge menu.
Replay plans may inherit the source sampler schedule by default, but fixed and
counterfactual replay can provide a different schedule or no source schedule
at all; that choice belongs to the plan, not to each tape step.

## Migration rules

1. New episode concepts belong under `trajectory_editor.core`.
2. Persistence, UI, concrete backends, and optional extensions depend on core;
   core does not import them.
3. Old module names remain as small compatibility shims during migration.
4. A rename is complete only when ownership and dependency direction change;
   a re-export alone is an intermediate step.
5. Existing behavior is the oracle while the architecture moves. Tests for
   removed extension capabilities are retired when those capabilities leave
   the active program.

## Core-only acceptance gate

The core distribution must be independently buildable and runnable without
vector-production or archived extension files. A clean core-only environment must be
able to import the package, run `policy-editor -h`, create and replay a minimal
episode with a supported backend, and export that episode through `projector`.
Optional entry points may be unavailable without their extensions; their
absence must not prevent the core command or core package from starting.

## First migration slice

The canonical action language now lives in `trajectory_editor.core.actions`.
The backend contract now lives in `trajectory_editor.core.backend`.
The replay-stable sampler kernels (candidate filtering, raw ranks, and
categorical/Gumbel draws) now live in `trajectory_editor.core.sampling`.
The core sampler contract now lives in `trajectory_editor.core.sampler_config`;
the old `SamplingConfig` is a wider compatibility record for historical episode
fields and is not part of the core install surface.
The authoritative replay/live runner now lives in
`trajectory_editor.episode_runner`; `episode_policy` is retained only in the
archive as historical extension code.
The installed `policy-editor` command advertises and accepts the core runtime
surface plus proper steering-vector loading. The core build contains the
artifact loader, while `policy-editor-vector` belongs to the optional vectors
build.
`episode_cli` uses `SamplerConfig` and core replay factories directly.
The live token ledger and branch cursor now begin their migration in
`trajectory_editor.core.trajectory.TrajectoryState`; `EpisodeEngine` forwards
its historical state attributes to this object for compatibility.
`episode_actions` and `episode_backend` remain compatibility import paths.
`projector` owns the typed episode projection implementation. The
steering/activation-vector command is supplied by the optional vectors build.
The package root and core `EpisodeRunner` do not eagerly import archived
learning, token-preference, comparison, or experimental vector extensions.

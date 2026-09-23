# Mutation testing

Mutation testing makes small changes to production code and checks whether the
selected tests notice. It is a way to find weak or missing assertions; the
mutation score is not a quality target by itself.

## Scope and pipeline

The pipeline mutates these modules in order, using `tests/core` as the test
selection at each stage:

1. `core/sampling.py`
2. `core/sampler_config.py`
3. `episode_history.py`
4. `episode_lineage.py`
5. `episode_replay_source.py`
6. `episode_lifecycle.py`
7. `episode_engine.py`

The list lives in [`setup.cfg`](setup.cfg); the stage order and commands live
in [`scripts/mutation_pipeline.sh`](../scripts/mutation_pipeline.sh). The
runner mutates a private copy of `core/src` inside `.mutmut-workspace/`, while
using the repository's tests. This keeps `mutmut apply` and browsing away from
the working source tree. The workspace and mutmut cache are generated and
ignored by Git.

## Set up and run

From the repository root, create the local environment and install the core
test dependencies and pinned mutmut version:

```bash
python3 -m venv .venv
(cd core && ../.venv/bin/python -m pip install -e '.[test]')
.venv/bin/python -m pip install 'mutmut==3.8.0'
```

If `.venv` already has the project test dependencies, only the last install
may be needed. Keep mutmut pinned when comparing runs; upgrading it can change
the generated mutants and result cache.

Run the full pipeline, or choose one stage while investigating:

```bash
scripts/mutation_pipeline.sh
scripts/mutation_pipeline.sh episode_lineage
```

The available stage names are `sampling`, `sampler_config`,
`episode_history`, `episode_lineage`, `episode_replay_source`,
`episode_lifecycle`, and `episode_engine`. With no argument, the script runs
all seven in order. To limit parallel workers:

```bash
MUTMUT_MAX_CHILDREN=2 scripts/mutation_pipeline.sh
```

Review results in the terminal or interactive browser:

```bash
scripts/mutation_pipeline.sh results
scripts/mutation_pipeline.sh browse
```

For a specific mutant, copy its full name from the result and inspect the
change and the tests mutmut associates with it:

```bash
(cd .mutmut-workspace && ../.venv/bin/mutmut show 'FULL_MUTANT_NAME')
(cd .mutmut-workspace && ../.venv/bin/mutmut tests-for-mutant 'FULL_MUTANT_NAME')
```

Mutmut caches verdicts in `.mutmut-workspace/`. It reuses that work on later
runs. If tests or mutation configuration have changed and a result looks
stale, remove this generated workspace and rerun to start with a clean cache:

```bash
rm -rf .mutmut-workspace
scripts/mutation_pipeline.sh episode_lineage
```

This removes only the ignored mutation workspace and its cached results.

## How to triage a result

**Not every mutant is a boundary worth fixing. Investigate each one and reason
about whether it changes behavior the program promises.** Do not add tests just
to raise the kill rate, and do not treat a survivor as a bug without checking
the contract.

For each interesting survivor:

1. Read the source change and the associated tests (`show` and
   `tests-for-mutant`, or `browse`).
2. Decide what observable behavior the mutant changes and whether that behavior
   is part of the module's contract.
3. If it violates an intended contract, add a focused regression test for that
   behavior, then fix the implementation if needed.
4. If it is equivalent, unreachable under supported inputs, or changes only an
   internal detail with no user-visible or caller-visible effect, leave it
   surviving. Record the reason when the case is likely to be revisited.
5. Rerun the relevant stage and confirm the regression test kills the mutant.

Interpret result categories carefully:

- **Killed:** at least one selected test failed with the mutation. This is
  useful evidence, but does not prove every nearby boundary is covered.
- **Survived:** selected tests passed. This may be a missing assertion, an
  equivalent mutation, or behavior outside the supported contract; inspect it
  before acting.
- **No tests:** mutmut did not associate a selected test with the mutant. This
  is not, by itself, evidence that the code is dead or that a new test is
  worthwhile.
- **Timeout:** the mutated version exceeded its time limit. Check whether the
  change causes nontermination or only makes a path substantially slower.

Good mutation tests describe meaningful edges rather than mirror an operator
change. Useful lineage regressions include equal-key siblings using episode ID
as the tie-breaker (`a-child` before `z-child`); creation key taking priority
over an oppositely sorted ID (`z-created-earlier` before `a-created-later`),
for ordinary and replay-derived forks; and mixed `None`/`False` creation keys
sorting deterministically without raising. A separate contract decision from
replay triage: a replay can remain related to its family through
`spr_source_id` when ordinary provenance metadata is missing. That resilience
is intentional. Other reviewed boundaries include strict finite-positive
`SamplerConfig.bias_step` validation, exact versus over-limit phrase lengths,
refusing observations after terminal/checkpoint states,
`EpisodeHistory.truncate` boundary validation, and replay source selection
through a requested boundary.

## Keeping the scope focused

If another production module becomes a good candidate, add it to both
`only_mutate` in `setup.cfg` and the stage mapping in
`scripts/mutation_pipeline.sh`. Routine unit tests should remain the fast
default; run the full mutation pipeline when reviewing the selected contracts
or when its findings justify the extra time.

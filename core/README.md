# Policy Editor Core

This subproject contains the standalone runtime: the menu-driven episode
editor, replay/resume/fork/rewind behavior, persistence, supported model
backends, core sampler, bias rules, and episode projector.

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
with an `EpisodeEngine`. It records action tape, outcomes, sampler/budget
control points, branch lineage, and rewind/fork state without importing or
requiring `EpisodeStore`. A fork is a lightweight session event: it retains a
reconstructible token prefix and reactivates on the one loaded backend. An
optional backend cache snapshot can make that faster, but is never branch
identity. Use `LiveSession.branch_handle(...)` for a branch-bound view.

The terminal editor exposes the same lifecycle with `--ephemeral`; it does
not open the default workspace (or create one) during a normal run:

```bash
policy-editor --ephemeral --model model.gguf --new-prompt 'Tell a story'
policy-editor --ephemeral --model model.gguf --teacher-plan plan.jsonl --new-prompt 'Tell a story'
```

At EDGE, `fork N` creates and selects a live branch, `rewind N` changes the
selected branch, and `branches`/`switch ID` navigate retained branches.
`export FILE` writes the selected portable teacher tape without opening a
workspace. `save WORKSPACE [ID]` materializes just that branch as one durable
episode; `save-family WORKSPACE [ROOT_ID]` materializes all retained branches
and their lineage. `quit` discards the entire in-memory session. Unless
`--plain-ui` is passed, `--ephemeral` uses the same live terminal interface as
ordinary episodes.

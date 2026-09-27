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

The terminal editor runs new sessions in memory by default and does not create
a workspace during a normal run. Pass `--workspace` to open or create the
default `episodes.sqlite3`, or pass `--workspace PATH` to choose another file.
Saved-episode operations such as resume, replay, fork-from, and listing require
a workspace:

```bash
policy-editor --model model.gguf --new-prompt 'Tell a story'
policy-editor --model model.gguf --teacher-plan plan.jsonl --new-prompt 'Tell a story'
policy-editor --workspace --model model.gguf --new-prompt 'Tell a story'
```

At EDGE, `fork N` creates and selects a live branch, `rewind N` changes the
selected branch, and `branches`/`switch N` navigate retained branches.
`export FILE` writes the selected portable teacher tape without opening a
workspace. `save [WORKSPACE [ID]]` materializes just that branch as one durable
episode; without a workspace argument it uses the `--workspace` selection or
defaults to `episodes.sqlite3`. `save-family [WORKSPACE [ROOT_ID]]` materializes
all retained branches and their lineage using the same workspace selection.
`quit` and process exit discard unsaved in-memory history. The live terminal
interface is the same whether or not a workspace is open.

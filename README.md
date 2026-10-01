# Serial Policy Editor

For the required validation workflow, commands and evidence index, see
[Terminal rendering validation and handoff](tests/TERMINAL_RENDERING_GUIDE.md).

Serial Policy Editor is a terminal editor for steering a local language model
one token, text insertion, or delegated span at a time. Episodes can be
replayed, rewound, forked, searched, and exported through the projector.

The active project is intentionally small:

- `core/` — the standalone runtime and `policy-editor` command;
- `vector/` — optional conventional activation/steering-vector production.

## Get only the files you need

These commands use Git sparse checkout. They leave the repository metadata and
top-level documentation available, but omit unrelated package trees and their
files from the working tree.

### Core only

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/finitestates/serial-policy-editor.git serial-policy-editor-core
cd serial-policy-editor-core
git sparse-checkout set core
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ./core
```

### Core plus vectors

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/finitestates/serial-policy-editor.git serial-policy-editor-vector
cd serial-policy-editor-vector
git sparse-checkout set core vector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ./core ./vector
```

If the repository has already been cloned, the essential commands are simply:

```bash
git sparse-checkout init --cone
git sparse-checkout set core                 # core only
# or: git sparse-checkout set core vector     # core plus vector tools
```

## Backend installation

All commands below are run from the repository root with the virtual
environment active. The base core install is model-free; choose the backend
extra that matches the model you intend to run.

```bash
# llama.cpp / GGUF
python -m pip install './core[llama]'

# Basic Transformers inference, CPU or GPU
python -m pip install './core[transformers]'

# Transformers plus Accelerate
python -m pip install './core[transformers-accelerate]'
```

The ordinary `transformers` extra does not install Accelerate. For the
explicit GGUF cross-backend conformance path, use:

```bash
python -m pip install './core[transformers-gguf]'
```

That path includes the GGUF reader and Accelerate. The optional bitsandbytes
path is available as `./core[transformers-bnb]`.

For the vector package, install core first with the backend it needs, then
install `vector`:

```bash
python -m pip install './core[llama]' ./vector
```

`policy-editor-vector` does not add Transformers, Torch, Accelerate, or
CUDA-related dependencies by default.

## 1.0 support and compatibility

The packages require CPython 3.10 or newer. The CI test matrix covers Python
3.10 and the latest CPython on Ubuntu Linux (`ubuntu-latest`). Ubuntu Linux is
the supported operating-system target for 1.0; macOS and Windows are not in the
release test matrix.

Core includes adapters for llama.cpp models in GGUF format and local
Transformers model directories. The core install itself remains model-free;
install the optional backend extra that matches the model. CPU/GPU availability
and native package installation depend on the selected backend and hardware.
Vector production uses Transformers, while the vector package can import
llama.cpp cvector GGUF artifacts.

The supported Python API is the public classes, functions, signatures, and
documented behavior exported by `trajectory_editor` and `trajectory_editor.core`
in their `__all__` lists. The supported CLI surface is the documented options
and command forms for `policy-editor` and
`policy-editor-vector`, including the documented teacher-plan and profile
formats. Within the 1.x series, incompatible changes to these documented
interfaces require a major-version change. Internal module paths, undocumented
aliases, and terminal layout details are not compatibility promises.

| File format | Use |
| --- | --- |
| JSONL | Existing portable teacher tapes |
| JSON | Optional teacher-plan envelope and portable steering-vector artifacts |
| YAML | Executable teacher plans and reusable CLI/controller profiles |
| SQLite | Explicit episode workspaces for save, resume, and saved-episode operations |
| GGUF | llama.cpp models and importable llama.cpp cvector artifacts |

SQLite workspaces are versioned application data, not an interchange format.
The current runtime rejects previous workspace schemas without modifying the
existing files; automatic migration is not part of the 1.x compatibility
promise. Check the changelog before reusing a workspace after an upgrade.

## Start the editor

With llama.cpp:

```bash
policy-editor \
  --backend llama.cpp \
  --model /path/to/model.gguf \
  --new-prompt 'Once upon a time'
```

Episodes run without a global token budget. Use `h N` for a finite delegated
span. Change settings at any decision with `s key=value`, replace the full
sampler configuration with `s {JSON}`, or change the draw seed with
`reroll [SEED]`. These are recorded in the replayable action sequence, so
rewind and fork restore settings from the retained action prefix.

With a local Hugging Face model directory:

```bash
policy-editor \
  --backend transformers \
  --model /path/to/model-directory \
  --new-prompt 'Once upon a time'
```

Running `policy-editor` without an episode source opens the normal initial
prompt flow. Reusable CLI values can be kept in a YAML profile and combined
with explicit flags; explicit flags win:

```yaml
model: /path/to/model.gguf
backend: llama.cpp
temperature: 0.80
top-k: 40
no-policy-view: true
```

Use `top-k: null` or `--top-k none` to disable only top-k. The
`--unfiltered` flag sets temperature to 1 and disables top-k, top-p, min-p,
typical-p, and tail-free filtering.

`--gumbel-top-k N` enables Gumbel-Max and limits the candidate menu to the top
N Gumbel-ranked choices from the active filtered candidate set. The menu starts
in Gumbel order; the first row remains the proposal, and displayed model ranks
stay as the candidates' selection addresses. `--gumbel-top-k none` removes this
menu limit.

By default, each candidate's Gumbel noise is addressed by token ID. Set
`--gumbel-noise-address model-rank` to address it by the candidate's one-based
full-vocabulary model rank instead. The option enables Gumbel-Max unless a draw
kernel is explicitly selected. Candidate filtering and each token's displayed
model rank stay as before, but the changed noise assignment can change which
rank wins. Candidates at the same model rank share the same Gumbel variate
across candidate sets if the seed, stream fingerprint, and boundary stay fixed.
`--gumbel-noise-scale SCALE` selects Gumbel-Max unless another draw kernel is
explicitly chosen, and scales perturbations after filtering. A scale of `1` is
standard Gumbel-Max; `0` removes noise. Larger values strengthen the
perturbation while leaving the active candidate set fixed. On a fixed set,
changing the scale is equivalent to changing the scores' effective temperature.
This separates noise strength from temperature-dependent filters that can
change candidate eligibility.

For Gaussian-noise argmax, use `--draw-kernel gaussian-max` with the optional
`--gaussian-noise-std` (default `1.0`). The standard deviation is applied to
temperature-scaled policy scores; random noise is keyed for deterministic
replay. This kernel defines a different draw distribution from softmax sampling.

Four additional perturb-and-argmax kernels share the `--perturb-noise-std`
control (default `1.0`): `logistic-max`, `student-t-max`, `laplace-max`, and
`uniform-max`. The Student-t kernel defaults to 3 degrees of freedom and
accepts any finite positive value via `--student-t-df` (also available as the
live `s student_t_df=...` setting). Student-t variates are divided by
`sqrt(3)` so the df=3 setting preserves the existing unit-variance behavior;
at other df values `perturb_noise_std` is a scale multiplier, not generally a
standard deviation. In particular, Student-t variance is infinite at df <= 2.
Logistic, Laplace, and Uniform noise are unit variance before applying the
standard-deviation control. Uniform noise is bounded, with half-width
`sqrt(3) * perturb_noise_std`; a score gap greater than or equal to twice that
half-width cannot be overcome. Like Gaussian-Max, perturbations are keyed for
deterministic replay and only rank candidates left by temperature scaling and
filtering.

```bash
policy-editor --profile profile.yaml --new-prompt 'Once upon a time'
```

The editor supports sequential token selection, full-vocabulary search, text
writes, check/force actions, chord previews, reusable profiles, bias and history
controls, CFG and perturb-and-argmax draws, rerolls, raw/model/gap logit views, teacher
replay, rewind, fork, and live-edge continuation.

On a usable TTY, the live editor draws full-screen views for choices, the live
edge, beam branches, and prompts. Press Ctrl+K to search command templates, F1
for the full command list, and Ctrl+L for captured output. Choose a theme with
`--theme` (`amber-cyan`, `chill`, `ink`, `monochrome`, `high-contrast`) or set a
default with `SPE_THEME=chill`. Piped and noninteractive runs keep the plain
terminal interface.

## Vectors

Install the optional vector package when you want to create or inspect
conventional hidden-state steering vectors:

```bash
python -m pip install './core[transformers]' ./vector
policy-editor-vector create \
  --model /path/to/model-directory \
  --prompt-a 'I am calm.' \
  --prompt-b 'I am angry.' \
  --layer-start 2 \
  --layer-end 2 \
  --output calm-vs-angry.json
policy-editor-vector inspect calm-vs-angry.json
```

The core editor can load an externally produced vector without the
vector package:

```bash
policy-editor \
  --backend llama.cpp \
  --model /path/to/model.gguf \
  --steering-vector calm-vs-angry.json \
  --new-prompt 'Hello'
```

The vector package also imports llama.cpp cvector GGUF files:

```bash
policy-editor-vector import-cvector control_vector.gguf \
  --output imported-cvector.json
```

Core preserves available artifact metadata but does not require producer
provenance or attempt to prove layer alignment from metadata alone. The
backend remains responsible for whether a vector can be applied.

## Replay and persistence

New sessions run in memory and do not create a database. Pass `--workspace`
(optionally followed by a path) to open or create an SQLite workspace for
saved-episode operations. At EDGE, `save` materializes the selected branch and
`save-family` materializes all retained branches; unsaved session history is
discarded when the process exits. Version 0.8.5 requires a fresh workspace:
previous-format workspace data is left untouched and is not migrated.

Replay tapes are storage-independent sequences of teacher actions with
optional expected results. Forks, rewinds, searches, and menus are editorial
moves, not tape entries. Handoff replay yields at the first divergence;
ballistic replay continues with teacher actions until the tape ends. Both modes
yield at the first unsupported action while preserving the remaining plan.

The `projector` command and the live edge expose plain-text, procedure, fork
map, and episode metadata views without making cache state part of the replay
contract.

## Active tests

The active suite covers the package contracts and the independent reference
kernel:

```bash
python -m pytest -q tests/core
python -m pytest -q tests/vectors
python -m pytest -q reference-kernel
```

Core tests cover the contracts documented in
[`tests/CORE_CONTRACTS.md`](tests/CORE_CONTRACTS.md). Vector tests cover artifact
interpretation, Transformers-based vector creation, cvector import, portable
artifact application, and optional production behavior. Real-model backend
checks are opt-in and skip when their local model is absent. The
`reference-kernel/` suite checks the standalone sampler/replay formulas and
their parity with the production runtime.

## Project documents

- [Core scope](CORE_SCOPE.md) — what belongs in the main runtime;
- [Cut notes](CUT_NOTES.md) — architectural decisions and migration notes;
- [Core package](core/README.md) — standalone core installation;
- [Vector package](vector/README.md) — optional steering-vector tooling;
- [Core contract matrix](tests/CORE_CONTRACTS.md) — the reduced core suite;
- [Reference kernel](reference-kernel/) — independent formulas and parity checks.

The version currently represented by the active package manifests is `0.8.5`.

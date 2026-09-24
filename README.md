# Serial Policy Editor

Serial Policy Editor is a terminal editor for steering a local language model
one token, text insertion, or delegated span at a time. Episodes can be
replayed, rewound, forked, searched, and exported through the projector.

The active project is intentionally small:

- `core/` — the standalone runtime and `policy-editor` command;
- `vector/` — optional conventional activation/steering-vector production;
- `archive/` — historical material, optional experiments, and retired tests.

The archive is not an install root and is not part of the normal test suite.

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

## Start the editor

With llama.cpp:

```bash
policy-editor \
  --backend llama.cpp \
  --model /path/to/model.gguf \
  --new-prompt 'Once upon a time' \
  --max-tokens 100
```

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

```bash
policy-editor --profile profile.yaml --new-prompt 'Once upon a time'
```

The editor supports sequential token selection, full-vocabulary search,
check/force actions, conditional and naive bias rules, CFG and Gumbel draws,
raw/model/gap logit views, replay, rewind, fork, and live-edge continuation.

## Vectors

Install the optional vector package when you want to create or inspect
conventional hidden-state steering vectors:

```bash
python -m pip install './core[llama]' ./vector
policy-editor-vector create \
  --model /path/to/model.gguf \
  --backend llama.cpp \
  --prompt-a 'I am calm.' \
  --prompt-b 'I am angry.' \
  --layer-start 2 \
  --layer-end 2 \
  --output calm-vs-angry.json
policy-editor-vector inspect calm-vs-angry.json
```

The core editor can load a compatible externally produced vector without the
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

Episodes are stored in an SQLite workspace. The replay tape is deliberately
small: a step number, teacher action, and an optional result used to detect
handoff divergence. Forks, rewinds, searches, and menus are editorial moves;
they are not replay tape entries. Handoff stops at the first divergence, while
ballistic replay continues with teacher actions and yields at the live edge
when the tape is exhausted.

The `projector` command and the live edge expose plain-text, procedure, fork
map, and episode metadata views without making cache state part of the replay
contract.

## Active tests

The active suite is divided by package boundary:

```bash
python -m pytest -q tests/core
python -m pytest -q tests/vectors
```

Core tests cover the 55 contract slots for sampling, actions, replay,
lifecycle, menus, vector loading, persistence, and generated cases. Vector
tests cover artifact interpretation, hidden-state capture, worker protocol,
and optional production behavior. Real-model backend checks are opt-in and
skip when their local model or worker is absent. The heavyweight cross-backend
worker check is documented separately under [optional cross-backend vector
conformance](#optional-cross-backend-vector-conformance).

### Optional cross-backend vector conformance

`tests/vectors/test_transformers_worker_interop.py` is intentionally not part
of the ordinary Transformers install or the default test run. It is an opt-in
check for people who want to compare a Transformers worker with a llama.cpp
GGUF worker. It requires a local model and worker command, plus the heavier
`./core[transformers-gguf]` extra, which includes GGUF support and Accelerate.
The normal `./core[transformers]` path does not install Accelerate and does not
need this test.

Historical tests and experimental material are retained under `archive/` and
are not collected by these commands.

## Project documents

- [Core scope](CORE_SCOPE.md) — what belongs in the main runtime;
- [Cut notes](CUT_NOTES.md) — architectural decisions and migration notes;
- [Core package](core/README.md) — standalone core installation;
- [Vector package](vector/README.md) — optional steering-vector tooling;
- [Core contract matrix](tests/CORE_CONTRACTS.md) — the reduced test budget;
- [Test inventory](tests/TEST_INVENTORY.md) — active and archived test buckets.

The version currently represented by the active package manifests is `0.7.5`.

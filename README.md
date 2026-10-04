# Serial Policy Editor

For the required validation workflow, commands and evidence index, see
[Terminal rendering validation and handoff](tests/TERMINAL_RENDERING_GUIDE.md).

This branch, `policy-editor-argmax`, experiments with a question: **if not the
argmax, then what?** The editor proposes the token with the highest score after
explicit policy adjustments and optional noise. You can accept that Argmax,
select another token by its raw model rank, write text, or delegate a finite
span. Episodes can be replayed, rewound, forked, searched, and exported.

Raw model argmax, adjusted argmax, and argmax after perturbation can differ.
The model logits stay available for full-vocabulary search even when proposal
eligibility is restricted. Selecting the best next score does not establish a
globally optimal sequence.

This is an incompatible experiment, not a validated 1.0 release. Package
manifests still say 1.0.0; the public records and workspace format have changed.
The inherited test harnesses and reference oracle await an explicit update.
See the [written contracts](tests/CORE_CONTRACTS.md) and
[documentation/testing checkpoint](docs/ARGMAX_DOCUMENTATION_PLAN.md).

The active project is intentionally small:

- `core/` — the standalone runtime and `policy-editor` command;
- `vector/` — optional conventional activation/steering-vector production.

## Get only the files you need

These commands describe the upstream checkout/install workflow. For this
experiment, use your local `policy-editor-argmax` checkout; the branch is not
assumed to be published upstream. These commands use Git sparse checkout. They leave the repository metadata and
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

## Platform and compatibility

The packages require CPython 3.10 or newer. The inherited CI matrix targets Python
3.10 and the latest CPython on Ubuntu Linux (`ubuntu-latest`). This experiment
has not passed that matrix. Ubuntu Linux is the supported platform target; macOS and Windows are not in the
release test matrix.

Core includes adapters for llama.cpp models in GGUF format and local
Transformers model directories. The core install itself remains model-free;
install the optional backend extra that matches the model. CPU/GPU availability
and native package installation depend on the selected backend and hardware.
Vector production uses Transformers, while the vector package can import
llama.cpp cvector GGUF artifacts.

The public Python surface is exported by `trajectory_editor` and
`trajectory_editor.core`. This experiment intentionally breaks the previous
API: `EligibleScores` replaces `SparseDistribution`, evidence uses nullable
`eligible_softmax`, and retired overlay/beam fields are removed. Compatibility
with the released 1.x interface is not claimed for this branch.

| File format | Use |
| --- | --- |
| JSONL | Existing portable teacher tapes |
| JSON | Optional teacher-plan envelope and portable steering-vector artifacts |
| YAML | Executable teacher plans and reusable CLI/controller profiles |
| SQLite | Explicit episode workspaces for save, resume, and saved-episode operations |
| GGUF | llama.cpp models and importable llama.cpp cvector artifacts |

SQLite workspaces are versioned application data, not an interchange format.
The current runtime rejects previous workspace schemas without modifying the
existing files; schema **3** is required on this branch and automatic migration is not
provided. Use a fresh workspace; old workspaces remain untouched.

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
sampler configuration with `s {JSON}`, or change the perturbation seed with
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
temperature: 1.0
draw-kernel: argmax
eligible-k: none
min-p: 0
```

```bash
policy-editor --profile examples/argmax.yaml --model /path/to/model.gguf \
  --new-prompt 'Once upon a time'
```

## Selection algorithm

At each boundary the engine:

1. Obtains full model logits, with configured guidance/steering.
2. Applies history penalties and explicit policy biases.
3. Divides adjusted scores by positive temperature.
4. Builds the eligible set using optional top-k and the min-p logit gap.
5. Adds the selected noise to all eligible scores, or only the leading
   `selective_noise_k` eligible scores ranked before noise.
6. Selects the highest final score. Exact ties use the lowest token ID.

Defaults are plain `argmax`, temperature `1`, top-k `none`, min-p `0`, and
selective-noise-k `none`: the full vocabulary is eligible and no noise is added.
Temperature zero instead reduces eligibility to one adjusted winner before
noise. Positive temperature alone cannot reorder plain argmax, but it changes
score gaps relative to fixed noise and an enabled min-p cutoff.

| Control | Meaning |
| --- | --- |
| `--eligible-k N` / `--top-k N` | Only the leading N adjusted-score tokens can be proposed |
| `--selective-noise-k N` | Perturb only the leading N eligible pre-noise scores; other eligible scores still compete |
| `--min-p P` | Keep scores within `-log(P)` of the eligible leader; `0` disables this logit-only cutoff |
| `--unfiltered` | Set temperature to 1 and disable top-k and min-p; noise remains configured |
| `--gumbel-top-k N` | Limit the displayed Gumbel-ranked menu; it does not define eligibility or noise membership |

With eligible-k 20 and selective-noise-k 5, only the leading 20 adjusted-score
tokens compete. Five receive noise; the other 15 keep their scores. With
unrestricted eligibility and selective-noise-k 5, original rank 6 in the
pre-noise eligible ordering can beat all five perturbed leaders. Lower untouched
tokens cannot beat that untouched leader. This probes robustness under the
chosen noise, not a universal measure of stability.

Use `s eligible_k=20 selective_noise_k=5 draw_kernel=gumbel-max` in the editor.
The stored eligibility field remains `top_k`. Full-vocabulary `/TERM` search and
raw-rank selection remain available outside the proposal set. Membership does
not guarantee that a particular token can win under bounded or selective noise.

### Noise and the Gumbel control

| Kernel | Active strength / construction |
| --- | --- |
| `argmax` | No perturbation |
| `gumbel-max` | `gumbel_noise_scale * Gumbel(0,1)` |
| `gaussian-max` | `gaussian_noise_std * Normal(0,1)` |
| `logistic-max` | Unit-variance Logistic noise times `perturb_noise_std` |
| `laplace-max` | Unit-variance Laplace noise times `perturb_noise_std` |
| `uniform-max` | Uniform noise on ±`sqrt(3) * perturb_noise_std` |
| `student-t-max` | `perturb_noise_std * t_df / sqrt(3)`, with `student_t_df > 0` |

All noise strengths default to 1 when their kernel is selected. Strength zero
returns the eligible adjusted argmax. Student-t's parameter is a scale
multiplier: it is a standard deviation at df 3, but not generally at other df.
Variance is infinite for 1 < df ≤ 2 and undefined for df ≤ 1. Its heavy tails
can promote tokens far below the leader, particularly over a large eligible
set. Gaussian noise has lighter tails; these are different experiments, not
unmeasured quality or speed guarantees.

Gumbel is the control: independent unit Gumbel noise on **every** eligible score
produces the softmax categorical winner distribution on that set. Selective
Gumbel noise does not retain that usual equivalence. No token CDF is computed.
Noise is addressed reproducibly by seed, root fingerprint, boundary, and token
ID; Gumbel optionally uses full-vocabulary model rank instead via
`--gumbel-noise-address model-rank`. A fixed set with full Gumbel noise and
positive scale has an effective noise temperature; changing the eligible set
or selective-noise membership is a separate change.

`reroll` records a new seed. `draw RAW_RANK` searches noise seeds that make a
target the proposal. Plain argmax ignores seed, so reroll cannot change its
winner and seed search is rejected. Some selective/bounded-noise targets are
unreachable; eligibility is not a promise of reachability.

Runnable configuration examples are in [`examples/`](examples/):
`argmax.yaml`, `gumbel-control.yaml`, `eligible-selective.yaml`, and
`gaussian-robustness.yaml`. Supply the model and prompt explicitly.

### Direct overlays and beam

The default table is **rank | token ID | text**. Overlay commands are submitted
with Enter; each shortcut toggles its own column group directly.

| Command | Effect |
| --- | --- |
| `l` | Raw model logits |
| `L` | Diff: token raw logit minus raw argmax logit |
| `~` | Added noise; untouched eligible scores show zero |
| `%` | Optional model/policy/eligible softmax diagnostics |
| `C` | Clear all diagnostic columns and restore three columns |
| `columns` | Show active overlays and shortcut guidance |
| `columns logit diff noise` | Select exactly those overlays |
| `overlay noise on` / `overlay noise off` | Explicitly set one overlay |
| `v` / `V` | Change row ordering / toggle policy-rank columns |

C does not change search, ordering, or proposal selection. The raw argmax's
diff is zero; lower raw logits have negative diff. Noise is a separate additive
change to the eligible pre-noise score. Excluded tokens have no noise value.

`model-softmax` normalizes full raw model logits; `policy-softmax` normalizes
full adjusted logits; `eligible-softmax` normalizes eligible, temperature-scaled,
**pre-noise** scores. They are optional diagnostics. Eligible softmax is not the
winner probability for arbitrary noise or selective perturbation. Unrequested
values remain unset, while a computed zero is displayed as zero.

`beam [WIDTH]` previews bounded sequence search by cumulative normalized policy
log-probability. It ignores proposal temperature, eligibility filters, and draw
noise. It retains one score per branch, protection/family controls, and recent
token rank details. Width bounds the retained frontier; no global optimum is
promised. Stochastic Gumbel beam without replacement has been removed.

The editor also supports writes, check/force actions, chord composition,
profiles, bias/history controls, CFG, teacher replay, rewind, fork, and EDGE
continuation. On a usable TTY, Ctrl+K opens command search, F1 help, F2 choice
details, and Ctrl+L captured output. Themes are `amber-cyan`, `chill`, `ink`,
`monochrome`, and `high-contrast`; `SPE_THEME` sets a default. Piped and
noninteractive runs use the plain interface.

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
discarded when the process exits. This branch requires workspace schema 3. Previous schemas are rejected
without migration. Normal evidence leaves `eligible_softmax` unset; explicitly
requested projector probability diagnostics replay the recorded procedure
under matching model identity to reconstruct missing values.

Replay tapes are storage-independent sequences of teacher actions with
optional expected results. Forks, rewinds, searches, and menus are editorial
moves, not tape entries. Handoff replay yields at the first divergence;
ballistic replay continues with teacher actions until the tape ends. Both modes
yield at the first unsupported action while preserving the remaining plan.

The `projector` command and the live edge expose plain-text, procedure, fork
map, and episode metadata views without making cache state part of the replay
contract.

## Validation status

The [contract matrix](tests/CORE_CONTRACTS.md) describes the intended current
behavior. Existing core/vector tests and the legacy reference suite contain
stale APIs and assertions. They have not been updated or run for this cut.
The reference kernel is not yet an oracle for the argmax experiment.

Source syntax parsing and diff review establish neither runtime correctness nor
model/terminal parity. Harness migration must precede new validation claims.
The terminal rendering guide remains the procedure for future terminal evidence.

## Project documents

- [Core scope](CORE_SCOPE.md) — what belongs in the main runtime;
- [Cut notes](CUT_NOTES.md) — architectural decisions and migration notes;
- [Core package](core/README.md) — standalone core installation;
- [Vector package](vector/README.md) — optional steering-vector tooling;
- [Core contract matrix](tests/CORE_CONTRACTS.md) — the reduced core suite;
- [Reference kernel](reference-kernel/) — legacy formulas awaiting an independent argmax port.

The version currently represented by the active package manifests is `1.0.0`.

# Serial Policy Editor 0.3.7

Serial Policy Editor (SPE) is a terminal editor for steering a local language
model one token, text insertion, or delegated span at a time. Save your choices,
rewind or fork a continuation, and replay the recorded editing procedure in a
new context.

**0.3.7 adds model-specific bias catalogs, named runtime groups, and portable
rules-only export.** It also includes the persistent full-screen editor,
scrollable context, multiline input,
Ctrl+E input expansion, and performance improvements.** The default hold is now
100 tokens; use `--hold-default` to choose another value. Transient busy feedback
has been removed to prevent layout shifts. Experimental history replacement and
fork-edit variations are excluded from this release.

Start with [installation](#install) and the [first-session walkthrough](#your-first-session).
The rest of this guide covers the [editor](#the-editor), [live-edge menu](#live-edge-menu),
[replay](#serial-policy-replay), [saved work](#persistence-and-plain-text),
[projection](#projector), [troubleshooting](#troubleshooting), and [tests](#tests).
See [the changelog](CHANGELOG.md) for release history.

## Core ideas

The program has four ideas:

1. **A live editor loop.** At every model boundary you can accept the sampled proposal, select any raw-rank token, insert text, search the full vocabulary with `/TERM`, delegate with `h`, open the edge menu with `q`, review boundaries, or fork.
2. **Checkpoints are not termination.** `--max-tokens` is only the number of visible tokens until control yields to the live-edge menu. Continue from that menu and the same episode remains live.
3. **Serial Policy Replay (SPR).** Reapply recorded policy actions from a stored episode. A changed outcome is a divergence/counterfactual, not a failure. `handoff` stops at the first changed action; `ballistic` records the divergence and continues. Replay exhaustion yields at a live edge.
4. **Compact evidence.** SQLite stores actions plus per-token proposal/NLL/rank evidence so episodes can be replayed and projected later. Replay is the equivalence check for the recorded policy path.

## Install

You need Python 3.10 or newer, a terminal, and local model files for one backend.
The source download does not include model weights. Run these commands from the
repository or extracted archive root:

```bash
cd serial-policy-editor
python3 -m venv .venv
source .venv/bin/activate
# Choose one backend:
python -m pip install -e '.[llama]'
# Or, for a local Hugging Face model directory:
python -m pip install -e '.[transformers]'
policy-editor --version
```

On Windows PowerShell, use `py -m venv .venv` and activate with
`.venv\Scripts\Activate.ps1`. Subsequent examples assume the environment is active.
Use `python -m pip install -e .` if you only need model-free workspace listing
and projection. Backend installation and hardware requirements depend on the
chosen inference library and model; SPE does not bundle an inference runtime.

## Start an episode

```bash
policy-editor \
  --backend llama.cpp \
  --model /path/to/model.gguf \
  --new-prompt 'Once upon a time' \
  --max-tokens 100 \
  --cache auto
```

Use `--seed N` for a reproducible sampler seed, or `--random-seed` to choose
and print a new seed from the supported signed 64-bit range.  The two options
are mutually exclusive.  The selected value is stored in the episode's
sampler segments for later replay or inspection.

For Transformers, `--model` is a local Hugging Face model directory.

`--cache auto` is the default: backends use incremental evaluation when they
support it and fall back to complete-prefix evaluation otherwise. Use
`--cache off` or `--no-cache` to force complete-prefix evaluation.

History rewind is always available in the live interface: `[` and `]` review
all retained token boundaries, and Enter deletes the continuation and resumes
from the selected boundary. Checkpoints do not limit navigation. You can cut
inside holds and multi-token writes. Partial writes become exact writes of the
retained text for replay, with the original submission kept in action metadata.
Rewind restores historical sampler settings, seed, and stream coordinates.
Token allowances restore their historical checkpoint on rewind and fork. `--seamless` remains
accepted for older launch commands but is no longer needed.

The EDGE command `rewind N` provides the same deletion in either interface.
Fork remains the explicit way to preserve multiple continuations.

Rewind reuses the backend's prefix cache when supported, with a complete-prefix
reset as the fallback. Resume batches already recorded tokens to restore context;
it does not regenerate them. These paths preserve the token ledger and sampler
coordinates. Small numerical differences from backend evaluation are assessed
through normal replay divergence handling.

## Your first session

1. Start the GGUF example above with your own model path. For Transformers, use
   `--backend transformers --model /path/to/local/model-directory` instead.
2. Press Enter to choose the prefilled proposal rank, or type a different raw
   rank and press Enter. A token may be a word fragment, punctuation, or whitespace.
3. Try `t The door opened` to insert text, then `h 10` to delegate up to ten
   tokens. Model end-of-generation can finish a live episode before that limit.
4. Enter `q` to reach EDGE. Type `name First experiment` to give the episode a
   title. Use `ls` to see its workspace number.
5. Type `quit` to leave the episode resumable. Later, from the same directory,
   run `policy-editor --resume '#1'`, replacing `#1` with the actual number.

The workspace path defaults to `episodes.sqlite3` in your **current working
directory**. Use the same `--workspace /path/to/episodes.sqlite3` on every command
when working from different directories. Quote episode numbers in shell commands.

To keep a second continuation, fork before changing the original: at EDGE,
`f 0` creates a child from the prompt boundary. Rewind, in contrast, deletes the
current episode's continuation after the chosen boundary. Inspect with `[` / `]`
in the live interface and press Esc to return without deleting anything.

## Choosing an operation

| Goal | Operation | Result |
| --- | --- | --- |
| Continue unfinished work | `--resume '#1'` | Restores the same episode |
| Keep an alternative | `--fork-from '#1' --at 0` | Creates a child at the selected boundary |
| Undo a continuation | EDGE `rewind N` | Deletes current text after boundary N |
| Replay from the original prompt | `--replay '#1'` | Creates a new episode from the source procedure |
| Apply a procedure here | EDGE `spr #1` | Inserts the source prompt and applies its actions in the current episode |
| Read without inference | `--project '#1'` | Prints a saved episode without loading a model |

## The editor

The familiar interaction UI is preserved. Important commands include:

- `1..N` select any valid raw rank; all numeric entries record that concrete
  rank, including selection of the sampled proposal
- `t TEXT` insert continuation text with automatic spacing
- `x TEXT` insert exact text without automatic spacing
- `/TERM` search the full vocabulary and display the target token's raw-rank neighborhood
- `m N` reveal more raw-ranked candidates
- `h [N]` delegate N tokens
- `h . [N]` delegate through the first token containing `.`, `!`, or `?`
- `h | [N]` delegate through the first token containing a newline
- `q` open the live-edge menu without generating tokens
- `e` preview/confirm teacher EOG
- `e!` commit teacher EOG immediately
- `[` / `]` review durable boundaries
- `f ...` fork from a boundary
- `?` show the complete command help

Conditional holds stop immediately after the matching token, up to their maximum
of N tokens. Matching checks the token's decoded text anywhere inside it, so
`\n`, `\n\n`, `"\n\n`, and `.\n\n` all trigger a newline hold. The whole token
is kept, including any text after the delimiter within that token. Sentence holds
leave separate closing quotes and trailing whitespace for the teacher; they do
not look ahead. Periods in abbreviations and decimals also match.

Boundary checks reuse recorded token text and cache classifications per token ID
within the engine, without additional decoding or rescanning the growing hold.

By default, the sampled proposal's raw rank is prefilled at each teacher
decision. Pressing Enter still explicitly commits that choice. Pass
`--manual-acceptance` to leave each command blank instead.

## Live-edge menu

Generation is unlimited by default. Set `--max-tokens N` for an optional
visible-token allowance. A live edge is reached on `q`, when an explicit budget
is exhausted, when SPR is exhausted, or when SPR hands off at a divergence. None of those events seals the episode.

The live-edge menu can:

- continue (preserving an early pause’s remaining allowance, or renewing an
  exhausted allowance);
- set a fresh allowance with `n N`, or remove it with `n off`;
- change sampler parameters (`s top_k=20 temperature=.8`);
- choose and record a new sampler seed (`s random-seed`, or `s random`);
- fork from a prior boundary, or use `fm` for a visual fork map;
- start SPR from another stored episode (`spr EPISODE_ID`);
- project the current episode;
- explicitly `end` and seal the episode;
- `quit` the process while leaving the episode resumable.

In live editing, explicit menu End or EOG terminates an episode. During replay,
any resolved EOG stops the tape and opens the live edge without committing the
terminal token or sealing the episode. Visible tokens already generated remain;
the remaining tape is not resumed automatically. Expected versus unexpected EOG
is retained as diagnostic evidence in a `replay-eog` interaction.

### Workspace navigation

Episodes have stable workspace numbers and optional titles; UUIDs remain valid
internal identifiers. At EDGE:

- `ls` lists open episodes; `ls all` includes finished and failed episodes.
- `#13` switches to that episode and stops at its EDGE without generating.
- `name The lighthouse` renames the current episode.
- `rewind 84` deletes the continuation from boundary 84.

Lists show recent visits first, text previews, and parent/fork information.
Selecting a finished episode opens a projection and offers an explicit fork.
The current episode is saved before switching. A cancelled or failed model load
returns to it. Forks inherit the current token allowance, not launch arguments.

Short numbers also work with CLI resume, fork, replay, projection, and lineage.
Quote them in the shell: `policy-editor --resume '#13'` (`#` starts a shell comment).

Resume defaults to the saved model path and backend, restoring recorded loading
options unless explicitly overridden. If loading fails, the program offers a
replacement path/backend. Choosing a different model requires confirmation.
For resume, the change creates a linked episode with the retained text retokenized
by the replacement model. The original token evidence stays in the source episode.
Episodes using the same model/loading options can reuse the loaded backend.

### Visual fork map

At a live edge, `fm` / `fork-map` renders every legal visible-token fork
boundary as an inline cut point and asks for the boundary to use:

```text
Once upon a time|0| there|1| was|2| a|3| ...
Fork boundary (0..3; blank cancels) > 2
```

The marker is literal: entering `2` forks at `|2|`, preserving everything to
its left and resuming immediately before the token to its right. `|0|` keeps
the original prompt/context and discards the entire generated continuation.
The ordinary `f N` command remains available and uses exactly the same absolute
boundary numbers.

### Fork lineage

The default episode list and projector remain compact. Query a complete
ordinary fork family explicitly with:

```bash
policy-editor --workspace episodes.sqlite3 --list --lineage EPISODE_ID
```

Append the same lineage metadata to a projected episode with:

```bash
policy-editor --workspace episodes.sqlite3 --project EPISODE_ID --with-lineage
```

The tree shows stable episode IDs, fork boundaries, visible-token counts, and
stored statuses. Serial Policy Replay episodes are listed separately with both
their policy source and execution context; they are not treated as ordinary
family branches.

## Serial Policy Replay

Replay an episode from its original entrance:

```bash
policy-editor --model MODEL --replay SOURCE_ID
```

By default, replay follows the source episode's recorded sampler segments at
their corresponding policy boundaries. Explicit sampler flags override only
those fields throughout replay: `--seed 77` holds seed at 77 while temperature,
top-k, and other settings continue to follow the source. `--random-seed` behaves
the same way after selecting and printing a concrete seed.

Add `--fixed-config` to freeze the source's initial sampler configuration, plus
any explicit sampler overrides, for the entire replay. For example:

```bash
policy-editor --replay '#13' --seed 77
policy-editor --replay '#13' --fixed-config
policy-editor --replay '#13' --seed 77 --fixed-config
policy-editor --replay '#13' --max-tokens 100
policy-editor --replay '#13' --max-tokens 100 --seed 77 --fixed-config
```

`--max-tokens` sets the target output allowance independently; it does not import
a source budget or change which sampler fields follow the source. `--fixed-config`
requires CLI `--replay`. On return to EDGE, the last active values remain, but
replay has no further authority: user settings changes and subsequent editing
work normally. Per-field overrides also apply to source settings recorded after
the last action on normal replay completion; early exits discard pending changes.

At a live edge, `spr SOURCE_ID` appends another episode's policy actions to the
current episode. It first inserts the source's initial prompt as an exact write
(`x` semantics, without added spacing), then executes the recorded actions.
Prompt-only episodes therefore insert their prompt. The destination's tokenizer
encodes that text without adding a beginning-of-sequence token; source prompt
token IDs are not copied. The insertion consumes the remaining token allowance
and can be rewound like any other write.

SPR preserves the destination's model, current sampler, and random stream;
source sampler transitions are not imported. Writes insert text, rank choices
resolve against the current context, and holds generate in that context.

The episode keeps its original boundary 0 and all prior text remains editable.
For example, invoking SPR at boundary 80 and producing 30 tokens reaches boundary
110 in the same episode. You can rewind inside that span, before boundary 80, or
back to 0. Fork explicitly before replay if you want to retain a separate path.

Each appended action records its immediate source episode and source boundary
in `replay_origin` metadata; prompt writes additionally carry `part="prompt"`.
A `replay-start` interaction records the invocation.
Partial rewinds retain provenance on surviving actions. Source coordinates are
provenance, not destination navigation coordinates. The procedure is snapshotted
before execution, so replaying the current episode is finite and does not append
its own newly generated actions to the plan. Composite episodes can themselves
be replayed as complete procedures.

This is counterfactual replay: normal divergence and EOG handoffs still apply.
At EDGE, the user can change settings and continue editing normally. CLI
`--replay` continues to create a new episode and use its source prompt only as
initial context, without inserting it again; EDGE `spr` does not create an episode.

`--divergence-policy handoff` yields at the first changed action. `ballistic` records the divergence and keeps applying the policy where structurally possible.

SPR in this reduced build intentionally does **not** reproduce the old 0.2
execution-path machinery or automatically traverse the old report forest.

## Persistence and plain text

The workspace defaults to `episodes.sqlite3`. It exists mainly to support SPR and retain useful editorial evidence:

- sampled proposal token;
- proposal agreement;
- raw-model NLL;
- raw rank;
- policy rank;
- decoder probability/support.

To also write plain text when an episode is sealed:

```bash
policy-editor ... --output result.txt
```

Quit without sealing and resume later:

```bash
policy-editor --model MODEL --resume EPISODE_ID
```

### Saved sampler validation and recovery

Saved sampler configurations must contain every recorded field and supported
RNG/policy identifiers. Stream identities must be lowercase SHA-256 digests;
coordinates and token IDs must be valid integers. Loading no longer silently
fills missing saved settings or regenerates an invalid stream identity.

Resume, fork, CLI replay, EDGE replay, and episode switching inspect the source
before execution. When recovery is possible, the editor lists every proposed
replacement and its replay implications, then asks `Do you wish to proceed?
[Y/n]`. Enter accepts; `n` or closed input cancels. Accepting creates a separately
numbered recovered copy and records the repair details in its metadata. The
source stays unchanged, and previous token evidence is retained as historical
evidence rather than recomputed. Unsupported explicit RNG/policy schemes stop
with an error requiring a compatible build; they are never silently substituted.

Defaults remain available for new configurations. The editor owns the sampling
seed for both backends; the llama.cpp adapter does not configure a second seed.

## Projector

The projector can render a seamless text view, legacy inline evidence, or a
teacher-focused statistical view. ``--full-evidence`` footnotes tokens chosen
by teacher actions with proposal agreement, NLL, raw rank, and policy rank.
Tokens produced inside autonomous ``hold`` spans remain unfootnoted.

```bash
policy-editor --workspace episodes.sqlite3 --project EPISODE_ID --full-evidence
```

Raw-model and final decoder probabilities are available as a separate opt-in
layer. They are marked as ``MODEL[...]`` so the distinction survives plain or
redirected output as well as terminal display:

```bash
policy-editor --workspace episodes.sqlite3 --project EPISODE_ID \
  --full-evidence --with-model-probs
```

``--with-model`` is a short alias for ``--with-model-probs``. At a live edge,
``p``/``project`` opens the full-evidence view without model probabilities.

Use ``--with-lineage`` to append fork-family and replay metadata to the
projection.

## Scope and compatibility

The 0.3.x editor does not ship the 0.2 verifier, execution replay, recovery
reconstruction, legacy report browser/replay command, Navigator, lineage
replay, or backward-compatibility schema machinery.

The inference adapters may evaluate incrementally or from complete prefixes as
their runtime permits. That choice is not episode state, evidence, or lineage;
the token/action ledger and Serial Policy Replay define the recorded behavior.

## Replay and editing details

Replay respects the target budget and never enlarges it to fit the source.
Budget exhaustion discards the remaining tape and returns live control.
Sampler changes do not reset the remaining allowance.

Writes and holds must fit the remaining allowance in full. Oversized requests
are rejected before generating tokens and open the edge menu; during replay
they stop the tape. Sentence/newline holds must also fit their requested maximum.
The remaining allowance is unchanged by rejection.

Recognized moves that cannot execute in the target (an unavailable raw rank,
a write producing no tokens, or no selectable EOG) stop replay and open the
live edge with an explanation. Rejected moves are recorded as interactions,
not executed tape actions. Unexpected backend and storage errors retain normal
error handling.

Replay holds execute their derived requested limit and current sentence/newline
stopping conditions. Recorded expectations only detect divergence: handoff
interrupts before a changed or extra token, while ballistic continues the move.
Expected token counts and stop reasons never replace the hold’s execution rules.

Sentence holds recorded before this change may replay with fewer tokens because
separate closing quotes and whitespace are no longer consumed. Existing replay
divergence handling reports the difference; stored episodes are not rewritten.

Replay uses an explicit finite plan. Empty plans open the live edge immediately.
Source-following plans apply the final recorded sampler settings on normal
completion, including changes made after the last move. Fixed-sampler plans
ignore source transitions only while replay is active. Early exits discard all
pending transitions; the edge menu displays the actual active configuration,
and continuing cannot restore the old plan or overwrite teacher changes.

Numeric entry accepts any raw rank from 1 through the model vocabulary size,
even if the token has not been displayed. Search and menu expansion are optional
inspection tools. Selecting the proposal’s rank records that concrete rank.

Teacher selections always record concrete raw ranks. Enter on the prefilled
proposal, blank Enter (including manual mode), and explicit `accept` resolve to
the current proposal’s raw rank. Proposal agreement remains independent token
evidence. Holds still delegate to the sampler. Existing stored legacy `Accept`
actions retain their original semantics.

Continuation spacing is intrinsic to `t`, not a run setting. It adds a space
before an alphanumeric continuation when the preceding text requires one;
existing whitespace and opening delimiters are respected. Use `x` to join text
directly, such as appending `nother` to `a`.

### Numeric preview and exploration

Menu expansion has no configured cap. Each `m N` adds N rows to the main menu,
up to the vocabulary size; previously explored neighborhoods do not affect that
count. The live interface windows the table around the selected row. Plain mode
prints the expanded table, and very large expansions can take time to calculate.

In the live interface, typing a valid raw rank previews that token and its
evidence, even when it is outside the displayed menu. Enter selects it. Ctrl+G
instead opens its neighborhood without selecting anything. The typed command
`ms N` opens the same neighborhood in either interface without a text search.
Existing `ms + [N]`, `ms - [N]`, and bare `ms` controls remain available.
Previewing does not expand the menu or change the token state. Ctrl+G does
nothing when the input is not a valid numeric rank or while reviewing history.

### Procedure view

Use `--project EPISODE_ID --procedure` to print a manual replay listing without
loading a model. The header shows the model basename, backend, initial sampler,
and prefix. Lines use `boundary : command #result`; boundaries are source fork-map
labels, not jump instructions. Comments preserve leading spaces (`#The` versus
`# The`). Control characters in comments and prefixes are escaped; literal
backslashes are doubled there. Ordinary writes preserve literal text and omit
duplicate result comments. Writes containing actual control characters are
explicitly marked as display-escaped, not paste-ready.

The view shares replay's surviving-move derivation. Partial handoffs become
finite holds; holds show their recorded output. Navigation and discarded attempts
are omitted. Sampler transitions use `q`, `s key=value`, `c`, with concrete seeds.
Trailing settings leave the procedure at the edge; empty procedures show `0 : q`.

This is an inspection format, not a loader or a prediction of target output.
Source budgets are not imported: the latest saved allowance need not be the
initial one. Legacy acceptance displays its recorded rank; legacy finish is
marked as a recorded span. EOG instructions are literal: manual live execution
can terminate on EOG, whereas machine replay yields to the edge.

### Token budget history

New episodes are unlimited unless a budget is requested. A finite allowance is
consumed by visible generated or written tokens, including EDGE SPR text.
Sampler changes and replay never replenish it or import a source budget.
Continuing before exhaustion preserves the remainder; continuing after exhaustion
renews the configured allowance. An explicit budget change starts a fresh allowance.

Rewind restores the allowance and remaining tokens at the selected boundary.
Forks inherit that same state, including an exhausted allowance, rebased to their
own boundary 0. Resume, workspace switching, and model replacement preserve the
remaining allowance. An explicit CLI budget overrides it with a fresh allowance.
Budget changes and renewals are stored at their boundary, just like sampler edits.

If historical budget evidence is absent, the operation uses unlimited tokens.
A notice appears only when the record lists a finite budget that cannot be
reconstructed; unlimited records need no notice. Old checkpoint values are not
used to guess historical allowances.

### Replay through a source boundary

Use `--replay EPISODE --until Y` from the CLI, or `spr #N --until Y` at EDGE,
to replay through source boundary Y and return to EDGE. `spr #N m` opens the
source's recorded token map and asks for a stopping boundary; blank cancels.
The map is a reference, since generated output can differ during replay.

Y uses source token positions, not destination positions. A cutoff inside a
write inserts its recorded text prefix; a cutoff inside a generated span makes
that span a finite hold. Earlier divergence and destination budget limits retain
their normal behavior. CLI replay follows source sampler changes only through Y,
subject to explicit overrides and `--fixed-config`.

At EDGE, `--until 0` inserts the entire source prompt as ordinary text, then
returns. Every inserted token has a normal destination boundary and can be
rewound or forked into. CLI `--until 0` loads the prompt as initial context and
returns without executing actions. Omitting `--until` replays the full procedure.

## Troubleshooting

- **`policy-editor` is not found:** activate the environment used for installation,
  or run `python -m trajectory_editor --help` from the package directory.
- **The terminal display is awkward:** launch with `--plain-ui` for the line-based
  interface. EDGE `rewind N` works there too.
- **An episode is missing:** check `--workspace` and the current directory. Workspace
  numbers belong to one database; they are not global identifiers.
- **A model has moved:** resume can offer a replacement path. Selecting a different
  model requires confirmation and creates a linked continuation with retokenized text.
- **A hold or write returns to EDGE without output:** check the remaining allowance.
  The whole request must fit, including a conditional hold's requested maximum.
  Use `n N` for a fresh allowance or `n off` for unlimited generation.
- **Replay stops earlier than expected:** inspect divergence, EOG, the target
  allowance, and any `--until` cutoff. Continuing at EDGE resumes live editing;
  it does not restart the discarded remainder of the replay plan.

To back up work, exit SPE and copy the workspace database. Keep the associated
model files available for future inference. Plain-text exports are readable
copies, not replayable workspaces. To export without ending an episode:

```bash
policy-editor --workspace episodes.sqlite3 --project '#1' > continuation.txt
```

## Tests

From `serial-policy-editor/`, install the test dependencies and run the complete
suite with the active virtual environment:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

All tests use pytest functions with plain assertions, `pytest.raises`, and
`pytest.mark.parametrize`. Filesystem tests use `tmp_path` for isolated workspaces.
`unittest.mock` remains in use for mocks and patches; there are no
`unittest.TestCase` classes or unittest runners. Preserve existing cases and
setup/cleanup behavior when changing tests.

Reference sampling calculations live in `tests/sampling_reference.py`; they check
the production observation statistics without adding unused runtime APIs.

### Real llama.cpp sampler smoke tests

The optional `llama_smoke` tests load an actual local GGUF through the normal
CLI backend loader. They compare plain-text EDGE output and the configuration
received by production sampling against scenario expectations. An independent
full-sort calculation checks history penalties, temperature, top-k, top-p,
min-p, and the seeded draw against real model logits. No model output is mocked.

The scenarios cover CLI initialization, EDGE edits, rewind, fork, reopening a
saved episode with a newly loaded backend, and replay seed overrides with and
without `--fixed-config`. Replay tests also verify that subsequent EDGE edits
remain authoritative. Reopening is currently within the test process; this does
not yet test fresh-process isolation or the full-screen terminal renderer.

From `serial-policy-editor/`, with the project virtual environment active:

```bash
SPE_LLAMA_SMOKE_MODEL=../Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf \
  python -m pytest -m llama_smoke -v
```

The reference run uses Llama 3.2 1B Instruct Q8_0, llama-cpp-python 0.3.35,
CPU execution with two threads, and a 256-token context. The test reports the
model file's SHA-256 and backend version for diagnosis. No downloads occur.
Without the environment variable, these tests skip; an explicitly configured
missing model or broken backend fails. Other GGUF files can be supplied, though
the short scripted journeys are validated against this reference model.

### Real Transformers sampler smoke tests

The optional `transformers_smoke` tests exercise an actual local Hugging Face
causal language model through SPE's Transformers backend. No model output is
mocked. In addition to the same edit, navigation, rewind, resume, and replay
journeys used by the llama.cpp smoke tests, the suite compares SPE's cached
backend logits with a fresh complete-prefix Hugging Face forward pass. It then
checks the consequential sampler behavior with an independent calculation.
The reference prefix comes from the episode engine and must match the backend's
token ledger and sampler history. The cached and fresh distributions must also
select the same token using the same seeded random draw, even when their
probabilities differ within the allowed numerical tolerance.

This is intended to catch both sampler errors and backend/cache errors: a KV
cache that is internally self-consistent but positioned at the wrong prefix
should disagree with the fresh full-prefix evaluation. The smoke run uses CPU
and float32 by default to keep the reference path straightforward.

Point `SPE_TRANSFORMERS_SMOKE_MODEL` at a local Hugging Face model directory:

```bash
SPE_TRANSFORMERS_SMOKE_MODEL=/path/to/local/model \
  python -m pytest -m transformers_smoke -v
```

The directory must already contain the model and tokenizer files required by
`transformers`; the smoke tests do not download models. Without the environment
variable, the tests skip.

### Recorded 0.3.4 verification matrix

The following results were recorded for 0.3.4; they are not new 0.3.5 runs
and are not a claim that every
model or runtime combination is supported. They are useful release checks
because they exercise real model weights and the normal inference adapters
rather than mocks. Re-run at least one model from each backend before a release;
additional architectures are useful when convenient.

| Model | Backend | Smoke suite | Result | Notes |
| --- | --- | --- | ---: | --- |
| Llama 3.2 1B Instruct Q8_0 GGUF | llama.cpp | `llama_smoke` | 5/5 passed | Small reference GGUF; CPU smoke path |
| GPT-2 | Transformers | `transformers_smoke` | 6/6 passed | Dense float32 baseline; Transformers 5.16.1 / torch 2.14.0+cu130 |
| SmolLM-360M-Instruct | Transformers | `transformers_smoke` | 6/6 passed | Newer causal-LM/cache implementation; Transformers 5.16.1 / torch 2.14.0+cu130 |
| Qwen3.5-9B-UD-Q4_K_XL GGUF | llama.cpp | `llama_smoke` | 5/5 passed | Larger, newer Qwen-family quantized GGUF; local run completed in about 93 s |

For a compact release check:

```bash
# llama.cpp / GGUF
SPE_LLAMA_SMOKE_MODEL=/path/to/model.gguf \
  python -m pytest -m llama_smoke -v

# Hugging Face Transformers
SPE_TRANSFORMERS_SMOKE_MODEL=/path/to/model-directory \
  python -m pytest -m transformers_smoke -v
```

A passing matrix is evidence that SPE agrees with real inference engines on the
paths exercised by these tests. It is not a bit-for-bit reproducibility claim:
different hardware, kernels, dtypes, quantization, and near-tied logits can
produce small numerical differences. The smoke suites emphasize consequential
agreement such as ranking, sampler support, deterministic selection, replay,
and cache/full-prefix consistency.

## License

Copyright (c) 2026 Graham Christopher Andrews.

Serial Policy Editor is released under the [MIT License](LICENSE).
Third-party dependencies and model weights remain subject to their own licenses.

### Context scrolling and multiline input

The live editor, historical review, and EDGE menu use the full terminal screen,
with input anchored at the bottom. Context and the writing area grow when the
terminal gets taller. The context window starts at the newest text; space is
reserved for candidate rows and feedback. **Page Up / Page Down** scroll through older
and newer context; the row indicator shows your position. All context is retained
by default. `--context-chars N` optionally limits it to the last N characters
(`0`, the default, means no character limit).

For `t TEXT` and `x TEXT`, **Alt+Enter** inserts a newline and **Enter** commits.
Raw-text commands start in the compact input with the normal candidate table.
**Ctrl+E** toggles a fixed writing area (about a third of the terminal) with three
candidate rows. Toggling preserves the draft, cursor, and undo history. The expanded
editor scrolls without moving surrounding sections as lines are added. Newlines
and multiline pastes do not expand it automatically. Removing the raw-text prefix
returns to compact input; a new raw-text command starts compact. Multiline paste
preserves newlines, blank lines, tabs, and leading/trailing whitespace. Tab inserts
a literal tab while editing raw text. If your terminal intercepts Alt+Enter,
press Escape followed by Enter to send the same key sequence.

## Headless preview

An experimental headless branch includes a local HTTP service and browser editor.
See [HEADLESS.md](HEADLESS.md) for launch instructions, the API contract, and preview limits.

### KV cache precision (llama.cpp)

Use `--cache-type-k` and `--cache-type-v` to choose `f16`, `q8_0`, or
`q4_0` independently. Omitting these options preserves the library defaults.
For example:

```bash
policy-editor --model /path/to/model.gguf --cache-type-k q8_0 --cache-type-v q8_0
```

Quantized caches reduce context-cache memory without changing model weights.
They can change token probabilities and replay results; speed and compatibility
depend on the model and backend. Quantized V requires Flash Attention, which SPE
requests by default; do not combine it with `--no-flash-attn`.
Settings are saved with episodes and restored on resume, fork, and replay;
explicit launch options override saved settings. The local headless server
accepts the same two cache-precision options. `--cache off` controls prefix reuse,
not cache precision.

### Single-token logit biases

At a live token position, use a **raw rank** followed by an operator:

- `12-` or `7+`: decrease or increase that token's bias by 0.5.
- `8-0.25` or `8+2`: adjust by an explicit positive amount.
- `12=`: clear that token's bias back to zero.

The rank identifies the token now; the bias then follows its token ID throughout
this episode. Commands accumulate without advancing, so you can enter `12-`,
`7-`, `201+`, then `6` to make your actual move. Raw ranks remain unchanged;
the sampled proposal and decoding probabilities refresh. Biased menu tokens show
`[bias +/-N]`. Use `--bias-step 1` to change the default increment (also available
as `s bias_step=1` in the EDGE menu).

Biases are added after history penalties and before temperature, top-k, top-p,
and min-p. Positive values encourage a token; negative values discourage it.
They do not ban tokens or prevent Teacher from selecting them. Bias commands
neither evaluate new tokens nor advance the sampling coordinate. Clearing all
biases restores the proposal for the unchanged context and sampler settings.

Resume restores biases. Fork and rewind use exactly the existing sampler-state
semantics: both select the state at the target boundary, including adjustments
already recorded at that boundary. Rewind removes changes at later boundaries.
Replay follows the recorded bias states unless explicitly overridden; in-place
SPR keeps the destination sampler, including its biases. Switching models clears
inherited token-ID biases.

### Experimental bias catalog compiler

The standalone `policy-editor-bias` tool can compile human-readable YAML
terms against a local model tokenizer. It currently produces a model-specific,
bias-free catalog for inspection and for the logical bias-rule runtime. A
simple YAML list becomes the automatic `global` group:

```yaml
- sky
- cloud
- fog
- aardvark
```

Named groups use the expanded mapping form:

```yaml
terms:
  - sky
  - mango
groups:
  nautical:
    - anchor
    - steamship
    - port of call
```

Compile the catalog with the normal tokenizer route search:

```bash
policy-editor-bias \
  --model /path/to/model.gguf \
  --input terms.yaml \
  --output catalog.json
```

Use `--level minimal`, `--level standard`, or `--level exhaustive` to select
canonical routes, bounded alternate routes, or a broader exact-route search.
`--max-route-tokens` controls decomposition depth, while `--max-routes`
controls how many routes are retained. `--term TEXT` may be repeated for
one-off compilation without a YAML file. YAML quoting is only YAML syntax:
quoted, unquoted, and single-quoted semantic terms receive the same spacing
and case expansion. The compiler adds leading-space variants automatically,
so users do not need to write them.
`max_routes` is a per-term cap across all generated case, spacing, plural, and
suffix forms; increase it only for terms where the extra routes are useful.
Canonical routes are reserved first, then deterministic alternate routes are
selected round-robin across generated forms until the budget is full.

The runtime matcher now has one logical route engine. A plain lexical target
that tokenizes into multiple pieces uses telescoping path semantics: `b
velociraptor +1` biases a viable starting token and then the next route token
after each matching prefix. Whitespace phrases retain tail semantics by
default, so a phrase continues to mean “bias the completion after this prefix.”
Alternate routes sharing an edge contribute once per logical rule rather than
once per route. Logical rules and named groups can be saved in the
`spe-bias-rules-v2` JSON preset format.

Load a compiled catalog into an interactive episode with
`--bias-catalog catalog.json`. A bare bias target uses a matching catalog entry
when one exists and otherwise falls back to one-shot tokenizer resolution;
`b @name +1` requires that catalog entry. Quoted runtime text bypasses catalog
lookup and remains an exact text target.

Named groups can also be created during an episode:

```text
b nautical -> {anchor, steamship, " port of call"}
b nautical +1
```

The arrow command creates or appends to a durable group. A later bare reference
to that name uses the same compiled routes as a catalog group; `@name` remains
strictly a catalog reference. The group has one shared bias amount, so adding a
member immediately inherits the group's current amount. Group definitions and
their bias changes are sampler state: replay, fork, and rewind restore them at
the relevant boundary. Runtime groups are included in the normal `--biases-only`
export.

Export the current surviving bias set as a JSON preset:

```bash
policy-editor --workspace episodes.sqlite3 --project '#1' --biases-only > biases.json
policy-editor --workspace episodes.sqlite3 --project '#1' --biases-only --rules-only > rules.json
policy-editor --model /path/to/model.gguf --biases biases.json --new-prompt 'Once upon a time'
```

`--biases` replaces the saved bias set, including on resume, fork, or replay.
An empty `bias_rules` list explicitly clears direct rules. A full preset contains
a format version, model metadata, direct logical rules, and any named groups.
`--rules-only` flattens the group's effective routes into ordinary logical rules
and omits group names and runtime metadata, producing a portable rules-only
preset that can be loaded like any other bias file. Loading checks vocabulary
size and supplied model metadata. Use presets with the model/tokenizer they
were made for; token IDs are not portable across tokenizers. Presets store bias
values, not the default interactive step. The supported preset format is
`spe-bias-rules-v2`.

### Logical bias rules

Every bias command creates one logical rule. A rule contains one or more token
routes, a mode, an amount, and optionally trigger/lifetime conditions. Raw model
probabilities and raw ranks remain untouched. The displayed `[bias +/-N]` is the
total currently active bias for that token.

Several ways to enter a rule:

```text
b New York +0.5
b {New York, New Jersey, "C"} +0.25
b " New York" +0.5
bl 3 -
12+0.5 ... " New"
```

- `b` bare text is continuation-oriented: surrounding whitespace is stripped and
  SPE supplies one leading space before tokenization. This makes `b wings +` mean
  the common token spelling `" wings"` without making you type the space or quotes.
- `{...}` applies the same edit to several comma-separated targets. Bare items get
  the same automatic leading space; quoted items are exact. Multiword bare items are
  fine because commas, not spaces, separate the group.
- A quoted `b` phrase remains exact. JSON quoting supports escaped quotes, newlines
  (`\n`), and intentional leading/no-leading whitespace. The older exact JSON-list
  form such as `[" wings", " scales"]` remains accepted.
- `bl X` captures the last X actual context tokens, including prompt tokens if
  the span reaches into the prompt. X must be positive and no greater than the
  context length. It does not insert text or retroactively change those tokens.
- The conditional rank form tokenizes the quoted prefix separately, then appends
  the exact token ID at that raw rank. It does not retokenize their combined text.

All forms accept bare `+`/`-` for the default increment and `=` to clear the exact
rule (`b " New York" =`, `bl 3 =`, or `12= ... " New"`). A one-token target is a
one-edge path rule. A lexical target that tokenizes into multiple pieces uses
telescoping path semantics: its head is biased when the term can begin, and each
continuation is biased after the matching prefix. Whitespace phrases default to
tail semantics, while `bl X` and ranked-prefix commands explicitly use tail
semantics. Empty phrases and empty conditional prefixes are rejected.

The commands stay at the same live position. Multiple rules may be edited before
making a move, and `v` still sorts by policy rank while raw ranks remain visible.
Rules follow the same sampler-state persistence, resume, replay, fork, and rewind
semantics as single-token biases. Matching is recomputed from the restored context;
there is no separate matcher state to restore. Rewinding retains rules at the
target boundary and discards later rule changes, exactly like other sampler settings.

`--project '#1' --biases-only` exports the complete logical rule and group set;
add `--rules-only` to flatten named groups into ordinary logical rules. Loading
a preset replaces the saved direct rules and named groups.

### Triggered biases with exact stop tokens

A scoped logical rule activates after any trigger appears since its most recent
stop token:

```text
b wings +0.5 after dragon until "."
b {scales, claws} + after {dragon, wyvern, winged serpent} until "."
12- after dragon until "\n"
```

Bare targets and triggers use the same continuation-friendly spelling as ordinary
`b` commands: `wings` means `" wings"`, while quoted strings are exact. Braces mean
"any/all of these comma-separated items": any trigger activates the gate, and the
same edit is compiled into one ordinary scoped rule per target. Quoted items inside
a brace group stay exact, for example `{hello, "Hello", "\n"}`.

The `until "TOKEN"` form names one exact stop token. SPE tokenizes the quoted text
without a BOS token and requires it to resolve to exactly one token; if it does
not, use `until #N` to name a token ID explicitly. `until .` and `until |` remain
available as sentence and newline lifetime heuristics. Because activation is
derived from token history, rewind, fork, resume, and replay need no separate
matcher state.

Targets and triggers are exact token sequences after the human spelling is expanded.
Repeated triggers do not multiply a rule's strength. Distinct active rules add
together. These rules do not perform semantic similarity matching.

Use bare `+`/`-` for the configured step, an explicit amount to change it, or `=`
to clear the exact rule. The target, trigger set, and stop condition identify the
rule; reordering trigger alternatives does not create a new rule. Multiple targets
are command-line sugar only and remain separate ordinary rules in saved sampler
state. Multiple targets also work without a scope:

```text
b {wings, scales, claws} +0.5
b {wings, scales} = after {wyvern, dragon} until "."
```

Scoped syntax supports human/quoted `b` targets and ranked targets. Use a `b` target
for scoped multi-token rules rather than combining the rank-prefix `...` form with
`after`.

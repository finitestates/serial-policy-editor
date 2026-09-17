# Serial Policy Editor 0.4.6

Serial Policy Editor (SPE) is a terminal editor for steering a local language
model one token, text insertion, or delegated span at a time. Save your choices,
rewind or fork a continuation, and replay the recorded editing procedure in a
new context.

**0.4.6 adds direct hidden-state vector capture for llama.cpp alongside the Transformers backend.**
Load relative term weights with `--reference`, define groups with YAML or
commands, and activate promote, suppress, or maintain objectives with bare
`b target +`, `-`, or `=`. Numeric amounts remain manual; `off` clears an
activation. The compiler uses canonical runtime routes and keeps optional
alternate decompositions for exploration.

Teacher preference learning remains independent, with optional fast/slow
memory, configurable severity and rejection, persisted projection seeds, and
learning from typed writes. Full `biases.json` presets preserve references,
groups, objectives, and learner vectors. The persistent terminal editor,
multiline input, browser preview, and replay workflows remain available.
The default hold is 100 tokens; use `--hold-default` to choose another value.
Phrase checks default to 16 tokens and a maximum required policy shift of 6.0;
use `--phrase-max-tokens` and `--phrase-max-shift` to change those limits.

Start with [installation](#install) and the [first-session walkthrough](#your-first-session).
The rest of this guide covers the [editor](#the-editor), [live-edge menu](#live-edge-menu),
[replay](#serial-policy-replay), [saved work](#persistence-and-plain-text),
[projection](#projector), [troubleshooting](#troubleshooting), and [tests](#tests).
See [the changelog](CHANGELOG.md) and [the 0.4.0 release notes](RELEASE_NOTES_0.4.0.md)
for release history and the structured-file formats.

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

For an interactive launch, omit the episode source to open the pre-runtime
setup menu:

```bash
policy-editor
```

Choose a prompt or replay source, model/backend, vector or bias artifacts,
common sampler settings, budget, seed, and learner toggles there. Type `show`
to review the selection and `go` to start. The same menu can be opened for an
otherwise flag-configured interactive launch with `--setup-menu`; scripted and
non-interactive launches continue to use the command-line interface directly.
The `workspace` command shows or switches the SQLite workspace, `ls` lists its
episodes, a bare `#N` (or `N`) prints episode information, and `fm [#N]` opens a
read-only fork map to help construct a later `source fork ... at N` command.
Typing `sampler` without arguments prints all sampler settings and their
planned values. The same pattern applies to the option-heavy learners:
`learning` (or `group`) without arguments prints the manual-group learning
settings, and `preference` prints the token-preference settings. Their
`key=value` forms update those controls in the setup plan, for example
`group on rate=.2 gate=sampler` or `preference on dimension=32 fast_slow=on`.
The main summary redraws after each change, so the enabled state is visible.
Use `controllers` (or `stack`) to inspect the ordered influence surfaces:
model logits, history penalties, hidden-state control, output-head steering, biases/groups,
reference prior, token-preference actuation, and final sampling, followed by
the feedback learners.

Reusable controller intent can be printed or saved as strict YAML from the
same menu:

```text
profile print
profile save profiles/calm.yaml
profile load profiles/calm.yaml
```

Profiles contain sampler, steering-artifact, group-learning, and token-
preference settings. They intentionally do not contain an episode source,
workspace, model/backend choice, or token budget, so loading one cannot move a
replay to another episode or workspace. The YAML document includes a
fingerprint derived from canonical, sorted JSON; hand-edited profiles may omit
that fingerprint, while a present fingerprint must match exactly.

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
- `check TEXT` probe and commit a continuation phrase when every token is within
  the configured policy-shift bound
- `checkx TEXT` do the same without implicit whitespace
- `force TEXT` commit a continuation phrase using temporary per-token policy
  shifts, then clear them
- `forcex TEXT` force an exact phrase without implicit whitespace
- `/TERM` search the full vocabulary and display the target token's raw-rank neighborhood
- `m N` reveal more candidates (raw top-N by default, policy top-N in policy order)
- `V` toggle policy diagnostics without reordering
- `v` toggle raw versus full-vocabulary policy top-N ordering
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

By default, the sampled proposal's backend rank is prefilled at each teacher
decision. Pressing Enter still explicitly commits that choice. Pass
`--manual-acceptance` to leave each command blank instead.

Policy diagnostics appear automatically when either learner is enabled, bias
rules/groups are configured, or the restored sampler has active adjustments
(including fast-only token preference memory). They appear before the first learning
update, even while the values are zero. Use `--policy-view` to show them
explicitly or `--no-policy-view` to hide them. `--show-policy-rank` remains an
alias for `--policy-view`.

| Column | Meaning |
| --- | --- |
| `Δrank` | Backend rank minus policy rank: positive means promoted, negative means demoted. |
| `Δlogit` | Adjusted logit minus backend logit, including penalties, biases, priors, and both preference memories. |
| `pol-rank` | Full-vocabulary rank under the adjusted policy. |
| `raw-p` / `backend-p` | Returned backend/model probability before temperature and filtering. `raw-p` and `raw_probability` are retained display/API aliases. |
| `pol-p` | Adjusted policy probability before temperature and filtering. |
| `decode-p` | Final sampling probability after temperature and filtering. |

The deltas compare the returned backend surface with the **current policy at the same
context**, not specifically the last learning update. A token can have a
positive adjustment yet be removed by decoder filtering. Narrow terminals
retain the two deltas and token text before secondary columns; plain mode
shows the full table. The expanded writing view also retains the deltas.

Ordering stays backend-ranked by default. In policy order, the table fetches the actual
policy top-N across the whole vocabulary, so strongly promoted tokens outside
the backend top-N become visible. Numeric selections always refer to absolute backend
rank, and search neighborhoods retain backend ordering. `V` and `v` preferences
persist across decisions and live edges for the session; they do not modify
sampler state, replay, or learning.

### Candidate filtering and draw kernels

The sampler has a formal boundary between candidate filtering and token
drawing. Temperature, top-k, typical-p, tail-free, top-p, and min-p produce a
deterministic candidate set; the draw kernel then chooses from that set.
Typical and tail-free are disabled at their neutral value `1.0`:

```text
sampler typical_p=.95 tail_free_z=.95
sampler draw_kernel=gumbel-max
```

`categorical` is the default kernel and uses SPE's deterministic prefix
quantile. `gumbel-max` uses a deterministic per-token Gumbel draw from the
same seed, stream fingerprint, and sampling coordinate, so replay remains
stable without depending on candidate-array order.

Classifier-free guidance is available as a model-phase prefix controller. It
runs a second copy of the active backend for an unconditional prompt and
applies `unconditional + scale * (conditional - unconditional)` for the first
configured number of generated tokens:

```bash
policy-editor --model model.gguf --new-prompt "A scene" \
  --cfg-unconditional-prompt "" --cfg-scale 1.8 --cfg-prefix-tokens 8
```

An empty unconditional prompt is valid and is the usual baseline; a short
neutral instruction can be used instead. After the prefix count is consumed,
the unconditional branch is dropped and ordinary autoregressive decoding continues. The CFG settings are part of the
runtime intent and sampler record, so replay can reproduce the schedule.

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

### Cross-backend hidden-state conformance

The optional `transformers_gguf_smoke` check loads the same local GGUF through
Transformers and the SPE llama.cpp worker. It verifies tokenizer identity and
compares the worker's pairwise directions against explicit Transformers hooks
for every one-based decoder-block output, the embedding output, the final
pre-normalization residual, and the post-normalization output-head input. This
is deliberately a slow sanity check for worker coordinates and prompt-pair
arithmetic, not part of normal vector creation or runtime generation. Use an
F16 GGUF for the clearest reference before repeating the check with a quantized
file:

```bash
SPE_TRANSFORMERS_GGUF_MODEL=/path/to/model-f16.gguf \
SPE_LLAMA_WORKER=/path/to/spe-llama-worker \
  python -m pytest -m transformers_gguf_smoke -v
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

### Lexical references and group objectives

Load a lexical universe independently with `--reference reference.yaml`.
Load group definitions directly with `--groups groups.yaml`, or assemble them
at the teacher prompt with `b atmosphere -> {shadow, silhouette}`.

```text
b atmosphere +       promote appearances adaptively
b atmosphere -       suppress appearances adaptively
b atmosphere =       maintain approximately the present rate
b atmosphere off     disable that group objective
b atmosphere +0.5    explicit manual logit bias
b                    inspect groups and controller diagnostics
```

All semantic operations support `after … until …`. Adaptive group objectives
operate during autonomous generation without teacher learning. Canonical
surface routes are used at runtime; alternate decompositions remain available
through `policy-editor-bias --explore` for inspection.

See [Steering](STEERING.md) for the complete command, reference, learning, and
`biases.json` workflow, and [Catalog YAML](BIAS_CATALOG_YAML.md) for definitions.

### Bias catalogs and structured policy files

The catalog compiler is optional. `--groups` reads YAML directly; `--bias-catalog`
still reads compiled JSON. Both use canonical runtime routes, including when
an exploratory catalog also contains other decompositions.

Export/import remains:

```bash
policy-editor --workspace episodes.sqlite3 --project '#1' --biases-only > biases.json
policy-editor --model model.gguf --new-prompt 'Another story' --biases biases.json
```

The v4 preset includes manual rules, groups, adaptive objectives, independent
reference weights, and token preference vectors/metadata. Older preset formats
are intentionally rejected. `--editor-friendly` exports membership YAML;
`--rules-only` is for manual rules and rejects active adaptive objectives.

### Opt-in online group learning

Appearance objectives run independently of teacher fitting. Start with
`--online-learning`, define or load a group, then use `b GROUP learn on` to let
your selections adjust its manual amount. Use `b GROUP learn off` to freeze the
amount, and `b` to inspect learning eligibility. New groups start with learning
off; membership and numeric edits preserve your explicit choice. Clear any
appearance objectives before enabling learning for that group.
See [teacher learning controls](STEERING.md#teacher-preference-learning).

### Opt-in token preference learning

The token preference learner is independent of group objectives. It derives a
fixed 64-dimensional, unit-normalized feature vector for every vocabulary
token from the loaded model's token/output embedding through a deterministic
projection. It learns only an anonymous vector `z`; there are no named preference
dimensions and no token-specific learned values:

```text
preference logit adjustment = preference strength * dot(z, token feature)
```

Enable it alongside or instead of group learning for a live session:

```bash
policy-editor --model /path/to/model.gguf --biases groups.json \
  --token-preference --token-preference-strength 1.0 \
  --token-preference-learning-rate 0.05 --token-preference-max-step 0.25 --token-preference-max-norm 4.0
```

It updates after each live teacher-selected token. The update moves
`z` toward the chosen token's fixed feature and away from the current
policy-weighted feature mean with full default severity; rank, sampler
eligibility, and the sampled proposal remain diagnostics. The vector is stored in sampler segments, included in
full `--biases-only` exports, and consumed during replay without rerunning the
learner. If the model changes, its token preference state is discarded because the
fixed feature space is model-specific. `--token-preference-dimension` can reduce the
projection size for a small experiment; 64 is the default.

Feature construction is chunked by default to cap its temporary projection
buffer. Use `--token-preference-projection-chunk-size` to lower the peak further (at the
cost of slower initialization) or raise it when startup speed matters more
than peak memory. The setting does not change the learned feature space and is
not persisted in presets.

### Offline vector workbench

SPE uses “vector” as an umbrella term for several deliberately different
control surfaces. They are typed in the portable artifact rather than being
presented as generic “activation vectors”:

| Vector kind | Coordinate space | Runtime effect |
| --- | --- | --- |
| `hidden-state-vector` | Decoder-block output residual at selected layers | Changes the model’s internal residual stream before later layers run. |
| `output-head-steering-vector` | Post-normalization output-head input | Projects through the output head into a vocabulary-wide logit adjustment. |
| Token-preference vector | SPE’s deterministic projected token-feature coordinates | Adjusts token logits directly through the preference learner surface. |
| Native cvector GGUF | llama.cpp’s native layerwise direction format | Imports native hidden-state directions and translates their slots into canonical block coordinates. |

The first two are model-state controls; only the output-head kind has a
static vocabulary-logit explanation. Token-preference vectors and learner
state are not hidden-state vectors, and manual/group biases are separate
logit-space controls. The controller stack and replay records preserve these
distinctions.

`policy-editor-vector` manages standalone, model-matched token-preference
artifacts. Extract a vector from an episode or a v4 preset, inspect its
coordinate identity and norms, and explain which vocabulary tokens it favors:

```bash
policy-editor-vector token-preference extract \
  --workspace episodes.sqlite3 --episode '#7' --output preference.json
policy-editor-vector token-preference inspect preference.json
policy-editor-vector token-preference explain preference.json \
  --model model.gguf --top 50
```

Use `validate` to check an artifact against a model, `blend` to combine
artifacts with compatible coordinates, and `apply` to write a vector into a
new v4 `biases.json` file. Artifacts record their model identity, projection
seed, feature scheme, coordinate identity, slow/fast vectors, strengths, and
source metadata. The workbench refuses to combine vectors from incompatible
models or feature bases.

The workbench also creates output-head steering vectors from two prompts. It
computes Prompt A minus Prompt B at the post-normalization output-head input,
then the runtime
projects that direction through the model output head into vocabulary logits.
The default artifact is unit-normalized and can be loaded directly by the
editor:

```bash
policy-editor-vector output-head create \
  --model model.gguf --backend llama.cpp \
  --prompt-a "Answer warmly and briefly." \
  --prompt-b "Answer coldly and at length." \
  --output style.json
policy-editor-vector output-head explain style.json \
  --model model.gguf --top 50
policy-editor --model model.gguf --new-prompt "Hello" \
  --steering-vector style.json
```

The artifact is explicitly typed as `output-head-steering-vector`; it is not a
vocabulary-sized logit vector. The selected vector, target, strength, and model
identity are persisted in sampler segments and included in replay state.
`inspect`, `validate`, and `blend` are available under `output-head` as well.

Paired episodes can now seed new vectors. Put desired or teacher-intervened
episodes on the positive side and baseline or contrasting episodes on the
negative side; entries are paired by list position. `derive` averages their
positive-minus-negative final hidden states for output-head steering, while
`export-pairs` writes the same episode texts as escaped prompt files for llama.cpp's
`llama-cvector-generator`:

```bash
policy-editor-vector output-head derive \
  --workspace episodes.sqlite3 \
  --positive desired-1 desired-2 \
  --negative baseline-1 baseline-2 \
  --model model.gguf --output desired-behavior.json

policy-editor-vector hidden-state export-pairs \
  --workspace episodes.sqlite3 \
  --positive desired-1 desired-2 \
  --negative baseline-1 baseline-2 \
  --output-dir cvector-input
llama-cvector-generator -m model.gguf \
  --positive-file cvector-input/positive.txt \
  --negative-file cvector-input/negative.txt \
  --method mean --output control_vector.gguf
policy-editor-vector hidden-state import-cvector control_vector.gguf \
  --model model.gguf --output desired-cvector.json
```

The pair exporter records a manifest with episode IDs, fork ancestry,
boundaries, text hashes, and token counts. Pairing is analysis-only: it does
not modify the workspace or infer that a teacher intervention is universally
correct. Evaluate derived vectors with `output-head validate`, `explain` (for
output-head vectors), and the `impact`/`compare` workbench reports before
using them on new episodes.

Both backends can create hidden-state vectors directly at the decoder-block
residual boundary. The portable coordinate is the residual stream immediately
after a canonical one-based decoder block; one width-sized direction is stored
for every model block, with only the requested range active. The native worker
also reports three explicitly separate endpoint sites: embedding output, the
final pre-output-normalization residual, and the post-normalization
output-head input. They are not interchangeable hidden-state coordinates.

Layer numbers are canonical one-based decoder-block numbers in this command:

```bash
policy-editor-vector hidden-state create \
  --model SmolLM-360M-Instruct --backend transformers \
  --prompt-a "I am calm." \
  --prompt-b "I am angry." \
  --layer 12 --output calm-vs-angry.json
policy-editor-vector hidden-state validate calm-vs-angry.json \
  --model SmolLM-360M-Instruct --backend transformers
policy-editor --model SmolLM-360M-Instruct --backend transformers \
  --new-prompt "How are you?" --steering-vector calm-vs-angry.json
```

Use `--layer-range START END` to activate several layers. The backend records
the native module path and per-layer type in its capabilities; this matters for
hybrid models such as Qwen3.5, where some blocks use linear attention and others
use full attention. The portable target remains the decoder block-output
residual stream; lower-level attention or recurrent-state vectors require an
explicit backend-specific coordinate.

For llama.cpp, the normal Python adapter captures the selected per-token
residual states directly through llama.cpp's native layer-input extraction API:

```bash
policy-editor-vector hidden-state create \
  --model model.gguf --backend llama.cpp \
  --prompt-a "I am calm." \
  --prompt-b "I am angry." \
  --layer-range 8 12 --output calm-vs-angry.json
policy-editor-vector hidden-state validate calm-vs-angry.json \
  --model model.gguf --backend llama.cpp
```

The llama.cpp adapter captures all requested token positions in one evaluation
per prompt, then the artifact creator selects `first` or `last` for the
prompt-pair direction. Its layer-input tap captures canonical block outputs
`1..N-1`. The native cvector slot is offset by one: canonical block output `N`
maps to native slot `N-1`, so the final block is runtime-steerable even though
it is not available through that capture tap. Canonical block output 1 remains
capture-only with the upstream cvector adapter because it has no slot 0.
Requests outside the adapter's capture range are rejected rather than silently
steering a different block. The final pre-normalization residual and the
post-normalization output-head input are separate sites.

There is also a narrow native worker for installations where the Python
binding's private capture symbols are not a comfortable compatibility boundary.
Build it against the llama.cpp checkout used by SPE, then ask the workbench to
use it for capture:

```bash
bash tools/build_spe_llama_worker.sh
policy-editor-vector hidden-state create \
  --model model.gguf --backend llama.cpp \
  --worker build/spe-llama-worker \
  --prompt-a "I am calm." \
  --prompt-b "I am angry." \
  --layer-range 8 12 --output calm-vs-angry.json
```

The worker is a one-shot native compatibility process, not an inference
server. It links directly to the selected llama.cpp build, captures the
requested canonical block range plus the embedding, final pre-normalization,
and post-normalization endpoint sites, and writes a small versioned JSON
response. The final block is captured from llama.cpp's evaluated graph output,
not guessed from a neighboring layer-input tap. The workbench validates the
selected range and converts it into the same portable
`spe-steering-vector-v1` artifact used by the normal backend. If llama.cpp's
internal capture interface changes, rebuilding the worker fails at its build
boundary instead of leaving a runtime symbol lookup silently pointed at the
wrong ABI. A worker JSON request may include canonical layer 1 for inspection,
but a loadable llama.cpp hidden-state artifact must target layers `2..N`,
because upstream cvector has no slot 0. Set `SPE_LLAMA_CPP_ROOT` or
`SPE_LLAMA_CPP_BUILD` when the checkout or build is not adjacent to the SPE
repository.

`export-pairs` plus `llama-cvector-generator` remains available when you want
llama.cpp's PCA/mean training workflow or a native GGUF cvector:

```bash
policy-editor-vector hidden-state import-cvector control_vector.gguf \
  --model model.gguf --backend llama.cpp --output mood-cvector.json
policy-editor-vector hidden-state validate mood-cvector.json \
  --model model.gguf --backend llama.cpp
policy-editor --model model.gguf --new-prompt "Hello" \
  --steering-vector mood-cvector.json
```

These artifacts translate native cvector directions into the canonical layout.
Native `direction.1` maps to canonical block output 2, and native
`direction.N-1` maps to canonical block output `N`; a leading zero chunk records
the capture-only canonical layer 1. The editor installs them through llama.cpp's
control-vector API and rebuilds the current prefix when that runtime state
changes. Hidden-state vectors are inspectable and replayable; token-level
`explain` is intentionally available only for output-head steering because a
hidden-state intervention is not a static output-logit offset.

The portable envelope is `spe-steering-vector-v1` with an explicit kind of
`output-head-steering-vector` or `hidden-state-vector`. Older ambiguous
`activation` artifacts are rejected with a recreation message rather than
silently guessing which runtime surface they meant.

Typed answers provide the same kind of live supervision when a learner is
enabled; learning from writes is enabled by default and `--learn-from-write`
may be used explicitly. Each typed token is
evaluated using the observation immediately before it, then receives its own
bounded update before the next token is observed.
There is no write-level sum, average, or square-root reduction, and the policy
can change between written tokens. Learnable groups update only when the
selected token is an active positive member; unrelated selections leave them
unchanged. Appearance objectives remain separate. Use `--no-learn-from-write`
to restore the legacy opt-out. Accept, EOG, and replay remain non-learning
paths.

Learning notices now summarize the cause of an update or skip. Type **`learning`**
at a token choice or the live edge for the latest teaching event's token evidence,
learning/decay/clipping breakdown, and explained slow/fast memory sizes. The report
opens on demand without generating tokens. See [reading learning feedback](STEERING.md#reading-learning-feedback).

Three further experiments are available for both learners:
`--token-preference-decay-on rejection` skips decay on agreement; `evidence` also skips it
inside the learning dead zone. The legacy `--token-preference-write-reduction mean` or `sqrt`
settings apply only to compatibility aggregate helpers, not normal live writes.
`--token-preference-rejection-target sampler` uses the actual sampler's
weighted alternatives as the negative target when rejection strength is nonzero.
Use `--learning-` in place of `--token-preference-` for manual group fitting. Defaults remain
`update`, `sum`, and `proposal`. See [semantics](STEERING.md#teacher-learning-experiments)
and [paired trial commands](CONFIG_TRIALS.md#14--protect-memory-on-acceptance-or-on-all-gate-skips).

Additional experimental token preference controls are optional:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--token-preference-decay` | `0` | Fraction of old memory forgotten per committed learning intervention, from 0 to 1. |
| `--token-preference-severity-cap` | `1000` | Positive rank distance above the dead zone where severity reaches 1. |
| `--token-preference-no-severity-attenuation` | on | Give every selection full severity, including rank-one and sampler-eligible choices. |
| `--token-preference-dead-zone-rank` | `0` | Legacy attenuation mode's rank threshold; zero leaves no rank in the dead zone, and the setting is ignored by the normal full-severity mode. |
| `--token-preference-rejection-strength` | `0` | Nonnegative rejection pressure when the proposal differs from the chosen token; the negative target defaults to that proposal. |
| `--token-preference-fast-slow` | off | Also learn an independent fast vector in the same feature space. |
| `--token-preference-projection-seed` | `9137` for new episodes | Signed 64-bit projection seed; restored episodes keep their saved seed. |
| `--token-preference-random-projection-seed` | off | Draw and print one concrete token preference projection seed at launch; mutually exclusive with `--token-preference-projection-seed`. |
| `--token-preference-feature-scheme whitened-projection-v2` | v1 | Center and whiten the projected feature vocabulary, with one global norm scale. |
| `--token-preference-learning-scheme fisher-kl-v2` | v1 | Use a damped Fisher direction and an exact canonical-policy KL line search. |
| `--token-preference-learning-metric fisher` | euclidean | Compatibility alias for the Fisher learner. |
| `--token-preference-learning-kl` | `0.05` | Exact canonical preference-policy KL budget per correction or atomic Write. |
| `--token-preference-fast-learning-kl` | slow budget | Separate fast-memory adaptation budget; deployment still uses one combined influence budget. |
| `--token-preference-fisher-mode diagonal` | diagonal | Diagonal Fisher is the inexpensive default; `full` uses configured policy support. |
| `--token-preference-fisher-ridge` | `0.001` | Damping used by the Fisher solve. |
| `--token-preference-influence-mode kl` | manual | Calibrate the combined token preference intervention to `--token-preference-influence-kl`. |
| `--token-preference-influence-kl` | `0.05` | Nominal total preference deployment KL target in automatic influence mode. |

Normal severity is `1.0` for every explicit teacher selection. With the default
rejection target, strength 1 gives a chosen-versus-proposal pairwise direction;
values above 1 add stronger rejection pressure. Selecting the sampled proposal
retains the expectation-based direction. The legacy severity formula is
`min(1, log1p(max(0, policy_rank - dead_zone_rank)) /
log1p(severity_cap))`, and is available with the explicit attenuation setting.
`--token-preference-no-severity-attenuation` is the normal default; it applies
to both token preference channels and typed-write evidence. It does not remove the
step or norm limits, alter decay, or change the named-group learner. Lowering
the severity cap also strengthens smaller misses; cap 1 already gives full
severity to every rank outside the dead zone. The explicit flag makes that
experimental intent visible in the launch command and learning diagnostics.
Disabled learners neither learn nor decay. Step clipping bounds new learning;
norm clipping bounds the combined state after forgetting and learning. Net
update norms include forgetting and any state clipping, so they can exceed
the learning step limit.

The v2 learner keeps preference memory separate from its deployment actuator.
It learns from the canonical policy `q(z) ∝ p0 exp(Fz)`, where `p0` is the
base policy before token preference actuation, so changing `--token-preference-strength` does not change the evidence
being learned. The Fisher matrix selects the natural-gradient direction, while
`--token-preference-learning-kl` is enforced by an exact policy-KL line search; the
reported `predicted_fisher_kl` is only the quadratic estimate. Rejections use a
pairwise logistic loss, so a well-separated chosen/proposal pair naturally
receives less additional pressure.

In automatic influence mode, slow and fast scores are first combined and one
global gain is calibrated against the base policy before token preference actuation. Thus the requested
influence describes the total token preference intervention. The fast strength is the
relative fast-channel weight in this mode; `--token-preference-strength` is the user
multiplier applied to the calibrated actuator. Learning notices expose both
the exact learning KL and the actual deployment KL, along with gain and safety
clamps.

With `--token-preference-fast-slow`, omitted fast controls resolve to:

| Flag | Initial experimental default |
| --- | --- |
| `--token-preference-fast-learning-rate` | 4 times the slow learning rate |
| `--token-preference-fast-decay` | `0.10` |
| `--token-preference-fast-strength` | Half the slow strength |
| `--token-preference-fast-max-step` | The slow max step |
| `--token-preference-fast-max-norm` | The smaller of 1 and the slow max norm |

The existing controls govern the slow vector, `token_preference_vector`. Fast
memory uses `token_preference_fast_vector`; both contribute to sampler logits
using their own strengths and share one projection. Restored fast state still
contributes to inference when fast learning is off; it is then left unchanged.
Saved nonempty vectors determine the feature dimension. Interaction payloads
record forgetting, learning step, net update, and memory norms for each
channel, plus per-token rejection information for Writes.

The concrete projection seed is persisted with the vectors in sampler
segments and full bias presets. Old records without a seed use 9137. Ordinary
replay follows each original segment's seed and vectors and never rerolls a
token preference projection seed or performs fresh learning. Conflicting explicit seeds (including
conflicts in later segments) are rejected; randomizing the projection seed
during replay requires `--fixed-config`. An explicit `--biases` preset replaces
its corresponding steering fields, including token preference vectors and seed, in every
replay segment. An explicit `--reference` similarly overrides reference state;
other fields continue to follow the source. Changing the seed on
resume, fork, or fixed-config replay clears both token preference vectors and prints a
reset notice, because their coordinates would otherwise have changed meaning.
The same versioned-coordinate check applies to preference dimension, feature
scheme, and whitening ridge. The saved identity includes the model embedding
fingerprint when the backend exposes it; incompatible nonzero slow/fast memory
is reset for explicit CLI/headless setting changes, while a loaded-model
fingerprint mismatch is rejected before the new basis is used. Ordinary
rewind/replay restores the historical identity and vectors exactly.

Export all active steering and vectors with `--project EPISODE --biases-only`,
and reload with `--biases biases.json`. See [Steering](STEERING.md#export-and-import-biasesjson)
for the complete v3 schema and portable-history behavior.

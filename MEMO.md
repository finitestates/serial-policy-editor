# Episode-local online SGD implementation

Implemented in `/home/realityisfire/online-learning` on the cloned
`policy-editor-argmax` branch, starting at commit
`781ec0b11470c2fc8f38447aebde05de1d49c6a2`. Changes remain local and uncommitted.
`ORDER.md` was preserved.

## What works

- Opt-in `--online-sgd`, `--online-lr`, and `--online-steps`, separate from
  `SamplerConfig`. Learning is disabled by default.
- Actual full-model PyTorch SGD, with no momentum, weight decay, KL loss,
  adapters, or bias-only substitute. The model remains in evaluation mode;
  the supervised forward explicitly enables autograd outside inference mode.
- Successful human rank overrides, written spans, forced phrases, and explicit
  EOG overrides supply supervision. Accept, autonomous Hold, selection of the
  existing proposal, diagnostic phrase checks, rejected actions, and replay
  do not train. Replay itself is explicitly unsupported in online mode.
- Written spans retain the existing batch-commit path. A separate forward
  uses the original prefix plus all corrected tokens except the last as
  input; logits starting at `len(prefix)-1` predict the corrected tokens.
  Cross-entropy is averaged over the span for every requested SGD step.
- `learn steps=N`, `learn lr=X`, `learn steps=0`, `learn status`, `learn reset`,
  and `teach last steps=N` work during an active episode. Pausing automatic
  updates still remembers the latest correction for `teach last`.
- Status reports correction count, optimizer steps, recent losses and gradient
  norms, configuration, and weight revision. Automatic updates print a brief
  training summary. Events retain exact pre-correction context, committed token
  IDs, original proposal, boundary, rate, step count, losses, gradients, and
  revision. `teach last` reuses the original context without changing defaults.
- Updates and reset invalidate KV, last-logit, activation-logit, observation,
  and speculative state. Revisions participate in observation validity.
  Speculative warming is disabled in this mode. Retained preexisting batch
  sessions are closed when learning is enabled; new batch sessions are rejected.
- Reset restores the original state dict, clears learned history and gradients,
  and retains the trajectory. Fresh episode roots and discard restore base
  weights; old episode handles cannot continue using another episode's weights.
- A training failure blocks inference, preserves the committed correction in
  the session action history, and offers reset or quit in the CLI. Reset
  recovers original weights even after an injected partial optimizer mutation.
- At EDGE, `export /absolute/path/experiment.json` writes a trajectory and
  diagnostics artifact (`online-sgd-experiment-v1`). It includes text, token IDs,
  actions, learning events, status, backend provenance, and terminal state.
  It explicitly declares `replay_supported: false`. Export before `learn reset`
  if the current learning history is valuable; reset deliberately clears it.

## Running it

Dependencies are installed in the workspace's ignored `.venv` (CPU Torch
2.14.1+cpu and Transformers 5.19.0). To recreate the environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -e 'core[test,transformers]' \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

For a local, unquantized float32 causal-LM checkpoint:

```bash
.venv/bin/policy-editor \
  --backend transformers --model /path/to/model \
  --transformers-device cpu --transformers-dtype float32 \
  --new-prompt 'The house had been empty for fifty years.' \
  --temperature 0 --online-sgd --online-lr 1e-5 --online-steps 1
```

The native terminal UI and `--plain-ui` both recognize the commands. Enter a
candidate's raw rank to override, or `x replacement text` to insert an exact
span. Use `q` to reach EDGE, then `export /path/experiment.json` to retain the
experiment or `c` to return to the episode. Run the same launch without
`--online-sgd` for the frozen comparison. The sampler is unchanged.

A fully offline smoke experiment generates a tiny random GPT-2 checkpoint
and exercises a real CLI subprocess:

```bash
.venv/bin/python benchmarks/online_sgd_smoke.py --output /tmp/online-sgd-demo
```

Its checkpoint can also be opened interactively using the launch above with
`--model /tmp/online-sgd-demo/checkpoint --new-prompt P --online-lr 0.05`.
This random eight-token model demonstrates mechanics, not useful writing.

## Validation and observed behavior

Fresh validation:

- `.venv/bin/pytest -q tests reference-kernel -rs`: **922 passed, 1 skipped**.
  The skip is the optional local GGUF oracle; `SPE_ORACLE_LLAMA_MODEL` is unset.
- After the last menu refresh and optional-dependency test adjustments,
  `.venv/bin/pytest -q tests/core/test_online_learning.py tests/core/test_menu_contracts.py`:
  **53 passed** (including all 17 real-model learning tests).
- The real CLI smoke subprocess exited **0** and verified export contents.
- `git diff --check` passed.


Focused tests use a real local, float32 GPT-2 with 4,016 parameters and a real
saved tokenizer/checkpoint. They verify parameter changes, nonzero gradients,
increased target likelihood, span alignment, exact step counts, original-context
retraining, reset, same-prefix cache invalidation, episode isolation, paused
learning, diagnostic/rejected/accepted negative controls, unsupported operations,
CLI configuration, dtype rejection, stale batches, interactive menu refresh,
discard, and recovery from failure after actual parameter mutation.

The reproducible smoke artifacts are retained in `evidence/online-sgd/`:
`summary.json`, `cli-experiment.json`, and `cli-transcript.txt`.
In the seeded five-step comparison at learning rate 0.05, span loss fell from
2.087945 to 1.076284; the first corrected token's probability at its original
prefix rose from 0.128470 to 0.330487. Frozen generation left those logits
unchanged. Both runs proposed token 4 after the written span. The CLI separately
applied 3 automatic steps plus 5 additional steps, exported all 8 steps, and
reset without losing the written trajectory.

These are real-model numerical and plain-CLI subprocess results. Menu tests
exercise the actual policy and request renderer with scripted input. No
physical terminal visual inspection, pretrained natural-language trajectory
study, GPU validation, or cross-backend learning claim is made.

## Difficulties and boundaries

The shared backend makes existing branch semantics unsafe without learning
reconstruction. This implements Phases A and B as a linear-episode prototype.
Rewind/review, fork, branch/episode switching, suspended-session reactivation,
replay/resume/teacher tapes, beam, chord, and replayable SQLite saves fail
explicitly in online mode, even when automatic steps are paused. Creating an
unrelated new root is supported and restores base weights. Experiment export
retains the valuable trajectory without pretending it can replay learned state.

Supported models must have entirely trainable float32 parameters, no
quantization, no device_map, and no multimodal wrapper. CFG is unsupported.
An in-memory CPU copy of the original full state dict enables exact reset;
this costs approximately another checkpoint's worth of host RAM, in addition
to gradients and training activations. Full-model training can be slow and
memory-intensive; no clipping or unrelated-capability preservation was added.

The broader tests exposed a regression in my initial switching guard: it
accessed the public engine after the frozen-mode session was detached. The
untouched branch passed the affected tests. The guard now uses retained engine
state, and the affected CFG switching tests pass.

## Outstanding work

Phase C remains: causally associate learning events with durable episode
records and reconstruct branch-local weights by resetting the base and
replaying updates. It needs explicit semantics for runtime reset/configuration,
retraining events, and failures before enabling time travel or saved replay.

The central research question remains untested: whether this reduces subsequent
human interventions on useful pretrained-model trajectories. Immediate loss
reduction is a sanity check, not evidence of that result. A next experiment
should compare frozen and learning runs from the same checkpoint/prompt, record
interventions and later failures, and report training time as well as trajectory
quality. No natural-language checkpoint was downloaded for this implementation.

# Real-model inference checks

This harness is opt-in. The ordinary `tests/core` suite remains model-free. Run
it from a development checkout with the package installed editable in `.venv`:

```bash
.venv/bin/python -m pip install -e './core[llama]'
# For a Transformers profile, also install the optional stack:
.venv/bin/python -m pip install -e './core[transformers-accelerate]'
```

The example root below is local to one machine. It is never a program default.
No model is downloaded by this harness. A selected missing path is an error.

```bash
.venv/bin/python -m benchmarks.real_model run \
  --model-root /home/realityisfire/Documents/serial-policy-editor-lab/models \
  --profile benchmarks/profiles/llama_1b_smoke.yaml \
  --profile benchmarks/profiles/gpt2_cpu_smoke.yaml
```

Profiles run sequentially; backend resources are released between profiles.
Use `--model /absolute/path` and `--backend llama.cpp|transformers` to override
the selection, `--scenario NAME` to select checks, `--repeats N` for more
samples, and `--output-dir PATH` for artifacts. Relative profile model paths
resolve only against `--model-root`. Launch settings are validated through the
same controller-profile and CLI parser as normal SPE. Harness settings stay at
the profile top level. The `*_benchmark.yaml` profiles use three repeats. The
`*_cfg.yaml` profiles require a second model copy and exercise CFG explicitly.

Each run produces a versioned JSON report and a short text summary in
`/tmp/spe-real-model-results` by default. Paths are unique and earlier runs are
not overwritten. A report includes raw samples, median and min/max for elapsed
time, model checksum and provenance, launch settings, hardware, actual tokens,
work counts, outcomes, failures, and exact measurement boundaries. A failure
returns nonzero and still leaves a partial report. An explicitly selected check
cannot silently skip. The `cli-jsonl` case has its own subprocess wall time;
that value includes interpreter startup and model loading.

Compare two compatible reports with:

```bash
.venv/bin/python -m benchmarks.real_model compare /path/to/before.json /path/to/after.json
```

The comparator rejects changed model content, backend/settings, hardware,
fixture version, failures, or realized token workload before calculating a
percentage. For meaningful timing comparisons, alternate before/after runs on
the same machine, use the same profile and harness version, and inspect all
individual samples. Three repeats give a median and range, not a reliable
tail percentile. The controlled-write scenario uses a fixed handwritten
procedure; free continuation is reported separately. A faster GPU may raise
the percentage outside inference even when absolute SPE overhead falls.

## What is timed

`active_wall_s` starts after model load, before scenario prompt/context
preparation, and ends after its specified final action. It includes normal
observation/sampling, token-ledger work, and session finalization. The in-process
JSONL scenarios also include parsing/export, each marked as a phase. Reference
full-prefix oracles, warmup, model hashing, report writing, and input waits are
outside this interval. A phase remainder remains explicit. Rendering a rich
terminal is outside these headless scenarios; `tui_transitions.py` is a
separately labeled terminal-only benchmark.

`backend_eval_wall_s` is the union of reset, eval, and branch-positioning
service intervals, including adapter work, cache handling, logit retrieval,
and transfer. Nested reset calls are counted once. Subtracting it from active
time gives the coarse `outside_backend_eval_wall_s`, reported in milliseconds,
per action, per committed token, and as a percentage. Model-call counts and
input positions are collected at the actual backend model-input call, including
full-prefix rebuilds and retries. Context length, kind, conditional/CFG role,
and cache fallbacks are recorded. Input positions are not attention FLOPs.

The narrower `model_call_wall_s` uses synchronous `Llama.eval` for llama.cpp
and the completed model forward on CPU Transformers. These include library
bookkeeping; neither is pure neural-network time. CPU logit transfer remains
inside the backend service. Accelerator forward calls may return before the
device finishes, so their narrow wall time is null; the coarse service remains
available. No global blocking or extra synchronization is enabled. This follows
[PyTorch's asynchronous CUDA timing guidance](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution).

## Check scope and current limit

The smoke set covers continuation, fixed writes, candidate preparation,
measurement-on/off semantic parity,
handwritten action-only JSONL, live-exported observed JSONL and deliberate
handoff, rewind/replacement, fork/switch, temporary-database save/resume,
longer context, cache-auto/off, and a real CLI subprocess. The CLI runs
action-only, embedded-envelope, and sidecar-envelope files with explicit
ballistic selection and a finite `end` response at the live edge. The source
envelopes contain deliberately nonexistent source paths; the destination model
is selected explicitly. Database construction is forbidden in the in-process
JSONL cases. A startup audit hook in each CLI subprocess rejects any SQLite
connection or open of the default/selected workspace; the fresh directory is
also checked for workspace, journal, and WAL files. Malformed files are checked
before model loading by the fast tests.

Numerical oracles use fresh full-prefix inference on exact token IDs. The
long-context case runs eight production continuation actions uninterrupted,
saves the logits at every prefix, then evaluates each saved prefix fresh outside
the timed interval. It collects the entire numerical/top-token trajectory
before deciding pass or fail, so an early numeric mismatch cannot hide a later
top-token divergence. The profile's `rtol` and `atol` apply to the entire logit
vector and are not automatically widened after failure.

Just as an example: On the local CPU Q4_K_M GGUF reference in one trial, we observed a numerical mismatch at checkpoint 2 (maximum 0.403), which peaked at 0.511 at checkpoint 5, and persisted through checkpoint 8 (0.433). The top token continued to agree throughout the duration of the run. The harness will report this as a failure, but the episode would still be replayable. One of the reasons that replay has the definition it does is to separate numerical failures from "practical effects." Interestingly, the numerical divergence widened, but then narrowed again, which suggests at least in this limited episode that the numerical divergence didn't "snowball" into greater downstream effects. (This matches what has been observed elsewhere about successful replay episodes being conducted using handoff rules at lengths far longer than what the harness currently tests for. The current "record" for maximum replay length without token selection divergence is ~700 matched teacher actions using GPT-2).

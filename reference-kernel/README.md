# Reference kernel and model oracle

`reference_kernel.py` is an independent standard-library implementation of the
editor's deterministic draw coordinates, candidate filters, history penalties,
and editing semantics. It imports neither production sampling nor the episode
engine. It supports categorical, Gumbel, Gaussian, logistic, Laplace, uniform,
and Student-t draws (including configurable degrees of freedom), Gumbel noise
scale and token-ID/model-rank addressing. Filter order matches production:
temperature → top-k → typical → tail-free → top-p → min-p. History penalties
precede direct token biases. Unfiltered categorical CDFs retain token-ID order.

`reference_kernel_oracle.py` adds only full-prefix model evaluation and two modes:

- `atomic`: evaluate once and print a candidate list with the proposal marked `*`.
- `pattern`: accept proposals for up to `--count` steps, stopping at EOG. Print
  each choice and a final result. EOG is recorded separately from visible tokens.

No UI, episode persistence, or incremental KV-cache bookkeeping is involved.
Full-prefix reevaluation is intentionally slower than the editor. Both llama.cpp
and Transformers causal language models are supported. Install the applicable
core extras (`llama` or `transformers`) in your Python environment.

From the repository root:

```sh
python reference-kernel/reference_kernel_oracle.py atomic \
  --backend llama --model /path/model.gguf --prefix 'The capital of France is' \
  --top-k 40 --top-p 0.95 --min-p 0.05 --menu-size 12

python reference-kernel/reference_kernel_oracle.py pattern \
  --backend transformers --model /path/hf-model --prefix-file prompt.txt \
  --count 20 --device cpu --dtype float32 --draw-kernel student-t-max \
  --student-t-df 7 --format jsonl --output continuation.jsonl
```

Use `--help` for all model/sampler flags. Oracle defaults leave filters disabled
(`top-k=none`, `top-p=1`, `min-p=0`); pass explicit production settings when
comparing. `--biases '[[123, 0.5]]'` adds direct token biases;
`--excluded-token-ids '[456]'` removes candidates.

`--menu-size N` counts **all** displayed entries, including the proposal. If the
proposal falls outside the first N candidates by probability, it replaces the
last entry. Zero displays the entire active support. Display limits never affect
the draw. `--detail tokens|probabilities|full` controls row fields. Full rows
include effective score, raw logit, model rank and EOG status. These are plain
lists, without the editor's grouped/chord/beam menu machinery.

JSONL begins with a `config` record (model settings, exact prefix token IDs,
policy, seed, root fingerprint and starting boundary), followed by `choice`
records, and a `result` for pattern mode. Plain text escapes candidate text so
newlines and control characters do not break the list; the final continuation
is decoded as a whole. Token IDs remain authoritative: individual decoded
fragments can contain replacement characters for incomplete UTF-8 bytes.

For exact replay, use `--prefix-token-ids '[1,2,3]'`. A resumed prefix also needs
`--stream-fingerprint ROOT_SHA256 --boundary X`: hashing the resumed prefix as a
new root changes its random coordinates. llama defaults to adding BOS; use
`--no-bos` to disable it and `--special` to recognize special-token spellings.
Transformers follows its tokenizer's `add_special_tokens` template unless
`--no-bos` is set (that template can add tokens beyond BOS). Use exact IDs to
avoid tokenizer-template differences. `--special` applies only to llama.

Model numeric settings, batching, precision and backend versions can change
logits and therefore trajectories near draw/filter thresholds. Match those
settings when comparing. CFG, grouped phrase biases, steering vectors, beam
search and editor actions are outside this minimal CLI's scope.

## Validation and coordinate demonstration

```sh
PYTHONPATH=core/src python -m pytest reference-kernel -q
python reference-kernel/draw_coordinates_demo.py
```

Parity tests compare independent formulas with production filters/draws and
scripted EpisodeEngine sessions. The coordinate demonstration keeps the
probability vector fixed and changes only the boundary; its JSON shows two
quantiles selecting different tokens. A seed alone does not specify a draw.

A real llama parity check is opt-in:

```sh
SPE_ORACLE_LLAMA_MODEL=/path/model.gguf PYTHONPATH=core/src \
  python -m pytest reference-kernel/test_reference_kernel_oracle.py -q
```

It compares the full-prefix oracle with production's incremental backend and
`Hold` under matching token IDs and sampler settings. No models are downloaded
by the tests.

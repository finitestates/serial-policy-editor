# Real-model Rust Choice boundary probe

**Status:** revalidated on 2026-10-03 with the corrected pre-0 prompt
coordinates for both smoke profiles.

The fixed prompt remains in the backend prefix and supplies the production
token-prefix fingerprint. It is outside generated visible history, so the two
Choice boundaries are `0` and `1`.

## Current result

The standalone Rust terminal completed two consecutive Choice turns through
one persistent Python worker for each profile. Rust calculated candidate
views and proposals from framed full-vocabulary logits. The worker used the
existing profile loader and production backend factory. Both accepted actions
used the Rust submit and typed episode-history path from fixture mode.

The exact live-logit Python oracle and a fresh-backend replay agreed with Rust
at both boundaries. The fresh replay prefills the exact root prompt, then
evaluates each captured generated token in its own incremental call. This
matches the live operation boundaries while rebuilding the complete prefix in
independent backend state.

| Profile | Vocabulary | Boundary 0 token | Boundary 1 token | Fresh replay max logit delta |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 Transformers CPU | 50,257 | 198 | 198 | 0.0; 0.0 |
| Llama 3.2 1B Q4_K_M CPU | 128,256 | 264 | 47218 | 0.0; 0.0 |

The sampler used temperature 0.8, `top_k: 5`, `top_p: 1.0`, `min_p: 0.0`,
`typical_p: 1.0`, `tail_free_z: 1.0`, categorical drawing, token-ID
addressing, and seed 17. Both backends returned float32 logits; the worker
converted them to f64le for Rust. Rust validated the vector shape, vocabulary
count, payload length, and finiteness.

The screen oracle checked six frames per profile at 100×30, 80×24, and
restored 100×30. It derived expected model text and candidate rows from the
Python backend and sampler, replayed raw PTY bytes with pyte, checked cursors
and complete cell rows, and rejected the negative control with the proposal
row removed.

Timing and transfer are boundary diagnostics, not speed comparisons. Totals
include worker startup and model loading.

| Profile | Backend service wall time | Outside-backend wall time | Round-trip wall time | Total bytes | Logit payload |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-2 Transformers CPU | 10.162977 s | 0.235573 s | 10.398550 s | 815,623 | 804,112 |
| Llama 1B Q4_K_M CPU | 1.850043 s | 0.230704 s | 2.080748 s | 2,063,409 | 2,052,096 |

One diagnostic run initially replayed the Llama prefix as a single reset
batch. The model returned the same candidate IDs but changed logits by up to
0.323943 and selected 2363 instead of the live token 47218 at boundary 1.
Replaying the same exact prefix with the root-prefill and one-token continuation
boundaries removed that batch-shape difference; both fresh decisions then
matched with zero logit delta. The failed diagnostic artifacts are retained in
`/tmp/spe-rust-real-model-choice/20261003T133917Z-1826352e/`.

The passing versioned reports retain per-operation timings, request/response
bytes, source and wire dtypes, model and tokenizer identities, provenance,
sampler settings, decision semantics, fresh-prefix diagnostics, and artifact
paths:

- [GPT-2 report](/tmp/spe-rust-real-model-choice/20261003T134408Z-5c03c44d/report.json)
- [Llama report](/tmp/spe-rust-real-model-choice/20261003T134245Z-bade53c9/report.json)

The saved October 2 captures below are historical: they counted the eight
prompt tokens as trajectory history and used boundaries 8 and 9. They remain
unchanged as historical evidence and do not replace the corrected runs above.

| Backend and model | Vocabulary | Historical boundaries and selected tokens | Historical fresh-prefix maximum logit delta |
| --- | ---: | --- | ---: |
| Transformers, local GPT-2 | 50,257 | 8 → 198; 9 → 198 | 0.0; 0.000579833984375 |
| llama.cpp, Llama 3.2 1B Q4_K_M | 128,256 | 8 → 362; 9 → 47218 | 0.0; 0.313051700592041 |

## Historical validation before the prompt-coordinate correction

Both optional inference extras were installed into core/.venv, the same
interpreter used by the runner. From the repository root, the completed
commands and outcomes were:

    cargo fmt --manifest-path Cargo.toml --all -- --check
    passed
    cargo test --manifest-path Cargo.toml --workspace --locked
    passed: 11 Rust tests across episode history, sampler, and worker protocol
    cargo clippy --manifest-path Cargo.toml --workspace --all-targets --all-features --locked -- -D warnings
    passed
    core/.venv/bin/python -m pytest -q tests/core/test_rust_real_model_protocol.py
    passed: 4
    core/.venv/bin/python -m pytest -q tests/core/test_live_terminal_pty.py -k compiled_rust
    passed: 2, 12 deselected
    core/.venv/bin/python -m pytest -q tests/core/test_live_terminal_pty.py
    passed: 14
    core/.venv/bin/python terminal-ui/scripts/real_model_choice.py --model-root /path/to/models --profile benchmarks/profiles/gpt2_cpu_smoke.yaml
    passed
    core/.venv/bin/python terminal-ui/scripts/real_model_choice.py --model-root /path/to/models --profile benchmarks/profiles/llama_1b_smoke.yaml
    passed

The real-model invocation creates an isolated run directory containing the
report, worker metadata and diagnostics, protocol metrics, semantic JSONL,
frame JSONL with cumulative byte offsets, raw PTY bytes, and the two live
f64le logit captures:

- [GPT-2 report](/tmp/spe-rust-real-model-choice/20261002T194714Z-7e9489be/report.json)
- [Llama report](/tmp/spe-rust-real-model-choice/20261002T194734Z-001c04ff/report.json)

The original acceptance scope and gates are archived in
[`REAL_MODEL_CHOICE_BRIEF.md`](REAL_MODEL_CHOICE_BRIEF.md). The direct/grouped
bias implementation is in [`BIAS_SLICE.md`](BIAS_SLICE.md), and Stage 2 numeric
policy evidence is in
[`POLICY_CALCULATIONS_SLICE.md`](POLICY_CALCULATIONS_SLICE.md). This PTY probe
does not select production packaging or claim a performance gain.

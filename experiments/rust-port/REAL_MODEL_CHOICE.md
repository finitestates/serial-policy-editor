# Completed slice: two real-model Rust Choice turns

**Status:** implemented and validated on 2026-10-02.

## Result

The standalone Rust terminal ran two consecutive Choice turns through one
persistent Python worker for both smoke profiles. Rust calculated candidate
views and proposals from framed full-vocabulary logits; the worker loaded the
selected profile through the existing SPE profile loader and production
backend factory. Both accepted actions used the same Rust submit and typed
episode-history path as fixture mode.

The live Rust decisions matched the independent Python sampler when both used
the exact captured live logits. After the live process exited, a fresh backend
replayed each full prefix. Every fresh proposal matched the Rust-selected
token. The fresh logit changes below are diagnostic measurements; they did not
change either sampled token or the top-ten token order.

The saved captures below are historical: that run counted the eight prompt
tokens as trajectory history and therefore recorded sampling boundaries `8`
and `9`. The harness now treats the prompt as pre-0 context. Its token-prefix
SHA-256 remains the 256-bit stream identity, while generated-token history and
sampling boundaries start at `0` and advance to `1`. This coordinate correction
was not rerun; the table below continues to describe the original captures.

| Backend and model | Vocabulary | Sampling boundaries and selected tokens | Fresh-prefix maximum logit delta |
| --- | ---: | --- | --- |
| Transformers, local GPT-2 | 50,257 | 8 → 198; 9 → 198 | 0.0; 0.000579833984375 |
| llama.cpp, Llama 3.2 1B Q4_K_M | 128,256 | 8 → 362; 9 → 47218 | 0.0; 0.313051700592041 |

The fixed sampler used temperature 0.8, top_k 5, top_p 1.0, min_p 0.0,
typical_p 1.0, tail_free_z 1.0, categorical drawing, token-ID addressing,
and seed 17. Its complete effective settings are in each decision record.
Both backends returned float32 logits; the worker converted them to f64le for
Rust. Rust validated the one-dimensional shape, vocabulary element count,
declared byte count, and finiteness before sampling.

The independent screen oracle passed six frames per profile at 100×30, 80×24,
and restored 100×30. It derived expected model text and candidate rows from
the Python backend transcript and sampler, replayed the raw PTY bytes with
pyte, checked cursors and complete cell rows, and rejected its negative
control with the proposal row removed. Model-provided control characters were
rendered as visible <U+....> text.

## Timing and transfer observations

These are boundary diagnostics, not a speed comparison. They include worker
startup and model loading in the backend and round-trip totals.

| Profile | Backend service wall time | Outside-backend round-trip wall time | Total transferred bytes | Logit payload |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 Transformers CPU | 10.095265 s | 0.238585 s | 817,713 | 804,112 |
| Llama 1B Q4_K_M CPU | 1.847259 s | 0.241856 s | 2,065,487 | 2,052,096 |

The versioned reports retain per-operation service and round-trip times,
outside-backend time, request/response bytes, source and wire dtypes, model and
tokenizer identities, provenance, sampler settings, decision semantics,
fresh-prefix diagnostics, and artifact paths.

## Validation

Both optional inference extras were installed into core/.venv, the same
interpreter used by the runner. From the repository root, the completed
commands and outcomes were:

    cargo fmt --manifest-path experiments/rust-port/Cargo.toml --all -- --check
    passed
    cargo test --manifest-path experiments/rust-port/Cargo.toml --workspace --locked
    passed: 11 Rust tests across episode history, sampler, and worker protocol
    cargo clippy --manifest-path experiments/rust-port/Cargo.toml --workspace --all-targets --all-features --locked -- -D warnings
    passed
    core/.venv/bin/python -m pytest -q tests/core/test_rust_real_model_protocol.py
    passed: 4
    core/.venv/bin/python -m pytest -q tests/core/test_live_terminal_pty.py -k compiled_rust
    passed: 2, 12 deselected
    core/.venv/bin/python -m pytest -q tests/core/test_live_terminal_pty.py
    passed: 14
    core/.venv/bin/python experiments/rust-port/terminal-ui/scripts/real_model_choice.py --model-root /path/to/models --profile benchmarks/profiles/gpt2_cpu_smoke.yaml
    passed
    core/.venv/bin/python experiments/rust-port/terminal-ui/scripts/real_model_choice.py --model-root /path/to/models --profile benchmarks/profiles/llama_1b_smoke.yaml
    passed

The real-model invocation creates an isolated run directory containing the
report, worker metadata and diagnostics, protocol metrics, semantic JSONL,
frame JSONL with cumulative byte offsets, raw PTY bytes, and the two live
f64le logit captures:

- [GPT-2 report](/tmp/spe-rust-real-model-choice/20261002T194714Z-7e9489be/report.json)
- [Llama report](/tmp/spe-rust-real-model-choice/20261002T194734Z-001c04ff/report.json)

The original acceptance scope and gates are preserved in
[NEXT_SLICE.md](NEXT_SLICE.md). This probe establishes the tested backend
process boundary for these two profiles; it does not finish roadmap stage 2,
select production packaging, or claim a performance gain.

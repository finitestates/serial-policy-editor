# Persistent TUI checkpoint

Branch: `codex/persistent-tui-controller`.

This checkout starts from `ef95aa9` and incorporates the uncommitted alternate-screen experiment from `fix/persistent-fullscreen-tui`. The original checkout and its edits were preserved.

## Try this checkout

Run from this directory so Python imports the refactored package:

```bash
cd /home/realityisfire/Documents/spe-branch/persistent-tui-refactor/serial-policy-editor
/home/realityisfire/Documents/spe-branch/venv/bin/python -m trajectory_editor.episode_cli --help
```

Replace `--help` with your usual launch arguments. An already installed `policy-editor` command may still refer to the original checkout.

## Implementation

- One prompt-toolkit Application, renderer and terminal raw-mode session spans choices, menu/search updates, review, the live edge, confirmations, notes and paged documents.
- Choice and edge layouts, bindings and buffers are reused. Choice presentation is explicit `ChoiceViewState`; policy interpretation remains in the existing synchronous episode code.
- The UI runs on a dedicated thread. The caller keeps exclusive ownership of the engine, backend and SQLite connection. Preview requests execute on that owner while it waits for input; rendering never invokes backend callbacks directly.
- Submitting a command disables the obsolete view immediately. Input arriving before the next view is ready is consumed, preventing accidental duplicate commits. Ctrl-C still interrupts the episode thread.
- Preview work is cached within each view, bounded, and superseded queued requests are cancelled. Missing insertion previews show a pending state until ready.
- The application remains able to resize while the episode owner is busy. The delayed working indicator was removed in `567aaca` to prevent layout shifts.
- Incidental printed output is captured during fullscreen operation and emitted after terminal restoration. CLI error handling runs after the session closes.

Standalone `read_live_choice()` and `read_live_edge_command()` remain available for callers that want a single synchronous prompt. The CLI uses the persistent session.

## Verification and measurements

Full regression suite: **657 passed, 12 optional tests skipped**. The new real-model smoke test was run separately and passed.

The focused integration tests cover application/renderer reuse, absence of terminal clears between views, stale typeahead, preview thread ownership and errors, interruption, render failures, CLI navigation and final error output. A Unix pseudoterminal test checks raw mode across action boundaries, resize during owner work, and terminal restoration.

The real GGUF smoke test passed with the existing model-enabled `venv` and local Llama-3.2-1B IQ4_XS model. It exercises expansion, rank lookup, insertion, review, rewind, fork, generation and sampler changes, asserting that backend calls stay on the episode thread.

A local comparison of 160 identical prepared-view transitions at 120×40 measured:

| Metric | Current alternate-screen experiment | Persistent application |
|---|---:|---:|
| Median transition | 25.54 ms | 6.46 ms |
| 95th percentile | 39.91 ms | 11.74 ms |
| Median ANSI bytes per transition | 6,188 | 1,643.5 |
| Erase commands across measured transitions | 480 | 0 |

These are indicative local UI measurements. They exclude inference, database work, preparation of view data and terminal-emulator painting. Real-model generation throughput has not been benchmarked. Backend and sampling algorithms were not changed.

Reproduce with `benchmarks/tui_transitions.py`. Its `--package-root` and `--implementation temporary` options compare the original checkout, which contains the alternate-screen experiment. Run the two measurements sequentially under the same interpreter and machine conditions.

The restricted execution sandbox denies writes to Python's internal socketpair, delaying cross-thread event-loop wakeups until an unrelated timer fires. Latency measurements must run in a normal local environment; unrestricted runs removed that test-environment artifact.

## Remaining review

This is an implemented, tested checkpoint ready for hands-on use. Further work should start with user testing in the preferred terminal and a real-model throughput comparison. Cross-platform terminal behavior and additional menu/history data caching have not been evaluated. No new third-party dependencies were added.

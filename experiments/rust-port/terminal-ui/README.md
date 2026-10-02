# Rust terminal UI PTY slice

**Status:** the first PTY smoke slice is implemented as an isolated Rust
binary and Python integration test. There is no production Rust UI or runtime
wiring yet.

## Current implementation

[`src/main.rs`](src/main.rs) is a small terminal process with one editable
ASCII input, Enter submission, a `quit` test command, a visible cursor, and a
geometry line updated after `SIGWINCH`. It renders complete rows in
synchronized, erase-free frame transactions and writes the intended grid and
cumulative PTY byte offset to `SPE_TERMINAL_FRAME_LOG`.

The `Session` in `tests/core/test_live_terminal_pty.py` accepts a child argv
while retaining its existing Python launch path. The Rust integration test
builds this crate once per pytest session, starts it under the same POSIX PTY,
checks input and both resize directions through the existing pyte oracle, and
changes an expected cursor as a negative control. Pytest keeps `pty.raw` and
`frames.jsonl` in its per-test temporary directory.

From the repository root, run the PTY integration test or the complete PTY
journey module with:

```sh
core/.venv/bin/python -m pytest -q tests/core/test_live_terminal_pty.py -k compiled_rust
core/.venv/bin/python -m pytest -q tests/core/test_live_terminal_pty.py
```

The smoke binary is not wired into the released Python package. The later
views, runtime ownership, and packaging steps in the roadmap remain future
work.

## Purpose

First prove that a compiled Rust terminal process can be driven by the current
Python PTY harness and checked with the existing `pyte` replay oracle. Keep
this slice small: it establishes the test seam, not a partial port of the live
editor.

Relevant existing pieces:

- PTY process driver, raw byte capture, frame replay, resize, transaction, and
  completeness checks: `tests/core/test_live_terminal_pty.py`;
- in-process Python-only view/parser harness: `tests/core/term_support.py`;
- required workflow and evidence handoff: `tests/TERMINAL_RENDERING_GUIDE.md`.

## First implementation

1. Generalize the PTY `Session` launcher to accept a child argv. Keep its
   current Python child path and add a path that launches the compiled Rust
   binary. Build that binary once for the test session.
2. Add a tiny Rust process with a fixed screen, a visible cursor, one editable
   input, and a size-dependent line. It should accept printable input and
   Enter, react to `SIGWINCH` by reading the PTY's current size, and exit
   cleanly on its test command.
3. Have the Rust renderer emit intended frames to the file named by
   `SPE_TERMINAL_FRAME_LOG`. Keep the log separate from PTY stdout/stderr so
   it cannot change the terminal stream under test.
4. Run the child through the existing PTY capture and `pyte` replay code. Save
   the raw bytes before assertions, including on failure.

## Frame-log contract

Match the shape consumed by `test_live_terminal_pty.py`:

```json
{
  "sequence": 1,
  "size": [80, 24],
  "lines": ["..."],
  "cursor": [12, 3],
  "end_offset": 184
}
```

- `sequence` increases once per presented frame.
- `size` is `[columns, rows]`.
- `lines` are the intended visible cell rows for that frame.
- `cursor` is `[x, y]` in terminal cells, or `null` while hidden.
- `end_offset` is the cumulative number of raw PTY-output bytes through the
  end of that frame's write transaction. It is a byte count, not a Unicode
  character count.

The Python oracle feeds bytes through each `end_offset`, resizes `pyte.Screen`
to the frame geometry, and compares displayed rows and cursor state. Keep this
as an implementation-neutral sidecar. If the shape needs to change, version
the format and update both Python and Rust producers/consumers together.

## Pass conditions for the smoke slice

- Python starts the compiled Rust child under a real POSIX PTY and captures its
  raw output without interpreting it first.
- The `pyte` replay matches every intended frame boundary, including after a
  key submission and at both directions of a resize.
- Each frame has valid cell geometry and cursor placement; frame transactions
  are complete; the renderer emits no post-startup screen erase.
- A negative control (for example, a changed expected cursor or truncated
  output) proves the replay assertion detects a mismatch.
- The raw capture and frame log remain available in pytest's temporary
  directory when a check fails.

Use deterministic fixture text and no model backend. Keep the current
`pyte.ByteStream` as the independent terminal emulator; do not replace it with
the Rust renderer's own canvas when checking actual output.

## After the smoke slice

Use Rust-side unit tests for view-state and input-reducer behavior, then grow
the real-process PTY journeys as views migrate: Choice, EDGE, prompt, review,
Beam, chord, mouse input, cancellation/suspend, and generated resize/input
sequences. Check every intermediate frame and retain deliberate negative
controls. PTY results establish emitted screen behavior for the tested
journeys and terminal sizes; they are one necessary part of evaluating
smoothness.

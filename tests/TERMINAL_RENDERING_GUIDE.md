# Terminal rendering validation and handoff

Read this before changing terminal rendering, layout, input, focus, request
handoffs, or the terminal driver. [TERMINAL_API.md](../TERMINAL_API.md)
describes the design. Root `AGENTS.md` governs evidence acceptance: a pyte
replay of the raw PTY bytes is evidence of what the terminal shows.

## The invariant that prevents flicker and tearing

Every frame is rendered from one UI state snapshot on the UI thread and written
as one synchronized, erase-free transaction. Keep it that way:

- Views may change state only in event handlers and may draw only in
  `render()`. `render()` reads state; it never mutates what another frame needs.
- Nothing outside the UI thread touches views, the canvas, or the terminal.
  Use `TerminalApp.post()` from other threads.
- Never emit erase sequences (`CSI J`, `CSI K`, `CSI 2J`) after startup; the
  writer overwrites whole lines.
- Do not add timers, animations, or sleeps to hide transitions.

## Checks to run

```bash
.venv/bin/python -m pytest tests/core/test_live_terminal.py tests/core/test_live_terminal_fuzz.py \
    tests/core/test_live_terminal_pty.py tests/core/test_search_warm_terminal.py \
    tests/core/test_selection_warm_terminal.py tests/core/test_terminal_architecture.py -q
```

The PTY tests compare every presented frame with an independent pyte replay and
check each frame for completeness, so a torn, blank, or mixed-state frame fails.
When you add a view or region, extend `assert_complete_frame` in
`test_live_terminal_pty.py` with what must be on screen, and add a negative
control showing the check rejects its absence.

For a deeper random search: `SPE_FUZZ_EXAMPLES=400 .venv/bin/python -m pytest
tests/core/test_live_terminal_fuzz.py`.

## Look at it

```bash
.venv/bin/python core/scripts/render_live_screens.py --output /tmp/spe-screens \
    --theme chill amber-cyan --size 80x24 120x40 --view choice beam --overlay help --png
```

Review the PNGs for layout and style. To inspect a real run, set
`SPE_TERMINAL_FRAME_LOG=/tmp/frames.jsonl`: the UI appends each presented frame
(text grid, cursor, and byte offset) as one JSON line.

## Handoff

Report the commands you ran and their results, the views and sizes you looked
at, and anything you could not verify. Keep generated captures out of tracked
source.

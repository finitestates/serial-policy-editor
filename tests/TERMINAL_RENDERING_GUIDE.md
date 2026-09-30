# Terminal rendering validation and handoff

This is the current workflow for agents changing terminal rendering, layout,
input, focus, request handoffs, asynchronous feedback, or driver lifecycle.
Read it before editing and complete it before handing back implementation.
The developer owns discovering and fixing visual defects; the user is not the
acceptance tester. Root [AGENTS.md](../AGENTS.md) governs evidence acceptance.

## The process

Use three complementary layers: widget tests to explain state and ownership,
settled snapshots to review composition, and production PTY capture with
independent terminal replay to check transitions. Endpoint tests alone cannot
establish transition correctness. This combines established
[Textual Pilot and snapshot testing](https://textual.textualize.io/guide/testing/)
with [pyte terminal replay](https://github.com/selectel/pyte/blob/master/docs/tutorial.rst).
The ordered markers and invariants are this repository's integration of those
techniques. [Synchronized output](https://contour-terminal.org/vt-extensions/synchronized-output/)
controls which internal updates a supporting terminal presents together; record
its negotiation and bracket boundaries rather than assuming support.

1. **Define the journey and failures.** Trace the production callers. List the
   affected requests, dimensions, themes, inputs, waiting states, and return
   paths. Express failures as observable assertions: a missing command bar,
   erased body, collapsed table, lost typed text, cursor/focus disappearance,
   changed geometry without a resize, or unnecessary same-kind teardown.
2. **Reproduce before fixing.** Use the production session/driver with a
   deterministic backend. Send real PTY input; synchronize on readiness and
   explicit gates. Exercise fast and delayed results, input during waits,
   repeated actions, and resize in both directions. For Beam use the existing
   30-advance journey. Preserve failures before assertions and in cleanup.
3. **Check the whole transition.** Keep raw bytes, initial dimensions, resize
   offsets, ordered write/display markers, generations, screen identity,
   geometry, grids, styles, and cursor/focus evidence. Mark displays through
   the same writer queue as the output. Decode UTF-8 incrementally. Inspect
   every display in the journey window, including unexpected screen kinds;
   do not filter away the very blank/idle/remounted states being tested.
   Inspect intervening clears and erases even if repaired before the next
   display marker. Evaluate unsynchronized writes and closed synchronized
   transactions appropriately. A marker inside an open bracket is not proof
   of a complete presentation.
4. **Prove the oracle rejects failure.** Run the same regression against an
   isolated pre-fix source without switching the user's checkout, or use a
   demonstrated injected defect when that baseline is unavailable. Add
   negative controls for the new invariant: one missing pane, standalone
   erase, lost focus, or lost cursor. Identity checks diagnose teardown but
   do not replace content/style/operation assertions.
5. **Review the reconstruction.** Read ordered text grids and render relevant
   before, intermediate, and after frames for style inspection. Replay all
   display grids for consistency. Report sequence numbers and what they show.
   Grid equality checks replay consistency; it does not establish independent
   capture correctness. For a cursor defect inspect the rendered TextArea
   cursor cell, focus and cursor position/style as well as relevant terminal
   cursor operations. A text grid or editor-focused flag alone is insufficient.
6. **Run the affected checks on the final tree.** Use the commands below;
   extend the production journey when it does not exercise the changed path.
   Inspect snapshot differences before accepting updates. Fix failures without
   weakening visibility assertions or hiding transitions with sleeps,
   animations, input throttling, or suppressed content.
7. **Deliver evidence.** Use the handoff template below. A missing capture,
   skipped production case, or untested changed path remains incomplete
   validation and must be stated explicitly. Do not call it visually verified.

## Required journey coverage

| Changed behavior | Required observations |
| --- | --- |
| Same-kind refresh/advance | Repeated requests, retained useful widgets, complete old view while waiting, complete replacement content |
| Typing/preview/status | Type, edit, paste and move cursor while feedback arrives; text, focus, cursor cell and command geometry persist |
| Different request kind | Complete outgoing view until complete destination; no idle/blank intermediate surface |
| Help/output/palette | Open, scroll, dismiss, immediately type; underlying content and input ownership restored |
| Layout/wrapping | Narrow and wide, resize both ways while active, long continuations, all required hints in bounds |
| Theme/style | Affected theme plus monochrome/high contrast when focus/cursor/styles change |
| Shutdown/error/interrupt | Terminal modes restored; no stray application output after teardown |

Apply these to affected Choice/live and historical review, Edge, Beam, ordinary
prompt, single-key, multiline, page and chord paths. Shared transition/driver
changes require the complete request matrix. Existing coverage is bounded:
Beam's frame oracles do not automatically verify every other screen, theme or
cursor transition. Add the missing journey and assertion before claiming it.

## Commands from the repository root

Use the environment with the core test dependencies installed. This checkout
has `.venv/bin/python`; older examples also use `core/.venv/bin/python`.
Keep generated runs outside source directories:

```bash
.venv/bin/python core/scripts/record_textual_journey.py --scenario beam --output /tmp/spe-terminal-review
.venv/bin/python core/scripts/record_textual_journey.py --scenario rewind --output /tmp/spe-terminal-review
```

Other scenarios are `mouse`, `runtime`, and `all`. `all` runs the PTY module;
it does not include the separate rewind script or every headless test. The
wrapper reports a unique run directory and retains the pytest log, capture
JSON, raw `.pty` bytes, text frames, frame index and `recording-summary.json`,
including on test failure. Check the exit code, replay errors and frame count;
an empty artifact directory is not a successful visual review.

```bash
.venv/bin/python core/scripts/render_textual_capture.py /path/to/example.capture.json --all
.venv/bin/python core/scripts/render_textual_capture.py /path/to/example.capture.json --sequence 137 141 --png
```

Select sequence numbers from that run's frame index; the numbers above are
examples. SVG export needs pyte; PNG additionally needs ffmpeg. The renderer
checks each selected grid against a fresh raw-byte replay. Review text grids
for geometry and SVG/PNG for styles; inspect operations separately for erases
and cursor behavior. Exporting images alone is not review.

For shared UI changes run the combined checks and any new regression:

```bash
.venv/bin/python -m pytest tests/core/test_textual_driver_pty.py tests/core/test_textual_driver_request_matrix.py tests/core/test_textual_render_stability.py tests/core/test_textual_migration_journey.py tests/core/test_textual_responsive_layout.py tests/core/test_textual_input_menus.py tests/core/test_terminal_lifecycle.py tests/core/test_terminal_architecture.py tests/core/test_textual_transition_matrix.py -q
```

For a bounded change select its affected tests and production scenario and
explain the scope. Benchmarks measure latency/work, not frame correctness;
input-ready counts must not be described as display-frame counts.

## Evidence and artifact map

| Location | Purpose |
| --- | --- |
| [VISIBLE_FRAME_CORRECTNESS.md](VISIBLE_FRAME_CORRECTNESS.md) | Detailed replay boundaries, presentation candidates, invariant rationale |
| [test_textual_driver_pty.py](core/test_textual_driver_pty.py) | Production input journeys, ordered capture, Beam completeness/style/erase oracles and negative controls |
| [record_textual_journey.py](../core/scripts/record_textual_journey.py) | Capture entry point and readable frame export |
| [render_textual_capture.py](../core/scripts/render_textual_capture.py) | Fresh replay agreement and styled SVG/PNG export |
| [test_runtime_rewind_journey.py](../core/scripts/test_runtime_rewind_journey.py) | Separate runtime editing/rewind reproduction |
| [textual_migration_snapshots](core/textual_migration_snapshots) | Settled snapshots and dated historical transition evidence |
| [TERMINAL_API.md](../TERMINAL_API.md) | Runtime/UI ownership and screen/driver architecture |
| [TUI_NO_FLASH_FIX_AND_ACCEPTANCE.md](../TUI_NO_FLASH_FIX_AND_ACCEPTANCE.md) | Historical Beam diagnosis, acceptance case and before/after inventory |

The top-level `TEXTUAL_UI_MIGRATION_*` documents are historical implementation
orders and follow-ups. Their dated results describe their recorded tree, not
current validation. This guide owns the current workflow; the detailed
observability note owns replay interpretation. Preserve historical evidence
rather than using old passing counts as a new acceptance result.

Generated runs belong in `/tmp/spe-terminal-review` or ignored local
`terminal-reconstructions/`. For a durable regression, retain a small dated
index and representative frames under the snapshot directory. Include source
revision plus dirty patch identity, command, dependency versions, dimensions,
scenario, results and oracle. Retain the raw bytes and capture JSON together;
repair `raw_pty_path` when relocating so fresh replay remains possible.
Absolute `/tmp` references in historical indexes may have expired and must be
checked before claiming the original raw capture is available. Do not commit
bulk recordings, generated images or repeated investigation logs by default.

## Evidence acceptance

Treat pyte encoding and terminal-display evidence as prima facie evidence of
what happens on the terminal screen. Review it and report what it shows. A
rebuttal must identify contrary evidence or a demonstrated reconstruction error
in that capture. Lack of direct access to the user's screen is not a rebuttal.
A structurally invalid frame needs no duration measurement to establish failure.

Use timing or a desktop recording when the question concerns elapsed duration
or an emulator-specific effect outside captured protocol state. Those are
additional measurements, not prerequisites for accepting PTY evidence or
closing a structural rendering defect. Distinguish writes inside supported
synchronization brackets from presentation candidates. Do not ask the user to
watch flashing output to validate the fix.

## Handoff template

- **Change and scope:** causes removed; changed paths and paths left unverified.
- **Reproduction:** source/environment, exact command, inputs, sizes and themes.
- **Failure detection:** baseline or injected control result and rejected invariant.
- **Final validation:** commands, actual results, skips/failures and scope.
- **Frame review:** artifact links, first/last relevant sequence and observed
  content, geometry, styles, focus/cursor and intervening terminal operations.
- **Remaining uncertainty:** specific missing journey or measurement, if any.

Leave changes uncommitted for review unless the user asks otherwise.

## Workflow verification on 2026-09-30

Running the documented Beam recorder on `luna/terminal-reboot` produced
**1 passed, 1 failed** in 33.14 seconds. The 160×50-to-80×24 case failed
`_assert_complete_beam_frame`: display 477 retained the wide Beam heading,
display 481 at 80×24 replaced the top rows with continuation text and lost
the heading, and display 485 restored the complete narrow heading. Display
481 was outside an active synchronization bracket. Fresh raw-byte replay
agreed with its stored grid and successfully exported SVG/PNG. This is an
open rendering failure; this documentation change does not fix it.

The retained run is `/tmp/spe-terminal-review/20260930-180438`, including
`recording-summary.json`, pytest log, and both size cases' raw bytes/captures.
These are local temporary artifacts, not a durable or portable baseline.
The result demonstrates that the documented workflow detects an intermediate
defect that a subsequent settled frame would hide.

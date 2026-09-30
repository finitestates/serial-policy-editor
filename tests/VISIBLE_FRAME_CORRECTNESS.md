# Visible-frame correctness in terminal tests

The terminal test suite observes several different layers of the UI. Those
layers answer different questions and should not be treated as interchangeable.

A passing state, layout, or screenshot test does not by itself establish that
every screen state emitted during a transition was valid. Conversely, a
reconstructed intermediate terminal state does not need to have been noticed by
a human observer, or assigned a duration in milliseconds, before it can be
classified as structurally invalid.

This note describes the observability boundary of the current harness and the
reason for retaining PTY + terminal-emulator replay even if the interactive UI
implementation changes.

## Three different kinds of evidence

### 1. Application and widget state

The in-process Textual pilot, lifecycle, request-matrix, and transition tests
observe application objects directly. They are good evidence for questions such
as:

- which screen is mounted;
- which widget owns focus;
- whether a request is accepting input;
- whether stale input is rejected at a request boundary;
- whether a table, editor, or screen instance was reused;
- whether prepared state reached the expected widget;
- whether a resize selected the intended layout class;
- whether regions fit inside the declared terminal dimensions.

Examples include
`test_terminal_lifecycle.py`, `test_textual_transition_matrix.py`,
`test_textual_driver_request_matrix.py`,
`test_textual_render_stability.py`, and
`test_textual_responsive_layout.py`.

These tests can establish logical and geometric invariants inside the
application. They do **not** observe the complete ordered byte stream sent to a
terminal. A framework can therefore be internally consistent at every sampled
application checkpoint while still producing an undesirable visible transition
between those checkpoints.

### 2. Settled visual checkpoints

SVG snapshots and post-display assertions answer questions about a completed
rendering:

- is expected content present;
- does the screen fit at a given size;
- are important regions visible;
- did a theme or layout change alter the settled presentation.

These are useful regression tests, but they are endpoint observations. Two
implementations can have identical settled screenshots while taking very
different routes between them.

For example, both of these transitions can end at the same final frame:

```text
A:  complete frame N  ---------------------->  complete frame N+1

B:  complete frame N
          |
          v
    collapsed / cleared frame
          |
          v
    partially repainted frame
          |
          v
    complete frame N+1
```

An endpoint snapshot cannot distinguish A from B.

### 3. Ordered terminal-visible state

The POSIX PTY harness in `test_textual_driver_pty.py` observes a lower
boundary. It records the bytes written through the production terminal driver,
ordered writer/display markers, resize offsets, synchronized-update operations,
and selected compositor metadata.

`_replay_terminal_frames()` then replays that byte stream through
`pyte.Screen`, an independent VT-style terminal state machine. At recorded
boundaries it reconstructs the terminal cell grid and can retain:

- character cells;
- selected cell styles;
- terminal dimensions;
- resize events;
- CSI operations and cursor state;
- synchronized-update bracket state;
- application generation and captured region geometry where available.

The Beam no-flash journey uses those reconstructed grids with
`_assert_complete_beam_frame()` to reject states in which, for example, the
Beam heading disappears, the selected-details pane disappears, the command bar
vanishes, a continuation column collapses, a live branch loses visible content,
or a captured region escapes the terminal.

This is evidence about the terminal protocol stream rather than only the
framework's object model.

## Invalid frames do not require timestamps

A visual defect can be hard for an automated agent to observe directly. That
does not make the defect subjective, and it does not mean the harness needs a
camera or precise timestamps before it can test it.

For many terminal transitions, the important property is **sequence**, not
duration.

Suppose replay reconstructs these ordered candidate display states:

```text
F0  complete Beam
F1  Beam body collapsed to zero/near-zero useful area
F2  command region repainted without the Beam body
F3  complete Beam
```

If the UI contract says that an active Beam transition must always retain a
complete usable Beam surface, then F1 and F2 are invalid states. Whether F1
lasted 0.5 ms, 5 ms, or 50 ms is a separate measurement question. A timestamp
may help characterize severity on a particular machine; it is not required to
establish that the renderer emitted a forbidden structural state.

This distinction is useful for automated contributors. An agent does not need
to "see the flash" in the human sense. It needs an observation boundary capable
of reconstructing the ordered terminal states, plus explicit invariants that
classify those states.

## Presented frames versus writes inside synchronized updates

The byte stream is finer-grained than what a conforming physical terminal is
necessarily expected to present.

When synchronized terminal updates are available, the harness records DEC
private mode 2026 boundaries and whether replay is inside a synchronization
bracket. Operations inside a supported synchronized-update bracket may alter
the emulator's internal reconstructed state without being intended as
individually presented frames.

Tests should therefore distinguish:

1. terminal operations that occurred;
2. reconstructed states at ordered write/resize boundaries;
3. completed compositor/display states;
4. states that are candidates for presentation given synchronized-update
   support.

That distinction prevents both false confidence and false alarms. It also
keeps the test useful on terminals that do not support synchronized updates:
the unsynchronized byte sequence still exists and can be inspected.

## Useful visible-frame invariants

The current Beam assertions are concrete examples, not an exhaustive contract.
The same approach can express renderer-independent properties such as:

- during an active request, required regions never disappear between valid
  endpoints;
- ordinary state changes repaint content but do not change region geometry;
- geometry changes occur only on an explicit resize or mode transition;
- a command/input region remains visible and in-bounds throughout a request;
- a live table does not transiently collapse its usable width or height;
- content required to identify the selected branch remains present in every
  completed display state;
- large clears or large-area style changes are not emitted as standalone
  presented states during routine updates;
- a resize produces a complete replacement geometry rather than a sequence of
  mutually inconsistent pane layouts.

These are stronger than "the final screen looks right" and different from
performance assertions. They describe states the UI is not allowed to expose.

## What remains unobservable

PTY + pyte replay deliberately stops short of claiming to model a person's
physical display.

It does not prove:

- exactly when a particular terminal emulator composites bytes to a monitor;
- the refresh rate, pixel response, compositor scheduling, or display hardware
  involved;
- whether a user perceived a particular transition;
- the precise duration for which a physical terminal displayed a reconstructed
  state;
- behavior caused by emulator-specific rendering outside the terminal protocol
  state represented by the harness.

Those limitations matter when characterizing physical presentation or
accessibility risk. They do not erase the software-level evidence. If the
application emits an ordered terminal state that violates a structural
visible-frame invariant, the defect can be detected and fixed without first
proving how many milliseconds a particular monitor displayed it.

## Why keep this below the UI framework

The PTY replay boundary is valuable precisely because it is external to
Textual's widget model. It should remain useful if the live renderer is later
simplified, rewritten with curses, or replaced by a smaller terminal
compositor.

The durable testing pipeline is:

```text
application / renderer
        |
        v
terminal byte stream
        |
        v
ordered PTY capture
        |
        v
independent terminal replay (pyte)
        |
        v
cell grids + styles + geometry
        |
        v
visible-frame invariants
```

Framework-level tests can explain *why* a transition happened. PTY replay can
show *what terminal state was emitted*. Both are useful, and neither substitutes
for the other.

A concise design rule for the live editing path is:

> **Reactive state, inert geometry.**

And the corresponding test rule is:

> **Every terminal state eligible to be presented to the user must satisfy the
> structural invariants of its active mode, including states between settled
> application endpoints.**

The goal is not to require an automated contributor to possess human visual
perception. The goal is to make the relevant visual property observable as
data.

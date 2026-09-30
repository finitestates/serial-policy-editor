# Terminal API and fallback behavior

`TerminalIO(...)` is the application's terminal entry point. Construct it once
per interactive run and enter `with io.session():` around that run. When both
standard streams are usable TTYs and Textual is installed, the session starts
one `PolicyEditorApp` on a dedicated UI thread and event loop. Piped or
noninteractive use keeps the synchronous `plain_tui.py` fallback.

## Thread and request model

The episode-owning thread keeps the model backend and SQLite connection. It
prepares the frozen view-state records in `terminal_contracts.py` and calls the
blocking `TerminalProtocol` methods: `read_choice`, `read_edge`, `read_beam`,
`prompt`, `read`, `read_key`, `write`, or `page`. The protocol also provides a
`session()` context manager and `capabilities` property.

For a live read, the owning thread posts a screen request to Textual with
`App.call_from_thread(...)` and waits on a `concurrent.futures.Future`. The
screen resolves that future once, when the user submits or cancels. A submitted
screen disables its input before dismissal, so later queued keystrokes cannot
become the next request's input. Choice and EDGE results are raw command text;
beam reads return a `BeamInput` with command text and selected branch. The
caller alone interprets commands and applies engine, navigation, replay, or
storage changes.

Token and insertion previews that need engine-owned state are queued back to
the episode thread. Search warming runs on a Textual-owned executor, and its
generation-tagged result returns to the UI through `call_from_thread`. The UI
thread never calls engine or backend state directly.

During a live session, incidental stdout and stderr are held in memory and do
not enter the live screen. The captured text is flushed to the original streams
only after the session exits, once Textual has restored the terminal. Explicit
`TerminalIO.write(...)` messages remain available in the app's docked `RichLog`.
The viewer retains the same trailing 16,000 characters whether it stays open or
is reopened; large writes and history rollover keep that limit.

## Screens and controls

Each request type has one Textual screen. Clicking a Choice, EDGE, or Beam row
keeps command focus in the editor through mouse-down and row selection. The
focused command bar remains read-only during owner-thread handoff, then becomes
editable when the next request is ready. The POSIX request matrix verifies a
gated stale paste is discarded and fresh multiline paste is preserved; the
runtime journey also checks queued key and mouse rejection during its tested
handoff. These cases do not cover every request and modal lifecycle.

| Screen | Contents and behavior |
| --- | --- |
| `ChoiceScreen` | Scrollable prepared context, candidate `DataTable`, preview and feedback, editable command area, and read-only historical review. Context follows the prepared tail, which is unlimited by default. PgUp/PgDn control context scrolling; Ctrl+E expands authored-text input; Alt+Enter inserts a newline in `t` and `x` commands. |
| `EdgeScreen` | Episode or session status, sampler summary, mode-specific commands from `edge_help()`, and a command input. Blank Enter submits an empty command so the engine can continue. |
| `BeamScreen` | Survivor table, shared context, selected-branch details, notice, and command input. Same-kind Beam commands update the mounted screen in a batch. Wrapped table rows and the flexible continuation column are fitted before the compositor pass. The table and details use the remaining terminal height; below 120 columns they stack. |
| `PromptScreen` | One parameterized screen for ordinary input, single keys, multiline composition, scrollable pages, and isolated chord composition. Page return and single-key capture use focused input targets so the production driver delivers keys through Textual's input-widget path. |
| `HelpScreen` | Scrollable modal help shown over the active request screen. |

The transition matrix verifies mounted-screen reuse for repeated Edge and
compatible Prompt requests. The production Beam journey separately asserts
screen/table/detail/editor identity through 30 advances and checks each
captured display pass at 80×24 and 160×50. These are bounded test journeys;
Choice reuse and every cancellation/modal combination are not all asserted by
that matrix.

Ctrl+K opens Textual's fuzzy command palette. Selecting a command inserts its
template into the active input. `?` opens the full command list as a modal.
Edge commands are displayed in the Edge screen and are also searchable in the
palette. Help is not printed into the Edge status area on every refresh.

Choice and EDGE reads return command text; beam reads return `BeamInput`.
`PromptRequest` covers ordinary input, confirmations and single keys,
multiline composition, pages, and isolated chord displays through
`io.prompt(request)`. The POSIX request matrix checks actual driver submission
for page return (`q` returns `""`; Enter/Esc also return), single-key values
(including Backspace as DEL), chord selection, and multiline paste. `read`,
`read_key`, and `page` are small adapters to this request. The chord flow
submits an isolated `PromptRequest` directly.

`SEAMLESS_REACTIVATE` distinguishes Enter in a seamless historical review from
Escape and ordinary command text. `io.capabilities.seamless_review` tells
callers whether that behavior is enabled. View-state records remain the
engine-facing seam: Textual screens render them, while the caller retains
command and state authority.

## Themes

`resolve_live_theme(requested, *, environment)` retains the names
`amber-cyan`, `monochrome`, and `high-contrast`. With no explicit choice,
`NO_COLOR` selects monochrome; otherwise amber-cyan is the default.
`COLORFGBG` selects a light or dark palette, with dark as the default. When
`COLORTERM=truecolor`, the Textual stylesheet and Rich spans use hex colors;
other terminals use named ANSI colors. Monochrome uses bold, underline, and
reverse video without color. High-contrast foreground and semantic colors meet
WCAG AA contrast against their light or dark background.

## Where changes belong

- Add teacher syntax and interpretation in `teacher_commands.py`, then handle
  the parsed command in `episode_ui.py`. Add EDGE syntax in `edge_commands.py`,
  dispatch it in `episode_cli.py` or `session_runtime.py`, and update
  `edge_help.py`.
- Add prepared display data to the relevant frozen record in
  `terminal_contracts.py`. Prepare it on the episode-owning thread. Shared Rich
  fragments and candidate table builders belong in `tui_render.py`; interactive
  widgets belong in `textual_tui.py`; plain rendering stays in `plain_tui.py`.
- Route new reads through `TerminalIO` in `tui.py`. Keep one app and session
  across request screens. Scripted CLI tests use `ScriptedIO` from
  `tests/fakes.py`.

`tests/core/test_terminal_architecture.py` checks that runtime imports stay at
the terminal boundary, Textual remains lazy, and the plain fallback stays
isolated. Pilot tests exercise screen state and layout; POSIX PTY tests exercise
the custom driver, multi-turn runtime bridge, tested handoff input rejection,
terminal cleanup, and output replayed through pyte. Resize checkpoints show the
screen at recorded resize offsets. Ordered writer markers identify individual
terminal writes; a test-only marker queued after Textual's `post_display_hook`
identifies a completed display pass in that same writer queue. The Beam journey
checks its gated wait, 30 advances, retained widget identity, complete captured
panes, and settled repaint after synchronized resizes. The retained comparison
shows the archived baseline failing the same remount assertion. These protocol
captures do not establish physical-terminal behavior or prove that no
intermediate partial display is visible.
`benchmarks/tui_transitions.py --package-root CHECKOUT` loads that checkout's
`core/src` directly and measures prepared-screen submission-to-next-render
latency at the requested console size. It excludes model, database, and
terminal-painting time.

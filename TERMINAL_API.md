# Terminal API and fallback behavior

`TerminalIO(...)` is the application's terminal entry point. Construct it once
per interactive run and enter `with io.session():` around that run. When both
standard streams are TTYs on a POSIX system, the session starts the live
interface in `trajectory_editor/term/`. Piped or noninteractive use keeps the
synchronous `plain_tui.py` fallback.

## How the live interface draws

The live interface is immediate-mode: every frame is a pure function of one UI
state snapshot.

- One UI thread (`spe-terminal-ui`) owns all UI state. Other threads only
  *post* work to it (`TerminalApp.post`); they never touch views or the
  terminal.
- The UI thread's loop drains every ready input byte and posted event, then
  renders the whole screen into a cell canvas (`term/canvas.py`) from the
  current state (`term/views.py`, `term/overlays.py`).
- `FrameWriter` (`term/driver.py`) compares the canvas with the previous frame
  and rewrites only the changed lines, inside one synchronized-output
  transaction (`CSI ? 2026 h … l`) and one `write()`. After the initial clear it
  never erases: lines are overwritten in place, so even a terminal without
  synchronized output never shows a blank or partly cleared screen.
- Resize is just another frame at the new size: the next render reads the
  terminal size and repaints every line.

Because a frame is computed from one snapshot, it cannot mix two states (for
example a Beam table selecting one branch while the details show another).

## Thread and request model

The episode-owning thread keeps the model backend and SQLite connection. It
prepares the frozen view-state records in `terminal_contracts.py` and calls the
blocking `TerminalProtocol` methods: `read_choice`, `read_edge`, `read_beam`,
`prompt`, `read`, `read_key`, `write`, or `page`.

For a live read, the owning thread posts the request to the UI thread and waits
on a `concurrent.futures.Future`. The view resolves that future once, when the
user submits or cancels. After submission the view stays on screen, marked
`working…`, until the next request replaces it; input typed meanwhile is
dropped so it cannot leak into the next request. Choice and EDGE results are raw
command text; beam reads return a `BeamInput`. The caller alone interprets
commands and applies engine, navigation, replay, or storage changes.

Token and insertion previews that need engine-owned state are queued back to the
owning thread while it waits. Search warming runs on a worker thread and posts
its generation-tagged result to the UI thread.

During a live session, incidental stdout and stderr are held in memory and do
not reach the screen. They are flushed to the original streams after the
session restores the terminal. Explicit `TerminalIO.write(...)` messages are
kept (trailing 16,000 characters) for the Ctrl+L output viewer.

Ctrl+C during a request raises `KeyboardInterrupt` from that request; while the
engine is working it interrupts the main thread, as in a plain CLI. Ctrl+Z
suspends to the shell and repaints on resume. Every exit path restores the
terminal modes (alternate screen, cursor, mouse, bracketed paste, autowrap).

## Views and keys

| View | Contents and keys |
| --- | --- |
| Choice | Context (follows the tail; PgUp/PgDn page it), proposal preview, feedback, candidate table, command line, hints. Tab/Shift+Tab and ↑/↓ cycle candidates and completions; click a row to stage its rank; Ctrl+G explores a rank; Ctrl+E expands authored text and Alt+Enter adds a newline; `[`/`]` open review from an empty command; F2 shows the full context, preview, and feedback. A prefilled command is replaced by typing. |
| Review | Historical boundary header, context, read-only command line. Enter, Esc, `[`, `]`, and `f` return the review contract values; any other key returns to the live edge. |
| EDGE | Header, command templates (↑/↓ or click to stage), command line. Blank Enter continues; Ctrl+D quits. |
| Beam | Survivor table and selected-branch details, side by side from 120 columns and stacked below that. ↑/↓ select, ←/→ rewind/advance, Enter commits, Backspace/`p`/`f` kill/protect/toggle families on an empty command, Esc returns. |
| Prompt | Ordinary input, single key, multiline composition (Esc then Enter submits), scrollable page, and isolated chord display. |

On every view: F1 help, Ctrl+K command search, Ctrl+L captured output.
Layouts allocate rows by priority, so the command line and hints stay visible at
every size; below 20×3 the screen asks to be enlarged.

## Themes

`resolve_live_theme(requested, *, environment)` accepts `amber-cyan` (default),
`chill`, `ink`, `monochrome`, and `high-contrast`; `SPE_THEME` sets a default, and
`NO_COLOR` selects monochrome. `COLORFGBG` selects the light or dark palette.
Truecolor and 256-color terminals get the theme's own palette and background;
basic terminals use ANSI colors on the terminal's own background. Every palette
color meets WCAG AA contrast against its background. The command field is a
quiet raised field (underlined on basic terminals); the colored prompt label and
the caret mark focus.

## Where changes belong

- Add teacher syntax and interpretation in `teacher_commands.py`, then handle
  the parsed command in `episode_ui.py`. Add EDGE syntax in `edge_commands.py`,
  dispatch it in `episode_cli.py` or `session_runtime.py`, and update
  `edge_help.py`.
- Add prepared display data to the relevant frozen record in
  `terminal_contracts.py`, prepared on the episode-owning thread. Shared Rich
  fragments and candidate rows belong in `tui_render.py`; live views belong in
  `term/views.py`; plain rendering stays in `plain_tui.py`.
- Route new reads through `TerminalIO` in `tui.py`. Scripted CLI tests use
  `ScriptedIO` from `tests/fakes.py`.

## Tests

- `tests/core/test_live_terminal.py` drives each view headlessly through the
  real key parser (`tests/core/term_support.py: Harness`) and asserts submitted
  values, layouts at several sizes, overlays, and styles.
- `tests/core/test_live_terminal_fuzz.py` runs Hypothesis-generated key, paste,
  click, wheel, and resize sequences on every view and checks each frame.
- `tests/core/test_live_terminal_pty.py` runs the production session on a real
  PTY (scripted requests and the real `run_session_roster` runtime), replays the
  raw bytes through pyte, and requires the terminal to equal the intended frame
  at every frame boundary; it also checks every frame for completeness, the
  absence of erase operations, and terminal restoration. Its negative controls
  show the checks reject a torn line, an erase, and a mixed-state Beam frame.
- `core/scripts/render_live_screens.py` renders any view, size, and theme to
  HTML/PNG for visual review.

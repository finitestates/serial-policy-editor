# Terminal API and fallback behavior

`TerminalIO(...)` is the application's terminal entry point. Construct it once
per interactive run and enter `with io.session():` around that run. When the
standard streams are usable TTYs and Python's curses module is available, the
session opens the synchronous curses renderer. Piped and noninteractive runs
use the synchronous `plain_tui.py` fallback.

## Request and action ownership

The episode-owning thread prepares the frozen view-state records in
`terminal_contracts.py` and calls the blocking `TerminalProtocol` methods:
`read_choice`, `read_edge`, `read_beam`, `prompt`, `read`, `read_key`, `write`,
or `page`. The protocol also provides a `session()` context manager and a
`capabilities` property.

Each live request renders and reads on that same thread. Choice and EDGE reads
return command text; Beam reads return a `BeamInput` with command text and the
selected branch. The caller interprets commands and applies engine,
navigation, replay, and storage changes. The terminal UI does not call model,
engine, or storage code while waiting for input.

While the live session is active, Python stdout and stderr and native file
descriptor output are captured so they do not overwrite the screen. Captured
text appears in the bounded output viewer and is written to the original
streams after the terminal session exits. Explicit `TerminalIO.write(...)`
messages use the same viewer. The viewer retains the trailing 16,000
characters.

## Controls

| Request | Controls |
| --- | --- |
| Choice | Tab/Shift+Tab browse candidates; Ctrl+G enters a raw rank; PgUp/PgDn scroll context; Ctrl+E expands authored-text input; Ctrl+O inserts a newline; Ctrl+D opens EDGE. |
| EDGE | Tab cycles commands; Up/Down selects command templates; Enter submits; Ctrl+D returns to the choice. |
| Beam | Up/Down selects a branch; Enter commits it; Left rewinds; Right advances; Backspace kills; `p` toggles protection; `f` toggles family details; PgUp/PgDn scroll details. |
| Prompts | Enter submits; Escape then Enter submits multiline text; Escape cancels where supported. |
| Any live view | Ctrl+K opens the command picker; F1 opens help; Ctrl+L opens captured output. |

The curses renderer updates through curses' virtual-screen refresh calls. It
uses the names `amber-cyan`, `monochrome`, and `high-contrast`. `NO_COLOR`
selects monochrome unless a theme is explicitly requested; otherwise
`amber-cyan` is the default. `COLORFGBG` selects the light or dark palette, with
dark as the default. Monochrome uses emphasis without color.

## Where changes belong

- Add teacher syntax and interpretation in `teacher_commands.py`, then handle
the parsed command in `episode_ui.py`. Add EDGE syntax in `edge_commands.py`,
dispatch it in `episode_cli.py` or `session_runtime.py`, and update
`edge_help.py`.
- Add prepared display data to the relevant frozen record in
`terminal_contracts.py`. Prepare it on the episode-owning thread. Shared Rich
fragments and candidate table builders belong in `tui_render.py`; live curses
rendering belongs in `curses_tui.py`; plain rendering stays in `plain_tui.py`.
- Route new reads through `TerminalIO` in `tui.py`. Scripted CLI flows use
`ScriptedIO` from `tests/fakes.py`.

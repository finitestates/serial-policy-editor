# Terminal API and fallback behavior

`TerminalIO(...)` is the application's terminal entry point. Construct it once
per interactive run and enter `with io.session():` around that run. A live
terminal keeps one `PersistentTerminalSession` and one prompt-toolkit
application until the context exits. A plain or piped terminal uses the same
context with no live application. Live choice, EDGE, and prompt reads require the
context to be active; plain requests use that same context.

The episode-owning thread prepares `ChoiceViewState` and `EdgeViewState` from
`terminal_contracts.py`, then calls `io.read_choice(state)` or
`io.read_edge(state)`. Both return raw command text or `None` on EOF. The
caller alone parses that text and applies engine, navigation, replay, or store
changes. `PromptRequest` covers ordinary input, confirmations and single keys,
multiline prompt composition, pages, and isolated chord displays through
`io.prompt(request)`. Existing `read`, `read_key`, `read_multiline_prompt`,
and `page` methods are small adapters to this request. The chord flow submits
an isolated `PromptRequest` directly.

The `TerminalProtocol` describes the shared API. `io.terminal_size()` supplies
the usable width and height; a missing size calls for a normal width fallback.
`io.capabilities.seamless_review` explicitly enables Enter to rewind from a
review boundary. Policy and EDGE callers do not choose a renderer. Adding
presentation fields changes the relevant request type; readers pass the
same request object through to the selected implementation.

## Where changes belong

- Add teacher syntax and help in `teacher_commands.py`, then handle the parsed
  command in `episode_ui.py`. Update the short plain action line in
  `plain_tui.py` and any live key help in `live_tui.py`. Add EDGE syntax in
  `edge_commands.py`, dispatch it
  in `episode_cli.py` and/or `ephemeral_runtime.py` as appropriate, and update
  `edge_help.py`. Keep episode, store, and backend effects in those callers.
- Add a decision field to `ChoiceViewState`, an EDGE field to `EdgeViewState`, or
  an input option to `PromptRequest` in `terminal_contracts.py`. Prepare its
  value on the episode-owning thread. Render the same request in `plain_tui.py`
  and the corresponding live view (`live_tui.py`, `edge_tui.py`, or the prompt
  surface in `persistent_tui.py`).
- Add an interactive surface to `PersistentTerminalSession` and route its
  request through `TerminalIO` in `tui.py`. Keep one application and session
  across surfaces. Its plain counterpart handles only text display and input.
  Scripted CLI tests use the request-level `ScriptedIO` adapter in
  `tests/fakes.py`; its `ScriptedTextIO` base is only for low-level text tests.

`tests/core/test_terminal_architecture.py` checks that production code imports
`plain_tui` only from terminal selection and does not branch on renderer mode in
runtime code. `tests/core/test_terminal_scenarios.py` runs common commands
through both scripted adapters and both episode workflows.

Live views are selected at construction only when enabled, both standard
streams are usable TTYs with file descriptors, and prompt-toolkit is installed.
Both backends accept the same teacher and
EDGE command grammar. Presentation may differ in these ways:

| Request | Live | Plain or piped |
| --- | --- | --- |
| Choice | Fullscreen layout, editable command buffer, non-mutating previews and optional initial command | Text table followed by ordinary input; no prefill or previews |
| Review and feedback | Dedicated review and status areas | Text lines prepared in the same choice request |
| EDGE | Fullscreen menu | Text menu with the same command meanings |
| Page | Scrollable in-application page | System text pager |
| Single key or confirmation | In-application key binding | Unbuffered key on a TTY; first character of an input line when piped |
| Multiline composition | In-application editor; Escape then Enter submits, Ctrl-D cancels, empty prompts remain editable | Line-oriented prompt; Enter submits one nonempty line, Ctrl-D cancels |
| Isolated chord | Dedicated prompt body, without prior status history | Body printed before input |

The choice request carries search results, errors, bias feedback, review
position, and column preferences. Both EDGE renderers use the same command
labels, with distinct durable episode and ephemeral session actions. Plain
input preserves command meanings but has no cursor navigation or previews.
Initial prompts and bare `new` use the same composition request. For exact
multiline startup text in plain mode, use `--new-prompt-file FILE` (or
`new TEXT` for a single-line root at EDGE). An interactive new launch enters
the terminal session before collecting its prompt or loading the model. A
piped launch without a source fails validation and does not read stdin.

Callers request pages explicitly with `io.page(text)` and give prompts their
own `PromptRequest.body` when they need accompanying context. Ordinary
`io.write(text)` remains status output regardless of line count. This keeps
help, fork-map selection, confirmations, and chord previews from carrying
the previous prompt's body into the next workflow.

The live application owns widgets, key bindings, surface transitions, input
gating, and preview scheduling. Preview callbacks execute on the episode-owning
thread. Expected preview validation errors remain editable feedback; unexpected
preview failures propagate, stop the live application, and restore the terminal.
EOF and interrupts also release the waiting episode thread. A live failure never
restarts input in plain mode. The active model-free checks in
`tests/core/test_terminal_lifecycle.py` cover those boundaries, resizing, narrow
layouts, multiline input, and every live theme. Use
`benchmarks/tui_transitions.py --package-root CHECKOUT` from the same interpreter
for each checkout. It loads `CHECKOUT/core/src` directly and reports application,
fullscreen, redraw, and prepared-view transition measurements. Compare the
same dimensions and iteration count; the benchmark excludes model, database,
and terminal-emulator painting time.

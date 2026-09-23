# Terminal API and fallback behavior

`TerminalIO(...)` is the application's terminal entry point. Construct it once
per interactive run and enter `with io.session():` around that run. A live
terminal keeps one `PersistentTerminalSession` and one prompt-toolkit
application until the context exits. A plain or piped terminal uses the same
context with no live application. Live choice, EDGE, and prompt reads require the
context to be active. `live_session()` remains an alias for older
callers; it enters the same context.

The episode-owning thread prepares `ChoiceViewState` and `EdgeViewState` from
`terminal_contracts.py`, then calls `io.read_choice(state)` or
`io.read_edge(state)`. Both return raw command text or `None` on EOF. The
caller alone parses that text and applies engine, navigation, replay, or store
changes. `PromptRequest` covers ordinary input, confirmations and single keys,
multiline prompt composition, pages, and isolated chord displays through
`io.prompt(request)`. Existing `read`, `read_key`, `read_multiline_prompt`,
and `page` methods are small adapters to this request. The chord flow submits
an isolated `PromptRequest` directly.
The `TerminalProtocol` describes the shared API; `io.capabilities` and
`io.terminal_size()` expose whether live views are available and the usable
size. A missing size means the caller should use its normal width fallback.
Adding presentation fields changes the relevant request type; readers pass the
same request object through to the selected implementation.

Live views are selected at construction only when enabled, both standard
streams are usable TTYs with file descriptors, and prompt-toolkit is installed.
Both backends accept the same teacher and
EDGE command grammar. Presentation may differ in these ways:

| Request | Live | Plain or piped |
| --- | --- | --- |
| Choice | Fullscreen layout, editable command buffer, non-mutating previews and optional initial command | Text table followed by ordinary input; no prefill or previews |
| Review and feedback | Dedicated review and status areas | Text lines and existing policy feedback messages |
| EDGE | Fullscreen menu | Text menu with the same command meanings |
| Page | Scrollable in-application page | System text pager |
| Single key or confirmation | In-application key binding | Unbuffered key on a TTY; first character of an input line when piped |
| Multiline composition | In-application editor | Existing line composer |
| Isolated chord | Dedicated prompt body, without prior status history | Body printed before input |

The live application owns widgets, key bindings, surface transitions, input
gating, and preview scheduling. Preview callbacks execute on the episode-owning
thread. Expected preview validation errors remain editable feedback; unexpected
preview failures propagate, stop the live application, and restore the terminal.
EOF and interrupts also release the waiting episode thread. A live failure never
restarts input in plain mode. The active model-free checks in
`tests/core/test_terminal_lifecycle.py` cover those boundaries. 

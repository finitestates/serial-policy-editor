"""Stable, two-pane terminal presentation for interactive beam search."""

from __future__ import annotations

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    HSplit,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension

from .terminal_contracts import BeamInput, BeamViewState
from .tui_views import ViewLifecycle


class LiveBeamView(ViewLifecycle):
    """Render the frontier and route input without reading model state."""

    def __init__(self, state: BeamViewState, *, submit, enabled, terminal_size):
        super().__init__(submit=submit)
        self.state = state
        self.enabled = enabled
        self.terminal_size = terminal_size
        self.selected_label = state.selected_label
        self.command_buffer = Buffer(
            multiline=False,
            read_only=Condition(lambda: not self.enabled()),
        )
        self.bindings = KeyBindings()

        wide = Condition(lambda: self._columns() >= 92)
        panes = HSplit(
            [
                ConditionalContainer(
                    VSplit(
                        [
                            Window(
                                FormattedTextControl(self._survivors),
                                wrap_lines=True,
                                style="class:beam-pane",
                            ),
                            Window(
                                FormattedTextControl(
                                    [("class:rule", "│")]
                                ),
                                width=Dimension.exact(1),
                                always_hide_cursor=True,
                            ),
                            Window(
                                FormattedTextControl(self._details),
                                wrap_lines=True,
                                style="class:beam-pane",
                            ),
                        ],
                        padding=1,
                    ),
                    wide,
                ),
                ConditionalContainer(
                    HSplit(
                        [
                            Window(
                                FormattedTextControl(self._survivors),
                                wrap_lines=True,
                                style="class:beam-pane",
                            ),
                            Window(
                                FormattedTextControl(
                                    [("class:rule", "─" * 72)]
                                ),
                                height=1,
                                always_hide_cursor=True,
                            ),
                            Window(
                                FormattedTextControl(self._details),
                                wrap_lines=True,
                                style="class:beam-pane",
                            ),
                        ]
                    ),
                    Condition(lambda: self._columns() < 92),
                ),
            ]
        )
        input_control = BufferControl(buffer=self.command_buffer)
        command_row = VSplit(
            [
                Window(
                    FormattedTextControl(
                        [("class:prompt-label", "Beam › ")]
                    ),
                    width=Dimension.exact(8),
                    height=1,
                ),
                Window(input_control, height=1, style="class:input"),
            ]
        )
        root = HSplit(
            [
                Window(
                    FormattedTextControl(self._header),
                    dont_extend_height=True,
                    always_hide_cursor=True,
                ),
                Window(
                    FormattedTextControl(self._context),
                    wrap_lines=True,
                    dont_extend_height=True,
                    always_hide_cursor=True,
                    style="class:muted",
                ),
                Window(
                    FormattedTextControl(
                        [("class:rule", "─" * max(1, self._columns()))]
                    ),
                    height=1,
                    always_hide_cursor=True,
                ),
                panes,
                Window(
                    FormattedTextControl(self._notice),
                    height=1,
                    dont_extend_height=True,
                    always_hide_cursor=True,
                    style="class:feedback-info",
                ),
                command_row,
                Window(
                    FormattedTextControl(self._footer),
                    height=1,
                    dont_extend_height=True,
                    always_hide_cursor=True,
                    style="class:hint",
                ),
            ]
        )
        self.layout = Layout(root, focused_element=input_control)

        @self.bindings.add("enter")
        def _enter(event):
            command = self.command_buffer.text.strip()
            if not command:
                command = "resume" if self.state.at_edge else (
                    f"select {self.selected_label}"
                    if self.selected_label is not None else ""
                )
            self._finish(
                event,
                result=BeamInput(command, self.selected_label),
            )

        @self.bindings.add("up")
        def _up(event):
            self._move_selection(-1, event)

        @self.bindings.add("down")
        def _down(event):
            self._move_selection(1, event)

        @self.bindings.add(
            "right",
            filter=Condition(lambda: not self.command_buffer.text.strip()),
        )
        def _advance(event):
            command = "resume" if self.state.at_edge else "advance 1"
            self._submit_command(event, command)

        @self.bindings.add(
            "left",
            filter=Condition(lambda: not self.command_buffer.text.strip()),
        )
        def _rewind(event):
            command = "resume" if self.state.at_edge else "rewind"
            self._submit_command(event, command)

        @self.bindings.add("escape", eager=True)
        def _return(event):
            self._submit_command(event, "return")

        @self.bindings.add("c-d")
        def _close(event):
            self._submit_command(event, "return")

    def update(self, state: BeamViewState) -> None:
        self.state = state
        self.selected_label = state.selected_label
        self.command_buffer.reset()

    def _columns(self) -> int:
        size = self.terminal_size()
        return size[0] if size else 100

    def _header(self) -> StyleAndTextTuples:
        title = self.state.title
        if self.state.at_edge:
            title = f"BEAM OPTIONS   ·   {title.removeprefix('BEAM   ')}"
        return [("class:status-strong", title)]

    def _context(self) -> StyleAndTextTuples:
        return [
            ("class:section", "Shared context: "),
            ("", self.state.shared_context),
        ]

    def _survivors(self) -> StyleAndTextTuples:
        selected = self.selected_label or "—"
        fragments: StyleAndTextTuples = [
            ("class:section", "SURVIVORS"),
            ("class:muted", f" · selected {selected}\n"),
        ]
        if not self.state.rows:
            fragments.append(("class:muted", "  No retained branches.\n"))
            return fragments
        columns = self._columns()
        pane_width = (columns - 5) // 2 if columns >= 92 else columns
        continuation_width = max(4, pane_width - 26)
        for rank, row in enumerate(self.state.rows, 1):
            selected_row = row.label == self.selected_label
            style = "class:beam-selected" if selected_row else "class:table-row"
            marker = ">" if selected_row else " "
            status = f" · {row.state}"
            score = row.score.replace("-", "−")
            continuation = row.continuation.replace("\n", " ↵ ")
            if len(continuation) > continuation_width:
                continuation = continuation[:continuation_width - 1] + "…"
            fragments.extend(
                [
                    (style, f"{marker} {rank:>2} {row.label:<4} "),
                    (style, f"{continuation:<{continuation_width}}"),
                    ("class:beam-score", f"  {score:>8}{status}\n"),
                ]
            )
        return fragments

    def _details(self) -> StyleAndTextTuples:
        label = self.selected_label or "—"
        row = next(
            (item for item in self.state.rows if item.label == self.selected_label),
            None,
        )
        continuation = row.continuation if row is not None else "(no retained branch)"
        recent_steps = row.recent_steps if row is not None else ()
        fragments: StyleAndTextTuples = [
            ("class:section", f"SELECTED: {label}\n"),
            ("class:beam-continuation", continuation + "\n\n"),
            ("class:section", "Recent steps:\n"),
        ]
        if recent_steps:
            fragments.extend(
                ("class:table-row", f"  {step}\n")
                for step in recent_steps
            )
        else:
            fragments.append(("class:muted", "  No generated steps yet.\n"))
        if self.state.at_edge:
            fragments.append(
                ("class:muted", "\nChoose c/resume, discard, or q to quit.\n")
            )
        else:
            fragments.append(
                (
                    "class:muted",
                    "\nEOS likelihood is scored; selection commits text before EOS.\n",
                )
            )
        return fragments

    def _notice(self) -> StyleAndTextTuples:
        if not self.state.notice:
            return [("", "")]
        return [("class:feedback-info", self.state.notice)]

    def _footer(self) -> StyleAndTextTuples:
        if self.state.at_edge:
            text = "Enter/→ resume · discard restore episode · Esc return · q quit editor"
        else:
            text = (
                "↑↓ inspect · → step · ← rewind · Enter commit · Esc return · "
                "advance/hold N · kill ID · q options"
            )
        return [("class:hint", text)]

    def _move_selection(self, direction: int, event) -> None:
        labels = [row.label for row in self.state.rows]
        if not labels:
            return
        try:
            current = labels.index(self.selected_label)
        except ValueError:
            current = 0 if direction >= 0 else len(labels) - 1
        self.selected_label = labels[max(0, min(len(labels) - 1, current + direction))]
        event.app.invalidate()

    def _submit_command(self, event, command: str) -> None:
        self._finish(
            event,
            result=BeamInput(command, self.selected_label),
        )


__all__ = ["LiveBeamView"]

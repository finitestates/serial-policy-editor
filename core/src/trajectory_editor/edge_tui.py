"""Prompt-toolkit presentation for the live-edge command surface.

The live-edge menu owns no episode behavior.  It only presents the current
checkpoint and returns the raw command to ``episode_cli`` for interpretation.
Keeping that boundary small lets the plain/non-TTY menu and all of the
episode-side effects continue to use the existing code path.
"""

from __future__ import annotations

from dataclasses import asdict

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout

from .edge_help import edge_help
from .terminal_contracts import EdgeViewState
from .tui_views import ViewLifecycle


def _command_row(command: str, description: str) -> StyleAndTextTuples:
    return [
        ("class:help-key", f"  {command:<14}"),
        ("class:muted", description),
    ]


def _edge_header(
    *,
    episode_id: str,
    boundary: int,
    current_budget: int | None,
    remaining_tokens: int | None,
    sampler_summary: str,
    mode: str = "episode",
) -> StyleAndTextTuples:
    remaining_label = "token" if remaining_tokens == 1 else "tokens"
    fragments: StyleAndTextTuples = [
        ("class:status-strong", "LIVE SESSION\n" if mode == "session" else "LIVE EDGE\n"),
        ("class:section", "Branch " if mode == "session" else "Episode "),
        ("", f"{episode_id}"),
        ("class:muted", f"  ·  boundary {boundary}\n"),
        ("class:rule", "────────────────────────────────────────\n"),
        ("class:section", "Budget\n"),
        (
            "class:proposal",
            (f"  {remaining_tokens} {remaining_label} remaining  " if remaining_tokens is not None else "  Unlimited  "),
        ),
        (
            "class:muted",
            (f"default next allowance: {current_budget} tokens\n" if current_budget is not None else "no automatic checkpoint\n"),
        ),
        ("class:section", "Sampler\n"),
        ("class:muted", f"  {sampler_summary}\n"),
        ("class:section", "Commands\n"),
    ]
    for item in edge_help(mode):
        fragments.extend(_command_row(item.command, item.description + "\n"))
    return fragments


class LiveEdgeView(ViewLifecycle):
    """Reusable edge layout; commands remain interpreted by the episode CLI."""

    def __init__(self, state: EdgeViewState, *, submit, enabled=lambda: True):
        self.state = state
        super().__init__(submit=submit)
        self.command_buffer = command_buffer = Buffer(multiline=False, read_only=Condition(lambda: not enabled()))
        self.bindings = bindings = KeyBindings()

        @bindings.add("enter")
        def _submit(event: object) -> None:
            self._finish(event, result=command_buffer.text)  # type: ignore[attr-defined]

        @bindings.add("c-c")
        def _interrupt(event: object) -> None:
            self._finish(event, exception=KeyboardInterrupt())  # type: ignore[attr-defined]

        @bindings.add("c-d")
        def _close(event: object) -> None:
            self._finish(event, result=None)  # type: ignore[attr-defined]

        header = FormattedTextControl(lambda: _edge_header(**asdict(self.state)))
        input_control = BufferControl(buffer=command_buffer, focusable=True)
        prompt_row = VSplit(
            [
                Window(
                    FormattedTextControl([("class:prompt-label", "Command › ")]),
                    width=Dimension.exact(11),
                    height=1,
                ),
                Window(input_control, height=1, style="class:input"),
            ]
        )
        footer = FormattedTextControl(
            [
                (
                    "class:hint",
                    "Enter submits · blank continues · Ctrl-D quits",
                )
            ]
        )
        root = HSplit(
            [
                Window(header, wrap_lines=True),
                prompt_row,
                Window(
                    footer,
                    wrap_lines=True,
                    dont_extend_height=True,
                    always_hide_cursor=True,
                ),
            ]
        )
        self.layout = Layout(root, focused_element=input_control)

    def update(self, state: EdgeViewState) -> None:
        self.state = state
        self.command_buffer.reset()

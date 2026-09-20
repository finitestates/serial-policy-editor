"""Prompt-toolkit presentation for the live-edge command surface.

The live-edge menu owns no episode behavior.  It only presents the current
checkpoint and returns the raw command to ``episode_cli`` for interpretation.
Keeping that boundary small lets the plain/non-TTY menu and all of the
episode-side effects continue to use the existing code path.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout

from .ui_themes import DEFAULT_LIVE_THEME
from .tui_views import ViewLifecycle, run_standalone_view


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
    if mode == "session":
        fragments.extend(_command_row("branches", "show retained live branches\n"))
        fragments.extend(_command_row("#N", "switch to any retained root or branch\n"))
        fragments.extend(_command_row("switch N", "compatible branch-switch alias\n"))
        fragments.extend(_command_row("new TEXT", "start an unrelated prompt root\n"))
        fragments.extend(_command_row("rewind N", "trim this branch back to token N\n"))
    else:
        fragments.extend(_command_row("ls / ls all", "list open / all episodes\n"))
        fragments.extend(_command_row("#N", "switch episode\n"))
        fragments.extend(_command_row("new TEXT", "start an unrelated episode\n"))
        fragments.extend(_command_row("name TITLE", "rename this episode\n"))
        fragments.extend(_command_row("rewind N", "delete continuation from token N\n"))
    fragments.extend(_command_row("c / continue", "resume the current tranche\n"))
    fragments.extend(_command_row("n N / n off", "set an allowance or remove the budget\n"))
    fragments.extend(_command_row("s key=value", "change sampler settings\n"))
    if mode != "session":
        fragments.extend(
            _command_row("s random-seed", "choose and record a new random seed\n")
        )
    fragments.extend(
        _command_row("f N", "fork at boundary N\n")
    )
    if mode == "session":
        fragments.extend(_command_row("fm", "show the fork map and choose a boundary\n"))
    if mode == "session":
        fragments.extend(_command_row("export FILE", "write the selected portable tape\n"))
        fragments.extend(_command_row("save WORKSPACE", "materialize this branch\n"))
        fragments.extend(_command_row("save-family WORKSPACE", "materialize this live family\n"))
        fragments.extend(_command_row("e / end", "end this branch and show its text\n"))
        fragments.extend(_command_row("q / quit", "discard the whole live session\n"))
    else:
        fragments.extend(_command_row("spr ID", "replay from another episode\n"))
        fragments.extend(_command_row("p / project", "view the episode record\n"))
        fragments.extend(_command_row("e / end", "end and seal the episode\n"))
        fragments.extend(_command_row("q / quit", "leave without sealing\n"))
    return fragments


@dataclass(frozen=True)
class EdgeViewState:
    episode_id: str
    boundary: int
    current_budget: int | None
    remaining_tokens: int | None
    sampler_summary: str
    mode: str = "episode"


class LiveEdgeView(ViewLifecycle):
    """Reusable edge layout; commands remain interpreted by the episode CLI."""

    def __init__(self, state: EdgeViewState, *, submit=None, enabled=lambda: True):
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


def read_live_edge_command(
    *,
    episode_id: str,
    boundary: int,
    current_budget: int | None,
    remaining_tokens: int | None,
    sampler_summary: str,
    mode: str = "episode",
    input_device: object | None = None,
    output_device: object | None = None,
    theme: str = DEFAULT_LIVE_THEME,
) -> str | None:
    """Standalone edge adapter for callers without an interactive session."""
    view = LiveEdgeView(EdgeViewState(
        episode_id, boundary, current_budget, remaining_tokens, sampler_summary, mode,
    ))
    return run_standalone_view(
        view,
        theme=theme,
        input_device=input_device,
        output_device=output_device,
    )

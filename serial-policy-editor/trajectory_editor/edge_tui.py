"""Prompt-toolkit presentation for the live-edge command surface.

The live-edge menu owns no episode behavior.  It only presents the current
checkpoint and returns the raw command to ``episode_cli`` for interpretation.
Keeping that boundary small lets the plain/non-TTY menu and all of the
episode-side effects continue to use the existing code path.
"""

from __future__ import annotations

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout

from .live_tui import _live_style
from .ui_themes import DEFAULT_LIVE_THEME


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
) -> StyleAndTextTuples:
    remaining_label = "token" if remaining_tokens == 1 else "tokens"
    fragments: StyleAndTextTuples = [
        ("class:status-strong", "LIVE EDGE\n"),
        ("class:section", "Episode "),
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
    fragments.extend(_command_row("ls / ls all", "list open / all episodes\n"))
    fragments.extend(_command_row("#N", "switch episode\n"))
    fragments.extend(_command_row("name TITLE", "rename this episode\n"))
    fragments.extend(_command_row("rewind N", "delete continuation from token N\n"))
    fragments.extend(_command_row("c / continue", "resume the current tranche\n"))
    fragments.extend(_command_row("n N / n off", "set an allowance or remove the budget\n"))
    fragments.extend(_command_row("s key=value", "change sampler settings\n"))
    fragments.extend(
        _command_row("s random-seed", "choose and record a new random seed\n")
    )
    fragments.extend(
        _command_row("f N", "fork at boundary N  ·  fm shows the fork map\n")
    )
    fragments.extend(_command_row("spr ID", "replay from another episode\n"))
    fragments.extend(_command_row("p / project", "view the episode record\n"))
    fragments.extend(_command_row("e / end", "end and seal the episode\n"))
    fragments.extend(_command_row("q / quit", "leave without sealing\n"))
    return fragments


def read_live_edge_command(
    *,
    episode_id: str,
    boundary: int,
    current_budget: int | None,
    remaining_tokens: int | None,
    sampler_summary: str,
    input_device: object | None = None,
    output_device: object | None = None,
    theme: str = DEFAULT_LIVE_THEME,
) -> str | None:
    """Display the live-edge surface and return one unparsed command.

    Enter is the only submit action.  A blank command is intentionally
    returned as ``""`` because the existing edge-menu parser treats it as
    ``continue``.  Ctrl-D returns ``None`` and therefore retains the existing
    quit behavior.
    """
    command_buffer = Buffer(multiline=False)
    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit(event: object) -> None:
        event.app.exit(result=command_buffer.text)  # type: ignore[attr-defined]

    @bindings.add("c-c")
    def _interrupt(event: object) -> None:
        event.app.exit(exception=KeyboardInterrupt())  # type: ignore[attr-defined]

    @bindings.add("c-d")
    def _close(event: object) -> None:
        event.app.exit(result=None)  # type: ignore[attr-defined]

    header = FormattedTextControl(
        _edge_header(
            episode_id=episode_id,
            boundary=boundary,
            current_budget=current_budget,
            remaining_tokens=remaining_tokens,
            sampler_summary=sampler_summary,
        ),
    )
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
    layout = Layout(root, focused_element=input_control)
    application: Application[str | None] = Application(
        layout=layout,
        key_bindings=bindings,
        style=_live_style(theme),
        full_screen=True,
        # The outer live session owns the alternate screen and the visible
        # frame should remain until the next edge surface is rendered.
        erase_when_done=False,
        mouse_support=False,
        input=input_device,  # type: ignore[arg-type]
        output=output_device,  # type: ignore[arg-type]
    )
    return application.run()

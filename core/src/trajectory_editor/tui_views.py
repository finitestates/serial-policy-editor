"""Shared prompt-toolkit plumbing for the live terminal surfaces."""

from __future__ import annotations

from typing import Any

from prompt_toolkit.application import Application

from .ui_themes import DEFAULT_LIVE_THEME


class ViewLifecycle:
    """Common completion bridge used by persistent and standalone views."""

    def __init__(self, *, submit=None) -> None:
        self.submit = submit

    def _finish(self, event, *, result=None, exception=None) -> None:
        if self.submit is None:
            event.app.exit(result=result, exception=exception)
        else:
            self.submit(result=result, exception=exception)


def run_standalone_view(
    view: Any,
    *,
    theme: str = DEFAULT_LIVE_THEME,
    input_device: object | None = None,
    output_device: object | None = None,
) -> str | None:
    """Run a live view outside the persistent terminal session."""

    from .live_tui import _live_style

    application: Application[str | None] = Application(
        layout=view.layout,
        key_bindings=view.bindings,
        style=_live_style(theme),
        full_screen=True,
        erase_when_done=False,
        mouse_support=False,
        input=input_device,
        output=output_device,
    )
    return application.run()


__all__ = ["ViewLifecycle", "run_standalone_view"]

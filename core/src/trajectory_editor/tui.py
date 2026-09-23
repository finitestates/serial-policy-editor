"""Public terminal selection, session, and request entry point."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
from typing import Iterator

from .terminal_contracts import (
    ChoiceViewState, EdgeViewState, PromptRequest, TerminalCapabilities,
)
from .ui_themes import resolve_live_theme

__all__ = ["TerminalIO"]


class _SessionOutput(StringIO):
    """Hold incidental print output until fullscreen exits, showing live status."""

    def __init__(self, session, original):
        super().__init__()
        self.session = session
        self.original = original

    def write(self, text):
        result = super().write(text)
        self.session.write(text, end="")
        return result

    def isatty(self):
        return self.original.isatty()

    def fileno(self):
        return self.original.fileno()


class TerminalIO:
    def __init__(
        self,
        *,
        live_choices: bool | None = None,
        live_theme: str | None = None,
    ) -> None:
        self._live_theme = resolve_live_theme(live_theme)
        requested = True if live_choices is None else live_choices
        self._live_choices = bool(
            requested
            and sys.stdin.isatty()
            and sys.stdout.isatty()
            and importlib.util.find_spec("prompt_toolkit") is not None
        )
        self._live_session: object | None = None

    @property
    def supports_live_choices(self) -> bool:
        return self._live_choices

    @property
    def live_theme(self) -> str:
        return self._live_theme

    @property
    def capabilities(self) -> TerminalCapabilities:
        size = self.terminal_size()
        return TerminalCapabilities(
            live_views=self._live_choices,
            columns=size[0] if size else None,
            rows=size[1] if size else None,
            single_key=sys.stdin.isatty(),
        )

    def terminal_size(self) -> tuple[int, int] | None:
        if self._live_session is not None:
            return self._live_session.terminal_size()
        if not sys.stdout.isatty():
            return None
        size = shutil.get_terminal_size(fallback=(100, 30))
        return size.columns, size.lines

    @contextmanager
    def session(self) -> Iterator[object | None]:
        """Keep one terminal application alive throughout the interactive loop."""
        if not self._live_choices:
            yield None
            return
        if self._live_session is not None:
            raise RuntimeError("live session is already active")
        from .persistent_tui import PersistentTerminalSession

        session = PersistentTerminalSession(theme=self._live_theme)
        stdout, stderr = sys.stdout, sys.stderr
        captured_out = _SessionOutput(session, stdout)
        captured_err = _SessionOutput(session, stderr)
        try:
            with session:
                self._live_session = session
                try:
                    with redirect_stdout(captured_out), redirect_stderr(captured_err):
                        yield session
                finally:
                    self._live_session = None
        finally:
            # CLI summaries and errors belong to the restored normal screen.
            # Prompt-toolkit writes through the output captured before redirection.
            stdout.write(captured_out.getvalue())
            stderr.write(captured_err.getvalue())
            stdout.flush()
            stderr.flush()

    # Existing integrations may still use this spelling; both enter one context.
    live_session = session

    def read_choice(self, state: ChoiceViewState) -> str | None:
        if not self._live_choices:
            from .plain_tui import read_choice
            return read_choice(self, state)
        if self._live_session is None:
            raise RuntimeError("enter TerminalIO.session() before live requests")
        return self._live_session.read_choice(state)

    def read_edge(self, state: EdgeViewState) -> str | None:
        if not self._live_choices:
            from .plain_tui import read_edge
            return read_edge(self, state)
        if self._live_session is None:
            raise RuntimeError("enter TerminalIO.session() before live requests")
        return self._live_session.read_edge(state)

    def prompt(self, request: PromptRequest) -> str | None:
        if self._live_session is not None:
            return self._live_session.prompt(request)
        from .plain_tui import prompt
        return prompt(self, request)

    def read(self, prompt: str) -> str | None:
        return self.prompt(PromptRequest(prompt))

    def read_multiline_prompt(self) -> str | None:
        """Read a new root with the same composer used for initial prompts."""
        return self.prompt(PromptRequest(
            "Write at least one character. Press Escape then Enter to continue.\n\n",
            multiline=True,
        ))

    def read_key(self, prompt: str) -> str | None:
        """Read one unbuffered key without echoing it on an interactive TTY."""
        return self.prompt(PromptRequest(prompt, single_key=True))

    def write(self, text: str = "", *, end: str = "\n") -> None:
        if self._live_session is not None:
            if len(text.splitlines()) > 6:
                self._live_session.page(text)
            else:
                self._live_session.write(text, end=end)
            return
        print(text, end=end, flush=True)

    def page(self, text: str) -> None:
        self.prompt(PromptRequest("", body=text, page=True))

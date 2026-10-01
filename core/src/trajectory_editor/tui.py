"""Public terminal selection, session, and request entry point."""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from .terminal_contracts import (
    BeamInput,
    BeamViewState,
    ChoiceViewState,
    EdgeViewState,
    PromptRequest,
    TerminalCapabilities,
)
from .ui_themes import resolve_live_theme

__all__ = ["TerminalIO"]


def _live_terminal_supported() -> bool:
    """The live interface needs POSIX termios; elsewhere use the plain UI."""
    return os.name == "posix" and importlib.util.find_spec("termios") is not None


def _live_stream_ready(stream) -> bool:
    try:
        return bool(stream.isatty()) and stream.fileno() >= 0
    except (AttributeError, OSError, TypeError, ValueError):
        return False


class _ProcessOutputSilencer:
    """Send incidental process output to /dev/null while the live UI owns the TTY."""

    def __init__(self):
        self.terminal_output = None
        self._saved_fds: dict[int, int] = {}
        self._redirected_fds: set[int] = set()

    def start(self):
        if os.name != "posix":
            return None
        stderr_fd = sys.__stderr__.fileno()
        self.terminal_output = os.fdopen(
            os.dup(stderr_fd),
            "w",
            buffering=1,
            encoding=sys.__stderr__.encoding or "utf-8",
            errors=sys.__stderr__.errors or "replace",
        )
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.__stdout__.flush()
            sys.__stderr__.flush()
            null_fd = os.open(os.devnull, os.O_WRONLY)
            try:
                for target_fd in (1, 2):
                    self._saved_fds[target_fd] = os.dup(target_fd)
                    os.dup2(null_fd, target_fd)
                    self._redirected_fds.add(target_fd)
            finally:
                os.close(null_fd)
            return self.terminal_output
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
            try:
                stream.flush()
            except (OSError, ValueError):
                pass
        for target_fd in tuple(self._redirected_fds):
            saved_fd = self._saved_fds.get(target_fd)
            if saved_fd is not None:
                os.dup2(saved_fd, target_fd)
            self._redirected_fds.discard(target_fd)
        for saved_fd in self._saved_fds.values():
            os.close(saved_fd)
        self._saved_fds.clear()
        if self.terminal_output is not None:
            self.terminal_output.close()
            self.terminal_output = None


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
            and _live_stream_ready(sys.stdin)
            and _live_stream_ready(sys.stdout)
            and _live_terminal_supported()
        )
        self._live_session: object | None = None
        self._silencing_process_output = False
        self._after_session_output: list[str] = []

    @property
    def live_theme(self) -> str:
        return self._live_theme

    @property
    def capabilities(self) -> TerminalCapabilities:
        return TerminalCapabilities(
            live_views=self._live_choices,
            single_key=sys.stdin.isatty(),
            seamless_review=self._live_choices,
        )

    @contextmanager
    def session(self) -> Iterator[object | None]:
        """Keep one terminal application alive throughout the interactive loop."""
        if not self._live_choices:
            yield None
            return
        if self._live_session is not None:
            raise RuntimeError("live session is already active")
        from .term.app import TerminalSession

        process_output = _ProcessOutputSilencer()
        stdout = sys.stdout
        self._silencing_process_output = True
        try:
            terminal_output = process_output.start()
            session = TerminalSession(
                theme=self._live_theme,
                terminal_output=terminal_output,
            )
            with session:
                self._live_session = session
                try:
                    yield session
                finally:
                    self._live_session = None
        finally:
            try:
                process_output.stop()
            finally:
                self._silencing_process_output = False
                queued, self._after_session_output = self._after_session_output, []
                for text in queued:
                    stdout.write(text)
                if queued:
                    stdout.flush()

    def write_after_session(self, text: str, *, end: str = "\n") -> None:
        """Emit intentional output after the live TUI restores the terminal."""
        output = text + end
        if self._silencing_process_output:
            self._after_session_output.append(output)
            return
        sys.stdout.write(output)
        sys.stdout.flush()

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

    def read_beam(self, state: BeamViewState) -> BeamInput | None:
        if not self._live_choices:
            from .plain_tui import read_beam
            return read_beam(self, state)
        if self._live_session is None:
            raise RuntimeError("enter TerminalIO.session() before live requests")
        return self._live_session.read_beam(state)

    def prompt(self, request: PromptRequest) -> str | None:
        if self._live_session is not None:
            return self._live_session.prompt(request)
        if self._live_choices:
            raise RuntimeError("enter TerminalIO.session() before live requests")
        from .plain_tui import prompt
        return prompt(self, request)

    def read(self, prompt: str) -> str | None:
        return self.prompt(PromptRequest(prompt))

    def read_multiline_prompt(self) -> str | None:
        """Read a new root with the same composer used for initial prompts."""
        from .episode_prompts import read_new_prompt
        return read_new_prompt(self)

    def read_key(self, prompt: str) -> str | None:
        """Read one unbuffered key without echoing it on an interactive TTY."""
        return self.prompt(PromptRequest(prompt, single_key=True))

    def write(self, text: str = "", *, end: str = "\n") -> None:
        if self._live_session is not None:
            self._live_session.write(text, end=end)
            return
        print(text, end=end, flush=True)

    def page(self, text: str) -> None:
        self.prompt(PromptRequest("", body=text, page=True))

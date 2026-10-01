"""Public terminal selection, session, and request entry point."""

from __future__ import annotations

import codecs
import importlib.util
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from io import StringIO

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


class _SessionOutput(StringIO):
    """Hold incidental print output until the fullscreen session exits."""

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()

    def write(self, text):
        with self._lock:
            return super().write(text)

    def isatty(self):
        return False

    def fileno(self):
        return -1


class _ProcessOutputCapture:
    """Keep native stdout/stderr writes off the TTY while the live UI is active."""

    def __init__(self, stdout_capture: _SessionOutput, stderr_capture: _SessionOutput):
        self.stdout_capture = stdout_capture
        self.stderr_capture = stderr_capture
        self.terminal_output = None
        self._saved_fds: dict[int, int] = {}
        self._redirected_fds: set[int] = set()
        self._read_fds: list[int] = []
        self._write_fds: list[int] = []
        self._readers: list[threading.Thread] = []

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
        sinks = (
            (1, self.stdout_capture),
            (2, self.stderr_capture),
        )
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.__stdout__.flush()
            sys.__stderr__.flush()
            for target_fd, sink in sinks:
                self._saved_fds[target_fd] = os.dup(target_fd)
                read_fd, write_fd = os.pipe()
                self._read_fds.append(read_fd)
                self._write_fds.append(write_fd)
                source_stream = sys.__stdout__ if target_fd == 1 else sys.__stderr__
                reader = threading.Thread(
                    target=self._read_output,
                    args=(
                        read_fd,
                        sink,
                        source_stream.encoding or "utf-8",
                        source_stream.errors or "replace",
                    ),
                    name=f"spe-output-capture-{target_fd}",
                    daemon=True,
                )
                reader.start()
                self._readers.append(reader)
                os.dup2(write_fd, target_fd)
                self._redirected_fds.add(target_fd)
                os.close(write_fd)
                self._write_fds.remove(write_fd)
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
        for write_fd in self._write_fds:
            os.close(write_fd)
        self._write_fds.clear()
        for reader in self._readers:
            reader.join()
        self._readers.clear()
        for read_fd in self._read_fds:
            try:
                os.close(read_fd)
            except OSError:
                pass
        self._read_fds.clear()
        if self.terminal_output is not None:
            self.terminal_output.close()
            self.terminal_output = None

    @staticmethod
    def _read_output(
        read_fd: int,
        sink: _SessionOutput,
        encoding: str,
        errors: str,
    ) -> None:
        decoder = codecs.getincrementaldecoder(encoding)(errors=errors)
        while True:
            try:
                chunk = os.read(read_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                sink.write(text)
        remaining = decoder.decode(b"", final=True)
        if remaining:
            sink.write(remaining)


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

        stdout, stderr = sys.stdout, sys.stderr
        captured_out = _SessionOutput()
        captured_err = _SessionOutput()
        process_output = _ProcessOutputCapture(captured_out, captured_err)
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
            process_output.stop()
            # CLI summaries and errors belong to the restored normal screen.
            # Native and Python output are flushed after the live UI restores the TTY.
            stdout.write(captured_out.getvalue())
            stderr.write(captured_err.getvalue())
            stdout.flush()
            stderr.flush()

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

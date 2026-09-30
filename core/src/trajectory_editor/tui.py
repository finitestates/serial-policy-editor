"""Terminal request entry point and synchronous curses session lifecycle."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
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


def _live_stream_ready(stream) -> bool:
    try:
        return bool(stream.isatty()) and stream.fileno() >= 0
    except (AttributeError, OSError, TypeError, ValueError):
        return False


class _SessionOutput(StringIO):
    """Buffer Python output during curses and mirror it to the output viewer."""

    def __init__(self, on_write) -> None:
        super().__init__()
        self._on_write = on_write

    def write(self, text):
        count = super().write(text)
        if text:
            self._on_write(text)
        return count

    def isatty(self):
        return False

    def fileno(self):
        return -1


class _NativeOutputCapture:
    """Redirect native fd writes only while the engine owns the terminal."""

    def __init__(self, on_output) -> None:
        self._on_output = on_output
        self._stdout_file = tempfile.TemporaryFile(mode="w+b")
        self._stderr_file = tempfile.TemporaryFile(mode="w+b")
        self._saved_fds: dict[int, int] = {}
        self._read_offsets = {1: 0, 2: 0}
        self._redirected: set[int] = set()

    def save(self) -> None:
        for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                pass
        self._saved_fds = {1: os.dup(1), 2: os.dup(2)}

    def capture_for_engine(self) -> None:
        targets = ((1, self._stdout_file), (2, self._stderr_file))
        try:
            for target, file in targets:
                file.seek(0, os.SEEK_END)
                os.dup2(file.fileno(), target)
                self._redirected.add(target)
        except BaseException:
            self.restore_for_ui()
            raise

    def restore_for_ui(self) -> None:
        for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                pass
        for target in tuple(self._redirected):
            saved = self._saved_fds.get(target)
            if saved is not None:
                os.dup2(saved, target)
            self._redirected.discard(target)
        self._collect(1, self._stdout_file)
        self._collect(2, self._stderr_file)

    def _collect(self, descriptor: int, file) -> None:
        current = file.tell()
        file.seek(self._read_offsets[descriptor])
        chunk = file.read()
        self._read_offsets[descriptor] = file.tell()
        file.seek(current)
        if chunk:
            self._on_output(chunk.decode("utf-8", errors="replace"))

    def final_output(self) -> tuple[str, str]:
        self.restore_for_ui()
        self._stdout_file.seek(0)
        stdout = self._stdout_file.read().decode("utf-8", errors="replace")
        self._stderr_file.seek(0)
        stderr = self._stderr_file.read().decode("utf-8", errors="replace")
        return stdout, stderr

    def close(self) -> None:
        for descriptor in self._saved_fds.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._saved_fds.clear()
        self._stdout_file.close()
        self._stderr_file.close()


class TerminalIO:
    """Keep the action protocol independent from the terminal renderer."""

    def __init__(
        self,
        *,
        live_choices: bool | None = None,
        live_theme: str | None = None,
    ) -> None:
        self._live_theme = resolve_live_theme(live_theme)
        requested = True if live_choices is None else live_choices
        try:
            curses_available = importlib.util.find_spec("curses") is not None
        except (ImportError, ValueError):
            curses_available = False
        self._live_choices = bool(
            requested
            and curses_available
            and _live_stream_ready(sys.stdin)
            and _live_stream_ready(sys.stdout)
        )
        self._live_session = None

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
        """Keep one curses screen alive around the synchronous episode loop."""
        if not self._live_choices:
            yield None
            return
        if self._live_session is not None:
            raise RuntimeError("live session is already active")

        from .curses_tui import CursesTerminalSession, curses

        original_stdout, original_stderr = sys.stdout, sys.stderr
        terminal: CursesTerminalSession | None = None
        capture: _NativeOutputCapture | None = None
        captured_stdout: _SessionOutput | None = None
        captured_stderr: _SessionOutput | None = None
        opened = False

        def record(text: str) -> None:
            if terminal is not None:
                terminal.add_output(text)

        try:
            capture = _NativeOutputCapture(record)
            capture.save()
            terminal = CursesTerminalSession(
                theme=self._live_theme,
                environment=dict(os.environ),
                restore_output=capture.restore_for_ui,
                capture_output=capture.capture_for_engine,
            )
            terminal.open()
            opened = True
        except (curses.error, OSError):
            self._live_choices = False
            if terminal is not None:
                try:
                    terminal.close()
                except curses.error:
                    pass
            if capture is not None:
                capture.close()
            yield None
            return

        try:
            captured_stdout = _SessionOutput(record)
            captured_stderr = _SessionOutput(record)
            sys.stdout = captured_stdout
            sys.stderr = captured_stderr
            capture.capture_for_engine()
            self._live_session = terminal
            yield terminal
        finally:
            self._live_session = None
            native_stdout = native_stderr = ""
            try:
                if capture is not None:
                    try:
                        native_stdout, native_stderr = capture.final_output()
                    except OSError:
                        pass
            finally:
                try:
                    if opened and terminal is not None:
                        terminal.close()
                finally:
                    sys.stdout, sys.stderr = original_stdout, original_stderr
                    try:
                        if captured_stdout is not None:
                            original_stdout.write(captured_stdout.getvalue())
                            original_stdout.flush()
                        if captured_stderr is not None:
                            original_stderr.write(captured_stderr.getvalue())
                            original_stderr.flush()
                        if native_stdout:
                            original_stdout.write(native_stdout)
                            original_stdout.flush()
                        if native_stderr:
                            original_stderr.write(native_stderr)
                            original_stderr.flush()
                    finally:
                        if captured_stdout is not None:
                            captured_stdout.close()
                        if captured_stderr is not None:
                            captured_stderr.close()
                        if capture is not None:
                            capture.close()

    def _session(self):
        if self._live_session is None:
            raise RuntimeError("enter TerminalIO.session() before live requests")
        return self._live_session

    def read_choice(self, state: ChoiceViewState) -> str | None:
        if not self._live_choices:
            from .plain_tui import read_choice

            return read_choice(self, state)
        return self._session().read_choice(state)

    def read_edge(self, state: EdgeViewState) -> str | None:
        if not self._live_choices:
            from .plain_tui import read_edge

            return read_edge(self, state)
        return self._session().read_edge(state)

    def read_beam(self, state: BeamViewState) -> BeamInput | None:
        if not self._live_choices:
            from .plain_tui import read_beam

            return read_beam(self, state)
        return self._session().read_beam(state)

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
        """Read one key without echoing it on an interactive terminal."""
        return self.prompt(PromptRequest(prompt, single_key=True))

    def write(self, text: str = "", *, end: str = "\n") -> None:
        if self._live_session is not None:
            self._live_session.write(text, end=end)
            return
        print(text, end=end, flush=True)

    def page(self, text: str) -> None:
        if self._live_session is not None:
            self._live_session.page(text)
            return
        from .plain_tui import prompt

        prompt(self, PromptRequest("", body=text, page=True))

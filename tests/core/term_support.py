"""Headless driver for the live terminal UI: real views, keys, and frames."""

from __future__ import annotations

import os
from concurrent.futures import Future
from queue import Queue
from typing import Any

from trajectory_editor.core.candidates import Candidate
from trajectory_editor.core.ui import ChoiceSet
from trajectory_editor.term.app import Lifecycle, TerminalApp
from trajectory_editor.term.canvas import Canvas
from trajectory_editor.term.keys import InputParser, Key, Mouse, Paste
from trajectory_editor.terminal_contracts import (
    BeamViewRow,
    BeamViewState,
    ChoiceViewState,
    EdgeViewState,
    PromptRequest,
)


def choice_state(**changes: Any) -> ChoiceViewState:
    candidates = (
        Candidate(1, 2, " alpha", .8, False, .8),
        Candidate(2, 3, " beta", .2, False, .2),
    )
    choice = ChoiceSet(
        "choice", "prompt", 0, 0, "0" * 64, "context", 2, " alpha",
        .8, .8, False, candidates, vocabulary_size=5, proposal_raw_rank=1,
    )
    state = ChoiceViewState(
        choice,
        candidates,
        lambda text, mode: text,
        resolve_candidate=lambda rank: candidates[rank - 1],
    )
    return ChoiceViewState(**{**state.__dict__, **changes})


def edge_state(*, mode: str = "episode") -> EdgeViewState:
    return EdgeViewState("episode-1", 3, "temperature=0.7 · top_k=20", mode)


def beam_state(
    *, at_edge: bool = False, stochastic: bool = False, row_count: int = 2
) -> BeamViewState:
    rows = tuple(
        BeamViewRow(
            f"b{rank}",
            (
                "alpha continuation" if rank == 1 else
                "beta continuation" if rank == 2 else
                f"continuation for branch {rank:02d}"
            ),
            (
                "-0.45" if rank == 1 else
                "-1.25" if rank == 2 else
                f"-{0.45 + (rank - 1) * 0.4:.2f}"
            ),
            "LIVE",
            ("alpha",) if rank == 1 else ("beta",) if rank == 2 else (f"step {rank}",),
            model_rank=2 if rank == 1 else 4 if rank == 2 else rank + 1,
            step_log_probability=-0.25 * rank,
            model_log_probability=-0.8 - (rank - 1) * 0.4,
            protected=rank == 1,
            family_metadata="family A" if rank == 1 else "family B" if rank == 2 else f"family {rank}",
        )
        for rank in range(1, row_count + 1)
    )
    return BeamViewState(
        "BEAM · width 2 · depth 1",
        "shared context",
        rows,
        "b1",
        notice="beam notice",
        at_edge=at_edge,
        stochastic=stochastic,
    )


def prompt_state(prompt: str = "Input › ", **flags: bool) -> PromptRequest:
    return PromptRequest(prompt, **flags)

_NAMED = {
    "enter": "\r", "tab": "\t", "shift+tab": "\x1b[Z", "backspace": "\x7f",
    "up": "\x1b[A", "down": "\x1b[B", "right": "\x1b[C", "left": "\x1b[D",
    "pageup": "\x1b[5~", "pagedown": "\x1b[6~", "home": "\x1b[H", "end": "\x1b[F",
    "delete": "\x1b[3~", "f1": "\x1bOP", "f2": "\x1bOQ", "alt+enter": "\x1b\r",
}


class Harness:
    """Drive one :class:`TerminalApp` without a terminal.

    Keys go through the real byte parser; ``frame()`` renders the canvas the
    writer would present.
    """

    def __init__(self, state: Any = None, *, size: tuple[int, int] = (120, 40),
                 theme: str = "amber-cyan", environment: dict[str, str] | None = None) -> None:
        self._null = os.open(os.devnull, os.O_RDWR)
        self.app = TerminalApp(
            theme=theme,
            environment=environment if environment is not None else {"COLORTERM": "truecolor"},
            input_fd=self._null, output_fd=self._null,
        )
        self.size = size
        self.parser = InputParser()
        self.generation = 0
        self.lifecycle: Lifecycle | None = None
        self.previews: list[Any] = []
        if state is not None:
            self.show(state)

    def close(self) -> None:
        self.app.close_executors()
        os.close(self._null)
        for fd in (self.app._wake_read, self.app._wake_write):
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> Harness:  # noqa: PYI034 - Python 3.10 support excludes typing.Self.
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- requests ----------------------------------------------------------

    def show(self, state: Any) -> Lifecycle:
        self.generation += 1
        self.lifecycle = Lifecycle(self.generation, state, Future(), Queue())
        self.app.show_request(self.lifecycle)
        self.frame()
        return self.lifecycle

    @property
    def view(self):
        return self.app.view

    @property
    def done(self) -> bool:
        assert self.lifecycle is not None
        return self.lifecycle.response.done()

    def result(self) -> Any:
        assert self.lifecycle is not None
        assert self.lifecycle.response.done(), "request was not submitted"
        return self.lifecycle.response.result()

    def run_previews(self) -> None:
        """Answer owner-thread preview requests as the engine thread would."""
        assert self.lifecycle is not None
        queue = self.lifecycle.owner_queue
        while not queue.empty():
            preview = queue.get_nowait()
            try:
                value, error = preview.callback(), None
            except BaseException as exc:  # noqa: BLE001
                value, error = None, exc
            self.app.deliver_owner_preview(preview.generation, preview.key, value, error)
        self.app._run_posted()

    # -- input -------------------------------------------------------------

    def feed(self, data: str) -> None:
        self.app._dispatch(self.parser.feed(data))
        self.app._dispatch(self.parser.flush())
        self.app._run_posted()
        self.frame()

    def press(self, *keys: str) -> None:
        for key in keys:
            if key in _NAMED:
                self.feed(_NAMED[key])
            elif key == "escape":
                self.app._dispatch([Key("escape")])
                self.frame()
            elif key.startswith("ctrl+") and len(key) == 6:
                self.feed(chr(ord(key[-1]) - 96))
            elif len(key) == 1:
                self.feed(key)
            else:
                self.app._dispatch([Key(key)])
                self.frame()

    def type(self, text: str) -> None:
        for character in text:
            self.feed(character)

    def paste(self, text: str) -> None:
        self.app._dispatch([Paste(text)])
        self.frame()

    def click(self, x: int, y: int) -> None:
        self.app._dispatch([Mouse("press", x, y)])
        self.frame()

    def wheel(self, x: int, y: int, *, down: bool = True) -> None:
        self.app._dispatch([Mouse("wheel_down" if down else "wheel_up", x, y)])
        self.frame()

    # -- frames ------------------------------------------------------------

    def frame(self, size: tuple[int, int] | None = None) -> Canvas:
        if size is not None:
            self.size = size
        self.canvas = self.app.render_canvas(*self.size)
        return self.canvas

    @property
    def lines(self) -> list[str]:
        return [line.rstrip() for line in self.canvas.text_lines()]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def row_of(self, needle: str) -> int:
        for index, line in enumerate(self.lines):
            if needle in line:
                return index
        raise AssertionError(f"{needle!r} not on screen:\n{self.text}")

    def find(self, needle: str) -> tuple[int, int]:
        y = self.row_of(needle)
        return self.lines[y].index(needle), y


class LiveLoop:
    """The real UI thread and event loop, fed through a pipe instead of a TTY.

    Enter it on the thread that will make engine requests. Keys are written to
    the input pipe from any thread; ``frame()`` renders on the UI thread.
    """

    def __init__(self, *, size: tuple[int, int] = (120, 40), theme: str = "amber-cyan") -> None:
        from trajectory_editor.term.app import TerminalSession

        self.size = size
        self._read_fd, self._write_fd = os.pipe()
        self._null = os.open(os.devnull, os.O_WRONLY)
        self.session = TerminalSession(
            theme=theme, environment={"COLORTERM": "truecolor"},
            input_fd=self._read_fd, output_fd=self._null, size=size,
        )

    @property
    def app(self) -> TerminalApp:
        assert self.session.application is not None
        return self.session.application

    def __enter__(self) -> LiveLoop:  # noqa: PYI034 - Python 3.10 support excludes typing.Self.
        self.session.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.session.__exit__(*exc)
        finally:
            for fd in (self._write_fd, self._read_fd, self._null):
                try:
                    os.close(fd)
                except OSError:
                    pass

    def keys(self, data: str) -> None:
        os.write(self._write_fd, data.encode())

    def on_ui(self, function):
        result: Future = Future()

        def run() -> None:
            try:
                result.set_result(function())
            except BaseException as error:  # noqa: BLE001
                result.set_exception(error)

        self.app.post(run)
        return result.result(timeout=5)

    def frame(self) -> str:
        canvas = self.on_ui(lambda: self.app.render_canvas(*self.size))
        return "\n".join(line.rstrip() for line in canvas.text_lines())

    def wait_for(self, predicate, timeout: float = 5.0) -> str:
        import time

        deadline = time.monotonic() + timeout
        text = ""
        while time.monotonic() < deadline:
            text = self.frame()
            if predicate(text):
                return text
            time.sleep(0.01)
        raise AssertionError(f"condition not met; last frame:\n{text}")

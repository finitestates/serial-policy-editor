"""The UI thread, its event loop, and the blocking engine-thread bridge.

Threading model: the engine-owning thread calls the blocking request methods
on :class:`TerminalSession`. Every UI state change happens on the single UI
thread, which alternates between (1) draining all ready input bytes and posted
engine events and (2) rendering one frame from the resulting state. Other
threads only *post* work to the UI thread; they never touch UI state or the
terminal.
"""

from __future__ import annotations

import _thread
import codecs
import logging
import os
import selectors
import signal
import sys
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any

from ..teacher_commands import CommandKind, interpret_command
from ..terminal_contracts import (
    BeamInput,
    BeamViewState,
    ChoiceViewState,
    EdgeViewState,
    PromptRequest,
)
from ..tui_render import _safe_context_text
from ..ui_themes import DEFAULT_LIVE_THEME
from .canvas import Canvas, TextLayout
from .driver import FrameWriter, detect_color_system, terminal_size
from .keys import InputParser, Key, Mouse, Paste
from .overlays import DocumentOverlay, OutputOverlay, Overlay, PaletteOverlay
from .palette import PaletteEntry
from .views import BeamView, ChoiceView, EdgeView, PromptView, RequestView
from .widgets import RenderContext, Styles

if os.name == "posix":
    import termios

_LOG = logging.getLogger(__name__)

ESCAPE_TIMEOUT = 0.04
SIZE_POLL_INTERVAL = 0.25
OUTPUT_HISTORY_LIMIT = 16_000


@dataclass(frozen=True)
class OwnerPreview:
    generation: int
    key: tuple[Any, ...]
    callback: Callable[[], Any]


@dataclass
class Lifecycle:
    generation: int
    state: ChoiceViewState | EdgeViewState | BeamViewState | PromptRequest
    response: Future[Any]
    owner_queue: Queue[OwnerPreview] | None
    warm_cancelled: threading.Event = field(default_factory=threading.Event)
    warm_future: Future[Any] | None = None
    warm_result: bool | None = None
    submitted_result: Any = None
    submitted_exception: BaseException | None = None
    submitted_target: tuple[int, int] | None = None


def _submission_target(state: ChoiceViewState, raw: Any) -> tuple[int, int] | None:
    if not isinstance(raw, str):
        return None
    interpretation = interpret_command(
        raw,
        menu_size=len(state.choice.candidates),
        default_hold_tokens=state.default_hold_tokens,
        vocabulary_size=state.choice.vocabulary_size or len(state.candidates),
        default_search_radius=state.default_search_radius,
    )
    command = interpretation.command
    if command is None or command.kind != CommandKind.EDIT or command.action is None:
        return None
    if command.action.kind.value == "accept":
        rank = state.choice.proposal_raw_rank
    elif command.action.kind.value == "select":
        rank = int(command.action.selected_rank)
    else:
        return None
    candidate = next((row for row in state.candidates if row.rank == rank), None)
    if candidate is None and rank == state.choice.proposal_raw_rank:
        return rank, state.choice.proposal_token_id
    return None if candidate is None else (candidate.rank, candidate.token_id)


class TerminalApp:
    """All UI state, owned and mutated only by the UI thread."""

    def __init__(
        self,
        *,
        theme: str = DEFAULT_LIVE_THEME,
        environment: dict[str, str] | None = None,
        input_fd: int | None = None,
        output_fd: int | None = None,
        interrupt_owner: bool = False,
        size: tuple[int, int] | None = None,
    ) -> None:
        self.environment = dict(os.environ if environment is None else environment)
        self.color_system = detect_color_system(self.environment)
        self.styles = Styles(theme, self.environment, self.color_system)
        self.layout = TextLayout(self.color_system)
        self.input_fd = sys.__stdin__.fileno() if input_fd is None else input_fd
        self.output_fd = sys.__stdout__.fileno() if output_fd is None else output_fd
        self.writer = FrameWriter(
            self.output_fd, self.color_system,
            frame_log=self.environment.get("SPE_TERMINAL_FRAME_LOG") or None,
        )
        self.interrupt_owner = interrupt_owner
        self.fixed_size = size
        self.stats: Counter[str] = Counter()
        self.view: RequestView | None = None
        self.overlay: Overlay | None = None
        self.active: Lifecycle | None = None
        self._posted: deque[tuple[Callable[..., Any], tuple[Any, ...]]] = deque()
        self._posted_lock = threading.Lock()
        self._wake_read, self._wake_write = os.pipe()
        os.set_blocking(self._wake_read, False)
        os.set_blocking(self._wake_write, False)
        self._parser = InputParser()
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._stop = False
        self._dirty = True
        self._size: tuple[int, int] | None = None
        self._last_size_check = 0.0
        self._hits: list = []
        self._wheels: list = []
        self._saved_attrs: Any = None
        self._output_chunks: deque[str] = deque()
        self._output_chars = 0
        self.warm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spe-search-warm")
        self.running = False

    # -- cross-thread entry points ------------------------------------------

    def post(self, function: Callable[..., Any], *args: Any) -> None:
        """Queue ``function(*args)`` to run on the UI thread."""
        with self._posted_lock:
            self._posted.append((function, args))
        try:
            os.write(self._wake_write, b"x")
        except (BlockingIOError, OSError):
            pass

    def stop(self) -> None:
        self.post(self._request_stop)

    def _request_stop(self) -> None:
        self._stop = True

    # -- terminal setup ------------------------------------------------------

    def _enter_terminal(self) -> None:
        if os.name == "posix" and os.isatty(self.input_fd):
            self._saved_attrs = termios.tcgetattr(self.input_fd)
            self._set_raw()
        self.writer.enter()

    def _set_raw(self) -> None:
        attrs = termios.tcgetattr(self.input_fd)
        attrs[0] &= ~(termios.BRKINT | termios.ICRNL | termios.INPCK | termios.ISTRIP | termios.IXON)
        attrs[2] = (attrs[2] & ~(termios.CSIZE | termios.PARENB)) | termios.CS8
        attrs[3] &= ~(termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG)
        attrs[6][termios.VMIN] = 1
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self.input_fd, termios.TCSANOW, attrs)

    def _leave_terminal(self) -> None:
        try:
            self.writer.exit()
        finally:
            if self._saved_attrs is not None:
                termios.tcsetattr(self.input_fd, termios.TCSADRAIN, self._saved_attrs)

    def suspend(self) -> None:
        """Ctrl+Z: restore the shell's terminal, stop, then repaint on resume."""
        self._leave_terminal()
        try:
            os.kill(os.getpid(), signal.SIGTSTP)
        finally:
            self.resume()

    def resume(self) -> None:
        if self._saved_attrs is not None:
            self._set_raw()
        self.writer.enter()
        self._dirty = True

    # -- event loop ----------------------------------------------------------

    def run(self, ready: Future[None]) -> None:
        selector = selectors.DefaultSelector()
        self._enter_terminal()
        self.running = True
        try:
            selector.register(self.input_fd, selectors.EVENT_READ, "input")
            selector.register(self._wake_read, selectors.EVENT_READ, "wake")
            ready.set_result(None)
            self._render()
            while not self._stop:
                timeout = ESCAPE_TIMEOUT if self._parser.pending else SIZE_POLL_INTERVAL
                ready_keys = selector.select(timeout)
                got_input = False
                for key, _mask in ready_keys:
                    if key.data == "input":
                        try:
                            data = os.read(self.input_fd, 65536)
                        except InterruptedError:
                            continue
                        if not data:
                            raise EOFError("terminal input closed")
                        got_input = True
                        self._dispatch(self._parser.feed(self._decoder.decode(data)))
                    else:
                        try:
                            while os.read(self._wake_read, 4096):
                                pass
                        except (BlockingIOError, OSError):
                            pass
                if not got_input and self._parser.pending:
                    self._dispatch(self._parser.flush())
                self._run_posted()
                self._check_size()
                if self._dirty:
                    self._render()
        except BaseException as error:
            self._fail(error)
            if not ready.done():
                ready.set_exception(error)
            raise
        finally:
            self.running = False
            selector.close()
            try:
                self._leave_terminal()
            finally:
                os.close(self._wake_read)
                os.close(self._wake_write)

    def _fail(self, error: BaseException) -> None:
        lifecycle = self.active
        if lifecycle is not None and not lifecycle.response.done():
            lifecycle.response.set_exception(error)

    def _run_posted(self) -> None:
        while True:
            with self._posted_lock:
                if not self._posted:
                    return
                function, args = self._posted.popleft()
            function(*args)
            self._dirty = True

    def _check_size(self) -> None:
        now = time.monotonic()
        if self._dirty or now - self._last_size_check >= SIZE_POLL_INTERVAL:
            self._last_size_check = now
            if self.terminal_size() != self._size:
                self._dirty = True

    # -- rendering -----------------------------------------------------------

    def terminal_size(self) -> tuple[int, int]:
        return self.fixed_size or terminal_size(self.output_fd, self.input_fd)

    def render_canvas(self, width: int, height: int) -> Canvas:
        """Compose one frame from the current state."""
        canvas = Canvas(width, height, self.styles.base)
        ctx = RenderContext(canvas, self.layout, self.styles)
        if width < 20 or height < 3:
            message = "Enlarge the terminal"
            canvas.put(max(0, (width - len(message)) // 2), height // 2, message, self.styles("feedback-info"))
        elif self.view is not None:
            self.view.render(ctx)
            if self.overlay is not None:
                canvas.cursor = None
                ctx.hits.clear()
                ctx.scroll_regions.clear()
                self.overlay.render(ctx)
        else:
            canvas.put(1, 0, "Policy editor · starting…", self.styles("muted"))
        self._hits = ctx.hits
        self._wheels = ctx.scroll_regions
        return canvas

    def _render(self) -> None:
        self._dirty = False
        size = self.terminal_size()
        if size != self._size:
            self._size = size
            self.writer.invalidate()
        canvas = self.render_canvas(*size)
        if self.writer.present(canvas):
            self.stats["frames"] += 1

    # -- input ---------------------------------------------------------------

    def _dispatch(self, events: list) -> None:
        for event in events:
            self._dirty = True
            self._handle(event)

    def _handle(self, event: Key | Paste | Mouse) -> None:
        if isinstance(event, Key) and event.name == "ctrl+z":
            self.suspend()
            return
        view = self.view
        if view is None or not view.accepting:
            self.stats["dropped_input"] += 1
            if isinstance(event, Key) and event.name == "ctrl+c" and self.interrupt_owner:
                # The engine is working; interrupt it as a plain CLI would.
                _thread.interrupt_main()
            return
        if isinstance(event, Key):
            self._handle_key(view, event)
        elif isinstance(event, Paste):
            if self.overlay is not None:
                self.overlay.on_paste(event.text)
            else:
                view.on_paste(event.text)
        else:
            self._handle_mouse(event)

    def _handle_key(self, view: RequestView, key: Key) -> None:
        if key.name == "ctrl+c":
            self.submit_exception(view, KeyboardInterrupt())
            return
        if self.overlay is not None:
            if self.overlay.on_key(key):
                self.overlay = None
            return
        if key.name == "f1":
            self.open_help()
        elif key.name == "ctrl+l":
            self.overlay = OutputOverlay(self.output_history)
        elif key.name == "ctrl+k":
            if view.palette_enabled and view.palette_entries():
                self.overlay = PaletteOverlay(view.palette_entries(), self._palette_chosen)
        else:
            view.on_key(key)

    def _handle_mouse(self, event: Mouse) -> None:
        if event.kind in {"wheel_up", "wheel_down"}:
            for region, scroll in reversed(self._wheels):
                if region.contains(event.x, event.y):
                    scroll(-3 if event.kind == "wheel_up" else 3)
                    return
            return
        if event.kind != "press" or event.button != 0:
            return
        for hit in reversed(self._hits):
            if hit.contains(event.x, event.y):
                hit.action()
                return

    def _palette_chosen(self, entry: PaletteEntry) -> None:
        view = self.view
        if view is None or not view.accepting:
            return
        if entry.title == "help":
            self.overlay = None
            self.open_help()
            return
        view.insert_command(entry.insert)

    def open_help(self) -> None:
        if self.view is not None:
            self.overlay = DocumentOverlay("Help", self.view.help_document())

    def open_details(self, document) -> None:
        self.overlay = DocumentOverlay("Details", document, close_keys={"escape", "q", "f2"})

    # -- requests ------------------------------------------------------------

    def show_request(self, lifecycle: Lifecycle) -> None:
        state = lifecycle.state
        if isinstance(state, ChoiceViewState):
            view: RequestView = ChoiceView(self, lifecycle)
        elif isinstance(state, EdgeViewState):
            view = EdgeView(self, lifecycle)
        elif isinstance(state, BeamViewState):
            view = BeamView(self, lifecycle)
        else:
            view = PromptView(self, lifecycle)
        self.active = lifecycle
        self.overlay = None
        self.view = view
        view.accepting = True
        if isinstance(state, ChoiceViewState):
            self._start_search_warm(lifecycle, view)

    def submit(self, view: RequestView, value: Any) -> None:
        if view is not self.view or not view.accepting:
            return
        view.accepting = False
        self.overlay = None
        lifecycle = view.lifecycle
        lifecycle.submitted_result = value
        state = lifecycle.state
        if isinstance(state, ChoiceViewState):
            lifecycle.submitted_target = _submission_target(state, value)
            self._update_warm_cancellation(lifecycle, value)
        if not lifecycle.response.done():
            lifecycle.response.set_result(value)

    def submit_exception(self, view: RequestView, error: BaseException) -> None:
        if view is not self.view or not view.accepting:
            return
        view.accepting = False
        self.overlay = None
        lifecycle = view.lifecycle
        lifecycle.submitted_exception = error
        if not lifecycle.response.done():
            lifecycle.response.set_exception(error)

    def _update_warm_cancellation(self, lifecycle: Lifecycle, result: Any) -> None:
        state = lifecycle.state
        target = state.search_warm_target
        if target is None:
            return
        submitted_target = lifecycle.submitted_target
        if (submitted_target is not None and submitted_target != target) or (
            isinstance(result, str)
            and result.strip().startswith("/")
            and result.strip() not in state.search_warm_commands
        ):
            lifecycle.warm_cancelled.set()

    def _start_search_warm(self, lifecycle: Lifecycle, view: RequestView) -> None:
        state = lifecycle.state
        if state.warm_search_token is None or state.search_warm_target is None:
            return
        target = state.search_warm_target
        if state.search_warm_prepared:
            lifecycle.warm_result = True
            if isinstance(view, ChoiceView):
                view.warm_completed(target, True, None)
            return
        generation = lifecycle.generation
        if isinstance(view, ChoiceView):
            view.warm_started(target)
        lifecycle.warm_future = self.warm_executor.submit(
            state.warm_search_token, target[0], target[1], generation,
            lifecycle.warm_cancelled.is_set,
        )
        self.stats["warm_dispatches"] += 1

        def finished(future: Future[Any]) -> None:
            try:
                value, error = bool(future.result()), None
            except FutureCancelledError:
                return
            except BaseException as exc:  # noqa: BLE001 - worker failures return through the UI.
                value, error = False, exc
            self.post(self._warm_completed, generation, target, value, error)

        lifecycle.warm_future.add_done_callback(finished)

    def _warm_completed(self, generation: int, target: tuple[int, int], value: bool,
                        error: BaseException | None) -> None:
        lifecycle = self.active
        if lifecycle is None or lifecycle.generation != generation:
            return
        lifecycle.warm_result = value and error is None
        if isinstance(self.view, ChoiceView) and self.view.generation == generation:
            self.view.warm_completed(target, value, error)

    def request_owner_preview(self, generation: int, key: tuple[Any, ...],
                              callback: Callable[[], Any]) -> None:
        lifecycle = self.active
        if lifecycle is None or lifecycle.generation != generation or lifecycle.owner_queue is None:
            return
        lifecycle.owner_queue.put(OwnerPreview(generation, key, callback))

    def deliver_owner_preview(self, generation: int, key: tuple[Any, ...], result: Any,
                              error: BaseException | None) -> None:
        view = self.view
        if view is not None and view.generation == generation:
            view.owner_preview_ready(key, result, error)

    # -- captured output -----------------------------------------------------

    def write_output(self, text: str) -> None:
        safe = _safe_context_text(text)
        if not safe:
            return
        safe = safe[-OUTPUT_HISTORY_LIMIT:]
        self._output_chunks.append(safe)
        self._output_chars += len(safe)
        while self._output_chars > OUTPUT_HISTORY_LIMIT:
            oldest = self._output_chunks.popleft()
            excess = self._output_chars - OUTPUT_HISTORY_LIMIT
            if len(oldest) > excess:
                self._output_chunks.appendleft(oldest[excess:])
                self._output_chars -= excess
            else:
                self._output_chars -= len(oldest)

    def output_history(self) -> str:
        if len(self._output_chunks) > 1:
            joined = "".join(self._output_chunks)
            self._output_chunks = deque([joined])
        return self._output_chunks[0] if self._output_chunks else ""

    def close_executors(self) -> None:
        self.warm_executor.shutdown(wait=True, cancel_futures=True)


class TerminalSession(AbstractContextManager["TerminalSession"]):
    """Run the UI thread while the calling thread owns engine work."""

    def __init__(
        self,
        *,
        theme: str = DEFAULT_LIVE_THEME,
        environment: dict[str, str] | None = None,
        terminal_output: Any = None,
        input_fd: int | None = None,
        output_fd: int | None = None,
        size: tuple[int, int] | None = None,
    ) -> None:
        self.theme = theme
        self._input_fd = input_fd
        self._output_fd = output_fd
        self._size = size
        self.environment = environment
        self.application: TerminalApp | None = None
        self._terminal_output = terminal_output
        self._ready: Future[None] = Future()
        self._thread: threading.Thread | None = None
        self._owner: int | None = None
        self._failure: BaseException | None = None
        self._closing = False
        self._generation = 0
        self._previous_signal_handlers: dict[signal.Signals, Any] = {}

    def __enter__(self) -> TerminalSession:  # noqa: PYI034
        if self._thread is not None:
            raise RuntimeError("terminal session cannot be entered twice")
        self._owner = threading.get_ident()
        on_main = threading.current_thread() is threading.main_thread()
        output_fd = self._output_fd
        if output_fd is None:
            output_fd = (
                self._terminal_output.fileno() if self._terminal_output is not None
                else sys.__stdout__.fileno()
            )
        self.application = TerminalApp(
            theme=self.theme,
            environment=self.environment,
            input_fd=self._input_fd,
            output_fd=output_fd,
            interrupt_owner=on_main,
            size=self._size,
        )
        self._install_signal_handlers()
        self._thread = threading.Thread(target=self._run_app, name="spe-terminal-ui")
        self._thread.start()
        try:
            self._ready.result()
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._closing = True
        app = self.application
        if app is not None and self._thread is not None and self._thread.is_alive():
            app.stop()
        if self._thread is not None:
            self._thread.join()
        if app is not None:
            app.close_executors()
        self._restore_signal_handlers()
        if exc_type is None and self._failure is not None and not isinstance(self._failure, EOFError):
            raise self._failure
        return False

    def _run_app(self) -> None:
        assert self.application is not None
        try:
            self.application.run(self._ready)
        except BaseException as error:  # noqa: BLE001 - signal UI failure to the blocked engine.
            self._failure = error
            if not self._ready.done():
                self._ready.set_exception(error)

    def _install_signal_handlers(self) -> None:
        if os.name != "posix" or threading.current_thread() is not threading.main_thread():
            return

        def wake(_signum: int, _frame: Any) -> None:
            app = self.application
            if app is not None and app.running:
                app.post(lambda: None)

        def resumed(_signum: int, _frame: Any) -> None:
            app = self.application
            if app is not None and app.running:
                app.post(app.resume)

        for signum, handler in ((signal.SIGWINCH, wake), (signal.SIGCONT, resumed)):
            self._previous_signal_handlers[signum] = signal.signal(signum, handler)

    def _restore_signal_handlers(self) -> None:
        if threading.current_thread() is threading.main_thread():
            for signum, handler in self._previous_signal_handlers.items():
                signal.signal(signum, handler)
        self._previous_signal_handlers.clear()

    def _read(self, state: Any) -> Any:
        if threading.get_ident() != self._owner:
            raise RuntimeError("terminal requests must come from the engine-owning thread")
        if self._closing or self._thread is None or not self._thread.is_alive():
            raise self._failure or EOFError("terminal session is closed")
        app = self.application
        assert app is not None
        self._generation += 1
        lifecycle = Lifecycle(
            generation=self._generation, state=state, response=Future(), owner_queue=Queue(),
        )
        try:
            app.post(app.show_request, lifecycle)
            while True:
                try:
                    return lifecycle.response.result(timeout=0.05)
                except TimeoutError:
                    pass
                if self._failure is not None:
                    raise self._failure
                if not self._thread.is_alive():
                    raise EOFError("terminal input closed")
                self._service_previews(app, lifecycle)
        finally:
            self._finish_warm(lifecycle)

    def _service_previews(self, app: TerminalApp, lifecycle: Lifecycle) -> None:
        assert lifecycle.owner_queue is not None
        while True:
            try:
                preview = lifecycle.owner_queue.get_nowait()
            except Empty:
                return
            try:
                value, error = preview.callback(), None
            except BaseException as exc:  # noqa: BLE001 - deliver preview failures to the UI.
                value, error = None, exc
            app.post(app.deliver_owner_preview, preview.generation, preview.key, value, error)

    def _finish_warm(self, lifecycle: Lifecycle) -> None:
        state = lifecycle.state
        if not isinstance(state, ChoiceViewState) or state.warm_search_token is None:
            return
        target = state.search_warm_target
        keep = False
        warm_future = lifecycle.warm_future
        if target is not None and lifecycle.submitted_exception is None:
            if warm_future is not None:
                try:
                    warm_value = bool(warm_future.result())
                except Exception:  # noqa: BLE001 - a failed warm search is a cache miss.
                    warm_value = False
            else:
                warm_value = bool(state.search_warm_prepared)
            raw = lifecycle.submitted_result
            if warm_value:
                if lifecycle.submitted_target is not None:
                    keep = lifecycle.submitted_target == target
                elif isinstance(raw, str):
                    keep = (
                        raw.strip() in state.search_warm_commands
                        if raw.strip().startswith("/")
                        else True
                    )
        if keep:
            if lifecycle.submitted_target == target and self.application is not None:
                self.application.stats["promotions"] += 1
            return
        lifecycle.warm_cancelled.set()
        if warm_future is not None:
            try:
                warm_future.result()
            except Exception:  # noqa: BLE001 - cancellation failures do not change the command.
                _LOG.warning("warm search failed while cancelling", exc_info=True)
        cancel = state.cancel_search_warm
        if callable(cancel) and self.application is not None:
            self.application.warm_executor.submit(cancel).result()

    def read_choice(self, state: ChoiceViewState) -> str | None:
        return self._read(state)

    def read_edge(self, state: EdgeViewState) -> str | None:
        return self._read(state)

    def read_beam(self, state: BeamViewState) -> BeamInput | None:
        return self._read(state)

    def prompt(self, request: PromptRequest) -> str | None:
        return self._read(request)

    def read(self, prompt: str) -> str | None:
        return self.prompt(PromptRequest(prompt))

    def read_key(self, prompt: str) -> str | None:
        return self.prompt(PromptRequest(prompt, single_key=True))

    def write(self, text: str = "", *, end: str = "\n") -> None:
        app = self.application
        if app is None or self._closing or self._thread is None or not self._thread.is_alive():
            return
        app.post(app.write_output, text + end)

    def page(self, text: str) -> None:
        self.prompt(PromptRequest("", body=text, page=True))

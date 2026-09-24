"""One terminal application, with synchronous episode work on its owning thread.

The UI thread owns widgets and terminal I/O. The caller owns the engine, its
backend and SQLite connection. Requests and preview futures are the only bridge;
no renderer or key handler calls the backend. A submitted view immediately stops
accepting input, so queued keystrokes cannot commit against an obsolete decision.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from _thread import interrupt_main
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from queue import Queue
from typing import Callable, Any

from prompt_toolkit.application import Application, create_app_session
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import (
    KeyBindings,
    DynamicKeyBindings,
    merge_key_bindings,
)
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import (
    HSplit,
    Window,
    DynamicContainer,
    ConditionalContainer,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import TextArea

from .edge_tui import LiveEdgeView
from .core.errors import EditorError
from .live_tui import (
    LiveChoiceView,
    PreviewPending,
    action_preview,
    _live_style,
    _safe_context_text,
)
from .terminal_contracts import ChoiceViewState, EdgeViewState, PromptRequest
from .ui_themes import DEFAULT_LIVE_THEME


# Default fallback matching the baseline
DEFAULT_WARM_SELECTION_DELAY = 0.15


@dataclass(eq=False)
class _Request:
    state: ChoiceViewState | EdgeViewState | PromptRequest
    response: Future = field(default_factory=Future)
    previews: OrderedDict = field(default_factory=OrderedDict)
    latest: dict[str, tuple] = field(default_factory=dict)
    insertion_display: dict[Any, str] = field(default_factory=dict)
    warm_target: tuple[int, int] | None = None
    submitted_target: tuple[int, int] | None = None
    warm_generation: int | None = None
    warm_cancelled: threading.Event = field(default_factory=threading.Event)
    warm_timer: asyncio.TimerHandle | None = None
    warm_future: Future | None = None


@dataclass
class _Resolution:
    request: _Request
    callback: Callable[[], Any]
    result: Future


class _PromptView:
    """In-application prompts, confirmations and scrollable documents."""

    def __init__(self, submit, enabled):
        self.state = PromptRequest("")
        self._error = ""
        self.submit = submit
        self.command_buffer = Buffer(
            multiline=True, read_only=Condition(lambda: not enabled())
        )
        self.body = TextArea(read_only=True, scrollbar=True, wrap_lines=True)
        self.input_control = BufferControl(buffer=self.command_buffer)
        self.layout = Layout(
            HSplit(
                [
                    ConditionalContainer(
                        self.body, Condition(lambda: bool(self.state.body))
                    ),
                    ConditionalContainer(
                        Window(
                            FormattedTextControl(lambda: self.state.prompt),
                            dont_extend_height=True,
                        ),
                        Condition(lambda: not self.state.page),
                    ),
                    ConditionalContainer(
                        Window(
                            self.input_control,
                            height=lambda: (
                                Dimension.exact(1)
                                if not self.state.multiline
                                else Dimension(min=3, preferred=8)
                            ),
                            style="class:input",
                            wrap_lines=True,
                        ),
                        Condition(lambda: not self.state.page),
                    ),
                    Window(
                        FormattedTextControl(
                            lambda: (
                                self._error
                                or (
                                    "↑/↓ · PgUp/PgDn scroll · Enter/Esc returns"
                                    if self.state.page
                                    else "Press a key"
                                    if self.state.single_key
                                    else "Escape then Enter submits · Ctrl-D cancels"
                                    if self.state.multiline
                                    else "PgUp/PgDn scroll · Enter submits · Ctrl-D cancels"
                                    if self.state.body
                                    else "Enter submits · Ctrl-D cancels"
                                )
                            )
                        ),
                        height=1,
                        style="class:hint",
                    ),
                ]
            )
        )
        self.bindings = KeyBindings()

        @self.bindings.add("enter", filter=Condition(lambda: not self.state.multiline))
        def enter(event):
            self.submit(
                result=""
                if self.state.page
                else "\n"
                if self.state.single_key
                else self.command_buffer.text
            )

        @self.bindings.add(
            "escape",
            "enter",
            eager=True,
            filter=Condition(lambda: self.state.multiline),
        )
        def submit_multiline(event):
            if not self.command_buffer.text:
                self._error = "Write at least one character."
                event.app.invalidate()
                return
            self.submit(result=self.command_buffer.text)

        @self.bindings.add(
            "pageup",
            filter=Condition(lambda: bool(self.state.body) and not self.state.page),
        )
        def scroll_up(event):
            self.body.window.vertical_scroll = max(
                0, self.body.window.vertical_scroll - 10
            )
            event.app.invalidate()

        @self.bindings.add(
            "pagedown",
            filter=Condition(lambda: bool(self.state.body) and not self.state.page),
        )
        def scroll_down(event):
            self.body.window.vertical_scroll += 10
            event.app.invalidate()

        @self.bindings.add(
            "escape", eager=True, filter=Condition(lambda: not self.state.multiline)
        )
        def escape(event):
            self.submit(
                result="\x1b"
                if self.state.single_key
                else ""
                if self.state.page
                else None
            )

        @self.bindings.add("c-d")
        def eof(event):
            self.submit(result=None)

        @self.bindings.add("q", filter=Condition(lambda: self.state.page))
        def close_page(event):
            self.submit(result="")

        @self.bindings.add(Keys.Any, filter=Condition(lambda: self.state.single_key))
        def key(event):
            self.submit(result=event.data)

        @self.bindings.add("backspace", filter=Condition(lambda: self.state.single_key))
        def backspace(event):
            self.submit(result="\x7f")

    def update(self, state: PromptRequest) -> None:
        self.state = state
        self._error = ""
        self.command_buffer.reset()
        self.body.buffer.set_document(
            Document(_safe_context_text(state.body)), bypass_readonly=True
        )
        self.layout.focus(self.body if state.page else self.input_control)


class PersistentTerminalSession(AbstractContextManager):
    """Keep one Application and renderer alive across every interactive surface.

    Enter and call read methods from the episode-owning thread. The background
    UI thread never accesses engine or store objects; preview callbacks execute
    only while that owner is waiting for the corresponding command.
    """

    def __init__(
        self,
        *,
        input_device=None,
        output_device=None,
        theme=DEFAULT_LIVE_THEME,
        warm_debounce_mode: str | None = None,
        fixed_delay: float = DEFAULT_WARM_SELECTION_DELAY,
        adaptive_min_delay: float = 0.08,
        adaptive_max_delay: float = 0.25,
        adaptive_burst_threshold: float = 0.18,
    ):
        self.input_device = input_device
        self.output_device = output_device
        self.theme = theme
        self.application: Application | None = None
        self.choice_view: LiveChoiceView | None = None
        self.edge_view: LiveEdgeView | None = None
        self._prompt_view = None
        self._surface = None
        self._current: _Request | None = None
        self._view_ready: _Request | None = None
        self._events: Queue = Queue()
        self._ready: Future = Future()
        self._thread: threading.Thread | None = None
        self._loop = None
        self._owner = None
        self._failure: BaseException | None = None
        self._interrupted = threading.Event()
        self._closing = False
        self._notice = ""
        self._warm_generation = 0

        # Debounce experiment controls (internal default is adaptive)
        resolved_mode = warm_debounce_mode
        if resolved_mode is None:
            resolved_mode = os.getenv("SPE_TEST_WARM_DEBOUNCE_MODE", "adaptive")

        self.warm_debounce_mode = resolved_mode.lower()
        self.fixed_delay = fixed_delay
        self.min_delay = adaptive_min_delay
        self.max_delay = adaptive_max_delay
        self.burst_threshold = adaptive_burst_threshold

        # Timing and telemetry state
        self._last_target_time: float | None = None
        self.stats = {
            "warm_dispatches": 0,
            "warm_aborts": 0,
            "promotions": 0,
        }

    @property
    def accepting_input(self) -> bool:
        return (
            self._current is not None
            and self._view_ready is self._current
            and not self._current.response.done()
        )

    def __enter__(self):
        if self._thread is not None:
            raise RuntimeError("terminal session cannot be entered twice")
        self._owner = threading.get_ident()
        self._thread = threading.Thread(target=self._run, name="spe-terminal")
        self._thread.start()
        try:
            self._ready.result()
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._closing = True
        if self._loop is not None and self._thread.is_alive():
            self._call(self._stop)
        if self._thread is not None:
            self._thread.join()
        if exc_type is None and self._failure is not None:
            raise self._failure
        return False

    def _call(self, callback, *args):
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(callback, *args)

    def _stop(self):
        if self.application.is_running and not self.application.is_done:
            self.application.exit()

    def _run(self):
        try:
            with create_app_session(
                input=self.input_device, output=self.output_device
            ) as session:
                self.input_device, self.output_device = session.input, session.output
                asyncio.run(self._run_application())
        except BaseException as exc:
            self._failure = exc
        finally:
            error = self._failure or EOFError("terminal input closed")
            if not self._ready.done():
                self._ready.set_exception(error)
            if self._current is not None and not self._current.response.done():
                self._current.response.set_exception(error)
            self._events.put(None)

    async def _run_application(self):
        self._loop = asyncio.get_running_loop()
        if self._closing:
            return
        self._loop.set_exception_handler(self._ui_exception)
        global_keys = KeyBindings()

        @global_keys.add("c-c", eager=True)
        @global_keys.add(Keys.SIGINT, eager=True)
        def interrupt(event):
            self._interrupt()

        busy_keys = KeyBindings()
        for key in Keys:
            busy_keys.add(key, eager=True)(lambda event: None)
        waiting = Window(FormattedTextControl("Preparing editor…"))
        root = HSplit(
            [
                ConditionalContainer(
                    Window(
                        FormattedTextControl(lambda: self._notice),
                        height=1,
                        style="class:hint",
                    ),
                    Condition(lambda: bool(self._notice)),
                ),
                DynamicContainer(
                    lambda: self._surface.layout.container if self._surface else waiting
                ),
            ]
        )
        active_keys = merge_key_bindings(
            [
                DynamicKeyBindings(
                    lambda: (
                        self._surface.bindings if self.accepting_input else busy_keys
                    )
                ),
                global_keys,
            ]
        )
        self.application = Application(
            layout=Layout(root),
            key_bindings=active_keys,
            style=_live_style(self.theme),
            full_screen=True,
            erase_when_done=True,
            input=self.input_device,
            output=self.output_device,
            before_render=self._before_render,
            after_render=self._rendered,
        )
        self.application.key_processor.after_key_press += self._after_key_press
        await self.application.run_async(
            set_exception_handler=False, handle_sigint=False
        )

    def _rendered(self, app):
        if not self._ready.done():
            self._ready.set_result(None)
        request = self._current
        if request is not None:
            self._refresh_warm_target(request)
        if self._closing:
            self._stop()

    def _after_key_press(self, key_processor):
        del key_processor
        request = self._current
        if request is not None:
            self._refresh_warm_target(request)

    @staticmethod
    def _is_warm_promotion(request: _Request) -> bool:
        if (
            request.warm_cancelled.is_set()
            or not request.response.done()
            or request.response.cancelled()
            or request.warm_target is None
            or request.submitted_target != request.warm_target
            or request.warm_future is None
            or not request.warm_future.done()
            or request.warm_future.cancelled()
        ):
            return False
        try:
            request.response.result()
            return bool(request.warm_future.result())
        except BaseException:
            return False

    def _choice_target(self, request: _Request, raw: str) -> tuple[int, int] | None:
        state = self.choice_view.state
        if request.state.review is not None:
            return None
        preview = action_preview(
            state.choice,
            raw,
            state.candidates,
            state.resolve_insertion,
            remaining_tokens=state.remaining_tokens,
            resolve_candidate=state.resolve_candidate,
            default_hold_tokens=state.default_hold_tokens,
            default_search_radius=state.default_search_radius,
        )
        if (
            not preview.valid
            or preview.state != "ready"
            or preview.kind != "candidate"
            or preview.candidate_rank is None
            or preview.token_id is None
        ):
            return None
        return preview.candidate_rank, preview.token_id

    def _refresh_warm_target(self, request: _Request) -> None:
        if (
            request is not self._current
            or request.response.done()
            or not isinstance(request.state, ChoiceViewState)
            or request.state.warm_selection is None
            or self.choice_view is None
        ):
            return
        target = self._choice_target(request, self.choice_view.command_buffer.text)
        if target == request.warm_target:
            return

        # Measure inter-arrival time across cursor shifts
        now = self._loop.time()
        delta = (
            now - self._last_target_time
            if self._last_target_time is not None
            else float("inf")
        )
        self._last_target_time = now

        # Cancel any pending timer or in-flight compute
        if request.warm_timer is not None or (
            request.warm_future and not request.warm_future.done()
        ):
            self.stats["warm_aborts"] += 1

        request.warm_cancelled.set()
        if request.warm_timer is not None:
            request.warm_timer.cancel()
            request.warm_timer = None
        if request.warm_future is not None and not request.warm_future.done():
            request.warm_future.cancel()
        if request.warm_target is not None:
            cancel = request.state.cancel_warm_selection
            if cancel is not None:
                self._events.put(_Resolution(request, cancel, Future()))

        self._warm_generation += 1
        request.warm_generation = self._warm_generation
        request.warm_target = target
        request.warm_cancelled = threading.Event()
        request.warm_future = None

        if target is not None:
            if self.warm_debounce_mode == "adaptive":
                if delta < self.burst_threshold:
                    speed_ratio = 1.0 - (max(0.0, delta) / self.burst_threshold)
                    delay = (
                        self.min_delay + (self.max_delay - self.min_delay) * speed_ratio
                    )
                else:
                    delay = self.min_delay
            else:
                delay = self.fixed_delay

            request.warm_timer = self._loop.call_later(
                delay,
                self._queue_warm_selection,
                request,
                request.warm_generation,
                target,
                request.warm_cancelled,
            )

    def _queue_warm_selection(
        self,
        request: _Request,
        generation: int,
        target: tuple[int, int],
        cancelled: threading.Event,
    ) -> None:
        request.warm_timer = None
        if (
            request is not self._current
            or request.response.done()
            or request.warm_generation != generation
            or request.warm_target != target
            or cancelled.is_set()
        ):
            return
        future = Future()
        request.warm_future = future
        self.stats["warm_dispatches"] += 1

        def warm():
            if cancelled.is_set():
                return None
            return request.state.warm_selection(
                target[0], target[1], generation, cancelled.is_set
            )

        self._events.put(_Resolution(request, warm, future))

    def _before_render(self, app):
        if self._current is not None and not self._current.response.done():
            self._view_ready = self._current

    def _ui_exception(self, loop, context):
        error = context.get("exception") or RuntimeError(
            context.get("message", "terminal error")
        )
        self._failure = error
        if self.application.is_running and not self.application.is_done:
            self.application.exit(exception=error)

    def _interrupt(self):
        if self._interrupted.is_set():
            return
        self._interrupted.set()
        if self._owner == threading.main_thread().ident:
            interrupt_main()
        self._events.put(None)

    def _read(self, state):
        if threading.get_ident() != self._owner:
            raise RuntimeError(
                "terminal requests must come from the episode-owning thread"
            )
        if self._closing or not self._thread.is_alive():
            raise self._failure or EOFError("terminal session is closed")
        request = _Request(state)
        self._call(self._show, request)
        try:
            while True:
                if self._interrupted.is_set():
                    raise KeyboardInterrupt
                if self._failure is not None:
                    raise self._failure
                if request.response.done():
                    return request.response.result()
                event = self._events.get()
                if isinstance(event, _Resolution):
                    if (
                        event.request is not request
                        or request.response.done()
                        or not event.result.set_running_or_notify_cancel()
                    ):
                        continue
                    try:
                        value = event.callback()
                    except EditorError as exc:
                        if not event.result.cancelled():
                            event.result.set_exception(exc)
                    except Exception as exc:
                        if not event.result.cancelled():
                            event.result.set_exception(exc)
                        self._failure = exc
                        self._call(self._stop)
                        raise
                    else:
                        if not event.result.cancelled():
                            event.result.set_result(value)
                    if not request.response.done():
                        self.application.invalidate()
                elif not self._thread.is_alive() and not request.response.done():
                    raise self._failure or EOFError("terminal input closed")
        finally:
            if not request.response.done():
                request.response.cancel()
            keep_prepared = self._is_warm_promotion(request)
            if keep_prepared:
                self.stats["promotions"] += 1
            if request.warm_timer is not None:
                self._call(request.warm_timer.cancel)
            if not keep_prepared:
                request.warm_cancelled.set()
                if request.warm_future is not None and not request.warm_future.done():
                    request.warm_future.cancel()
                if isinstance(request.state, ChoiceViewState):
                    cancel_warm = request.state.cancel_warm_selection
                    if callable(cancel_warm):
                        cancel_warm()
            for preview in request.previews.values():
                if not preview.done():
                    preview.cancel()

    def read_choice(self, state: ChoiceViewState):
        return self._read(state)

    def read_edge(self, state: EdgeViewState):
        return self._read(state)

    def prompt(self, request: PromptRequest):
        return self._read(request)

    def read(self, prompt: str, *, single_key=False):
        return self.prompt(PromptRequest(prompt, single_key=single_key))

    def read_multiline_prompt(
        self,
        prompt: str = "Write at least one character. Press Escape then Enter to continue.\n\n",
    ):
        return self.prompt(PromptRequest(prompt, multiline=True))

    def page(self, text: str):
        return self.prompt(PromptRequest("", body=text, page=True))

    def write(self, text: str, *, end="\n"):
        self._call(self._write, text + end)

    def _write(self, text):
        lines = _safe_context_text(text).splitlines()
        if lines:
            self._notice = next(
                (line for line in reversed(lines) if line), self._notice
            )
        self.application.invalidate()

    def _show(self, request):
        if self._closing:
            return
        self._view_ready = None
        self._current = request
        state = request.state
        if isinstance(state, ChoiceViewState):
            resolver = state.resolve_candidate
            state = replace(
                state,
                resolve_insertion=lambda text, mode: self._preview(
                    request,
                    ("insertion", text, mode),
                    lambda: request.state.resolve_insertion(text, mode),
                ),
                resolve_candidate=(
                    lambda rank: self._preview(
                        request, ("candidate", rank), lambda: resolver(rank)
                    )
                )
                if resolver
                else None,
            )
            if self.choice_view is None:
                self.choice_view = LiveChoiceView(
                    state,
                    submit=self._submit,
                    enabled=lambda: self.accepting_input,
                    terminal_size=self._surface_size,
                )
            else:
                self.choice_view.update(state)
            self._surface = self.choice_view
        elif isinstance(state, EdgeViewState):
            if self.edge_view is None:
                self.edge_view = LiveEdgeView(
                    state, submit=self._submit, enabled=lambda: self.accepting_input
                )
            else:
                self.edge_view.update(state)
            self._surface = self.edge_view
        else:
            if state.isolated:
                self._notice = ""
            if self._prompt_view is None:
                self._prompt_view = _PromptView(
                    self._submit, lambda: self.accepting_input
                )
            self._prompt_view.update(state)
            self._surface = self._prompt_view
        self.application.layout.focus(self._surface.layout.current_control)
        self.application.invalidate()

    def _surface_size(self):
        size = self.output_device.get_size()
        return size.columns, max(1, size.rows - bool(self._notice))

    def terminal_size(self) -> tuple[int, int]:
        return self._surface_size()

    def _submit(self, *, result=None, exception=None):
        if isinstance(exception, KeyboardInterrupt):
            self._interrupt()
            return
        if not self.accepting_input:
            return
        request = self._current
        if (
            exception is None
            and isinstance(request.state, ChoiceViewState)
            and request.state.warm_selection is not None
            and isinstance(result, str)
        ):
            request.submitted_target = self._choice_target(request, result)
        if not request.response.set_running_or_notify_cancel():
            return
        if exception is None:
            request.response.set_result(result)
        else:
            request.response.set_exception(exception)
        self._notice = ""
        self._events.put(None)

    def _preview(self, request, key, callback):
        cached = request.previews.get(key)
        if cached is not None and not cached.cancelled() and cached.done():
            result = cached.result()
            if key[0] == "insertion":
                request.insertion_display[key[2]] = result
            return result
        if not request.response.done() and (cached is None or cached.cancelled()):
            previous = request.latest.get(key[0])
            if previous is not None:
                old = request.previews.get(previous)
                if old is not None and not old.done():
                    old.cancel()
            cached = Future()
            request.previews[key] = cached
            request.latest[key[0]] = key
            while len(request.previews) > 128:
                _, expired = request.previews.popitem(last=False)
                expired.cancel()
            self._events.put(_Resolution(request, callback, cached))
        if key[0] == "candidate":
            return None
        raise PreviewPending(request.insertion_display.get(key[2]))

"""One terminal application, with synchronous episode work on its owning thread.

The UI thread owns widgets and terminal I/O. The caller owns the engine, its
backend and SQLite connection. Requests and preview futures are the only bridge;
no renderer or key handler calls the backend. A submitted view immediately stops
accepting input, so queued keystrokes cannot commit against an obsolete decision.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from _thread import interrupt_main
from collections import OrderedDict, deque
from concurrent.futures import Future
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from queue import Queue
from typing import Callable, Any

from prompt_toolkit.application import Application, create_app_session
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings, DynamicKeyBindings, merge_key_bindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window, DynamicContainer, ConditionalContainer
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.widgets import TextArea

from .edge_tui import EdgeViewState, LiveEdgeView
from .live_tui import ChoiceViewState, LiveChoiceView, PreviewPending, _live_style, _safe_context_text
from .ui_themes import DEFAULT_LIVE_THEME


@dataclass(frozen=True)
class PromptState:
    prompt: str
    body: str = ""
    single_key: bool = False
    page: bool = False


@dataclass(eq=False)
class _Request:
    state: ChoiceViewState | EdgeViewState | PromptState
    response: Future = field(default_factory=Future)
    previews: OrderedDict = field(default_factory=OrderedDict)
    latest: dict[str, tuple] = field(default_factory=dict)


@dataclass
class _Resolution:
    request: _Request
    callback: Callable[[], Any]
    result: Future


class _PromptView:
    """In-application prompts, confirmations and scrollable documents."""

    def __init__(self, submit, enabled):
        self.state = PromptState("")
        self.submit = submit
        self.command_buffer = Buffer(read_only=Condition(lambda: not enabled()))
        self.body = TextArea(read_only=True, scrollbar=True, wrap_lines=True)
        self.input_control = BufferControl(buffer=self.command_buffer)
        self.layout = Layout(HSplit([
            self.body,
            ConditionalContainer(Window(FormattedTextControl(lambda: self.state.prompt),
                                        dont_extend_height=True),
                                 Condition(lambda: not self.state.page)),
            ConditionalContainer(Window(self.input_control, height=1, style="class:input"),
                                 Condition(lambda: not self.state.page)),
            Window(FormattedTextControl(lambda: (
                "↑/↓ · PgUp/PgDn scroll · Enter/Esc returns" if self.state.page else
                "Press a key" if self.state.single_key else "Enter submits · Ctrl-D cancels"
            )), height=1, style="class:hint"),
        ]))
        self.bindings = KeyBindings()

        @self.bindings.add("enter")
        def enter(event):
            self.submit(result="" if self.state.page else "\n" if self.state.single_key
                        else self.command_buffer.text)

        @self.bindings.add("escape", eager=True)
        def escape(event):
            self.submit(result="\x1b" if self.state.single_key else "" if self.state.page else None)

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

    def update(self, state: PromptState) -> None:
        self.state = state
        self.command_buffer.reset()
        self.body.buffer.set_document(Document(_safe_context_text(state.body)), bypass_readonly=True)
        self.layout.focus(self.body if state.page else self.input_control)


class PersistentTerminalSession(AbstractContextManager):
    """Keep one Application and renderer alive across every interactive surface.

    Enter and call read methods from the episode-owning thread. The background
    UI thread never accesses engine or store objects; preview callbacks execute
    only while that owner is waiting for the corresponding command.
    """

    def __init__(self, *, input_device=None, output_device=None, theme=DEFAULT_LIVE_THEME):
        self.input_device = input_device
        self.output_device = output_device
        self.theme = theme
        self.application: Application | None = None
        self.choice_view: LiveChoiceView | None = None
        self.edge_view: LiveEdgeView | None = None
        self._prompt_view = None
        self._surface = None
        self._current: _Request | None = None
        self._events: Queue = Queue()
        self._ready: Future = Future()
        self._thread: threading.Thread | None = None
        self._loop = None
        self._owner = None
        self._failure: BaseException | None = None
        self._interrupted = threading.Event()
        self._closing = False
        self._notice = ""
        self._messages = deque(maxlen=80)
        self._prompt_context = ""
        self._busy_timer = None

    @property
    def accepting_input(self) -> bool:
        return self._current is not None and not self._current.response.done()

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
            with create_app_session(input=self.input_device, output=self.output_device) as session:
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
        # Override special/default bindings too (including Ctrl-L's forced
        # repaint). Ordinary typeahead is consumed until a fresh view is ready.
        for key in Keys:
            busy_keys.add(key, eager=True)(lambda event: None)
        waiting = Window(FormattedTextControl("Preparing editor…"))
        root = HSplit([
            ConditionalContainer(Window(FormattedTextControl(lambda: self._notice),
                                        height=1, style="class:hint"),
                                 Condition(lambda: bool(self._notice))),
            DynamicContainer(lambda: self._surface.layout.container if self._surface else waiting),
        ])
        active_keys = merge_key_bindings([
            DynamicKeyBindings(lambda: self._surface.bindings if self.accepting_input else busy_keys),
            global_keys,
        ])
        self.application = Application(
            layout=Layout(root), key_bindings=active_keys, style=_live_style(self.theme),
            full_screen=True, erase_when_done=True,
            input=self.input_device, output=self.output_device,
            after_render=self._rendered,
        )
        await self.application.run_async(set_exception_handler=False, handle_sigint=False)

    def _rendered(self, app):
        if not self._ready.done():
            self._ready.set_result(None)
        if self._closing:
            self._stop()

    def _ui_exception(self, loop, context):
        error = context.get("exception") or RuntimeError(context.get("message", "terminal error"))
        self._failure = error
        if self.application.is_running and not self.application.is_done:
            self.application.exit(exception=error)

    def _interrupt(self):
        if self._interrupted.is_set():
            return
        self._interrupted.set()
        # Signal before waking a blocked read: the caller must not miss an
        # interrupt in the small handoff between returning a command and apply().
        if self._owner == threading.main_thread().ident:
            interrupt_main()
        self._events.put(None)

    def _read(self, state):
        if threading.get_ident() != self._owner:
            raise RuntimeError("terminal requests must come from the episode-owning thread")
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
                    if (event.request is not request or request.response.done()
                            or not event.result.set_running_or_notify_cancel()):
                        continue
                    try:
                        value = event.callback()
                    except Exception as exc:
                        if not event.result.cancelled():
                            event.result.set_exception(exc)
                    else:
                        if not event.result.cancelled():
                            event.result.set_result(value)
                    if not request.response.done():
                        self.application.invalidate()
                elif not self._thread.is_alive() and not request.response.done():
                    raise self._failure or EOFError("terminal input closed")
        finally:
            # Prevent any late paint from scheduling callbacks while the
            # caller is applying an action or repositioning the backend.
            if not request.response.done():
                request.response.cancel()

    def read_choice(self, state: ChoiceViewState):
        return self._read(state)

    def read_edge(self, state: EdgeViewState):
        return self._read(state)

    def read(self, prompt: str, *, single_key=False):
        return self._read(PromptState(prompt, single_key=single_key))

    def page(self, text: str):
        return self._read(PromptState("", body=text, page=True))

    def write(self, text: str, *, end="\n"):
        self._call(self._write, text + end)

    def _write(self, text):
        lines = _safe_context_text(text).splitlines()
        self._messages.extend(line for line in lines if line)
        if lines:
            self._notice = next((line for line in reversed(lines) if line), self._notice)
        self.application.invalidate()

    def _show(self, request):
        if self._closing:
            return
        if self._busy_timer is not None:
            self._busy_timer.cancel()
        self._current = request
        if self._notice == "Working…":
            self._notice = ""
        state = request.state
        if isinstance(state, ChoiceViewState):
            self._prompt_context = ""
            resolver = state.resolve_candidate
            state = replace(state,
                resolve_insertion=lambda text, mode: self._preview(
                    request, ("insertion", text, mode),
                    lambda: request.state.resolve_insertion(text, mode)),
                resolve_candidate=(lambda rank: self._preview(
                    request, ("candidate", rank), lambda: resolver(rank))) if resolver else None,
            )
            if self.choice_view is None:
                self.choice_view = LiveChoiceView(state, submit=self._submit,
                                                  enabled=lambda: self.accepting_input,
                                                  terminal_size=self._surface_size)
            else:
                self.choice_view.update(state)
            self._surface = self.choice_view
        elif isinstance(state, EdgeViewState):
            self._prompt_context = ""
            if self.edge_view is None:
                self.edge_view = LiveEdgeView(state, submit=self._submit,
                                              enabled=lambda: self.accepting_input)
            else:
                self.edge_view.update(state)
            self._surface = self.edge_view
        else:
            if state.page:
                self._prompt_context = state.body
            else:
                state = replace(state, body=self._prompt_context or "\n".join(self._messages))
            if self._prompt_view is None:
                self._prompt_view = _PromptView(self._submit, lambda: self.accepting_input)
            self._prompt_view.update(state)
            self._surface = self._prompt_view
        self.application.layout.focus(self._surface.layout.current_control)
        self.application.invalidate()

    def _surface_size(self):
        size = self.output_device.get_size()
        return size.columns, max(1, size.rows - bool(self._notice))

    def _submit(self, *, result=None, exception=None):
        if isinstance(exception, KeyboardInterrupt):
            self._interrupt()
            return
        if not self.accepting_input:
            return
        request = self._current
        if not request.response.set_running_or_notify_cancel():
            return
        if exception is None:
            request.response.set_result(result)
        else:
            request.response.set_exception(exception)
        self._notice = ""
        self._events.put(None)
        self._busy_timer = self._loop.call_later(0.15, self._show_busy, request)

    def _show_busy(self, request):
        if (self._current is request and not self.accepting_input
                and not self._closing and not self._notice):
            self._notice = "Working…"
            self.application.invalidate()

    def _preview(self, request, key, callback):
        # UI-only cache containing thread-safe futures. A newer preview of the
        # same kind cancels obsolete queued work, and each view has its own cache.
        cached = request.previews.get(key)
        if cached is not None and not cached.cancelled() and cached.done():
            return cached.result()
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
        raise PreviewPending

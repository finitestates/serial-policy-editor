"""Textual screens and the synchronous engine-to-UI thread bridge."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from queue import Empty, Queue
from typing import Any, ClassVar

from rich.markup import escape as escape_markup
from rich.text import Text
from textual import constants, events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import DiscoveryHit, Hit, Provider
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen, Screen
from textual.theme import Theme
from textual.widgets import DataTable, Input, RichLog, Static, TextArea

if os.name == "posix":
    import termios
    import tty

    from textual.driver import Driver
    from textual.drivers._writer_thread import WriterThread
    from textual.drivers.linux_driver import (
        KITTY_DISAMBIGUATE_ESCAPE_CODES,
        KITTY_REPORT_ALL_KEYS,
        KITTY_REPORT_ASSOCIATED_TEXT,
        LinuxDriver,
    )
    from textual.geometry import Size

from .candidate_columns import CandidateColumns
from .core.candidates import Candidate
from .core.errors import EditorError
from .edge_help import edge_help
from .teacher_commands import HELP_TEXT, CommandKind, interpret_command
from .terminal_contracts import (
    SEAMLESS_REACTIVATE,
    BeamInput,
    BeamViewState,
    ChoiceFeedback,
    ChoiceViewState,
    EdgeViewState,
    PromptRequest,
)
from .tui_render import (
    ActionPreview,
    PreviewPending,
    _candidate_preview,
    _is_writing,
    _navigation_command_cycle,
    _preview_fragments,
    _render_review,
    _safe_context_text,
    action_preview,
    candidate_table_plan,
    candidate_table_row,
)
from .ui_themes import (
    DEFAULT_LIVE_THEME,
    semantic_style,
    textual_theme_values,
    theme_palette,
    theme_stylesheet,
)

_LOG = logging.getLogger(__name__)


if os.name == "posix":

    class _ThreadedLinuxDriver(LinuxDriver):
        """Linux terminal driver whose process signals are owned by the caller thread."""

        def __init__(
            self,
            app: App[Any],
            *,
            debug: bool = False,
            mouse: bool = True,
            size: tuple[int, int] | None = None,
        ) -> None:
            Driver.__init__(self, app, debug=debug, mouse=mouse, size=size)
            self._file = getattr(app, "terminal_output", None) or sys.__stderr__
            self._metrics_app = app
            self.fileno = sys.__stdin__.fileno()
            self.input_tty = sys.__stdin__.isatty()
            self.attrs_before: list[Any] | None = None
            self.exit_event = threading.Event()
            self._key_thread: threading.Thread | None = None
            self._writer_thread: WriterThread | None = None
            self._must_signal_resume = False
            self._in_band_window_resize = False
            self._mouse_pixels = False
            self._count_driver_writes = False

        def write(self, data: str) -> None:
            """Count driver writes after terminal startup is complete."""
            if self._count_driver_writes:
                self._metrics_app.stats["driver_write_calls"] += 1
                self._metrics_app.stats["driver_write_characters"] += len(data)
            super().write(data)

        def _get_terminal_size(self) -> tuple[int, int]:
            """Read the TTY geometry, ignoring potentially stale LINES/COLUMNS."""
            try:
                size = os.get_terminal_size(self.fileno)
            except OSError:
                return self._size or super()._get_terminal_size()
            return size.columns, size.lines

        def _send_terminal_resize(self) -> None:
            if self._in_band_window_resize:
                return
            width, height = self._get_terminal_size()
            size = Size(width, height)
            self.send_message(events.Resize(size, size))

        def start_application_mode(self) -> None:
            """Start terminal mode without registering signals on the UI thread."""
            if os.isatty(self.fileno):
                try:
                    termios.tcsetattr(
                        self.fileno, termios.TCSANOW, termios.tcgetattr(self.fileno)
                    )
                except termios.error:
                    return

            loop = asyncio.get_running_loop()
            self._writer_thread = WriterThread(self._file)
            self._writer_thread.start()
            self._send_terminal_resize()

            self.write("\x1b[?1049h")
            self._enable_mouse_support()
            try:
                self.attrs_before = termios.tcgetattr(self.fileno)
            except termios.error:
                self.attrs_before = None

            try:
                newattr = termios.tcgetattr(self.fileno)
            except termios.error:
                pass
            else:
                newattr[tty.LFLAG] = self._patch_lflag(newattr[tty.LFLAG])
                newattr[tty.IFLAG] = self._patch_iflag(newattr[tty.IFLAG])
                newattr[tty.CC][termios.VMIN] = 1
                try:
                    termios.tcsetattr(self.fileno, termios.TCSANOW, newattr)
                except termios.error:
                    pass

            self.write("\x1b[?25l")
            self.write("\x1b[?1004h")
            if not constants.DISABLE_KITTY_KEY:
                kitty_protocol_flag = (
                    KITTY_DISAMBIGUATE_ESCAPE_CODES
                    | KITTY_REPORT_ALL_KEYS
                    | KITTY_REPORT_ASSOCIATED_TEXT
                )
                self.write(f"\x1b[>{kitty_protocol_flag}u")
            self.flush()
            self._key_thread = threading.Thread(
                target=self._run_input_thread, name="textual-input"
            )
            self._key_thread.start()
            self._request_terminal_sync_mode_support()
            self._query_in_band_window_resize()
            self._enable_bracketed_paste()
            self._disable_line_wrap()
            self._enable_mouse_support()

            if self._must_signal_resume:
                self._must_signal_resume = False
                asyncio.run_coroutine_threadsafe(
                    self._app._post_message(self.SignalResume()), loop=loop
                )
            self._count_driver_writes = True

        def stop_application_mode(self) -> None:
            self._count_driver_writes = False
            super().stop_application_mode()

        def disable_input(self) -> None:
            """Stop the input reader without changing process signal handlers."""
            try:
                if not self.exit_event.is_set():
                    self._disable_mouse_support()
                    self.exit_event.set()
                    if self._key_thread is not None:
                        self._key_thread.join()
                    self.exit_event.clear()
                    try:
                        termios.tcflush(self.fileno, termios.TCIFLUSH)
                    except termios.error:
                        pass
            except (OSError, RuntimeError):
                _LOG.exception("could not stop Textual terminal input")


@dataclass(frozen=True)
class _TerminalResponse:
    value: Any = None
    exception: BaseException | None = None


def _stretch_data_table_column(
    table: DataTable, key: str
) -> None:
    """Fit one flexible DataTable column using the pinned Textual 8.2.8 internals.

    DataTable 8.2.8 has no public API for a column that consumes the remaining
    viewport width. Keep this version-sensitive operation in this adapter and
    pin the supported Textual version in core/pyproject.toml.
    """
    column = table.columns[key]
    available = table.scrollable_content_region.width
    fixed = sum(
        item.get_render_width(table)
        for item in table.columns.values()
        if item is not column
    )
    fixed += table._row_label_column_width
    width = max(1, available - fixed - 2 * table.cell_padding)
    if column.width == width and not column.auto_width:
        return
    column.width = width
    column.auto_width = False
    # Auto-height rows (Beam continuations) need remeasurement after the fill
    # column changes width, or cached no-wrap cells keep their old one-line height.
    for row in table.rows.values():
        if row.auto_height:
            row.height = 0
    table._row_renderable_cache.clear()
    table._cell_render_cache.clear()
    table._new_rows.update(table.rows)
    table._require_update_dimensions = True
    table.refresh(layout=True)
    app = table.app
    if isinstance(app, PolicyEditorApp):
        app.stats["table_layout_refresh_requests"] += 1


class _FluidDataTable(DataTable):
    """DataTable adapter for one flexible column on Textual 8.2.8."""

    def __init__(self, *args: Any, stretch_column: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stretch_column = stretch_column
        self._fit_after_resize_scheduled = False

    def on_resize(self, _event: events.Resize) -> None:
        # Resize handlers run before all child regions settle. Coalesce resize
        # bursts and fit once against the final viewport reported after refresh.
        if self._fit_after_resize_scheduled:
            return
        self._fit_after_resize_scheduled = True
        self.call_after_refresh(self._fit_after_resize)

    def _fit_after_resize(self) -> None:
        self._fit_after_resize_scheduled = False
        self.fit_column()

    def fit_column(self) -> None:
        if (
            self._stretch_column in self.columns
            and self.scrollable_content_region.width > 0
        ):
            column = self.columns[self._stretch_column]
            previous_width = column.width
            _stretch_data_table_column(self, self._stretch_column)
            if isinstance(self.app, PolicyEditorApp):
                self.app.stats["table_fit_attempts"] += 1
                if column.width != previous_width:
                    self.app.stats["table_column_resizes"] += 1


class _CommandTemplateDataTable(_FluidDataTable):
    """Stage row commands without taking focus away from their editor."""

    def focus_on_click(self) -> bool:
        return False


@dataclass(frozen=True)
class _OwnerPreview:
    generation: int
    key: tuple[Any, ...]
    callback: Callable[[], Any]


@dataclass
class _RequestLifecycle:
    generation: int
    state: ChoiceViewState | EdgeViewState | BeamViewState | PromptRequest
    response: Future[Any]
    owner_queue: Queue[_OwnerPreview] | None
    warm_cancelled: threading.Event = field(default_factory=threading.Event)
    warm_future: Future[Any] | None = None
    warm_result: bool | None = None
    submitted_result: Any = None
    submitted_exception: BaseException | None = None
    submitted_target: tuple[int, int] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass(frozen=True)
class _PaletteEntry:
    title: str
    insert: str
    help: str


def _teacher_palette_entries() -> tuple[_PaletteEntry, ...]:
    """Derive searchable command examples from the shared teacher help text."""
    replacements = {
        "RANK+ / RANK-": ("1+", "Adjust the selected token's bias."),
        "RANK=VALUE": ("1=0", "Set or clear a direct token adjustment."),
        "1..N": ("1", "Commit a candidate by raw rank."),
        "s top_k=20|none": ("s top_k=20", "Change sampler settings."),
        "t TEXT": ("t ", "Insert continuation text."),
        "x TEXT": ("x ", "Insert exact text."),
        "check TEXT": ("check ", "Commit a checked continuation phrase."),
        "checkx TEXT": ("checkx ", "Commit a checked exact phrase."),
        "force TEXT": ("force ", "Force a continuation phrase."),
        "forcex TEXT": ("forcex ", "Force exact text."),
        "h [N]": ("h", "Release control for the requested token count."),
        "f N": ("f ", "Fork at an absolute boundary."),
        "f - N": ("f - ", "Fork relative to this boundary."),
        "n [TEXT]": ("n ", "Add a note before the current decision."),
        "p [TEXT]": ("p ", "Add a note after the most recent update."),
        "e | eog": ("e", "Preview and confirm a teacher selected end token."),
        "e! | eog!": ("e!", "Commit a teacher selected end token."),
        "q | finish": ("q", "Open the live edge menu."),
        "[ / ]": ("[", "Review a token boundary."),
    }
    result: list[_PaletteEntry] = []
    for line in HELP_TEXT.splitlines():
        if not line.startswith("  "):
            continue
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        head = parts[0]
        if head in {"READY", "Tab", "Ctrl+G", "numeric", "terms", "multi-token", "after"}:
            continue
        if head not in {
            "accept", "groups", "b", "RANK+", "RANK=VALUE", "s", "reroll",
            "draw", "1..N", "chord", "beam", "gbeam", "t", "x", "check",
            "checkx", "force", "forcex", "h", "m", "/TERM", "ms", "c", "C",
            "overlay", "context", "v", "V", "l", "L", "%", "[", "f", "n",
            "p", "e", "e!", "q",
        }:
            continue
        title, _, description = stripped.partition("  ")
        if not description:
            # Help columns use spacing, but compact text can make the split one
            # space after strip. Keep a short, useful description in that case.
            description = parts[1] if len(parts) > 1 else title
        insert, help_text = replacements.get(title, (title, description.strip()))
        if title.startswith(("b NAME", "b {", "b token")):
            insert = "b "
        elif title.startswith(("beam", "gbeam")):
            insert = title.split()[0]
        elif title.startswith(("chord",)):
            insert = "chord "
        elif title.startswith(("reroll", "draw", "overlay", "context", "ms")):
            insert = title.split()[0] + " "
        elif title.startswith(("m N", "m ")):
            insert = "m "
        elif title.startswith("/TERM"):
            insert = "/"
        elif title.startswith("[ / ]"):
            insert = "["
        result.append(_PaletteEntry(title, insert, help_text.strip()))
    # Descriptive lines and aliases that share a parser prefix need their own
    # searchable entry even if the help table formats their first column tightly.
    known = {entry.title for entry in result}
    for title, insert, detail in (
        ("sampler settings", "s ", "Change one or more sampler settings."),
        ("token bias", "b ", "Change a group or token bias."),
        ("fork boundary", "f", "Fork at the current boundary."),
        ("help", "", "Show the full command list."),
    ):
        if title not in known:
            result.append(_PaletteEntry(title, insert, detail))
    return tuple(result)


_TEACHER_PALETTE_ENTRIES = _teacher_palette_entries()
_BEAM_PALETTE_ENTRIES = (
    _PaletteEntry("resume", "resume", "Continue from the beam edge."),
    _PaletteEntry("select <label>", "select ", "Commit the selected beam branch."),
    _PaletteEntry("advance N", "advance 1", "Advance one or more beam steps."),
    _PaletteEntry("rewind", "rewind", "Return one beam step."),
    _PaletteEntry("kill <label>", "kill ", "Remove a beam branch."),
    _PaletteEntry("protect", "protect", "Protect the selected deterministic lineage."),
    _PaletteEntry("families", "families", "Toggle branch family details."),
    _PaletteEntry("return", "return", "Return to the teacher decision."),
)
_HELP_PALETTE_ENTRY = _PaletteEntry("help", "", "Show the full command list.")


def _edge_insert_command(command: str) -> str:
    if command.startswith("s "):
        return "s "
    if command.startswith("reroll"):
        return "reroll "
    if command.startswith("f "):
        return "f "
    if command.startswith("#"):
        return "#"
    if command.startswith("new "):
        return "new "
    if command.startswith("name "):
        return "name "
    if command.startswith("rewind "):
        return "rewind "
    if command.startswith("export "):
        return "export "
    if command.startswith("save"):
        return "save "
    if command.startswith("spr"):
        return "spr "
    if command.startswith("switch "):
        return "switch "
    if " / " in command:
        return command.split(" / ", 1)[0]
    return command.split()[0]


def _palette_entries(state: Any) -> tuple[_PaletteEntry, ...]:
    if isinstance(state, EdgeViewState):
        commands = tuple(
            _PaletteEntry(item.command, _edge_insert_command(item.command), item.description)
            for item in edge_help(state.mode)
        )
        return (*commands, _HELP_PALETTE_ENTRY)
    if isinstance(state, BeamViewState):
        return (*_BEAM_PALETTE_ENTRIES, _HELP_PALETTE_ENTRY)
    if isinstance(state, ChoiceViewState):
        return _TEACHER_PALETTE_ENTRIES
    return ()


def _help_document(state: Any) -> Text:
    document = Text(HELP_TEXT)
    document.append("\nTerminal output\n", style="bold underline")
    document.append("  Ctrl+L                   Open captured output\n")
    if isinstance(state, EdgeViewState):
        heading = f"\n{state.mode.title()} edge commands\n"
        document.append(heading, style="bold underline")
        for item in edge_help(state.mode):
            document.append(f"  {item.command:<24} {item.description}\n")
    elif isinstance(state, BeamViewState):
        document.append("\nBeam commands\n", style="bold underline")
        for item in _BEAM_PALETTE_ENTRIES:
            document.append(f"  {item.title:<24} {item.help}\n")
    return document


class _CommandProvider(Provider):
    """Fuzzy-search teacher, edge, and beam commands for the active surface."""

    @staticmethod
    def _activate(screen: _RequestScreen, entry: _PaletteEntry) -> None:
        if screen.terminal_app._active_screen is not screen or not screen.accepting_input:
            return
        if entry.title == "help":
            screen.action_show_help()
        else:
            screen.insert_command(entry.insert)

    async def discover(self):
        screen = self.screen
        if not isinstance(screen, _RequestScreen):
            return
        for entry in _palette_entries(screen.lifecycle.state):
            yield DiscoveryHit(
                Text(entry.title),
                lambda item=entry: self._activate(screen, item),
                text=entry.title,
                help=escape_markup(entry.help),
            )

    async def search(self, query: str):
        screen = self.screen
        if not isinstance(screen, _RequestScreen):
            return
        matcher = self.matcher(query)
        for entry in _palette_entries(screen.lifecycle.state):
            score = matcher.match(entry.title)
            if score <= 0:
                continue
            yield Hit(
                score,
                matcher.highlight(escape_markup(entry.title)),
                lambda item=entry: self._activate(screen, item),
                text=entry.title,
                help=escape_markup(entry.help),
            )


class PolicyEditorApp(App[None]):
    """One Textual application running independently of the synchronous engine."""

    TITLE = "Policy Editor"
    COMMAND_PALETTE_BINDING = "ctrl+k"
    COMMANDS = App.COMMANDS | {_CommandProvider}
    CSS = ""
    OUTPUT_HISTORY_LIMIT = 16_000

    def __init__(
        self,
        *,
        theme: str = DEFAULT_LIVE_THEME,
        environment: dict[str, str] | None = None,
        ready: Future[None] | None = None,
        terminal_output: Any = None,
    ) -> None:
        driver_class = _ThreadedLinuxDriver if os.name == "posix" else None
        super().__init__(driver_class=driver_class)
        self.terminal_output = terminal_output
        self.theme_name = theme
        self.environment = dict(os.environ if environment is None else environment)
        self.palette = theme_palette(theme, environment=self.environment)
        self.CSS = theme_stylesheet(theme, environment=self.environment)
        self.register_theme(Theme(**textual_theme_values(self.palette)))
        self.theme = theme
        self._ready_future = ready
        self._active_request: _RequestLifecycle | None = None
        self._active_screen: _RequestScreen | None = None
        self._input_event_cutoff: float | None = None
        self._output_chunks: deque[str] = deque()
        self._output_history_chars = 0
        self._warm_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="spe-search-warm")
        self._preview_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="spe-preview")
        self.stats = {
            "warm_dispatches": 0,
            "promotions": 0,
            "stale_input_events": 0,
            "stale_key_events": 0,
            "stale_mouse_events": 0,
            "driver_write_calls": 0,
            "driver_write_characters": 0,
            "screen_layout_refreshes": 0,
            "table_fit_attempts": 0,
            "table_column_resizes": 0,
            "table_layout_refresh_requests": 0,
            "rich_log_writes": 0,
            "output_history_high_water_chars": 0,
            "context_append_characters": 0,
            "context_high_water_characters": 0,
            "context_rendered_characters": 0,
            "beam_detail_renders": 0,
        }

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            yield Static("Waiting for an editor request…", id="idle-status")

    def on_mount(self) -> None:
        if self._ready_future is not None and not self._ready_future.done():
            self._ready_future.set_result(None)

    def _screen_ready(self, screen: _RequestScreen) -> None:
        if screen is self._active_screen:
            # Input that reached the driver before readiness belongs to the
            # previous handoff, even if Textual dispatches it later.
            self._input_event_cutoff = time.monotonic()
            if not screen._layout_metrics_subscribed:
                screen.screen_layout_refresh_signal.subscribe(
                    self, self._note_screen_layout_refresh,
                )
                screen._layout_metrics_subscribed = True
            self._set_command_palette_for(screen)

    def _note_screen_layout_refresh(self, _screen: Screen[Any]) -> None:
        self.stats["screen_layout_refreshes"] += 1

    def _set_command_palette_for(self, screen: _RequestScreen) -> None:
        state = getattr(screen, "state", None)
        self.use_command_palette = not bool(getattr(state, "single_key", False))

    def on_unmount(self) -> None:
        # The UI loop must finish unmounting before owner threads wait for
        # callbacks that may currently be returning through call_from_thread.
        self.close_executors(wait=False)

    async def on_event(self, event: events.Event) -> None:
        if isinstance(event, (events.InputEvent, events.Paste)):
            if (
                self._input_event_cutoff is not None
                and event.time <= self._input_event_cutoff
            ):
                self.stats["stale_input_events"] += 1
                if isinstance(event, events.Key):
                    self.stats["stale_key_events"] += 1
                elif isinstance(event, events.MouseEvent):
                    self.stats["stale_mouse_events"] += 1
                event.stop()
                event.prevent_default()
                return
            screen = self._active_screen
            if screen is None or not screen.accepting_input:
                event.stop()
                event.prevent_default()
                self.stats["stale_input_events"] += 1
                if isinstance(event, events.Key):
                    self.stats["stale_key_events"] += 1
                elif isinstance(event, events.MouseEvent):
                    self.stats["stale_mouse_events"] += 1
                return
            self._input_event_cutoff = None
        await super().on_event(event)

    def on_key(self, event: events.Key) -> None:
        screen = self._active_screen
        if screen is None:
            event.stop()
            event.prevent_default()

    def _mark_input_cutoff(self) -> None:
        """Keep already queued key events attached to the request that submitted."""
        self._input_event_cutoff = time.monotonic()

    def show_request(self, lifecycle: _RequestLifecycle) -> Any:
        """Install or update a request screen and return its mount awaitable, if any."""
        self._active_request = lifecycle
        state = lifecycle.state
        current = self.screen
        reuse_choice = (
            isinstance(state, ChoiceViewState)
            and state.review is None
            and isinstance(current, ChoiceScreen)
            and current.state.review is None
            and not current.accepting_input
        )
        if reuse_choice:
            assert isinstance(current, ChoiceScreen)
            screen = current
            self._active_screen = screen
            self._set_command_palette_for(screen)
            screen.update_request(lifecycle)
            mount = None
        elif isinstance(state, ChoiceViewState):
            screen = ChoiceScreen(self, lifecycle)
        elif isinstance(state, EdgeViewState):
            screen = EdgeScreen(self, lifecycle)
        elif isinstance(state, BeamViewState):
            screen = BeamScreen(self, lifecycle)
        else:
            screen = PromptScreen(self, lifecycle)
        if not reuse_choice:
            self._active_screen = screen
            self._set_command_palette_for(screen)
            if isinstance(current, _RequestScreen) and not current.accepting_input:
                mount = self.switch_screen(screen)
            else:
                mount = self.push_screen(screen)
        if isinstance(state, ChoiceViewState):
            self._start_search_warm(lifecycle, screen)
        return mount

    def _screen_result(self, lifecycle: _RequestLifecycle, value: Any) -> None:
        if lifecycle is not self._active_request:
            return
        self._active_screen = None
        self.use_command_palette = False
        if not isinstance(value, _TerminalResponse):
            value = _TerminalResponse(value=value)
        lifecycle.submitted_result = value.value
        lifecycle.submitted_exception = value.exception
        if isinstance(lifecycle.state, ChoiceViewState):
            lifecycle.submitted_target = self._submission_target(lifecycle.state, value.value)
        if value.exception is not None:
            if not lifecycle.response.done():
                lifecycle.response.set_exception(value.exception)
            return
        self._update_warm_cancellation(lifecycle, value.value)
        if not lifecycle.response.done():
            lifecycle.response.set_result(value.value)

    @staticmethod
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

    def _update_warm_cancellation(self, lifecycle: _RequestLifecycle, result: Any) -> None:
        state = lifecycle.state
        if not isinstance(state, ChoiceViewState):
            return
        target = state.search_warm_target
        if target is None:
            return
        submitted_target = lifecycle.submitted_target
        if (
            submitted_target is not None
            and submitted_target != target
        ) or (
            isinstance(result, str)
            and result.strip().startswith("/")
            and result.strip() not in state.search_warm_commands
        ):
            lifecycle.warm_cancelled.set()

    def _start_search_warm(self, lifecycle: _RequestLifecycle, screen: _RequestScreen) -> None:
        state = lifecycle.state
        if (
            not isinstance(state, ChoiceViewState)
            or state.warm_search_token is None
            or state.search_warm_target is None
        ):
            return
        if state.search_warm_prepared:
            lifecycle.warm_result = True
            if isinstance(screen, ChoiceScreen):
                screen.warm_completed(state.search_warm_target, True, None)
            return
        target = state.search_warm_target
        generation = lifecycle.generation
        cancelled = lifecycle.warm_cancelled
        callback = state.warm_search_token
        if isinstance(screen, ChoiceScreen):
            screen.warm_started(target)
        lifecycle.warm_future = self._warm_executor.submit(
            callback, target[0], target[1], generation, cancelled.is_set
        )
        self.stats["warm_dispatches"] += 1

        def finished(future: Future[Any]) -> None:
            try:
                value = bool(future.result())
                error = None
            except BaseException as exc:  # noqa: BLE001 - worker failures return through the UI lifecycle.
                value, error = False, exc
            try:
                if not self.is_running:
                    return
                self.call_from_thread(
                    self._warm_completed,
                    generation,
                    target,
                    value,
                    error,
                )
            except (FutureCancelledError, RuntimeError):
                return

        lifecycle.warm_future.add_done_callback(finished)

    def _warm_completed(
        self,
        generation: int,
        target: tuple[int, int],
        value: bool,
        error: BaseException | None,
    ) -> None:
        lifecycle = self._active_request
        if lifecycle is None or lifecycle.generation != generation:
            return
        lifecycle.warm_result = value and error is None
        screen = self._active_screen
        if isinstance(screen, ChoiceScreen):
            screen.warm_completed(target, value, error)

    def request_owner_preview(
        self,
        generation: int,
        key: tuple[Any, ...],
        callback: Callable[[], Any],
    ) -> None:
        lifecycle = self._active_request
        if lifecycle is None or lifecycle.generation != generation:
            return
        if lifecycle.owner_queue is not None:
            lifecycle.owner_queue.put(_OwnerPreview(generation, key, callback))
            return

        future = self._preview_executor.submit(callback)

        def finished(result_future: Future[Any]) -> None:
            try:
                result = result_future.result()
                error = None
            except BaseException as exc:  # noqa: BLE001 - worker failures return through the UI lifecycle.
                result, error = None, exc
            try:
                if not self.is_running:
                    return
                self.call_from_thread(
                    self.deliver_owner_preview,
                    generation,
                    key,
                    result,
                    error,
                )
            except (FutureCancelledError, RuntimeError):
                return

        future.add_done_callback(finished)

    def deliver_owner_preview(
        self,
        generation: int,
        key: tuple[Any, ...],
        result: Any,
        error: BaseException | None,
    ) -> None:
        screen = self._active_screen
        if screen is not None and screen.lifecycle.generation == generation:
            screen.owner_preview_ready(key, result, error)

    def write_output(self, text: str) -> None:
        safe_text = _safe_context_text(text)
        if not safe_text:
            return
        if len(safe_text) > self.OUTPUT_HISTORY_LIMIT:
            safe_text = safe_text[-self.OUTPUT_HISTORY_LIMIT:]
        self._output_chunks.append(safe_text)
        self._output_history_chars += len(safe_text)
        history_trimmed = False
        while self._output_history_chars > self.OUTPUT_HISTORY_LIMIT:
            history_trimmed = True
            oldest = self._output_chunks.popleft()
            excess = self._output_history_chars - self.OUTPUT_HISTORY_LIMIT
            if len(oldest) > excess:
                oldest = oldest[excess:]
                self._output_chunks.appendleft(oldest)
                self._output_history_chars -= excess
            else:
                self._output_history_chars -= len(oldest)
        self.stats["output_history_high_water_chars"] = max(
            self.stats["output_history_high_water_chars"], self._output_history_chars,
        )
        if isinstance(self.screen, OutputScreen):
            if history_trimmed:
                self.screen.refresh_output()
            else:
                self.screen.append_output(safe_text)

    def start_output(self) -> None:
        if self._active_screen is not None and self._active_screen.accepting_input:
            self.push_screen(OutputScreen(self))

    def start_help(self, screen: _RequestScreen) -> None:
        self.push_screen(HelpScreen(_help_document(screen.lifecycle.state)))

    def close_executors(self, *, wait: bool = True) -> None:
        self._warm_executor.shutdown(wait=wait, cancel_futures=True)
        self._preview_executor.shutdown(wait=wait, cancel_futures=True)


class _RequestScreen(Screen[_TerminalResponse]):
    """A request screen whose result is the only input returned to the engine."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("f1", "show_help", "Help", show=False, priority=True),
        Binding("ctrl+l", "show_output", "Captured output", show=False, priority=True),
        Binding("ctrl+c", "interrupt", "Interrupt", show=False, priority=True),
    ]

    def __init__(self, terminal_app: PolicyEditorApp, lifecycle: _RequestLifecycle) -> None:
        super().__init__()
        self.terminal_app = terminal_app
        self.lifecycle = lifecycle
        self._layout_metrics_subscribed = False
        self.accepting_input = False
        self._submitted = False
        self._owner_preview_values: dict[tuple[Any, ...], Any] = {}
        self._owner_preview_errors: dict[tuple[Any, ...], BaseException] = {}
        self._owner_preview_pending: set[tuple[Any, ...]] = set()

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            yield from self.compose_request()

    def compose_request(self) -> ComposeResult:
        raise NotImplementedError

    def on_mount(self) -> None:
        self.call_after_refresh(self._enable_input)

    def _enable_input(self) -> None:
        if self._submitted:
            return
        self.accepting_input = True
        self.set_input_enabled(True)
        self.terminal_app._screen_ready(self)

    def set_input_enabled(self, enabled: bool) -> None:
        del enabled

    async def on_event(self, event: events.Event) -> None:
        if isinstance(event, (events.InputEvent, events.Paste)) and not self.accepting_input:
            event.stop()
            event.prevent_default()
            return
        await super().on_event(event)

    def on_key(self, event: events.Key) -> None:
        if not self.accepting_input:
            event.stop()
            event.prevent_default()

    def action_show_help(self) -> None:
        if self.accepting_input:
            self.terminal_app.start_help(self)

    def action_show_output(self) -> None:
        if self.accepting_input:
            self.terminal_app.start_output()

    def action_interrupt(self) -> None:
        self.submit_exception(KeyboardInterrupt())

    def submit(self, value: Any = None) -> None:
        if not self.accepting_input:
            return
        self.accepting_input = False
        self._submitted = True
        self.set_input_enabled(False)
        self.terminal_app._mark_input_cutoff()
        self.terminal_app._screen_result(
            self.lifecycle,
            _TerminalResponse(value=value),
        )

    def submit_exception(self, error: BaseException) -> None:
        if not self.accepting_input:
            return
        self.accepting_input = False
        self._submitted = True
        self.set_input_enabled(False)
        self.terminal_app._mark_input_cutoff()
        self.terminal_app._screen_result(
            self.lifecycle,
            _TerminalResponse(exception=error),
        )

    def insert_command(self, command: str) -> None:
        if self.accepting_input:
            self.apply_palette_command(command)

    def apply_palette_command(self, command: str) -> None:
        del command

    def owner_preview_ready(
        self,
        key: tuple[Any, ...],
        result: Any,
        error: BaseException | None,
    ) -> None:
        self._owner_preview_pending.discard(key)
        if error is None:
            self._owner_preview_values[key] = result
            self._owner_preview_errors.pop(key, None)
        else:
            self._owner_preview_errors[key] = error


class ChoiceScreen(_RequestScreen):
    """Candidate decision and read-only historical review surface."""

    BINDINGS: ClassVar[list[Binding]] = [
        *_RequestScreen.BINDINGS,
        Binding("enter", "submit_command", "Commit", show=False, priority=True),
        Binding("tab", "cycle_next", "Next choice", show=False, priority=True),
        Binding("shift+tab", "cycle_previous", "Previous choice", show=False, priority=True),
        Binding("ctrl+g", "explore_rank", "Explore rank", show=False, priority=True),
        Binding("ctrl+e", "toggle_editor", "Expand editor", show=False, priority=True),
        Binding("alt+enter", "insert_newline", "New line", show=False, priority=True),
        Binding("pageup", "context_up", "Context up", show=False, priority=True),
        Binding("pagedown", "context_down", "Context down", show=False, priority=True),
        Binding("escape", "leave_review", "Back", show=False, priority=True),
        Binding("[", "review_back", "Previous boundary", show=False, priority=True),
        Binding("]", "review_forward", "Next boundary", show=False, priority=True),
        Binding("f", "review_fork", "Fork boundary", show=False, priority=True),
        Binding("ctrl+d", "close_choice", "Close", show=False, priority=True),
    ]

    def __init__(self, terminal_app: PolicyEditorApp, lifecycle: _RequestLifecycle) -> None:
        super().__init__(terminal_app, lifecycle)
        assert isinstance(lifecycle.state, ChoiceViewState)
        self.state = lifecycle.state
        self.generation = lifecycle.generation
        self._completion_owned = bool(self.state.initial_command and self.state.review is None)
        self._expanded = False
        self._follow_tail = True
        self._preview_key: tuple[int, str] | None = None
        self._preview: ActionPreview | None = None
        self._local_feedback: ChoiceFeedback | None = None
        self._command_text = self.state.initial_command if self._completion_owned else ""
        self._navigation = self._navigation_commands()
        self._suppress_changed = False
        self._preview_error: str | None = None
        self._warm_pending_target: tuple[int, int] | None = None
        self._candidate_by_rank: dict[int, Candidate] = {}
        self._candidate_table_columns: CandidateColumns | None = None
        self._candidate_column_keys: tuple[str, ...] = ()
        self._candidate_focus_rank: int | None = None
        self._programmatic_candidate_highlights: set[str] = set()
        self._candidate_table_ready = False
        self._rendered_context_tail: str | None = None
        self._rendered_context: Text | None = None

    def compose_request(self) -> ComposeResult:
        if self.state.review is not None:
            yield Static(
                _render_review(
                    self.state.review,
                    seamless=self.state.seamless,
                    theme=self.terminal_app.theme_name,
                    environment=self.terminal_app.environment,
                ),
                id="review-header",
                classes="status-strong",
            )
            yield VerticalScroll(Static(id="review-context"), id="context-scroll")
            yield Static(id="choice-feedback")
            yield TextArea(
                self._command_text,
                id="choice-input",
                compact=True,
                read_only=True,
                soft_wrap=True,
                placeholder="Historical review is read-only",
            )
        else:
            yield Static(id="choice-heading", classes="status-strong")
            yield VerticalScroll(Static(id="context"), id="context-scroll")
            yield Static(id="choice-preview")
            yield Static("Candidates", classes="section")
            yield _CommandTemplateDataTable(
                id="choice-table", cursor_type="row", zebra_stripes=False,
                stretch_column="text",
            )
            yield Static(id="choice-feedback")
            with Horizontal(id="choice-command-row"):
                yield Static("Command >", classes="prompt-label")
                yield TextArea(
                    self._command_text,
                    id="choice-input",
                    compact=True,
                    read_only=self._completion_owned,
                    soft_wrap=True,
                    tab_behavior="focus",
                )
        yield Static(id="hint", classes="hint")

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one("#choice-feedback", Static).display = False
        input_widget = self.query_one("#choice-input", TextArea)
        # Historical review still receives screen-level keyboard commands.
        # Keep its editor read-only while giving the active screen a focused
        # target so terminal-driver key events reach ChoiceScreen.on_key.
        input_widget.focus()
        if self.state.review is None and self._completion_owned:
            input_widget.read_only = True
        if self.state.review is not None:
            self.query_one("#review-context", Static).update(
                Text(
                    "HISTORICAL CONTEXT\n" + _safe_context_text(self.state.review.context_text_tail),
                    style=semantic_style(
                        "section", self.terminal_app.theme_name,
                        environment=self.terminal_app.environment,
                    ),
                )
            )
            self._render_warm_status()
            self._render_feedback()
            return
        self._build_candidate_table()
        self._render_boundary_context()
        self._render_warm_status()

    def update_request(self, lifecycle: _RequestLifecycle) -> None:
        """Refresh a consecutive live boundary without replacing this screen."""
        assert isinstance(lifecycle.state, ChoiceViewState)
        assert lifecycle.state.review is None
        self.lifecycle = lifecycle
        self.state = lifecycle.state
        self.generation = lifecycle.generation
        self.accepting_input = False
        self._submitted = False
        self._owner_preview_values.clear()
        self._owner_preview_errors.clear()
        self._owner_preview_pending.clear()
        self._completion_owned = bool(self.state.initial_command)
        self._expanded = False
        self._follow_tail = True
        self._preview_key = None
        self._preview = None
        self._local_feedback = None
        self._preview_error = None
        self._warm_pending_target = None
        self._candidate_by_rank.clear()
        self._candidate_table_columns = None
        self._candidate_column_keys = ()
        self._candidate_focus_rank = None
        self._programmatic_candidate_highlights.clear()
        self._candidate_table_ready = False
        self._command_text = self.state.initial_command if self._completion_owned else ""
        self._navigation = self._navigation_commands()

        input_widget = self.query_one("#choice-input", TextArea)
        input_widget.disabled = False
        input_widget.read_only = True
        input_widget.remove_class("expanded")
        self._suppress_changed = True
        input_widget.text = self._command_text
        lines = self._command_text.splitlines() or [""]
        input_widget.move_cursor((len(lines) - 1, len(lines[-1])))
        self._suppress_changed = False
        input_widget.focus()

        self.query_one("#choice-table", DataTable).clear(columns=True)
        self._build_candidate_table()
        self._render_boundary_context()
        self._render_warm_status()
        self.call_after_refresh(self._enable_input)

    def set_input_enabled(self, enabled: bool) -> None:
        widget = self.query_one("#choice-input", TextArea)
        # The app and screen reject input while a request is submitted. Keep
        # the field focused/read-only through the owner-thread handoff so its
        # focused command bar does not blink out between consecutive choices.
        widget.disabled = False
        if self.state.review is None:
            widget.read_only = not enabled or self._completion_owned
            if enabled:
                widget.focus()
        else:
            widget.read_only = True

    def _navigation_commands(self) -> tuple[str, ...]:
        visible = tuple(
            self.state.candidates
            if self.state.display_candidates is None
            else self.state.display_candidates
        )
        return _navigation_command_cycle(
            self.state.choice,
            visible,
            self.state.feedback,
            sort_by_policy=self.state.sort_by_policy,
            sort_by_gumbel=self.state.sort_by_gumbel,
            search_lens_active=self.state.search_lens_active,
        )

    def _candidate_table_plan(self):
        return candidate_table_plan(
            self.state.choice,
            self.state.candidates,
            self._command_text,
            target_token_id=self.state.target_token_id,
            policy_active=self.state.policy_active,
            show_policy_rank=self.state.show_policy_rank,
            sort_by_policy=self.state.sort_by_policy,
            sort_by_gumbel=self.state.sort_by_gumbel,
            logit_view=self.state.logit_view,
            show_model_probabilities=self.state.show_model_probabilities,
            column_focus=self.state.column_focus,
            overlays=self.state.overlays,
            display_candidates=self.state.display_candidates,
            search_lens_active=self.state.search_lens_active,
            preview=self._preview,
        )

    def _candidate_row(self, candidate: Candidate, focus_rank: int | None) -> tuple[Text, ...]:
        assert self._candidate_table_columns is not None
        return candidate_table_row(
            candidate,
            self._candidate_table_columns,
            focus_rank=focus_rank,
            target_token_id=self.state.target_token_id,
            theme=self.terminal_app.theme_name,
            environment=self.terminal_app.environment,
        )

    def _build_candidate_table(self) -> None:
        table = self.query_one("#choice-table", _FluidDataTable)
        self._refresh_preview()
        plan = self._candidate_table_plan()
        self._candidate_table_columns = plan.columns
        self._candidate_by_rank = {candidate.rank: candidate for candidate in plan.candidates}
        self._candidate_column_keys = (
            "marker", "rank", *(label for label, _width in plan.columns.columns), "text"
        )
        table.add_column("", width=2, key="marker")
        table.add_column("rank", width=6, key="rank")
        for label, width in plan.columns.columns:
            table.add_column(label, width=width, key=label)
        table.add_column("text", key="text")
        for candidate in plan.candidates:
            cells = self._candidate_row(candidate, plan.focus_rank)
            table.add_row(*cells, key=str(candidate.rank))
        self._candidate_focus_rank = plan.focus_rank
        self._candidate_table_ready = True
        self._move_candidate_cursor(table, plan.candidates, plan.focus_rank)
        table.fit_column()
        self._render_preview()
        self.query_one("#choice-heading", Static).update(
            f"Step {self.state.choice.aligned_step} · teacher track"
        )

    def _move_candidate_cursor(
        self, table: DataTable, candidates, focus_rank: int | None
    ) -> None:
        focus_row = next(
            (index for index, candidate in enumerate(candidates) if candidate.rank == focus_rank),
            None,
        )
        if focus_row is None:
            table.cursor_type = "none"
            return
        table.cursor_type = "row"
        if (
            self.accepting_input
            and table.cursor_coordinate.row != focus_row
        ):
            self._programmatic_candidate_highlights.add(
                str(candidates[focus_row].rank)
            )
        table.move_cursor(row=focus_row, column=0, animate=False)

    @on(DataTable.RowHighlighted, "#choice-table")
    def _on_candidate_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if not self.accepting_input or self.state.review is not None:
            return
        try:
            rank = int(event.row_key.value)
        except (TypeError, ValueError):
            return
        table = self.query_one("#choice-table", DataTable)
        if not table.is_valid_row_index(table.cursor_coordinate.row):
            return
        cursor_row_key = table.coordinate_to_cell_key(
            table.cursor_coordinate
        ).row_key.value
        if str(cursor_row_key) != str(rank):
            return
        if str(rank) in self._programmatic_candidate_highlights:
            self._programmatic_candidate_highlights.discard(str(rank))
            return
        if rank not in self._candidate_by_rank:
            return
        if self._preview is not None and self._preview.candidate_rank == rank:
            return
        # A mouse selection stages the command in the editor and returns the
        # caret there so the next typed character has an obvious destination.
        self._set_command(str(rank), owned=False, focus=True)

    @on(DataTable.RowSelected, "#choice-table")
    def _on_candidate_selected(self, _event: DataTable.RowSelected) -> None:
        if self.accepting_input and self.state.review is None:
            self.query_one("#choice-input", TextArea).focus()

    def _refresh_candidate_focus(self) -> None:
        if not self._candidate_table_ready or not self.is_mounted:
            return
        table = self.query_one("#choice-table", DataTable)
        plan = self._candidate_table_plan()
        changed_ranks = {self._candidate_focus_rank, plan.focus_rank}
        for rank in changed_ranks:
            if rank is None:
                continue
            candidate = self._candidate_by_rank.get(rank)
            if candidate is None:
                continue
            for column_key, cell in zip(
                self._candidate_column_keys,
                self._candidate_row(candidate, plan.focus_rank),
                strict=True,
            ):
                table.update_cell(str(rank), column_key, cell)
        self._candidate_focus_rank = plan.focus_rank
        self._move_candidate_cursor(table, plan.candidates, plan.focus_rank)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "choice-input" or self._suppress_changed:
            return
        input_had_focus = event.text_area.has_focus
        self._command_text = event.text_area.text
        self._local_feedback = None
        self._preview_error = None
        self._expanded = self._expanded and _is_writing(self._command_text)
        if self._expanded:
            event.text_area.add_class("expanded")
        else:
            event.text_area.remove_class("expanded")
        self._follow_tail = True
        self._refresh_preview()
        if input_had_focus and self.accepting_input and self.state.review is None:
            event.text_area.focus()

    def on_key(self, event: events.Key) -> None:
        super().on_key(event)
        if not self.accepting_input or self.state.review is not None:
            if self.accepting_input and self.state.review is not None:
                if event.key == "f" and not self._command_text:
                    self._set_command("f", owned=False)
                    event.stop()
                    event.prevent_default()
                elif event.key not in {
                    "enter", "escape", "ctrl+c", "ctrl+d", "tab", "shift+tab",
                    "[", "]", "question_mark", "ctrl+k",
                }:
                    self.submit("\x1b")
                    event.stop()
                    event.prevent_default()
            return
        if self._completion_owned and event.character:
            self._set_command(event.character, owned=False)
            event.stop()
            event.prevent_default()

    def _set_command(self, value: str, *, owned: bool, focus: bool = True) -> None:
        command_changed = value != self._command_text
        self._suppress_changed = True
        self._command_text = value
        self._completion_owned = owned
        if command_changed:
            self._preview_key = None
        widget = self.query_one("#choice-input", TextArea)
        widget.text = value
        lines = value.splitlines() or [""]
        widget.move_cursor((len(lines) - 1, len(lines[-1])))
        widget.read_only = self.state.review is not None or owned
        self._suppress_changed = False
        self._local_feedback = None
        self._refresh_preview()
        if self.accepting_input and self.state.review is None and focus:
            widget.focus()

    def _preview_request_key(self, command_text: str, kind: str, value: Any) -> tuple[Any, ...]:
        return (self.generation, kind, command_text, value)

    def _owner_resolver(self, command_text: str):
        def resolve_insertion(text: str, mode) -> str:
            key = self._preview_request_key(command_text, "insertion", (text, mode))
            if key in self._owner_preview_errors:
                raise self._owner_preview_errors[key]
            if key in self._owner_preview_values:
                return self._owner_preview_values[key]
            if key not in self._owner_preview_pending:
                self._owner_preview_pending.add(key)
                self.terminal_app.request_owner_preview(
                    self.generation,
                    key,
                    lambda: self.state.resolve_insertion(text, mode),
                )
            previous = self._preview.appended_text if self._preview is not None else None
            raise PreviewPending(previous)

        def resolve_candidate(rank: int):
            if self.state.resolve_candidate is None:
                raise EditorError(f"raw rank {rank} is not available in the current choice")
            key = self._preview_request_key(command_text, "candidate", rank)
            if key in self._owner_preview_errors:
                raise self._owner_preview_errors[key]
            if key in self._owner_preview_values:
                return self._owner_preview_values[key]
            if key not in self._owner_preview_pending:
                self._owner_preview_pending.add(key)
                self.terminal_app.request_owner_preview(
                    self.generation,
                    key,
                    lambda: self.state.resolve_candidate(rank),
                )
            raise PreviewPending()

        return resolve_insertion, resolve_candidate

    def _refresh_preview(self) -> None:
        if self.state.review is not None:
            return
        key = (self.generation, self._command_text)
        if self._preview_key == key and self._preview is not None:
            return
        resolve_insertion, resolve_candidate = self._owner_resolver(self._command_text)
        self._preview = action_preview(
            self.state.choice,
            self._command_text,
            self.state.candidates,
            resolve_insertion,
            resolve_candidate=resolve_candidate,
            default_hold_tokens=self.state.default_hold_tokens,
            default_search_radius=self.state.default_search_radius,
        )
        self._preview_key = key
        if self.is_mounted and self._candidate_table_ready:
            self._render_preview()

    def _render_preview(self) -> None:
        if self._preview is None:
            return
        preview_text = _preview_fragments(
            self._preview,
            policy_active=self.state.policy_active,
            theme=self.terminal_app.theme_name,
            environment=self.terminal_app.environment,
        )
        self.query_one("#choice-preview", Static).update(preview_text)
        self._refresh_candidate_focus()
        self._render_feedback()

    def _render_boundary_context(self) -> None:
        if self.state.review is not None:
            return
        context_tail = _safe_context_text(self.state.choice.context_text_tail)
        context_widget = self.query_one("#context", Static)
        rendered = False
        if (
            self._rendered_context is not None
            and self._rendered_context_tail is not None
            and context_tail.startswith(self._rendered_context_tail)
        ):
            addition = context_tail[len(self._rendered_context_tail):]
            if addition:
                self._rendered_context.append(addition)
                self.terminal_app.stats["context_append_characters"] += len(addition)
                context_widget.update(self._rendered_context)
                rendered = True
        else:
            context = Text(
                "DECISION BOUNDARY\n",
                style=semantic_style(
                    "section", self.terminal_app.theme_name,
                    environment=self.terminal_app.environment,
                ),
            )
            context.append(context_tail)
            self._rendered_context = context
            context_widget.update(context)
            rendered = True
        if rendered:
            self.terminal_app.stats["context_rendered_characters"] += len(
                self._rendered_context
            )
        self._rendered_context_tail = context_tail
        self.terminal_app.stats["context_high_water_characters"] = max(
            self.terminal_app.stats["context_high_water_characters"],
            len(context_tail),
        )
        scroll = self.query_one("#context-scroll", VerticalScroll)
        if self._follow_tail:
            scroll.scroll_end(animate=False)

    def _render_feedback(self) -> None:
        target = self.query_one("#choice-feedback", Static)
        feedback = self._local_feedback or self.state.feedback
        rendered = Text()
        if feedback is not None:
            category = feedback.category if feedback.category in {"error", "info", "search"} else "info"
            rendered.append(
                _safe_context_text(feedback.title) + "\n",
                style=semantic_style(
                    f"feedback-{category}", self.terminal_app.theme_name,
                    environment=self.terminal_app.environment,
                ),
            )
            for line in feedback.lines:
                rendered.append(
                    "  " + _safe_context_text(line) + "\n",
                    style=semantic_style(
                        "feedback-detail", self.terminal_app.theme_name,
                        environment=self.terminal_app.environment,
                    ),
                )
        if self._preview_error:
            rendered.append(
                self._preview_error + "\n",
                style=semantic_style(
                    "feedback-error", self.terminal_app.theme_name,
                    environment=self.terminal_app.environment,
                ),
            )
        visible = bool(rendered.plain)
        if target.content == rendered and target.display == visible:
            return
        if not visible:
            if target.display:
                target.display = False
            if target.content != rendered:
                target.update(rendered, layout=False)
            return
        if target.content != rendered:
            target.update(rendered)
        if not target.display:
            target.display = True

    def _render_warm_status(self) -> None:
        try:
            target = self.query_one("#hint", Static)
        except NoMatches:
            return
        if self._warm_pending_target is None:
            target.update(self._choice_hint())
            return
        pending = Text(no_wrap=True, overflow="ellipsis")
        pending.append(
            f"Resolving raw rank {self._warm_pending_target[0]}…",
            style=semantic_style(
                "pending", self.terminal_app.theme_name,
                environment=self.terminal_app.environment,
            ),
        )
        pending.append(" · " + self._choice_hint())
        target.update(pending)

    def _choice_hint(self) -> str:
        if self.state.review is not None:
            return "Enter resumes · f forks · Esc returns\nPgUp/PgDn scroll · Ctrl+L output · F1 help"
        if self._expanded:
            return "Alt+Enter newline · Enter commits\nTab inserts · Ctrl+K menu · Ctrl+L output · F1 help"
        return "Tab cycles · Enter commits · Ctrl+G rank\nPgUp/PgDn context · Ctrl+K menu · Ctrl+L output · F1 help"

    def warm_started(self, target: tuple[int, int]) -> None:
        self._warm_pending_target = target
        self._render_warm_status()

    def owner_preview_ready(
        self,
        key: tuple[Any, ...],
        result: Any,
        error: BaseException | None,
    ) -> None:
        super().owner_preview_ready(key, result, error)
        if key[0] != self.generation or key[2] != self._command_text:
            return
        preview = self._preview
        if preview is None:
            return
        if error is not None:
            if key[1] == "candidate" and isinstance(error, EditorError):
                self._preview = ActionPreview(
                    kind="effect", label="selected raw rank",
                    detail=f"Candidate preview unavailable: {error}",
                    valid=False, state="invalid", command=preview.command,
                )
            else:
                self._preview_error = f"Preview unavailable: {type(error).__name__}: {error}"
        elif key[1] == "insertion":
            self._preview = ActionPreview(
                kind="insertion",
                label=preview.label,
                detail=preview.detail,
                appended_text=result,
                command=preview.command,
            )
        elif key[1] == "candidate":
            self._preview = replace(
                _candidate_preview(result, label="selected candidate"),
                command=preview.command,
            )
        self._render_preview()

    def warm_completed(
        self,
        target: tuple[int, int],
        value: bool,
        error: BaseException | None,
    ) -> None:
        if target != self.state.search_warm_target:
            return
        self._warm_pending_target = None
        if error is not None:
            self._preview_error = f"Search warm-up unavailable: {type(error).__name__}: {error}"
        elif value:
            self._preview_error = None
        self._render_warm_status()
        self._render_feedback()

    def _navigate(self, direction: int) -> None:
        if not self.accepting_input or not self._navigation:
            return
        if self.state.review is not None:
            self.submit("\x1b")
            return
        current = self._command_text
        if not current:
            suggestions = self.state.feedback.completion_commands if self.state.feedback else ()
            if suggestions:
                self._set_command(suggestions[0] if direction > 0 else suggestions[-1], owned=False)
                return
            if direction > 0 and self.state.search_lens_active:
                match_command = self.state.feedback.initial_tab_command if self.state.feedback else None
                if match_command is None and self.state.target_token_id is not None:
                    match_command = next(
                        (str(candidate.rank) for candidate in self.state.display_candidates or ()
                         if candidate.token_id == self.state.target_token_id),
                        None,
                    )
                if match_command is not None:
                    self._set_command(match_command, owned=False)
                    return
        if not current:
            if self.state.search_lens_active and self.state.target_token_id is not None:
                match_command = next(
                    (str(candidate.rank) for candidate in self.state.display_candidates or ()
                     if candidate.token_id == self.state.target_token_id),
                    self._navigation[0],
                )
                index = self._navigation.index(match_command) if match_command in self._navigation else 0
            else:
                self._set_command(self._navigation[0] if direction > 0 else self._navigation[-1], owned=False)
                return
        else:
            try:
                index = self._navigation.index(current)
            except ValueError:
                if self.state.search_lens_active:
                    match_command = next(
                        (str(candidate.rank) for candidate in self.state.display_candidates or ()
                         if candidate.token_id == self.state.target_token_id),
                        self._navigation[0],
                    )
                    self._set_command(match_command, owned=False)
                    return
                proposal_rank = next(
                    (candidate.rank for candidate in self.state.display_candidates or ()
                     if candidate.token_id == self.state.choice.proposal_token_id),
                    None,
                )
                if current.isdigit() and int(current) == proposal_rank:
                    index = 0
                else:
                    return
        self._set_command(
            self._navigation[(index + direction) % len(self._navigation)],
            owned=False,
        )

    def action_cycle_next(self) -> None:
        if _is_writing(self._command_text):
            self.query_one("#choice-input", TextArea).insert("\t")
        else:
            self._navigate(1)

    def action_cycle_previous(self) -> None:
        if _is_writing(self._command_text):
            self.query_one("#choice-input", TextArea).insert("\t")
        else:
            self._navigate(-1)

    def action_explore_rank(self) -> None:
        raw = self._command_text.strip()
        if self.accepting_input and self.state.review is None and raw.isdigit():
            rank = int(raw)
            size = self.state.choice.vocabulary_size
            if rank >= 1 and (size is None or rank <= size):
                self.submit(f"ms {rank}")

    def action_toggle_editor(self) -> None:
        if not self.accepting_input or self.state.review is not None or not _is_writing(self._command_text):
            return
        self._expanded = not self._expanded
        widget = self.query_one("#choice-input", TextArea)
        if self._expanded:
            widget.add_class("expanded")
        else:
            widget.remove_class("expanded")
        self._render_warm_status()

    def action_insert_newline(self) -> None:
        if self.accepting_input and self.state.review is None and _is_writing(self._command_text):
            self.query_one("#choice-input", TextArea).insert("\n")

    def _scroll_context(self, amount: int) -> None:
        if not self.accepting_input:
            return
        scroll = self.query_one("#context-scroll", VerticalScroll)
        self._follow_tail = False
        scroll.scroll_relative(y=amount, animate=False, immediate=True)
        if amount > 0 and scroll.scroll_y >= scroll.max_scroll_y:
            self._follow_tail = True

    def action_context_up(self) -> None:
        if self.state.review is None:
            self._scroll_context(-max(1, self.query_one("#context-scroll", VerticalScroll).size.height - 1))

    def action_context_down(self) -> None:
        if self.state.review is None:
            self._scroll_context(max(1, self.query_one("#context-scroll", VerticalScroll).size.height - 1))

    def action_review_back(self) -> None:
        if not self.accepting_input:
            return
        if self.state.review is None:
            if self._completion_owned or not self._command_text:
                self.submit("[")
            else:
                self.query_one("#choice-input", TextArea).insert("[")

    def action_review_forward(self) -> None:
        if not self.accepting_input:
            return
        if self.state.review is None:
            if self._completion_owned or not self._command_text:
                self.submit("]")
            else:
                self.query_one("#choice-input", TextArea).insert("]")

    def action_review_fork(self) -> None:
        if self.accepting_input and self.state.review is not None and not self._command_text:
            self._set_command("f", owned=False)
        elif self.accepting_input and self.state.review is None:
            if self._completion_owned:
                self._set_command("f", owned=False)
            else:
                self.query_one("#choice-input", TextArea).insert("f")

    def action_leave_review(self) -> None:
        if self.accepting_input and self.state.review is not None or self.accepting_input and self.state.search_lens_active:
            self.submit("\x1b")

    def action_close_choice(self) -> None:
        if self.accepting_input and not self._command_text:
            self.submit(None)

    def action_submit_command(self) -> None:
        if not self.accepting_input:
            return
        raw = self._command_text
        if self.state.review is not None:
            if not raw.strip() and self.state.seamless and self.state.reactivate_on_review_enter:
                self.submit(SEAMLESS_REACTIVATE)
            elif raw.strip().lower() in {"f", "fork"}:
                self.submit(raw)
            elif not raw.strip():
                self.submit("\x1b")
            else:
                self.submit("\x1b")
            return
        self._refresh_preview()
        if self._preview is not None and self._preview.state in {"invalid", "incomplete"}:
            self._local_feedback = ChoiceFeedback(
                "error" if self._preview.state == "invalid" else "info",
                self._preview.label.upper(),
                (self._preview.detail,),
            )
            self._render_feedback()
            return
        if self._preview is not None and self._preview.state == "pending":
            self._local_feedback = ChoiceFeedback(
                "info", "PREVIEW PENDING", (self._preview.detail,),
            )
            self._render_feedback()
            return
        self.submit(raw)

    def _render_candidate_feedback(self) -> None:
        self._render_feedback()

    def apply_palette_command(self, command: str) -> None:
        self._set_command(command, owned=False)
        self.query_one("#choice-input", TextArea).focus()


class EdgeScreen(_RequestScreen):
    """Live edge command entry with mode-specific commands."""

    BINDINGS: ClassVar[list[Binding]] = [
        *_RequestScreen.BINDINGS,
        Binding("enter", "submit_command", "Submit", show=False, priority=True),
        Binding("ctrl+d", "close_edge", "Quit", show=False, priority=True),
    ]

    def __init__(self, terminal_app: PolicyEditorApp, lifecycle: _RequestLifecycle) -> None:
        super().__init__(terminal_app, lifecycle)
        assert isinstance(lifecycle.state, EdgeViewState)
        self.state = lifecycle.state

    def compose_request(self) -> ComposeResult:
        title = "LIVE SESSION" if self.state.mode == "session" else "LIVE EDGE"
        entity = "Branch" if self.state.mode == "session" else "Episode"
        yield Static(
            Text(f"{title}\n{entity} {self.state.episode_id} · boundary {self.state.boundary}\nSampler · {self.state.sampler_summary}"),
            id="edge-header",
            classes="status-strong",
        )
        table = _CommandTemplateDataTable(
            id="edge-commands", cursor_type="row", stretch_column="description"
        )
        yield table
        with Horizontal(id="command-row"):
            yield Static("Command >", classes="prompt-label")
            yield TextArea("", id="edge-input", compact=True, soft_wrap=False, tab_behavior="focus")
        yield Static(id="hint", classes="hint")

    def on_mount(self) -> None:
        super().on_mount()
        table = self.query_one("#edge-commands", _FluidDataTable)
        table.add_column("command", key="command")
        table.add_column("description", key="description")
        for item in edge_help(self.state.mode):
            table.add_row(item.command, item.description, key=item.command, height=None)
        self.call_after_refresh(table.fit_column)
        self.query_one("#edge-input", TextArea).focus()
        self.query_one("#hint", Static).update(
            "Click a template · type to edit · Enter submits\n"
            "Blank continues · Ctrl+C interrupt · Ctrl+D quit\n"
            "Ctrl+L output · F1 help"
        )

    def set_input_enabled(self, enabled: bool) -> None:
        widget = self.query_one("#edge-input", TextArea)
        # Keep the command bar visibly focused across request handoff while
        # making it read-only until the next request is ready.
        widget.disabled = False
        widget.read_only = not enabled
        if not enabled:
            widget.focus()

    @on(DataTable.RowHighlighted, "#edge-commands")
    def _on_command_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if not self.accepting_input:
            return
        widget = self.query_one("#edge-input", TextArea)
        command = _edge_insert_command(str(event.row_key.value))
        widget.text = command
        widget.move_cursor((0, len(command)))
        widget.focus()

    @on(DataTable.RowSelected, "#edge-commands")
    def _on_command_selected(self, _event: DataTable.RowSelected) -> None:
        if self.accepting_input:
            self.query_one("#edge-input", TextArea).focus()

    def action_submit_command(self) -> None:
        if self.accepting_input:
            self.submit(self.query_one("#edge-input", TextArea).text)

    def action_close_edge(self) -> None:
        if self.accepting_input:
            self.submit(None)

    def apply_palette_command(self, command: str) -> None:
        widget = self.query_one("#edge-input", TextArea)
        widget.text = command
        widget.focus()


class BeamInputArea(TextArea):
    """Preserve branch shortcuts before a blank TextArea consumes a letter."""

    async def _on_key(self, event: events.Key) -> None:
        screen = self.screen
        shortcut = getattr(screen, "handle_empty_key", None)
        if event.key in {"p", "f"} and not self.text and callable(shortcut):
            shortcut(event.key)
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)


class BeamScreen(_RequestScreen):
    """Beam survivor table and branch controls."""

    SIDE_BY_SIDE_MIN_WIDTH = 120
    HORIZONTAL_BREAKPOINTS: ClassVar[list[tuple[int, str]]] = [
        (0, "-stacked"), (SIDE_BY_SIDE_MIN_WIDTH, "-side-by-side"),
    ]

    BINDINGS: ClassVar[list[Binding]] = [
        *_RequestScreen.BINDINGS,
        Binding("enter", "submit_command", "Submit", show=False, priority=True),
        Binding("up", "selection_up", "Previous branch", show=False, priority=True),
        Binding("down", "selection_down", "Next branch", show=False, priority=True),
        Binding("pageup", "detail_page_up", "Scroll details up", show=False, priority=True),
        Binding("pagedown", "detail_page_down", "Scroll details down", show=False, priority=True),
        Binding("backspace", "kill_selected", "Kill selected", show=False, priority=True),
        Binding("p", "protect_selected", "Protect selected", show=False, priority=True),
        Binding("f", "toggle_families", "Toggle families", show=False, priority=True),
        Binding("right", "advance", "Advance", show=False, priority=True),
        Binding("left", "rewind", "Rewind", show=False, priority=True),
        Binding("escape", "return_to_choice", "Return", show=False, priority=True),
        Binding("ctrl+d", "return_to_choice", "Return", show=False, priority=True),
    ]

    def __init__(self, terminal_app: PolicyEditorApp, lifecycle: _RequestLifecycle) -> None:
        super().__init__(terminal_app, lifecycle)
        assert isinstance(lifecycle.state, BeamViewState)
        self.state = lifecycle.state
        self.selected_label = self.state.selected_label
        self._labels: list[str] = []

    def compose_request(self) -> ComposeResult:
        title = self.state.title
        if self.state.at_edge:
            title = f"BEAM OPTIONS   ·   {title.removeprefix('BEAM   ')}"
        yield Static(title, id="beam-heading", classes="status-strong")
        yield Static(id="beam-context", classes="muted")
        with Horizontal(id="beam-body"):
            yield _CommandTemplateDataTable(
                id="beam-table", cursor_type="row", stretch_column="continuation"
            )
            with VerticalScroll(id="beam-detail-pane"):
                yield Static(id="beam-detail")
        yield Static(id="beam-notice")
        with Horizontal(id="command-row"):
            yield Static("Beam >", classes="prompt-label")
            yield BeamInputArea("", id="beam-input", compact=True, soft_wrap=False, tab_behavior="focus")
        yield Static(id="hint", classes="hint")

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one("#beam-context", Static).update(
            Text(
                "Shared context: " + _safe_context_text(self.state.shared_context),
                style=semantic_style(
                    "section", self.terminal_app.theme_name,
                    environment=self.terminal_app.environment,
                ),
            )
        )
        table = self.query_one("#beam-table", _CommandTemplateDataTable)
        for label, width in (
            ("", 2), ("#", 3), ("label", None), ("state", None),
            ("score", None), ("continuation", 1),
        ):
            table.add_column(label, width=width, key=label or "marker")
        self._labels = [row.label for row in self.state.rows]
        if self.selected_label not in self._labels:
            self.selected_label = self._labels[0] if self._labels else None
        self.query_one("#beam-input", TextArea).focus()
        for rank, row in enumerate(self.state.rows, 1):
            marker = ">" if row.label == self.selected_label else " "
            score = row.score.replace("-", "−")
            if self.state.stochastic:
                model_logp = (
                    "—" if row.model_log_probability is None
                    else f"{row.model_log_probability:.3f}"
                )
                score = f"G {score} · log-p {model_logp}"
            if row.protected:
                marker = "◆" if marker == " " else ">◆"
            continuation = row.continuation.replace("\n", " ↵ ")
            table.add_row(
                marker,
                str(rank),
                row.label,
                row.state,
                score,
                continuation,
                height=None,
                key=row.label,
            )
        selected_index = next(
            (index for index, row in enumerate(self.state.rows)
             if row.label == self.selected_label),
            0,
        )
        if self.state.rows:
            table.move_cursor(row=selected_index, column=0, animate=False)
        table.fit_column()
        self._render_details()
        self.query_one("#beam-notice", Static).update(self.state.notice)
        self.query_one("#beam-notice", Static).display = bool(self.state.notice)
        hint = (
            "Enter/→ resume · Esc/Ctrl+D return\nCtrl+K commands · Ctrl+L output · F1 help"
            if self.state.at_edge else
            "↑↓ select · ←/→ step · Enter commit\n"
            "PgUp/Dn details · Esc/Ctrl+D return\n"
            "Backspace kill · "
            + ("p protect · " if not self.state.stochastic else "")
            + "f family\nCtrl+K commands · Ctrl+L output · F1 help"
        )
        self.query_one("#hint", Static).update(hint)

    def set_input_enabled(self, enabled: bool) -> None:
        widget = self.query_one("#beam-input", TextArea)
        widget.disabled = False
        widget.read_only = not enabled
        if not enabled:
            widget.focus()

    def handle_empty_key(self, key: str) -> None:
        if key == "p":
            self.action_protect_selected()
        elif key == "f":
            self.action_toggle_families()

    def _render_details(self) -> None:
        self.terminal_app.stats["beam_detail_renders"] += 1
        row = next((item for item in self.state.rows if item.label == self.selected_label), None)
        label = self.selected_label or "—"
        protected = " · PROTECTED" if row is not None and row.protected else ""
        rendered = Text(f"SELECTED: {label}{protected}\n", style="bold underline")
        if row is None:
            rendered.append("(no retained branch)")
        else:
            rendered.append(f"STATE: {row.state} · SCORE: {row.score.replace('-', '−')}\n")
            if self.state.stochastic:
                model_logp = (
                    "—" if row.model_log_probability is None
                    else f"{row.model_log_probability:.3f}"
                )
                rendered.append(f"G {row.score.replace('-', '−')} · log-p {model_logp}\n")
            model_rank = "—" if row.model_rank is None else str(row.model_rank)
            step_logp = (
                "—" if row.step_log_probability is None
                else f"{row.step_log_probability:.3f}".replace("-", "−")
            )
            rendered.append(f"Model rank: {model_rank} · Step log-p: {step_logp}\n")
            if row.model_log_probability is not None and not self.state.stochastic:
                model_logp = f"{row.model_log_probability:.3f}".replace("-", "−")
                rendered.append(f"Model log-p: {model_logp}\n")
            if row.family_metadata:
                rendered.append(
                    row.family_metadata + "\n",
                    style=semantic_style(
                        "beam-family", self.terminal_app.theme_name,
                        environment=self.terminal_app.environment,
                    ),
                )
            rendered.append(row.continuation + "\n\n")
            rendered.append("Recent steps:\n", style="bold")
            if row.recent_steps:
                for step in row.recent_steps:
                    rendered.append(f"  {step}\n")
            else:
                rendered.append("  No generated steps yet.\n")
        self.query_one("#beam-detail", Static).update(rendered)

    def action_detail_page_up(self) -> None:
        if self.accepting_input:
            pane = self.query_one("#beam-detail-pane", VerticalScroll)
            pane.scroll_relative(y=-max(1, pane.size.height - 1), animate=False, immediate=True)

    def action_detail_page_down(self) -> None:
        if self.accepting_input:
            pane = self.query_one("#beam-detail-pane", VerticalScroll)
            pane.scroll_relative(y=max(1, pane.size.height - 1), animate=False, immediate=True)

    @on(DataTable.RowHighlighted, "#beam-table")
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._select_row(str(event.row_key.value))

    @on(DataTable.RowSelected, "#beam-table")
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        self._select_row(str(event.row_key.value))

    def _select_row(self, label: str) -> None:
        if label in self._labels and label != self.selected_label:
            self.selected_label = label
            self._render_details()
        if label in self._labels:
            self.query_one("#beam-input", TextArea).focus()

    def action_selection_up(self) -> None:
        self._move_selection(-1)

    def action_selection_down(self) -> None:
        self._move_selection(1)

    def _move_selection(self, direction: int) -> None:
        if not self.accepting_input or not self._labels:
            return
        try:
            current = self._labels.index(self.selected_label)
        except ValueError:
            current = 0 if direction >= 0 else len(self._labels) - 1
        selected = max(0, min(len(self._labels) - 1, current + direction))
        selected_label = self._labels[selected]
        self._select_row(selected_label)
        self.query_one("#beam-table", DataTable).move_cursor(row=selected, column=0, animate=False)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "beam-input":
            return
        if event.text_area.text:
            return
        self._refresh_marker()

    def _refresh_marker(self) -> None:
        table = self.query_one("#beam-table", DataTable)
        if self.selected_label in self._labels:
            table.move_cursor(row=self._labels.index(self.selected_label), column=0, animate=False)

    def action_submit_command(self) -> None:
        if not self.accepting_input:
            return
        command = self.query_one("#beam-input", TextArea).text.strip()
        if not command:
            command = (
                "resume" if self.state.at_edge
                else f"select {self.selected_label}" if self.selected_label is not None
                else ""
            )
        self.submit(BeamInput(command, self.selected_label))

    def action_kill_selected(self) -> None:
        widget = self.query_one("#beam-input", TextArea)
        if self.accepting_input and not self.state.at_edge and not widget.text:
            command = f"kill {self.selected_label}" if self.selected_label is not None else "k"
            self.submit(BeamInput(command, self.selected_label))
        elif self.accepting_input:
            widget.action_delete_left()

    def action_protect_selected(self) -> None:
        if (
            self.accepting_input and not self.state.at_edge and not self.state.stochastic
            and not self.query_one("#beam-input", TextArea).text
        ):
            self.submit(BeamInput("protect", self.selected_label))
        elif self.accepting_input:
            self.query_one("#beam-input", TextArea).insert("p")

    def action_toggle_families(self) -> None:
        widget = self.query_one("#beam-input", TextArea)
        if self.accepting_input and not self.state.at_edge and not widget.text:
            self.submit(BeamInput("families", self.selected_label))
        elif self.accepting_input:
            widget.insert("f")

    def action_advance(self) -> None:
        widget = self.query_one("#beam-input", TextArea)
        if self.accepting_input and not widget.text.strip():
            self.submit(BeamInput("resume" if self.state.at_edge else "advance 1", self.selected_label))
        elif self.accepting_input:
            widget.action_cursor_right()

    def action_rewind(self) -> None:
        widget = self.query_one("#beam-input", TextArea)
        if self.accepting_input and not widget.text.strip():
            self.submit(BeamInput("resume" if self.state.at_edge else "rewind", self.selected_label))
        elif self.accepting_input:
            widget.action_cursor_left()

    def action_return_to_choice(self) -> None:
        if self.accepting_input:
            self.submit(BeamInput("return", self.selected_label))

    def apply_palette_command(self, command: str) -> None:
        widget = self.query_one("#beam-input", TextArea)
        widget.text = command
        widget.focus()


class _SingleKeyHint(Static):
    """Focusable target for single-key requests, which have no editor widget."""

    can_focus = True


class PromptScreen(_RequestScreen):
    """One parameterized screen for ordinary, key, multiline, page, and chord input."""

    BINDINGS: ClassVar[list[Binding]] = [
        *_RequestScreen.BINDINGS,
        Binding("enter", "enter_prompt", "Submit", show=False, priority=True),
        Binding("escape", "escape_prompt", "Cancel", show=False, priority=True),
        Binding("ctrl+d", "cancel_prompt", "Cancel", show=False, priority=True),
        Binding("backspace", "single_backspace", "Backspace", show=False),
        Binding("pageup", "page_up", "Scroll up", show=False, priority=True),
        Binding("pagedown", "page_down", "Scroll down", show=False, priority=True),
    ]
    VERTICAL_BREAKPOINTS: ClassVar[list[tuple[int, str]]] = [
        (0, "-short"), (18, "-regular"),
    ]

    def __init__(self, terminal_app: PolicyEditorApp, lifecycle: _RequestLifecycle) -> None:
        super().__init__(terminal_app, lifecycle)
        assert isinstance(lifecycle.state, PromptRequest)
        self.state = lifecycle.state
        self._escape_pending = False
        self._error = ""

    def compose_request(self) -> ComposeResult:
        group_classes = "page" if self.state.page else ""
        with Vertical(id="prompt-group", classes=group_classes):
            if self.state.page:
                yield VerticalScroll(
                    Static(Text(self.state.body), id="page-body"),
                    id="page-scroll",
                )
            else:
                if self.state.body:
                    yield VerticalScroll(Static(Text(self.state.body)), id="prompt-body")
                if self.state.multiline:
                    yield Static(
                        "Write the new prompt. Enter adds a line; Esc then Enter submits.",
                        id="prompt-instructions",
                        classes="muted",
                    )
                yield Static(self.state.prompt, id="prompt-label", classes="prompt-label")
                if not self.state.single_key:
                    if self.state.multiline:
                        yield TextArea(
                            "", id="multiline-input", soft_wrap=True,
                            tab_behavior="indent", placeholder="Write at least one character",
                        )
                    else:
                        yield Input("", id="prompt-input", placeholder="Response")
                else:
                    yield _SingleKeyHint("Press a key", id="single-key-hint")
            yield Static(id="prompt-status", classes="feedback-info")
        yield Static(id="hint", classes="hint")

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one("#prompt-status", Static).display = False
        if self.state.page:
            hint = "PgUp/PgDn scroll · Enter/Esc/q return · Ctrl+L output · F1 help"
        elif self.state.single_key:
            hint = "Press a key · Backspace returns DEL\nEsc returns ESC · Ctrl+D cancels · Ctrl+L output · F1 help"
        elif self.state.multiline:
            hint = "Enter adds a line · Esc then Enter submits\nCtrl+D cancels · Ctrl+L output · F1 help"
        else:
            hint = "Enter submits · Ctrl+D cancels · PgUp/PgDn scroll · Ctrl+L output · F1 help"
        self.query_one("#hint", Static).update(hint)
        self.call_after_refresh(self._focus_prompt_target)

    def _focus_prompt_target(self) -> None:
        if self.state.multiline:
            self.query_one("#multiline-input", TextArea).focus()
        elif self.state.page:
            self.query_one("#page-scroll", VerticalScroll).focus()
        elif self.state.single_key:
            self.query_one("#single-key-hint", _SingleKeyHint).focus()
        else:
            self.query_one("#prompt-input", Input).focus()

    def set_input_enabled(self, enabled: bool) -> None:
        if self.state.single_key or self.state.page:
            return
        selector, widget_type = (
            ("#multiline-input", TextArea)
            if self.state.multiline
            else ("#prompt-input", Input)
        )
        self.query_one(selector, widget_type).disabled = not enabled

    def on_key(self, event: events.Key) -> None:
        super().on_key(event)
        if not self.accepting_input:
            return
        if self.state.page and event.key == "q":
            self.submit("")
            event.stop()
            event.prevent_default()
            return
        if self.state.single_key:
            if event.key in {"escape", "ctrl+d", "ctrl+c", "backspace", "enter", "pageup", "pagedown", "ctrl+k", "f1"}:
                return
            if event.character is not None:
                self.submit(event.character)
            else:
                self.submit(event.key)
            event.stop()
            event.prevent_default()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "multiline-input":
            return
        if self._escape_pending:
            self._escape_pending = False
        status = self.query_one("#prompt-status", Static)
        if not status.display:
            return
        status.update("")
        status.display = False
        status.remove_class("feedback-error")
        status.add_class("feedback-info")

    def action_enter_prompt(self) -> None:
        if not self.accepting_input:
            return
        if self.state.page:
            self.submit("")
        elif self.state.single_key:
            self.submit("\n")
        elif self.state.multiline:
            widget = self.query_one("#multiline-input", TextArea)
            if self._escape_pending:
                self._escape_pending = False
                if not widget.text.strip():
                    self._error = "Write at least one character."
                    status = self.query_one("#prompt-status", Static)
                    status.remove_class("feedback-info")
                    status.add_class("feedback-error")
                    status.update(self._error)
                    status.display = True
                    return
                self.submit(widget.text)
            else:
                widget.insert("\n")
        else:
            self.submit(self.query_one("#prompt-input", Input).value)

    def action_escape_prompt(self) -> None:
        if not self.accepting_input:
            return
        if self.state.page:
            self.submit("")
        elif self.state.single_key:
            self.submit("\x1b")
        elif self.state.multiline:
            self._escape_pending = True
            status = self.query_one("#prompt-status", Static)
            status.remove_class("feedback-error")
            status.add_class("feedback-info")
            status.update("Escape pressed · press Enter to submit")
            status.display = True
        else:
            self.submit(None)

    def action_cancel_prompt(self) -> None:
        if self.accepting_input:
            self.submit(None)

    def action_single_backspace(self) -> None:
        if self.accepting_input and self.state.single_key:
            self.submit("\x7f")

    def _scroll_document(self, amount: int) -> None:
        if self.state.page:
            self.query_one("#page-scroll", VerticalScroll).scroll_relative(
                y=amount, animate=False, immediate=True,
            )
            return
        if self.state.body:
            self.query_one("#prompt-body", VerticalScroll).scroll_relative(
                y=amount, animate=False, immediate=True,
            )

    def action_page_up(self) -> None:
        if self.accepting_input:
            self._scroll_document(-10)

    def action_page_down(self) -> None:
        if self.accepting_input:
            self._scroll_document(10)

class HelpScreen(ModalScreen[None]):
    """Scrollable help overlay shared by every interactive screen."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("pageup", "scroll_up", "Scroll up", show=False, priority=True),
        Binding("pagedown", "scroll_down", "Scroll down", show=False, priority=True),
        Binding("escape", "close_help", "Close help", show=False, priority=True),
        Binding("q", "close_help", "Close help", show=False, priority=True),
        Binding("question_mark", "close_help", "Close help", show=False, priority=True),
    ]

    def __init__(self, document: Text) -> None:
        super().__init__()
        self.document = document

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield VerticalScroll(
                Static(self.document, id="help-body"),
                id="help-scroll",
            )
            yield Static("PgUp/PgDn scroll · Esc/q/? closes", id="hint", classes="hint")

    def action_scroll_up(self) -> None:
        self.query_one("#help-scroll", VerticalScroll).scroll_relative(
            y=-10, animate=False, immediate=True,
        )

    def action_scroll_down(self) -> None:
        self.query_one("#help-scroll", VerticalScroll).scroll_relative(
            y=10, animate=False, immediate=True,
        )

    def action_close_help(self) -> None:
        self.dismiss(None)


class OutputScreen(ModalScreen[None]):
    """Bounded captured output, opened deliberately without reflowing requests."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("pageup", "scroll_up", "Scroll up", show=False, priority=True),
        Binding("pagedown", "scroll_down", "Scroll down", show=False, priority=True),
        Binding("escape", "close_output", "Close output", show=False, priority=True),
        Binding("q", "close_output", "Close output", show=False, priority=True),
        Binding("ctrl+l", "close_output", "Close output", show=False, priority=True),
    ]

    def __init__(self, terminal_app: PolicyEditorApp) -> None:
        super().__init__()
        self.terminal_app = terminal_app

    def compose(self) -> ComposeResult:
        with Vertical(id="output-dialog"):
            yield Static("Captured output", classes="section")
            yield RichLog(
                id="output-body", max_lines=self.terminal_app.OUTPUT_HISTORY_LIMIT + 1,
                wrap=True,
                markup=False, highlight=False,
            )
            yield Static("PgUp/PgDn scroll · Esc/q/Ctrl+L closes", id="hint", classes="hint")

    def on_mount(self) -> None:
        self.refresh_output()

    def refresh_output(self) -> None:
        log = self.query_one("#output-body", RichLog)
        previous_scroll_y = log.scroll_y
        following_tail = previous_scroll_y >= log.max_scroll_y
        log.clear()
        history = "".join(self.terminal_app._output_chunks)
        if history:
            log.write(Text(history), scroll_end=False)
            self.terminal_app.stats["rich_log_writes"] += 1

        def restore_scroll() -> None:
            if following_tail:
                log.scroll_end(animate=False, immediate=True)
            else:
                log.scroll_to(
                    y=min(previous_scroll_y, log.max_scroll_y),
                    animate=False,
                    immediate=True,
                )

        self.call_after_refresh(restore_scroll)

    def append_output(self, text: str) -> None:
        log = self.query_one("#output-body", RichLog)
        was_at_tail = log.scroll_y >= log.max_scroll_y
        log.write(Text(text), scroll_end=was_at_tail)
        self.terminal_app.stats["rich_log_writes"] += 1

    def action_scroll_up(self) -> None:
        log = self.query_one("#output-body", RichLog)
        log.scroll_relative(y=-max(1, log.size.height - 1), animate=False, immediate=True)

    def action_scroll_down(self) -> None:
        log = self.query_one("#output-body", RichLog)
        log.scroll_relative(y=max(1, log.size.height - 1), animate=False, immediate=True)

    def action_close_output(self) -> None:
        self.dismiss(None)


class TextualTerminalSession(AbstractContextManager["TextualTerminalSession"]):
    """Run Textual in a dedicated thread while the caller owns engine work."""

    def __init__(
        self,
        *,
        theme: str = DEFAULT_LIVE_THEME,
        environment: dict[str, str] | None = None,
        terminal_output: Any = None,
    ) -> None:
        self.theme = theme
        self.environment = environment
        self.application: PolicyEditorApp | None = None
        self._ready: Future[None] = Future()
        self._thread: threading.Thread | None = None
        self._owner: int | None = None
        self._failure: BaseException | None = None
        self._closing = False
        self._generation = 0
        self._active_lifecycle: _RequestLifecycle | None = None
        self._previous_signal_handlers: dict[signal.Signals, Any] = {}
        self._terminal_output = terminal_output

    def __enter__(self) -> TextualTerminalSession:  # noqa: PYI034 - Python 3.10 support excludes typing.Self.
        if self._thread is not None:
            raise RuntimeError("terminal session cannot be entered twice")
        self._owner = threading.get_ident()
        self.application = PolicyEditorApp(
            theme=self.theme,
            environment=self.environment,
            ready=self._ready,
            terminal_output=self._terminal_output,
        )
        self._install_process_signal_handlers()
        self._thread = threading.Thread(target=self._run_app, name="spe-textual-terminal")
        self._thread.start()
        try:
            self._ready.result()
        except BaseException:
            self.__exit__(*__import__("sys").exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._closing = True
        app = self.application
        if app is not None and self._thread is not None and self._thread.is_alive():
            try:
                app.call_from_thread(app.exit)
            except RuntimeError:
                pass
        if self._thread is not None:
            self._thread.join()
        if app is not None:
            app.close_executors()
        self._restore_process_signal_handlers()
        if exc_type is None and self._failure is not None:
            raise self._failure
        return False

    def _install_process_signal_handlers(self) -> None:
        if os.name != "posix" or threading.current_thread() is not threading.main_thread():
            return

        def dispatch(method_name: str) -> None:
            app = self.application
            if app is None or not app.is_running:
                return
            driver = app._driver
            if not isinstance(driver, _ThreadedLinuxDriver):
                return
            try:
                app.call_from_thread(getattr(driver, method_name))
            except RuntimeError:
                # The app may be exiting as this signal is delivered.
                pass

        def stop_foreground_job(_signum: int, _frame: Any) -> None:
            os.kill(os.getpid(), signal.SIGSTOP)

        def suspend(_signum: int, _frame: Any) -> None:
            dispatch("_sigtstp_application")

        def resume(_signum: int, _frame: Any) -> None:
            dispatch("_sigcont_application")

        def resize(_signum: int, _frame: Any) -> None:
            dispatch("_send_terminal_resize")

        handlers = (
            (signal.SIGTSTP, suspend),
            (signal.SIGCONT, resume),
            (signal.SIGWINCH, resize),
            (signal.SIGTTIN, stop_foreground_job),
            (signal.SIGTTOU, stop_foreground_job),
        )
        try:
            for signum, handler in handlers:
                self._previous_signal_handlers[signum] = signal.signal(signum, handler)
        except BaseException:
            self._restore_process_signal_handlers()
            raise

    def _restore_process_signal_handlers(self) -> None:
        if threading.current_thread() is threading.main_thread():
            for signum, handler in self._previous_signal_handlers.items():
                signal.signal(signum, handler)
        self._previous_signal_handlers.clear()

    def _run_app(self) -> None:
        try:
            assert self.application is not None
            asyncio.run(self.application.run_async())
        except BaseException as exc:  # noqa: BLE001 - signal app failure to every blocked engine request.
            self._failure = exc
            if not self._ready.done():
                self._ready.set_exception(exc)
            lifecycle = self._active_lifecycle
            if lifecycle is not None and not lifecycle.response.done():
                lifecycle.response.set_exception(exc)

    def _read(self, state: ChoiceViewState | EdgeViewState | BeamViewState | PromptRequest) -> Any:
        if threading.get_ident() != self._owner:
            raise RuntimeError("terminal requests must come from the engine-owning thread")
        if self._closing or self._thread is None or not self._thread.is_alive():
            raise self._failure or EOFError("terminal session is closed")
        self._generation += 1
        lifecycle = _RequestLifecycle(
            generation=self._generation,
            state=state,
            response=Future(),
            owner_queue=Queue(),
        )
        self._active_lifecycle = lifecycle
        assert self.application is not None
        try:
            self.application.call_from_thread(self.application.show_request, lifecycle)
            while True:
                try:
                    result = lifecycle.response.result(timeout=0.05)
                    break
                except TimeoutError:
                    if self._failure is not None:
                        raise self._failure
                    try:
                        assert lifecycle.owner_queue is not None
                        preview = lifecycle.owner_queue.get_nowait()
                    except Empty:
                        if not self._thread.is_alive():
                            raise self._failure or EOFError("terminal input closed")
                        continue
                    try:
                        value = preview.callback()
                        error = None
                    except BaseException as exc:  # noqa: BLE001 - preserve callback failures for UI delivery.
                        value, error = None, exc
                    try:
                        self.application.call_from_thread(
                            self.application.deliver_owner_preview,
                            preview.generation,
                            preview.key,
                            value,
                            error,
                        )
                    except RuntimeError:
                        if not lifecycle.response.done():
                            raise
            return result
        finally:
            self._finish_warm(lifecycle)
            self._active_lifecycle = None

    def _finish_warm(self, lifecycle: _RequestLifecycle) -> None:
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
                except Exception:  # noqa: BLE001 - a failed warm search is a non-fatal cache miss.
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
            except Exception:  # noqa: BLE001 - warm cancellation failures do not change the submitted command.
                _LOG.warning("warm search failed while cancelling", exc_info=True)
        cancel = state.cancel_search_warm
        if callable(cancel) and self.application is not None:
            self.application._warm_executor.submit(cancel).result()

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
        try:
            app.call_from_thread(app.write_output, text + end)
        except RuntimeError:
            return

    def page(self, text: str) -> None:
        self.prompt(PromptRequest("", body=text, page=True))

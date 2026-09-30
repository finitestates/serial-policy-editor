"""Exercise the custom Textual driver through real POSIX terminal input."""

from __future__ import annotations

import base64
import codecs
import errno
import fcntl
import json
import os
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import pyte
import pytest

pytestmark = pytest.mark.current_workflow
_FRAME_SNAPSHOT_DIR = Path(__file__).with_name("textual_migration_snapshots")

_CHILD = r'''
import json
import sys
import threading
import time
from pathlib import Path

from textual.widgets import DataTable, Input, TextArea

from trajectory_editor.terminal_contracts import (
    BeamInput,
    BeamViewRow,
    BeamViewState,
    PromptRequest,
)
from trajectory_editor.textual_tui import (
    BeamScreen,
    HelpScreen,
    PolicyEditorApp,
    PromptScreen,
    TextualTerminalSession,
)

result_path = Path(sys.argv[1])
snapshot_path = result_path.with_suffix(".snapshot.json")
mode = sys.argv[2]

CAPTURE_MARKER = "\x1b]777;SPEPTY;"
DISPLAY_BOUNDARY_SENTINEL = "\x1b]777;SPEFRAME\x07"


class CaptureOutput:
    """Forward terminal output and add invisible, ordered capture markers."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sequence = 0
        self._generation = lambda: 0

    def write(self, data):
        with self._lock:
            if DISPLAY_BOUNDARY_SENTINEL in data:
                visible = data.replace(DISPLAY_BOUNDARY_SENTINEL, "")
                if visible:
                    written = sys.__stderr__.write(visible)
                    sys.__stderr__.flush()
                else:
                    written = len(data)
                self._mark_locked("display", self._generation())
                return written
            written = sys.__stderr__.write(data)
            sys.__stderr__.flush()
            self._mark_locked("write", self._generation())
        return written

    def flush(self):
        sys.__stderr__.flush()

    def mark(self, event, generation):
        with self._lock:
            self._mark_locked(event, generation)

    def _mark_locked(self, event, generation):
        self._sequence += 1
        marker = f"{CAPTURE_MARKER}{self._sequence};{generation};{event}\x07"
        sys.__stderr__.write(marker)
        sys.__stderr__.flush()


capture_output = CaptureOutput()


_original_post_display_hook = PolicyEditorApp.post_display_hook
def _capture_compositor_boundary(app):
    _original_post_display_hook(app)
    if app.is_running and app._driver is not None:
        # Use the same WriterThread queue as compositor output so this marker
        # denotes a completed Textual display write, unlike a parent-side read
        # offset or an out-of-band test marker.
        app._driver.write(DISPLAY_BOUNDARY_SENTINEL)


PolicyEditorApp.post_display_hook = _capture_compositor_boundary


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


if mode == "interrupt":
    with TextualTerminalSession() as terminal:
        try:
            terminal.prompt(PromptRequest("Interrupt now? "))
        except KeyboardInterrupt:
            interrupted = True
        else:
            interrupted = False
        app = terminal.application
    save_json(result_path, {"interrupted": interrupted, "stats": dict(app.stats)})
else:
    rows = (
        BeamViewRow(
            "b1", "alpha continuation", "-0.45", "LIVE", ("alpha",),
            model_rank=1,
        ),
        BeamViewRow(
            "b2", "beta continuation", "-1.25", "LIVE",
            tuple(f"generated step {index}" for index in range(80)),
            model_rank=2,
        ),
    )
    state = BeamViewState(
        "BEAM · width 2 · depth 1", "PTY generated context", rows, "b1",
    )
    observations = []
    stop_watcher = threading.Event()
    ready_generations = set()
    last_top_screen = [None]

    terminal = TextualTerminalSession(terminal_output=capture_output)
    with terminal:
        app = terminal.application
        capture_output._generation = lambda: (
            terminal._active_lifecycle.generation
            if terminal._active_lifecycle is not None else 0
        )

        def inspect_screen():
            screen = app._active_screen
            if isinstance(screen, BeamScreen) and screen.accepting_input:
                driver = app._driver
                table = screen.query_one("#beam-table", DataTable)
                pane = screen.query_one("#beam-detail-pane")
                editor = screen.query_one("#beam-input", TextArea)
                return {
                    "kind": "beam",
                    "generation": screen.lifecycle.generation,
                    "top_screen": type(app.screen).__name__,
                    "size": [app.size.width, app.size.height],
                    "driver_size": list(driver._get_terminal_size()),
                    "table": [
                        table.region.x,
                        table.region.y,
                        table.region.width,
                        table.region.height,
                        table.header_height,
                    ],
                    "editor": [
                        editor.region.x,
                        editor.region.y,
                        editor.region.width,
                        editor.region.height,
                    ],
                    "pane": [
                        pane.region.x,
                        pane.region.y,
                        pane.region.width,
                        pane.region.height,
                    ],
                    "selected": screen.selected_label,
                    "scroll_y": pane.scroll_y,
                    "max_scroll_y": pane.max_scroll_y,
                    "stacked": screen.has_class("-stacked"),
                    "side_by_side": screen.has_class("-side-by-side"),
                    "table_focused": table.has_focus,
                    "editor_focused": editor.has_focus,
                    "command_text": editor.text,
                    "detail_renders": app.stats["beam_detail_renders"],
                    "layout_refreshes": app.stats["screen_layout_refreshes"],
                    "help_scroll_y": (
                        app.screen.query_one("#help-scroll").scroll_y
                        if isinstance(app.screen, HelpScreen)
                        else None
                    ),
                    "help_max_scroll_y": (
                        app.screen.query_one("#help-scroll").max_scroll_y
                        if isinstance(app.screen, HelpScreen)
                        else None
                    ),
                    "help_scroll_region": (
                        [
                            app.screen.query_one("#help-scroll").region.x,
                            app.screen.query_one("#help-scroll").region.y,
                            app.screen.query_one("#help-scroll").region.width,
                            app.screen.query_one("#help-scroll").region.height,
                        ]
                        if isinstance(app.screen, HelpScreen)
                        else None
                    ),
                }
            if isinstance(screen, PromptScreen) and screen.accepting_input:
                editor = screen.query_one("#prompt-input", Input)
                return {
                    "kind": "prompt",
                    "top_screen": type(app.screen).__name__,
                    "size": [app.size.width, app.size.height],
                    "text": editor.value,
                    "editor_focused": editor.has_focus,
                }
            return None

        def watch_screen():
            while not stop_watcher.is_set():
                try:
                    current = app.call_from_thread(inspect_screen)
                except RuntimeError:
                    return
                except BaseException as error:
                    save_json(snapshot_path, {"watch_error": repr(error)})
                    return
                if current is not None:
                    observations.append(current)
                    save_json(snapshot_path, {"observations": observations})
                    top_screen = current.get("top_screen")
                    if top_screen != last_top_screen[0]:
                        if top_screen == "HelpScreen":
                            capture_output.mark(
                                "overlay-open", current.get("generation", 0),
                            )
                        elif last_top_screen[0] == "HelpScreen":
                            capture_output.mark(
                                "overlay-close", current.get("generation", 0),
                            )
                        last_top_screen[0] = top_screen
                    if current.get("kind") == "beam":
                        generation = current["generation"]
                        if generation not in ready_generations:
                            ready_generations.add(generation)
                            capture_output.mark("ready", generation)
                time.sleep(.02)

        watcher = threading.Thread(target=watch_screen, name="pty-state-observer")
        watcher.start()
        beam_result = terminal.read_beam(state)
        terminal.write("PTY_CAPTURE_MARKER")
        prompt_result = terminal.prompt(PromptRequest("Finish? "))
        stop_watcher.set()
        watcher.join(timeout=2)

    save_json(
        result_path,
        {
            "beam": {
                "command": beam_result.command,
                "selected_label": beam_result.selected_label,
            },
            "prompt": prompt_result,
            "stats": dict(app.stats),
            "observations": observations,
        },
    )
'''


_CHOICE_CHILD = r'''
import json
import sys
import threading
import time
from pathlib import Path

from tests.core.textual_support import choice_state
from trajectory_editor.textual_tui import ChoiceScreen, TextualTerminalSession

result_path = Path(sys.argv[1])
snapshot_path = result_path.with_suffix(".snapshot.json")
observations = []
stop_watcher = threading.Event()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


with TextualTerminalSession() as terminal:
    app = terminal.application

    def inspect_screen():
        screen = app._active_screen
        if not isinstance(screen, ChoiceScreen) or not screen.accepting_input:
            return None
        editor = screen.query_one("#choice-input")
        table = screen.query_one("#choice-table")
        return {
            "kind": "choice",
            "generation": screen.lifecycle.generation,
            "text": editor.text,
            "editor_focused": editor.has_focus,
            "read_only": editor.read_only,
            "table": [
                table.region.x,
                table.region.y,
                table.region.width,
                table.region.height,
                table.header_height,
            ],
        }

    def watch_screen():
        previous = None
        while not stop_watcher.is_set():
            try:
                current = app.call_from_thread(inspect_screen)
            except RuntimeError:
                return
            if current is not None and current != previous:
                observations.append(current)
                save_json(snapshot_path, {"observations": observations})
                previous = current
            time.sleep(.02)

    watcher = threading.Thread(target=watch_screen, name="choice-pty-observer")
    watcher.start()
    command = terminal.read_choice(choice_state())
    stop_watcher.set()
    watcher.join(timeout=2)

save_json(
    result_path,
    {"command": command, "observations": observations, "stats": dict(app.stats)},
)
'''

_EDGE_CHILD = r'''
import json
import sys
from pathlib import Path

from tests.core.textual_support import edge_state
from trajectory_editor.edge_commands import parse_edge_command
from trajectory_editor.textual_tui import EdgeScreen, TextualTerminalSession

result_path = Path(sys.argv[1])
snapshot_path = result_path.with_suffix(".snapshot.json")
observations = []

with TextualTerminalSession() as terminal:
    app = terminal.application
    screen = None

    def inspect_screen():
        nonlocal_screen = app._active_screen
        if not isinstance(nonlocal_screen, EdgeScreen) or not nonlocal_screen.accepting_input:
            return None
        editor = nonlocal_screen.query_one("#edge-input")
        table = nonlocal_screen.query_one("#edge-commands")
        return {
            "kind": "edge",
            "text": editor.text,
            "editor_focused": editor.has_focus,
            "read_only": editor.read_only,
            "table": [
                table.region.x, table.region.y, table.region.width,
                table.region.height, table.header_height,
                [str(key.value) for key in table.rows],
            ],
        }

    import threading
    import time
    stop = threading.Event()

    def watch():
        previous = None
        while not stop.is_set():
            try:
                current = app.call_from_thread(inspect_screen)
            except RuntimeError:
                return
            if current is not None and current != previous:
                observations.append(current)
                temporary = snapshot_path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"observations": observations}), encoding="utf-8")
                temporary.replace(snapshot_path)
                previous = current
            time.sleep(.02)

    watcher = threading.Thread(target=watch, name="edge-pty-observer")
    watcher.start()
    command = terminal.read_edge(edge_state(mode="session"))
    stop.set()
    watcher.join(timeout=2)
    parsed = parse_edge_command(command)

result_path.write_text(
    json.dumps({"command": command, "parsed": str(parsed), "observations": observations}),
    encoding="utf-8",
)
'''

_RUNTIME_CHILD = r'''
import contextlib
import base64
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests.fakes import ConformingFakeBackend
from trajectory_editor.beam import BeamSearch
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_session import LiveSession, LiveSessionRoster
from trajectory_editor.session_runtime import run_session_roster
from trajectory_editor.textual_tui import (
    BeamScreen, ChoiceScreen, EdgeScreen, PromptScreen, TextualTerminalSession,
    PolicyEditorApp,
)
from trajectory_editor.tui import TerminalIO
from textual._ansi_sequences import SYNC_END, SYNC_START

result_path = Path(sys.argv[1])
mode = sys.argv[2]
snapshot_path = result_path.with_suffix(".snapshot.json")
arm_path = result_path.with_suffix(".arm")
waiting_path = result_path.with_suffix(".waiting")
release_path = result_path.with_suffix(".release")
observations = []
audit = {}
beam_inputs = []
mount_events = []

CAPTURE_MARKER = "\x1b]777;SPEPTY;"
FRAME_MARKER = re.compile(
    r"\x1b\]777;SPEFRAME;(\d+);(\d+);(\d+);(\d+);(\d+);([A-Za-z0-9_=-]+)\x07"
)
PHASE_MARKER = "\x1b]777;SPEPHASE;"
frame_path = result_path.with_suffix(".frames.json")


class CaptureOutput:
    """Write live PTY output and mark qualified compositor frames in order."""

    def __init__(self):
        self.lock = threading.Lock()
        self.sequence = 0
        self.frames = []

    def write(self, data):
        matches = list(FRAME_MARKER.finditer(data))
        with self.lock:
            written = sys.__stderr__.write(data)
            sys.__stderr__.flush()
            if not matches:
                self._mark_locked(0, "write")
            else:
                for match in matches:
                    generation, columns, rows, screen_id, characters = map(
                        int, match.groups()[:5],
                    )
                    frame = {
                        "generation": generation,
                        "size": [columns, rows],
                        "screen_id": screen_id,
                        "characters": characters,
                        "geometry": json.loads(
                            base64.urlsafe_b64decode(
                                match.group(6) + "=" * (-len(match.group(6)) % 4)
                            ).decode("utf-8")
                        ),
                    }
                    self._mark_locked(generation, "display")
                    frame["sequence"] = self.sequence
                    self.frames.append(frame)
                temporary = frame_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.frames), encoding="utf-8")
                temporary.replace(frame_path)
        return written

    def flush(self):
        sys.__stderr__.flush()

    def _mark_locked(self, generation, event):
        self.sequence += 1
        marker = f"{CAPTURE_MARKER}{self.sequence};{generation};{event}\x07"
        sys.__stderr__.write(marker)
        sys.__stderr__.flush()


capture_output = CaptureOutput()


if mode == "beam-no-flash":
    original_display = PolicyEditorApp._display
    original_post_display_hook = PolicyEditorApp.post_display_hook

    def qualified_post_display_hook(app):
        original_post_display_hook(app)
        before = getattr(app, "_beam_capture_before", 0)
        written = app.stats["driver_write_characters"] - before
        if app._sync_available:
            written -= len(SYNC_START) + len(SYNC_END)
        if written <= 0 or app._driver is None:
            return
        try:
            screen = app.screen
        except Exception:
            screen = None
        lifecycle = app._active_request
        generation = (
            screen.lifecycle.generation if isinstance(screen, BeamScreen)
            else lifecycle.generation if lifecycle is not None else 0
        )
        geometry = None
        if isinstance(screen, BeamScreen):
            table = screen.query_one("#beam-table")
            heading = screen.query_one("#beam-heading")
            detail = screen.query_one("#beam-detail")
            detail_pane = screen.query_one("#beam-detail-pane")
            editor = screen.query_one("#beam-input")
            command_bar = screen.query_one("#command-row")
            continuation_index = table.get_column_index("continuation")
            continuation = table._get_column_region(continuation_index)
            continuation_height = max(
                0,
                min(
                    continuation.height,
                    table.region.height - table.header_height,
                ),
            )
            continuation_index = table.get_column_index("continuation")
            continuation_column = table.columns["continuation"]
            rendered_continuations = {}
            for row in screen.state.rows:
                if row.state != "LIVE" or row.label not in table.rows:
                    continue
                rendered_lines = table._render_cell(
                    table.get_row_index(row.label),
                    continuation_index,
                    table.rich_style,
                    continuation_column.get_render_width(table),
                    cursor=False,
                    hover=False,
                )
                rendered_continuations[row.label] = [
                    "".join(segment.text for segment in line).strip()
                    for line in rendered_lines
                ]
            geometry = {
                "kind": "beam",
                "active": app._active_screen is screen,
                "accepting": screen.accepting_input,
                "app_focus": app.app_focus,
                "focused_widget": getattr(screen.focused, "id", None),
                "editor_focused": editor.has_focus,
                "cursor_screen_offset": [
                    editor.cursor_screen_offset.x,
                    editor.cursor_screen_offset.y,
                ],
                "screen_generation": screen.lifecycle.generation,
                "title": screen.state.title,
                "selected_label": screen.selected_label,
                "heading_region": [
                    heading.region.x, heading.region.y,
                    heading.region.width, heading.region.height,
                ],
                "table_region": [
                    table.region.x, table.region.y,
                    table.region.width, table.region.height,
                ],
                "table_header_region": [
                    table.region.x, table.region.y,
                    table.region.width, table.header_height,
                ],
                "detail_region": [
                    detail_pane.region.x, detail_pane.region.y,
                    detail_pane.region.width, detail_pane.region.height,
                ],
                "detail_content_region": [
                    detail.region.x, detail.region.y,
                    detail.region.width, detail.region.height,
                ],
                "editor_region": [
                    editor.region.x, editor.region.y,
                    editor.region.width, editor.region.height,
                ],
                "command_bar_region": [
                    command_bar.region.x, command_bar.region.y,
                    command_bar.region.width, command_bar.region.height,
                ],
                "continuation_region": [
                    table.region.x + continuation.x,
                    table.region.y + table.header_height,
                    continuation.width,
                    continuation_height,
                ],
                "table_header_height": table.header_height,
                "table_scroll_y": int(round(table.scroll_y)),
                "continuation_width": table.columns[
                    "continuation"
                ].get_render_width(table),
                "column_layout": {
                    (column.label.plain or "marker"): {
                        "width": column.width,
                        "content_width": column.content_width,
                        "auto_width": column.auto_width,
                        "render_width": column.get_render_width(table),
                    }
                    for column in table.columns.values()
                },
                "row_label_column_width": table._row_label_column_width,
                "scrollable_content_region": [
                    table.scrollable_content_region.x,
                    table.scrollable_content_region.y,
                    table.scrollable_content_region.width,
                    table.scrollable_content_region.height,
                ],
                "cell_padding": table.cell_padding,
                "live_labels": [
                    row.label for row in screen.state.rows if row.state == "LIVE"
                ],
                "row_order": [row.label for row in screen.state.rows],
                "row_heights": {
                    row.label: table.rows[row.label].height
                    for row in screen.state.rows
                    if row.label in table.rows
                },
                "row_states": {
                    row.label: row.state for row in screen.state.rows
                },
                "live_continuations": {
                    row.label: row.continuation
                    for row in screen.state.rows if row.state == "LIVE"
                },
                "rendered_continuations": rendered_continuations,
                    "live_row_heights": {
                    row.label: table.rows[row.label].height
                    for row in screen.state.rows
                    if row.state == "LIVE" and row.label in table.rows
                    },
                "live_protected": {
                    row.label: row.protected
                    for row in screen.state.rows if row.state == "LIVE"
                },
                "show_family_metadata": screen.state.show_family_metadata,
            }
        geometry_payload = base64.urlsafe_b64encode(
            json.dumps(geometry, separators=(",", ":")).encode("utf-8")
        ).decode("ascii").rstrip("=")
        marker = (
            f"\x1b]777;SPEFRAME;{generation};{app.size.width};"
            f"{app.size.height};{id(app.screen)};{max(0, written)};"
            f"{geometry_payload}\x07"
        )
        # The frame metadata is captured on the UI thread and follows the
        # compositor bytes in Textual's own WriterThread queue.
        app._driver.write(marker)

    def qualified_display(app, screen, renderable):
        app._beam_capture_before = app.stats["driver_write_characters"]
        original_display(app, screen, renderable)

    PolicyEditorApp.post_display_hook = qualified_post_display_hook
    PolicyEditorApp._display = qualified_display

    original_beam_mount = BeamScreen.on_mount

    def counted_beam_mount(screen):
        mount_event = {
            "event": "mount",
            "order": len(mount_events),
            "screen_id": id(screen),
            "generation": screen.lifecycle.generation,
        }
        mount_events.append(mount_event)
        original_beam_mount(screen)

    BeamScreen.on_mount = counted_beam_mount
    original_beam_unmount = getattr(BeamScreen, "on_unmount", None)

    def counted_beam_unmount(screen):
        mount_events.append({
            "event": "unmount",
            "order": len(mount_events),
            "screen_id": id(screen),
            "generation": screen.lifecycle.generation,
        })
        if original_beam_unmount is not None:
            original_beam_unmount(screen)

    BeamScreen.on_unmount = counted_beam_unmount

    original_app_on_event = PolicyEditorApp.on_event

    async def repaint_rejected_waiting_input(app, event):
        await original_app_on_event(app, event)
        # Force a real, test-only compositor pass after each rejected input so
        # the gate assertion examines painted states, not elapsed time or one
        # stale snapshot from before the backend wait.
        if waiting_path.exists() and not release_path.exists() and app._active_screen is None:
            app.refresh()

    PolicyEditorApp.on_event = repaint_rejected_waiting_input

original_read_beam = TextualTerminalSession.read_beam
def audited_read_beam(target, state):
    value = original_read_beam(target, state)
    beam_inputs.append({
        "command": value.command if value is not None else None,
        "selected_label": value.selected_label if value is not None else None,
        "state_selected_label": state.selected_label,
        "at_edge": state.at_edge,
        "stochastic": state.stochastic,
        "title": state.title,
        "rows": [
            {
                "label": row.label,
                "continuation": row.continuation,
                "state": row.state,
                "score": row.score,
                "protected": row.protected,
                "family_metadata": row.family_metadata,
            }
            for row in state.rows
        ],
    })
    return value
TextualTerminalSession.read_beam = audited_read_beam

original_beam_view_state = BeamSearch.view_state
def audited_beam_view_state(target, *, notice="", at_edge=False):
    state = original_beam_view_state(target, notice=notice, at_edge=at_edge)
    rows_by_label = {row.label: row for row in state.rows}
    audit.setdefault("runtime_beam_states", []).append({
        "title": state.title,
        "selected_label": state.selected_label,
        "at_edge": state.at_edge,
        "stochastic": state.stochastic,
        "show_family_metadata": state.show_family_metadata,
        "base_visible": list(target.base_visible),
        "base_prefix": list(target.base_prefix),
        "paths": [
            {
                "label": path.label,
                "token_ids": list(target._path_token_ids(path)),
                "state": rows_by_label[path.label].state,
                "search_state": path.state,
                "score": path.score,
                "continuation": rows_by_label[path.label].continuation,
                "protected": rows_by_label[path.label].protected,
                "family_metadata": rows_by_label[path.label].family_metadata,
            }
            for path in target.ordered_paths()
        ],
    })
    return state
BeamSearch.view_state = audited_beam_view_state


class GatedBackend(ConformingFakeBackend):
    pieces = {
        0: "<EOG>",
        1: " amber continuation words wrap across the beam pane",
        2: " bronze continuation also wraps across the narrow terminal",
        3: " cedar continuation remains a distinct surviving branch",
        4: " hello",
        5: "!",
        6: "Q",
        7: "P",
    }
    app = None

    def tokenize(self, text, *, add_bos=False, special=False):
        if add_bos and text == "Q":
            return [6]
        return super().tokenize(text, add_bos=add_bos, special=special)

    def last_logits(self):
        logits = np.full(self.vocabulary_size(), -20.0, dtype=np.float32)
        if mode == "beam-no-flash":
            generated = self.tokens[1:]
            if not generated:
                logits[1], logits[2], logits[3], logits[0] = 10.0, 9.0, 8.0, -20.0
            elif len(generated) == 1:
                # Keep one live descendant from each root branch on the first
                # expansion so the next scoring round can deterministically
                # exchange their visible order.
                logits[1], logits[2], logits[3], logits[0] = 20.0, 0.0, -5.0, -40.0
            elif len(generated) == 2 and generated[0] == 1:
                logits[1], logits[2], logits[3], logits[0] = 0.0, 0.0, 0.0, -40.0
            elif len(generated) == 2 and generated[0] == 2:
                logits[1], logits[2], logits[3], logits[0] = 20.0, -20.0, -25.0, -40.0
            else:
                logits[1], logits[2], logits[3], logits[0] = 10.0, 9.0, 8.0, -20.0
        elif self.tokens[:1] == [6] and len(self.tokens) == 2:
            # The fresh Q root is tokenized to the fake question-mark token.
            # Keep its audited two-token result distinct from the first root.
            logits[2], logits[1], logits[3], logits[0] = 10.0, 9.0, 8.0, -20.0
        elif len(self.tokens) >= 3:
            logits[0] = 20.0
        else:
            logits[1], logits[2], logits[3], logits[0] = 10.0, 9.0, 8.0, -20.0
        return logits

    def eval(self, token_ids):
        if arm_path.exists() and not waiting_path.exists():
            if mode == "beam-no-flash" and self.app is not None:
                self.app.call_from_thread(
                    lambda: self.app._driver.write(
                        f"{PHASE_MARKER}WAIT\x07"
                    )
                )
            waiting_path.write_text("waiting", encoding="utf-8")
            if mode != "beam-no-flash":
                os.write(1, b"PTY_NATIVE_CAPTURE_MARKER")
            while not release_path.exists():
                time.sleep(.01)
            if mode == "beam-no-flash" and self.app is not None:
                self.app.call_from_thread(
                    lambda: self.app._driver.write(
                        f"{PHASE_MARKER}RELEASE\x07"
                    )
                )
        super().eval(token_ids)


def save_snapshot(value):
    temporary = snapshot_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(snapshot_path)


def inspect_screen(app):
    screen = app._active_screen
    if mode == "beam-no-flash" and screen is None:
        try:
            screen = app.screen
        except Exception:
            screen = None
    if screen is None:
        return {
            "kind": "handoff", "accepting": False,
            "stale_input": app.stats["stale_input_events"],
            "stale_keys": app.stats["stale_key_events"],
            "stale_mouse": app.stats["stale_mouse_events"],
        }
    if isinstance(screen, ChoiceScreen):
        kind, selector = "choice", "#choice-input"
        generation = screen.lifecycle.generation
        selected = screen._candidate_focus_rank
    elif isinstance(screen, EdgeScreen):
        kind, selector = "edge", "#edge-input"
        generation, selected = screen.lifecycle.generation, None
    elif isinstance(screen, BeamScreen):
        kind, selector = "beam", "#beam-input"
        generation, selected = screen.lifecycle.generation, screen.selected_label
    elif isinstance(screen, PromptScreen):
        kind = "prompt"
        selector = "#multiline-input" if screen.state.multiline else "#prompt-input"
        generation, selected = screen.lifecycle.generation, None
    else:
        return None
    try:
        editor = screen.query_one(selector)
    except Exception:
        # A screen enters the stack before compose/mount has attached its
        # request widgets; skip this intermediate observer sample.
        return None
    value = {
        "kind": kind,
        "generation": generation,
        "accepting": screen.accepting_input,
        "size": [app.size.width, app.size.height],
        "text": getattr(editor, "text", getattr(editor, "value", "")),
        "focused": editor.has_focus,
        "read_only": getattr(editor, "read_only", False),
        "disabled": getattr(editor, "disabled", False),
        "selected": selected,
    }
    if mode == "beam-no-flash" and isinstance(screen, BeamScreen):
        table = screen.query_one("#beam-table")
        detail = screen.query_one("#beam-detail")
        heading = screen.query_one("#beam-heading")
        value.update({
            "active": app._active_screen is screen,
            "screen_id": id(screen),
            "table_id": id(table),
            "detail_id": id(detail),
            "editor_id": id(editor),
            "regions": {
                "heading": [
                    screen.query_one("#beam-heading").region.x,
                    screen.query_one("#beam-heading").region.y,
                    screen.query_one("#beam-heading").region.width,
                    screen.query_one("#beam-heading").region.height,
                ],
                "table": [table.region.x, table.region.y, table.region.width, table.region.height],
                "detail": [detail.region.x, detail.region.y, detail.region.width, detail.region.height],
                "editor": [editor.region.x, editor.region.y, editor.region.width, editor.region.height],
            },
            "editor_region": [
                editor.region.x, editor.region.y,
                editor.region.width, editor.region.height,
            ],
            "heading": screen.state.title,
            "at_edge": screen.state.at_edge,
            "show_family_metadata": screen.state.show_family_metadata,
            "rows": [
                {
                    "label": row.label,
                    "continuation": row.continuation,
                    "score": row.score,
                    "state": row.state,
                    "protected": row.protected,
                    "family_metadata": row.family_metadata,
                }
                for row in screen.state.rows
            ],
            "table_scores": [
                str(table.get_cell(row.label, "score"))
                for row in screen.state.rows
                if row.label in table.rows
            ],
            "mount_events": list(mount_events),
            "stale_input": app.stats["stale_input_events"],
            "stale_keys": app.stats["stale_key_events"],
            "heading_render": str(heading.render()),
            "detail_render": str(detail.render()),
        })
    return value


io = TerminalIO()
if mode == "beam-no-flash":
    @contextlib.contextmanager
    def direct_textual_session():
        terminal = TextualTerminalSession(terminal_output=capture_output)
        io._live_session = terminal
        try:
            with terminal:
                yield terminal
        finally:
            io._live_session = None

    io.session = direct_textual_session

backend = GatedBackend()
engine = EpisodeEngine(
    backend,
    initial_token_ids=[7],
    sampling=SamplerConfig(temperature=0.0),
)
session = LiveSession(engine)
roster = LiveSessionRoster(session)
args = SimpleNamespace(
    divergence_policy="handoff", table_depth=3, search_radius=3,
    hold_default=20, context_chars=0, manual_acceptance=False,
    show_policy_rank=None, logit_view="none", output=None,
    phrase_max_tokens=16, phrase_max_shift=6.0,
)

original_discard = LiveSessionRoster.discard
def audited_discard(target):
    audit["branches"] = [
        {
            "prompt": entry.session.prompt,
            "branch_id": entry.branch_id,
            "boundary": entry.session.branch_state(entry.branch_id).boundary,
            "tokens": list(entry.session.branch_state(entry.branch_id).visible_token_ids),
            "actions": [type(step.action).__name__ for step in entry.session.branch_state(entry.branch_id).tape],
        }
        for entry in target.entries()
    ]
    original_discard(target)
LiveSessionRoster.discard = audited_discard

with io.session() as terminal:
    app = terminal.application
    backend.app = app
    stopped = threading.Event()

    def watch():
        previous = None
        while not stopped.is_set():
            try:
                current = app.call_from_thread(lambda: inspect_screen(app))
            except RuntimeError:
                return
            if current is not None and current != previous:
                observations.append(current)
                save_snapshot({"observations": observations})
                previous = current
            time.sleep(.01)

    watcher = threading.Thread(target=watch, name="runtime-pty-observer")
    watcher.start()
    result = run_session_roster(
        args, io=io, roster=roster, backend_provenance={"backend": "fake"},
    )
    stopped.set()
    watcher.join(timeout=2)
    audit["return_code"] = result
    audit["stats"] = dict(app.stats)
    audit["sync_available"] = app._sync_available
    audit["beam_inputs"] = beam_inputs
    audit["mount_events"] = list(mount_events)
    audit["capture_frames"] = list(capture_output.frames)

result_path.write_text(
    json.dumps({"audit": audit, "observations": observations}), encoding="utf-8",
)
'''


def _set_size(fd: int, columns: int, rows: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))


def _pump(master: int, output: bytearray, timeout: float = .05) -> None:
    readable, _, _ = select.select([master], [], [], timeout)
    if not readable:
        return
    try:
        chunk = os.read(master, 65536)
    except OSError as error:
        if error.errno == errno.EIO:
            return
        raise
    output.extend(chunk)


def _answer_synchronized_output_probe(
    master: int,
    process: subprocess.Popen,
    output: bytearray,
    timeout: float = 8,
) -> None:
    """Answer Textual's DCS probe with synchronized output recognized/reset."""
    query = b"\x1b[?2026$p"
    reply = b"\x1b[?2026;2$y"
    deadline = time.monotonic() + timeout
    while query not in output and process.poll() is None and time.monotonic() < deadline:
        _pump(master, output, .05)
    assert query in output, (
        "Textual did not issue its synchronized-output query before the runtime journey; "
        f"returncode={process.poll()}\n"
        + output[-4000:].decode("utf-8", errors="replace")
    )
    _send(master, reply)


def _drain_available(master: int, output: bytearray) -> None:
    """Read PTY bytes already available before recording an external resize."""
    while True:
        readable, _, _ = select.select([master], [], [], 0)
        if not readable:
            return
        try:
            output.extend(os.read(master, 65536))
        except OSError as error:
            if error.errno == errno.EIO:
                return
            raise


def _drain_to_quiet_display_boundary(master: int, output: bytearray) -> None:
    """Drain queued PTY output through a completed compositor display marker."""
    marker = re.compile(rb"\x1b\]777;SPEPTY;\d+;\d+;display\x07$")
    quiet_since = None
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        readable, _, _ = select.select([master], [], [], 0.02)
        if readable:
            _drain_available(master, output)
            quiet_since = None
            continue
        if quiet_since is None:
            quiet_since = time.monotonic()
        if time.monotonic() - quiet_since >= 0.04 and marker.search(output):
            return
    raise AssertionError("PTY output did not settle at a completed driver-write boundary")


def _resize_preserving_top(screen: pyte.Screen, rows: int, columns: int) -> None:
    """Control replay resize policy that clips the bottom, not the top."""
    old_lines = screen.lines
    old_buffer = screen.buffer
    new_buffer = defaultdict(
        lambda: pyte.screens.StaticDefaultDict(screen.default_char),
    )
    for row_index in range(min(old_lines, rows)):
        line = old_buffer[row_index]
        for column in range(columns, screen.columns):
            line.pop(column, None)
        new_buffer[row_index] = line
    screen.buffer = new_buffer
    screen.lines = rows
    screen.columns = columns
    screen.cursor.x = min(screen.cursor.x, columns - 1)
    screen.cursor.y = min(screen.cursor.y, rows - 1)
    screen.dirty = set(range(rows))
    screen.set_margins()


def _replay_terminal_frames(
    output: bytes,
    resizes,
    *,
    preserve_top_on_resize=False,
    style_region=None,
    style_regions=None,
    display_metadata=None,
    sync_available=False,
):
    """Replay ordered driver-write and resize boundaries from a PTY capture."""
    if not resizes or resizes[0][0] != 0:
        raise AssertionError("the PTY capture must start with its initial dimensions")
    columns, rows = resizes[0][1]
    screen = pyte.Screen(columns, rows)
    stream = pyte.Stream(screen)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    checkpoints = []
    segment_start = 0
    pending_csi = ""
    sync_active = False
    sync_bracket_counter = 0
    sync_open_bracket_id = None

    csi_pattern = re.compile(r"\x1b\[([0-?]*)([ -/]*)?([@-~])")

    def feed_decoded(decoded, *, final=False):
        """Feed text in order and attach cursor/sync state to every CSI op."""
        nonlocal pending_csi, sync_active
        nonlocal sync_bracket_counter, sync_open_bracket_id
        text = pending_csi + decoded
        pending_csi = ""
        cursor = 0
        operations = []
        for match in csi_pattern.finditer(text):
            prefix = text[cursor:match.start()]
            if prefix:
                stream.feed(prefix)
            parameters = match.group(1)
            intermediates = match.group(2) or ""
            final_byte = match.group(3)
            sync_before = sync_active
            bracket_id = sync_open_bracket_id
            if parameters == "?2026" and final_byte == "h":
                if not sync_active:
                    sync_bracket_counter += 1
                    sync_open_bracket_id = sync_bracket_counter
                sync_active = True
                bracket_id = sync_open_bracket_id
            elif parameters == "?2026" and final_byte == "l":
                bracket_id = sync_open_bracket_id
                sync_active = False
                sync_open_bracket_id = None
            operation = {
                "parameters": parameters,
                "intermediates": intermediates,
                "final": final_byte,
                "cursor_before": [screen.cursor.x, screen.cursor.y],
                "sync_available": bool(sync_available),
                "sync_active_before": sync_before,
                "sync_active_after": sync_active,
                "sync_bracket_id": bracket_id,
            }
            operations.append(operation)
            stream.feed(match.group(0))
            cursor = match.end()

        tail = text[cursor:]
        incomplete_start = tail.rfind("\x1b[")
        if incomplete_start >= 0 and not final:
            pending_csi = tail[incomplete_start:]
            tail = tail[:incomplete_start]
        if tail:
            stream.feed(tail)
        if final and pending_csi:
            stream.feed(pending_csi)
            pending_csi = ""
        return operations

    def feed_until(offset):
        nonlocal segment_start
        if offset < segment_start:
            raise AssertionError("PTY capture boundaries must be ordered")
        segment = output[segment_start:offset]
        decoded = decoder.decode(segment, final=False)
        operations = feed_decoded(decoded)
        segment_start = offset
        return operations

    def prompt_cell_styles():
        for row_index, row in enumerate(screen.display):
            column = row.find("Beam >")
            if column >= 0:
                return [
                    {
                        "data": cell.data,
                        "fg": cell.fg,
                        "bg": cell.bg,
                        "bold": cell.bold,
                        "reverse": cell.reverse,
                        "underscore": cell.underscore,
                    }
                    for cell in (
                        screen.buffer[row_index][column + offset]
                        for offset in range(len("Beam >"))
                    )
                ]
        return None

    def region_cell_styles(region):
        if region is None:
            return None
        x, y, width, _height = region
        if not (0 <= y < screen.lines and 0 <= x < screen.columns):
            return None
        return [
            {
                "data": screen.buffer[y][x + offset].data,
                "fg": screen.buffer[y][x + offset].fg,
                "bg": screen.buffer[y][x + offset].bg,
                "bold": screen.buffer[y][x + offset].bold,
                "reverse": screen.buffer[y][x + offset].reverse,
                "underscore": screen.buffer[y][x + offset].underscore,
            }
            for offset in range(min(max(0, width), 12, screen.columns - x))
        ]

    def named_region_styles(geometry=None):
        if geometry:
            regions = {
                name.removesuffix("_region"): tuple(geometry[name])
                for name in (
                    "heading_region", "table_region", "detail_region",
                    "table_header_region", "editor_region", "command_bar_region",
                    "continuation_region",
                )
                if geometry.get(name) is not None
            }
        elif style_regions:
            regions = style_regions.get(tuple(current_size), {})
        else:
            return {}
        return {
            name: region_cell_styles(region)
            for name, region in regions.items()
        }

    current_size = resizes[0][1]
    events = [
        (offset, 2, "resize", {"size": size})
        for offset, size in resizes[1:]
    ]
    marker_pattern = re.compile(
        rb"\x1b\]777;SPEPTY;(\d+);(\d+);([a-z-]+)\x07"
    )
    for match in marker_pattern.finditer(output):
        sequence = int(match.group(1))
        events.append((
            match.end(), 1, match.group(3).decode("ascii"),
            {
                "sequence": sequence,
                "generation": int(match.group(2)),
                "frame_metadata": (display_metadata or {}).get(sequence),
            },
        ))
    events.sort(key=lambda event: (event[0], event[1]))

    for offset, _order, kind, metadata in events:
        operations = feed_until(offset)
        if kind == "resize":
            current_size = metadata["size"]
            columns, rows = current_size
            if preserve_top_on_resize:
                _resize_preserving_top(screen, rows, columns)
            else:
                screen.resize(lines=rows, columns=columns)
        checkpoints.append({
            "kind": kind,
            "offset": offset,
            "size": current_size,
            "generation": metadata.get("generation"),
            "sequence": metadata.get("sequence"),
            "parser_ground": stream._taking_plain_text is True,
            "sync_available": bool(sync_available),
            "sync_active": sync_active,
            "beam_prompt_styles": prompt_cell_styles(),
            "beam_editor_styles": region_cell_styles(style_region),
            "region_styles": named_region_styles(
                (metadata.get("frame_metadata") or {}).get("geometry")
            ),
            "frame_metadata": metadata.get("frame_metadata"),
            "terminal_operations": operations,
            "grid": tuple(screen.display),
        })

    eof_operations = feed_until(len(output))
    final_text = decoder.decode(b"", final=True)
    eof_operations.extend(feed_decoded(final_text, final=True))
    checkpoints.append({
        "kind": "eof",
        "offset": len(output),
        "size": current_size,
        "generation": None,
        "sequence": None,
        "parser_ground": stream._taking_plain_text is True,
        "sync_available": bool(sync_available),
        "sync_active": sync_active,
        "beam_prompt_styles": prompt_cell_styles(),
        "beam_editor_styles": region_cell_styles(style_region),
        "region_styles": named_region_styles(),
        "frame_metadata": None,
        "terminal_operations": eof_operations,
        "grid": tuple(screen.display),
    })
    return checkpoints


_BEAM_IDENTITY_KEYS = ("screen_id", "table_id", "detail_id", "editor_id")


def _assert_stable_beam_identity(
    observations, expected, mount_events, *, through_generation=None,
    expect_final_unmount=False,
):
    identities = [
        tuple(item[key] for key in _BEAM_IDENTITY_KEYS)
        for item in observations
    ]
    assert len(identities) >= 2
    assert set(identities) == {tuple(expected[key] for key in _BEAM_IDENTITY_KEYS)}, (
        "same-kind Beam advancement remounted the request widgets: "
        f"identities={identities!r}"
    )
    mounts = [
        item for item in mount_events
        if item["event"] == "mount" and item["screen_id"] == expected["screen_id"]
    ]
    assert len(mounts) == 1, f"Beam screen mounted more than once: {mounts!r}"
    unmounts = [
        item for item in mount_events
        if item["event"] == "unmount" and item["screen_id"] == expected["screen_id"]
    ]
    if through_generation is not None:
        assert not [
            item for item in unmounts
            if item["generation"] <= through_generation
        ], f"Beam unmounted during advancement: {unmounts!r}"
    assert len(unmounts) <= int(expect_final_unmount), (
        f"unexpected Beam unmount sequence: {unmounts!r}"
    )
    assert all(item["order"] > mounts[0]["order"] for item in unmounts)


def _beam_style_signature(cells, *, ignore_indices=()):
    if not cells:
        return None
    return tuple(
        (cell["fg"], cell["bg"], cell["bold"], cell["reverse"], cell["underscore"])
        for index, cell in enumerate(cells)
        if index not in ignore_indices
    )


def _beam_region_style_cells(frame, region_name):
    geometry = frame.get("geometry") or {}
    cells = frame.get("region_styles", {}).get(region_name)
    region = geometry.get(f"{region_name}_region")
    cursor = geometry.get("cursor_screen_offset")
    ignored = ()
    if cells and region and cursor and geometry.get("editor_focused"):
        x, y, _width, height = region
        cursor_x, cursor_y = cursor
        cursor_index = cursor_x - x
        if y <= cursor_y < y + height and 0 <= cursor_index < len(cells):
            ignored = (cursor_index,)
    return [cell for index, cell in enumerate(cells or ()) if index not in ignored]


def _beam_region_style_signature(frame, region_name):
    return _beam_style_signature(_beam_region_style_cells(frame, region_name))


def _assert_focused_editor_styles(frames):
    signatures = [
        _beam_region_style_signature(frame, "editor")
        for frame in frames
    ]
    assert signatures and all(signature is not None for signature in signatures), (
        "editor style cells were not captured for every display pass"
    )
    geometry = [frame.get("geometry") or {} for frame in frames]
    observed_focus = [
        state["editor_focused"] for state in geometry
        if "editor_focused" in state
    ]
    assert observed_focus and all(observed_focus), (
        "a gated or ready Beam display pass lost command-editor focus"
    )
    assert len(set(signatures)) == 1, f"editor focus style changed: {signatures!r}"
    assert any(
        cell["bg"] not in {"default", "121212", None} or cell["bold"]
        for cell in _beam_region_style_cells(frames[0], "editor")
    ), "the Beam editor's focus styling was absent"


def _assert_stable_prompt_styles(frames):
    signatures = [
        _beam_style_signature(frame.get("beam_prompt_styles"))
        for frame in frames
    ]
    assert signatures and all(signature is not None for signature in signatures), (
        "Beam prompt style cells were not captured for every display pass"
    )
    assert len(set(signatures)) == 1, (
        f"Beam command prompt styles changed: {signatures!r}"
    )


def _assert_complete_beam_frame(frame, expected_screen_id):
    geometry = frame.get("geometry") or {}
    assert frame.get("screen_id") == expected_screen_id
    assert geometry.get("kind") == "beam"
    assert geometry.get("screen_generation") == frame.get("generation")
    grid = frame["grid"]

    def region_text(name):
        x, y, width, height = geometry[f"{name}_region"]
        return "\n".join(
            row[max(0, x):max(0, x) + max(0, width)]
            for row in grid[max(0, y):max(0, y) + max(0, height)]
        )

    heading = region_text("heading")
    detail = region_text("detail")
    continuation_x, _continuation_y, continuation_width, _ = geometry[
        "continuation_region"
    ]
    table_x, table_y, table_width, table_height = geometry["table_region"]
    header_height = geometry.get("table_header_height", 1)
    assert "BEAM" in heading, f"Beam heading vanished in frame {frame['sequence']}"
    assert "SELECTED:" in detail, f"Beam details vanished in frame {frame['sequence']}"
    assert f"SELECTED: {geometry['selected_label']}" in detail
    assert "Beam >" in "\n".join(grid), (
        f"Beam command bar vanished in frame {frame['sequence']}"
    )
    assert "command_bar_region" in geometry
    assert "table_header_region" in geometry
    assert continuation_width >= 8, (
        "continuation column collapsed during a completed display pass: "
        f"frame={frame['sequence']} size={frame['size']} width={continuation_width}"
    )
    assert table_x <= continuation_x
    assert continuation_x + continuation_width <= table_x + table_width
    assert table_height > 0 and table_width > 0
    columns, rows = frame["size"]
    for name in (
        "heading", "table", "table_header", "detail", "editor", "command_bar",
    ):
        x, y, width, height = geometry[f"{name}_region"]
        assert x >= 0 and y >= 0 and width > 0 and height > 0
        assert x + width <= columns and y + height <= rows, (
            f"{name} pane escaped the terminal in frame {frame['sequence']}: "
            f"region={(x, y, width, height)!r}, size={frame['size']!r}"
        )

    table_bottom = min(len(grid), table_y + table_height)
    table_body_y = table_y + header_height
    table_scroll_y = geometry.get("table_scroll_y", 0)
    live_labels = geometry.get("live_labels", [])
    assert live_labels, f"Beam has no live candidate in frame {frame['sequence']}"
    complete_rows = set()
    for label in live_labels:
        row_order = geometry.get("row_order", [])
        assert label in row_order, (
            f"live branch {label} has no captured row order in frame {frame['sequence']}"
        )
        row_heights = geometry.get("row_heights", {})
        assert label in row_heights, (
            f"live branch {label} has no captured row height in frame {frame['sequence']}"
        )
        row_index = row_order.index(label)
        row_y = table_body_y + sum(
            row_heights[previous] for previous in row_order[:row_index]
        ) - table_scroll_y
        row_height = row_heights[label]
        expected_continuation = " ".join(
            geometry["live_continuations"][label].replace("\n", " ↵ ").split()
        )
        rendered_lines = geometry.get("rendered_continuations", {}).get(label)
        assert rendered_lines is not None, (
            f"branch {label} has no Textual wrapped-row evidence in frame "
            f"{frame['sequence']}"
        )
        assert len(rendered_lines) == row_height, (
            f"branch {label} row height does not preserve its wrapped content "
            f"on frame {frame['sequence']}: height={row_height}, "
            f"rendered_lines={len(rendered_lines)}"
        )
        rendered_continuation = " ".join(" ".join(rendered_lines).split())
        assert rendered_continuation == expected_continuation, (
            f"branch {label} lost or changed wrapped content on frame "
            f"{frame['sequence']}: expected={expected_continuation!r}, "
            f"rendered={rendered_continuation!r}"
        )

        visible_top = max(table_body_y, row_y, 0)
        visible_bottom = min(table_bottom, row_y + row_height)
        if visible_top < visible_bottom:
            if table_body_y <= row_y < table_bottom:
                label_line = grid[row_y][table_x:table_x + table_width]
                assert label in label_line, (
                    f"visible live branch {label} is missing from its table row "
                    f"in frame {frame['sequence']}"
                )
            for screen_y in range(visible_top, visible_bottom):
                rendered_index = screen_y - row_y
                expected_line = rendered_lines[rendered_index].strip()
                painted_line = grid[screen_y][
                    continuation_x:continuation_x + continuation_width
                ].strip()
                assert painted_line == expected_line, (
                    f"branch {label} has an incorrect visible continuation slice "
                    f"on frame {frame['sequence']} at row {screen_y}: "
                    f"expected={expected_line!r}, painted={painted_line!r}"
                )
            if visible_top == row_y and visible_bottom == row_y + row_height:
                complete_rows.add(label)

    return complete_rows


def _assert_stable_beam_region_styles(frames):
    for region_name in ("heading", "table_header", "command_bar"):
        signatures = [
            _beam_region_style_signature(frame, region_name)
            for frame in frames
        ]
        assert signatures and all(signature is not None for signature in signatures), (
            f"{region_name} style cells were not captured on every frame"
        )
        assert len(set(signatures)) == 1, (
            f"Beam {region_name} styles changed across compositor frames"
        )


def _focused_beam_frames(frames):
    focused = [
        frame for frame in frames
        if (frame.get("geometry") or {}).get("app_focus")
        and (frame.get("geometry") or {}).get("editor_focused")
    ]
    assert focused, "no app-focused Beam compositor passes were captured"
    return focused


def _assert_fixed_beam_geometry(frames):
    by_size = defaultdict(list)
    for frame in frames:
        geometry = frame.get("geometry") or {}
        size = tuple(frame["size"])
        heading = geometry.get("heading_region", [0, 0, 0, 0])
        table = geometry.get("table_region", [0, 0, 0, 0])
        table_header = geometry.get("table_header_region", [0, 0, 0, 0])
        detail = geometry.get("detail_region", [0, 0, 0, 0])
        editor = geometry.get("editor_region", [0, 0, 0, 0])
        command_bar = geometry.get("command_bar_region", [0, 0, 0, 0])
        continuation = geometry.get("continuation_region", [0, 0, 0, 0])
        by_size[size].append((
            tuple(heading[:3]), tuple(table[:3]), tuple(table_header[:3]),
            tuple(detail[:3]), tuple(editor[:3]), tuple(command_bar[:3]),
            tuple(continuation[:3]),
        ))
    for size, signatures in by_size.items():
        assert signatures and len(set(signatures)) == 1, (
            f"Beam pane geometry changed at fixed terminal size {size}: "
            f"{signatures!r}"
        )


def _assert_no_beam_terminal_erases(
    ordered_events, display_frames, *, sync_available=None,
):
    """Reject unbracketed erases of preserved Beam panes, including repaints."""
    sequences = [frame.get("sequence") for frame in display_frames]
    assert sequences and all(isinstance(sequence, int) for sequence in sequences)
    first_sequence, last_sequence = min(sequences), max(sequences)
    if sync_available is None:
        sync_available = any(frame.get("sync_available") for frame in display_frames)
    if sync_available:
        open_markers = [
            frame.get("sequence") for frame in display_frames
            if frame.get("sync_active") is not False
        ]
        assert not open_markers, (
            "a compositor marker landed inside an open synchronized-output bracket: "
            f"{open_markers!r}"
        )
    journey_events = [
        event for event in ordered_events
        if isinstance(event.get("sequence"), int)
        and first_sequence <= event["sequence"] <= last_sequence
    ]
    opened_brackets = {
        operation.get("sync_bracket_id")
        for event in journey_events
        for operation in event.get("terminal_operations", ())
        if operation.get("parameters") == "?2026"
        and operation.get("final") == "h"
        and operation.get("sync_bracket_id") is not None
    }
    closed_brackets = {
        operation.get("sync_bracket_id")
        for event in journey_events
        for operation in event.get("terminal_operations", ())
        if operation.get("parameters") == "?2026"
        and operation.get("final") == "l"
        and operation.get("sync_bracket_id") is not None
    }
    erase_operations = []
    preserved_region_erases = []
    unbracketed_preserved_erases = []

    def stable_regions(geometry, event_size):
        if not geometry:
            return {}
        table = geometry.get("table_region")
        header = geometry.get("table_header_region")
        if header is None and table is not None:
            header = [
                table[0], table[1], table[2],
                geometry.get("table_header_height", 1),
            ]
        editor = geometry.get("editor_region")
        command_bar = geometry.get("command_bar_region")
        if command_bar is None and editor is not None:
            # Older retained captures predate the explicit command-bar field;
            # the command prompt occupies the editor's terminal row there.
            command_bar = [0, editor[1], event_size[0], editor[3]]
        return {
            name: tuple(region)
            for name, region in (
                ("heading", geometry.get("heading_region")),
                ("table_header", header),
                ("details", geometry.get("detail_region")),
                ("editor", editor),
                ("command_bar", command_bar),
            )
            if region is not None and len(region) == 4
        }

    def intersecting_regions(cursor, span, regions):
        _x, y = cursor
        left, right = span
        return [
            name for name, (region_x, region_y, width, height) in regions.items()
            if region_y <= y < region_y + height
            and left < region_x + width
            and region_x < right
        ]

    beam_frame_events = sorted(display_frames, key=lambda frame: frame["sequence"])
    def frame_for_sequence(sequence):
        previous = [frame for frame in beam_frame_events if frame["sequence"] <= sequence]
        return previous[-1] if previous else beam_frame_events[0]

    for event in journey_events:
        for operation in event.get("terminal_operations", ()):
            final = operation.get("final")
            parameters = operation.get("parameters", "")
            if final == "J" and parameters in {"", "0", "1", "2", "3"}:
                erase_operations.append({
                    "sequence": event["sequence"],
                    "kind": event["kind"],
                    "operation": operation,
                    "reason": "display-clear",
                })
                continue
            if not (
                (final == "K" and parameters in {"", "0", "1", "2"})
                or final in {"X", "P"}
            ):
                continue
            cursor = operation.get("cursor_before")
            if not cursor or len(cursor) != 2:
                erase_operations.append({
                    "sequence": event["sequence"],
                    "kind": event["kind"],
                    "operation": operation,
                    "reason": "erase-cursor-unavailable",
                })
                continue
            x, y = cursor
            columns = event.get("size", frame_for_sequence(event["sequence"])["size"])[0]
            if final == "K":
                mode = int(parameters or "0")
                span = (
                    (x, columns) if mode == 0
                    else (0, min(columns, x + 1)) if mode == 1
                    else (0, columns)
                )
            elif final == "X":
                count = int(parameters or "1")
                span = (x, min(columns, x + max(1, count)))
            else:  # CSI P deletes cells and shifts the remainder of the line.
                span = (x, columns)
            geometry_frame = frame_for_sequence(event["sequence"])
            regions = stable_regions(
                geometry_frame.get("geometry"), tuple(event.get("size", geometry_frame["size"])),
            )
            overlaps = intersecting_regions((x, y), span, regions)
            if not overlaps:
                erase_operations.append({
                    "sequence": event["sequence"],
                    "kind": event["kind"],
                    "operation": operation,
                    "reason": "changing-table-or-unpreserved-area",
                    "regions": [],
                    "sync_bracketed": False,
                })
                continue
            bracket_id = operation.get("sync_bracket_id")
            bracketed = (
                bool(sync_available)
                and operation.get("sync_active_before") is True
                and bracket_id in opened_brackets
                and bracket_id in closed_brackets
            )
            item = {
                "sequence": event["sequence"],
                "kind": event["kind"],
                "operation": operation,
                "regions": overlaps,
                "sync_bracketed": bracketed,
            }
            erase_operations.append(item)
            if bracketed:
                preserved_region_erases.append(item)
            else:
                unbracketed_preserved_erases.append(item)
    assert not [
        item for item in erase_operations if item.get("reason") == "display-clear"
    ], (
        "a Beam journey emitted a CSI J0-J3 display clear between compositor "
        f"markers, including a clear repainted before the next frame: {erase_operations!r}"
    )
    assert not unbracketed_preserved_erases, (
        "a Beam journey erased a preserved heading/header/details/editor/command "
        "bar outside a proven synchronized-output bracket: "
        f"{unbracketed_preserved_erases!r}"
    )
    return {
        "sequence_window": [first_sequence, last_sequence],
        "events_scanned": len(journey_events),
        "erase_operations": erase_operations,
        "preserved_region_erases": preserved_region_erases,
        "sync_bracket_boundaries": [
            {
                "sequence": event["sequence"],
                "offset": event["offset"],
                "final": operation["final"],
                "bracket_id": operation.get("sync_bracket_id"),
            }
            for event in journey_events
            for operation in event.get("terminal_operations", ())
            if operation.get("parameters") == "?2026"
            and operation.get("final") in {"h", "l"}
        ],
    }


def test_beam_no_flash_oracles_reject_mutated_one_pass_controls():
    expected = {
        "screen_id": 10, "table_id": 11, "detail_id": 12, "editor_id": 13,
    }
    observations = [dict(expected), dict(expected)]
    mounts = [{"event": "mount", "screen_id": expected["screen_id"]}]
    _assert_stable_beam_identity(observations, expected, mounts)
    remounted = [dict(observations[0]), dict(observations[1])]
    remounted[1]["screen_id"] += 1
    with pytest.raises(AssertionError, match="remounted"):
        _assert_stable_beam_identity(
            remounted, expected,
            mounts + [{"event": "mount", "screen_id": remounted[1]["screen_id"]}],
        )

    def cell_style(data, fg="f0f0f0", bg="121212", bold=False):
        return {
            "data": data, "fg": fg, "bg": bg, "bold": bold,
            "reverse": False, "underscore": False,
        }

    grid = [" " * 80 for _ in range(24)]
    grid[0] = "BEAM · width 2 · depth 1".ljust(80)
    grid[3] = "label state score continuation".ljust(80)
    grid[4] = "b1   LIVE  0.0    amber wrapped continuation".ljust(80)
    grid[5] = "                        words stay in the cell".ljust(80)
    grid[3] = grid[3][:52] + "SELECTED: b1" + grid[3][64:]
    grid[20] = "Beam >".ljust(80)
    frame = {
        "sequence": 1, "generation": 1, "size": [80, 24],
        "screen_id": expected["screen_id"], "beam_prompt_styles": [
            cell_style(char, fg="f0f0f0") for char in "Beam >"
        ],
        "grid": grid,
        "geometry": {
            "kind": "beam", "screen_generation": 1,
            "selected_label": "b1", "live_labels": ["b1"],
            "live_continuations": {
                "b1": " amber wrapped continuation words stay in the cell",
            },
            "rendered_continuations": {
                "b1": ["amber wrapped continuation", "words stay in the cell"],
            },
            "table_scroll_y": 0,
            "row_order": ["b1"], "row_heights": {"b1": 2},
            "table_header_height": 1,
            "heading_region": [0, 0, 80, 1],
            "table_region": [0, 3, 48, 8],
            "table_header_region": [0, 3, 48, 1],
            "detail_region": [52, 3, 28, 8],
            "editor_region": [6, 20, 70, 1],
            "command_bar_region": [0, 20, 80, 1],
            "continuation_region": [18, 4, 30, 7],
        },
        "region_styles": {
            "heading": [cell_style("B", bold=True)],
            "table_header": [cell_style("l", bold=True)],
            "command_bar": [cell_style("B", bold=True)],
            "editor": [cell_style(" ", fg="default", bg="b4b4b4")],
        },
    }
    _assert_complete_beam_frame(frame, expected["screen_id"])
    blank_table = json.loads(json.dumps(frame))
    x, y, width, height = blank_table["geometry"]["table_region"]
    for row in range(y, y + height):
        blank_table["grid"][row] = (
            blank_table["grid"][row][:x]
            + " " * width
            + blank_table["grid"][row][x + width:]
        )
    with pytest.raises(AssertionError, match="live branch b1"):
        _assert_complete_beam_frame(blank_table, expected["screen_id"])

    display_markers = [
        {"sequence": 10, "kind": "display", "terminal_operations": []},
        {"sequence": 12, "kind": "display", "terminal_operations": []},
    ]
    repainted_clear = [
        *display_markers,
        {
            "sequence": 11, "kind": "write",
            "terminal_operations": [{
                "parameters": "2", "intermediates": "", "final": "J",
            }],
        },
    ]
    with pytest.raises(AssertionError, match="CSI J0-J3 display clear"):
        _assert_no_beam_terminal_erases(repainted_clear, display_markers)

    stable_geometry = {
        "heading_region": [0, 0, 80, 1],
        "table_region": [0, 3, 48, 8],
        "table_header_region": [0, 3, 48, 1],
        "detail_region": [52, 3, 28, 8],
        "editor_region": [6, 20, 70, 1],
        "command_bar_region": [0, 20, 80, 1],
    }
    stable_markers = [
        {
            "sequence": sequence, "kind": "display", "size": [80, 24],
            "geometry": stable_geometry, "sync_available": False,
            "sync_active": False,
        }
        for sequence in (20, 22)
    ]
    unbracketed_editor_erase = [
        stable_markers[0],
        {
            "sequence": 21, "kind": "write", "size": [80, 24],
            "terminal_operations": [{
                "parameters": "0", "intermediates": "", "final": "K",
                "cursor_before": [10, 20], "sync_active_before": False,
                "sync_active_after": False, "sync_bracket_id": None,
            }],
        },
        stable_markers[1],
    ]
    with pytest.raises(AssertionError, match="outside a proven synchronized-output"):
        _assert_no_beam_terminal_erases(
            unbracketed_editor_erase, stable_markers, sync_available=False,
        )
    candidate_cleanup = [
        stable_markers[0],
        {
            "sequence": 21, "kind": "write", "size": [80, 24],
            "terminal_operations": [{
                "parameters": "10", "intermediates": "", "final": "X",
                "cursor_before": [18, 4], "sync_active_before": False,
                "sync_active_after": False, "sync_bracket_id": None,
            }],
        },
        stable_markers[1],
    ]
    candidate_report = _assert_no_beam_terminal_erases(
        candidate_cleanup, stable_markers, sync_available=False,
    )
    assert candidate_report["erase_operations"][0]["reason"] == (
        "changing-table-or-unpreserved-area"
    )
    synchronized_markers = [
        {**marker, "sync_available": True} for marker in stable_markers
    ]
    synchronized_erase = [
        synchronized_markers[0],
        {
            "sequence": 21, "offset": 1, "kind": "write", "size": [80, 24],
            "terminal_operations": [{
                "parameters": "?2026", "intermediates": "", "final": "h",
                "cursor_before": [0, 0], "sync_available": True,
                "sync_active_before": False, "sync_active_after": True,
                "sync_bracket_id": 1,
            }],
        },
        {
            "sequence": 22, "offset": 2, "kind": "write", "size": [80, 24],
            "terminal_operations": [{
                "parameters": "0", "intermediates": "", "final": "K",
                "cursor_before": [10, 20], "sync_available": True,
                "sync_active_before": True, "sync_active_after": True,
                "sync_bracket_id": 1,
            }],
        },
        {
            "sequence": 23, "offset": 3, "kind": "write", "size": [80, 24],
            "terminal_operations": [{
                "parameters": "?2026", "intermediates": "", "final": "l",
                "cursor_before": [10, 20], "sync_available": True,
                "sync_active_before": True, "sync_active_after": False,
                "sync_bracket_id": 1,
            }],
        },
        {**synchronized_markers[1], "sequence": 24},
    ]
    bracketed_report = _assert_no_beam_terminal_erases(
        synchronized_erase, [synchronized_erase[0], synchronized_erase[-1]],
        sync_available=True,
    )
    assert len(bracketed_report["preserved_region_erases"]) == 1

    stable_pair = [
        {
            "geometry": {
                "accepting": False,
                "editor_focused": True,
                "editor_region": [0, 0, 2, 1],
                "cursor_screen_offset": [1, 0],
            },
            "region_styles": {
                "editor": [
                    cell_style(" ", bg="b4b4b4"),
                    cell_style(" ", bg="f0f0f0"),
                ],
            },
        }
        for _ in range(2)
    ]
    _assert_focused_editor_styles(stable_pair)
    caret_blink = json.loads(json.dumps(stable_pair))
    caret_blink[1]["region_styles"]["editor"][1] = cell_style(
        " ", bg="a4a4a4",
    )
    _assert_focused_editor_styles(caret_blink)
    focus_removed = json.loads(json.dumps(stable_pair))
    focus_removed[1]["geometry"]["editor_focused"] = False
    with pytest.raises(AssertionError, match="lost command-editor focus"):
        _assert_focused_editor_styles(focus_removed)
    unstyled = json.loads(json.dumps(stable_pair))
    unstyled[1]["region_styles"]["editor"][0] = cell_style(
        " ", fg="f0f0f0", bg="121212",
    )
    with pytest.raises(AssertionError):
        _assert_focused_editor_styles(unstyled)


def test_resize_checkpoint_decoder_preserves_split_utf8():
    frames = _replay_terminal_frames(
        "€".encode(),
        [(0, (10, 2)), (1, (10, 2))],
    )
    assert "€" in frames[-1]["grid"][0]
    assert "�" not in "\n".join(frames[-1]["grid"])


def _read_until(
    master: int,
    process: subprocess.Popen,
    output: bytearray,
    needle: bytes,
    timeout: float = 8,
) -> None:
    deadline = time.monotonic() + timeout
    while needle not in output and time.monotonic() < deadline:
        if process.poll() is not None:
            break
        _pump(master, output, .05)
    assert needle in output, (
        f"PTY did not display {needle!r}; returncode={process.poll()}\n"
        + output[-4000:].decode("utf-8", errors="replace")
    )


def _wait_json(
    master: int,
    process: subprocess.Popen,
    output: bytearray,
    path: Path,
    predicate,
    timeout: float = 8,
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _pump(master, output, .03)
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                value = None
            if value is not None and predicate(value):
                return value
        if process.poll() is not None:
            break
    assert False, (
        f"PTY state did not reach the requested checkpoint; returncode={process.poll()}\n"
        f"state={path.read_text(encoding='utf-8') if path.exists() else '<missing>'}\n"
        + output[-4000:].decode("utf-8", errors="replace")
    )


def _launch(tmp_path: Path, name: str, mode: str, size=(120, 40), *, child=_CHILD):
    master, slave = os.openpty()
    _set_size(master, *size)
    initial_attributes = termios.tcgetattr(slave)

    def acquire_controlling_tty():
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)

    repository = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository / "core" / "src"), str(repository), environment.get("PYTHONPATH", ""))
    )
    result_path = tmp_path / f"{name}.json"
    process = subprocess.Popen(
        [sys.executable, "-c", child, str(result_path), mode],
        cwd=repository,
        env=environment,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
        start_new_session=True,
        preexec_fn=acquire_controlling_tty,  # noqa: PLW1509 - PTY session setup
    )
    return process, master, slave, initial_attributes, result_path


def _send(master: int, text: bytes) -> None:
    os.write(master, text)


def _latest_snapshot(value):
    observations = value.get("observations", ())
    return observations[-1] if observations else {}


@pytest.mark.skipif(os.name != "posix", reason="the custom driver requires a POSIX PTY")
def test_choice_mouse_selection_keeps_command_editor_focused(tmp_path):
    process, master, slave, initial_attributes, result_path = _launch(
        tmp_path, "choice-row", "choice-row", child=_CHOICE_CHILD,
    )
    output = bytearray()
    try:
        snapshot_path = result_path.with_suffix(".snapshot.json")
        initial = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "choice"
            and _latest_snapshot(value).get("editor_focused"),
        )
        table_x, table_y, _width, _height, header_height = _latest_snapshot(initial)["table"]
        click_x = table_x + 5
        click_y = table_y + header_height + 2
        click = f"\x1b[<0;{click_x};{click_y}M\x1b[<0;{click_x};{click_y}m".encode()

        _send(master, click)
        selected = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: any(
                item.get("kind") == "choice"
                and item.get("text") == "2"
                and item.get("editor_focused")
                for item in value.get("observations", ())
            ),
        )
        assert all(item["editor_focused"] for item in selected["observations"])

        # Re-selecting the highlighted row emits RowSelected instead of
        # RowHighlighted. Typing immediately must still land in the editor.
        _send(master, click + b" ")
        typed = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: _latest_snapshot(value).get("text") == "2 "
            and _latest_snapshot(value).get("editor_focused"),
        )
        assert all(item["editor_focused"] for item in typed["observations"])
        _send(master, b"\r")

        deadline = time.monotonic() + 8
        while process.poll() is None and time.monotonic() < deadline:
            _pump(master, output, .05)
        assert process.poll() == 0, output[-4000:].decode("utf-8", errors="replace")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["command"] == "2 "
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        final_attributes = termios.tcgetattr(slave)
        os.close(master)
        os.close(slave)
    assert final_attributes == initial_attributes


@pytest.mark.skipif(os.name != "posix", reason="the custom driver requires a POSIX PTY")
def test_edge_template_click_keeps_editor_focused_for_immediate_command_entry(tmp_path):
    process, master, slave, initial_attributes, result_path = _launch(
        tmp_path, "edge-row", "edge-row", child=_EDGE_CHILD,
    )
    output = bytearray()
    try:
        snapshot_path = result_path.with_suffix(".snapshot.json")
        initial = _wait_json(
            master, process, output, snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "edge"
            and _latest_snapshot(value).get("editor_focused"),
        )
        table = _latest_snapshot(initial)["table"]
        row_index = table[5].index("new TEXT")
        click_x = table[0] + 5
        click_y = table[1] + table[4] + row_index + 1
        _send(master, f"\x1b[<0;{click_x};{click_y}M".encode())
        down = _wait_json(
            master, process, output, snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "edge"
            and _latest_snapshot(value).get("editor_focused"),
        )
        assert _latest_snapshot(down)["editor_focused"]
        _send(master, f"\x1b[<0;{click_x};{click_y}m".encode())
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "edge"
            and _latest_snapshot(value).get("text") == "new ",
        )
        _send(master, b"branch")
        staged = _wait_json(
            master, process, output, snapshot_path,
            lambda value: _latest_snapshot(value).get("text") == "new branch"
            and _latest_snapshot(value).get("editor_focused"),
        )
        assert _latest_snapshot(staged)["read_only"] is False
        _send(master, b"\r")
        deadline = time.monotonic() + 8
        while process.poll() is None and time.monotonic() < deadline:
            _pump(master, output, .05)
        assert process.poll() == 0, output[-4000:].decode("utf-8", errors="replace")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["command"] == "new branch"
        assert result["parsed"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        final_attributes = termios.tcgetattr(slave)
        os.close(master)
        os.close(slave)
    assert final_attributes == initial_attributes


@pytest.mark.skipif(os.name != "posix", reason="the custom driver requires a POSIX PTY")
def test_textual_bridge_drives_a_multi_turn_runtime_and_rejects_handoff_input(tmp_path):
    process, master, slave, initial_attributes, result_path = _launch(
        tmp_path, "runtime", "runtime", child=_RUNTIME_CHILD,
    )
    output = bytearray()
    snapshot_path = result_path.with_suffix(".snapshot.json")
    try:
        def wait_for(kind, after_generation=-1, *, accepting=True):
            value = _wait_json(
                master, process, output, snapshot_path,
                lambda snapshot: any(
                    item.get("kind") == kind
                    and item.get("generation", -1) > after_generation
                    and item.get("accepting") is accepting
                    for item in snapshot.get("observations", ())
                ),
            )
            return next(
                item for item in reversed(value["observations"])
                if item.get("kind") == kind
                and item.get("generation", -1) > after_generation
                and item.get("accepting") is accepting
            )

        def enter_choice_command(command):
            first, rest = command[:1], command[1:]
            _send(master, first.encode())
            _wait_json(
                master, process, output, snapshot_path,
                lambda value: _latest_snapshot(value).get("kind") == "choice"
                and _latest_snapshot(value).get("accepting")
                and _latest_snapshot(value).get("text") == first,
            )
            if rest:
                _send(master, rest.encode())
                _wait_json(
                    master, process, output, snapshot_path,
                    lambda value: _latest_snapshot(value).get("kind") == "choice"
                    and _latest_snapshot(value).get("text") == command,
                )
            _send(master, b"\r")

        choice = wait_for("choice")
        generation = choice["generation"]
        for _ in range(2):
            _send(master, b"\r")
            choice = wait_for("choice", generation)
            generation = choice["generation"]
        _send(master, b"\r")
        edge = wait_for("edge", generation)
        generation = edge["generation"]
        _send(master, b"new")
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "edge"
            and _latest_snapshot(value).get("text") == "new"
            and _latest_snapshot(value).get("focused"),
        )
        _send(master, b"\r")
        prompt = wait_for("prompt", generation)
        generation = prompt["generation"]
        _send(master, b"Q")
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "prompt"
            and _latest_snapshot(value).get("text") == "Q",
        )
        _send(master, b"\x1b")
        _pump(master, output, .15)
        _send(master, b"\r")
        choice = wait_for("choice", generation)
        generation = choice["generation"]
        enter_choice_command("beam 2")
        beam = wait_for("beam", generation)
        generation = beam["generation"]

        result_path.with_suffix(".arm").touch()
        _send(master, b"\x1b[C")
        deadline = time.monotonic() + 8
        while not result_path.with_suffix(".waiting").exists() and time.monotonic() < deadline:
            _pump(master, output, .03)
        assert result_path.with_suffix(".waiting").exists(), output[-3000:].decode(
            "utf-8", errors="replace"
        )
        _send(
            master,
            b"Z\x1b[<0;10;8M\x1b[<0;10;8m"
            b"\x1b[200~PASTE_LEAK\x1b[201~",
        )
        handoff = _wait_json(
            master, process, output, snapshot_path,
            lambda value: any(
                item.get("kind") == "handoff"
                and item.get("stale_keys", 0) >= 1
                and item.get("stale_mouse", 0) >= 1
                and item.get("stale_input", 0) >= 3
                for item in value.get("observations", ())
            ),
        )
        assert next(
            item for item in reversed(handoff["observations"])
            if item.get("kind") == "handoff"
        )["accepting"] is False
        result_path.with_suffix(".release").touch()

        next_beam = wait_for("beam", generation)
        assert next_beam["text"] == ""
        assert next_beam["focused"]
        generation = next_beam["generation"]
        _send(master, b"select b5")
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "beam"
            and _latest_snapshot(value).get("text") == "select b5",
        )
        _send(master, b"\r")
        choice = wait_for("choice", generation)
        generation = choice["generation"]
        enter_choice_command("beam 2")
        beam = wait_for("beam", generation)
        generation = beam["generation"]
        _send(master, b"\x1b")
        choice = wait_for("choice", generation)
        generation = choice["generation"]
        _send(master, b"e\r")
        edge = wait_for("edge", generation)
        generation = edge["generation"]
        _send(master, b"q")
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "edge"
            and _latest_snapshot(value).get("text") == "q",
        )
        _send(master, b"\r")

        deadline = time.monotonic() + 10
        while process.poll() is None and time.monotonic() < deadline:
            _pump(master, output, .05)
        assert process.poll() == 0, output[-4000:].decode("utf-8", errors="replace")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        audit = result["audit"]
        assert audit["return_code"] == 0
        assert audit["stats"]["stale_key_events"] >= 1
        assert audit["stats"]["stale_mouse_events"] >= 1
        branches = audit["branches"]
        root = next(item for item in branches if item["prompt"] == "P")
        fresh = next(item for item in branches if item["prompt"] == "Q")
        assert root["actions"] == ["SelectRawRank", "SelectRawRank", "SelectRawRank"]
        assert root["boundary"] == 2
        assert fresh["tokens"] == [1, 2]
        assert any(item["kind"] == "prompt" for item in result["observations"])
        assert any(
            item["kind"] == "beam" and item["text"] == "select b5"
            for item in result["observations"]
        )
        assert sum(item["kind"] == "beam" for item in result["observations"]) >= 3
        assert [item["command"] for item in audit["beam_inputs"]] == [
            "advance 1", "select b5", "return",
        ]
        assert b"PTY_NATIVE_CAPTURE_MARKER" in output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        final_attributes = termios.tcgetattr(slave)
        os.close(master)
        os.close(slave)
    assert final_attributes == initial_attributes


@pytest.mark.skipif(os.name != "posix", reason="the custom driver requires a POSIX PTY")
@pytest.mark.parametrize(
    ("initial_size", "other_size"),
    [((80, 24), (160, 50)), ((160, 50), (80, 24))],
)
def test_runtime_beam_advances_keep_one_complete_painted_screen(
    tmp_path, initial_size, other_size,
):
    process, master, slave, initial_attributes, result_path = _launch(
        tmp_path, "beam-no-flash", "beam-no-flash", size=initial_size,
        child=_RUNTIME_CHILD,
    )
    output = bytearray()
    snapshot_path = result_path.with_suffix(".snapshot.json")
    waiting_path = result_path.with_suffix(".waiting")
    release_path = result_path.with_suffix(".release")
    frame_path = result_path.with_suffix(".frames.json")
    resizes = [(0, initial_size)]
    result = None
    return_code = None

    def wait_for(kind, after_generation=-1, *, accepting=True, size=None):
        def matches(item):
            return (
                item.get("kind") == kind
                and item.get("generation", -1) > after_generation
                and item.get("accepting") is accepting
                and (size is None or item.get("size") == list(size))
            )

        state_file = _wait_json(
            master, process, output, snapshot_path,
            lambda value: any(matches(item) for item in value.get("observations", ())),
            timeout=12,
        )
        return next(
            item for item in reversed(state_file["observations"])
            if matches(item)
        )

    def write_ordered_capture():
        snapshot = {}
        if snapshot_path.exists():
            try:
                snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        audit = (result or {}).get("audit", {})
        if not audit and result_path.exists():
            try:
                audit = json.loads(result_path.read_text(encoding="utf-8")).get("audit", {})
            except (json.JSONDecodeError, OSError):
                pass
        child_frames = audit.get("capture_frames", [])
        if not child_frames and frame_path.exists():
            try:
                child_frames = json.loads(frame_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        regions_by_size = {}
        for item in snapshot.get("observations", ()):
            if item.get("kind") == "beam" and item.get("regions"):
                regions_by_size[tuple(item["size"])] = {
                    name: tuple(region)
                    for name, region in item["regions"].items()
                }
        replay_error = None
        ordered_frames = []
        by_sequence = {
            int(frame["sequence"]): frame for frame in child_frames
            if "sequence" in frame
        }
        try:
            ordered_frames = _replay_terminal_frames(
                bytes(output), resizes, style_regions=regions_by_size,
                display_metadata=by_sequence,
                sync_available=bool(audit.get("sync_available")),
            )
        except Exception as error:  # noqa: BLE001 - preserve raw evidence on replay failures
            replay_error = f"{type(error).__name__}: {error}"
        for frame in ordered_frames:
            metadata = frame.get("frame_metadata")
            if metadata is not None:
                frame["screen_id"] = metadata.get("screen_id")
                frame["characters"] = metadata.get("characters")
                frame["generation"] = metadata.get("generation")
                frame["geometry"] = metadata.get("geometry")
        raw_path = result_path.with_suffix(".beam-no-flash.pty")
        raw_path.write_bytes(bytes(output))
        capture_path = result_path.with_suffix(".beam-no-flash.capture.json")
        capture_path.write_text(json.dumps({
            "raw_pty_path": str(raw_path),
            "resize_sequence": [
                {"offset": offset, "size": list(size)} for offset, size in resizes
            ],
            "sync_probe_query_seen": b"\x1b[?2026$p" in output,
            "sync_probe_reply_sent": "\x1b[?2026;2$y",
            "sync_available": audit.get("sync_available"),
            "sync_protocol_boundaries": [
                {
                    "sequence": frame.get("sequence"),
                    "offset": frame["offset"],
                    "kind": frame["kind"],
                    "active_at_marker": frame.get("sync_active"),
                    "operation": operation,
                }
                for frame in ordered_frames
                for operation in frame.get("terminal_operations", ())
                if operation.get("parameters") == "?2026"
                and operation.get("final") in {"h", "l"}
            ],
            "audit": audit,
            "child_frame_metadata": child_frames,
            "observations": snapshot.get("observations", []),
            "ordered_events": ordered_frames,
            "replay_error": replay_error,
        }, indent=2), encoding="utf-8")

    try:
        _answer_synchronized_output_probe(master, process, output)

        choice = wait_for("choice")
        generation = choice["generation"]
        def enter_choice_command(command):
            first, remainder = command[:1], command[1:]
            _send(master, first.encode())
            _wait_json(
                master, process, output, snapshot_path,
                lambda value: _latest_snapshot(value).get("kind") == "choice"
                and _latest_snapshot(value).get("generation") == generation
                and _latest_snapshot(value).get("accepting")
                and _latest_snapshot(value).get("text") == first,
            )
            if remainder:
                _send(master, remainder.encode())
                _wait_json(
                    master, process, output, snapshot_path,
                    lambda value: _latest_snapshot(value).get("kind") == "choice"
                    and _latest_snapshot(value).get("generation") == generation
                    and _latest_snapshot(value).get("accepting")
                    and _latest_snapshot(value).get("text") == command,
                )
            _send(master, b"\r")

        enter_choice_command("beam 2")
        beam = wait_for("beam", generation)
        initial_generation = beam["generation"]
        initial_identity = {
            key: beam[key] for key in ("screen_id", "table_id", "detail_id", "editor_id")
        }
        assert beam["size"] == list(initial_size)
        assert beam["focused"]

        result_path.with_suffix(".arm").touch()
        _send(master, b"\x1b[C")
        deadline = time.monotonic() + 12
        while not waiting_path.exists() and process.poll() is None and time.monotonic() < deadline:
            _pump(master, output, .03)
        assert waiting_path.exists(), (
            "the deterministic backend did not enter the explicit Beam wait gate; "
            + output[-4000:].decode("utf-8", errors="replace")
        )
        waiting = _wait_json(
            master, process, output, snapshot_path,
            lambda value: any(
                item.get("kind") == "beam"
                and item.get("generation") == initial_generation
                and item.get("active") is False
                and item.get("accepting") is False
                for item in value.get("observations", ())
            ),
        )
        waiting_view = next(
            item for item in reversed(waiting["observations"])
            if item.get("kind") == "beam"
            and item.get("generation") == initial_generation
            and item.get("active") is False
        )
        assert {
            key: waiting_view[key] for key in initial_identity
        } == initial_identity
        _send(master, b"Z\x1b[200~STALE_PASTE\x1b[201~")
        stale = _wait_json(
            master, process, output, snapshot_path,
            lambda value: any(
                item.get("kind") == "beam"
                and item.get("active") is False
                and item.get("stale_keys", 0) >= 1
                and item.get("stale_input", 0) >= 2
                for item in value.get("observations", ())
            ),
        )
        assert next(
            item for item in reversed(stale["observations"])
            if item.get("kind") == "beam" and item.get("active") is False
        )["accepting"] is False
        wait_marker = b"\x1b]777;SPEPHASE;WAIT\x07"
        deadline = time.monotonic() + 8
        while (
            wait_marker not in output
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            _pump(master, output, .03)
        wait_offset = output.find(wait_marker)
        assert wait_offset >= 0, "the backend gate marker was not visible in the PTY"
        after_wait = wait_offset + len(wait_marker)
        while (
            output[after_wait:].count(b"\x1b]777;SPEFRAME;") < 2
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            _pump(master, output, .03)
        assert output[after_wait:].count(b"\x1b]777;SPEFRAME;") >= 2, (
            "the backend gate did not produce two more completed compositor frames"
        )
        release_path.touch()

        beam = wait_for("beam", initial_generation)
        generations = [beam["generation"]]
        # One gated result plus 29 fast results makes 30 actual Right-key
        # submissions in a single production runtime and Textual driver.
        for _ in range(29):
            _send(master, b"\x1b[C")
            beam = wait_for("beam", beam["generation"])
            generations.append(beam["generation"])

        _drain_to_quiet_display_boundary(master, output)
        _set_size(master, *other_size)
        resizes.append((len(output), other_size))
        os.kill(process.pid, signal.SIGWINCH)
        wide = _wait_json(
            master, process, output, snapshot_path,
            lambda value: any(
                item.get("kind") == "beam"
                and item.get("accepting")
                and item.get("size") == list(other_size)
                and item.get("generation") == beam["generation"]
                for item in value.get("observations", ())
            ),
        )
        beam = next(
            item for item in reversed(wide["observations"])
            if item.get("kind") == "beam"
            and item.get("size") == list(other_size)
            and item.get("generation") == generations[-1]
            and item.get("accepting")
        )
        _drain_to_quiet_display_boundary(master, output)
        _set_size(master, *initial_size)
        resizes.append((len(output), initial_size))
        os.kill(process.pid, signal.SIGWINCH)
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: any(
                item.get("kind") == "beam"
                and item.get("accepting")
                and item.get("size") == list(initial_size)
                and item.get("generation") == generations[-1]
                for item in value.get("observations", ())
            ),
        )

        # Exercise the full deterministic Beam control set after the repeated
        # advances and separate resize cycle.
        depth_after_30 = int(re.search(r"depth (\d+)", beam["heading"]).group(1))
        _send(master, b"\x1b[D")
        rewound = wait_for("beam", generations[-1])
        assert int(re.search(r"depth (\d+)", rewound["heading"]).group(1)) == depth_after_30 - 1
        _send(master, b"\x1b[C")
        readvanced = wait_for("beam", rewound["generation"])
        assert int(re.search(r"depth (\d+)", readvanced["heading"]).group(1)) == depth_after_30

        _send(master, b"p")
        protected = wait_for("beam", readvanced["generation"])
        selected_row = next(
            row for row in protected["rows"]
            if row["label"] == protected["selected"]
        )
        assert selected_row["protected"] is True
        _send(master, b"f")
        family_view = wait_for("beam", protected["generation"])
        assert family_view["show_family_metadata"] is True
        assert any(row["family_metadata"] for row in family_view["rows"])
        _send(master, b"p")
        unprotected = wait_for("beam", family_view["generation"])
        assert next(
            row for row in unprotected["rows"]
            if row["label"] == unprotected["selected"]
        )["protected"] is False

        labels = [row["label"] for row in unprotected["rows"]]
        assert len(labels) >= 2, f"deterministic beam lost branch choices: {labels!r}"
        current_index = labels.index(unprotected["selected"])
        prune_target = labels[(current_index + 1) % len(labels)]
        _send(master, b"\x1b[B")
        selected_next = _wait_json(
            master, process, output, snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "beam"
            and _latest_snapshot(value).get("generation") == unprotected["generation"]
            and _latest_snapshot(value).get("selected") == prune_target,
        )
        assert _latest_snapshot(selected_next)["selected"] == prune_target
        _send(master, b"\x7f")
        pruned = wait_for("beam", unprotected["generation"])
        live_rows = [row for row in pruned["rows"] if row["state"] == "LIVE"]
        assert len(live_rows) >= 1
        assert prune_target not in [row["label"] for row in pruned["rows"]]

        _send(master, b"q")
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "beam"
            and _latest_snapshot(value).get("generation") == pruned["generation"]
            and _latest_snapshot(value).get("text") == "q"
            and _latest_snapshot(value).get("accepting"),
        )
        _send(master, b"\r")
        options = wait_for("beam", pruned["generation"])
        assert options["at_edge"] is True
        depth_at_edge = int(re.search(r"depth (\d+)", options["heading"]).group(1))
        _send(master, b"\x1b[C")
        resumed = wait_for("beam", options["generation"])
        assert resumed["at_edge"] is False
        assert int(re.search(r"depth (\d+)", resumed["heading"]).group(1)) == depth_at_edge
        deterministic_end_generation = resumed["generation"]

        _send(master, b"\x1b")
        choice = wait_for("choice", resumed["generation"])

        # Exercise the second same-kind runtime with stochastic scores, then
        # return through Choice and leave by its explicit EDGE action.
        generation = choice["generation"]
        enter_choice_command("gbeam 2")
        stochastic = wait_for("beam", generation)
        stochastic_initial_generation = stochastic["generation"]
        stochastic_initial_identity = {
            key: stochastic[key] for key in _BEAM_IDENTITY_KEYS
        }
        assert "STOCHASTIC" in stochastic["heading"]
        assert any("G " in score for score in stochastic["table_scores"])
        for _ in range(2):
            _send(master, b"\x1b[C")
            stochastic = wait_for("beam", stochastic["generation"])
        stochastic_end_generation = stochastic["generation"]
        _send(master, b"\x1b")
        choice = wait_for("choice", stochastic["generation"])
        generation = choice["generation"]
        enter_choice_command("q")
        wait_for("edge", choice["generation"])
        _send(master, b"\x04")
        deadline = time.monotonic() + 12
        while process.poll() is None and time.monotonic() < deadline:
            _pump(master, output, .05)
        return_code = process.poll()
        if return_code == 0 and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        write_ordered_capture()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        final_attributes = termios.tcgetattr(slave)
        os.close(master)
        os.close(slave)

    assert return_code == 0, output[-5000:].decode("utf-8", errors="replace")
    assert final_attributes == initial_attributes
    assert result is not None
    audit = result["audit"]
    deterministic_inputs = [
        item for item in audit["beam_inputs"] if not item["stochastic"]
    ]
    stochastic_inputs = [
        item for item in audit["beam_inputs"] if item["stochastic"]
    ]
    assert [item["command"] for item in deterministic_inputs] == [
        "advance 1",
    ] * 30 + [
        "rewind", "advance 1", "protect", "families", "protect",
        f"kill {prune_target}", "q", "resume", "return",
    ]
    assert [item["command"] for item in stochastic_inputs] == [
        "advance 1", "advance 1", "return",
    ]
    selection_changes = [
        index for index, item in enumerate(deterministic_inputs)
        if item["selected_label"] != item["state_selected_label"]
    ]
    assert selection_changes == [35], (
        "only the explicit Down-before-prune action may change the submitted "
        f"selection: {selection_changes!r}"
    )
    prune_input = deterministic_inputs[35]
    assert prune_input["command"] == f"kill {prune_target}"
    assert prune_input["selected_label"] == prune_target
    assert prune_input["state_selected_label"] == unprotected["selected"]
    assert all(
        item["selected_label"] == item["state_selected_label"]
        for item in stochastic_inputs
    )
    assert len(generations) == 30
    assert generations == sorted(generations)
    assert audit["sync_available"] is True

    observations = result["observations"]
    beam_ready = [
        item for item in observations
        if item.get("kind") == "beam"
        and item.get("accepting")
        and initial_generation <= item.get("generation", -1) <= generations[-1]
    ]
    assert len(beam_ready) >= 31
    _assert_stable_beam_identity(
        beam_ready, initial_identity, audit["mount_events"],
        through_generation=generations[-1], expect_final_unmount=True,
    )
    all_deterministic_ready = [
        item for item in observations
        if item.get("kind") == "beam" and item.get("accepting")
        and initial_generation <= item.get("generation", -1)
        <= deterministic_end_generation
    ]
    _assert_stable_beam_identity(
        all_deterministic_ready, initial_identity, audit["mount_events"],
        expect_final_unmount=True,
    )
    initial_screen_events = [
        event for event in audit["mount_events"]
        if event["screen_id"] == initial_identity["screen_id"]
    ]
    assert [event["event"] for event in initial_screen_events] == ["mount", "unmount"]
    assert initial_screen_events[0]["order"] < initial_screen_events[1]["order"]
    deterministic_states = [
        item for item in audit["runtime_beam_states"] if not item["stochastic"]
    ]
    stochastic_states = [
        item for item in audit["runtime_beam_states"] if item["stochastic"]
    ]
    assert len(deterministic_states) == len(deterministic_inputs)
    assert len(stochastic_states) == 3
    depths = [
        int(re.search(r"depth (\d+)", item["title"]).group(1))
        for item in deterministic_states
    ]
    start_depth = depths[0]
    assert depths == (
        list(range(start_depth, start_depth + 30))
        + [start_depth + 30, start_depth + 29]
        + [start_depth + 30] * 7
    ), depths
    root_orders = []
    for index, (beam_input, state) in enumerate(zip(
        deterministic_inputs, deterministic_states,
    )):
        labels_from_input = [row["label"] for row in beam_input["rows"]]
        labels_from_search = [path["label"] for path in state["paths"]]
        assert labels_from_input == labels_from_search, (
            f"Beam row order diverged from BeamSearch at input {index}: "
            f"{labels_from_input!r} != {labels_from_search!r}"
        )
        for beam_row, path in zip(beam_input["rows"], state["paths"]):
            assert beam_row["continuation"] == path["continuation"]
            assert beam_row["state"] == path["state"]
            assert path["state"] == ("EOS" if path["search_state"] == "finished" else "LIVE")
            assert round(path["score"], 6) == float(beam_row["score"])
            assert beam_row["protected"] is path["protected"]
            assert beam_row["family_metadata"] == path["family_metadata"]
        roots = []
        for path in state["paths"]:
            if path["state"] == "LIVE" and path["token_ids"]:
                root = path["token_ids"][0]
                if root not in roots:
                    roots.append(root)
        root_orders.append(tuple(roots))
        assert beam_input["state_selected_label"] == state["selected_label"]
        if index == 35:
            assert beam_input["selected_label"] == prune_target
        else:
            assert beam_input["selected_label"] == state["selected_label"]
        assert state["selected_label"] in set(labels_from_search)
        assert state["base_visible"] == deterministic_states[0]["base_visible"]
        assert state["base_prefix"] == deterministic_states[0]["base_prefix"]
        for path in state["paths"]:
            token_count = len(path["token_ids"])
            if path["state"] == "LIVE":
                assert token_count == depths[index]
            else:
                assert path["state"] == "EOS"
                assert 1 <= token_count <= depths[index]
                assert path["token_ids"][-1] == 0
    assert any(
        left != right for left, right in pairwise(root_orders[:30])
    ), f"deterministic beam never reordered surviving root branches: {root_orders[:30]!r}"
    for index in range(30):
        previous = deterministic_states[index]["paths"]
        following = deterministic_states[index + 1]["paths"]
        previous_live = [
            tuple(path["token_ids"]) for path in previous
            if path["state"] == "LIVE"
        ]
        previous_finished = {
            tuple(path["token_ids"]) for path in previous
            if path["state"] == "EOS"
        }
        for path in following:
            tokens = tuple(path["token_ids"])
            if path["state"] == "EOS" and tokens in previous_finished:
                continue
            assert any(
                len(tokens) == len(prefix) + 1 and tokens[:-1] == prefix
                for prefix in previous_live
            ), (
                f"path {path['label']} did not extend a prior live branch at "
                f"Right step {index + 1}: {tokens!r}"
            )
    assert deterministic_states[33]["paths"] and any(
        path["protected"] for path in deterministic_states[33]["paths"]
    )
    assert deterministic_states[34]["show_family_metadata"] is True
    assert any(
        path["family_metadata"] for path in deterministic_states[34]["paths"]
    )
    assert all(not path["protected"] for path in deterministic_states[35]["paths"])
    assert all(path["label"] != prune_target for path in deterministic_states[36]["paths"])
    assert all(item["stochastic"] for item in stochastic_states)
    stochastic_depths = [
        len(state["paths"][0]["token_ids"]) for state in stochastic_states
    ]
    assert stochastic_depths == list(range(stochastic_depths[0], stochastic_depths[0] + 3))
    for beam_input, state in zip(stochastic_inputs, stochastic_states):
        assert [row["label"] for row in beam_input["rows"]] == [
            path["label"] for path in state["paths"]
        ]
        assert beam_input["selected_label"] == state["selected_label"]
        for beam_row, path in zip(beam_input["rows"], state["paths"]):
            assert beam_row["continuation"] == path["continuation"]
            assert beam_row["state"] == path["state"]
            assert path["state"] == ("EOS" if path["search_state"] == "finished" else "LIVE")
            assert round(path["score"], 6) == float(beam_row["score"])
            assert beam_row["protected"] is path["protected"]
            assert beam_row["family_metadata"] == path["family_metadata"]
    assert deterministic_inputs[30]["command"] == "rewind"
    assert deterministic_inputs[37]["at_edge"] is True
    assert deterministic_inputs[37]["command"] == "resume"
    assert any(item["stochastic"] for item in audit["runtime_beam_states"])
    assert any(item.get("kind") == "choice" for item in observations)

    capture_path = result_path.with_suffix(".beam-no-flash.capture.json")
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    display_frames = [
        item for item in capture["ordered_events"]
        if item["kind"] == "display"
        and initial_generation <= item.get("generation", -1)
        <= deterministic_end_generation
    ]
    assert display_frames, "no ordered compositor display frames covered the Beam journey"
    selected_long_continuations_visible = []
    for frame in display_frames:
        complete_rows = _assert_complete_beam_frame(
            frame, initial_identity["screen_id"],
        )
        geometry = frame.get("geometry") or {}
        selected = geometry.get("selected_label")
        selected_continuation = geometry.get(
            "live_continuations", {},
        ).get(selected)
        if (
            selected in complete_rows
            and selected_continuation
            and len(selected_continuation) > geometry.get("continuation_width", 0)
        ):
            selected_long_continuations_visible.append(frame["sequence"])
    assert selected_long_continuations_visible, (
        "the PTY journey never scrolled a selected long continuation fully "
        "into the table viewport"
    )
    erase_report = _assert_no_beam_terminal_erases(
        capture["ordered_events"], display_frames,
        sync_available=capture.get("sync_available"),
    )
    _assert_fixed_beam_geometry(display_frames)
    for size in {(80, 24), (160, 50)}:
        sized_frames = [
            frame for frame in display_frames if tuple(frame["size"]) == size
        ]
        focused_frames = _focused_beam_frames(sized_frames)
        _assert_stable_beam_region_styles(focused_frames)
        _assert_focused_editor_styles(focused_frames)
        _assert_stable_prompt_styles(focused_frames)
    stochastic_ready = [
        item for item in observations
        if item.get("kind") == "beam" and item.get("accepting")
        and stochastic_initial_generation <= item.get("generation", -1)
        <= stochastic_end_generation
    ]
    assert len(stochastic_ready) >= 3
    _assert_stable_beam_identity(
        stochastic_ready, stochastic_initial_identity, audit["mount_events"],
        through_generation=stochastic_initial_generation,
        expect_final_unmount=True,
    )
    stochastic_screen_events = [
        event for event in audit["mount_events"]
        if event["screen_id"] == stochastic_initial_identity["screen_id"]
    ]
    assert [event["event"] for event in stochastic_screen_events] == [
        "mount", "unmount",
    ]
    assert stochastic_screen_events[0]["order"] < stochastic_screen_events[1]["order"]
    stochastic_frames = [
        item for item in capture["ordered_events"]
        if item["kind"] == "display"
        and stochastic_initial_generation <= item.get("generation", -1)
        <= stochastic_end_generation
    ]
    assert stochastic_frames
    for frame in stochastic_frames:
        _assert_complete_beam_frame(
            frame, stochastic_initial_identity["screen_id"],
        )
    stochastic_erase_report = _assert_no_beam_terminal_erases(
        capture["ordered_events"], stochastic_frames,
        sync_available=capture.get("sync_available"),
    )
    _assert_fixed_beam_geometry(stochastic_frames)
    focused_stochastic_frames = _focused_beam_frames(stochastic_frames)
    _assert_stable_beam_region_styles(focused_stochastic_frames)
    _assert_focused_editor_styles(focused_stochastic_frames)
    _assert_stable_prompt_styles(focused_stochastic_frames)
    wait_offset = output.find(b"\x1b]777;SPEPHASE;WAIT\x07")
    release_offset = output.find(b"\x1b]777;SPEPHASE;RELEASE\x07")
    assert 0 <= wait_offset < release_offset
    gated_frames = [
        item for item in display_frames
        # Display-event offsets are the end of the SPEPTY marker; phase
        # offsets are the start of their marker. Equality means the completed
        # display marker ended immediately before RELEASE began.
        if wait_offset < item["offset"] <= release_offset
    ]
    assert len(gated_frames) >= 2, (
        "stale-input probes did not produce multiple completed compositor passes "
        f"while the backend was gated: {gated_frames!r}"
    )
    assert all(frame["grid"] == gated_frames[0]["grid"] for frame in gated_frames), (
        "the previously complete Beam changed while stale input was rejected"
    )
    focused_gated_frames = _focused_beam_frames(gated_frames)
    _assert_focused_editor_styles(focused_gated_frames)
    _assert_stable_beam_region_styles(focused_gated_frames)
    _assert_stable_prompt_styles(focused_gated_frames)
    assert all(frame["characters"] > 0 for frame in display_frames)
    assert not erase_report["erase_operations"] or all(
        item.get("reason") == "changing-table-or-unpreserved-area"
        or item.get("sync_bracketed")
        for item in erase_report["erase_operations"]
    )
    assert not stochastic_erase_report["erase_operations"] or all(
        item.get("reason") == "changing-table-or-unpreserved-area"
        or item.get("sync_bracketed")
        for item in stochastic_erase_report["erase_operations"]
    )
    assert {tuple(item["size"]) for item in capture["resize_sequence"]} == {
        (80, 24), (160, 50),
    }


@pytest.mark.skipif(os.name != "posix", reason="the custom driver requires a POSIX PTY")
def test_custom_driver_handles_sgr_mouse_resize_output_and_terminal_cleanup(tmp_path):
    process, master, slave, initial_attributes, result_path = _launch(
        tmp_path, "normal", "normal",
    )
    output = bytearray()
    resizes = [(0, (120, 40))]
    try:
        snapshot_path = result_path.with_suffix(".snapshot.json")
        snapshot_file = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: any(
                observation["kind"] == "beam"
                for observation in value.get("observations", ())
            ),
        )
        initial = next(
            item for item in snapshot_file["observations"]
            if item["kind"] == "beam" and item["size"] == [120, 40]
        )
        table_x, table_y, _table_width, _table_height, header_height = initial["table"]
        click_x = table_x + 5 + 1
        click_y = table_y + header_height + 1 + 1
        _send(master, f"\x1b[<0;{click_x};{click_y}M\x1b[<0;{click_x};{click_y}m".encode())

        selected = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: any(
                item.get("kind") == "beam" and item.get("selected") == "b2"
                for item in value.get("observations", ())
            ),
        )
        selected_b2 = next(
            item for item in reversed(selected["observations"])
            if item.get("kind") == "beam" and item.get("selected") == "b2"
        )
        assert selected_b2["editor_focused"]
        assert selected_b2["detail_renders"] >= 2

        pane_x, pane_y, pane_width, pane_height = selected_b2["pane"]
        wheel_x = pane_x + max(1, pane_width // 2) + 1
        wheel_y = pane_y + max(1, pane_height // 2) + 1
        _send(master, f"\x1b[<65;{wheel_x};{wheel_y}M".encode())
        scrolled = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: any(
                item.get("kind") == "beam"
                and item.get("selected") == "b2"
                and item.get("scroll_y", 0) > 0
                for item in value.get("observations", ())
            ),
        )
        assert any(
            item.get("kind") == "beam" and item.get("max_scroll_y", 0) > 0
            for item in scrolled["observations"]
        )

        # Re-click the already-selected row. No RowHighlighted event fires;
        # mouse-down and immediate command typing must still use the editor.
        beam_row_x = table_x + 7
        beam_row_y = table_y + header_height + 2
        _send(master, f"\x1b[<0;{beam_row_x};{beam_row_y}M".encode())
        repeated_click_focus = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "beam"
            and _latest_snapshot(value).get("selected") == "b2"
            and _latest_snapshot(value).get("editor_focused"),
        )
        assert _latest_snapshot(repeated_click_focus)["editor_focused"]
        _send(
            master,
            f"\x1b[<0;{beam_row_x};{beam_row_y}m".encode() + b"advance 2",
        )
        typed_command = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "beam"
            and _latest_snapshot(value).get("command_text") == "advance 2"
            and _latest_snapshot(value).get("editor_focused"),
        )
        assert _latest_snapshot(typed_command)["selected"] == "b2"

        _drain_to_quiet_display_boundary(master, output)
        _set_size(master, 80, 24)
        resizes.append((len(output), (80, 24)))
        os.kill(process.pid, signal.SIGWINCH)
        narrow = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "beam"
            and _latest_snapshot(value).get("size") == [80, 24]
            and _latest_snapshot(value).get("stacked")
            and _latest_snapshot(value).get("editor_focused")
            and _latest_snapshot(value).get("selected") == "b2"
            and _latest_snapshot(value).get("command_text") == "advance 2",
        )
        assert narrow["observations"][-1]["size"] == [80, 24]

        _drain_to_quiet_display_boundary(master, output)
        _set_size(master, 120, 40)
        resizes.append((len(output), (120, 40)))
        os.kill(process.pid, signal.SIGWINCH)
        _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "beam"
            and _latest_snapshot(value).get("size") == [120, 40]
            and _latest_snapshot(value).get("side_by_side")
            and _latest_snapshot(value).get("editor_focused")
            and _latest_snapshot(value).get("selected") == "b2"
            and _latest_snapshot(value).get("command_text") == "advance 2",
        )
        _drain_to_quiet_display_boundary(master, output)
        _set_size(master, 80, 24)
        resizes.append((len(output), (80, 24)))
        os.kill(process.pid, signal.SIGWINCH)
        narrow_again = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "beam"
            and _latest_snapshot(value).get("size") == [80, 24]
            and _latest_snapshot(value).get("stacked")
            and _latest_snapshot(value).get("editor_focused")
            and _latest_snapshot(value).get("selected") == "b2"
            and _latest_snapshot(value).get("command_text") == "advance 2",
        )

        latest_beam = next(
            item for item in reversed(narrow_again["observations"])
            if item.get("kind") == "beam" and item.get("size") == [80, 24]
        )
        editor_x, editor_y, _editor_width, _editor_height = latest_beam["editor"]
        _send(master, f"\x1b[<0;{editor_x + 4};{editor_y + 1}M".encode())
        _send(master, f"\x1b[<0;{editor_x + 4};{editor_y + 1}m".encode())
        editor_focused = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "beam"
            and _latest_snapshot(value).get("size") == [80, 24]
            and _latest_snapshot(value).get("editor_focused"),
        )
        assert any(
            item.get("kind") == "beam" and item.get("editor_focused")
            for item in editor_focused["observations"]
        )
        assert _latest_snapshot(editor_focused)["command_text"] == "advance 2"

        _drain_to_quiet_display_boundary(master, output)
        _set_size(master, 120, 40)
        resizes.append((len(output), (120, 40)))
        os.kill(process.pid, signal.SIGWINCH)
        _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "beam"
            and _latest_snapshot(value).get("size") == [120, 40]
            and _latest_snapshot(value).get("side_by_side")
            and _latest_snapshot(value).get("editor_focused"),
        )
        _drain_to_quiet_display_boundary(master, output)
        _set_size(master, 80, 24)
        resizes.append((len(output), (80, 24)))
        os.kill(process.pid, signal.SIGWINCH)
        _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "beam"
            and _latest_snapshot(value).get("size") == [80, 24]
            and _latest_snapshot(value).get("stacked")
            and _latest_snapshot(value).get("editor_focused"),
        )
        _send(master, b"\x1b[11~")
        help_open = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "beam"
            and _latest_snapshot(value).get("top_screen") == "HelpScreen"
            and _latest_snapshot(value).get("help_max_scroll_y", 0) > 0,
        )
        help_geometry = _latest_snapshot(help_open)["help_scroll_region"]
        help_x = help_geometry[0] + max(1, help_geometry[2] // 2) + 1
        help_y = help_geometry[1] + max(1, help_geometry[3] // 2) + 1
        _send(master, f"\x1b[<65;{help_x};{help_y}M".encode())
        _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "beam"
            and _latest_snapshot(value).get("top_screen") == "HelpScreen"
            and _latest_snapshot(value).get("help_scroll_y", 0) > 0,
        )
        _send(master, b"\x1b")
        _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "beam"
            and _latest_snapshot(value).get("top_screen") == "BeamScreen"
            and _latest_snapshot(value).get("editor_focused"),
        )
        typed_command = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "beam"
            and _latest_snapshot(value).get("command_text") == "advance 2",
        )
        assert _latest_snapshot(typed_command)["editor_focused"]
        _send(master, b"\rZ\x1b[<0;40;12M\x1b[<0;40;12m")

        prompt_state = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: value.get("observations", ())
            and value["observations"][-1].get("kind") == "prompt",
        )
        assert prompt_state["observations"][-1]["text"] == ""
        assert prompt_state["observations"][-1]["editor_focused"]
        _read_until(master, process, output, b"Finish? ")
        assert b"PTY_CAPTURE_MARKER" not in output

        before_log = len(output)
        _send(master, b"\x0c")
        while b"PTY_CAPTURE_MARKER" not in output[before_log:]:
            _pump(master, output, .05)
            if process.poll() is not None:
                break
        assert b"PTY_CAPTURE_MARKER" in output[before_log:]
        _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "prompt"
            and _latest_snapshot(value).get("top_screen") == "OutputScreen",
        )
        _send(master, b"q")
        _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: value.get("observations", ())
            and value["observations"][-1].get("kind") == "prompt"
            and value["observations"][-1].get("top_screen") == "PromptScreen"
            and any(
                item.get("top_screen") == "OutputScreen"
                for item in value["observations"]
            ),
        )
        _send(master, b"d")
        _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "prompt"
            and _latest_snapshot(value).get("top_screen") == "PromptScreen"
            and _latest_snapshot(value).get("text") == "d",
        )
        _send(master, b"one")
        _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: _latest_snapshot(value).get("kind") == "prompt"
            and _latest_snapshot(value).get("top_screen") == "PromptScreen"
            and _latest_snapshot(value).get("text") == "done",
        )
        _send(master, b"\r")

        deadline = time.monotonic() + 8
        while process.poll() is None and time.monotonic() < deadline:
            _pump(master, output, .05)
        assert process.poll() == 0, output[-4000:].decode("utf-8", errors="replace")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["beam"] == {
            "command": "advance 2",
            "selected_label": "b2",
        }
        assert result["prompt"] == "done"
        assert result["stats"]["stale_key_events"] > 0
        assert result["stats"]["stale_mouse_events"] > 0
        assert result["stats"]["driver_write_calls"] > 0
        assert result["stats"]["driver_write_characters"] > 0
        assert result["stats"]["screen_layout_refreshes"] > 0
        assert result["stats"]["table_fit_attempts"] > 0
        assert result["stats"]["table_column_resizes"] > 0
        assert result["stats"]["table_layout_refresh_requests"] > 0
        assert result["stats"]["beam_detail_renders"] == 2
        assert result["stats"]["rich_log_writes"] > 0
        assert result["stats"]["output_history_high_water_chars"] > 0
        assert b"\x1b[?1006h" in output
        assert b"\x1b[?1006l" in output
        assert b"\x1b[?1000l" in output
        assert b"PTY_CAPTURE_MARKER" in output
        frames = _replay_terminal_frames(bytes(output), resizes)
        top_anchored_frames = _replay_terminal_frames(
            bytes(output), resizes, preserve_top_on_resize=True,
        )
        resize_replay_comparison = []
        for offset, _size in resizes[1:]:
            pyte_resize = next(
                frame for frame in frames
                if frame["kind"] == "resize" and frame["offset"] == offset
            )
            top_resize = next(
                frame for frame in top_anchored_frames
                if frame["kind"] == "resize" and frame["offset"] == offset
            )
            pyte_text = "\n".join(pyte_resize["grid"])
            top_text = "\n".join(top_resize["grid"])
            resize_replay_comparison.append({
                "offset": offset,
                "pyte_resize_kept_heading": "BEAM" in pyte_text,
                "top_anchored_control_kept_heading": "BEAM" in top_text,
                "pyte_resize_kept_candidate": "b1" in pyte_text or "b2" in pyte_text,
                "top_anchored_control_kept_candidate": (
                    "b1" in top_text or "b2" in top_text
                ),
                "no_output_between_control_states": True,
            })
        # Preserve the raw ordered capture before any visibility assertion can
        # abort the journey. The OSC markers identify completed WriterThread
        # writes; retaining bytes alongside replay grids lets us check the
        # terminal protocol independently when a boundary looks suspicious.
        early_evidence = {
            "pty_bytes_base64": base64.b64encode(output).decode("ascii"),
            "resize_sequence": [
                {"offset": offset, "size": list(size)}
                for offset, size in resizes
            ],
            "resize_replay_comparison": resize_replay_comparison,
            "frames": [
                {
                    "kind": frame["kind"],
                    "offset": frame["offset"],
                    "generation": frame["generation"],
                    "sequence": frame["sequence"],
                    "size": list(frame["size"]),
                    "parser_ground": frame["parser_ground"],
                    "rows": list(frame["grid"]),
                }
                for frame in frames
            ],
        }
        result_path.with_suffix(".ordered-capture.json").write_text(
            json.dumps(early_evidence), encoding="utf-8",
        )
        assert frames
        assert {
            tuple(frame["size"]) for frame in frames if frame["kind"] == "resize"
        } == {(120, 40), (80, 24)}
        beam_ready = next(frame for frame in frames if frame["kind"] == "ready")
        beam_generation = beam_ready["generation"]
        compositor_boundaries = {
            frame["offset"] for frame in frames
            if frame["kind"] == "display"
        }
        assert all(offset in compositor_boundaries for offset, _size in resizes[1:]), (
            "each resize must follow a completed Textual post_display_hook boundary; "
            f"resize offsets={resizes!r}, compositor boundaries="
            f"{sorted(compositor_boundaries)!r}"
        )
        assert any(
            not item["pyte_resize_kept_heading"]
            and item["top_anchored_control_kept_heading"]
            for item in resize_replay_comparison
        ), (
            "the capture no longer reproduces pyte's resize-only heading loss; "
            f"comparison={resize_replay_comparison!r}"
        )
        beam_transitions = [
            frame for frame in frames
            if frame["kind"] == "write"
            and frame["generation"] == beam_generation
            and frame["offset"] >= beam_ready["offset"]
        ]
        assert len(beam_transitions) >= 3
        # pyte's resize policy clips removed rows from the top of its screen.
        # That operation can hide a previously rendered Beam heading without
        # any terminal bytes. Assert a complete post-resize repaint at every
        # synchronized size boundary instead of treating partial writer deltas
        # or the emulator's in-place resize as compositor display frames.
        settled_repaints = []
        for index, (resize_offset, size) in enumerate(resizes[1:]):
            next_resize = (
                resizes[index + 2][0]
                if index + 2 < len(resizes) else len(output)
            )
            in_interval = [
                frame for frame in frames
                if frame["kind"] == "display"
                and frame["generation"] == beam_generation
                and resize_offset < frame["offset"] <= next_resize
                and tuple(frame["size"]) == size
            ]
            settled = next((
                frame for frame in in_interval
                if "BEAM" in "\n".join(frame["grid"])
                and ("b1" in "\n".join(frame["grid"])
                     or "b2" in "\n".join(frame["grid"]))
                and "Beam > advance 2" in "\n".join(frame["grid"])
            ), None)
            assert settled is not None, (
                f"no complete compositor repaint after synchronized resize to {size} "
                f"at PTY offset {resize_offset}; writes={in_interval!r}"
            )
            settled_repaints.append({
                "resize_offset": resize_offset,
                "size": list(size),
                "write_offset": settled["offset"],
                "sequence": settled["sequence"],
            })
        rendered_frames = ["\n".join(frame["grid"]) for frame in frames]
        assert any("BEAM" in frame for frame in rendered_frames)
        assert any("Finish?" in frame for frame in rendered_frames)
        evidence = {
            "pty_bytes": len(output),
            "driver_write_transitions": len(beam_transitions),
            "resize_checkpoints": sum(frame["kind"] == "resize" for frame in frames),
            "resize_offset_semantics": (
                "captured after a quiet period ending at a complete writer-thread marker"
            ),
            "resize_offsets_follow_compositor_boundaries": True,
            "compositor_boundary_source": (
                "Textual 8.2.8 App.post_display_hook marker queued through the "
                "custom driver WriterThread after each App._display call"
            ),
            "settled_post_resize_repaints": settled_repaints,
            "resize_sequence": [
                {"offset": offset, "size": list(size)}
                for offset, size in resizes
            ],
            "driver_write_calls": result["stats"]["driver_write_calls"],
            "layout_refreshes": result["stats"]["screen_layout_refreshes"],
            "table_fit_attempts": result["stats"]["table_fit_attempts"],
            "frames": [
                {
                    "kind": frame["kind"],
                    "offset": frame["offset"],
                    "generation": frame["generation"],
                    "size": list(frame["size"]),
                    "rows": list(frame["grid"]),
                }
                for frame in frames
            ],
        }
        result_path.with_suffix(".frames.json").write_text(
            json.dumps(evidence), encoding="utf-8",
        )
        visual_capture = []
        for size in sorted({tuple(frame["size"]) for frame in frames}):
            grids = [frame["grid"] for frame in frames if tuple(frame["size"]) == size]
            rows = [row for grid in grids for row in grid]
            visual_capture.append({
                "size": list(size),
                "beam_visible": any("BEAM" in row for row in rows),
                "command_visible": any("Beam > advance 2" in row for row in rows),
                "finish_visible": any("Finish?" in row for row in rows),
                "max_rendered_row_width": max(map(len, rows), default=0),
                "occupied_rows": max(
                    (sum(bool(row.strip()) for row in grid) for grid in grids),
                    default=0,
                ),
            })
        assert all(frame["beam_visible"] for frame in visual_capture)
        assert all(frame["command_visible"] for frame in visual_capture)
        assert all(frame["max_rendered_row_width"] <= frame["size"][0] for frame in visual_capture)
        frame_snapshot = _FRAME_SNAPSHOT_DIR / "beam-pty-resize-frames.json"
        if os.environ.get("UPDATE_TEXTUAL_SNAPSHOTS") == "1":
            frame_snapshot.write_text(
                json.dumps(visual_capture, indent=2) + "\n", encoding="utf-8",
            )
        assert frame_snapshot.exists(), f"missing PTY frame snapshot: {frame_snapshot}"
        assert visual_capture == json.loads(frame_snapshot.read_text(encoding="utf-8"))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        final_attributes = termios.tcgetattr(slave)
        os.close(master)
        os.close(slave)
    assert final_attributes == initial_attributes


@pytest.mark.skipif(os.name != "posix", reason="the custom driver requires a POSIX PTY")
def test_custom_driver_restores_terminal_modes_after_interrupt(tmp_path):
    process, master, slave, initial_attributes, result_path = _launch(
        tmp_path, "interrupt", "interrupt", size=(80, 24),
    )
    output = bytearray()
    try:
        _read_until(master, process, output, b"Interrupt now? ")
        time.sleep(.2)
        active_attributes = termios.tcgetattr(slave)
        assert not (active_attributes[3] & termios.ISIG)
        _send(master, b"\x03")
        deadline = time.monotonic() + 8
        while process.poll() is None and time.monotonic() < deadline:
            _pump(master, output, .05)
        assert process.poll() == 0, output[-4000:].decode("utf-8", errors="replace")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["interrupted"]
        assert result["stats"]["driver_write_calls"] > 0
        assert b"\x1b[?1006l" in output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        final_attributes = termios.tcgetattr(slave)
        os.close(master)
        os.close(slave)
    assert final_attributes == initial_attributes

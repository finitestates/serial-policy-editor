"""Exercise the custom Textual driver through real POSIX terminal input."""

from __future__ import annotations

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


class CaptureOutput:
    """Forward terminal output and add invisible, ordered capture markers."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sequence = 0
        self._generation = lambda: 0

    def write(self, data):
        with self._lock:
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
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from tests.fakes import ConformingFakeBackend
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_session import LiveSession, LiveSessionRoster
from trajectory_editor.session_runtime import run_session_roster
from trajectory_editor.textual_tui import (
    BeamScreen, ChoiceScreen, EdgeScreen, PromptScreen, TextualTerminalSession,
)
from trajectory_editor.tui import TerminalIO

result_path = Path(sys.argv[1])
snapshot_path = result_path.with_suffix(".snapshot.json")
arm_path = result_path.with_suffix(".arm")
waiting_path = result_path.with_suffix(".waiting")
release_path = result_path.with_suffix(".release")
observations = []
audit = {}
beam_inputs = []

original_read_beam = TextualTerminalSession.read_beam
def audited_read_beam(target, state):
    value = original_read_beam(target, state)
    beam_inputs.append({
        "command": value.command if value is not None else None,
        "selected_label": value.selected_label if value is not None else None,
    })
    return value
TextualTerminalSession.read_beam = audited_read_beam


class GatedBackend(ConformingFakeBackend):
    def eval(self, token_ids):
        if arm_path.exists() and not waiting_path.exists():
            waiting_path.write_text("waiting", encoding="utf-8")
            os.write(1, b"PTY_NATIVE_CAPTURE_MARKER")
            while not release_path.exists():
                time.sleep(.01)
        super().eval(token_ids)


def save_snapshot(value):
    temporary = snapshot_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(snapshot_path)


def inspect_screen(app):
    screen = app._active_screen
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
    editor = screen.query_one(selector)
    return {
        "kind": kind,
        "generation": generation,
        "accepting": screen.accepting_input,
        "text": getattr(editor, "text", getattr(editor, "value", "")),
        "focused": editor.has_focus,
        "read_only": getattr(editor, "read_only", False),
        "selected": selected,
    }


io = TerminalIO()
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
    audit["beam_inputs"] = beam_inputs

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


def _replay_terminal_frames(output: bytes, resizes):
    """Replay ordered driver-write and resize boundaries from a PTY capture."""
    if not resizes or resizes[0][0] != 0:
        raise AssertionError("the PTY capture must start with its initial dimensions")
    columns, rows = resizes[0][1]
    screen = pyte.Screen(columns, rows)
    stream = pyte.Stream(screen)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    checkpoints = []
    segment_start = 0

    def feed_until(offset):
        nonlocal segment_start
        if offset < segment_start:
            raise AssertionError("PTY capture boundaries must be ordered")
        segment = output[segment_start:offset]
        decoded = decoder.decode(segment, final=False)
        if decoded:
            stream.feed(decoded)
        segment_start = offset

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

    current_size = resizes[0][1]
    events = [
        (offset, 2, "resize", {"size": size})
        for offset, size in resizes[1:]
    ]
    marker_pattern = re.compile(
        rb"\x1b\]777;SPEPTY;(\d+);(\d+);(write|ready)\x07"
    )
    for match in marker_pattern.finditer(output):
        events.append((
            match.end(), 1, match.group(3).decode("ascii"),
            {
                "sequence": int(match.group(1)),
                "generation": int(match.group(2)),
            },
        ))
    events.sort(key=lambda event: (event[0], event[1]))

    for offset, _order, kind, metadata in events:
        feed_until(offset)
        if kind == "resize":
            current_size = metadata["size"]
            columns, rows = current_size
            screen.resize(lines=rows, columns=columns)
        checkpoints.append({
            "kind": kind,
            "offset": offset,
            "size": current_size,
            "generation": metadata.get("generation"),
            "sequence": metadata.get("sequence"),
            "parser_ground": stream._taking_plain_text is True,
            "beam_prompt_styles": prompt_cell_styles(),
            "grid": tuple(screen.display),
        })

    feed_until(len(output))
    final_text = decoder.decode(b"", final=True)
    if final_text:
        stream.feed(final_text)
    checkpoints.append({
        "kind": "eof",
        "offset": len(output),
        "size": current_size,
        "generation": None,
        "sequence": None,
        "parser_ground": stream._taking_plain_text is True,
        "beam_prompt_styles": prompt_cell_styles(),
        "grid": tuple(screen.display),
    })
    return checkpoints


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

        _set_size(master, 80, 24)
        _drain_available(master, output)
        resizes.append((len(output), (80, 24)))
        os.kill(process.pid, signal.SIGWINCH)
        narrow = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: any(
                item.get("kind") == "beam"
                and item.get("size") == [80, 24]
                and item.get("stacked")
                and item.get("editor_focused")
                and item.get("selected") == "b2"
                for item in value.get("observations", ())
            ),
        )
        assert narrow["observations"][-1]["size"] == [80, 24]

        _set_size(master, 120, 40)
        _drain_available(master, output)
        resizes.append((len(output), (120, 40)))
        os.kill(process.pid, signal.SIGWINCH)
        _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: any(
                item.get("kind") == "beam"
                and item.get("size") == [120, 40]
                and item.get("side_by_side")
                and item.get("editor_focused")
                and item.get("selected") == "b2"
                for item in value.get("observations", ())
            ),
        )
        _set_size(master, 80, 24)
        _drain_available(master, output)
        resizes.append((len(output), (80, 24)))
        os.kill(process.pid, signal.SIGWINCH)
        narrow_again = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: any(
                item.get("kind") == "beam"
                and item.get("size") == [80, 24]
                and item.get("stacked")
                and item.get("selected") == "b2"
                for item in value.get("observations", ())
            ),
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

        _set_size(master, 120, 40)
        _drain_available(master, output)
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
        _set_size(master, 80, 24)
        _drain_available(master, output)
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
        assert frames
        assert {
            tuple(frame["size"]) for frame in frames if frame["kind"] == "resize"
        } == {(120, 40), (80, 24)}
        beam_ready = next(frame for frame in frames if frame["kind"] == "ready")
        beam_generation = beam_ready["generation"]
        beam_transitions = [
            frame for frame in frames
            if frame["kind"] == "write"
            and frame["generation"] == beam_generation
            and frame["offset"] >= beam_ready["offset"]
        ]
        assert len(beam_transitions) >= 3
        # A pyte resize checkpoint models viewport reflow. The next driver
        # write can be a no-op at the screen level, so assert the actual
        # write transitions whose visible grid changed.
        visible_beam_transitions = []
        prior_grid = beam_ready["grid"]
        for frame in frames:
            if frame["offset"] < beam_ready["offset"]:
                continue
            if (
                frame["kind"] == "write"
                and frame["generation"] == beam_generation
                and frame["grid"] != prior_grid
            ):
                visible_beam_transitions.append(frame)
            prior_grid = frame["grid"]
        assert visible_beam_transitions
        for frame in visible_beam_transitions:
            rendered = "\n".join(frame["grid"])
            frame_index = frames.index(frame)
            nearby = frames[max(0, frame_index - 1):frame_index + 2]
            transition_context = [
                {
                    "kind": item["kind"],
                    "offset": item["offset"],
                    "size": item["size"],
                    "generation": item["generation"],
                    "parser_ground": item["parser_ground"],
                    "beam_prompt_styles": item["beam_prompt_styles"],
                    "grid": [
                        (row_number, row.rstrip())
                        for row_number, row in enumerate(item["grid"])
                        if row.strip()
                    ],
                }
                for item in nearby
            ]
            assert "BEAM" in rendered, (
                f"Beam heading disappeared at PTY offset {frame['offset']} "
                f"during request generation {beam_generation}; "
                f"surrounding transitions={transition_context!r}"
            )
            assert "b1" in rendered or "b2" in rendered, (
                f"Beam candidate pane disappeared at PTY offset {frame['offset']} "
                f"during request generation {beam_generation}; "
                f"surrounding transitions={transition_context!r}"
            )
            assert "Beam >" in rendered, (
                f"Beam command editor disappeared at PTY offset {frame['offset']} "
                f"during request generation {beam_generation}; "
                f"surrounding transitions={transition_context!r}"
            )
        rendered_frames = ["\n".join(frame["grid"]) for frame in frames]
        assert any("BEAM" in frame for frame in rendered_frames)
        assert any("Finish?" in frame for frame in rendered_frames)
        evidence = {
            "pty_bytes": len(output),
            "driver_write_transitions": len(beam_transitions),
            "resize_checkpoints": sum(frame["kind"] == "resize" for frame in frames),
            "resize_offset_semantics": (
                "best synchronized after draining currently readable PTY bytes"
            ),
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

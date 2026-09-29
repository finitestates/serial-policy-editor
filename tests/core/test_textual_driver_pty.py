"""Exercise the custom Textual driver through real POSIX terminal input."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import select
import signal
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.current_workflow

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

    with TextualTerminalSession() as terminal:
        app = terminal.application

        def inspect_screen():
            screen = app._active_screen
            if isinstance(screen, BeamScreen) and screen.accepting_input:
                driver = app._driver
                table = screen.query_one("#beam-table", DataTable)
                pane = screen.query_one("#beam-detail-pane")
                editor = screen.query_one("#beam-input", TextArea)
                return {
                    "kind": "beam",
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


def _launch(tmp_path: Path, name: str, mode: str, size=(120, 40)):
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
        [sys.executable, "-c", _CHILD, str(result_path), mode],
        cwd=repository,
        env=environment,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
        start_new_session=True,
        preexec_fn=acquire_controlling_tty,
    )
    return process, master, slave, initial_attributes, result_path


def _send(master: int, text: bytes) -> None:
    os.write(master, text)


def _latest_snapshot(value):
    observations = value.get("observations", ())
    return observations[-1] if observations else {}


@pytest.mark.skipif(os.name != "posix", reason="the custom driver requires a POSIX PTY")
def test_custom_driver_handles_sgr_mouse_resize_output_and_terminal_cleanup(tmp_path):
    process, master, slave, initial_attributes, result_path = _launch(
        tmp_path, "normal", "normal",
    )
    output = bytearray()
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

        # Clicking the table header focuses the table without changing its
        # selected row; row clicks intentionally return focus to the editor.
        _send(master, f"\x1b[<0;{table_x + 6};{table_y + 1}M".encode())
        _send(master, f"\x1b[<0;{table_x + 6};{table_y + 1}m".encode())
        table_focused = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: any(
                item.get("kind") == "beam"
                and item.get("selected") == "b2"
                and item.get("table_focused")
                for item in value.get("observations", ())
            ),
        )

        _set_size(master, 80, 24)
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
                and item.get("table_focused")
                and item.get("selected") == "b2"
                for item in value.get("observations", ())
            ),
        )
        assert narrow["observations"][-1]["size"] == [80, 24]

        _set_size(master, 120, 40)
        os.kill(process.pid, signal.SIGWINCH)
        side_by_side = _wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: any(
                item.get("kind") == "beam"
                and item.get("size") == [120, 40]
                and item.get("side_by_side")
                and item.get("table_focused")
                and item.get("selected") == "b2"
                for item in value.get("observations", ())
            ),
        )
        _set_size(master, 80, 24)
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
        _send(master, b"advance 2")
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

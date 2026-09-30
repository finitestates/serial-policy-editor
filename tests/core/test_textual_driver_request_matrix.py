"""Production-driver coverage for less common request and paste lifecycles."""

from __future__ import annotations

import json
import os
import termios
import time
from pathlib import Path

import pytest

from tests.core.test_textual_driver_pty import _launch, _pump, _wait_json

pytestmark = [pytest.mark.current_workflow, pytest.mark.skipif(
    os.name != "posix", reason="the custom driver requires a POSIX PTY",
)]

_MATRIX_CHILD = r'''
import json
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from textual.containers import VerticalScroll
from trajectory_editor.terminal_contracts import BoundaryReview, PromptRequest
from trajectory_editor.textual_tui import ChoiceScreen, PromptScreen, TextualTerminalSession
from tests.core.textual_support import choice_state

result_path = Path(sys.argv[1])
snapshot_path = result_path.with_suffix(".snapshot.json")
mode = sys.argv[2]
observations = []
stop = threading.Event()

def save(value):
    temporary = snapshot_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(snapshot_path)

with TextualTerminalSession() as terminal:
    app = terminal.application

    def inspect():
        screen = app._active_screen
        if screen is None:
            return {"kind": "handoff", "accepting": False}
        if isinstance(screen, ChoiceScreen):
            kind = "review" if screen.state.review is not None else "choice"
            editor = screen.query_one("#choice-input")
            return {
                "kind": kind, "generation": screen.lifecycle.generation,
                "accepting": screen.accepting_input, "text": editor.text,
            }
        if isinstance(screen, PromptScreen):
            request = screen.state
            kind = (
                "page" if request.page else
                "single" if request.single_key else
                "multiline" if request.multiline else
                "chord" if request.isolated else "prompt"
            )
            selector = "#multiline-input" if request.multiline else "#prompt-input"
            text = screen.query_one(selector).text if request.multiline else (
                screen.query_one(selector).value if not request.page and not request.single_key else ""
            )
            return {
                "kind": kind, "generation": screen.lifecycle.generation,
                "accepting": screen.accepting_input, "text": text,
                "escape_pending": screen._escape_pending,
                "focused": app.focused.id if app.focused is not None else None,
                "stale_input": app.stats["stale_input_events"],
                "top_screen": type(app.screen).__name__,
                "page_scroll_y": (
                    screen.query_one("#page-scroll", VerticalScroll).scroll_y
                    if request.page else None
                ),
            }
        return None

    def watch():
        previous = None
        while not stop.is_set():
            try:
                current = app.call_from_thread(inspect)
            except RuntimeError:
                return
            if current != previous:
                observations.append(current)
                save({"observations": observations})
                previous = current
            time.sleep(.01)

    watcher = threading.Thread(target=watch, name="request-matrix-observer")
    watcher.start()
    if mode == "review":
        state = replace(
            choice_state(),
            review=BoundaryReview(2, 1, "historical context", {"kind": "token-boundary"}),
            seamless=False,
            reactivate_on_review_enter=False,
        )
        value = terminal.read_choice(state)
    elif mode == "page":
        value = terminal.prompt(PromptRequest("Page", body="page body", page=True))
    elif mode == "chord":
        value = terminal.prompt(PromptRequest(
            "Chord > ", body="a (1) | b (2)", isolated=True,
        ))
    elif mode == "single":
        value = terminal.prompt(PromptRequest("Press one key", single_key=True))
    elif mode == "boundaries":
        long_page = "\n".join(f"Page line {line}" for line in range(100))
        first_page = terminal.prompt(PromptRequest("First page", body=long_page, page=True))
        second_page = terminal.prompt(PromptRequest("Second page", body="second", page=True))
        first_key = terminal.prompt(PromptRequest("First key", single_key=True))
        second_key = terminal.prompt(PromptRequest("Second key", single_key=True))
        value = [first_page, second_page, first_key, second_key]
    else:
        value = terminal.prompt(PromptRequest(
            "New prompt > ", multiline=True, isolated=True,
        ))
    stop.set()
    watcher.join(timeout=2)
    driver = app._driver
    session_thread = terminal._thread

workers = {
    "session": session_thread.is_alive(),
    "input": bool(driver._key_thread and driver._key_thread.is_alive()),
    "output": bool(driver._writer_thread and driver._writer_thread.is_alive()),
}
result_path.write_text(json.dumps({"value": value, "workers": workers}), encoding="utf-8")
'''


_PASTE_CHILD = r'''
import json
import sys
import threading
import time
from pathlib import Path

from textual import events
from trajectory_editor.terminal_contracts import PromptRequest
from trajectory_editor.textual_tui import BeamScreen, PromptScreen, TextualTerminalSession
from tests.core.textual_support import beam_state

result_path = Path(sys.argv[1])
snapshot_path = result_path.with_suffix(".snapshot.json")
waiting_path = result_path.with_suffix(".waiting")
release_path = result_path.with_suffix(".release")
paste_audit = []
observations = []
stop = threading.Event()

def save(value):
    temporary = snapshot_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(snapshot_path)

with TextualTerminalSession() as terminal:
    app = terminal.application
    original_on_event = app.on_event

    async def audited_on_event(event):
        if isinstance(event, events.Paste):
            screen = app._active_screen
            before = {
                "text": event.text,
                "kind": type(screen).__name__ if screen is not None else None,
                "generation": screen.lifecycle.generation if screen is not None else None,
                "accepting": screen.accepting_input if screen is not None else False,
                "stale_before": app.stats["stale_input_events"],
            }
            await original_on_event(event)
            before["stale_after"] = app.stats["stale_input_events"]
            if not any(
                item.get("text") == before["text"]
                and item.get("kind") == before["kind"]
                and item.get("generation") == before["generation"]
                for item in paste_audit
            ):
                paste_audit.append(before)
            save({"paste_audit": paste_audit, "observations": observations})
            return
        await original_on_event(event)

    app.on_event = audited_on_event

    def inspect():
        screen = app._active_screen
        if isinstance(screen, BeamScreen):
            return {
                "kind": "beam", "generation": screen.lifecycle.generation,
                "accepting": screen.accepting_input,
                "text": screen.query_one("#beam-input").text,
            }
        if isinstance(screen, PromptScreen) and screen.state.multiline:
            return {
                "kind": "multiline", "generation": screen.lifecycle.generation,
                "accepting": screen.accepting_input,
                "text": screen.query_one("#multiline-input").text,
                "escape_pending": screen._escape_pending,
            }
        return None

    def watch():
        previous = None
        while not stop.is_set():
            try:
                current = app.call_from_thread(inspect)
            except RuntimeError:
                return
            if current is not None and current != previous:
                observations.append(current)
                save({"paste_audit": paste_audit, "observations": observations})
                previous = current
            time.sleep(.01)

    watcher = threading.Thread(target=watch, name="paste-lifecycle-observer")
    watcher.start()
    first = terminal.read_beam(beam_state())
    waiting_path.write_text("waiting", encoding="utf-8")
    while not release_path.exists():
        time.sleep(.005)
    second = terminal.prompt(PromptRequest("Compose", multiline=True, isolated=True))
    stop.set()
    watcher.join(timeout=2)
    driver = app._driver
    session_thread = terminal._thread

workers = {
    "session": session_thread.is_alive(),
    "input": bool(driver._key_thread and driver._key_thread.is_alive()),
    "output": bool(driver._writer_thread and driver._writer_thread.is_alive()),
}
result_path.write_text(json.dumps({
    "first": {"command": first.command, "selected": first.selected_label},
    "second": second, "paste_audit": paste_audit, "workers": workers,
}), encoding="utf-8")
'''


def _latest(value: dict) -> dict:
    rows = value.get("observations", ())
    return rows[-1] if rows else {}


def _finish(process, master: int, output: bytearray, result_path: Path) -> dict:
    deadline = time.monotonic() + 3
    while process.poll() is None and time.monotonic() < deadline:
        _pump(master, output, .05)
    assert process.poll() == 0, output[-4000:].decode("utf-8", errors="replace")
    return json.loads(result_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("mode", "kind", "expected"),
    [
        ("review", "review", "\x1b"),
        ("page", "page", ""),
        ("chord", "chord", "a"),
        ("single", "single", "z"),
        ("multiline", "multiline", "first line\nsecond \u20ac line"),
    ],
)
def test_actual_driver_submits_review_page_chord_single_key_and_multiline_values(
    tmp_path, mode, kind, expected,
):
    process, master, slave, initial_attributes, result_path = _launch(
        tmp_path, f"request-{mode}", mode, child=_MATRIX_CHILD,
    )
    output = bytearray()
    snapshot_path = result_path.with_suffix(".snapshot.json")
    try:
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: any(
                row.get("kind") == kind
                and row.get("generation") == 1
                and row.get("accepting")
                and (
                    mode not in {"page", "single"}
                    or row.get("focused") == (
                        "page-return" if mode == "page" else "single-key-hint"
                    )
                )
                for row in value.get("observations", ())
            ),
        )
        if mode == "review":
            _send(master, b"\r")
        elif mode == "page":
            _send(master, b"q")
        elif mode == "chord":
            _send(master, b"a")
            _wait_json(
                master, process, output, snapshot_path,
                lambda value: _latest(value).get("kind") == "chord"
                and _latest(value).get("text") == "a",
            )
            _send(master, b"\r")
        elif mode == "single":
            _send(master, b"z")
        else:
            _send(master, b"\x1b[200~first line\nsecond \xe2\x82\xac line\x1b[201~")
            _wait_json(
                master, process, output, snapshot_path,
                lambda value: _latest(value).get("kind") == "multiline"
                and _latest(value).get("text") == expected,
            )
            _send(master, b"\x1b")
            _wait_json(
                master, process, output, snapshot_path,
                lambda value: _latest(value).get("kind") == "multiline"
                and _latest(value).get("escape_pending"),
            )
            _send(master, b"\r")
        result = _finish(process, master, output, result_path)
        assert result["value"] == expected
        assert result["workers"] == {"session": False, "input": False, "output": False}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        final_attributes = termios.tcgetattr(slave)
        os.close(master)
        os.close(slave)
    assert final_attributes == initial_attributes


def test_actual_driver_page_and_single_key_ownership_survives_boundaries_and_modals(tmp_path):
    process, master, slave, initial_attributes, result_path = _launch(
        tmp_path, "request-boundaries", "boundaries", child=_MATRIX_CHILD,
    )
    output = bytearray()
    snapshot_path = result_path.with_suffix(".snapshot.json")
    try:
        def wait_for(kind, generation, **conditions):
            return _wait_json(
                master, process, output, snapshot_path,
                lambda value: (
                    (row := _latest(value)).get("kind") == kind
                    and row.get("generation") == generation
                    and row.get("accepting")
                    and all(row.get(name) == expected for name, expected in conditions.items())
                ),
            )

        wait_for("page", 1, focused="page-return")
        _send(master, b"\x1b[6~")
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: any(
                row.get("generation") == 1 and row.get("page_scroll_y", 0) > 0
                for row in value.get("observations", ())
            ),
        )
        _send(master, b"\x1b[5~")
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: any(
                row.get("generation") == 1 and row.get("page_scroll_y") == 0
                for row in value.get("observations", ())
            ),
        )
        _send(master, b"\x1b[11~")
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: _latest(value).get("top_screen") == "HelpScreen",
        )
        _send(master, b"\x1b")
        wait_for("page", 1, focused="page-return", top_screen="PromptScreen")
        _send(master, b"q")
        wait_for("page", 2, focused="page-return")
        _send(master, b"\r")
        wait_for("single", 3, focused="single-key-hint")
        _send(master, b"\x1b[11~")
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: _latest(value).get("top_screen") == "HelpScreen",
        )
        _send(master, b"\x1b")
        wait_for("single", 3, focused="single-key-hint", top_screen="PromptScreen")
        _send(master, b"z")
        wait_for("single", 4, focused="single-key-hint")
        _send(master, b"\x0c")
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: _latest(value).get("top_screen") == "OutputScreen",
        )
        _send(master, b"q")
        wait_for("single", 4, focused="single-key-hint", top_screen="PromptScreen")
        _send(master, b"\x7f")
        result = _finish(process, master, output, result_path)
        assert result["value"] == ["", "", "z", "\x7f"]
        assert result["workers"] == {"session": False, "input": False, "output": False}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        final_attributes = termios.tcgetattr(slave)
        os.close(master)
        os.close(slave)
    assert final_attributes == initial_attributes


def test_actual_driver_rejects_stale_bracketed_paste_and_keeps_fresh_paste_exact(tmp_path):
    process, master, slave, initial_attributes, result_path = _launch(
        tmp_path, "paste-handoff", "paste", child=_PASTE_CHILD,
    )
    output = bytearray()
    snapshot_path = result_path.with_suffix(".snapshot.json")
    try:
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: any(
                row.get("kind") == "beam" and row.get("accepting")
                for row in value.get("observations", ())
            ),
        )
        _send(master, b"advance 1\r")
        deadline = time.monotonic() + 8
        while not result_path.with_suffix(".waiting").exists() and time.monotonic() < deadline:
            _pump(master, output, .03)
        assert result_path.with_suffix(".waiting").exists()

        stale_text = "STALE LINE ONE\nSTALE LINE TWO"
        _send(master, b"\x1b[200~" + stale_text.encode() + b"\x1b[201~")
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: any(
                item.get("text") == stale_text
                and item.get("kind") is None
                and item.get("stale_after", 0) > item.get("stale_before", 0)
                for item in value.get("paste_audit", ())
            ),
        )
        result_path.with_suffix(".release").touch()
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: any(
                row.get("kind") == "multiline"
                and row.get("generation") == 2
                and row.get("accepting")
                and row.get("text") == ""
                for row in value.get("observations", ())
            ),
        )

        fresh_text = "FRESH LINE ONE\nFRESH \u20ac LINE TWO"
        _send(master, b"\x1b[200~" + fresh_text.encode() + b"\x1b[201~")
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: any(
                row.get("kind") == "multiline"
                and row.get("generation") == 2
                and row.get("text") == fresh_text
                for row in value.get("observations", ())
            ) and sum(item.get("text") == fresh_text for item in value.get("paste_audit", ())) == 1,
        )
        _send(master, b"\x1b")
        _wait_json(
            master, process, output, snapshot_path,
            lambda value: _latest(value).get("kind") == "multiline"
            and _latest(value).get("escape_pending"),
        )
        _send(master, b"\r")
        result = _finish(process, master, output, result_path)
        assert result["first"] == {"command": "select b1", "selected": "b1"}
        assert result["second"] == fresh_text
        assert [item["text"] for item in result["paste_audit"]] == [stale_text, fresh_text]
        assert result["paste_audit"][0]["kind"] is None
        assert result["paste_audit"][0]["stale_after"] > result["paste_audit"][0]["stale_before"]
        assert result["paste_audit"][1]["kind"] == "PromptScreen"
        assert result["paste_audit"][1]["accepting"] is True
        assert result["workers"] == {"session": False, "input": False, "output": False}
        assert "STALE LINE" not in result["second"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        final_attributes = termios.tcgetattr(slave)
        os.close(master)
        os.close(slave)
    assert final_attributes == initial_attributes


def _send(master: int, value: bytes) -> None:
    os.write(master, value)

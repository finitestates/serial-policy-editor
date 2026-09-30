"""Reproducible Beam edit/rewind journeys used by record_textual_journey.py."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPOSITORY), str(REPOSITORY / "core" / "src")]
SUPPORT_PATH = REPOSITORY / "tests" / "core" / "test_textual_driver_pty.py"
SPEC = importlib.util.spec_from_file_location("_textual_driver_pty_support", SUPPORT_PATH)
assert SPEC is not None and SPEC.loader is not None
SUPPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUPPORT
SPEC.loader.exec_module(SUPPORT)


def _wait_for(
    master: int,
    process,
    output: bytearray,
    snapshot_path: Path,
    kind: str,
    after_generation: int = -1,
    *,
    text: str | None = None,
    timeout: float = 8,
) -> dict:
    def matches(item: dict) -> bool:
        return (
            item.get("kind") == kind
            and item.get("generation", -1) > after_generation
            and item.get("accepting") is True
            and (text is None or item.get("text") == text)
        )

    value = SUPPORT._wait_json(
        master,
        process,
        output,
        snapshot_path,
        lambda state: any(matches(item) for item in state.get("observations", ())),
        timeout=timeout,
    )
    return next(
        item for item in reversed(value.get("observations", ())) if matches(item)
    )


def _enter_choice_command(
    master: int,
    process,
    output: bytearray,
    snapshot_path: Path,
    command: str,
    generation: int,
) -> None:
    first, rest = command[:1], command[1:]
    SUPPORT._send(master, first.encode())
    SUPPORT._wait_json(
        master,
        process,
        output,
        snapshot_path,
        lambda value: any(
            item.get("kind") == "choice"
            and item.get("generation") == generation
            and item.get("accepting") is True
            and item.get("text") == first
            for item in value.get("observations", ())
        ),
    )
    if rest:
        SUPPORT._send(master, rest.encode())
        SUPPORT._wait_json(
            master,
            process,
            output,
            snapshot_path,
            lambda value: any(
                item.get("kind") == "choice"
                and item.get("generation") == generation
                and item.get("accepting") is True
                and item.get("text") == command
                for item in value.get("observations", ())
            ),
        )
    SUPPORT._send(master, b"\r")


def _depth(screen: dict) -> int:
    match = re.search(r"depth (\d+)", screen.get("heading", ""))
    assert match is not None, screen.get("heading")
    return int(match.group(1))


def _save_capture(
    result_path: Path,
    output: bytearray,
    snapshot_path: Path,
    size: tuple[int, int],
) -> None:
    snapshot = {}
    if snapshot_path.exists():
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    result = {}
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    audit = result.get("audit", {})
    frame_path = result_path.with_suffix(".frames.json")
    child_frames = audit.get("capture_frames", [])
    if not child_frames and frame_path.exists():
        try:
            child_frames = json.loads(frame_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            child_frames = []
    regions_by_size = {}
    for item in snapshot.get("observations", ()):
        if item.get("kind") == "beam" and item.get("regions"):
            regions_by_size[tuple(item["size"])] = {
                name: tuple(region) for name, region in item["regions"].items()
            }
    replay_error = None
    ordered_events = []
    try:
        ordered_events = SUPPORT._replay_terminal_frames(
            bytes(output),
            [(0, size)],
            style_regions=regions_by_size,
            display_metadata={
                int(frame["sequence"]): frame
                for frame in child_frames
                if "sequence" in frame
            },
            sync_available=bool(audit.get("sync_available", True)),
        )
    except Exception as error:  # retain the PTY bytes even if the parser fails
        replay_error = f"{type(error).__name__}: {error}"
    for frame in ordered_events:
        metadata = frame.get("frame_metadata")
        if metadata is not None:
            frame["screen_id"] = metadata.get("screen_id")
            frame["characters"] = metadata.get("characters")
            frame["generation"] = metadata.get("generation")
            frame["geometry"] = metadata.get("geometry")
    raw_path = result_path.with_suffix(".rewind.pty")
    raw_path.write_bytes(bytes(output))
    capture_path = result_path.with_suffix(".rewind.capture.json")
    capture_path.write_text(
        json.dumps(
            {
                "raw_pty_path": str(raw_path),
                "resize_sequence": [{"offset": 0, "size": list(size)}],
                "sync_available": audit.get("sync_available"),
                "child_frame_metadata": child_frames,
                "observations": snapshot.get("observations", []),
                "ordered_events": ordered_events,
                "replay_error": replay_error,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _close_runtime(master: int, process, output: bytearray, snapshot_path: Path) -> None:
    if process.poll() is not None:
        return
    state = {}
    try:
        state = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    latest = state.get("observations", [{}])[-1]
    try:
        if latest.get("kind") == "beam":
            SUPPORT._send(master, b"\x1b")
            _wait_for(
                master, process, output, snapshot_path, "choice",
                latest.get("generation", -1), timeout=2,
            )
            latest = json.loads(snapshot_path.read_text(encoding="utf-8"))[
                "observations"
            ][-1]
        if latest.get("kind") == "choice":
            generation = latest.get("generation", -1)
            SUPPORT._send(master, b"q")
            SUPPORT._wait_json(
                master, process, output, snapshot_path,
                lambda value: any(
                    item.get("kind") == "choice"
                    and item.get("generation") == generation
                    and item.get("text") == "q"
                    for item in value.get("observations", ())
                ),
                timeout=2,
            )
            SUPPORT._send(master, b"\r")
            latest = _wait_for(
                master, process, output, snapshot_path, "edge", generation,
                timeout=2,
            )
        if latest.get("kind") == "edge":
            SUPPORT._send(master, b"\x04")
            deadline = time.monotonic() + 2
            while process.poll() is None and time.monotonic() < deadline:
                SUPPORT._pump(master, output, .03)
    except (AssertionError, OSError, KeyError, IndexError):
        pass
    if process.poll() is None:
        process.kill()
        process.wait(timeout=2)


@pytest.mark.skipif(os.name != "posix", reason="the custom driver requires a POSIX PTY")
@pytest.mark.parametrize("edit", ["delete-row", "delete-count"])
def test_runtime_can_rewind_after_a_beam_row_edit(tmp_path, edit):
    size = (80, 24)
    process, master, slave, initial_attributes, result_path = SUPPORT._launch(
        tmp_path,
        f"rewind-{edit}",
        "beam-no-flash",
        size=size,
        child=SUPPORT._RUNTIME_CHILD,
    )
    output = bytearray()
    snapshot_path = result_path.with_suffix(".snapshot.json")
    try:
        SUPPORT._answer_synchronized_output_probe(master, process, output)
        choice = _wait_for(master, process, output, snapshot_path, "choice")
        generation = choice["generation"]
        _enter_choice_command(
            master, process, output, snapshot_path, "beam 2", generation,
        )
        beam = _wait_for(master, process, output, snapshot_path, "beam", generation)
        generation = beam["generation"]
        for _ in range(3):
            SUPPORT._send(master, b"\x1b[C")
            beam = _wait_for(
                master, process, output, snapshot_path, "beam", generation,
            )
            generation = beam["generation"]
        depth_before_edit = _depth(beam)

        if edit == "delete-row":
            removed_label = beam["selected"]
            SUPPORT._send(master, b"\x7f")
            edited = _wait_for(
                master, process, output, snapshot_path, "beam", generation,
            )
            assert removed_label not in [row["label"] for row in edited["rows"]]
        else:
            SUPPORT._send(master, b"advance 1")
            edited = _wait_for(
                master, process, output, snapshot_path, "beam", generation,
                text="advance 1",
            )
            SUPPORT._send(master, b"\x7f")
            edited = _wait_for(
                master, process, output, snapshot_path, "beam", generation,
                text="advance ",
            )

        SUPPORT._send(master, b"\x1b[D")
        rewound = _wait_for(
            master, process, output, snapshot_path, "beam",
            edited["generation"],
            timeout=3,
        )
        assert _depth(rewound) == depth_before_edit - 1, (
            f"Left Arrow did not rewind after {edit}: "
            f"before={depth_before_edit}, after={_depth(rewound)}, "
            f"input={edited.get('text')!r}"
        )
    finally:
        _close_runtime(master, process, output, snapshot_path)
        try:
            final_attributes = __import__("termios").tcgetattr(slave)
        finally:
            os.close(master)
            os.close(slave)
        _save_capture(result_path, output, snapshot_path, size)
    assert final_attributes == initial_attributes

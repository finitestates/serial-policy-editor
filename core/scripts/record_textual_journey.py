#!/usr/bin/env python3
"""Record scripted Textual terminal journeys and preserve every captured frame.

Run from the repository root, using the core test environment::

    core/.venv/bin/python core/scripts/record_textual_journey.py --scenario beam

The selected pytest journey launches the production Textual session in a POSIX
PTY, sends scripted input, and replays the emitted terminal stream. This wrapper
keeps pytest's raw captures, exports every marked display grid as a readable
text file, and reports the artifact directory even when an assertion fails.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import strftime


BEAM_TEST = (
    "tests/core/test_textual_driver_pty.py::"
    "test_runtime_beam_advances_keep_one_complete_painted_screen"
)
SCENARIOS = {
    "beam": [BEAM_TEST],
    "rewind": ["core/scripts/test_runtime_rewind_journey.py"],
    "mouse": [
        "tests/core/test_textual_driver_pty.py::"
        "test_choice_mouse_selection_keeps_command_editor_focused",
        "tests/core/test_textual_driver_pty.py::"
        "test_edge_template_click_keeps_editor_focused_for_immediate_command_entry",
        "tests/core/test_textual_driver_pty.py::"
        "test_custom_driver_handles_sgr_mouse_resize_output_and_terminal_cleanup",
    ],
    "runtime": [
        "tests/core/test_textual_driver_pty.py::"
        "test_textual_bridge_drives_a_multi_turn_runtime_and_rejects_handoff_input",
        BEAM_TEST,
    ],
    "all": ["tests/core/test_textual_driver_pty.py"],
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIOS),
        default="beam",
        help="scripted PTY encounter to record (default: beam)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(tempfile.gettempdir()) / "serial-policy-editor-recordings",
        help="directory for raw PTY data, frame text files, and the pytest log",
    )
    return parser


def _new_run_directory(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    stamp = strftime("%Y%m%d-%H%M%S")
    for suffix in range(1000):
        candidate = output / (stamp if suffix == 0 else f"{stamp}-{suffix}")
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"could not create a unique run directory under {output}")


def _export_frames(run_directory: Path) -> list[dict[str, object]]:
    recordings: list[dict[str, object]] = []
    for capture_path in sorted(run_directory.rglob("*.capture.json")):
        try:
            capture = json.loads(capture_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            recordings.append({"capture": str(capture_path), "error": str(error)})
            continue

        frame_directory = capture_path.with_name(
            capture_path.name.removesuffix(".capture.json") + ".frames"
        )
        frame_directory.mkdir(exist_ok=True)
        index: list[dict[str, object]] = []
        for event in capture.get("ordered_events", ()):
            if event.get("kind") != "display" or not event.get("grid"):
                continue
            sequence = int(event.get("sequence", len(index) + 1))
            generation = event.get("generation", "?")
            columns, rows = event.get("size", (0, 0))
            filename = f"{sequence:04d}-g{generation}-{columns}x{rows}.txt"
            geometry = event.get("geometry") or {}
            header = (
                f"display={sequence} generation={generation} "
                f"screen={event.get('screen_id')} kind={geometry.get('kind')} "
                f"size={columns}x{rows} sync_active={event.get('sync_active')}"
            )
            (frame_directory / filename).write_text(
                header + "\n" + "\n".join(event["grid"]) + "\n",
                encoding="utf-8",
            )
            index.append({
                "sequence": sequence,
                "generation": generation,
                "size": [columns, rows],
                "screen_id": event.get("screen_id"),
                "screen_kind": geometry.get("kind"),
                "sync_active": event.get("sync_active"),
                "file": filename,
            })

        (frame_directory / "index.json").write_text(
            json.dumps(index, indent=2) + "\n",
            encoding="utf-8",
        )
        recordings.append({
            "capture": str(capture_path),
            "raw_pty": capture.get("raw_pty_path"),
            "frame_directory": str(frame_directory),
            "display_frames": len(index),
            "replay_error": capture.get("replay_error"),
        })
    return recordings


def _run(args: argparse.Namespace) -> int:
    repository = Path(__file__).resolve().parents[2]
    run_directory = _new_run_directory(args.output.expanduser().resolve())
    pytest_temp = run_directory / "pytest"
    log_path = run_directory / "pytest.log"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-q",
        "--tb=short",
        "--basetemp",
        str(pytest_temp),
        *SCENARIOS[args.scenario],
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    print(f"Recording {args.scenario!r} PTY journey from {repository}")
    print(f"Artifacts: {run_directory}")
    print("Command:", " ".join(command))
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=repository,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()

    recordings = _export_frames(run_directory)
    summary = {
        "scenario": args.scenario,
        "command": command,
        "pytest_exit_code": return_code,
        "pytest_log": str(log_path),
        "recordings": recordings,
    }
    summary_path = run_directory / "recording-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nRecording summary: {summary_path}")
    if recordings:
        for recording in recordings:
            if "error" in recording:
                print(f"Could not read {recording['capture']}: {recording['error']}")
                continue
            print(
                f"{recording['display_frames']} display frames: "
                f"{recording['frame_directory']}"
            )
            print(f"Raw PTY stream: {recording['raw_pty']}")
    else:
        print("No frame capture was produced; see pytest.log and the pytest folder.")
    return return_code


def main() -> int:
    args = _parser().parse_args()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())

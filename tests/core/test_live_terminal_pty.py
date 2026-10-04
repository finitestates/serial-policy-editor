"""Production terminal journeys on a real PTY, checked frame by frame.

The child process runs the real ``TerminalIO`` session. The UI logs every frame
it intends to show (``SPE_TERMINAL_FRAME_LOG``); the parent records the raw PTY
bytes, replays them through pyte, and requires the reconstructed screen to equal
the intended frame at every frame boundary. Every intended frame is then checked
for completeness, so a torn, blank, or mixed-state frame fails the test.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import select
import signal
import struct
import sys
import termios
import textwrap
import time
import unicodedata
from pathlib import Path

import pytest
from rich.cells import cell_len

pyte = pytest.importorskip("pyte")

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX PTY journeys")

ROOT = Path(__file__).resolve().parents[2]

CHILD = textwrap.dedent(
    r"""
    import json, os, signal, sys, time
    from dataclasses import replace
    sys.path[:0] = [os.environ["SPE_ROOT"] + "/core/src", os.environ["SPE_ROOT"]]
    from tests.core.term_support import beam_state, choice_state, edge_state, prompt_state
    from trajectory_editor.terminal_contracts import BeamViewRow, BeamViewState
    from trajectory_editor.tui import TerminalIO

    def beam(step, selected):
        rows = tuple(
            BeamViewRow(
                f"b{(step + index) % 9}",
                f"step {step} branch {index} " + "continuation " * (index + step % 3),
                f"-{0.3 * index + step / 10:.2f}", "LIVE", (f"step {step}",),
            )
            for index in range(6)
        )
        labels = [row.label for row in rows]
        return BeamViewState(f"BEAM · width 6 · depth {step}", "shared context " * 3, rows,
                             selected if selected in labels else labels[step % 6], notice=f"advance {step}")

    def gate(name):
        path = os.environ["SPE_GATE"] + "." + name
        while not os.path.exists(path):
            time.sleep(0.01)

    mode = sys.argv[1]
    if mode == "cli-interrupt":
        from unittest.mock import patch
        from tests.fakes import ConformingFakeBackend
        from trajectory_editor.episode_cli import main

        with patch(
            "trajectory_editor.episode_backend_loader.load_backend",
            side_effect=lambda _args: ConformingFakeBackend(),
        ):
            raise SystemExit(main(["--model", "fake", "--new-prompt", "P"]))

    io = TerminalIO()
    if mode == "fatal":
        with io.session():
            io.read_choice(choice_state())
            print("incidental before fatal")
            os.write(2, b"native stderr before fatal\n")
            raise ValueError("fatal sentinel")
    if mode == "explicit-output":
        from pathlib import Path
        from types import SimpleNamespace
        from trajectory_editor.session_runtime import _print_final_text

        with io.session():
            io.read_choice(choice_state())
            print("incidental before saved output")
            _print_final_text(
                SimpleNamespace(engine=SimpleNamespace(text="saved final text")),
                Path(os.environ["SPE_FINAL_OUTPUT"]),
                io,
            )
        print("outside output")
        raise SystemExit(0)

    results = []
    try:
        with io.session():
            if mode == "journey":
                results.append(io.read_choice(choice_state()))
                gate("choice")  # the engine is slow; keys typed now must be dropped
                io.write("captured line")
                results.append(io.read_edge(edge_state()))
                selected = None
                for step in range(31):
                    answer = io.read_beam(beam(step, selected))
                    results.append([answer.command, answer.selected_label])
                    selected = answer.selected_label
                    if answer.command != "advance 1":
                        break
                results.append(io.read(prompt_state().prompt))
                results.append(io.prompt(prompt_state(multiline=True)))
                results.append(io.read_key("Key? "))
                print("printed during session")
                os.write(2, b"native stderr during session\n")
            elif mode == "interrupt":
                io.read_choice(choice_state())
            elif mode == "unicode-geometry":
                results.append(io.prompt(prompt_state("Unicode geometry › ")))
            elif mode == "choice-pane":
                state = choice_state()
                context = "\n".join(
                    f"layout history item {index:03} " + "word " * 12
                    for index in range(80)
                )
                results.append(io.read_choice(replace(
                    state, choice=replace(state.choice, context_text_tail=context),
                )))
            elif mode == "suspend":
                # pty.fork() makes this child an orphaned job-control group,
                # where the kernel ignores SIGTSTP. Preserve app.suspend()'s
                # stop/resume flow by substituting an unignorable stop here.
                kill = os.kill
                def kill_for_test(pid, signum):
                    kill(pid, signal.SIGSTOP if signum == signal.SIGTSTP else signum)
                os.kill = kill_for_test
                io.read_choice(choice_state())
            elif mode == "diagnostic":
                from trajectory_editor.core.errors import EditorError
                state = choice_state()
                def fail_candidate(rank):
                    raise EditorError(f"engine detail for rank {rank}")
                results.append(io.read_choice(replace(state, resolve_candidate=fail_candidate)))
            elif mode == "runtime":
                from types import SimpleNamespace
                from tests.fakes import ConformingFakeBackend
                from trajectory_editor.core.sampler_config import SamplerConfig
                from trajectory_editor.episode_engine import EpisodeEngine
                from trajectory_editor.episode_session import LiveSession, LiveSessionRoster
                from trajectory_editor.session_runtime import run_session_roster
                engine = EpisodeEngine(ConformingFakeBackend(), initial_token_ids=[7],
                                       sampling=SamplerConfig(temperature=0.0))
                args = SimpleNamespace(
                    divergence_policy="handoff", table_depth=3, hold_default=20,
                    context_chars=0, manual_acceptance=False, show_policy_rank=None,
                    logit_view="none", output=None, phrase_max_tokens=16, phrase_max_shift=6.0,
                )
                results.append(run_session_roster(
                    args, io=io, roster=LiveSessionRoster(LiveSession(engine)),
                    backend_provenance={"backend": "fake"},
                ))
    except KeyboardInterrupt:
        results.append("interrupted")
    print("RESULTS " + json.dumps(results))
    """
)


def _set_size(fd: int, columns: int, rows: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))


class Session:
    def __init__(self, tmp_path: Path, mode: str, size=(100, 30), env=None) -> None:
        import pty

        self.frames_path = tmp_path / "frames.jsonl"
        self.gate = tmp_path / "gate"
        self.output = bytearray()
        self.size = size
        pid, fd = pty.fork()
        if pid == 0:  # pragma: no cover - child
            os.environ.update({
                "SPE_ROOT": str(ROOT), "SPE_GATE": str(self.gate),
                "SPE_TERMINAL_FRAME_LOG": str(self.frames_path),
                "TERM": "xterm-256color", "COLORTERM": "truecolor",
                **(env or {}),
            })
            os.execv(sys.executable, [sys.executable, "-c", CHILD, mode])
        self.pid, self.fd = pid, fd
        _set_size(fd, *size)
        self.status: int | None = None

    def pump(self, seconds: float = 0.05) -> None:
        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            ready, _, _ = select.select([self.fd], [], [], remaining)
            if not ready:
                return
            try:
                data = os.read(self.fd, 65536)
            except OSError:
                return
            if not data:
                return
            self.output.extend(data)

    def frames(self) -> list[dict]:
        if not self.frames_path.exists():
            return []
        return [json.loads(line) for line in self.frames_path.read_text().splitlines() if line]

    def wait_for(self, predicate, timeout: float = 10.0, what: str = "") -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.pump(0.02)
            frames = self.frames()
            if frames and predicate(frames[-1]):
                return frames[-1]
        last = self.frames()[-1]["lines"] if self.frames() else []
        raise AssertionError(f"timed out waiting for {what}; last frame:\n" + "\n".join(last))

    def wait_stopped(self, timeout: float = 10.0) -> int:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.pump(0.02)
            done, status = os.waitpid(self.pid, os.WNOHANG | os.WUNTRACED)
            if done:
                if os.WIFSTOPPED(status):
                    return status
                self.status = status
                raise AssertionError(f"child exited instead of stopping: {status}")
        excerpt = self.output[-2000:].decode("utf-8", "replace")
        raise AssertionError("child did not stop after Ctrl+Z:\n" + excerpt)

    def send(self, data: bytes | str, *, settle: float = 0.0) -> None:
        os.write(self.fd, data.encode() if isinstance(data, str) else data)
        if settle:
            self.pump(settle)

    def resize(self, columns: int, rows: int) -> None:
        self.size = (columns, rows)
        _set_size(self.fd, columns, rows)
        os.kill(self.pid, signal.SIGWINCH)

    def open_gate(self, name: str) -> None:
        Path(f"{self.gate}.{name}").touch()

    def finish(self, timeout: float = 10.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.pump(0.05)
            done, status = os.waitpid(self.pid, os.WNOHANG)
            if done:
                self.status = status
                self.pump(0.2)
                return self.output.decode("utf-8", "replace")
        os.kill(self.pid, signal.SIGKILL)
        os.waitpid(self.pid, 0)
        raise AssertionError("child did not exit:\n" + self.output[-2000:].decode("utf-8", "replace"))


def text(frame: dict) -> str:
    return "\n".join(frame["lines"])


def replay_and_compare(output: bytes, frames: list[dict]) -> None:
    """The terminal must show exactly each intended frame at its boundary."""
    assert frames, "no frames were presented"
    screen = pyte.Screen(*frames[0]["size"])
    stream = pyte.ByteStream(screen)
    position = 0
    for frame in frames:
        width, height = frame["size"]
        if (screen.columns, screen.lines) != (width, height):
            screen.resize(height, width)
        stream.feed(bytes(output[position:frame["end_offset"]]))
        position = frame["end_offset"]
        got = [line.rstrip() for line in screen.display]
        want = [line.rstrip() for line in frame["lines"]]
        assert got == want, f"terminal differs from intended frame {frame['sequence']}"
        if frame["cursor"] is not None:
            assert (screen.cursor.x, screen.cursor.y) == tuple(frame["cursor"])
            assert not screen.cursor.hidden
        else:
            assert screen.cursor.hidden


ERASES = re.compile(rb"\x1b\[[0-3]?[JK]")


def assert_no_erase_after_entry(output: bytes) -> None:
    entry = output.find(b"\x1b[?1049h")
    assert entry >= 0
    first_clear = output.find(b"\x1b[2J", entry)
    later = output[first_clear + 4:]
    exit_index = later.find(b"\x1b[?1049l")
    body = later if exit_index < 0 else later[:exit_index]
    assert not ERASES.search(body), "the UI erased the screen after startup"


def assert_frames_are_transactions(output: bytes) -> None:
    starts = [match.start() for match in re.finditer(re.escape(b"\x1b[?2026h"), output)]
    ends = [match.start() for match in re.finditer(re.escape(b"\x1b[?2026l"), output)]
    assert len(starts) == len(ends)
    assert all(start < end for start, end in zip(starts, ends))
    assert all(end < next_start for end, next_start in zip(ends, starts[1:]))


def assert_complete_frame(frame: dict) -> None:
    lines = frame["lines"]
    body = text(frame)
    width, height = frame["size"]
    assert "working…" not in body, f"internal busy status in frame {frame['sequence']}"
    assert "Resolving raw rank" not in body, f"internal warm status in frame {frame['sequence']}"
    assert len(lines) == height
    for row, line in enumerate(lines):
        cells = cell_len(line)
        assert cells <= width, (
            f"frame {frame['sequence']} row {row} occupies {cells} cells in a {width}-cell terminal"
        )
    if frame["cursor"] is not None:
        cursor_x, cursor_y = frame["cursor"]
        assert 0 <= cursor_x < width and 0 <= cursor_y < height, (
            f"frame {frame['sequence']} cursor {(cursor_x, cursor_y)} is outside {width}x{height}"
        )
    if any("╭─ " in line for line in lines):
        assert any("closes" in line for line in lines), f"overlay without its hint in frame {frame['sequence']}"
        return
    if lines[0].startswith("BEAM"):
        assert "Beam >" in body, f"beam command row missing in frame {frame['sequence']}"
        marked = [line for line in lines if line.startswith(">")]
        assert len(marked) == 1, f"beam selection marker count {len(marked)} in frame {frame['sequence']}"
        label = marked[0].split()[2]
        if height >= 20:
            assert f"SELECTED: {label}" in body, (
                f"frame {frame['sequence']}: table selects {label} but details show "
                + next((line.strip() for line in lines if "SELECTED:" in line), "nothing")
            )
        # Heading and notice come from the same request: never mixed generations.
        depth = re.search(r"depth (\d+)", lines[0])
        notice = next((re.fullmatch(r"advance (\d+)", line.strip()) for line in lines
                       if re.fullmatch(r"advance (\d+)", line.strip())), None)
        assert depth is not None and notice is not None, f"beam frame {frame['sequence']} incomplete"
        assert depth.group(1) == notice.group(1), f"mixed beam generations in frame {frame['sequence']}"
    elif lines[0].startswith("Step "):
        if height >= 8:
            assert "─" * width in lines, (
                f"context/candidate divider missing in frame {frame['sequence']}"
            )
        assert "Command >" in body and "Candidates" in body or height < 18
    elif lines[0].startswith("LIVE EDGE"):
        assert "Command >" in body


def test_frame_completeness_rejects_cell_overflow():
    """Negative control: a row may fit by codepoints and still exceed terminal cells."""
    frame = {
        "sequence": 1,
        "size": [3, 1],
        "lines": ["界界"],
        "cursor": None,
    }
    assert len(frame["lines"][0]) <= frame["size"][0]
    with pytest.raises(AssertionError, match="occupies 4 cells"):
        assert_complete_frame(frame)


def test_frame_completeness_rejects_a_missing_choice_divider():
    """Negative control: a Choice screen without its declared seam is incomplete."""
    frame = {
        "sequence": 1,
        "size": [40, 8],
        "lines": [
            "Step 0 · teacher track", "DECISION BOUNDARY", "context", "preview",
            "Candidates", "rank token-id text", "Command >", "Enter commit",
        ],
        "cursor": None,
    }

    with pytest.raises(AssertionError, match="context/candidate divider missing"):
        assert_complete_frame(frame)


def test_production_prompt_geometry_for_wide_and_combining_text(tmp_path):
    ui = Session(tmp_path, "unicode-geometry", size=(80, 16))
    entered = "A界e\u0301B"
    visible = unicodedata.normalize("NFC", entered)
    try:
        ui.wait_for(
            lambda frame: "Unicode geometry ›" in text(frame) and frame["cursor"],
            what="Unicode geometry prompt",
        )
        ui.send(entered)
        frame = ui.wait_for(
            lambda candidate: visible in text(candidate) and candidate["cursor"],
            what="wide and combining input",
        )
        row = next(index for index, line in enumerate(frame["lines"]) if visible in line)
        line = frame["lines"][row]
        expected_column = cell_len(line[:line.index(visible)]) + cell_len(visible)
        assert tuple(frame["cursor"]) == (expected_column, row), (
            f"cursor {frame['cursor']} does not follow {visible!r} at cell column {expected_column}"
        )
        ui.send("\r")
        output = ui.finish()
    finally:
        if ui.status is None:
            try:
                os.kill(ui.pid, signal.SIGKILL)
                os.waitpid(ui.pid, 0)
                ui.pump(0.2)
            except OSError:
                pass
        (tmp_path / "pty.raw").write_bytes(bytes(ui.output))

    assert os.WIFEXITED(ui.status) and os.WEXITSTATUS(ui.status) == 0, output[-2000:]
    results = json.loads(output.rsplit("RESULTS ", 1)[1].splitlines()[0])
    assert results == [entered]

    raw = bytes(ui.output)
    frames = ui.frames()
    replay_and_compare(raw, frames)
    assert_no_erase_after_entry(raw)
    assert_frames_are_transactions(raw)
    for captured in frames:
        assert_complete_frame(captured)

    # Negative control: pyte must reject a cursor column that disagrees with
    # the raw PTY stream at this Unicode editor frame boundary.
    altered = [dict(captured) for captured in frames]
    target = next(index for index, captured in enumerate(altered) if visible in text(captured))
    wrong_cursor = list(altered[target]["cursor"])
    wrong_cursor[0] = (
        wrong_cursor[0] + 1
        if wrong_cursor[0] + 1 < altered[target]["size"][0]
        else wrong_cursor[0] - 1
    )
    altered[target]["cursor"] = wrong_cursor
    with pytest.raises(AssertionError):
        replay_and_compare(raw, altered)


def test_journey_every_frame_is_complete_and_matches_the_terminal(tmp_path):
    ui = Session(tmp_path, "journey", size=(100, 30))
    try:
        ui.wait_for(lambda f: f["lines"][0].startswith("Step 0") and f["cursor"], what="choice")
        ui.send("\t")
        ui.wait_for(lambda f: "Command > 1" in text(f), what="tab staged rank 1")
        ui.send("\r")
        submitted = ui.wait_for(
            lambda f: f["lines"][0].startswith("Step 0")
            and f["cursor"] is None and "Command > 1" in text(f),
            what="submitted choice without a status marker",
        )
        # The engine is still working: typed keys are dropped, the view stays whole.
        ui.send("zzz\r", settle=0.3)
        busy = ui.frames()[-1]
        assert "zzz" not in text(busy) and busy["lines"][0].startswith("Step 0")
        assert busy["cursor"] is None
        ui.open_gate("choice")

        ui.wait_for(lambda f: f["lines"][0].startswith("LIVE EDGE") and f["cursor"], what="edge")
        ui.send("\x0c")  # Ctrl+L: captured output
        ui.wait_for(lambda f: "captured line" in text(f), what="output overlay")
        ui.send("q")
        ui.wait_for(lambda f: "Captured output" not in text(f), what="overlay closed")
        ui.send("#3\r")

        ui.wait_for(lambda f: "depth 0" in f["lines"][0] and f["cursor"], what="beam")
        for step in range(30):
            if step == 4:
                ui.resize(140, 40)
            elif step == 10:
                ui.resize(80, 24)
            elif step == 16:
                ui.resize(160, 50)
            elif step == 22:
                ui.resize(100, 30)
            if step % 5 == 2:
                ui.send("\x1b[B")  # move the selection before advancing
            ui.send("\x1b[C")
            ui.wait_for(lambda f, s=step: f"depth {s + 1}" in f["lines"][0] and f["cursor"],
                        what=f"beam step {step + 1}")
        # A mouse click on another branch, then Enter commits it.
        frame = ui.frames()[-1]
        row = next(index for index, line in enumerate(frame["lines"]) if line.startswith("  ") and " b" in line[:12])
        ui.send(f"\x1b[<0;8;{row + 1}M\x1b[<0;8;{row + 1}m")
        ui.wait_for(lambda f, r=row: f["lines"][r].startswith(">"), what="clicked branch")
        ui.send("\r")

        ui.wait_for(lambda f: "Response" in text(f) or "›" in text(f), what="prompt")
        ui.send("\x1b[200~pasted\nline\x1b[201~\r")
        ui.wait_for(lambda f: "Write the new prompt" in text(f), what="multiline")
        ui.send("first\rsecond")
        ui.send("\x1b")
        time.sleep(0.1)
        ui.send("\r")
        ui.wait_for(lambda f: "Press a key" in text(f), what="single key")
        ui.send("k")
        output = ui.finish()
    finally:
        if ui.status is None:
            try:
                os.kill(ui.pid, signal.SIGKILL)
                os.waitpid(ui.pid, 0)
                ui.pump(0.2)
            except OSError:
                pass
        (tmp_path / "pty.raw").write_bytes(bytes(ui.output))

    assert os.WIFEXITED(ui.status) and os.WEXITSTATUS(ui.status) == 0, output[-2000:]
    results = json.loads(output.rsplit("RESULTS ", 1)[1].splitlines()[0])
    assert results[0] == "1"
    assert results[1] == "#3"
    beam_results = results[2:-3]
    assert len(beam_results) == 31
    assert all(command == "advance 1" for command, _label in beam_results[:30])
    assert beam_results[-1][0].startswith("select ")
    assert results[-3:] == ["pasted line", "first\nsecond", "k"]

    raw = bytes(ui.output)
    frames = ui.frames()
    replay_and_compare(raw, frames)
    assert_no_erase_after_entry(raw)
    assert_frames_are_transactions(raw)
    for frame in frames:
        assert_complete_frame(frame)
    tainted = dict(submitted)
    tainted["lines"] = list(submitted["lines"])
    command_row = next(index for index, line in enumerate(tainted["lines"]) if "Command >" in line)
    tainted["lines"][command_row] = tainted["lines"][command_row].replace(
        "Command >", "working… Command >", 1,
    )
    with pytest.raises(AssertionError):
        assert_complete_frame(tainted)
    sizes = {tuple(frame["size"]) for frame in frames}
    assert {(100, 30), (140, 40), (80, 24), (160, 50)} <= sizes

    # Incidental process output is silenced; explicit output after the session survives.
    exit_index = raw.rfind(b"\x1b[?1049l")
    assert exit_index > 0
    assert raw.rfind(b"\x1b[?25h") > raw.rfind(b"\x1b[?25l")
    for enabled, restored in ((b"\x1b[?2004h", b"\x1b[?2004l"), (b"\x1b[?1006h", b"\x1b[?1006l"),
                              (b"\x1b[?1000h", b"\x1b[?1000l"), (b"\x1b[?7l", b"\x1b[?7h")):
        assert raw.rfind(restored) > raw.rfind(enabled)
    def assert_process_output_was_silenced(output):
        assert b"printed during session" not in output
        assert b"native stderr during session" not in output

    assert_process_output_was_silenced(raw)
    with pytest.raises(AssertionError):
        assert_process_output_was_silenced(raw + b"printed during session")
    assert raw.find(b"RESULTS") > exit_index


def test_choice_pane_content_updates_and_resize_keep_ordered_frames_complete(tmp_path):
    ui = Session(tmp_path, "choice-pane", size=(80, 24))
    try:
        initial = ui.wait_for(
            lambda frame: frame["lines"][0].startswith("Step 0") and frame["cursor"],
            what="long-context Choice",
        )
        assert "layout history item 079" in text(initial)
        assert next(row for row, line in enumerate(initial["lines"]) if line == "─" * 80) == 7

        ui.send("\x1b[5~")  # page the context without moving the pane boundary
        scrolled = ui.wait_for(
            lambda frame: frame["sequence"] > initial["sequence"],
            what="scrolled context",
        )
        assert scrolled["lines"][1:9] != initial["lines"][1:9]
        assert next(row for row, line in enumerate(scrolled["lines"]) if line == "─" * 80) == 7

        ui.send("2")
        preview = ui.wait_for(
            lambda frame: "Command > 2" in text(frame), what="updated candidate preview",
        )
        assert next(row for row, line in enumerate(preview["lines"]) if line == "─" * 80) == 7

        ui.send("\x7f9\r")  # clear the command, then create editable validation feedback
        feedback = ui.wait_for(
            lambda frame: "rank must be 1..5" in text(frame).lower(),
            what="updated Choice feedback",
        )
        assert next(row for row, line in enumerate(feedback["lines"]) if line == "─" * 80) == 7

        ui.resize(120, 40)
        resized = ui.wait_for(
            lambda frame: tuple(frame["size"]) == (120, 40)
            and "rank must be 1..5" in text(frame).lower(),
            what="resized Choice with feedback",
        )
        assert next(row for row, line in enumerate(resized["lines"]) if line == "─" * 120) == 9

        ui.send("\x7f1\r")
        submitted = ui.wait_for(
            lambda frame: frame["cursor"] is None and "Command > 1" in text(frame),
            what="submitted Choice after resize",
        )
        assert next(row for row, line in enumerate(submitted["lines"]) if line == "─" * 120) == 9
        output = ui.finish()
    finally:
        if ui.status is None:
            try:
                os.kill(ui.pid, signal.SIGKILL)
                os.waitpid(ui.pid, 0)
                ui.pump(0.2)
            except OSError:
                pass
        (tmp_path / "pty.raw").write_bytes(bytes(ui.output))

    assert os.WIFEXITED(ui.status) and os.WEXITSTATUS(ui.status) == 0, output[-2000:]
    assert json.loads(output.rsplit("RESULTS ", 1)[1].splitlines()[0]) == ["1"]
    raw = bytes(ui.output)
    frames = ui.frames()
    replay_and_compare(raw, frames)
    assert_no_erase_after_entry(raw)
    assert_frames_are_transactions(raw)
    choice_frames = [frame for frame in frames if frame["lines"][0].startswith("Step 0")]
    assert len(choice_frames) >= 6
    for frame in choice_frames:
        assert_complete_frame(frame)
        divider = "─" * frame["size"][0]
        expected_divider_row = 7 if frame["size"][1] == 24 else 9
        assert next(row for row, line in enumerate(frame["lines"]) if line == divider) == expected_divider_row


def test_uncaught_fatal_traceback_survives_after_terminal_restore(tmp_path):
    ui = Session(tmp_path, "fatal", size=(80, 24))
    ui.wait_for(lambda f: f["lines"][0].startswith("Step 0") and f["cursor"], what="choice")
    ui.send("1\r")
    output = ui.finish()
    assert os.WIFEXITED(ui.status) and os.WEXITSTATUS(ui.status) == 1, output[-2000:]
    raw = bytes(ui.output)
    exit_index = raw.rfind(b"\x1b[?1049l")
    traceback_index = raw.find(b"Traceback")
    assert exit_index > 0 and traceback_index > exit_index
    # Python 3.13+ may colorize the exception type and message on a PTY.
    traceback_text = re.sub(rb"\x1b\[[0-9;]*m", b"", raw[traceback_index:])
    assert b"ValueError: fatal sentinel" in traceback_text
    assert b"incidental before fatal" not in raw
    assert b"native stderr before fatal" not in raw
    frames = ui.frames()
    replay_and_compare(raw, frames)
    assert_no_erase_after_entry(raw)
    assert_frames_are_transactions(raw)
    for frame in frames:
        assert_complete_frame(frame)


def test_explicit_saved_output_is_emitted_after_terminal_restore(tmp_path):
    saved_text = tmp_path / "final.txt"
    ui = Session(
        tmp_path, "explicit-output", size=(80, 24),
        env={"SPE_FINAL_OUTPUT": str(saved_text)},
    )
    ui.wait_for(lambda f: f["lines"][0].startswith("Step 0") and f["cursor"], what="choice")
    ui.send("1\r")
    output = ui.finish()
    assert os.WIFEXITED(ui.status) and os.WEXITSTATUS(ui.status) == 0, output[-2000:]
    assert saved_text.read_text() == "saved final text"
    raw = bytes(ui.output)
    exit_index = raw.rfind(b"\x1b[?1049l")
    assert exit_index > 0
    assert raw.find(f"Text: {saved_text}".encode()) > exit_index
    assert raw.find(b"outside output") > exit_index
    assert b"incidental before saved output" not in raw
    frames = ui.frames()
    replay_and_compare(raw, frames)
    assert_no_erase_after_entry(raw)
    assert_frames_are_transactions(raw)
    for frame in frames:
        assert_complete_frame(frame)


def test_ctrl_c_interrupts_and_restores_the_terminal(tmp_path):
    ui = Session(tmp_path, "interrupt", size=(80, 24))
    ui.wait_for(lambda f: f["lines"][0].startswith("Step 0") and f["cursor"], what="choice")
    ui.send("\x03")
    output = ui.finish()
    assert os.WIFEXITED(ui.status) and os.WEXITSTATUS(ui.status) == 0
    assert 'RESULTS ["interrupted"]' in output
    raw = bytes(ui.output)
    exit_index = raw.rfind(b"\x1b[?1049l")
    assert exit_index > raw.rfind(b"\x1b[?1049h")
    assert raw.find(b"RESULTS") > exit_index
    replay_and_compare(raw, ui.frames())


def test_cli_ctrl_c_exits_cleanly_with_brief_message(tmp_path):
    ui = Session(tmp_path, "cli-interrupt", size=(80, 24))
    ui.wait_for(lambda f: f["lines"][0].startswith("Step 0") and f["cursor"], what="choice")
    ui.send("\x03")
    output = ui.finish()
    assert os.WIFEXITED(ui.status) and os.WEXITSTATUS(ui.status) == 130, output[-2000:]
    assert "Interrupted." in output
    assert "Traceback" not in output
    raw = bytes(ui.output)
    exit_index = raw.rfind(b"\x1b[?1049l")
    assert exit_index > 0
    assert raw.find(b"Interrupted.") > exit_index
    frames = ui.frames()
    replay_and_compare(raw, frames)
    assert_no_erase_after_entry(raw)
    assert_frames_are_transactions(raw)
    for frame in frames:
        assert_complete_frame(frame)


def test_preview_diagnostic_is_off_the_live_canvas_until_ctrl_l(tmp_path):
    ui = Session(tmp_path, "diagnostic", size=(80, 24))
    try:
        ui.wait_for(lambda f: f["lines"][0].startswith("Step 0") and f["cursor"], what="choice")
        ui.send("5")
        preview = ui.wait_for(
            lambda f: "preview unavailable" in text(f).lower(), what="generic preview notice",
        )
        assert "engine detail for rank 5" not in text(preview)
        ui.send("\x0c")
        diagnostic = ui.wait_for(
            lambda f: "[diagnostic] candidate preview failed" in text(f),
            what="captured preview diagnostic",
        )
        assert "engine detail for rank 5" in text(diagnostic)
        ui.send("q")
        ui.wait_for(
            lambda f: f["lines"][0].startswith("Step 0") and "Captured output" not in text(f),
            what="choice restored after diagnostics",
        )
        ui.send("\x03")
        output = ui.finish()
    finally:
        if ui.status is None:
            try:
                os.kill(ui.pid, signal.SIGKILL)
                os.waitpid(ui.pid, 0)
            except OSError:
                pass
        ui.pump(0.2)
        (tmp_path / "pty.raw").write_bytes(bytes(ui.output))
    assert os.WIFEXITED(ui.status) and os.WEXITSTATUS(ui.status) == 0, output[-2000:]
    assert 'RESULTS ["interrupted"]' in output
    raw = bytes(ui.output)
    frames = ui.frames()
    replay_and_compare(raw, frames)
    assert_no_erase_after_entry(raw)
    assert_frames_are_transactions(raw)
    for frame in frames:
        assert_complete_frame(frame)


def test_replay_oracle_rejects_a_torn_frame(tmp_path):
    """Negative control: corrupting one written line must fail the comparison."""
    ui = Session(tmp_path, "interrupt", size=(80, 24))
    ui.wait_for(lambda f: f["lines"][0].startswith("Step 0") and f["cursor"], what="choice")
    ui.send("2")
    ui.wait_for(lambda f: "Command > 2" in text(f), what="typed")
    ui.send("\x03")
    ui.finish()
    raw = bytes(ui.output)
    frames = ui.frames()
    replay_and_compare(raw, frames)
    torn = raw.replace(b"Step 0 \xc2\xb7 teacher track", b"Step 0 \xc2\xb7 teacher tr\x1b[K", 1)
    with pytest.raises(AssertionError):
        replay_and_compare(torn, frames)
    with pytest.raises(AssertionError):
        assert_no_erase_after_entry(torn)
    mixed = dict(frames[-1])
    mixed["lines"] = ["BEAM · depth 1", ">◆   1 b1 LIVE", "     2 b2 LIVE", "SELECTED: b2", "advance 1", "Beam >"] + [""] * 18
    with pytest.raises(AssertionError):
        assert_complete_frame(mixed)


def test_real_runtime_journey_through_choice_beam_and_edge(tmp_path):
    ui = Session(tmp_path, "runtime", size=(100, 30))
    try:
        for step in range(2):
            ui.wait_for(lambda f, s=step: f["lines"][0].startswith(f"Step {s} ") and f["cursor"], what=f"step {step}")
            ui.send("\r")
        ui.wait_for(lambda f: f["lines"][0].startswith("Step 2 ") and f["cursor"], what="step 2")
        # Optional diagnostic overlays can be enabled directly, then cleared.
        for command, label in (("l", "model-logit"), ("L", "model-gap"),
                               ("~", "noise"), ("%", "model-softmax")):
            ui.send(command + "\r")
            ui.wait_for(lambda f, label=label: label in text(f) and f["cursor"],
                        what=f"overlay {command}")
        ui.send("C\r")
        ui.wait_for(lambda f: "model-softmax" not in text(f)
                    and "model-logit" not in text(f) and f["cursor"], what="clear overlays")
        ui.send("beam\r")
        before = ui.wait_for(lambda f: "depth 1" in f["lines"][0] and f["cursor"], what="beam")
        def labels(frame):
            return set(re.findall(r"^\s*[>◆ ]*\d+\s+(b\d+)\s", text(frame), re.M))
        marked = next(line for line in before["lines"] if line.startswith(">"))
        if "LIVE" not in marked:
            first_live_rank = next(int(match.group(1)) for line in before["lines"]
                                   if (match := re.search(r"^\s*[>◆ ]*(\d+)\s+b\d+\s+LIVE", line)))
            ui.send("\x1b[B" * (first_live_rank - 1))
            before = ui.wait_for(lambda f: any(line.startswith(">") and "LIVE" in line
                                              for line in f["lines"]), what="select live branch")
            marked = next(line for line in before["lines"] if line.startswith(">"))
        killed = marked.split()[2]
        previous_labels = labels(before)
        ui.send("\x7f")
        after = ui.wait_for(lambda f: "depth 1" in f["lines"][0]
                           and killed not in labels(f) and f["cursor"], what="same-depth backfill")
        assert len(labels(after)) == len(previous_labels)
        assert previous_labels - {killed} <= labels(after)
        ui.resize(130, 40)
        ui.send("\x1b[C")
        ui.wait_for(lambda f: "depth 2" in f["lines"][0] and f["cursor"], what="beam advance")
        ui.resize(80, 24)
        ui.send("\x1b[B\x1b[C")
        ui.wait_for(lambda f: "depth 3" in f["lines"][0] and f["cursor"], what="beam advance")
        ui.send("\x1b")
        time.sleep(0.1)
        ui.wait_for(lambda f: f["lines"][0].startswith("Step 2 ") and f["cursor"], what="back to choice")
        ui.send("q\r")
        ui.wait_for(lambda f: f["lines"][0].startswith("LIVE SESSION") and f["cursor"], what="edge")
        ui.send("q\r")
        output = ui.finish()
    finally:
        if ui.status is None:
            try:
                os.kill(ui.pid, signal.SIGKILL)
                os.waitpid(ui.pid, 0)
            except OSError:
                pass
        ui.pump(0.2)
        (tmp_path / "pty.raw").write_bytes(bytes(ui.output))
    assert os.WIFEXITED(ui.status) and os.WEXITSTATUS(ui.status) == 0, output[-2000:]
    assert "RESULTS [" in output
    raw = bytes(ui.output)
    frames = ui.frames()
    replay_and_compare(raw, frames)
    assert_no_erase_after_entry(raw)
    assert_frames_are_transactions(raw)
    for frame in frames:
        lines = frame["lines"]
        if lines[0].startswith("BEAM"):
            assert "Beam >" in "\n".join(lines)
            assert sum(line.startswith(">") for line in lines) == 1
            if frame["size"][1] >= 20:
                marked = next(line for line in lines if line.startswith(">"))
                assert f"SELECTED: {marked.split()[2]}" in "\n".join(lines)
        elif lines[0].startswith("Step "):
            assert "Command >" in "\n".join(lines)


def test_resize_signals_coalesce_and_preserve_editor_state(tmp_path):
    ui = Session(tmp_path, "interrupt", size=(80, 24))
    try:
        ui.wait_for(lambda f: f["lines"][0].startswith("Step 0") and f["cursor"], what="choice")
        ui.send("2")
        ui.wait_for(lambda f: "Command > 2" in text(f), what="typed command")

        # A resize notification with unchanged geometry should not repaint.
        ui.pump(0.1)
        frame_count = len(ui.frames())
        output_count = len(ui.output)
        os.kill(ui.pid, signal.SIGWINCH)
        ui.pump(0.2)
        assert len(ui.frames()) == frame_count
        assert len(ui.output) == output_count

        # The small-terminal screen is temporary; request/editor state survives it.
        ui.resize(30, 2)
        small = ui.wait_for(
            lambda f: tuple(f["size"]) == (30, 2) and "Enlarge the terminal" in text(f),
            what="minimum-size screen",
        )
        assert "Command > 2" not in text(small)

        # Several geometry changes arrive without waiting for intermediate
        # frames. The UI must settle on the TTY's latest size and retain input.
        for size in ((110, 32), (70, 20), (120, 40)):
            ui.resize(*size)
        final = ui.wait_for(
            lambda f: tuple(f["size"]) == (120, 40) and "Command > 2" in text(f),
            what="latest resize with editor text preserved",
        )
        assert tuple(final["size"]) == (120, 40)
        assert "Command > 2" in text(final)
        ui.send("\x03")
        output = ui.finish()
    finally:
        if ui.status is None:
            try:
                os.kill(ui.pid, signal.SIGKILL)
                os.waitpid(ui.pid, 0)
            except OSError:
                pass
        ui.pump(0.2)
        (tmp_path / "pty.raw").write_bytes(bytes(ui.output))

    assert os.WIFEXITED(ui.status) and os.WEXITSTATUS(ui.status) == 0, output[-2000:]
    raw = bytes(ui.output)
    frames = ui.frames()
    replay_and_compare(raw, frames)
    assert_no_erase_after_entry(raw)
    assert_frames_are_transactions(raw)
    for frame in frames:
        assert_complete_frame(frame)


def test_ctrl_z_sigcont_reenters_terminal_once(tmp_path):
    ui = Session(tmp_path, "suspend", size=(80, 24))
    try:
        ui.wait_for(lambda f: f["lines"][0].startswith("Step 0") and f["cursor"], what="choice")
        initial_frames = len(ui.frames())
        ui.send("\x1a")
        stopped = ui.wait_stopped()
        assert os.WSTOPSIG(stopped) == signal.SIGSTOP
        before_resume = len(ui.frames())

        os.kill(ui.pid, signal.SIGCONT)
        resumed = ui.wait_for(
            lambda f: f["sequence"] > before_resume
            and f["lines"][0].startswith("Step 0") and f["cursor"],
            what="choice after SIGCONT",
        )
        ui.pump(0.15)
        assert len(ui.frames()) == before_resume + 1
        assert resumed["sequence"] == initial_frames + 1
        ui.send("\x03")
        output = ui.finish()
    finally:
        if ui.status is None:
            try:
                os.kill(ui.pid, signal.SIGKILL)
                os.waitpid(ui.pid, 0)
            except OSError:
                pass
        ui.pump(0.2)
        (tmp_path / "pty.raw").write_bytes(bytes(ui.output))

    assert os.WIFEXITED(ui.status) and os.WEXITSTATUS(ui.status) == 0, output[-2000:]
    assert 'RESULTS ["interrupted"]' in output
    raw = bytes(ui.output)
    frames = ui.frames()
    replay_and_compare(raw, frames)
    assert_frames_are_transactions(raw)
    assert raw.count(b"\x1b[?1049h") == 2
    assert raw.count(b"\x1b[?1049l") == 2
    assert raw.count(b"\x1b[2J") == 2
    assert ERASES.findall(raw) == [b"\x1b[2J", b"\x1b[2J"]
    for frame in frames:
        assert_complete_frame(frame)

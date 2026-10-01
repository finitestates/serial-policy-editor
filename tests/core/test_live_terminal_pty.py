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
from pathlib import Path

import pytest

pyte = pytest.importorskip("pyte")

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX PTY journeys")

ROOT = Path(__file__).resolve().parents[2]

CHILD = textwrap.dedent(
    r"""
    import json, os, sys, time
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
                model_rank=index + 1, step_log_probability=-0.1 * index,
                model_log_probability=-0.5 - index / 10,
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
    io = TerminalIO()
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
            elif mode == "interrupt":
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
                    divergence_policy="handoff", table_depth=3, search_radius=3, hold_default=20,
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
    assert len(lines) == height and all(len(line) <= width for line in lines)
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
        assert "Command >" in body and "Candidates" in body or height < 18
    elif lines[0].startswith("LIVE EDGE"):
        assert "Command >" in body


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
            except OSError:
                pass

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

    # The terminal is restored and process output appears only afterwards.
    exit_index = raw.rfind(b"\x1b[?1049l")
    assert exit_index > 0
    assert raw.rfind(b"\x1b[?25h") > raw.rfind(b"\x1b[?25l")
    for enabled, restored in ((b"\x1b[?2004h", b"\x1b[?2004l"), (b"\x1b[?1006h", b"\x1b[?1006l"),
                              (b"\x1b[?1000h", b"\x1b[?1000l"), (b"\x1b[?7l", b"\x1b[?7h")):
        assert raw.rfind(restored) > raw.rfind(enabled)
    assert raw.find(b"printed during session") > exit_index


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
        ui.send("beam\r")
        ui.wait_for(lambda f: "depth 1" in f["lines"][0] and f["cursor"], what="beam")
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
            os.kill(ui.pid, signal.SIGKILL)
            os.waitpid(ui.pid, 0)
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

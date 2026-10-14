"""Actual SGD and panel inspection over the production terminal bridge."""

import os
from pathlib import Path
import signal
import textwrap

from tests.core import test_live_terminal_pty as pty
from tests.core.test_online_learning import checkpoint  # shared real tiny model fixture


ONLINE_CHILD = textwrap.dedent(r'''
            elif mode == "online-panels":
                from types import SimpleNamespace
                from trajectory_editor.transformers_backend import TransformersBackend, TransformersSettings
                from trajectory_editor.online_learning import LearningConfig
                from trajectory_editor.core.sampler_config import SamplerConfig
                from trajectory_editor.episode_engine import EpisodeEngine
                from trajectory_editor.episode_session import LiveSession, LiveSessionRoster
                from trajectory_editor.session_runtime import run_session_roster
                backend = TransformersBackend(os.environ["SPE_ONLINE_MODEL"],
                    TransformersSettings(device="cpu", dtype="float32", torch_num_threads=1))
                backend.enable_online_learning(LearningConfig(lr=0.05, steps=3, max_step=3))
                engine = EpisodeEngine(backend, initial_token_ids=[2], initial_text="P",
                    sampling=SamplerConfig(temperature=0.0, top_k=8, min_p=0.0))
                rank = engine.observe().policy_calculations.raw_rank(3)
                from pathlib import Path
                Path(os.environ["SPE_GATE"] + ".rank").write_text(str(rank))
                diagnostic = engine.learner._diagnostic
                def paced(step, logits):
                    diagnostic(step, logits)
                    gate("sgd-" + str(step.step))
                engine.learner._diagnostic = paced
                args = SimpleNamespace(
                    divergence_policy="handoff", table_depth=8, hold_default=20,
                    context_chars=0, manual_acceptance=False, show_policy_rank=None,
                    logit_view="none", output=None, phrase_max_tokens=16, phrase_max_shift=6.0,
                )
                results.append(run_session_roster(
                    args, io=io, roster=LiveSessionRoster(LiveSession(engine)),
                    backend_provenance={"backend": "transformers"},
                ))
                results.append(engine.learner.events)
''')


def test_real_sgd_dock_overview_resize_and_reset_over_pty(tmp_path, checkpoint, monkeypatch):
    branch = '        elif mode == "runtime":'
    assert branch in pty.CHILD
    child = pty.CHILD.replace(branch, textwrap.indent(ONLINE_CHILD.strip(), "        ") + "\n" + branch)
    # Preserve the injected branch's relative indentation.
    compile(child, "online-panels-child", "exec")
    monkeypatch.setattr(pty, "CHILD", child)
    ui = pty.Session(tmp_path, "online-panels", size=(80, 24),
                     env={"SPE_ONLINE_MODEL": str(checkpoint)})
    try:
        initial = ui.wait_for(lambda f: "No SGD trace yet" in pty.text(f) and f["cursor"],
                              timeout=20, what="reserved learning dock")
        rank = int(Path(str(ui.gate) + ".rank").read_text())
        ui.send(f"{rank} ? 5.0\r")
        start = ui.wait_for(lambda f: "0/3 · training" in pty.text(f), what="initial target / rival")
        assert start["lines"][:3] == initial["lines"][:3]
        assert "Target 'red'" in pty.text(start)
        ui.send("\x1b[13~")  # F3 while the engine is training
        ui.wait_for(lambda f: "Learning overview" in pty.text(f) and "busy" in pty.text(f),
                    what="inspection during SGD")
        ui.send("\x1b[13~")
        ui.wait_for(lambda f: "Learning overview" not in pty.text(f), what="close overview")
        for step in (1, 2):
            ui.open_gate(f"sgd-{step - 1}")
            frame = ui.wait_for(lambda f, s=step: f"{s}/3 · training" in pty.text(f), what=f"step {step}")
            assert frame["lines"][:3] == initial["lines"][:3]
            assert "Rival" in pty.text(frame) and "Margin" in pty.text(frame)
        ui.resize(120, 40)
        ui.wait_for(lambda f: f["size"] == [120, 40] and "2/3 · training" in pty.text(f), what="resize")
        ui.send("\x1b[13~")
        ui.wait_for(lambda f: "2 SGD steps" in pty.text(f), what="live step count")
        ui.open_gate("sgd-2")
        ui.wait_for(lambda f: "3 SGD steps" in pty.text(f), what="live overlay update")
        ui.open_gate("sgd-3")
        ui.wait_for(lambda f: f["lines"][0].startswith("Step 1 ") and "ceiling" in pty.text(f), what="next choice after ceiling")
        ui.send("R")
        ui.wait_for(lambda f: "Comparisons refreshed." in pty.text(f), what="owner-thread refresh")
        ui.send("\x1b[F")  # scroll to the latest correction
        ui.wait_for(lambda f: "correction · boundary 0" in pty.text(f), what="correction readout")
        ui.send("\x1b[13~")
        ui.wait_for(lambda f: "Learning overview" not in pty.text(f) and f["cursor"], what="return to choice")
        ui.resize(80, 24)
        ui.send("learn reset\r")
        ui.wait_for(lambda f: "No SGD trace yet" in pty.text(f) and "Original episode weights restored" in pty.text(f),
                    what="reset clears trace without removing dock")
        ui.send("q\r")
        ui.wait_for(lambda f: f["lines"][0].startswith("LIVE SESSION") and f["cursor"], what="edge")
        ui.send("q\r")
        output = ui.finish()
    finally:
        if ui.status is None:
            os.kill(ui.pid, signal.SIGKILL)
            os.waitpid(ui.pid, 0)
        ui.pump(0.2)
        (tmp_path / "pty.raw").write_bytes(bytes(ui.output))
    assert os.WIFEXITED(ui.status) and os.WEXITSTATUS(ui.status) == 0, output[-2000:]
    raw = bytes(ui.output)
    frames = ui.frames()
    pty.replay_and_compare(raw, frames)
    pty.assert_no_erase_after_entry(raw)
    pty.assert_frames_are_transactions(raw)
    assert all(any(line.strip() for line in f["lines"]) for f in frames)

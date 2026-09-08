from dataclasses import replace
from unittest.mock import patch

import pytest

from tests.test_replay_eog import engine, create, NeverChoose
from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.episode_actions import Accept, EndGeneration, Hold, SelectRawRank
from trajectory_editor.episode_engine import ReplayExpectation
from trajectory_editor.episode_policy import EpisodeRunner, ReplayPlan, TapeStep
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_cli import main


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_final_transition_and_empty_plan(tmp_path, fixed, empty):
    e = engine()
    original = e.sampling
    final = replace(original, temperature=0.7)
    step_config = replace(original, top_k=3)
    steps = () if empty else (TapeStep(Accept(), ReplayExpectation((1,), None, "completed"), step_config),)
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        identifier = create(store, e)
        result = EpisodeRunner(e, store, identifier).run(
            tape=ReplayPlan(steps, not fixed, final), live_policy=NeverChoose(),
            stop_after_tape=False,
        )
        assert result.replay_exhausted
        assert e.boundary == (0 if empty else 1)
        assert e.sampling == (original if fixed else final)
        assert store.get_episode(identifier)["status"] == "replay-edge"
        assert store.final_sampling(identifier) == e.sampling
        # Continuing without a plan cannot reapply its settings.
        teacher = replace(e.sampling, temperature=0.2)
        e.sampling = teacher
        EpisodeRunner(e, store, identifier).run()
        assert e.sampling == teacher


@pytest.mark.parametrize("kind", ["mismatch", "eog", "budget", "invalid", "checkpoint"])
def test_early_exit_never_applies_final_settings(tmp_path, kind):
    e = engine()
    initial = e.sampling
    final = replace(initial, temperature=0.7)
    action = {"mismatch": Accept(), "eog": EndGeneration(), "budget": Hold(3),
              "invalid": SelectRawRank(999), "checkpoint": Hold(1)}[kind]
    if kind in {"budget", "checkpoint"}:
        e.resume(max_tokens=1)
    expected = ReplayExpectation((2,) if kind == "mismatch" else (1,), None, "completed")
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        identifier = create(store, e)
        result = EpisodeRunner(e, store, identifier).run(
            tape=ReplayPlan((TapeStep(action, expected),), True, final),
            live_policy=NeverChoose(),
        )
        assert not result.replay_exhausted
        assert e.sampling == initial
        assert store.final_sampling(identifier) == initial


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_cli_plan_yields_then_teacher_owns_settings(tmp_path, fixed, empty):
    path = tmp_path / "episodes.sqlite3"
    with EpisodeStore(path) as store:
        source = engine()
        identifier = create(store, source, "source")
        if not empty:
            store.record_action(identifier, 0, source.apply(Accept()))
        store.record_sampling_segment(
            identifier, start_boundary=source.boundary,
            sampling=replace(source.sampling, temperature=0.7),
            stream_fingerprint=source.stream_fingerprint, coordinate_offset=0,
        )
    io = ScriptedIO(["s temperature=0.2", "c", "q", "q"])
    args = ["--workspace", str(path), "--model", "fake", "--replay", "source",
            "--episode-id", "target", "--plain-ui"]
    if fixed:
        args += ["--temperature", "0.4", "--fixed-config"]
    with patch("trajectory_editor.episode_cli._backend", return_value=ConformingFakeBackend()), patch(
        "trajectory_editor.episode_cli.TerminalIO", return_value=io
    ):
        assert main(args) == 0
    headers = [line for line in io.output if "Live edge @ boundary" in line]
    assert ("temp=0.4" if fixed else "temp=0.7") in headers[0]
    assert "temp=0.2" in headers[-1]
    with EpisodeStore(path) as store:
        assert store.final_sampling("target").temperature == 0.2
        assert len(store.actions("target")) == (0 if empty else 1)

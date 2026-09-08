from unittest.mock import patch

import pytest

from tests.test_episode_runtime import NoEogBackend
from tests.fakes import ScriptedIO
from trajectory_editor.domain import EditorError, SamplingConfig
from trajectory_editor.episode_actions import Hold, Write
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_cli import main, _restore_engine, _spr_engine_from_source
from trajectory_editor.episode_store import EpisodeStore


def runtime(budget=None):
    return EpisodeEngine(NoEogBackend(), initial_token_ids=[7],
                         sampling=SamplingConfig(temperature=0), max_tokens=budget)


def test_unlimited_hold_exceeds_old_default():
    e = runtime()
    result = e.apply(Hold(200))
    assert len(result.visible_token_ids) == 200
    assert e.remaining is None
    assert not e.checkpointed
    assert result.stop_reason == "requested-length"


def test_allowance_preserved_renewed_and_removed():
    e = runtime(5)
    e.apply(Hold(2))
    e.resume()
    assert e.remaining == 3
    result = e.apply(Hold(3))
    assert len(result.visible_token_ids) == 3
    assert e.checkpointed
    e.resume()
    assert e.remaining == 5
    e.resume(max_tokens=None)
    assert e.remaining is None


def test_write_budget_is_atomic():
    e = runtime(1)
    with pytest.raises(EditorError, match="budget"):
        e.apply(Write(" A B", mode="exact"))
    assert e.boundary == 0


@pytest.mark.parametrize("budget", [None, 5])
def test_q_and_resume_preserve_budget_and_do_not_record_moves(tmp_path, budget):
    path = tmp_path / "episodes.sqlite3"
    io = ScriptedIO(["h 2", "q", "s temperature=0.5", "c", "q", "q"])
    args = ["--workspace", str(path), "--model", "fake", "--new-prompt", "P",
            "--episode-id", "test", "--plain-ui", "--temperature", "0"]
    if budget is not None:
        args += ["--max-tokens", str(budget)]
    backend = NoEogBackend()
    with patch("trajectory_editor.episode_cli._backend", return_value=backend), patch(
        "trajectory_editor.episode_cli.TerminalIO", return_value=io
    ):
        assert main(args) == 0
    with EpisodeStore(path) as store:
        assert [row["kind"] for row in store.actions("test")] == ["hold"]
        restored = _restore_engine(store, "test", NoEogBackend(),
                                   max_tokens=None, sampling_override=None)
        assert restored.boundary == 2
        assert restored.remaining == (None if budget is None else 3)
        assert restored.sampling.temperature == 0.5
        replay, _ = _spr_engine_from_source(
            store, "test", NoEogBackend(), sampling=restored.sampling,
            max_tokens=1,
        )
        assert replay.remaining == 1


def test_menu_can_remove_budget(tmp_path):
    io = ScriptedIO(["q", "n off", "h 4", "q", "q"])
    path = tmp_path / "episodes.sqlite3"
    with patch("trajectory_editor.episode_cli._backend", return_value=NoEogBackend()), patch(
        "trajectory_editor.episode_cli.TerminalIO", return_value=io
    ):
        assert main(["--workspace", str(path), "--model", "fake", "--new-prompt", "P",
                     "--episode-id", "test", "--plain-ui", "--max-tokens", "1",
                     "--temperature", "0"]) == 0
    with EpisodeStore(path) as store:
        episode = store.get_episode("test")
        assert episode["max_tokens"] is None
        assert episode["checkpoint_boundary"] is None
        assert len(store.tokens("test")) == 4


@pytest.mark.parametrize("oversized_write", [False, True])
@pytest.mark.parametrize("mode", ["handoff", "ballistic"])
def test_replay_budget_hands_off_without_sealing(tmp_path, oversized_write, mode):
    from trajectory_editor.episode_policy import EpisodeRunner, TapeStep
    from trajectory_editor.episode_engine import ReplayExpectation
    e = runtime(1)
    with EpisodeStore(tmp_path / "episode.sqlite3") as store:
        identifier = store.create_episode(
            initial_text="P", initial_token_ids=[7], sampling=e.sampling,
            stream_fingerprint=e.stream_fingerprint, coordinate_offset=0,
            max_tokens=1, backend={},
        )
        action = Write(" A B", mode="exact") if oversized_write else Hold(10)
        tape = [TapeStep(action, ReplayExpectation((1, 2), None, "requested-length"))]
        result = EpisodeRunner(e, store, identifier, divergence_policy=mode).run(tape=tape)
        assert not e.ended
        assert e.boundary == 0
        assert e.remaining == 1
        assert store.get_episode(identifier)["status"] != "failed"
        assert result.handed_off
        assert store.actions(identifier) == []



@pytest.mark.parametrize("boundary", [None, "sentence", "newline"])
def test_oversized_hold_rejected_before_observation(boundary):
    e = runtime(7)
    with patch.object(e.backend, "last_logits", side_effect=AssertionError("must not observe")):
        with pytest.raises(EditorError, match="20 tokens but only 7 remain"):
            e.apply(Hold(20, boundary))
    assert e.boundary == 0
    assert e.remaining == 7
    assert e.backend.tokens == [7]


def test_live_oversized_hold_opens_edge_and_can_be_replaced(tmp_path):
    io = ScriptedIO(["h 20", "c", "h 3", "q", "q"])
    path = tmp_path / "episodes.sqlite3"
    with patch("trajectory_editor.episode_cli._backend", return_value=NoEogBackend()), patch(
        "trajectory_editor.episode_cli.TerminalIO", return_value=io
    ):
        assert main(["--workspace", str(path), "--model", "fake", "--new-prompt", "P",
                     "--episode-id", "test", "--plain-ui", "--max-tokens", "7",
                     "--temperature", "0"]) == 0
    with EpisodeStore(path) as store:
        assert len(store.tokens("test")) == 3
        assert len(store.actions("test")) == 1
        restored = _restore_engine(store, "test", NoEogBackend(), max_tokens=None,
                                   sampling_override=None)
        assert restored.remaining == 4

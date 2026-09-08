from unittest.mock import patch
import pytest

from tests.test_replay_eog import engine, create, NeverChoose
from trajectory_editor.episode_actions import Accept, EndGeneration, SelectRawRank, Write
from trajectory_editor.episode_engine import ReplayExpectation
from trajectory_editor.episode_policy import EpisodeRunner, TapeStep
from trajectory_editor.episode_store import EpisodeStore


@pytest.mark.parametrize("mode", ["handoff", "ballistic"])
@pytest.mark.parametrize("kind", ["rank", "empty-write", "no-eog"])
def test_known_rejection_preserves_prior_work_and_returns_live_edge(tmp_path, mode, kind):
    runtime = engine()
    action = {"rank": SelectRawRank(999), "empty-write": Write("hello"),
              "no-eog": EndGeneration()}[kind]
    with EpisodeStore(tmp_path / "episode.sqlite3") as store:
        identifier = create(store, runtime)
        with patch.object(runtime.backend, "tokenize", return_value=[]), patch.object(
            runtime.backend, "eog_token_ids", return_value=()
        ):
            result = EpisodeRunner(runtime, store, identifier, divergence_policy=mode).run(
                tape=[TapeStep(Accept(), ReplayExpectation((1,), None, "completed")),
                      TapeStep(action, ReplayExpectation((4,))),
                      TapeStep(Accept(), ReplayExpectation((2,)))],
                live_policy=NeverChoose(),
            )
        assert result.handed_off
        assert result.handoff_reason
        assert runtime.visible_token_ids == [1]
        assert not runtime.ended
        assert len(store.actions(identifier)) == 1
        assert len(store.replay_tape(identifier)) == 1
        assert store.get_episode(identifier)["status"] == "replay-edge"
        event = store.interactions(identifier)[-1]
        assert event["kind"] == "instruction-rejected"
        assert event["payload"]["action"] == action.to_dict()
        assert event["payload"]["replay"]
        runtime.apply(Write("hello"))
        assert runtime.visible_token_ids == [1, 4]


def test_unexpected_tokenizer_failure_is_not_rejection(tmp_path):
    runtime = engine()
    with EpisodeStore(tmp_path / "episode.sqlite3") as store:
        identifier = create(store, runtime)
        with patch.object(runtime.backend, "tokenize", side_effect=RuntimeError("backend broken")):
            with pytest.raises(RuntimeError, match="backend broken"):
                EpisodeRunner(runtime, store, identifier).run(
                    tape=[TapeStep(Write("hello"), ReplayExpectation((4,)))])
        assert store.get_episode(identifier)["status"] == "failed"
        assert not store.interactions(identifier)
        assert not store.tokens(identifier)

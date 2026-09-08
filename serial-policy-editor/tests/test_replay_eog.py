from unittest.mock import patch
import pytest

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.episode_actions import Accept, EndGeneration, Hold, SelectRawRank, Write
from trajectory_editor.episode_engine import EpisodeEngine, ReplayExpectation
from trajectory_editor.episode_policy import EpisodeRunner, TapeStep
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_cli import main


def engine(prefix=7):
    return EpisodeEngine(ConformingFakeBackend(), initial_token_ids=[prefix],
                         sampling=SamplingConfig(temperature=0))


def create(store, runtime, identifier="test"):
    return store.create_episode(
        initial_text=runtime.text, initial_token_ids=list(runtime.initial_token_ids),
        sampling=runtime.sampling, stream_fingerprint=runtime.stream_fingerprint,
        coordinate_offset=0, max_tokens=None, backend={}, episode_id=identifier,
    )


@pytest.mark.parametrize("mode", ["handoff", "ballistic"])
@pytest.mark.parametrize("matches", [False, True])
@pytest.mark.parametrize("action,prefix,visible", [
    (Accept(), 2, ()),
    (SelectRawRank(1), 2, ()),
    (EndGeneration(), 7, ()),
    (Hold(5), 7, (1, 2)),
])
def test_eog_stops_tape_without_committing_terminal(tmp_path, mode, matches, action, prefix, visible):
    runtime = engine(prefix)
    expected = ReplayExpectation(visible, 0, "eog") if matches else ReplayExpectation(
        (*visible, 3), None, "requested-length"
    )
    with EpisodeStore(tmp_path / "episode.sqlite3") as store:
        identifier = create(store, runtime)
        result = EpisodeRunner(runtime, store, identifier, divergence_policy=mode).run(
            tape=[TapeStep(action, expected), TapeStep(Write("hello"), ReplayExpectation((4,)))],
            stop_after_tape=False,
            live_policy=NeverChoose(),
        )
        assert result.handed_off
        assert not result.replay_exhausted
        assert len(result.outcomes) == 1
        outcome = result.outcomes[0]
        assert outcome.stop_reason == "replay-eog"
        assert outcome.replay_eog_token_id == 0
        assert (outcome.divergence is None) == matches
        assert tuple(runtime.visible_token_ids) == visible
        assert runtime.terminal_token_id is None
        assert not runtime.ended
        assert runtime.backend.tokens == [prefix, *visible]
        assert all(not row["is_eog"] for row in store.tokens(identifier))
        assert store.get_episode(identifier)["status"] == "replay-edge"
        event = store.interactions(identifier)[-1]
        assert event["kind"] == "replay-eog"
        assert event["payload"]["matched_expectation"] == matches
        # A derived tape retains only visible work, never the attempted EOG.
        tape = store.replay_tape(identifier)
        assert len(tape) == (1 if visible else 0)
        if visible:
            assert tape[0][0] == Hold(len(visible))
        runtime.apply(Write("hello"))
        assert not runtime.ended


class NeverChoose:
    def choose(self, *args):
        raise AssertionError("Replay EOG must reach the menu before requesting a live move")


def test_live_eog_still_terminates():
    runtime = engine(2)
    outcome = runtime.apply(Accept())
    assert outcome.stop_reason == "eog"
    assert runtime.ended
    assert runtime.terminal_token_id == 0


def test_cli_replay_eog_returns_live_control(tmp_path):
    path = tmp_path / "episode.sqlite3"
    with EpisodeStore(path) as store:
        source = engine()
        identifier = create(store, source, "source")
        store.record_action(identifier, 0, source.apply(EndGeneration()))
    io = ScriptedIO(["c", "x hello", "q", "q"])
    with patch("trajectory_editor.episode_cli._backend", return_value=ConformingFakeBackend()), patch(
        "trajectory_editor.episode_cli.TerminalIO", return_value=io
    ):
        assert main(["--workspace", str(path), "--model", "fake", "--replay", "source",
                     "--episode-id", "target", "--plain-ui"]) == 0
    with EpisodeStore(path) as store:
        target = store.get_episode("target")
        assert target["terminal_token_id"] is None
        assert target["status"] == "open"
        assert [row["token_id"] for row in store.tokens("target")] == [4]
        assert len(store.actions("target")) == 2
    assert any("Replay encountered EOG" in text for text in io.output)

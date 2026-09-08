from unittest.mock import patch

import numpy as np
import pytest

from tests.fakes import ScriptedIO
from tests.test_episode_runtime import engine, LiveScriptedIO
from trajectory_editor.episode_actions import SelectRawRank
from trajectory_editor.episode_policy import EpisodeRunner
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_ui import InteractivePolicy
from tests.test_replay_eog import create


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("text", ["", "accept", "1"])
def test_teacher_entry_records_rank_and_agreement(tmp_path, live, manual, text):
    runtime = engine()
    io = (LiveScriptedIO if live else ScriptedIO)([text])
    action = InteractivePolicy(io=io, manual_acceptance=manual).choose(runtime, runtime.observe())
    assert action == SelectRawRank(1)
    outcome = runtime.apply(action)
    assert outcome.evidence[0].proposal_agreement
    with EpisodeStore(tmp_path / "episode.sqlite3") as store:
        identifier = create(store, runtime)
        store.record_action(identifier, 0, outcome)
        assert store.replay_tape(identifier)[0][0] == SelectRawRank(1)
        assert store.tokens(identifier)[0]["proposal_agreement"]


@pytest.mark.parametrize("mode", ["handoff", "ballistic"])
def test_same_proposal_at_changed_rank_does_not_rescue_replay(mode):
    source = engine()
    action = InteractivePolicy(io=ScriptedIO([""])).choose(source, source.observe())
    original = source.apply(action)
    target = engine()
    logits = np.full(target.backend.vocabulary_size(), -10.0)
    logits[2], logits[1] = 10.0, 9.0
    with patch.object(target.backend, "last_logits", return_value=logits), patch(
        "trajectory_editor.episode_engine.draw_token", return_value=1
    ):
        observation = target.observe()
        assert observation.proposal_token_id == original.resolved_token_ids[0] == 1
        assert observation.proposal_raw_rank == 2
        result = target.apply(action, replay=True, expectation=original.expectation(),
                              divergence_policy=mode)
    assert result.divergence is not None
    if mode == "handoff":
        assert result.visible_token_ids == ()
        assert result.status == "handed-off"
    else:
        assert result.visible_token_ids == (2,)
        assert not result.evidence[0].proposal_agreement

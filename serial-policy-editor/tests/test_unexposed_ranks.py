import pytest

from tests.fakes import ScriptedIO
from tests.test_episode_runtime import engine, LiveScriptedIO
from trajectory_editor.episode_actions import SelectRawRank
from trajectory_editor.episode_ui import InteractivePolicy, _choice_from_observation
from trajectory_editor.live_tui import action_preview


@pytest.mark.parametrize("live", [False, True])
def test_unexposed_rank_can_be_selected_after_invalid_rank(live):
    runtime = engine()
    observation = runtime.observe()
    io = (LiveScriptedIO if live else ScriptedIO)(["999", "8"])
    policy = InteractivePolicy(io=io, menu_size=1)
    action = policy.choose(runtime, observation)
    assert action == SelectRawRank(8)
    expected = runtime.candidates(observation, start_rank=8, count=1)[0]
    result = runtime.apply(action)
    assert result.resolved_token_ids == (expected.token_id,)


def test_numeric_proposal_records_concrete_rank():
    runtime = engine()
    observation = runtime.observe()
    policy = InteractivePolicy(io=ScriptedIO([str(observation.proposal_raw_rank)]), menu_size=1)
    assert policy.choose(runtime, observation) == SelectRawRank(observation.proposal_raw_rank)


def test_unexposed_preview_is_selectable_and_bounded():
    runtime = engine()
    observation = runtime.observe()
    rows = runtime.candidates(observation, count=1)
    choice = _choice_from_observation(runtime, observation, rows, context_characters=100, serial=1)
    preview = action_preview(choice, "8", rows, lambda *args: None)
    assert preview.valid
    assert "Press Enter to select raw rank 8" in preview.detail
    assert not action_preview(choice, "9", rows, lambda *args: None).valid
    assert not action_preview(choice, "0", rows, lambda *args: None).valid

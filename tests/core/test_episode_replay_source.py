from __future__ import annotations

import pytest

from tests.core.runtime_helpers import NoEogBackend
from trajectory_editor.core.actions import SetSampler, Write
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_replay_source import (
    build_source_replay_recipe,
    replay_procedure,
)
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.run_loop import run_plan
from trajectory_editor.episode_session import LiveSession
from trajectory_editor.spr_recipe import ReplaySamplerPolicy, ReplayPlacement, compose_replay_plan

pytestmark = pytest.mark.invariant

class MemoryReader:
    """Minimal durable-record reader; deliberately not an EpisodeStore."""

    def __init__(self) -> None:
        root = SamplerConfig(temperature=0.0, seed=3)
        changed = SamplerConfig(temperature=0.0, seed=5)
        self._episode = {
            "initial_text": "P",
            "initial_sampling": root.to_dict(),
            "initial_stream_fingerprint": "a" * 64,
        }
        self._actions = [
            {
                "ordinal": 0,
                "boundary_before": 0,
                "status": "completed",
                "stop_reason": "completed",
                "arguments": Write(" A B C", mode="exact").to_dict(),
            }
        ]
        self._tokens = [
            {
                "action_ordinal": 0,
                "token_id": 1,
                "text": " A",
                "realized_visible": True,
                "is_eog": False,
            },
            {
                "action_ordinal": 0,
                "token_id": 2,
                "text": " B",
                "realized_visible": True,
                "is_eog": False,
            },
            {
                "action_ordinal": 0,
                "token_id": 3,
                "text": " C",
                "realized_visible": True,
                "is_eog": False,
            },
        ]
        self.changed_sampling = changed

    def get_episode(self, episode_id):
        assert episode_id == "source"
        return self._episode

    def actions(self, episode_id):
        assert episode_id == "source"
        return self._actions

    def tokens(self, episode_id):
        assert episode_id == "source"
        return self._tokens

def test_reader_adapter_builds_a_root_relative_recipe_without_a_store():
    recipe = build_source_replay_recipe(MemoryReader(), "source", until=1)

    assert recipe.source_prompt == "P"
    assert recipe.source_visible_boundary == 3
    assert recipe.source_end_boundary == 1
    assert recipe.procedure.steps[0].action == Write(" A", mode="exact")
    assert recipe.procedure.steps[0].expectation.token_ids == (1,)
    assert recipe.initial_sampling.seed == 3


def test_reader_adapter_joins_all_text_pieces_for_a_partial_prefix():
    recipe = build_source_replay_recipe(MemoryReader(), "source", until=2)

    assert recipe.procedure.steps[0].action == Write(" A B", mode="exact")
    assert recipe.procedure.steps[0].expectation.token_ids == (1, 2)


def _two_step_reader() -> MemoryReader:
    reader = MemoryReader()
    reader._actions[0]["arguments"] = Write(" A B", mode="exact").to_dict()
    reader._tokens = reader._tokens[:2]
    reader._actions.extend([
        {
            "ordinal": 1,
            "boundary_before": 2,
            "status": "completed",
            "stop_reason": "completed",
            "arguments": {"kind": "future-action"},
        },
        {
            "ordinal": 2,
            "boundary_before": 2,
            "status": "completed",
            "stop_reason": "completed",
            "arguments": Write(" C", mode="exact").to_dict(),
        },
    ])
    return reader


def _run_reader(reader: MemoryReader, *, until: int | None = None):
    recipe = build_source_replay_recipe(reader, "source", until=until)
    plan = compose_replay_plan(
        recipe, ReplayPlacement.SOURCE_ROOT, ReplaySamplerPolicy.FOLLOW_SOURCE
    )
    session = LiveSession(EpisodeEngine(
        NoEogBackend(),
        sampling=recipe.initial_sampling,
        initial_text="P",
        initial_token_ids=[7],
    ))
    result = run_plan(session, divergence_policy="handoff", tape=plan)
    return recipe, plan, session, result


def test_unsupported_first_step_hands_off_without_action_or_sampler_change():
    reader = _two_step_reader()
    reader._actions = reader._actions[:1]
    reader._actions[0]["arguments"] = {"kind": "future-action"}
    reader._tokens = []

    recipe, plan, session, result = _run_reader(reader)

    assert recipe.procedure.steps == ()
    assert plan.steps == ()
    assert result.handed_off and not result.replay_exhausted
    assert result.replayed_actions == 0
    assert result.outcomes == ()
    assert session.engine.boundary == 0
    assert session.engine.sampling.seed == 3
    assert "source step 1" in result.handoff_reason
    assert "future-action" in result.handoff_reason


def test_unsupported_after_prefix_hands_off_before_later_actions_and_final_sampler():
    reader = _two_step_reader()

    recipe, plan, session, result = _run_reader(reader)

    assert len(recipe.procedure.steps) == len(plan.steps) == 1
    assert result.handed_off and not result.replay_exhausted
    assert result.replayed_actions == 1
    assert [outcome.action for outcome in result.outcomes] == [Write(" A B", "exact")]
    assert session.engine.visible_token_ids == [1, 2]
    assert session.engine.boundary == 2
    assert session.engine.sampling.seed == 3
    assert "source step 2" in result.handoff_reason
    assert "future-action" in result.handoff_reason
    with pytest.raises(EditorError, match="source step 2"):
        replay_procedure(reader, "source")


def test_supported_action_without_source_expectation_executes_without_comparison():
    reader = _two_step_reader()
    reader._actions = reader._actions[:1]
    reader._actions[0].pop("stop_reason")
    reader._tokens[0]["token_id"] = 5

    recipe, plan, session, result = _run_reader(reader)

    assert recipe.procedure.steps[0].expectation is None
    assert plan.steps[0].expectation is None
    assert result.replayed_actions == 1
    assert result.replay_exhausted and not result.handed_off
    assert result.outcomes[0].divergence is None
    assert session.engine.visible_token_ids == [1, 2]
    assert session.engine.sampling.seed == 3


def test_explicit_endpoint_before_unsupported_step_exhausts_normally():
    reader = _two_step_reader()

    _, _, session, result = _run_reader(reader, until=2)

    assert result.replayed_actions == 1
    assert result.replay_exhausted and not result.handed_off
    assert session.engine.visible_token_ids == [1, 2]
    assert session.engine.sampling.seed == 3


@pytest.mark.parametrize("arguments, message", [
    ({"kind": "hold", "limit": "bad"}, "valid limit"),
    ({"kind": ""}, "nonempty string"),
])
def test_malformed_action_is_not_unsupported_handoff(arguments, message):
    reader = _two_step_reader()
    reader._actions[1]["arguments"] = arguments

    with pytest.raises(EditorError, match=message):
        build_source_replay_recipe(reader, "source")


def test_source_sampler_transitions_within_supported_prefix_still_apply():
    reader = MemoryReader()
    reader._actions = [
        {**reader._actions[0], "arguments": Write(" A", "exact").to_dict()},
        {
            "ordinal": 1,
            "boundary_before": 1,
            "status": "completed",
            "stop_reason": "sampler-change",
            "arguments": SetSampler(reader.changed_sampling).to_dict(),
        },
        {
            "ordinal": 2,
            "boundary_before": 1,
            "status": "completed",
            "stop_reason": "completed",
            "arguments": Write(" B", "exact").to_dict(),
        },
        {
            "ordinal": 3,
            "boundary_before": 2,
            "status": "completed",
            "stop_reason": "completed",
            "arguments": {"kind": "future-action"},
        },
    ]
    reader._tokens[1]["action_ordinal"] = 2
    reader._tokens[2]["action_ordinal"] = 3

    _, plan, session, result = _run_reader(reader)

    assert len(plan.steps) == result.replayed_actions == 3
    assert result.handed_off and not result.replay_exhausted
    assert session.engine.visible_token_ids == [1, 2]
    assert session.engine.sampling.seed == 5


def test_supported_action_without_any_recorded_result_runs_at_full_endpoint():
    reader = MemoryReader()
    reader._actions[0]["arguments"] = Write(" A", "exact").to_dict()
    reader._actions[0].pop("stop_reason")
    reader._tokens = []

    recipe, plan, session, result = _run_reader(reader)

    assert recipe.source_visible_boundary == 0
    assert len(plan.steps) == 1
    assert plan.steps[0].expectation is None
    assert result.replayed_actions == 1
    assert result.replay_exhausted and not result.handed_off
    assert session.engine.visible_token_ids == [1]
    assert result.outcomes[0].divergence is None

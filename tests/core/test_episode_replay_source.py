from __future__ import annotations

from trajectory_editor.core.actions import Write
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_replay_source import (
    build_source_replay_recipe,
    final_sampling,
)


class MemoryReader:
    """Minimal durable-record reader; deliberately not an EpisodeStore."""

    def __init__(self) -> None:
        root = SamplerConfig(temperature=0.0, seed=3)
        changed = SamplerConfig(temperature=0.0, seed=5)
        self._episode = {"initial_text": "P"}
        self._actions = [
            {
                "ordinal": 0,
                "boundary_before": 0,
                "status": "completed",
                "stop_reason": "completed",
                "arguments": Write(" A B", mode="exact").to_dict(),
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
        ]
        self._samplers = [
            {
                "start_boundary": 0,
                "sampling": root.to_dict(),
                "stream_fingerprint": "a" * 64,
                "coordinate_offset": 11,
            },
            {
                "start_boundary": 1,
                "sampling": changed.to_dict(),
                "stream_fingerprint": "a" * 64,
                "coordinate_offset": 11,
            },
        ]
        self._budgets = [
            {
                "start_boundary": 0,
                "max_tokens": None,
                "checkpoint_boundary": None,
            }
        ]

    def get_episode(self, episode_id):
        assert episode_id == "source"
        return self._episode

    def actions(self, episode_id):
        assert episode_id == "source"
        return self._actions

    def tokens(self, episode_id):
        assert episode_id == "source"
        return self._tokens

    def sampler_segments(self, episode_id):
        assert episode_id == "source"
        return self._samplers

    def budget_segments(self, episode_id):
        assert episode_id == "source"
        return self._budgets


def test_reader_adapter_builds_a_root_relative_recipe_without_a_store():
    recipe = build_source_replay_recipe(MemoryReader(), "source", until=1)

    assert recipe.source_prompt == "P"
    assert recipe.source_visible_boundary == 2
    assert recipe.source_end_boundary == 1
    assert recipe.procedure.steps[0].action == Write(" A", mode="exact")
    assert recipe.procedure.steps[0].expectation.token_ids == (1,)
    assert recipe.controls.effective_at(0).coordinate_offset == 11
    assert recipe.controls.effective_at(0).sampling.seed == 3


def test_reader_adapter_preserves_trailing_source_sampler_state():
    assert final_sampling(MemoryReader(), "source").seed == 5

"""Regression coverage for the executable controller-stack contract.

These tests deliberately describe the runtime surfaces, not the first-slice
display-only stack.  A few assertions are expected to fail until the runtime
seam is introduced; keeping them strict prevents the display order from
becoming an accidental compatibility contract.
"""

from dataclasses import replace
from math import log
from unittest.mock import patch

import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend, ScriptedIO
from tests.research.test_token_preference import FEATURES
from trajectory_editor.bias_rules import BiasGroup, BiasRule
from trajectory_editor.controller_stack import build_controller_stack
from trajectory_editor.domain import EditorError, SamplingConfig
from trajectory_editor.episode_actions import Accept, Hold, SelectRawRank
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_lifecycle import (
    _create_episode,
    _fork_engine,
    _rewind_episode,
)
from trajectory_editor.episode_policy import EpisodeRunner, TapeStep
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_ui import InteractivePolicy
from trajectory_editor.group_control import GroupControl


def plain_sampling(**kwargs) -> SamplingConfig:
    values = dict(
        temperature=0.0,
        top_k=8,
        top_p=1.0,
        min_p=0.0,
    )
    values.update(kwargs)
    return SamplingConfig(**values)


def cvector_sampling(values, *, digest="a" * 64) -> SamplingConfig:
    return plain_sampling(
        activation_vector=tuple(values),
        activation_vector_strength=1.0,
        activation_vector_layer="control-vector",
        activation_vector_position="layers",
        activation_vector_layer_start=1,
        activation_vector_layer_end=2,
        activation_vector_digest=digest,
    )


class StickyControlBackend(ConformingFakeBackend):
    """Fake backend whose installed cvector changes the returned logits."""

    def __init__(self):
        super().__init__()
        self.current_control = None
        self.set_calls = []
        self.clear_calls = 0
        self.branch_prefixes = []

    def activation_control_vector_width(self):
        return 3

    def activation_control_vector_layer_count(self):
        return 2

    def set_activation_control_vector(
        self, vector, *, layer_start, layer_end, strength
    ):
        del layer_start, layer_end, strength
        self.current_control = tuple(float(value) for value in vector)
        self.set_calls.append(self.current_control)

    def clear_activation_control_vector(self):
        self.current_control = None
        self.clear_calls += 1

    def branch_to_prefix(self, prefix_token_ids):
        self.branch_prefixes.append(list(prefix_token_ids))
        self.tokens = list(prefix_token_ids)

    def last_logits(self):
        logits = super().last_logits()
        if self.current_control is not None:
            weights = np.arange(1, len(self.current_control) + 1, dtype=float)
            logits[1] += float(np.dot(weights, self.current_control))
        return logits


class ActivationPreferenceBackend(ConformingFakeBackend):
    matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 0.5, 0.25],
            [0.5, -1.0, 0.5],
            [0.25, 0.5, -1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    def activation_logit_adjustments(
        self, vector, *, layer="output", position="current"
    ):
        assert layer == "output"
        assert position == "current"
        return self.matrix @ np.asarray(vector, dtype=np.float32)

    def token_preference_features(self, *, feature_dimension, projection_seed):
        del projection_seed
        assert feature_dimension == FEATURES.shape[1]
        return FEATURES








def test_controller_display_contract_exposes_missing_model_and_group_stages():
    group = BiasGroup("g", (BiasRule(routes=((2,),), bias=0.0),))
    config = cvector_sampling((1, 0, 0, 0, 0, 0), digest="a" * 64)
    config = replace(config, bias_groups=(group,), group_controls=(GroupControl("g", "promote", 0.1),))
    stack = build_controller_stack(sampling=config, backend="llama.cpp")

    policy_names = [entry.name for entry in stack.entries if entry.phase == "policy"]
    assert policy_names == [
        "base model",
        "history penalties",
        "output-head steering",
        "manual biases/groups",
        "reference prior",
        "token preference actuator",
        "group control",
        "sampler / token draw",
    ]
    assert next(entry for entry in stack.entries if entry.name == "layerwise hidden-state control").phase == "model"


def test_raw_rank_and_policy_rank_actions_remain_distinct():
    config = plain_sampling(
        bias_rules=(BiasRule(routes=((3,),), bias=20.0),),
    )
    policy_runtime = EpisodeEngine(
        ConformingFakeBackend(), initial_token_ids=[7], sampling=config
    )
    observation = policy_runtime.observe()
    assert observation.proposal_token_id == 3
    assert observation.proposal_raw_rank == 3
    assert observation.proposal_policy_rank == 1
    assert policy_runtime.apply(Accept()).resolved_token_ids == (3,)

    raw_runtime = EpisodeEngine(
        ConformingFakeBackend(), initial_token_ids=[7], sampling=config
    )
    assert raw_runtime.apply(SelectRawRank(1)).resolved_token_ids == (1,)

    ui_runtime = EpisodeEngine(
        ConformingFakeBackend(), initial_token_ids=[7], sampling=config
    )
    ui_observation = ui_runtime.observe()
    ui_action = InteractivePolicy(io=ScriptedIO(["accept"])).choose(
        ui_runtime, ui_observation
    )
    assert ui_action == SelectRawRank(ui_observation.proposal_raw_rank)


def test_reused_backend_control_vector_to_plain_runtime_is_isolated():
    backend = StickyControlBackend()
    active = EpisodeEngine(
        backend,
        initial_token_ids=[7],
        sampling=cvector_sampling((1, 0, 0, 0, 0, 0)),
    )
    active.observe()
    assert backend.current_control is not None

    plain = EpisodeEngine(backend, initial_token_ids=[7], sampling=plain_sampling())
    observed = plain.observe()
    fresh = EpisodeEngine(
        StickyControlBackend(), initial_token_ids=[7], sampling=plain_sampling()
    ).observe()

    assert backend.current_control is None
    np.testing.assert_allclose(observed.statistics.logits, fresh.statistics.logits)


def test_reused_backend_plain_to_control_vector_installs_control():
    backend = StickyControlBackend()
    plain = EpisodeEngine(backend, initial_token_ids=[7], sampling=plain_sampling())
    plain.observe()
    active = EpisodeEngine(
        backend,
        initial_token_ids=[7],
        sampling=cvector_sampling((0, 1, 0, 0, 0, 0)),
    )
    active.observe()

    assert backend.current_control == (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert len(backend.set_calls) == 1


def test_control_vector_a_to_b_reinstalls_the_changed_vector():
    backend = StickyControlBackend()
    first = cvector_sampling((1, 0, 0, 0, 0, 0), digest="a" * 64)
    second = cvector_sampling((0, 1, 0, 0, 0, 0), digest="b" * 64)
    runtime = EpisodeEngine(backend, initial_token_ids=[7], sampling=first)
    runtime.observe()
    runtime.sampling = second
    runtime.observe()

    assert backend.current_control == second.activation_vector
    assert backend.set_calls == [first.activation_vector, second.activation_vector]


def test_changed_vector_with_unchanged_digest_is_not_silently_cached():
    backend = StickyControlBackend()
    first = cvector_sampling((1, 0, 0, 0, 0, 0), digest="a" * 64)
    changed = cvector_sampling((0, 1, 0, 0, 0, 0), digest="a" * 64)
    runtime = EpisodeEngine(backend, initial_token_ids=[7], sampling=first)
    runtime.observe()
    runtime.sampling = changed
    runtime.observe()

    assert backend.current_control == changed.activation_vector


def test_active_control_vector_requires_a_verified_digest():
    with pytest.raises(EditorError):
        cvector_sampling((1, 0, 0, 0, 0, 0), digest="")


def test_rewind_from_control_vector_to_plain_sampler_clears_backend_state(tmp_path):
    backend = StickyControlBackend()
    runtime = EpisodeEngine(backend, initial_token_ids=[7], sampling=plain_sampling())
    with EpisodeStore(tmp_path / "rewind.db") as store:
        episode_id = _create_episode(
            store, runtime, backend_provenance=backend.provenance()
        )
        outcome = runtime.apply(Hold(1))
        store.record_action(episode_id, 0, outcome)
        active = cvector_sampling((1, 0, 0, 0, 0, 0))
        runtime.sampling = active
        store.record_sampling_segment(
            episode_id,
            start_boundary=1,
            sampling=active,
            stream_fingerprint=runtime.stream_fingerprint,
            coordinate_offset=runtime.coordinate_offset,
        )
        runtime.observe()
        assert backend.current_control is not None

        _rewind_episode(store, episode_id, runtime, 0)
        runtime.observe()
        assert backend.current_control is None


def test_fork_branch_from_control_vector_to_plain_sampler_clears_backend_state(tmp_path):
    backend = StickyControlBackend()
    active = cvector_sampling((1, 0, 0, 0, 0, 0))
    runtime = EpisodeEngine(backend, initial_token_ids=[7], sampling=active)
    with EpisodeStore(tmp_path / "fork.db") as store:
        episode_id = _create_episode(
            store, runtime, backend_provenance=backend.provenance()
        )
        outcome = runtime.apply(Hold(1))
        store.record_action(episode_id, 0, outcome)
        plain = plain_sampling()
        runtime.sampling = plain
        store.record_sampling_segment(
            episode_id,
            start_boundary=1,
            sampling=plain,
            stream_fingerprint=runtime.stream_fingerprint,
            coordinate_offset=runtime.coordinate_offset,
        )
        # The parent has not observed the new plain boundary, so the backend
        # still carries the cvector when the fork reuses its branch cache.
        assert backend.current_control is not None
        child = _fork_engine(
            store,
            episode_id,
            runtime,
            1,
            backend=backend,
            max_tokens=None,
        )
        child.observe()

        assert backend.branch_prefixes[-1] == [7, *runtime.visible_token_ids[:1]]
        assert backend.current_control is None


class ExplodingLearner:
    enabled = True

    def update(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("replay must not update learners")


def replay_sampling():
    group = BiasGroup("g", (BiasRule(routes=((2,),), bias=0.0),))
    return plain_sampling(
        bias_rules=(BiasRule(routes=((3,),), bias=0.5),),
        bias_groups=(group,),
        group_controls=(GroupControl("g", "promote", 0.1),),
        reference_prior_routes=(((4,), 1.0), ((5,), 2.0)),
        reference_prior_scope="global",
        reference_prior_mode="contrastive",
        activation_vector=(1.0, 0.0, 0.0),
        activation_vector_strength=0.5,
        activation_vector_digest="c" * 64,
        token_preference_vector=(0.2, -0.1),
    )


def test_replay_preserves_controller_state_without_updating_learners(tmp_path):
    sampling = replay_sampling()
    source_backend = ActivationPreferenceBackend()
    source = EpisodeEngine(source_backend, initial_token_ids=[7], sampling=sampling)
    source_observation = source.observe()
    source_outcome = source.apply(Hold(1))

    with EpisodeStore(tmp_path / "replay.db") as store:
        source_id = _create_episode(
            store, source, backend_provenance=source_backend.provenance()
        )
        store.record_action(source_id, 0, source_outcome)

        replay_backend = ActivationPreferenceBackend()
        replay = EpisodeEngine(
            replay_backend, initial_token_ids=[7], sampling=source.sampling
        )
        replay_observation = replay.observe()
        replay_id = _create_episode(
            store, replay, backend_provenance=replay_backend.provenance()
        )
        result = EpisodeRunner(
            replay,
            store,
            replay_id,
            learner=ExplodingLearner(),
            token_preference_learner=ExplodingLearner(),
        ).run(
            tape=[TapeStep(source_outcome.action, source_outcome.expectation())]
        )

    assert result.replayed_actions == 1
    assert replay.sampling == source.sampling
    assert replay.visible_token_ids == source.visible_token_ids
    np.testing.assert_allclose(
        replay_observation.statistics.adjusted,
        source_observation.statistics.adjusted,
    )

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor.bias_rules import BiasMatcher, BiasRule
from trajectory_editor.core.actions import (
    Accept,
    EndGeneration,
    Finish,
    Hold,
    Phrase,
    SelectRawRank,
    Write,
    action_from_dict,
)
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.results import ReplayExpectation
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.core.observation import ObservationStatistics
from trajectory_editor.core.sampling import (
    MAX_SEED,
    MIN_SEED,
    SparseDistribution,
    draw_token,
    position_uniform_token,
    raw_rank,
    top_raw_ids,
)
from trajectory_editor.episode_engine import EpisodeEngine


def test_s01_sampler_config_accepts_rejects_and_round_trips_core_state():
    config = SamplerConfig(
        temperature=0.7,
        top_k=12,
        top_p=0.9,
        min_p=0.05,
        typical_p=0.8,
        tail_free_z=0.7,
        draw_kernel="gumbel-max",
        cfg_scale=1.4,
        cfg_prefix_tokens=3,
        repeat_penalty=1.1,
        seed=77,
    )
    record = config.to_dict()
    record.update(
        token_preference_vector=[1, 2, 3],
        reference_prior_routes=[],
        group_controls=[{"research": "ignored"}],
    )
    assert SamplerConfig.from_record(record) == config
    for invalid in (MIN_SEED - 1, MAX_SEED + 1):
        with pytest.raises(EditorError):
            SamplerConfig(seed=invalid)
    for invalid in (math.nan, math.inf, -math.inf):
        with pytest.raises(EditorError):
            SamplerConfig(temperature=invalid)
    with pytest.raises(ValueError, match="decoder logits"):
        ObservationStatistics(np.asarray([0.0, math.nan]), SamplerConfig(), [])


def test_s02_sampler_draws_and_candidate_filters_are_deterministic():
    logits = np.asarray([4.0, 3.0, 2.0, 1.0, 0.0])
    baseline = ObservationStatistics(
        logits, SamplerConfig(top_k=5, top_p=1.0, min_p=0.0), []
    )
    neutral = ObservationStatistics(
        logits,
        SamplerConfig(top_k=5, top_p=1.0, min_p=0.0, typical_p=1.0, tail_free_z=1.0),
        [],
    )
    assert neutral.distribution.ids.tolist() == baseline.distribution.ids.tolist()
    typical = ObservationStatistics(
        np.asarray([6.0, 3.0, 2.0, 1.0, 0.0, -1.0]),
        SamplerConfig(top_k=6, top_p=1.0, min_p=0.0, typical_p=0.35),
        [],
    )
    tail_free = ObservationStatistics(
        np.asarray([6.0, 3.0, 2.0, 1.0, 0.0, -1.0]),
        SamplerConfig(top_k=6, top_p=1.0, min_p=0.0, tail_free_z=0.35),
        [],
    )
    assert len(typical.distribution.ids) < 6
    assert len(tail_free.distribution.ids) < 6
    assert top_raw_ids(np.zeros(16), 5) == [0, 1, 2, 3, 4]
    assert [raw_rank(logits, token_id) for token_id in range(5)] == [1, 2, 3, 4, 5]

    distribution = SparseDistribution(
        np.asarray([4, 1, 7], dtype=np.int64),
        np.asarray([0.2, 0.5, 0.3], dtype=np.float64),
        np.asarray([0.1, 0.9, 0.4], dtype=np.float64),
    )
    kwargs = dict(seed=17, stream_fingerprint="a" * 64, aligned_step=3)
    assert draw_token(distribution, kernel="gumbel-max", **kwargs) == draw_token(
        distribution, kernel="gumbel-max", **kwargs
    )
    assert 0.0 < position_uniform_token(17, "a" * 64, 3, 4) < 1.0


def test_s03_cfg_is_scoped_to_its_configured_prefix():
    sampling = SamplerConfig(
        temperature=0.0,
        top_k=8,
        top_p=1.0,
        min_p=0.0,
        cfg_unconditional_prompt=" A",
        cfg_scale=2.0,
        cfg_prefix_tokens=1,
    )
    runtime = EpisodeEngine(
        ConformingFakeBackend(),
        guidance_backend=ConformingFakeBackend(),
        sampling=sampling,
        initial_token_ids=[7],
    )
    assert runtime.observe().proposal_token_id == 1
    runtime.apply(Accept())
    assert not runtime._cfg_active()


def test_s04_history_penalties_change_policy_order_not_raw_rank():
    logits = np.asarray([2.0, 1.0, 0.0, -1.0])
    config = SamplerConfig(
        temperature=1.0,
        top_k=2,
        top_p=1.0,
        min_p=0.0,
        repeat_penalty=2.0,
        repeat_last_n=-1,
        presence_penalty=0.5,
        frequency_penalty=0.25,
    )
    observation = ObservationStatistics(logits, config, [0, 0, 1])
    assert [observation.policy_rank(token) for token in [0, 1, 2]] == [1, 3, 2]
    np.testing.assert_allclose(observation.policy_logits - observation.backend_logits,
                               [-2.0, -1.25, 0.0, 0.0])
    assert observation.raw_rank(1) == 2
    assert observation.distribution.ids.tolist() == [0, 2]
    with pytest.raises(ValueError, match="exact prefix token ids"):
        ObservationStatistics(np.asarray([1.0, 0.0]), SamplerConfig(repeat_penalty=1.1), None)


def test_s05_tail_bias_assigns_multi_token_credit_only_to_final_token():
    matcher = BiasMatcher((BiasRule(routes=((10, 11),), bias=2.0, mode="tail"),))
    assert matcher.active_biases([]) == {}
    assert matcher.active_biases([10]) == {11: 2.0}

    tail = BiasMatcher((BiasRule(routes=((10, 11, 12),), bias=3.0, mode="tail"),))
    assert tail.active_biases([10]) == {}
    assert tail.active_biases([10, 11]) == {12: 3.0}


def test_s06_conditional_bias_waits_for_trigger_and_stops_at_terminator():
    matcher = BiasMatcher((BiasRule(
        routes=((20,),),
        bias=2.0,
        mode="tail",
        triggers=((7, 1, 2),),
        until=6,
    ),))
    assert matcher.active_biases([7, 1]) == {}
    assert matcher.active_biases([7, 1, 2]) == {20: 2.0}
    assert matcher.active_biases([7, 1, 2, 6]) == {}


@pytest.mark.parametrize(
    "action",
    [
        Accept(),
        SelectRawRank(3),
        Write(" hello", mode="exact"),
        Phrase("hello", mode="exact", force=True),
        Hold(2, boundary="sentence"),
        Finish(),
        EndGeneration(),
    ],
)
def test_s07_core_actions_and_replay_expectations_round_trip(action):
    assert action_from_dict(action.to_dict()) == action
    expectation = ReplayExpectation((1, 2), terminal_token_id=0, stop_reason="eog")
    assert ReplayExpectation.from_mapping({
        "token_ids": [1, 2],
        "terminal_token_id": 0,
        "stop_reason": "eog",
    }) == expectation
    with pytest.raises(EditorError):
        SelectRawRank(0)


def test_s08_engine_uses_core_sampler_without_research_fields():
    assert "token_preference_vector" not in SamplerConfig.__dataclass_fields__
    assert "reference_prior_routes" not in SamplerConfig.__dataclass_fields__
    runtime = EpisodeEngine(
        ConformingFakeBackend(),
        sampling=SamplerConfig(temperature=0.0, top_k=8, top_p=1.0, min_p=0.0),
        initial_token_ids=[7],
    )
    runtime.apply(Accept())
    assert runtime.boundary == 1

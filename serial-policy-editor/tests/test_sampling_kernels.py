from __future__ import annotations

import numpy as np

from tests.fakes import ConformingFakeBackend
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.episode_actions import Accept
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.controller_pipeline import ControllerPipeline
from trajectory_editor.sampling import (
    ObservationStatistics,
    SparseDistribution,
    draw_token,
    position_uniform_token,
)


def test_typical_and_tail_free_are_neutral_at_one():
    logits = np.asarray([4.0, 3.0, 2.0, 1.0, 0.0])
    baseline = ObservationStatistics(
        logits, SamplingConfig(top_k=5, top_p=1.0, min_p=0.0), []
    )
    configured = ObservationStatistics(
        logits,
        SamplingConfig(
            top_k=5, top_p=1.0, min_p=0.0, typical_p=1.0, tail_free_z=1.0
        ),
        [],
    )
    np.testing.assert_array_equal(configured.distribution.ids, baseline.distribution.ids)
    np.testing.assert_allclose(
        configured.distribution.probabilities, baseline.distribution.probabilities
    )


def test_typical_and_tail_free_reduce_candidates_before_final_filters():
    logits = np.asarray([6.0, 3.0, 2.0, 1.0, 0.0, -1.0])
    typical = ObservationStatistics(
        logits,
        SamplingConfig(top_k=6, top_p=1.0, min_p=0.0, typical_p=0.35),
        [],
    )
    tail_free = ObservationStatistics(
        logits,
        SamplingConfig(top_k=6, top_p=1.0, min_p=0.0, tail_free_z=0.35),
        [],
    )
    assert len(typical.distribution.ids) < len(logits)
    assert len(tail_free.distribution.ids) < len(logits)
    assert typical.candidate_filter_diagnostics["boundary"] == "candidate-filter -> draw-kernel"


def test_gumbel_max_is_deterministic_and_order_independent():
    distribution = SparseDistribution(
        np.asarray([4, 1, 7], dtype=np.int64),
        np.asarray([0.2, 0.5, 0.3], dtype=np.float64),
        np.asarray([0.1, 0.9, 0.4], dtype=np.float64),
    )
    kwargs = dict(seed=17, stream_fingerprint="a" * 64, aligned_step=3)
    first = draw_token(distribution, kernel="gumbel-max", **kwargs)
    assert first == draw_token(distribution, kernel="gumbel-max", **kwargs)
    assert 0.0 < position_uniform_token(17, "a" * 64, 3, 4) < 1.0


def test_cfg_guides_only_the_requested_prefix():
    sampling = SamplingConfig(
        temperature=0.0,
        top_k=8,
        top_p=1.0,
        min_p=0.0,
        cfg_unconditional_prompt=" A",
        cfg_scale=2.0,
        cfg_prefix_tokens=1,
    )
    engine = EpisodeEngine(
        ConformingFakeBackend(),
        guidance_backend=ConformingFakeBackend(),
        sampling=sampling,
        initial_token_ids=[7],
    )
    first = engine.observe()
    assert first.proposal_token_id == 1
    engine.apply(Accept())
    assert not engine._cfg_active()


def test_cfg_trace_names_the_model_phase_when_trace_capture_is_enabled():
    sampling = SamplingConfig(
        temperature=0.0,
        top_k=8,
        top_p=1.0,
        min_p=0.0,
        cfg_unconditional_prompt=" A",
        cfg_scale=1.5,
        cfg_prefix_tokens=2,
    )
    engine = EpisodeEngine(
        ConformingFakeBackend(),
        guidance_backend=ConformingFakeBackend(),
        sampling=sampling,
        initial_token_ids=[7],
        controller_pipeline=ControllerPipeline(capture_trace=True),
    )
    trace = engine.observe().statistics.controller_trace
    assert trace is not None
    assert trace.stages[0].name == "classifier-free guidance"
    assert trace.stages[0].phase == "model"
    assert trace.stage("classifier-free guidance").diagnostics["branch_scope"] == (
        "conditional-only hidden-state controls"
    )


def test_canonical_model_and_policy_names_alias_legacy_fields():
    stats = ObservationStatistics(
        np.asarray([1.0, 0.0]), SamplingConfig(top_k=2, top_p=1.0, min_p=0.0), []
    )
    np.testing.assert_array_equal(stats.backend_logits, stats.logits)
    np.testing.assert_array_equal(stats.policy_logits, stats.adjusted)
    assert stats.model_rank(0) == stats.raw_rank(0)


def test_new_sampler_and_cfg_fields_round_trip_through_saved_records():
    config = SamplingConfig(
        typical_p=0.8,
        tail_free_z=0.7,
        draw_kernel="gumbel-max",
        cfg_unconditional_prompt="neutral",
        cfg_scale=1.6,
        cfg_prefix_tokens=5,
    )
    restored = SamplingConfig.from_record(config.to_dict())
    assert restored == config

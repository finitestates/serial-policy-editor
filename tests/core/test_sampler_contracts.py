from __future__ import annotations

import math

import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor.bias_commands import format_group_report, format_token_report
from trajectory_editor.bias_groups import (
    BiasGroup,
    BiasMember,
    BiasRoute,
    BiasToken,
    member_surfaces,
)
from trajectory_editor.core.actions import (
    Accept,
    EndGeneration,
    Hold,
    Phrase,
    SelectRawRank,
    Write,
    action_from_dict,
)
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.results import ReplayExpectation
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.core.policy_calculations import PolicyCalculations
from trajectory_editor.core.sampling import (
    MAX_SEED,
    MIN_SEED,
    EligibleScores,
    draw_token,
    find_seed_for_token,
    perturbation_ranking_scores,
    position_uniform_token,
    raw_rank,
    top_raw_ids,
)
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.teacher_plan import load_teacher_plan


@pytest.mark.invariant
def test_s01_sampler_config_accepts_rejects_and_round_trips_core_state():
    config = SamplerConfig(
        temperature=0.7,
        top_k=12,
        min_p=0.05,
        draw_kernel='gumbel-max',
        cfg_scale=1.4,
        cfg_prefix_tokens=3,
        repeat_penalty=1.1,
        seed=77,
    )
    assert SamplerConfig.from_record(config.to_dict()) == config
    for retired in ('top_p', 'typical_p', 'tail_free_z'):
        with pytest.raises(EditorError):
            SamplerConfig.from_record(config.to_dict() | {retired: 1.0})
    for invalid in (MIN_SEED - 1, MAX_SEED + 1):
        with pytest.raises(EditorError):
            SamplerConfig(seed=invalid)
    for invalid in (math.nan, math.inf, -math.inf):
        with pytest.raises(EditorError):
            SamplerConfig(temperature=invalid)
    with pytest.raises(ValueError, match="logits"):
        PolicyCalculations(np.asarray([0.0, math.nan]), SamplerConfig(), [])


def test_student_t_df_is_positive_finite_serialized_and_legacy_defaults_to_three():
    config = SamplerConfig(draw_kernel="student-t-max", student_t_df=1.0)
    assert SamplerConfig.from_record(config.to_dict()) == config
    legacy_record = config.to_dict()
    legacy_record.pop("student_t_df")
    assert SamplerConfig.from_record(legacy_record).student_t_df == 3.0
    for invalid in (0.0, -1.0, math.nan, math.inf, -math.inf, 10**10000):
        with pytest.raises(EditorError, match="student_t_df"):
            SamplerConfig(student_t_df=invalid)


def test_s02_sampler_draws_and_candidate_filters_are_deterministic():
    logits = np.asarray([4., 3., 2., 1., 0.])
    baseline = PolicyCalculations(logits, SamplerConfig(), [])
    assert baseline.distribution.ids.tolist() == [0, 1, 2, 3, 4]
    assert 'softmax' not in baseline.distribution.__dict__
    gap = PolicyCalculations(logits, SamplerConfig(top_k=3, min_p=math.exp(-1.)), [])
    assert gap.distribution.ids.tolist() == [0, 1]
    assert top_raw_ids(np.zeros(16), 5) == [0, 1, 2, 3, 4]
    assert [raw_rank(logits, token_id) for token_id in range(5)] == [1, 2, 3, 4, 5]
    distribution = EligibleScores(np.asarray([4, 1, 7]), np.asarray([.9, .9, .4]))
    kwargs = dict(seed=17, stream_fingerprint='a' * 64, aligned_step=3)
    assert draw_token(distribution, kernel='argmax', **kwargs) == 1
    assert draw_token(distribution, kernel='gumbel-max', **kwargs) == draw_token(distribution, kernel='gumbel-max', **kwargs)
    assert 'softmax' not in distribution.__dict__



def test_targeted_seed_search_finds_only_active_candidates():
    distribution = EligibleScores(
        np.asarray([1, 4, 7], dtype=np.int64),
        np.asarray([0.1, 0.9, 0.4], dtype=np.float64),
    )
    fingerprint = "a" * 64
    target = 4
    draw = lambda seed: draw_token(
        distribution,
        seed=seed,
        stream_fingerprint=fingerprint,
        aligned_step=3, kernel="gumbel-max",
    )
    nonmatching = next(seed for seed in range(1000) if draw(seed) != target)
    matching = next(seed for seed in range(1000) if draw(seed) == target)
    candidate_seeds = iter((12345, nonmatching, matching))

    found, checked = find_seed_for_token(
        distribution,
        target,
        current_seed=12345,
        stream_fingerprint=fingerprint,
        aligned_step=3,
        kernel="gumbel-max",
        next_seed=lambda: next(candidate_seeds),
    )

    assert found == matching
    assert checked == 3
    with pytest.raises(EditorError, match="outside the active truncated candidate set"):
        find_seed_for_token(
            distribution,
            2,
            current_seed=12345,
            stream_fingerprint=fingerprint,
            aligned_step=3,
            kernel="gumbel-max",
            next_seed=lambda: pytest.fail("ineligible targets must not search seeds"),
        )


def test_student_t_sampler_preserves_df3_draws_and_supports_cauchy_df1():
    distribution = EligibleScores(
        np.asarray([5, 2, 9], dtype=np.int64),
        np.asarray([1.0, 0.0, -2.0], dtype=np.float64),
    )
    kwargs = dict(
        seed=19,
        stream_fingerprint="a" * 64,
        aligned_step=4,
        kernel="student-t-max",
    )
    legacy_df3 = perturbation_ranking_scores(distribution, **kwargs)
    np.testing.assert_allclose(
        legacy_df3,
        [1.3532073552692647, 1.9757514253850954, -2.3560151756666734],
        rtol=0.0,
        atol=1e-15,
    )

    cauchy_scores = perturbation_ranking_scores(
        distribution, **kwargs, student_t_df=1.0
    )
    np.testing.assert_array_equal(
        cauchy_scores,
        perturbation_ranking_scores(distribution, **kwargs, student_t_df=1.0),
    )
    assert np.all(np.isfinite(cauchy_scores))
    assert not np.array_equal(cauchy_scores, legacy_df3)
    fractional_scores = perturbation_ranking_scores(
        distribution, **kwargs, student_t_df=0.5
    )
    assert np.all(np.isfinite(fractional_scores))
    selected = draw_token(distribution, **kwargs, student_t_df=1.0)
    assert selected in distribution.ids

    target = draw_token(distribution, **kwargs, student_t_df=1.0)
    found, checked = find_seed_for_token(
        distribution,
        target,
        current_seed=12345,
        stream_fingerprint="a" * 64,
        aligned_step=4,
        kernel="student-t-max",
        student_t_df=1.0,
        next_seed=lambda: 19,
    )
    assert found == 19
    assert checked == 1


def test_episode_engine_uses_configured_student_t_df_for_proposals():
    sampling = SamplerConfig(draw_kernel='student-t-max', student_t_df=0.5, top_k=5, min_p=0.0)
    runtime = EpisodeEngine(
        ConformingFakeBackend(), sampling=sampling, initial_token_ids=[7]
    )
    observation = runtime.observe()
    expected = draw_token(
        observation.distribution,
        seed=sampling.seed,
        stream_fingerprint=runtime.stream_fingerprint,
        aligned_step=observation.sampling_boundary,
        kernel="student-t-max",
        perturb_noise_std=sampling.perturb_noise_std,
        student_t_df=0.5,
    )
    assert observation.proposal_token_id == expected


@pytest.mark.invariant
def test_s03_cfg_is_scoped_to_its_configured_prefix():
    sampling = SamplerConfig(
        temperature=0.0,
        top_k=8,
        min_p=0.0,
        cfg_unconditional_prompt=' A',
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


@pytest.mark.invariant
def test_s04_history_penalties_change_policy_order_not_raw_rank():
    logits = np.asarray([2.0, 1.0, 0.0, -1.0])
    config = SamplerConfig(
        temperature=1.0,
        top_k=2,
        min_p=0.0,
        repeat_penalty=2.0,
        repeat_last_n=-1,
        presence_penalty=0.5,
        frequency_penalty=0.25,
    )
    policy_calculations = PolicyCalculations(logits, config, [0, 0, 1])
    assert [policy_calculations.policy_rank(token) for token in [0, 1, 2]] == [1, 3, 2]
    np.testing.assert_allclose(policy_calculations.adjusted - policy_calculations.logits,
                               [-2.0, -1.25, 0.0, 0.0])
    assert policy_calculations.raw_rank(1) == 2
    assert policy_calculations.distribution.ids.tolist() == [0, 2]
    with pytest.raises(ValueError, match="exact prefix token ids"):
        PolicyCalculations(np.asarray([1.0, 0.0]), SamplerConfig(repeat_penalty=1.1), None)


@pytest.mark.current_workflow
def test_s04b_policy_probabilities_are_on_demand_not_dense():
    """Dense vocabulary soft-max is not stored; on-demand probs match logsumexp."""

    logits = np.asarray([2.0, 1.0, 0.0, -1.0])
    plain = PolicyCalculations(
        logits, SamplerConfig(temperature=1.0, top_k=4, min_p=0.0), []
    )
    assert "policy_probabilities" not in plain.__dict__

    expected_raw = np.exp(logits - float(np.max(logits))) / float(
        np.sum(np.exp(logits - float(np.max(logits))))
    )
    np.testing.assert_allclose(plain.raw_probabilities([0, 2, 3]), expected_raw[[0, 2, 3]])
    np.testing.assert_allclose(plain.policy_probabilities_at([1, 3]), expected_raw[[1, 3]])
    np.testing.assert_allclose(plain.raw_nll(0), plain.log_z - float(logits[0]))

    # Sparse draw distribution still carries a soft-max over filtered candidates.
    assert len(plain.distribution.ids) >= 1
    assert plain.distribution.softmax.shape == plain.distribution.ids.shape
    np.testing.assert_allclose(plain.distribution.softmax.sum(), 1.0, atol=1e-12)

    penalized = PolicyCalculations(
        logits,
        SamplerConfig(
            temperature=1.0,
            top_k=4,
            min_p=0.0,
            repeat_penalty=2.0,
            repeat_last_n=-1,
            presence_penalty=0.5,
            frequency_penalty=0.25,
        ),
        [0, 0, 1],
    )
    assert "policy_probabilities" not in penalized.__dict__
    adjusted = np.asarray(penalized.adjusted, dtype=np.float64)
    policy_max = float(np.max(adjusted))
    expected_policy = np.exp(adjusted - policy_max) / float(np.sum(np.exp(adjusted - policy_max)))
    np.testing.assert_allclose(
        penalized.policy_probabilities_at([0, 1, 2]), expected_policy[[0, 1, 2]]
    )
    np.testing.assert_allclose(
        penalized.raw_probabilities([0, 1]), expected_raw[[0, 1]]
    )

    runtime = EpisodeEngine(
        ConformingFakeBackend(),
        sampling=SamplerConfig(temperature=0.8, top_k=8, min_p=0.0),
        initial_token_ids=[7],
    )
    observation = runtime.observe()
    assert "policy_probabilities" not in observation.policy_calculations.__dict__
    assert 0 <= observation.proposal_token_id < len(observation.logits)
    np.testing.assert_allclose(
        observation.proposal_raw_probability,
        float(observation.policy_calculations.raw_probabilities([observation.proposal_token_id])[0]),
    )


@pytest.mark.current_workflow
def test_s04c_logsumexp_deferred_until_nll_or_probabilities():
    """Draw / ranks / top-ids work without dense V exp+sum; NLL triggers once."""

    logits = np.asarray([2.0, 1.0, 0.0, -1.0])
    policy_calculations = PolicyCalculations(
        logits, SamplerConfig(temperature=1.0, top_k=2, min_p=0.0), []
    )
    assert policy_calculations._raw_logsumexp_ready is False
    assert policy_calculations._log_z is None
    assert policy_calculations.top_raw_ids(2) == [0, 1]
    assert policy_calculations.raw_rank(1) == 2
    assert policy_calculations._raw_logsumexp_ready is False
    assert len(policy_calculations.distribution.ids) >= 1
    np.testing.assert_allclose(policy_calculations.distribution.softmax.sum(), 1.0, atol=1e-12)
    assert policy_calculations._raw_logsumexp_ready is False

    # Cheap top-logit peek must not force exp+sum.
    assert policy_calculations.maximum == 2.0
    assert policy_calculations._raw_logsumexp_ready is False

    expected_log_z = 2.0 + float(np.log(float(np.sum(np.exp(logits - 2.0)))))
    nll = policy_calculations.raw_nll(0)
    assert policy_calculations._raw_logsumexp_ready is True
    np.testing.assert_allclose(policy_calculations.log_z, expected_log_z)
    np.testing.assert_allclose(nll, expected_log_z - 2.0)
    # Second access reuses the same scalars.
    assert policy_calculations.raw_nll(0) == nll

    runtime = EpisodeEngine(
        ConformingFakeBackend(),
        sampling=SamplerConfig(temperature=0.0, top_k=8, min_p=0.0),
        initial_token_ids=[7],
    )
    observation = runtime.observe()
    assert observation.policy_calculations._raw_logsumexp_ready is False
    proposal = observation.proposal_token_id
    assert observation.proposal_raw_rank >= 1
    assert observation.policy_calculations._raw_logsumexp_ready is False
    # A commit records the decision without calculating report statistics.
    outcome = runtime.apply(Accept())
    assert observation.policy_calculations._raw_logsumexp_ready is False
    assert outcome.evidence and outcome.evidence[0].raw_model_nll is None
    assert outcome.evidence[0].raw_rank is None



def test_s04d_candidates_skip_probabilities_until_requested():
    from trajectory_editor.candidate_columns import CandidateColumns, OVERLAYS
    runtime = EpisodeEngine(ConformingFakeBackend(), sampling=SamplerConfig(), initial_token_ids=[7])
    observation = runtime.observe()
    assert set(OVERLAYS) == {'logit', 'diff', 'noise', 'probability'}
    assert dict(CandidateColumns().columns).keys() == {'token-id'}
    for overlays in (frozenset(), frozenset({'logit', 'diff', 'noise'})):
        rows = runtime.candidates(observation, count=3, metrics=CandidateColumns(overlays=overlays).plan.metrics)
        assert all(row.eligible_softmax is None for row in rows)
        assert not observation.policy_calculations._raw_logsumexp_ready
        assert 'softmax' not in observation.policy_calculations.distribution.__dict__
    rows = runtime.candidates(observation, count=3, metrics=CandidateColumns(overlays=frozenset({'probability'})).plan.metrics)
    assert all(row.raw_probability is not None for row in rows)
    assert all(row.eligible_softmax is not None for row in rows)
    assert observation.policy_calculations._raw_logsumexp_ready



def test_s04e_direct_overlays_persist_and_clear_in_view_preferences():
    from trajectory_editor.episode_ui import InteractivePolicy
    from tests.fakes import ScriptedIO
    runtime = EpisodeEngine(ConformingFakeBackend(), initial_token_ids=[7], sampling=SamplerConfig())
    policy = InteractivePolicy(io=ScriptedIO(['l', 'L', '~', '1']), menu_size=2)
    policy.choose(runtime, runtime.observe())
    assert policy.view_preferences.overlays == frozenset({'logit', 'diff', 'noise'})
    policy.io = ScriptedIO(['1'])
    policy.choose(runtime, runtime.observe())
    assert policy.view_preferences.overlays == frozenset({'logit', 'diff', 'noise'})
    policy.io = ScriptedIO(['%', 'C', '1'])
    policy.choose(runtime, runtime.observe())
    assert policy.view_preferences.overlays == frozenset()










@pytest.mark.invariant
def test_s05_group_member_biases_only_its_final_token_after_exact_prefix():
    member = BiasMember(
        text="the steamship",
        routes=(BiasRoute((10, 11, 12), ("the steamship",)),),
    )
    config = SamplerConfig(bias_groups=(BiasGroup("ships", (member,), 0.5),))
    assert config.active_biases([]) == {}
    assert config.active_biases([10]) == {}
    assert config.active_biases([10, 11]) == {12: 0.5}
    assert config.active_biases([7, 10, 11]) == {12: 0.5}
    assert config.active_biases([10, 9, 11]) == {}

    single = BiasMember("ship", (BiasRoute((12,), ("ship",)),))
    single_config = SamplerConfig(
        bias_groups=(BiasGroup("ships", (single,), 0.5),)
    )
    assert single_config.active_biases([]) == {12: 0.5}


@pytest.mark.invariant
def test_s06_surface_variants_are_bounded_and_bias_sources_are_explainable():
    assert member_surfaces("my favorite couch") == (
        "my favorite couch",
        " my favorite couch",
        "My favorite couch",
        " My favorite couch",
        "MY FAVORITE COUCH",
        " MY FAVORITE COUCH",
    )
    assert member_surfaces("my favorite couch", literal=True) == (
        "my favorite couch",
    )

    member = BiasMember(
        "steamship", (BiasRoute((1, 2, 3), ("steamship",)),)
    )
    overlapping_member = BiasMember(
        "steamship suffix", (BiasRoute((2, 3), ("steamship suffix",)),)
    )
    config = SamplerConfig(
        bias_groups=(
            BiasGroup("nautical", (member, overlapping_member), 0.5),
            BiasGroup("ships", (member,), 0.25),
        ),
        token_biases=(BiasToken(3, 0.125),),
    )
    assert config.active_biases([1, 2]) == {3: 0.875}
    sources = config.bias_contributions([1, 2], include_inactive=True)
    assert sum(item.amount for item in sources if item.active) == pytest.approx(0.875)
    assert {item.source for item in sources if item.active} == {
        "group 'nautical'",
        "group 'ships'",
        "token #3",
    }
    inactive = config.bias_contributions([1], include_inactive=True)
    assert sum(item.amount for item in inactive if item.active) == pytest.approx(0.125)
    assert sum(item.amount for item in inactive if not item.active) == pytest.approx(0.75)

    backend = ConformingFakeBackend()
    group_view = format_group_report(config, "nautical", [1, 2], backend)
    assert "token total now: +0.875" in group_view
    assert "group 'ships'" in group_view
    assert "'steamship suffix'" in group_view
    token_view = format_token_report(config, 3, [1, 2], backend)
    assert "active total: +0.875" in token_view
    assert "token #3" in token_view
    dormant_view = format_token_report(config, 3, [1], backend)
    assert "active total: +0.125" in dormant_view
    assert "inactive until its prefix matches" in dormant_view
    assert SamplerConfig.from_record(config.to_dict()) == config


@pytest.mark.parametrize(
    "action",
    [
        Accept(),
        SelectRawRank(3),
        Write(" hello", mode="exact"),
        Phrase("hello", mode="exact", force=True),
        Hold(2, boundary="sentence"),
        EndGeneration(),
    ],
)
@pytest.mark.invariant
def test_s07_core_actions_and_replay_expectations_round_trip(action):
    assert action_from_dict(action.to_dict()) == action
    expectation = ReplayExpectation((1, 2), terminal_token_id=0, stop_reason="eog")
    assert ReplayExpectation.from_mapping({
        "token_ids": [1, 2],
        "terminal_token_id": 0,
        "stop_reason": "eog",
    }) == expectation


@pytest.mark.invariant
def test_replay_plan_rejects_finish_as_unrecognized_action():
    with pytest.raises(
        EditorError,
        match="teacher plan step 0: invalid action: unsupported policy action kind 'finish'",
    ):
        load_teacher_plan([{"step": 0, "action": {"kind": "finish"}}])
    with pytest.raises(EditorError):
        SelectRawRank(0)


@pytest.mark.current_workflow
def test_s08_engine_uses_core_sampler_without_research_fields():
    assert "token_preference_vector" not in SamplerConfig.__dataclass_fields__
    assert "reference_prior_routes" not in SamplerConfig.__dataclass_fields__
    runtime = EpisodeEngine(
        ConformingFakeBackend(),
        sampling=SamplerConfig(temperature=0.0, top_k=8, min_p=0.0),
        initial_token_ids=[7],
    )
    runtime.apply(Accept())
    assert runtime.boundary == 1

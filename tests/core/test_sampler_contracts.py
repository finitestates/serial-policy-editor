from __future__ import annotations

import math

import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend
from trajectory_editor.bias_rules import BiasMatcher, BiasRule
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
from trajectory_editor.teacher_plan import load_teacher_plan


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


def test_s04b_observation_probabilities_are_on_demand_not_dense():
    """Dense vocabulary soft-max is not stored; on-demand probs match logsumexp."""

    logits = np.asarray([2.0, 1.0, 0.0, -1.0])
    plain = ObservationStatistics(
        logits, SamplerConfig(temperature=1.0, top_k=4, top_p=1.0, min_p=0.0), []
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
    assert plain.distribution.probabilities.shape == plain.distribution.ids.shape
    np.testing.assert_allclose(plain.distribution.probabilities.sum(), 1.0, atol=1e-12)

    penalized = ObservationStatistics(
        logits,
        SamplerConfig(
            temperature=1.0,
            top_k=4,
            top_p=1.0,
            min_p=0.0,
            repeat_penalty=2.0,
            repeat_last_n=-1,
            presence_penalty=0.5,
            frequency_penalty=0.25,
        ),
        [0, 0, 1],
    )
    assert "policy_probabilities" not in penalized.__dict__
    adjusted = np.asarray(penalized.policy_logits, dtype=np.float64)
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
        sampling=SamplerConfig(temperature=0.8, top_k=8, top_p=1.0, min_p=0.0),
        initial_token_ids=[7],
    )
    observation = runtime.observe()
    assert "policy_probabilities" not in observation.statistics.__dict__
    assert 0 <= observation.proposal_token_id < len(observation.logits)
    np.testing.assert_allclose(
        observation.proposal_raw_probability,
        float(observation.statistics.raw_probabilities([observation.proposal_token_id])[0]),
    )


def test_s04c_logsumexp_deferred_until_nll_or_probabilities():
    """Draw / ranks / top-ids work without dense V exp+sum; NLL triggers once."""

    logits = np.asarray([2.0, 1.0, 0.0, -1.0])
    stats = ObservationStatistics(
        logits, SamplerConfig(temperature=1.0, top_k=2, top_p=1.0, min_p=0.0), []
    )
    assert stats._raw_logsumexp_ready is False
    assert stats._log_z is None
    assert stats.top_raw_ids(2) == [0, 1]
    assert stats.raw_rank(1) == 2
    assert stats._raw_logsumexp_ready is False
    assert len(stats.distribution.ids) >= 1
    np.testing.assert_allclose(stats.distribution.probabilities.sum(), 1.0, atol=1e-12)
    assert stats._raw_logsumexp_ready is False

    # Cheap top-logit peek must not force exp+sum.
    assert stats.maximum == 2.0
    assert stats._raw_logsumexp_ready is False

    expected_log_z = 2.0 + float(np.log(float(np.sum(np.exp(logits - 2.0)))))
    nll = stats.raw_nll(0)
    assert stats._raw_logsumexp_ready is True
    np.testing.assert_allclose(stats.log_z, expected_log_z)
    np.testing.assert_allclose(nll, expected_log_z - 2.0)
    # Second access reuses the same scalars.
    assert stats.raw_nll(0) == nll

    runtime = EpisodeEngine(
        ConformingFakeBackend(),
        sampling=SamplerConfig(temperature=0.0, top_k=8, top_p=1.0, min_p=0.0),
        initial_token_ids=[7],
    )
    observation = runtime.observe()
    assert observation.statistics._raw_logsumexp_ready is False
    proposal = observation.proposal_token_id
    assert observation.proposal_raw_rank >= 1
    assert observation.statistics._raw_logsumexp_ready is False
    # A commit records the decision without calculating report statistics.
    outcome = runtime.apply(Accept())
    assert observation.statistics._raw_logsumexp_ready is False
    assert outcome.evidence and outcome.evidence[0].raw_model_nll is None
    assert outcome.evidence[0].raw_rank is None



def test_s04d_candidates_skip_probabilities_until_requested():
    """Menu/search rows can ship ranks+logits without waking dense logsumexp."""
    from trajectory_editor.candidate_columns import (
        COLUMN_FOCUS_CYCLE,
        CandidateColumns,
        OVERLAYS,
        next_column_focus,
        overlays_from_preferences,
    )

    def needs_raw(**options):
        return CandidateColumns(**options).plan.needs("raw_probability")

    # Default identity-only: no soft-max % wake.
    assert needs_raw(logit_view="none") is False
    assert needs_raw(logit_view="raw") is False
    assert needs_raw(logit_view="gap") is False
    assert needs_raw(logit_view="both") is False
    assert needs_raw(policy=True, logit_view="none") is False
    assert needs_raw(show_model_probabilities=True) is True
    assert needs_raw(policy=True, show_model_probabilities=True) is True
    # Narrow width that drops raw-p must not claim soft-max is needed.
    assert needs_raw(
        policy=True, show_model_probabilities=True, width=40
    ) is False

    default_labels = dict(CandidateColumns().columns)
    assert "raw-p" not in default_labels
    assert "decode-p" not in default_labels
    assert "token-id" in default_labels
    assert "raw-p" not in dict(CandidateColumns(logit_view="raw").columns)
    assert "raw-p" in dict(
        CandidateColumns(show_model_probabilities=True).columns
    )
    assert "decode-p" in dict(
        CandidateColumns(show_model_probabilities=True).columns
    )
    assert overlays_from_preferences(logit_view="none") == frozenset()
    assert "pct" in overlays_from_preferences(show_model_probabilities=True)
    assert "raw_probability" in OVERLAYS["pct"].metrics
    assert OVERLAYS["margin_neighbor"].wired is True
    assert "raw_probability" not in OVERLAYS["margin_neighbor"].metrics
    assert OVERLAYS["z"].wired is True
    assert "raw_probability" not in OVERLAYS["z"].metrics

    # column_focus: single overlay; pct wakes soft-max, decode_pct does not.
    assert needs_raw(column_focus="pct") is True
    assert needs_raw(column_focus="decode_pct") is False
    assert needs_raw(column_focus="logit") is False
    assert needs_raw(column_focus="margin_neighbor") is False
    assert needs_raw(column_focus="z") is False
    assert "raw-p" in dict(CandidateColumns(column_focus="pct").columns)
    assert "decode-p" in dict(CandidateColumns(column_focus="decode_pct").columns)
    assert "raw-p" not in dict(CandidateColumns(column_focus="decode_pct").columns)
    assert "model-logit" in dict(CandidateColumns(column_focus="logit").columns)
    assert "margin" in dict(CandidateColumns(column_focus="margin_neighbor").columns)
    assert "z" in dict(CandidateColumns(column_focus="z").columns)
    # Focus and shortcuts combine.
    assert overlays_from_preferences(
        logit_view="both", show_model_probabilities=True, column_focus="logit"
    ) == frozenset({"logit", "gap_k1", "pct", "decode_pct"})
    assert next_column_focus(None) == "logit"
    assert next_column_focus("logit") == "gap_k1"
    assert next_column_focus("gap_k1") == "margin_neighbor"
    assert next_column_focus("margin_neighbor") == "z"
    assert next_column_focus("z") == "pct"
    assert next_column_focus("decode_pct") == "logit"
    assert COLUMN_FOCUS_CYCLE == (
        "logit", "gap_k1", "margin_neighbor", "z", "pct", "decode_pct"
    )

    runtime = EpisodeEngine(
        ConformingFakeBackend(),
        sampling=SamplerConfig(temperature=0.0, top_k=8, top_p=1.0, min_p=0.0),
        initial_token_ids=[7],
    )
    observation = runtime.observe()
    assert observation.statistics._raw_logsumexp_ready is False

    lean = runtime.candidates(observation, count=3)
    assert observation.statistics._raw_logsumexp_ready is False
    assert len(lean) == 3
    assert all(row.raw_probability is None for row in lean)
    assert all(row.policy_probability is None for row in lean)
    assert all(row.raw_logit is None for row in lean)
    assert lean[0].rank == 1

    filled = runtime.candidates(
        observation, count=3,
        view=CandidateColumns(show_model_probabilities=True).plan,
    )
    assert observation.statistics._raw_logsumexp_ready is True
    assert all(row.raw_probability is not None and row.raw_probability > 0 for row in filled)
    assert [row.token_id for row in filled] == [row.token_id for row in lean]

    # Logit-centric interactive choose must not wake soft-max before commit.
    from tests.fakes import ScriptedIO
    from trajectory_editor.episode_ui import InteractivePolicy

    episode = EpisodeEngine(
        ConformingFakeBackend(),
        initial_text="P",
        initial_token_ids=[7],
        sampling=SamplerConfig(temperature=0.0),
    )
    observed = episode.observe()
    assert observed.statistics._raw_logsumexp_ready is False
    policy = InteractivePolicy(io=ScriptedIO(["m 4", "1"]), menu_size=2, logit_view="raw")
    action = policy.choose(episode, observed)
    assert action.kind == "select-raw-rank"
    assert observed.statistics._raw_logsumexp_ready is False


def test_s04e_column_focus_persists_in_view_preferences():
    """c cycles / C clears column_focus on PolicyViewPreferences across chooses."""
    from tests.fakes import ScriptedIO
    from trajectory_editor.episode_ui import InteractivePolicy
    from trajectory_editor.teacher_commands import CommandKind, parse_command

    def parse(raw: str):
        return parse_command(
            raw, menu_size=12, default_hold_tokens=24, vocabulary_size=1000
        )

    assert parse("c").kind == CommandKind.COLUMN_FOCUS
    assert parse("C").kind == CommandKind.COLUMN_FOCUS
    assert parse("C").invoked_as == "C"
    assert parse("context").kind == CommandKind.CONTEXT
    assert parse("c 900").kind == CommandKind.CONTEXT
    assert parse("c all").kind == CommandKind.CONTEXT

    episode = EpisodeEngine(
        ConformingFakeBackend(),
        initial_text="P",
        initial_token_ids=[7],
        sampling=SamplerConfig(temperature=0.0),
    )
    io = ScriptedIO(["c", "c", "1"])
    policy = InteractivePolicy(io=io, menu_size=2)
    assert policy.view_preferences.column_focus is None
    action = policy.choose(episode, episode.observe())
    # After two c presses before commit: logit → gap_k1.
    assert policy.view_preferences.column_focus == "gap_k1"
    episode.apply(action)

    # Focus survives the next observe/choose without re-pressing c.
    io_persist = ScriptedIO(["1"])
    policy.io = io_persist
    policy.choose(episode, episode.observe())
    assert policy.view_preferences.column_focus == "gap_k1"

    # C clears to identity and resets l / % prefs.
    episode2 = EpisodeEngine(
        ConformingFakeBackend(),
        initial_text="P",
        initial_token_ids=[7],
        sampling=SamplerConfig(temperature=0.0),
    )
    io_clear = ScriptedIO(["c", "c", "c", "%", "C", "1"])
    policy2 = InteractivePolicy(io=io_clear, menu_size=2)
    policy2.choose(episode2, episode2.observe())
    assert policy2.view_preferences.column_focus is None
    assert policy2.view_preferences.logit_view == "none"
    assert policy2.view_preferences.show_model_probabilities is False


def test_s04f_neighbor_margin_matches_consecutive_logit_gaps():
    """Consecutive margin is logit[i]-logit[i+1] on the ordered menu; last is None."""
    from trajectory_editor.candidate_columns import CandidateColumns

    runtime = EpisodeEngine(
        ConformingFakeBackend(),
        sampling=SamplerConfig(temperature=0.0, top_k=8, top_p=1.0, min_p=0.0),
        initial_token_ids=[7],
    )
    observation = runtime.observe()
    assert observation.statistics._raw_logsumexp_ready is False

    lean = runtime.candidates(observation, count=4)
    assert all(row.neighbor_margin is None for row in lean)

    rows = runtime.candidates(
        observation, count=4, view=CandidateColumns(column_focus="margin_neighbor").plan
    )
    assert observation.statistics._raw_logsumexp_ready is False
    assert len(rows) == 4
    for i in range(len(rows) - 1):
        expected = float(observation.logits[rows[i].token_id] - observation.logits[rows[i + 1].token_id])
        assert rows[i].neighbor_margin == pytest.approx(expected)
        assert rows[i].neighbor_margin >= 0.0  # raw-rank order: better ≥ next-worse
    assert rows[-1].neighbor_margin is None

    rendered = CandidateColumns(column_focus="margin_neighbor").values(rows[0])
    assert f"{rows[0].neighbor_margin:+.3f}" in rendered
    assert CandidateColumns(column_focus="margin_neighbor").values(rows[-1]).count("--") >= 1




def test_s04g_logit_z_score_uses_full_vocab_mean_std_without_softmax_wake():
    """z = (logit - mean) / std over full-vocab raw logits (population ddof=0)."""
    from trajectory_editor.candidate_columns import CandidateColumns

    logits = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    stats = ObservationStatistics(
        logits, SamplerConfig(temperature=1.0, top_k=4, top_p=1.0, min_p=0.0), []
    )
    assert stats._raw_logsumexp_ready is False
    expected_mean = float(np.mean(logits))
    expected_std = float(np.std(logits, ddof=0))
    assert expected_std > 0.0
    pair = stats._ensure_logit_mean_std()
    assert pair is not None
    mean, std = pair
    assert mean == pytest.approx(expected_mean)
    assert std == pytest.approx(expected_std)
    assert stats._raw_logsumexp_ready is False

    for token_id, logit in enumerate(logits):
        expected_z = (float(logit) - expected_mean) / expected_std
        assert stats.logit_z(token_id) == pytest.approx(expected_z)
    zs = stats.logit_z_scores([0, 3, 1])
    assert zs[0] == pytest.approx((1.0 - expected_mean) / expected_std)
    assert zs[1] == pytest.approx((4.0 - expected_mean) / expected_std)
    assert zs[2] == pytest.approx((2.0 - expected_mean) / expected_std)
    assert stats._raw_logsumexp_ready is False

    # Degenerate vocab (std ~ 0) → undefined z.
    flat = ObservationStatistics(
        np.asarray([5.0, 5.0, 5.0]), SamplerConfig(temperature=1.0, top_k=3, top_p=1.0, min_p=0.0), []
    )
    assert flat._ensure_logit_mean_std() is None
    assert flat.logit_z(0) is None
    assert flat.logit_z_scores([0, 1]) == [None, None]
    assert flat._raw_logsumexp_ready is False

    runtime = EpisodeEngine(
        ConformingFakeBackend(),
        sampling=SamplerConfig(temperature=0.0, top_k=8, top_p=1.0, min_p=0.0),
        initial_token_ids=[7],
    )
    observation = runtime.observe()
    assert observation.statistics._raw_logsumexp_ready is False

    lean = runtime.candidates(observation, count=4)
    assert all(row.logit_z is None for row in lean)
    assert observation.statistics._raw_logsumexp_ready is False
    assert observation.statistics._logit_mean_std_ready is False

    rows = runtime.candidates(
        observation, count=4, view=CandidateColumns(column_focus="z").plan
    )
    assert observation.statistics._raw_logsumexp_ready is False
    assert observation.statistics._logit_mean_std_ready is True
    assert len(rows) == 4
    assert all(row.logit_z is not None for row in rows)
    for row in rows:
        assert row.logit_z == pytest.approx(observation.statistics.logit_z(row.token_id))

    rendered = CandidateColumns(column_focus="z").values(rows[0])
    assert f"{rows[0].logit_z:+.2f}" in rendered

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


def test_replay_plan_rejects_finish_as_unrecognized_action():
    with pytest.raises(
        EditorError,
        match="teacher plan step 0: invalid action: unsupported policy action kind 'finish'",
    ):
        load_teacher_plan([{"step": 0, "action": {"kind": "finish"}}])
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

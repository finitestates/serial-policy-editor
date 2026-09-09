"""Regression checks for boundary-local numeric snapshots."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from tests.sampling_reference import (
    policy_token_evidence, raw_nll, raw_probability, sampling_distribution,
)
from tests.test_episode_runtime import engine
from trajectory_editor.domain import EditorError, SamplingConfig
from trajectory_editor.episode_actions import Accept, Write
from trajectory_editor.sampling import (
    ObservationStatistics, raw_rank,
)


@pytest.mark.parametrize("temperature", [0.0, 0.8, 2.0])
@pytest.mark.parametrize("penalties", [False, True])
def test_statistics_match_existing_math(temperature, penalties):
    values = np.random.default_rng(123).normal(size=1024)
    values[:4] = 2.0
    config = SamplingConfig(
        temperature=temperature, top_k=40, top_p=0.8, min_p=0.1,
        repeat_penalty=1.2 if penalties else 1.0,
        presence_penalty=0.4 if penalties else 0.0,
        frequency_penalty=0.2 if penalties else 0.0,
    )
    history = [0, 0, 3, 6]
    stats = ObservationStatistics(values, config, history)
    expected = sampling_distribution(values, config, history)
    np.testing.assert_array_equal(stats.distribution.ids, expected.ids)
    np.testing.assert_array_equal(stats.distribution.probabilities, expected.probabilities)
    ids = [0, 1, 3, 23, 900]
    rows = policy_token_evidence(values, config, history, ids)
    assert stats.raw_probabilities(ids).tolist() == [raw_probability(values, i) for i in ids]
    for token, row in zip(ids, rows):
        assert stats.raw_nll(token) == raw_nll(values, token)
        assert stats.raw_rank(token) == raw_rank(values, token)
        assert stats.policy_rank(token) == row["policy_rank"]
        assert stats.policy_probabilities[token] == row["policy_probability"]


def test_menu_and_accept_prepare_once():
    runtime = engine()
    runtime.sampling = replace(runtime.sampling, repeat_penalty=1.2)
    with patch(
        "trajectory_editor.sampling._history_penalty_surface",
        wraps=__import__("trajectory_editor.sampling", fromlist=[""])._history_penalty_surface,
    ) as prepare:
        observation = runtime.observe()
        first = runtime.candidates(observation, count=3)
        assert runtime.candidates(observation, count=3) == first
        assert runtime.observe() is observation
        outcome = runtime.apply(Accept())
        assert prepare.call_count == 1
        assert outcome.evidence[0].raw_rank == observation.proposal_raw_rank
        runtime.observe()
        assert prepare.call_count == 2


def test_sampler_replacement_invalidates_even_when_restored():
    runtime = engine()
    observation = runtime.observe()
    original = runtime.sampling
    runtime.sampling = replace(original, seed=42)
    with pytest.raises(EditorError, match="stale"):
        runtime.candidates(observation)
    runtime.sampling = original
    with pytest.raises(EditorError, match="stale"):
        runtime._commit_token(observation, observation.proposal_token_id)


def test_rewind_foreign_prefix_and_coordinate_validation():
    runtime = engine(max_tokens=4)
    observation = runtime.observe()
    with pytest.raises(EditorError, match="stale"):
        engine().candidates(observation)
    runtime.rewind_to(0)
    with pytest.raises(EditorError, match="stale"):
        runtime.candidates(observation)
    observation = runtime.observe()
    runtime.coordinate_offset += 1
    with pytest.raises(EditorError, match="stale"):
        runtime.candidates(observation)
    observation = runtime.observe()
    runtime.stream_fingerprint = "c" * 64
    with pytest.raises(EditorError, match="stale"):
        runtime.candidates(observation)
    observation = runtime.observe()
    runtime.initial_token_ids = (6,)
    with pytest.raises(EditorError, match="stale"):
        runtime.candidates(observation)


def test_snapshot_owns_readonly_arrays():
    runtime = engine()
    values = runtime.backend.last_logits().astype(np.float64)
    with patch.object(runtime.backend, "last_logits", return_value=values):
        observation = runtime.observe()
    expected = observation.logits.copy()
    values[:] = -500
    np.testing.assert_array_equal(observation.logits, expected)
    for array in (observation.logits, observation.distribution.ids,
                  observation.distribution.probabilities):
        with pytest.raises(ValueError):
            array[0] = 0


def test_termination_rejects_current_observation():
    runtime = engine()
    observation = runtime.observe()
    runtime.terminate()
    with pytest.raises(EditorError, match="stale"):
        runtime.candidates(observation)


@pytest.mark.parametrize("config", [
    SamplingConfig(),
    SamplingConfig(repeat_last_n=0, repeat_penalty=1.5,
                   presence_penalty=0.5, frequency_penalty=0.2),
])
def test_inactive_penalties_reuse_raw_statistics(config):
    values = np.array([2.0, 2.0, -1.0, 0.0])
    with patch("trajectory_editor.sampling._history_penalty_surface") as prepare:
        stats = ObservationStatistics(values, config, [0, 0, 2])
        prepare.assert_not_called()
    assert stats.adjusted is stats.logits
    assert stats._policy_ranks is stats._raw_ranks
    assert not stats.adjusted.flags.writeable
    assert not stats.policy_probabilities.flags.writeable
    ids = stats.top_raw_ids(4)
    rows = policy_token_evidence(values, config, [0, 0, 2], ids)
    assert stats.raw_probabilities(ids).tolist() == [raw_probability(values, i) for i in ids]
    for token, row in zip(ids, rows):
        assert stats.policy_rank(token) == row["policy_rank"]
        assert stats.policy_probabilities[token] == row["policy_probability"]


@pytest.mark.parametrize("history", [[-1], [4], [[0]]])
def test_inactive_penalties_still_validate_history(history):
    with pytest.raises(ValueError, match="history token ids"):
        ObservationStatistics(np.zeros(4), SamplingConfig(), history)


def test_history_penalties_toggle_mid_episode():
    runtime = engine(max_tokens=8)
    runtime.apply(Accept())
    off = runtime.observe()
    config = runtime.sampling
    runtime.sampling = replace(config, presence_penalty=20.0, repeat_last_n=-1)
    with pytest.raises(EditorError, match="stale"):
        runtime.candidates(off)
    on = runtime.observe()
    token = runtime.visible_token_ids[0]
    assert on.statistics.adjusted[token] == on.logits[token] - 20.0
    assert on.statistics.policy_rank(token) != on.statistics.raw_rank(token)
    ids = list(range(len(on.logits)))
    rows = policy_token_evidence(on.logits, runtime.sampling, runtime.token_ids, ids)
    for i, row in zip(ids, rows):
        assert on.statistics.policy_rank(i) == row["policy_rank"]
        assert on.statistics.policy_probabilities[i] == row["policy_probability"]
        assert on.statistics.raw_rank(i) == off.statistics.raw_rank(i)
    np.testing.assert_array_equal(on.statistics.raw_probabilities(ids), off.statistics.raw_probabilities(ids))
    expected = sampling_distribution(on.logits, runtime.sampling, runtime.token_ids)
    np.testing.assert_array_equal(on.distribution.ids, expected.ids)
    np.testing.assert_array_equal(on.distribution.probabilities, expected.probabilities)
    runtime.sampling = config
    with pytest.raises(EditorError, match="stale"):
        runtime.candidates(on)
    restored = runtime.observe()
    assert restored.statistics.adjusted is restored.statistics.logits
    np.testing.assert_array_equal(restored.distribution.ids, off.distribution.ids)
    np.testing.assert_array_equal(restored.distribution.probabilities, off.distribution.probabilities)
    np.testing.assert_array_equal(restored.statistics.policy_probabilities, off.statistics.policy_probabilities)

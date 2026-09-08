from __future__ import annotations

import math

import numpy as np
import pytest

from trajectory_editor.domain import MAX_SEED, MIN_SEED, EditorError, SamplingConfig
from trajectory_editor.sampling import ObservationStatistics, raw_rank, top_raw_ids


@pytest.mark.parametrize("seed", [MIN_SEED, MAX_SEED])
def test_seed_has_one_portable_signed_64_bit_contract(seed):
    assert SamplingConfig(seed=seed).seed == seed


@pytest.mark.parametrize("seed", [MIN_SEED - 1, MAX_SEED + 1])
def test_out_of_range_seed_is_rejected(seed):
    with pytest.raises(EditorError):
        SamplingConfig(seed=seed)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_configuration_is_rejected(value):
    with pytest.raises(EditorError):
        SamplingConfig(temperature=value)


def test_nonfinite_logits_are_rejected():
    with pytest.raises(ValueError):
        ObservationStatistics(np.asarray([0.0, math.nan]), SamplingConfig(), [])


def test_equal_logits_use_token_id_ascending_tie_break():
    logits = np.zeros(16, dtype=np.float32)
    assert top_raw_ids(logits, 5) == [0, 1, 2, 3, 4]
    stats = ObservationStatistics(
        logits, SamplingConfig(temperature=1.0, top_k=5, top_p=1.0, min_p=0.0), []
    )
    assert stats.distribution.ids.tolist() == [0, 1, 2, 3, 4]
    assert [raw_rank(logits, token_id) for token_id in range(5)] == [1, 2, 3, 4, 5]


def test_raw_rank_is_absolute_over_the_full_vocabulary():
    logits = np.asarray([1.0, 5.0, 3.0, 3.0, -2.0])
    assert [raw_rank(logits, token_id) for token_id in range(5)] == [4, 1, 2, 3, 5]


def test_history_penalties_reorder_without_changing_raw_rank():
    logits = np.asarray([2.0, 1.0, 0.0, -1.0])
    config = SamplingConfig(
        temperature=1.0, top_k=2, top_p=1.0, min_p=0.0,
        repeat_penalty=2.0, repeat_last_n=-1,
        presence_penalty=0.5, frequency_penalty=0.25,
    )
    stats = ObservationStatistics(logits, config, [0, 0, 1])
    assert [stats.policy_rank(token) for token in [0, 1, 2]] == [1, 3, 2]
    np.testing.assert_allclose(stats.adjusted - stats.logits, [-2.0, -1.25, 0.0, 0.0])
    assert stats.raw_rank(1) == 2
    assert stats.distribution.ids.tolist() == [0, 2]


def test_active_history_policy_fails_without_exact_prefix():
    config = SamplingConfig(repeat_penalty=1.1)
    with pytest.raises(ValueError, match="exact prefix token ids"):
        ObservationStatistics(np.asarray([1.0, 0.0]), config, None)


def test_legacy_sampling_mapping_uses_neutral_penalties():
    config = SamplingConfig.from_mapping(
        {"temperature": 0.5, "top_k": 4, "top_p": 0.9, "min_p": 0.1, "seed": 7}
    )
    assert not config.history_penalties_active
    assert config.repeat_penalty == 1.0
    assert config.repeat_last_n == 64

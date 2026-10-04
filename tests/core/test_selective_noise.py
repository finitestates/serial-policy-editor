from __future__ import annotations

import argparse

import numpy as np
import pytest

from trajectory_editor.core.cli_config import (
    add_core_sampler_arguments,
    sampler_from_args,
    sampler_override,
    sampler_overrides_from_args,
)
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.core.sampling import (
    DRAW_KERNELS,
    SparseDistribution,
    draw_token,
    find_seed_for_token,
    gaussian_ranking_scores,
    gumbel_ranking_scores,
    perturbation_ranking_scores,
)


@pytest.mark.parametrize('kernel', [k for k in DRAW_KERNELS if k != 'categorical'])
def test_selective_noise_preserves_addresses_ties_and_untouched_scores(kernel):
    distribution = SparseDistribution(
        np.array([8, 3, 9]), np.array([0.4, 0.4, 0.2]), np.array([2., 2., 1.]),
    )
    kwargs = dict(seed=19, stream_fingerprint='a' * 64, aligned_step=4)
    if kernel == 'gaussian-max':
        rank = gaussian_ranking_scores
    elif kernel == 'gumbel-max':
        rank = gumbel_ranking_scores
    else:
        rank = perturbation_ranking_scores
        kwargs['kernel'] = kernel
    full = rank(distribution, **kwargs)
    selective = rank(distribution, **kwargs, selective_noise_k=1)
    # The lower token ID wins the pre-noise tie regardless of input order.
    assert selective[1] == full[1]
    np.testing.assert_array_equal(selective[[0, 2]], distribution.scores[[0, 2]])
    np.testing.assert_array_equal(rank(distribution, **kwargs, selective_noise_k=10), full)


def test_untouched_challenger_can_win_but_lower_untouched_token_cannot():
    distribution = SparseDistribution(
        np.array([0, 1, 2]), np.array([0.34, 0.33, 0.33]), np.array([0.01, 0., -1.]),
    )
    kwargs = dict(stream_fingerprint='a' * 64, aligned_step=0,
                  kernel='gaussian-max', selective_noise_k=1)
    winners = {draw_token(distribution, seed=seed, **kwargs) for seed in range(40)}
    assert winners == {0, 1}
    with pytest.raises(EditorError, match='unchanged higher-ranked'):
        find_seed_for_token(distribution, 2, current_seed=0, next_seed=lambda: 1, **kwargs)


def test_cli_explicit_none_and_saved_records():
    parser = argparse.ArgumentParser()
    add_core_sampler_arguments(parser)
    current = SamplerConfig(draw_kernel='gumbel-max', selective_noise_k=5)
    args = parser.parse_args(['--eligible-k', '20', '--selective-noise-k', 'none'])
    config = sampler_from_args(args, current)
    assert config.eligible_k == 20
    assert config.selective_noise_k is None
    assert sampler_overrides_from_args(args) == {'top_k': 20, 'selective_noise_k': None}
    changed = sampler_override(current, 'eligible_k=12 selective_noise_k=3')
    assert changed.eligible_k == 12
    assert changed.selective_noise_k == 3
    assert SamplerConfig.from_record(changed.to_dict()) == changed
    legacy = changed.to_dict()
    del legacy['selective_noise_k']
    assert SamplerConfig.from_record(legacy).selective_noise_k is None
    with pytest.raises(EditorError, match='perturb-and-argmax'):
        SamplerConfig(selective_noise_k=3)


@pytest.mark.parametrize('kernel', ['gumbel-max', 'gaussian-max', 'laplace-max'])
def test_engine_competes_with_untouched_eligible_scores_and_keeps_raw_logits(kernel):
    from tests.fakes import ConformingFakeBackend
    from trajectory_editor.episode_engine import EpisodeEngine

    sampling = SamplerConfig(
        draw_kernel=kernel, selective_noise_k=1, top_k=5, top_p=1., min_p=0.,
    )
    engine = EpisodeEngine(ConformingFakeBackend(), sampling=sampling, initial_token_ids=[7])
    observation = engine.observe()
    assert len(observation.distribution.ids) == 5
    assert len(observation.policy_calculations.logits) > 5
    assert observation.proposal_token_id == draw_token(
        observation.distribution, seed=sampling.seed,
        stream_fingerprint=engine.stream_fingerprint,
        aligned_step=observation.sampling_boundary, kernel=kernel, selective_noise_k=1,
    )
    if kernel == 'gumbel-max':
        ids = observation.distribution.ids
        scores = observation.distribution.scores
        leader = np.lexsort((ids, -scores))[0]
        unchanged = np.arange(len(ids)) != leader
        np.testing.assert_array_equal(observation.gumbel_scores[unchanged], scores[unchanged])

"""Independent reference formulas checked against the production primitives."""

from dataclasses import replace

import numpy as np
import pytest

from test_reference_kernel import ScriptedBackend, start
from trajectory_editor.core.actions import Accept as ProductionAccept
from trajectory_editor.core.actions import Hold as ProductionHold
from trajectory_editor.core.actions import Reroll as ProductionReroll
from trajectory_editor.core.actions import action_from_dict
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.core.sampling import (
    EligibleScores, draw_token,
    position_uniform_token as production_uniform_token,
)
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_hash import token_prefix_sha256 as production_sha256
from reference_kernel import (
    Accept, Branch, Distribution, Hold, Policy, Reroll, State, World, apply, draw, observe,
    position_uniform, position_uniform_token, rewind, token_prefix_sha256,
)


SCRIPTED_TOKENIZER_ID = "reference-kernel-scripted-tokenizer-v1"


class ProductionScriptedBackend(ScriptedBackend):
    """The same pure logits function behind EpisodeEngine's incremental API."""

    def __init__(self):
        self.tokens = []

    def vocabulary_size(self):
        return 10

    def reset(self, prefix_token_ids):
        self.tokens = list(prefix_token_ids)

    def eval(self, token_ids):
        self.tokens.extend(token_ids)

    def last_logits(self):
        return np.asarray(self.logits(self.tokens), dtype=np.float64)

    def tokenize(self, text, *, add_bos=False, special=False):
        raise AssertionError("parity run supplies the initial token ledger")

    def render(self, token_ids, *, special=False):
        return ",".join(str(token_id) for token_id in token_ids)

    def token_text(self, token_id):
        return str(token_id)

    def eog_token_ids(self):
        return ()

    def tokenizer_id(self):
        return SCRIPTED_TOKENIZER_ID

    def provenance(self, *, include_model_sha256=True):
        return {"backend": "scripted", "vocabulary_size": 10}


@pytest.mark.parametrize("prefix", [(), (0,), (1, 2, 3), ((1 << 63) - 1,)])
def test_token_prefix_hash_matches_production(prefix):
    assert token_prefix_sha256(prefix) == production_sha256(prefix)


@pytest.mark.parametrize("seed", [-(1 << 63), -5, 0, 17, (1 << 63) - 1])
def test_random_access_uniforms_match_production(seed):
    world = World.for_prefix(seed, (1, 2, 3))
    for boundary in (0, 1, 29, 500, 1000000):
        args = (seed, world.stream_fingerprint, boundary)
        for token_id in (0, 4, 999):
            assert position_uniform_token(world, boundary, token_id) == production_uniform_token(*args, token_id)


@pytest.mark.parametrize("kernel", ["argmax", "gumbel-max"])
def test_fixed_distribution_draws_match_production(kernel):
    reference = Distribution((4, 1, 7), (0.1, 0.9, 0.4))
    production = EligibleScores(
        np.asarray(reference.ids, dtype=np.int64),
        np.asarray(reference.scores, dtype=np.float64),
    )
    for seed in (-3, 17, 12345):
        world = World.for_prefix(seed, (1, 2, 3))
        for boundary in (0, 1, 2, 500, 1000000):
            expected = draw_token(production, seed=seed, stream_fingerprint=world.stream_fingerprint,
                                  aligned_step=boundary, kernel=kernel)
            assert draw(reference, world, boundary, kernel) == expected


@pytest.mark.parametrize("kernel", ["argmax", "gumbel-max"])
@pytest.mark.parametrize("seed", [12345, 67890])
def test_production_sampling_boundaries_observations_holds_and_rewind(seed, kernel):
    reference_backend = ScriptedBackend()
    reference = start(seed, policy=Policy(top_k=6, draw_kernel=kernel))
    production = EpisodeEngine(
        ProductionScriptedBackend(), initial_token_ids=[1, 2, 3],
        sampling=SamplerConfig(seed=seed, temperature=1.0, top_k=6, min_p=0.0, draw_kernel=kernel),
    )
    assert production.stream_fingerprint == reference.world.stream_fingerprint

    for boundary in range(5):
        expected, actual = observe(reference_backend, reference), production.observe()
        assert expected.boundary == actual.boundary == boundary
        assert expected.sampling_boundary == actual.sampling_boundary == boundary
        assert expected.prefix_token_ids == tuple(actual.prefix_token_ids)
        assert expected.proposal_token_id == actual.proposal_token_id
        assert expected.distribution.ids == tuple(actual.distribution.ids)
        assert expected.distribution.probabilities == pytest.approx(actual.distribution.softmax)
        reference, _ = apply(reference_backend, reference, Accept())
        production.apply(ProductionAccept())

    reference, held = apply(reference_backend, reference, Hold(7))
    production_held = production.apply(ProductionHold(7))
    assert held.visible_token_ids == production_held.visible_token_ids
    assert tuple(production.token_ids) == reference.state.token_ids
    assert held.stop_reason == production_held.stop_reason == "requested-length"

    reference = rewind(reference, 2)
    production.rewind_to(2)
    assert observe(reference_backend, reference).sampling_boundary == production.observe().sampling_boundary == 2
    assert observe(reference_backend, reference).proposal_token_id == production.observe().proposal_token_id
    reference, held_again = apply(reference_backend, reference, Hold(10))
    production_again = production.apply(ProductionHold(10))
    assert held_again.visible_token_ids == production_again.visible_token_ids
    assert tuple(production.token_ids) == reference.state.token_ids


def test_top_k_membership_preserves_plain_argmax_in_production():
    class Flat(ProductionScriptedBackend):
        def logits(self, prefix):
            return (0.0, 0.0, 0.0)

        def vocabulary_size(self):
            return 3

    backend = Flat()
    reference_backend = Flat()
    # Pin the same world in both implementations. This test isolates policy
    # remapping from root fingerprint construction, which has its own parity
    # assertion above.
    stream_fingerprint = "a" * 64
    branch = Branch(
        State((1, 2)),
        Policy(top_k=2),
        World(5, stream_fingerprint),
    )
    production = EpisodeEngine(
        backend,
        initial_token_ids=[1, 2],
        stream_fingerprint=stream_fingerprint,
        sampling=SamplerConfig(seed=5, top_k=2, min_p=0.0),
    )
    a = observe(reference_backend, branch).proposal_token_id
    assert a == production.observe().proposal_token_id
    branch = replace(branch, policy=Policy(top_k=3))
    production.sampling = replace(production.sampling, top_k=3)
    b = observe(reference_backend, branch).proposal_token_id
    assert b == production.observe().proposal_token_id
    assert a == b == 0
    branch = replace(branch, policy=Policy(top_k=2))
    production.sampling = replace(production.sampling, top_k=2)
    assert observe(reference_backend, branch).proposal_token_id == production.observe().proposal_token_id == a


@pytest.mark.parametrize("kernel", ["argmax", "gumbel-max"])
@pytest.mark.parametrize("seed,new_seed", [(12345, 999), (67890, -42)])
def test_reroll_matches_reference_kernel(kernel, seed, new_seed):
    reference_backend = ScriptedBackend()
    reference = start(seed, policy=Policy(top_k=6, draw_kernel=kernel))
    production = EpisodeEngine(
        ProductionScriptedBackend(), initial_token_ids=[1, 2, 3],
        sampling=SamplerConfig(seed=seed, temperature=1.0, top_k=6, min_p=0.0, draw_kernel=kernel),
    )
    # Draw once under the original seed so a stale proposal exists to discard.
    before = observe(reference_backend, reference).proposal_token_id
    assert before == production.observe().proposal_token_id

    reference, kernel_result = apply(reference_backend, reference, Reroll(new_seed))
    outcome = production.apply(ProductionReroll(new_seed))

    assert kernel_result.resolved_token_ids == outcome.resolved_token_ids == ()
    assert kernel_result.visible_token_ids == outcome.visible_token_ids == ()
    assert production.sampling.seed == new_seed
    assert production.boundary == 0
    assert production.sampling.top_k == 6  # policy untouched

    expected = observe(reference_backend, reference)
    actual = production.observe()
    assert expected.proposal_token_id == actual.proposal_token_id
    assert expected.distribution.ids == tuple(actual.distribution.ids)
    assert expected.distribution.probabilities == pytest.approx(actual.distribution.softmax)


def test_reroll_round_trips_through_action_dict():
    action = ProductionReroll(4242)
    assert action.to_dict() == {"kind": "reroll", "seed": 4242}
    assert action_from_dict(action.to_dict()) == action


@pytest.mark.parametrize("seed", ["7", 7.0, None, 1 << 63, -(1 << 63) - 1])
def test_reroll_rejects_bad_seeds(seed):
    with pytest.raises(EditorError):
        ProductionReroll(seed)
    with pytest.raises(EditorError):
        action_from_dict({"kind": "reroll", "seed": seed})


@pytest.mark.parametrize('kernel', ['gumbel-max', 'gumbel-max', 'gaussian-max', 'logistic-max', 'laplace-max', 'uniform-max', 'student-t-max'])
@pytest.mark.parametrize('df', [0.5, 3.0, 7.0])
def test_all_draw_kernels_and_scales(kernel, df):
    from reference_kernel import DRAW_KERNELS
    from trajectory_editor.core.sampling import DRAW_KERNELS as production_kernels
    assert set(DRAW_KERNELS) == set(production_kernels)
    reference = Distribution((4, 1, 7), (0.1, 0.9, 0.4))
    production = EligibleScores(np.array(reference.ids), np.array(reference.scores))
    for scale in (0., .7, 1., 2.):
        policy = Policy(
            draw_kernel=kernel,
            gaussian_noise_std=scale,
            perturb_noise_std=scale,
            student_t_df=df,
            gumbel_noise_scale=scale,
        )
        for seed in (-3, 17, 12345):
            world = World.for_prefix(seed, (1, 2, 3))
            for boundary in (0, 1, 29):
                expected = draw_token(production, seed=seed, stream_fingerprint=world.stream_fingerprint, aligned_step=boundary, kernel=kernel, gaussian_noise_std=scale, perturb_noise_std=scale, student_t_df=df, gumbel_noise_scale=scale)
                assert draw(reference, world, boundary, kernel, policy=policy) == expected


@pytest.mark.parametrize("settings", [
    {}, {'min_p': .4}, {'top_k': 6, 'min_p': .1},
    {'repeat_penalty': 1.2, 'presence_penalty': .2, 'frequency_penalty': .3},
])
def test_filter_and_history_parity(settings):
    from trajectory_editor.core.policy_calculations import PolicyCalculations
    logits = np.array([.2, -.5, 3., 1.7, .4, 1., -.8, .5])
    history = (1, 2, 1, 4)
    config = SamplerConfig(**({'top_k': None, 'min_p': 0.} | settings))
    actual = PolicyCalculations(logits, config, history).distribution
    expected = Policy(**settings).distribution(logits, history)
    assert expected.ids == tuple(actual.ids)
    assert expected.probabilities == pytest.approx(actual.softmax)
    assert expected.scores == pytest.approx(actual.scores)


def test_model_rank_gumbel_matches_production():
    dist = Distribution((4, 1, 7), (0.1, 0.9, 0.4))
    production = EligibleScores(np.array(dist.ids), np.array(dist.scores))
    ranks = {4: 2, 1: 5, 7: 1}
    world = World.for_prefix(17, (1, 2))
    policy = Policy(draw_kernel='gumbel-max', gumbel_noise_address='model-rank')
    for step in range(20):
        assert draw(dist, world, step, 'gumbel-max', policy=policy, model_ranks=ranks) == draw_token(production, seed=17, stream_fingerprint=world.stream_fingerprint, aligned_step=step, kernel='gumbel-max', gumbel_noise_address='model-rank', candidate_model_ranks=np.array([ranks[i] for i in dist.ids]))


@pytest.mark.parametrize('kernel', ['gumbel-max', 'gaussian-max', 'logistic-max', 'laplace-max', 'uniform-max', 'student-t-max'])
@pytest.mark.parametrize('eligible_k,noise_k', [(None, None), (None, 1), (4, 2), (2, 10)])
def test_independent_eligibility_selective_noise_and_winner_parity(kernel, eligible_k, noise_k):
    from trajectory_editor.core.policy_calculations import PolicyCalculations
    logits = np.asarray([2., 2., 1.9, -1., -2.])
    settings = dict(draw_kernel=kernel, top_k=eligible_k, selective_noise_k=noise_k,
                    min_p=.01, temperature=.8)
    reference = Policy(**settings).distribution(logits, ())
    actual = PolicyCalculations(logits, SamplerConfig(**settings), []).distribution
    assert tuple(actual.ids) == reference.ids
    assert actual.scores == pytest.approx(reference.scores)
    for seed in (-7, 0, 17, 999):
        world = World(seed, 'a' * 64)
        assert draw(reference, world, 3, kernel, policy=Policy(**settings)) == draw_token(
            actual, seed=seed, stream_fingerprint=world.stream_fingerprint,
            aligned_step=3, kernel=kernel, selective_noise_k=noise_k)

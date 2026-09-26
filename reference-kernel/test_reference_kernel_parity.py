"""Independent reference formulas checked against the production primitives."""

from dataclasses import replace

import numpy as np
import pytest

from test_reference_kernel import ScriptedBackend, start
from trajectory_editor.core.actions import Accept as ProductionAccept
from trajectory_editor.core.actions import Hold as ProductionHold
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.core.sampling import (
    SparseDistribution, draw_token, position_uniform as production_uniform,
    position_uniform_token as production_uniform_token,
)
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_hash import token_prefix_sha256 as production_sha256
from reference_kernel import (
    Accept, Branch, Distribution, Hold, Policy, State, World, apply, draw, observe,
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
        assert position_uniform(world, boundary) == production_uniform(*args)
        for token_id in (0, 4, 999):
            assert position_uniform_token(world, boundary, token_id) == production_uniform_token(*args, token_id)


@pytest.mark.parametrize("kernel", ["categorical", "gumbel-max"])
def test_fixed_distribution_draws_match_production(kernel):
    reference = Distribution((4, 1, 7), (0.2, 0.5, 0.3), (0.1, 0.9, 0.4))
    production = SparseDistribution(
        np.asarray(reference.ids, dtype=np.int64),
        np.asarray(reference.probabilities, dtype=np.float64),
        np.asarray(reference.scores, dtype=np.float64),
    )
    for seed in (-3, 17, 12345):
        world = World.for_prefix(seed, (1, 2, 3))
        for boundary in (0, 1, 2, 500, 1000000):
            expected = draw_token(production, seed=seed, stream_fingerprint=world.stream_fingerprint,
                                  aligned_step=boundary, kernel=kernel)
            assert draw(reference, world, boundary, kernel) == expected


@pytest.mark.parametrize("kernel", ["categorical", "gumbel-max"])
@pytest.mark.parametrize("seed", [12345, 67890])
def test_production_sampling_boundaries_observations_holds_and_rewind(seed, kernel):
    reference_backend = ScriptedBackend()
    reference = start(seed, policy=Policy(top_k=6, draw_kernel=kernel))
    production = EpisodeEngine(
        ProductionScriptedBackend(), initial_token_ids=[1, 2, 3],
        sampling=SamplerConfig(seed=seed, temperature=1.0, top_k=6, top_p=1.0,
                               min_p=0.0, draw_kernel=kernel),
    )
    assert production.stream_fingerprint == reference.world.stream_fingerprint

    for boundary in range(5):
        expected, actual = observe(reference_backend, reference), production.observe()
        assert expected.boundary == actual.boundary == boundary
        assert expected.sampling_boundary == actual.sampling_boundary == boundary
        assert expected.prefix_token_ids == tuple(actual.prefix_token_ids)
        assert expected.proposal_token_id == actual.proposal_token_id
        assert expected.distribution.ids == tuple(actual.distribution.ids)
        assert expected.distribution.probabilities == pytest.approx(actual.distribution.probabilities)
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


def test_top_k_policy_remaps_same_quantile_in_production():
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
        sampling=SamplerConfig(seed=5, top_k=2, top_p=1.0, min_p=0.0),
    )
    a = observe(reference_backend, branch).proposal_token_id
    assert a == production.observe().proposal_token_id
    branch = replace(branch, policy=Policy(top_k=3))
    production.sampling = replace(production.sampling, top_k=3)
    b = observe(reference_backend, branch).proposal_token_id
    assert b == production.observe().proposal_token_id
    assert a != b
    branch = replace(branch, policy=Policy(top_k=2))
    production.sampling = replace(production.sampling, top_k=2)
    assert observe(reference_backend, branch).proposal_token_id == production.observe().proposal_token_id == a

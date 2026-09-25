"""Selected-token warm-up uses one in-place evaluation and ordinary actions."""

from __future__ import annotations

import pytest

from tests.fakes import ConformingFakeBackend, SpeculativeFakeBackend
from trajectory_editor import EpisodeEngine, SamplerConfig
from trajectory_editor.core.actions import Accept, SelectRawRank


def engine(
    backend: ConformingFakeBackend,
    *,
    max_tokens: int | None = None,
    sampling: SamplerConfig | None = None,
    initial_token_ids: list[int] | None = None,
    guidance_backend: ConformingFakeBackend | None = None,
) -> EpisodeEngine:
    return EpisodeEngine(
        backend,
        initial_token_ids=initial_token_ids or [7],
        sampling=sampling or SamplerConfig(temperature=0.0),
        max_tokens=max_tokens,
        guidance_backend=guidance_backend,
    )


@pytest.mark.invariant
def test_selected_raw_rank_commits_the_prepared_in_place_token():
    backend = SpeculativeFakeBackend()
    episode = engine(backend)
    observation = episode.observe()
    assert observation.proposal_token_id == 1

    assert episode.speculate_accept(observation, raw_rank=2, token_id=2, generation=1)
    assert episode.token_ids == [7]
    assert episode.boundary == 0
    assert backend.tokens == [7, 2]
    assert backend.eval_calls == [(2,)]
    assert episode.has_prepared_accept(observation, 2, 2)

    outcome = episode.apply(SelectRawRank(2))

    assert outcome.resolved_token_ids == (2,)
    assert backend.tokens == [7, 2]
    assert backend.eval_calls == [(2,)]
    assert backend._speculation_prefix is None


@pytest.mark.invariant
def test_repeated_identical_warm_reuses_the_in_place_evaluation():
    backend = SpeculativeFakeBackend()
    episode = engine(backend)
    observation = episode.observe()

    assert episode.speculate_accept(observation, raw_rank=2, token_id=2, generation=1)
    assert episode.speculate_accept(observation, raw_rank=2, token_id=2, generation=2)

    assert backend.eval_calls == [(2,)]
    assert backend.tokens == [7, 2]
    assert episode.has_prepared_accept(observation, 2, 2)
    episode.apply(SelectRawRank(2))
    assert backend.eval_calls == [(2,)]
    assert backend.tokens == [7, 2]


@pytest.mark.invariant
def test_accept_rolls_back_a_different_prepared_token_before_committing_proposal():
    backend = SpeculativeFakeBackend()
    episode = engine(backend)
    observation = episode.observe()

    assert episode.speculate_accept(observation, raw_rank=2, token_id=2, generation=1)
    outcome = episode.apply(Accept())

    assert outcome.resolved_token_ids == (1,)
    assert backend.eval_calls == [(2,), (1,)]
    assert backend.tokens == [7, 1]
    assert backend._speculation_prefix is None


@pytest.mark.invariant
def test_changed_raw_rank_rolls_back_the_prepared_token_and_evaluates_new_choice():
    backend = SpeculativeFakeBackend()
    episode = engine(backend)
    observation = episode.observe()

    assert episode.speculate_accept(observation, raw_rank=2, token_id=2, generation=1)
    outcome = episode.apply(SelectRawRank(3))

    assert outcome.resolved_token_ids == (3,)
    assert backend.eval_calls == [(2,), (3,)]
    assert backend.tokens == [7, 3]
    assert backend._speculation_prefix is None


@pytest.mark.invariant
def test_old_generation_cannot_replace_newer_in_place_selection():
    backend = SpeculativeFakeBackend()
    episode = engine(backend)
    observation = episode.observe()

    assert episode.speculate_accept(observation, raw_rank=2, token_id=2, generation=2)
    assert not episode.speculate_accept(observation, raw_rank=3, token_id=3, generation=1)
    assert backend.tokens == [7, 2]
    assert episode.has_prepared_accept(observation, 2, 2)
    episode.apply(SelectRawRank(2))

    assert backend.eval_calls == [(2,)]
    assert backend.tokens == [7, 2]


@pytest.mark.invariant
def test_cancellation_during_backend_evaluation_rolls_back_the_result():
    cancelled = False

    def cancel_during_eval() -> None:
        nonlocal cancelled
        cancelled = True

    backend = SpeculativeFakeBackend(on_eval=cancel_during_eval)
    episode = engine(backend)
    observation = episode.observe()

    assert not episode.speculate_accept(
        observation,
        raw_rank=2,
        token_id=2,
        generation=1,
        cancelled=lambda: cancelled,
    )
    assert backend.tokens == [7]
    assert not episode.has_prepared_accept(observation, 2, 2)
    backend.on_eval = None
    episode.apply(SelectRawRank(2))

    assert backend.eval_calls == [(2,), (2,)]
    assert backend.tokens == [7, 2]


@pytest.mark.invariant
def test_invalid_rank_token_pair_does_no_backend_work():
    backend = SpeculativeFakeBackend()
    episode = engine(backend)
    observation = episode.observe()

    assert not episode.speculate_accept(observation, raw_rank=2, token_id=3, generation=1)
    assert not episode.speculate_accept(observation, raw_rank=0, token_id=2, generation=1)
    assert backend.eval_calls == []
    assert backend.tokens == [7]


@pytest.mark.invariant
def test_default_proposal_warm_up_commits_through_accept():
    backend = SpeculativeFakeBackend()
    episode = engine(backend)
    observation = episode.observe()

    assert episode.speculate_accept(observation)
    assert backend.tokens == [7, 1]
    episode.apply(Accept())

    assert backend.eval_calls == [(1,)]
    assert backend.tokens == [7, 1]


@pytest.mark.invariant
def test_final_checkpoint_token_is_not_warmed():
    backend = SpeculativeFakeBackend()
    episode = engine(backend, max_tokens=1)
    observation = episode.observe()

    assert not episode.speculate_accept(observation, raw_rank=2, token_id=2)
    assert backend.eval_calls == []
    assert backend.tokens == [7]


@pytest.mark.invariant
def test_checkpointed_episode_declines_a_stale_warm_up_request():
    backend = SpeculativeFakeBackend()
    episode = engine(backend, max_tokens=1)
    observation = episode.observe()
    episode.apply(Accept())
    assert episode.checkpointed

    assert not episode.speculate_accept(observation, raw_rank=2, token_id=2)
    assert backend.eval_calls == [(1,)]
    assert backend.tokens == [7, 1]


@pytest.mark.invariant
def test_selected_eog_token_is_not_warmed():
    backend = SpeculativeFakeBackend()
    episode = engine(backend)
    observation = episode.observe()
    eog_rank = observation.statistics.raw_rank(0)

    assert not episode.speculate_accept(observation, raw_rank=eog_rank, token_id=0)
    assert backend.eval_calls == []
    assert backend.tokens == [7]


@pytest.mark.invariant
def test_active_cfg_is_not_warmed():
    backend = SpeculativeFakeBackend()
    episode = engine(
        backend,
        sampling=SamplerConfig(
            temperature=0.0,
            cfg_unconditional_prompt="P",
            cfg_scale=1.0,
        ),
        guidance_backend=ConformingFakeBackend(),
    )
    observation = episode.observe()

    assert not episode.speculate_accept(observation, raw_rank=2, token_id=2)
    assert backend.eval_calls == []
    assert backend.tokens == [7]


@pytest.mark.invariant
def test_backend_without_in_place_speculation_support_is_not_warmed():
    backend = ConformingFakeBackend()
    episode = engine(backend)
    observation = episode.observe()

    assert not episode.speculate_accept(observation, raw_rank=2, token_id=2)
    assert backend.tokens == [7]


@pytest.mark.invariant
def test_backend_declining_in_place_speculation_is_not_warmed():
    class DecliningSpeculationBackend(SpeculativeFakeBackend):
        def __init__(self):
            super().__init__()
            self.speculation_attempts = 0

        def speculate(self, token_id: int) -> bool:
            del token_id
            self.speculation_attempts += 1
            return False

    backend = DecliningSpeculationBackend()
    episode = engine(backend)
    observation = episode.observe()

    assert not episode.speculate_accept(observation, raw_rank=2, token_id=2)
    assert backend.speculation_attempts == 1
    assert backend.eval_calls == []
    assert backend.tokens == [7]

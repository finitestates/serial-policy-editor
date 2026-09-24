"""Selected-token speculative warm-up keeps ordinary action resolution authoritative."""

from __future__ import annotations

import pytest

from tests.fakes import ConformingFakeBackend, SnapshotFakeBackend
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
def test_selected_raw_rank_promotes_prepared_state_without_second_evaluation():
    backend = SnapshotFakeBackend()
    episode = engine(backend)
    observation = episode.observe()
    assert observation.proposal_token_id == 1

    assert episode.speculate_accept(observation, raw_rank=2, token_id=2, generation=1)
    assert backend.tokens == [7]
    assert backend.eval_calls == [(2,)]
    assert backend.snapshot_calls == 2
    assert backend.restore_calls == 1

    outcome = episode.apply(SelectRawRank(2))

    assert outcome.resolved_token_ids == (2,)
    assert backend.tokens == [7, 2]
    assert backend.eval_calls == [(2,)]
    assert backend.restore_calls == 2


@pytest.mark.invariant
def test_accept_cannot_promote_a_different_selected_token():
    backend = SnapshotFakeBackend()
    episode = engine(backend)
    observation = episode.observe()

    assert episode.speculate_accept(observation, raw_rank=2, token_id=2, generation=1)
    outcome = episode.apply(Accept())

    assert outcome.resolved_token_ids == (1,)
    assert backend.eval_calls == [(2,), (1,)]
    assert backend.restore_calls == 1


@pytest.mark.invariant
def test_changed_raw_rank_discards_prepared_state():
    backend = SnapshotFakeBackend()
    episode = engine(backend)
    observation = episode.observe()

    assert episode.speculate_accept(observation, raw_rank=2, token_id=2, generation=1)
    outcome = episode.apply(SelectRawRank(3))

    assert outcome.resolved_token_ids == (3,)
    assert backend.eval_calls == [(2,), (3,)]
    assert backend.restore_calls == 1


@pytest.mark.invariant
def test_old_generation_cannot_replace_newer_prepared_selection():
    backend = SnapshotFakeBackend()
    episode = engine(backend)
    observation = episode.observe()

    assert episode.speculate_accept(observation, raw_rank=2, token_id=2, generation=2)
    assert not episode.speculate_accept(observation, raw_rank=3, token_id=3, generation=1)
    episode.apply(SelectRawRank(2))

    assert backend.eval_calls == [(2,)]
    assert backend.restore_calls == 2


@pytest.mark.invariant
def test_cancellation_during_backend_evaluation_discards_the_result():
    cancelled = False

    def cancel_during_eval() -> None:
        nonlocal cancelled
        cancelled = True

    backend = SnapshotFakeBackend(on_eval=cancel_during_eval)
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
    backend.on_eval = None
    episode.apply(SelectRawRank(2))

    assert backend.eval_calls == [(2,), (2,)]
    assert backend.restore_calls == 1


@pytest.mark.invariant
def test_invalid_rank_token_pair_does_no_backend_work():
    backend = SnapshotFakeBackend()
    episode = engine(backend)
    observation = episode.observe()

    assert not episode.speculate_accept(observation, raw_rank=2, token_id=3, generation=1)
    assert not episode.speculate_accept(observation, raw_rank=0, token_id=2, generation=1)
    assert backend.eval_calls == []
    assert backend.snapshot_calls == 0


@pytest.mark.invariant
def test_default_proposal_warm_up_remains_compatible_with_accept():
    backend = SnapshotFakeBackend()
    episode = engine(backend)
    observation = episode.observe()

    assert episode.speculate_accept(observation)
    episode.apply(Accept())

    assert backend.eval_calls == [(1,)]
    assert backend.restore_calls == 2


@pytest.mark.invariant
def test_final_checkpoint_token_is_not_warmed():
    backend = SnapshotFakeBackend()
    episode = engine(backend, max_tokens=1)
    observation = episode.observe()

    assert not episode.speculate_accept(observation, raw_rank=2, token_id=2)
    assert backend.snapshot_calls == 0
    assert backend.eval_calls == []


@pytest.mark.invariant
def test_checkpointed_episode_declines_a_stale_warm_up_request():
    backend = SnapshotFakeBackend()
    episode = engine(backend, max_tokens=1)
    observation = episode.observe()
    episode.apply(Accept())
    assert episode.checkpointed

    assert not episode.speculate_accept(observation, raw_rank=2, token_id=2)
    assert backend.snapshot_calls == 0


@pytest.mark.invariant
def test_selected_eog_token_is_not_warmed():
    backend = SnapshotFakeBackend()
    episode = engine(backend)
    observation = episode.observe()
    eog_rank = observation.statistics.raw_rank(0)

    assert not episode.speculate_accept(observation, raw_rank=eog_rank, token_id=0)
    assert backend.snapshot_calls == 0
    assert backend.eval_calls == []


@pytest.mark.invariant
def test_active_cfg_is_not_warmed():
    backend = SnapshotFakeBackend()
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
    assert backend.snapshot_calls == 0
    assert backend.eval_calls == []


@pytest.mark.invariant
def test_backend_without_exact_snapshot_support_is_not_warmed():
    backend = ConformingFakeBackend()
    episode = engine(backend)
    observation = episode.observe()

    assert not episode.speculate_accept(observation, raw_rank=2, token_id=2)
    assert backend.tokens == [7]


@pytest.mark.invariant
def test_backend_declining_exact_snapshot_is_not_warmed():
    class DecliningSnapshotBackend(SnapshotFakeBackend):
        def snapshot_state(self):
            self.snapshot_calls += 1
            return None

    backend = DecliningSnapshotBackend()
    episode = engine(backend)
    observation = episode.observe()

    assert not episode.speculate_accept(observation, raw_rank=2, token_id=2)
    assert backend.snapshot_calls == 1
    assert backend.eval_calls == []

from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.core.actions import Accept
from trajectory_editor.domain import SamplingConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.research_adapter import (
    core_sampling,
    final_sampling,
    sampling_from_record,
    sampling_at,
)

from tests.fakes import ConformingFakeBackend


def test_research_adapter_round_trips_wide_sampling_records(tmp_path):
    backend = ConformingFakeBackend()
    research_sampling = SamplingConfig(
        reference_prior_routes=(((3,), 2.0),),
        reference_prior_scope="global",
        reference_prior_mode="lexical",
        token_preference_projection_seed=42,
    )
    engine = EpisodeEngine(
        backend,
        sampling=research_sampling,
        initial_token_ids=[7],
    )

    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        episode_id = store.create_episode(
            initial_text="P",
            initial_token_ids=[7],
            sampling=engine.sampling,
            stream_fingerprint=engine.stream_fingerprint,
            coordinate_offset=engine.coordinate_offset,
            max_tokens=None,
            backend=backend.provenance(),
        )

        # The core projection is intentionally narrow.
        assert store.final_sampling(episode_id) == SamplerConfig.from_record(
            research_sampling.to_dict()
        )

        # The research adapter reconstructs the opaque extension fields when
        # a research consumer explicitly asks for them.
        restored = final_sampling(store, episode_id)
        assert restored == research_sampling
        assert sampling_at(store, episode_id).token_preference_projection_seed == 42
        assert core_sampling(restored) == store.final_sampling(episode_id)


def test_research_adapter_accepts_a_future_sampler_without_core_knowledge():
    record = SamplingConfig().to_dict()
    record.update(
        {
            "future_sampler_method": "fictional-quantum-draw-v9",
            "future_sampler_options": {"phase": "imaginary"},
        }
    )

    research_sampling = sampling_from_record(record)
    projected = core_sampling(research_sampling)
    engine = EpisodeEngine(
        ConformingFakeBackend(),
        sampling=projected,
        initial_token_ids=[7],
    )

    assert projected == SamplerConfig.from_record(record)
    outcome = engine.apply(Accept())
    assert len(outcome.visible_token_ids) == 1
    assert engine.boundary == 1

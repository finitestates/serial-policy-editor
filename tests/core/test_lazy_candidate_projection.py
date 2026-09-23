"""Display demand and deferred durable report metrics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.candidate_columns import CandidateColumns
from trajectory_editor.core.actions import Accept, SelectRawRank
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.episode_ui import InteractivePolicy
from trajectory_editor.projector import project_episode

pytestmark = pytest.mark.current_workflow

def test_candidate_view_computes_only_visible_metrics_and_reuses_normalizer(monkeypatch):
    engine = EpisodeEngine(
        ConformingFakeBackend(),
        sampling=SamplerConfig(temperature=0.0, presence_penalty=1.0),
        initial_token_ids=[7],
    )
    observation = engine.observe()
    stats = observation.statistics
    plain = engine.candidates(observation, count=3, view=CandidateColumns().plan)
    assert [row.rank for row in plain] == [1, 2, 3]
    assert all(row.raw_probability is None for row in plain)
    assert not stats._raw_logsumexp_ready

    sparse = engine.candidates(
        observation, count=3,
        view=CandidateColumns(overlays=frozenset({"decode_pct"})).plan,
    )
    assert sparse[0].decoder_probability == observation.distribution.probability(sparse[0].token_id)
    assert not stats._raw_logsumexp_ready

    original_exp = np.exp
    dense_calls = 0

    def counted_exp(values, *args, **kwargs):
        nonlocal dense_calls
        if np.size(values) == len(stats.logits):
            dense_calls += 1
        return original_exp(values, *args, **kwargs)

    monkeypatch.setattr(np, "exp", counted_exp)
    pct = CandidateColumns(show_model_probabilities=True).plan
    first = engine.candidates(observation, count=3, view=pct)
    second = engine.candidates(observation, start_rank=2, count=2, view=pct)
    assert dense_calls == 1
    assert first[1].raw_probability == second[0].raw_probability
    assert [row.token_id for row in first] == [row.token_id for row in plain]
    assert not stats._policy_logsumexp_ready
    with_policy = CandidateColumns(policy=True, show_model_probabilities=True).plan
    policy_rows = engine.candidates(observation, count=3, view=with_policy)
    assert all(row.policy_probability is not None for row in policy_rows)
    assert stats._policy_logsumexp_ready
    assert dense_calls == 2


def test_named_overlays_combine_and_default_table_is_identity_only():
    engine = EpisodeEngine(
        ConformingFakeBackend(),
        sampling=SamplerConfig(temperature=0.0, presence_penalty=1.0),
        initial_token_ids=[7],
    )
    observation = engine.observe()
    io = ScriptedIO(["overlay z", "overlay decode_pct", "1"])
    policy = InteractivePolicy(io=io, menu_size=2)
    policy.choose(engine, observation)
    assert policy.view_preferences.overlays == frozenset({"z", "decode_pct"})
    output = "".join(io.output)
    assert "rank  token-id  text" in output
    assert "Δrank" not in output
    assert "decode-p" in output and "z" in output
    assert not observation.statistics._raw_logsumexp_ready


def test_projector_replays_missing_metrics_without_persisting_them(tmp_path):
    backend = ConformingFakeBackend()
    engine = EpisodeEngine(
        backend, sampling=SamplerConfig(temperature=0.0), initial_token_ids=[7],
    )
    observation = engine.observe()
    outcome = engine.apply(Accept())
    assert not observation.statistics._raw_logsumexp_ready
    assert outcome.evidence[0].raw_model_nll is None

    with EpisodeStore(tmp_path / "episode.db") as store:
        episode_id = store.create_episode(
            initial_text=engine.initial_text,
            initial_token_ids=engine.initial_token_ids,
            sampling=engine.sampling,
            stream_fingerprint=engine.stream_fingerprint,
            coordinate_offset=engine.coordinate_offset,
            max_tokens=engine.max_tokens,
            backend=backend.provenance(),
        )
        store.record_action(episode_id, 0, outcome)
        before = store.tokens(episode_id)[0]
        assert (before["raw_model_nll"], before["raw_rank"], before["policy_rank"]) == (None, None, None)

        report = project_episode(
            store, episode_id, with_loss=True, with_rank=True,
            with_policy_rank=True, backend=ConformingFakeBackend(),
        ).text
        assert "nll=0.1698" in report
        assert "raw-rank=1" in report and "policy-rank=1" in report
        assert store.tokens(episode_id)[0]["raw_model_nll"] is None

        store.connection.execute(
            "UPDATE episodes SET backend_json = ? WHERE episode_id = ?",
            (json.dumps({**backend.provenance(), "model_sha256": "recorded"}), episode_id),
        )

        class DifferentBackend(ConformingFakeBackend):
            def provenance(self, *, include_model_sha256=True):
                return {**super().provenance(), "model_sha256": "different"}

        with pytest.raises(EditorError, match="model identity"):
            project_episode(store, episode_id, with_loss=True, backend=DifferentBackend())

        store.connection.execute(
            "UPDATE tokens SET raw_model_nll = ?, raw_rank = ?, policy_rank = ?",
            (1.25, 7, 6),
        )
        legacy = project_episode(
            store, episode_id, with_loss=True, with_rank=True, with_policy_rank=True,
        ).text
        assert "nll=1.2500" in legacy
        assert "raw-rank=7" in legacy and "policy-rank=6" in legacy


def test_projector_follows_recorded_budget_and_sampler_changes(tmp_path):
    backend = ConformingFakeBackend()
    engine = EpisodeEngine(
        backend, sampling=SamplerConfig(temperature=0.0),
        initial_token_ids=[7], max_tokens=1,
    )
    with EpisodeStore(tmp_path / "renewed.db") as store:
        episode_id = store.create_episode(
            initial_text=engine.initial_text,
            initial_token_ids=engine.initial_token_ids,
            sampling=engine.sampling,
            stream_fingerprint=engine.stream_fingerprint,
            coordinate_offset=engine.coordinate_offset,
            max_tokens=engine.max_tokens,
            backend=backend.provenance(),
        )
        store.record_action(episode_id, 0, engine.apply(Accept()))
        assert engine.checkpointed
        engine.resume(max_tokens=1)
        store.record_budget(
            episode_id, engine.boundary, engine.max_tokens, engine.checkpoint_boundary,
        )
        engine.sampling = SamplerConfig(temperature=0.0, presence_penalty=100.0)
        store.record_sampling_segment(
            episode_id, start_boundary=engine.boundary, sampling=engine.sampling,
            stream_fingerprint=engine.stream_fingerprint,
            coordinate_offset=engine.coordinate_offset,
        )
        expected_policy_rank = engine.observe().statistics.policy_rank(1)
        assert expected_policy_rank > 3
        store.record_action(episode_id, 1, engine.apply(SelectRawRank(3)))
        report = project_episode(
            store, episode_id, with_loss=True, with_policy_rank=True,
            backend=ConformingFakeBackend(),
        ).text
        assert report.count("nll=") == 2
        assert f"policy-rank={expected_policy_rank}" in report


def test_v1_token_table_migrates_to_nullable_report_columns(tmp_path):
    path = tmp_path / "legacy.db"
    with EpisodeStore(path) as store:
        engine = EpisodeEngine(
            ConformingFakeBackend(), sampling=SamplerConfig(temperature=0.0),
            initial_token_ids=[7],
        )
        episode_id = store.create_episode(
            initial_text=engine.initial_text, initial_token_ids=engine.initial_token_ids,
            sampling=engine.sampling, stream_fingerprint=engine.stream_fingerprint,
            coordinate_offset=engine.coordinate_offset, max_tokens=engine.max_tokens,
            backend=engine.backend.provenance(),
        )
        store.record_action(episode_id, 0, engine.apply(Accept()))
        db = store.connection
        db.execute("UPDATE tokens SET raw_model_nll = 1.5, raw_rank = 2, policy_rank = 3")
        db.execute("ALTER TABLE tokens RENAME TO tokens_old")
        db.execute("""CREATE TABLE tokens (
            episode_id TEXT NOT NULL, action_ordinal INTEGER NOT NULL,
            action_token_index INTEGER NOT NULL, boundary INTEGER NOT NULL,
            token_id INTEGER NOT NULL, text TEXT NOT NULL,
            realized_visible INTEGER NOT NULL, is_eog INTEGER NOT NULL,
            sampling_coordinate INTEGER NOT NULL, proposal_token_id INTEGER NOT NULL,
            raw_model_nll REAL NOT NULL, raw_rank INTEGER NOT NULL,
            policy_rank INTEGER NOT NULL, decoder_probability REAL NOT NULL,
            proposal_agreement INTEGER NOT NULL,
            PRIMARY KEY (episode_id, action_ordinal, action_token_index),
            FOREIGN KEY (episode_id, action_ordinal) REFERENCES actions(episode_id, ordinal)
        )""")
        db.execute("INSERT INTO tokens SELECT * FROM tokens_old")
        db.execute("DROP TABLE tokens_old")
        db.execute("UPDATE schema_info SET version = 1")
        db.commit()

    with EpisodeStore(path) as store:
        assert store.tokens(episode_id)[0]["raw_model_nll"] == 1.5
        assert store.connection.execute("SELECT version FROM schema_info").fetchone()[0] == 2
        assert store.connection.execute("PRAGMA table_info(tokens)").fetchall()[10][3] == 0

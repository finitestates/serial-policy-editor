"""Display demand and deferred durable report metrics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.fakes import ConformingFakeBackend, ScriptedIO
from trajectory_editor.candidate_columns import CandidateColumns
from trajectory_editor.core.actions import Accept, SelectRawRank, SetSampler
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
    calculations = observation.policy_calculations
    plain = engine.candidates(observation, count=3, metrics=CandidateColumns().plan.metrics)
    assert [row.rank for row in plain] == [1, 2, 3]
    assert all(row.raw_probability is None for row in plain)
    assert not calculations._raw_logsumexp_ready

    sparse = engine.candidates(
        observation, count=3,
        metrics=frozenset({"eligible_softmax"}),
    )
    assert sparse[0].eligible_softmax == observation.distribution.softmax_at(sparse[0].token_id)
    assert not calculations._raw_logsumexp_ready

    original_exp = np.exp
    dense_calls = 0

    def counted_exp(values, *args, **kwargs):
        nonlocal dense_calls
        if np.size(values) == len(calculations.logits):
            dense_calls += 1
        return original_exp(values, *args, **kwargs)

    monkeypatch.setattr(np, "exp", counted_exp)
    pct = CandidateColumns(overlays=frozenset({"probability"})).plan
    first = engine.candidates(observation, count=3, metrics=pct.metrics)
    second = engine.candidates(observation, start_rank=2, count=2, metrics=pct.metrics)
    assert dense_calls == 1
    assert first[1].raw_probability == second[0].raw_probability
    assert [row.token_id for row in first] == [row.token_id for row in plain]
    assert not calculations._policy_logsumexp_ready
    with_policy = CandidateColumns(policy=True, overlays=frozenset({"probability"})).plan
    policy_rows = engine.candidates(observation, count=3, metrics=with_policy.metrics)
    assert all(row.policy_probability is not None for row in policy_rows)
    assert calculations._policy_logsumexp_ready
    assert dense_calls == 2


def test_named_overlays_combine_and_default_table_is_identity_only():
    engine = EpisodeEngine(
        ConformingFakeBackend(),
        sampling=SamplerConfig(temperature=0.0, presence_penalty=1.0),
        initial_token_ids=[7],
    )
    observation = engine.observe()
    io = ScriptedIO(["overlay noise", "overlay diff", "1"])
    policy = InteractivePolicy(io=io, menu_size=2)
    policy.choose(engine, observation)
    assert policy.view_preferences.overlays == frozenset({"noise", "diff"})
    output = "".join(io.output)
    assert "rank  token-id  text" in output
    assert "Δrank" not in output
    assert "noise" in output and "model-gap" in output
    assert not observation.policy_calculations._raw_logsumexp_ready


def test_projector_replays_missing_metrics_without_persisting_them(tmp_path):
    backend = ConformingFakeBackend()
    engine = EpisodeEngine(
        backend, sampling=SamplerConfig(temperature=0.0), initial_token_ids=[7],
    )
    observation = engine.observe()
    outcome = engine.apply(Accept())
    assert not observation.policy_calculations._raw_logsumexp_ready
    assert outcome.evidence[0].raw_model_nll is None

    with EpisodeStore(tmp_path / "episode.db") as store:
        episode_id = store.create_episode(
            initial_text=engine.initial_text,
            initial_token_ids=engine.initial_token_ids,
            sampling=engine.sampling,
            stream_fingerprint=engine.stream_fingerprint,
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


def test_projector_replays_sampler_changes_as_ordered_actions(tmp_path):
    backend = ConformingFakeBackend()
    engine = EpisodeEngine(
        backend, sampling=SamplerConfig(temperature=0.0),
        initial_token_ids=[7],
    )
    with EpisodeStore(tmp_path / "renewed.db") as store:
        episode_id = store.create_episode(
            initial_text=engine.initial_text,
            initial_token_ids=engine.initial_token_ids,
            sampling=engine.sampling,
            stream_fingerprint=engine.stream_fingerprint,
            backend=backend.provenance(),
        )
        store.record_action(episode_id, 0, engine.apply(Accept()))
        sampler_outcome = engine.apply(
            SetSampler(SamplerConfig(temperature=0.0, presence_penalty=100.0))
        )
        store.record_action(episode_id, 1, sampler_outcome)
        expected_policy_rank = engine.observe().policy_calculations.policy_rank(1)
        assert expected_policy_rank > 3
        store.record_action(episode_id, 2, engine.apply(SelectRawRank(3)))
        report = project_episode(
            store, episode_id, with_loss=True, with_policy_rank=True,
            backend=ConformingFakeBackend(),
        ).text
        assert report.count("nll=") == 2
        assert f"policy-rank={expected_policy_rank}" in report


def test_projector_refuses_to_reconstruct_metrics_past_unknown_saved_action(tmp_path):
    backend = ConformingFakeBackend()
    engine = EpisodeEngine(
        backend, sampling=SamplerConfig(temperature=0.0), initial_token_ids=[7],
    )
    outcome = engine.apply(Accept())
    with EpisodeStore(tmp_path / "episode.db") as store:
        episode_id = store.create_episode(
            initial_text=engine.initial_text,
            initial_token_ids=engine.initial_token_ids,
            sampling=engine.sampling,
            stream_fingerprint=engine.stream_fingerprint,
            backend=backend.provenance(),
        )
        store.record_action(episode_id, 0, outcome)
        with store.transaction() as db:
            db.execute(
                "UPDATE actions SET arguments_json = ? WHERE episode_id = ?",
                ('{"kind":"future-action"}', episode_id),
            )

        with pytest.raises(EditorError, match="source step 1"):
            project_episode(
                store, episode_id, with_loss=True, backend=ConformingFakeBackend(),
            )

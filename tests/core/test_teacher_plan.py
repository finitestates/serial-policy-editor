from __future__ import annotations

import json

import pytest
from trajectory_editor.core.actions import Accept, Hold, Write
from trajectory_editor.core.errors import EditorError
from trajectory_editor.core.results import ReplayExpectation
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_materializer import materialize_live_branch
from trajectory_editor.episode_session import LiveSession
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.run_loop import run_plan
from trajectory_editor.teacher_plan import (
    export_live_teacher_tape,
    export_teacher_tape,
    load_teacher_tape,
    load_teacher_tape_jsonl,
    load_teacher_tape_yaml,
)

from tests.fakes import ConformingFakeBackend

pytestmark = pytest.mark.invariant

def _episode() -> EpisodeEngine:
    return EpisodeEngine(ConformingFakeBackend(), initial_text="Prompt", initial_token_ids=[7], sampling=SamplerConfig(temperature=0.0))


def _create(store: EpisodeStore, engine: EpisodeEngine) -> str:
    return store.create_episode(
        episode_id="source", initial_text=engine.initial_text,
        initial_token_ids=engine.initial_token_ids, sampling=engine.sampling,
        stream_fingerprint=engine.stream_fingerprint,
        backend=engine.backend.provenance(),
    )


def test_exported_teacher_tape_loads_with_its_embedded_envelope(tmp_path):
    path = tmp_path / "source.jsonl"
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        engine = _episode()
        identifier = _create(store, engine)
        store.record_action(identifier, 0, engine.apply(Hold(1)))
        export_teacher_tape(store, identifier, path)

    tape = load_teacher_tape_jsonl(path, require_observations=True)

    assert tape.envelope["format"] == "serial-policy-tape"
    assert tape.envelope["prompt"] == "Prompt"
    assert tape.plan[0].action == Hold(1)
    assert tape.plan[0].expectation is not None


def test_teacher_tape_accepts_a_json_sidecar_envelope(tmp_path):
    plan = tmp_path / "plan.jsonl"
    plan.write_text(json.dumps({"step": 0, "action": {"kind": "accept"}}) + "\n", encoding="utf-8")
    envelope = tmp_path / "plan.json"
    envelope.write_text(json.dumps({"format": "serial-policy-tape", "version": 1, "prompt": "P"}), encoding="utf-8")

    tape = load_teacher_tape_jsonl(plan, envelope_path=envelope)

    assert tape.envelope["prompt"] == "P"
    assert tape.plan[0].action.kind == "accept"


def test_live_export_projects_zero_width_handoff_like_durable_export(tmp_path):
    session = LiveSession(_episode())
    outcome = session.generate(
        Accept(),
        expectation=ReplayExpectation((2,), None, "completed"),
        replay=True,
    )
    assert outcome.status == "handed-off"

    live_path = tmp_path / "live.jsonl"
    stored_path = tmp_path / "stored.jsonl"
    export_live_teacher_tape(session, live_path)
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        identifier = materialize_live_branch(
            store,
            session,
            session.branch_state(),
            {},
            episode_id="saved",
        )
        export_teacher_tape(store, identifier, stored_path)

    assert len(load_teacher_tape_jsonl(live_path).plan) == 0
    assert len(load_teacher_tape_jsonl(stored_path).plan) == 0


def test_live_export_preserves_expectations_for_surviving_steps(tmp_path):
    session = LiveSession(_episode())
    expected = ReplayExpectation((1,), None, "completed")
    outcome = session.generate(Accept(), expectation=expected, replay=True)
    assert outcome.status == "completed"

    path = tmp_path / "live.jsonl"
    export_live_teacher_tape(session, path)

    plan = load_teacher_tape_jsonl(path).plan
    assert plan.steps[0].expectation == expected


def test_live_export_uses_durable_finite_hold_projection_for_partial_handoff(tmp_path):
    session = LiveSession(_episode())
    outcome = session.generate(
        Hold(2),
        expectation=ReplayExpectation((1, 3), None, "requested-length"),
        replay=True,
    )
    assert outcome.status == "handed-off"
    assert outcome.visible_token_ids == (1,)

    live_path = tmp_path / "live.jsonl"
    stored_path = tmp_path / "stored.jsonl"
    export_live_teacher_tape(session, live_path)
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        identifier = materialize_live_branch(
            store,
            session,
            session.branch_state(),
            {},
            episode_id="saved",
        )
        export_teacher_tape(store, identifier, stored_path)

    live = load_teacher_tape_jsonl(live_path).plan
    stored = load_teacher_tape_jsonl(stored_path).plan
    assert live.steps == stored.steps
    assert live.steps[0].action == Hold(1)
    assert live.steps[0].expectation == ReplayExpectation((1,), None, "requested-length")


def test_yaml_plan_preserves_multiline_prompt_write_and_multi_token_observation(tmp_path):
    path = tmp_path / "teacher.yaml"
    path.write_text(
        """format: serial-policy-tape
version: 1
prompt: |-
  First line.
  Second line.
environment:
  sampler:
    temperature: 0.25
    seed: 19
steps:
  - step: 0
    action:
      kind: write
      mode: exact
      text: |2-
          A
          B
    observation:
      token_ids: [1, 2]
      stop_reason: completed
""",
        encoding="utf-8",
    )

    tape = load_teacher_tape_yaml(path, require_observations=True)

    assert tape.envelope["prompt"] == "First line.\nSecond line."
    assert tape.initial_sampling == SamplerConfig(temperature=0.25, seed=19)
    assert len(tape.plan) == 1
    assert tape.plan[0].action == Write("  A\n  B", mode="exact")
    assert tape.plan[0].expectation == ReplayExpectation((1, 2), None, "completed")


def test_yaml_teacher_plan_rejects_duplicate_and_unknown_fields(tmp_path):
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "format: serial-policy-tape\nformat: serial-policy-tape\nversion: 1\nsteps: []\n",
        encoding="utf-8",
    )
    with pytest.raises(EditorError, match="duplicate key"):
        load_teacher_tape_yaml(duplicate)

    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(
        """format: serial-policy-tape
version: 1
steps:
  - step: 0
    action:
      kind: accept
      accepet: true
""",
        encoding="utf-8",
    )
    with pytest.raises(EditorError, match="unknown action fields"):
        load_teacher_tape_yaml(unknown)


def test_yaml_plan_allows_missing_observation_only_when_not_required(tmp_path):
    path = tmp_path / "ballistic.yaml"
    path.write_text(
        """format: serial-policy-tape
version: 1
prompt: P
steps:
  - step: 0
    action:
      kind: accept
""",
        encoding="utf-8",
    )

    assert load_teacher_tape_yaml(path).plan[0].expectation is None
    with pytest.raises(EditorError, match="teacher plan step 0: missing observation"):
        load_teacher_tape_yaml(path, require_observations=True)


def test_yaml_export_roundtrips_one_multi_token_write_action(tmp_path):
    path = tmp_path / "source.yaml"
    with EpisodeStore(tmp_path / "episodes.sqlite3") as store:
        engine = _episode()
        identifier = _create(store, engine)
        outcome = engine.apply(Write(" A B", mode="exact"))
        store.record_action(identifier, 0, outcome)
        store.update_episode(identifier, visible_text=engine.text)
        export_teacher_tape(store, identifier, path)

    tape = load_teacher_tape(path, require_observations=True)

    assert len(tape.plan) == 1
    assert tape.plan[0].action == Write(" A B", mode="exact")
    assert tape.plan[0].expectation == ReplayExpectation((1, 2), None, "completed")

    replay = LiveSession(_episode())
    result = run_plan(replay, divergence_policy="handoff", tape=tape.plan)
    assert result.replayed_actions == 1
    assert replay.engine.boundary == 2

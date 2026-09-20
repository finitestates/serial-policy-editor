from __future__ import annotations

import json

from trajectory_editor.core.actions import Hold
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_store import EpisodeStore
from trajectory_editor.teacher_plan import export_teacher_tape, load_teacher_tape_jsonl
from tests.fakes import ConformingFakeBackend


def _episode() -> EpisodeEngine:
    return EpisodeEngine(ConformingFakeBackend(), initial_text="Prompt", initial_token_ids=[7], sampling=SamplerConfig(temperature=0.0))


def _create(store: EpisodeStore, engine: EpisodeEngine) -> str:
    return store.create_episode(episode_id="source", initial_text=engine.initial_text, initial_token_ids=engine.initial_token_ids, sampling=engine.sampling, stream_fingerprint=engine.stream_fingerprint, coordinate_offset=engine.coordinate_offset, max_tokens=engine.max_tokens, backend=engine.backend.provenance())


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

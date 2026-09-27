"""Fast, model-free checks for the optional real-model harness."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from benchmarks.real_model import ProfileError, compare, load_profile
from benchmarks.real_model_metrics import Measurement, union_ns
from benchmarks.real_model_scenarios import (
    _capture_decision,
    _require_matching_decisions,
    compare_decision_trajectory,
    save_resume,
)
from tests.fakes import ConformingFakeBackend
from trajectory_editor.core.sampler_config import SamplerConfig
from trajectory_editor.episode_engine import EpisodeEngine
from trajectory_editor.episode_cli import main


class Clock:
    value = 0

    def __call__(self):
        return self.value

    def advance(self, amount):
        self.value += amount


def test_nested_service_intervals_count_once_and_keep_residuals():
    clock = Clock()
    meter = Measurement(clock)
    with meter.active():
        clock.advance(10)
        with meter.phase("generation"):
            with meter.service():
                clock.advance(10)
                with meter.service():
                    with meter.model_call("incremental", 1, 8):
                        clock.advance(30)
                    clock.advance(10)
            clock.advance(10)
        clock.advance(30)
    result = meter.result(actions=2, committed_tokens=1)
    assert result["active_wall_s"] == pytest.approx(100e-9)
    assert result["backend_eval_wall_s"] == pytest.approx(50e-9)
    assert result["outside_backend_eval_wall_s"] == pytest.approx(50e-9)
    assert result["phase_wall_s"]["generation"] == pytest.approx(60e-9)
    assert result["unclassified_phase_wall_s"] == pytest.approx(40e-9)
    assert result["work"][0]["context_length"] == 8
    assert union_ns([(0, 10), (5, 20), (30, 35)]) == 25


def test_seeded_decision_parity_ignores_logit_deltas_and_catches_draw_changes():
    class OffsetBackend(ConformingFakeBackend):
        logit_offset = 0.0
        logits_override = None

        def last_logits(self):
            if self.logits_override is not None:
                return self.logits_override.copy()
            return super().last_logits() + np.float32(self.logit_offset)

    backend = OffsetBackend()
    engine = EpisodeEngine(
        backend,
        initial_text="P",
        initial_token_ids=[7],
        sampling=SamplerConfig(temperature=1.0, seed=19),
    )
    capture = _capture_decision(engine, engine.observe())

    # A constant offset produces a large vector delta while preserving the
    # normalized sampling distribution and its replay-stable draw.
    backend.logit_offset = 50.0
    points = compare_decision_trajectory(backend, [capture])
    assert len(points) == 1
    assert points[0]["max_abs_logit_delta"] == pytest.approx(50.0)
    assert points[0]["top_token_agrees"]
    assert points[0]["selected_token_agrees"]
    _require_matching_decisions(points, "test")

    alternate_token = next(
        token_id for token_id in (1, 2, 3)
        if token_id != capture["observed_token_id"]
    )
    backend.logits_override = np.full(backend.vocabulary_size(), -1e6, dtype=np.float32)
    backend.logits_override[alternate_token] = 0.0
    changed_points = compare_decision_trajectory(backend, [capture])
    assert changed_points[0]["fresh_selected_token_id"] == alternate_token
    assert not changed_points[0]["selected_token_agrees"]
    with pytest.raises(AssertionError, match="sampled different tokens"):
        _require_matching_decisions(changed_points, "test")


def test_save_resume_compares_same_seeded_token_decision():
    backend = ConformingFakeBackend()
    result = save_resume(
        backend,
        SamplerConfig(temperature=0.0, seed=17),
        provenance=backend.provenance(),
    )
    assert result["resumed_visible_token_ids"][:len(result["saved_visible_token_ids"])] == result["saved_visible_token_ids"]
    assert result["decision_trajectory"]
    assert all(point["selected_token_agrees"] for point in result["decision_trajectory"])


def test_attach_preserves_optional_branch_capability():
    class WithoutBranch:
        def reset(self, values):
            return list(values)
        def eval(self, values):
            return list(values)

    backend = WithoutBranch()
    meter = Measurement()
    assert not hasattr(backend, "branch_to_prefix")
    with meter.attach(backend):
        assert not hasattr(backend, "branch_to_prefix")
        assert backend.eval([1]) == [1]
    assert not hasattr(backend, "branch_to_prefix")


def test_profile_needs_explicit_root_and_never_replaces_missing_selected_path(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmarks.real_model.importlib.util.find_spec", lambda _name: object())
    (tmp_path / "model.gguf").write_bytes(b"model")
    profile = tmp_path / "profile.yaml"
    profile.write_text("model: model.gguf\nbackend: llama.cpp\nlaunch:\n  n-ctx: 64\n  seed: 7\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="model-root"):
        load_profile(profile, None, None, None, None, None)
    resolved = load_profile(profile, tmp_path, None, None, None, None)
    assert resolved["model_path"] == (tmp_path / "model.gguf").resolve()
    assert resolved["options"].n_ctx == 64
    with pytest.raises(ProfileError, match="missing"):
        load_profile(profile, tmp_path, "missing.gguf", None, None, None)


@pytest.mark.parametrize("content,reason", [
    ("{broken\n", "invalid JSON"),
    (json.dumps({"step": 1, "action": {"kind": "accept"}}) + "\n", "expected step=0"),
    (json.dumps({"step": 0, "action": {"kind": "unknown"}}) + "\n", "invalid action"),
    (json.dumps({"step": 0, "action": {"kind": "accept"}}) + "\n", "missing observation"),
])
def test_invalid_teacher_plan_fails_before_backend_load(tmp_path, content, reason, capsys):
    plan = tmp_path / "bad.jsonl"
    plan.write_text(content, encoding="utf-8")
    with patch("trajectory_editor.episode_backend_loader.load_backend", side_effect=AssertionError("backend loaded")):
        status = main(["--model", "nonexistent.gguf", "--new-prompt", "P",
                       "--teacher-plan", str(plan)])
    assert status == 2
    assert reason in capsys.readouterr().err


def test_compare_rejects_changed_model_and_workload(tmp_path, capsys):
    base = {"format": "spe-real-model-report-v2", "harness_version": 3,
            "scenario_fixture_version": 3, "model": {"model_sha256": "first"}, "hardware": {"cpu": "same"},
            "profile": {"backend": "llama.cpp", "launch": {}, "scenarios": ["continuation"]},
            "samples": [{"scenario": "continuation", "status": "passed", "result": {"visible_token_ids": [1]}}],
            "summary": {}}
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(base), encoding="utf-8")
    changed = json.loads(json.dumps(base))
    changed["model"]["model_sha256"] = "second"
    changed["samples"][0]["result"]["visible_token_ids"] = [2]
    second.write_text(json.dumps(changed), encoding="utf-8")
    assert compare(first, second) == 2
    assert "model identity" in capsys.readouterr().out


def test_compare_ignores_logit_diagnostics_but_tracks_sampled_decisions(tmp_path, capsys):
    base = {
        "format": "spe-real-model-report-v2",
        "harness_version": 3,
        "scenario_fixture_version": 3,
        "model": {"model_sha256": "same"},
        "runtime_identity": {},
        "hardware": {"cpu": "same"},
        "profile": {"backend": "llama.cpp", "launch": {}, "scenarios": ["fork-switch"]},
        "counts": {"failed": 0},
        "samples": [{
            "scenario": "fork-switch",
            "status": "passed",
            "result": {
                "child_visible_token_ids": [1],
                "decision_trajectory": [{
                    "boundary": 0,
                    "selected_token_id": 7,
                    "fresh_selected_token_id": 7,
                    "max_abs_logit_delta": 0.0,
                    "top_token_agrees": True,
                }],
            },
        }],
        "summary": {},
    }
    first = tmp_path / "first-decisions.json"
    second = tmp_path / "second-decisions.json"
    first.write_text(json.dumps(base), encoding="utf-8")
    diagnostic_change = json.loads(json.dumps(base))
    point = diagnostic_change["samples"][0]["result"]["decision_trajectory"][0]
    point["max_abs_logit_delta"] = 50.0
    point["top_token_agrees"] = False
    second.write_text(json.dumps(diagnostic_change), encoding="utf-8")
    assert compare(first, second) == 0

    draw_change = json.loads(json.dumps(diagnostic_change))
    point = draw_change["samples"][0]["result"]["decision_trajectory"][0]
    point["selected_token_id"] = 8
    point["fresh_selected_token_id"] = 8
    second.write_text(json.dumps(draw_change), encoding="utf-8")
    assert compare(first, second) == 2
    assert "fork-switch realized workload" in capsys.readouterr().out

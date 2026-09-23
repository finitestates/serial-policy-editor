"""Fast, model-free checks for the optional real-model harness."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from benchmarks.real_model import ProfileError, compare, load_profile
from benchmarks.real_model_metrics import Measurement, union_ns
from benchmarks.real_model_scenarios import compare_prefix_trajectory
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


def test_prefix_trajectory_collects_later_top_token_divergence():
    class FreshBackend:
        def reset(self, prefix):
            self.prefix = tuple(prefix)

        def last_logits(self):
            return {
                (1,): np.array([4.0, 1.0]),
                (1, 2): np.array([1.0, 4.0]),
                (1, 2, 3): np.array([1.0, 4.0]),
            }[self.prefix]

    captures = [
        ((1,), np.array([4.2, 1.0]), None),
        ((1, 2), np.array([4.0, 1.0]), 2),
        ((1, 2, 3), np.array([1.0, 4.0]), 3),
    ]
    points = compare_prefix_trajectory(FreshBackend(), captures, rtol=0.0, atol=0.01)
    assert len(points) == 3
    assert [point["within_tolerance"] for point in points] == [False, False, True]
    assert [point["top_token_agrees"] for point in points] == [True, False, True]
    assert points[0]["next_committed_token_id"] == 2


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
        status = main(["--ephemeral", "--model", "nonexistent.gguf", "--new-prompt", "P",
                       "--teacher-plan", str(plan)])
    assert status == 2
    assert reason in capsys.readouterr().err


def test_compare_rejects_changed_model_and_workload(tmp_path, capsys):
    base = {"format": "spe-real-model-report-v1", "harness_version": 2,
            "scenario_fixture_version": 2, "model": {"model_sha256": "first"}, "hardware": {"cpu": "same"},
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

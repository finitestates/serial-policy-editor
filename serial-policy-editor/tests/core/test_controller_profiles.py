from pathlib import Path

import pytest
import yaml

from trajectory_editor.core.errors import EditorError
from trajectory_editor.runtime_setup import (
    RuntimePlan,
    apply_controller_profile,
    apply_setup_command,
    controller_profile_fingerprint,
    controller_profile_json,
    controller_profile_yaml,
    load_controller_profile,
    save_controller_profile,
)


def _core_plan(tmp_path: Path) -> RuntimePlan:
    return RuntimePlan(
        workspace=tmp_path / "episodes.sqlite3",
        model=tmp_path / "model.gguf",
        backend="llama.cpp",
        max_tokens=32,
        activation_vector=tmp_path / "style.json",
        temperature=0.72,
        top_k=23,
        cfg_scale=1.2,
        explicit_options={
            "workspace", "model", "backend", "max_tokens", "activation_vector",
            "temperature", "top_k", "cfg_scale",
        },
    )


def test_core_profile_yaml_round_trips_sampler_and_vector_intent(tmp_path):
    plan = _core_plan(tmp_path)
    rendered = controller_profile_yaml(plan)
    document = yaml.safe_load(rendered)

    assert document["format"] == "spe-controller-profile-v1"
    assert document["controllers"]["sampler"]["temperature"] == 0.72
    assert document["controllers"]["steering"]["vector"] == str(
        tmp_path / "style.json"
    )
    assert "workspace" not in rendered
    assert "model" not in rendered
    assert document["fingerprint"] == controller_profile_fingerprint(plan)

    path = tmp_path / "profile.yaml"
    assert save_controller_profile(plan, path) == document["fingerprint"]
    payload, fingerprint = load_controller_profile(path)
    restored = RuntimePlan()
    assert apply_controller_profile(restored, payload) == fingerprint
    assert controller_profile_json(restored) == controller_profile_json(plan)


def test_core_profile_load_does_not_replace_launch_context(tmp_path):
    source = _core_plan(tmp_path)
    path = tmp_path / "profile.yaml"
    save_controller_profile(source, path)
    target = RuntimePlan(
        replay="#2",
        workspace=tmp_path / "other.sqlite3",
        model=tmp_path / "other.gguf",
        backend="transformers",
        temperature=0.3,
        explicit_options={"replay", "workspace", "model", "backend", "temperature"},
    )

    assert apply_setup_command(f"profile load {path}", target) == "profile-loaded"
    assert target.replay == "#2"
    assert target.workspace == tmp_path / "other.sqlite3"
    assert target.model == tmp_path / "other.gguf"
    assert target.backend == "transformers"
    assert target.temperature == source.temperature
    assert target.top_k == source.top_k


def test_core_profile_rejects_unknown_malformed_and_tampered_documents(tmp_path):
    invalid_documents = (
        (
            "format: spe-controller-profile-v1\ncontrollers: {}\nextra: true\n",
            "unknown top-level",
        ),
        (
            "format: spe-controller-profile-v1\ncontrollers:\n  sampler:\n    mystery: 1\n",
            "unknown fields",
        ),
        (
            "format: spe-controller-profile-v1\ncontrollers:\n  sampler:\n    top_k: many\n",
            "must be an integer",
        ),
    )
    for index, (body, message) in enumerate(invalid_documents):
        path = tmp_path / f"bad-{index}.yaml"
        path.write_text(body, encoding="utf-8")
        with pytest.raises(EditorError, match=message):
            load_controller_profile(path)

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "format: spe-controller-profile-v1\n"
        "controllers: {}\ncontrollers: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(EditorError, match="duplicate key"):
        load_controller_profile(duplicate)

    tampered = tmp_path / "tampered.yaml"
    tampered.write_text(
        controller_profile_yaml(RuntimePlan(temperature=0.8)).replace("0.8", "0.7"),
        encoding="utf-8",
    )
    with pytest.raises(EditorError, match="fingerprint mismatch"):
        load_controller_profile(tampered)


def test_core_profile_load_is_transactional_and_save_rejects_directories(tmp_path):
    plan = _core_plan(tmp_path)
    before = (plan.workspace, plan.model, plan.temperature, plan.top_k)
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "format: spe-controller-profile-v1\n"
        "controllers:\n"
        "  sampler:\n"
        "    top_k: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(EditorError, match="must be positive"):
        apply_setup_command(f"profile load {bad}", plan)
    assert (plan.workspace, plan.model, plan.temperature, plan.top_k) == before

    with pytest.raises(EditorError, match="cannot read"):
        apply_setup_command(f"profile load {tmp_path / 'missing.yaml'}", plan)
    assert (plan.workspace, plan.model, plan.temperature, plan.top_k) == before

    with pytest.raises(EditorError, match="is a directory"):
        save_controller_profile(RuntimePlan(), tmp_path)

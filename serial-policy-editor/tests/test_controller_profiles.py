from pathlib import Path

import pytest
import yaml

from trajectory_editor.domain import EditorError
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


def _profile_plan(tmp_path: Path) -> RuntimePlan:
    return RuntimePlan(
        replay="#7",
        at=12,
        workspace=tmp_path / "episodes.sqlite3",
        model=tmp_path / "model.gguf",
        backend="llama.cpp",
        max_tokens=32,
        biases=tmp_path / "biases.json",
        groups=tmp_path / "groups.yaml",
        activation_vector=tmp_path / "style.json",
        temperature=0.72,
        top_k=23,
        online_learning=True,
        learning_rate=0.2,
        learnable_groups=("concrete", "abstract"),
        token_preference=True,
        token_preference_dimension=32,
        token_preference_fast_slow=True,
        explicit_options={
            "replay", "at", "workspace", "model", "backend", "max_tokens",
            "biases", "groups", "activation_vector", "temperature", "top_k",
            "online_learning", "learning_rate", "learnable_groups",
            "token_preference", "token_preference_dimension",
            "token_preference_fast_slow",
        },
    )


def test_profile_yaml_round_trip_and_canonical_identity(tmp_path):
    plan = _profile_plan(tmp_path)
    rendered = controller_profile_yaml(plan)
    document = yaml.safe_load(rendered)

    assert document["format"] == "spe-controller-profile-v1"
    assert document["controllers"]["sampler"]["temperature"] == 0.72
    assert document["controllers"]["steering"]["activation_vector"] == str(
        tmp_path / "style.json"
    )
    assert "replay" not in rendered
    assert "workspace" not in rendered
    assert "model" not in rendered
    assert document["fingerprint"] == controller_profile_fingerprint(plan)

    path = tmp_path / "profiles" / "calm.yaml"
    assert save_controller_profile(plan, path) == document["fingerprint"]
    payload, fingerprint = load_controller_profile(path)
    assert fingerprint == document["fingerprint"]

    restored = RuntimePlan()
    assert apply_controller_profile(restored, payload) == fingerprint
    assert restored.temperature == plan.temperature
    assert restored.top_k == plan.top_k
    assert restored.activation_vector == plan.activation_vector
    assert restored.learnable_groups == plan.learnable_groups
    assert restored.token_preference_fast_slow is True
    assert controller_profile_json(restored) == controller_profile_json(plan)


def test_profile_commands_print_save_and_load_without_episode_context_mutation(tmp_path):
    source = _profile_plan(tmp_path)
    path = tmp_path / "profile.yaml"

    assert apply_setup_command("profile print", source) == "show-profile"
    assert apply_setup_command(f"profile save {path}", source) == "profile-saved"
    assert path.exists()

    target = RuntimePlan(
        replay="#2",
        at=4,
        workspace=tmp_path / "other.sqlite3",
        model=tmp_path / "other.gguf",
        backend="transformers",
        max_tokens=9,
        top_k=99,
        temperature=0.3,
        explicit_options={"replay", "at", "workspace", "model", "backend", "max_tokens", "top_k", "temperature"},
    )
    assert apply_setup_command(f"profile load {path}", target) == "profile-loaded"
    assert target.replay == "#2"
    assert target.at == 4
    assert target.workspace == tmp_path / "other.sqlite3"
    assert target.model == tmp_path / "other.gguf"
    assert target.backend == "transformers"
    assert target.max_tokens == 9
    assert target.temperature == source.temperature
    assert target.top_k == source.top_k


@pytest.mark.parametrize(
    "body, message",
    [
        (
            "format: spe-controller-profile-v1\ncontrollers: {}\nextra: true\n",
            "unknown top-level",
        ),
        (
            "format: spe-controller-profile-v1\ncontrollers:\n  mystery: {}\n",
            "unknown controller sections",
        ),
        (
            "format: spe-controller-profile-v1\ncontrollers:\n  sampler:\n    mystery: 1\n",
            "unknown fields",
        ),
        (
            "format: spe-controller-profile-v1\ncontrollers:\n  sampler:\n    top_k: many\n",
            "must be an integer",
        ),
    ],
)
def test_profile_load_rejects_unknown_or_malformed_yaml(tmp_path, body, message):
    path = tmp_path / "bad.yaml"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(EditorError, match=message):
        load_controller_profile(path)


def test_profile_load_rejects_duplicate_keys_and_fingerprint_tampering(tmp_path):
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "format: spe-controller-profile-v1\n"
        "controllers: {}\ncontrollers: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(EditorError, match="duplicate key"):
        load_controller_profile(duplicate)

    plan = RuntimePlan(temperature=0.8, explicit_options={"temperature"})
    tampered = tmp_path / "tampered.yaml"
    tampered.write_text(controller_profile_yaml(plan).replace("0.8", "0.7"), encoding="utf-8")
    with pytest.raises(EditorError, match="fingerprint mismatch"):
        load_controller_profile(tampered)


def test_profile_load_is_transactional_on_read_or_validation_failure(tmp_path):
    plan = _profile_plan(tmp_path)
    before = (
        plan.replay, plan.workspace, plan.model, plan.temperature, plan.top_k,
        set(plan.explicit_options),
    )
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

    assert (
        plan.replay, plan.workspace, plan.model, plan.temperature, plan.top_k,
        set(plan.explicit_options),
    ) == before

    with pytest.raises(EditorError, match="cannot read"):
        apply_setup_command(f"profile load {tmp_path / 'missing.yaml'}", plan)
    assert plan.replay == before[0]
    assert plan.workspace == before[1]


def test_profile_path_and_replay_safety_are_explicit(tmp_path):
    plan = RuntimePlan(
        replay="#3",
        workspace=tmp_path / "episodes.sqlite3",
        model=tmp_path / "model.gguf",
        temperature=0.4,
        top_k=17,
        explicit_options={"replay", "workspace", "model", "temperature", "top_k"},
    )
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "format: spe-controller-profile-v1\n"
        "controllers:\n"
        "  sampler:\n"
        "    temperature: 0.9\n"
        "  steering:\n"
        f"    activation_vector: {str(tmp_path / 'vector.json')!r}\n",
        encoding="utf-8",
    )

    apply_setup_command(f"profile load {profile}", plan)
    assert plan.replay == "#3"
    assert plan.workspace == tmp_path / "episodes.sqlite3"
    assert plan.model == tmp_path / "model.gguf"
    assert plan.temperature == 0.9
    assert plan.top_k is None  # omitted intent does not become a replay override
    assert plan.activation_vector == tmp_path / "vector.json"
    assert "top_k" not in plan.explicit_options
    assert "replay" in plan.explicit_options


def test_profile_save_rejects_directory_path(tmp_path):
    with pytest.raises(EditorError, match="is a directory"):
        save_controller_profile(RuntimePlan(), tmp_path)

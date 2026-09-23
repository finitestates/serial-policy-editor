from pathlib import Path

import pytest

from trajectory_editor.core.errors import EditorError
from trajectory_editor.controller_profiles import (
    controller_profile_fingerprint,
    explicit_option_dests,
    load_controller_profile,
    profile_arguments,
)
from trajectory_editor.episode_cli import build_parser

pytestmark = pytest.mark.current_workflow

def test_profile_values_are_cli_typed_using_visible_option_names(tmp_path):
    parser = build_parser(include_vector=False)
    path = tmp_path / "profile.yaml"
    path.write_text(
        "model: /tmp/model.gguf\n"
        "temperature: 0.80\n"
        "top-k: 24\n"
        "new-prompt: A saved prompt\n",
        encoding="utf-8",
    )

    values, fingerprint = load_controller_profile(path, parser)

    assert values == {
        "model": Path("/tmp/model.gguf"),
        "temperature": 0.8,
        "top-k": 24,
        "new-prompt": "A saved prompt",
    }
    assert fingerprint == controller_profile_fingerprint(values)


def test_visible_negated_flags_are_presence_booleans(tmp_path):
    parser = build_parser(include_vector=False)
    path = tmp_path / "profile.yaml"
    path.write_text(
        "ephemeral: true\n"
        "no-flash-attn: true\n"
        "no-policy-view: true\n"
        "temperature: 0.8\n",
        encoding="utf-8",
    )

    values, _ = load_controller_profile(path, parser)
    tokens, applied = profile_arguments(parser, values)
    args = parser.parse_args(tokens)

    assert values["ephemeral"] is True
    assert values["no-flash-attn"] is True
    assert values["no-policy-view"] is True
    assert args.ephemeral is True
    assert args.no_flash_attn is True
    assert args.show_policy_rank is False
    assert args.temperature == 0.8
    assert {"ephemeral", "no_flash_attn", "show_policy_rank"} <= applied


def test_false_profile_flags_are_omitted(tmp_path):
    parser = build_parser(include_vector=False)
    path = tmp_path / "profile.yaml"
    path.write_text(
        "no-flash-attn: false\n"
        "no-policy-view: false\n",
        encoding="utf-8",
    )

    values, _ = load_controller_profile(path, parser)
    tokens, applied = profile_arguments(parser, values)
    args = parser.parse_args(tokens)

    assert "--no-flash-attn" not in tokens
    assert "--no-policy-view" not in tokens
    assert args.no_flash_attn is False
    assert args.show_policy_rank is None
    assert "no_flash_attn" not in applied
    assert "show_policy_rank" not in applied


def test_cli_values_override_profile_values_even_in_mutually_exclusive_groups(tmp_path):
    parser = build_parser(include_vector=False)
    path = tmp_path / "profile.yaml"
    path.write_text(
        "model: profile.gguf\n"
        "temperature: 0.80\n"
        "new-prompt: from profile\n"
        "seed: 11\n",
        encoding="utf-8",
    )
    values, _ = load_controller_profile(path, parser)
    argv = ["--profile", str(path), "--model", "cli.gguf", "--replay", "#2", "--random-seed"]
    explicit = explicit_option_dests(parser, argv)
    profile_tokens, applied = profile_arguments(parser, values, overridden=explicit)
    args = parser.parse_args([*profile_tokens, *argv])

    assert args.model == Path("cli.gguf")
    assert args.temperature == 0.8
    assert args.new_prompt is None
    assert args.replay == "#2"
    assert args.seed is None
    assert args.random_seed is True
    assert {"model", "replay", "random_seed"}.isdisjoint(applied)
    assert "temperature" in applied


def test_profile_fingerprint_can_be_embedded_and_is_checked(tmp_path):
    parser = build_parser(include_vector=False)
    values = {"model": Path("model.gguf"), "temperature": 0.8}
    path = tmp_path / "profile.yaml"
    profile = (
        "format: spe-controller-profile-v2\n"
        "model: model.gguf\n"
        "temperature: 0.8\n"
        f"fingerprint: {controller_profile_fingerprint(values)}\n"
    )
    path.write_text(profile, encoding="utf-8")

    loaded, fingerprint = load_controller_profile(path, parser)

    assert loaded == values
    assert fingerprint == controller_profile_fingerprint(values)

    path.write_text(
        profile.replace("temperature: 0.8", "temperature: 0.7"),
        encoding="utf-8",
    )
    with pytest.raises(EditorError, match="fingerprint mismatch"):
        load_controller_profile(path, parser)


@pytest.mark.parametrize(
    "body, message",
    [
        ("unknown: true\n", "unknown option"),
        ("temp: 0.8\n", "unknown option"),
        ("top-k: many\n", "invalid controller profile CLI value"),
        ("top-k: 0\n", "top_k must be at least 1"),
        ("format: other\nmodel: model.gguf\n", "unsupported controller profile format"),
    ],
)
def test_profile_validation_rejects_unknown_or_malformed_values(tmp_path, body, message):
    path = tmp_path / "bad.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(EditorError, match=message):
        load_controller_profile(path, build_parser(include_vector=False))


def test_profile_validation_rejects_duplicate_yaml_keys(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("model: one\nmodel: two\n", encoding="utf-8")

    with pytest.raises(EditorError, match="duplicate key"):
        load_controller_profile(path, build_parser(include_vector=False))

"""Portable logical bias-rule presets (token IDs are model-specific)."""

from __future__ import annotations

import json
from pathlib import Path

from .bias_rules import BiasGroup, BiasRule
from .domain import EditorError, SamplingConfig


FORMAT = "spe-bias-rules-v2"
IDENTITY_FIELDS = ("backend", "filename", "file_size_bytes", "vocabulary_size")


def _validate_rule_tokens(rules: tuple[BiasRule, ...], backend) -> None:
    tokens = [
        token
        for rule in rules
        for route in rule.routes
        for token in route
    ]
    tokens.extend(
        token
        for rule in rules
        for trigger in rule.triggers
        for token in trigger
    )
    tokens.extend(rule.until for rule in rules if type(rule.until) is int)
    if any(token >= backend.vocabulary_size() for token in tokens):
        raise EditorError("Bias rule token ID is outside the loaded vocabulary")


def _validate_group_tokens(groups: tuple[BiasGroup, ...], backend) -> None:
    _validate_rule_tokens(
        tuple(rule for group in groups for rule in group.rules), backend
    )


def load_bias_preset(path: Path, backend, provenance: dict) -> SamplingConfig:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EditorError(f"Could not read bias preset: {exc}") from exc
    if not isinstance(value, dict) or value.get("format") != FORMAT:
        raise EditorError(f"Bias preset must use format {FORMAT}")
    model = value.get("model")
    rows = value.get("bias_rules")
    group_rows = value.get("bias_groups", [])
    if not isinstance(model, dict) or not isinstance(rows, list):
        raise EditorError("Bias preset requires model metadata and a bias_rules list")
    if not isinstance(group_rows, list):
        raise EditorError("Bias preset bias_groups must be a list")
    if type(model.get("vocabulary_size")) is not int or model["vocabulary_size"] != backend.vocabulary_size():
        raise EditorError("Bias preset vocabulary does not match the loaded model")
    for field in IDENTITY_FIELDS:
        if field in model and model[field] != provenance.get(field):
            raise EditorError(f"Bias preset model mismatch: {field}")
    try:
        rules = tuple(BiasRule.from_record(row) for row in rows)
    except (EditorError, TypeError, ValueError) as exc:
        raise EditorError(f"Invalid bias rule preset: {exc}") from exc
    try:
        groups = tuple(BiasGroup.from_record(row) for row in group_rows)
    except (EditorError, TypeError, ValueError) as exc:
        raise EditorError(f"Invalid bias group preset: {exc}") from exc
    _validate_rule_tokens(rules, backend)
    _validate_group_tokens(groups, backend)
    return SamplingConfig(bias_rules=rules, bias_groups=groups)


def project_biases(store, episode_id: str, *, rules_only: bool = False) -> str:
    episode = store.get_episode(episode_id)
    config = store.final_sampling(episode_id)
    model = {
        key: episode["backend"][key]
        for key in IDENTITY_FIELDS
        if episode["backend"].get(key) is not None
    }
    result = {
        "format": FORMAT,
        "model": model,
        "bias_rules": [
            rule.to_dict()
            for rule in (config.effective_bias_rules if rules_only else config.bias_rules)
        ],
    }
    if not rules_only:
        result["bias_groups"] = [group.to_dict() for group in config.bias_groups]
    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

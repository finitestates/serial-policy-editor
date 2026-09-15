"""Portable logical bias-rule presets (token IDs are model-specific)."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

from .bias_rules import BiasGroup, BiasRule
from .domain import EditorError, SamplingConfig
from .latent_features import DEFAULT_PROJECTION_SEED


FORMAT = "spe-bias-rules-v3"
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
    if not isinstance(value, dict) or value.get("format") not in {FORMAT, "spe-bias-rules-v2", "spe-preference-weights-v1"}:
        raise EditorError(f"Bias preset must use format {FORMAT}")
    model = value.get("model")
    rows = value.get("bias_rules", [])
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
    if model.get("tokenizer_fingerprint"):
        from .bias_catalog import tokenizer_fingerprint
        if model["tokenizer_fingerprint"] != tokenizer_fingerprint(backend):
            raise EditorError("Bias preset tokenizer fingerprint does not match the loaded model")
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
    state = SamplingConfig(
        bias_rules=rules,
        bias_groups=groups,
        group_controls=value.get("group_controls", ()),
        latent_preference_z=value.get("latent_preference_z", value.get("weights", ())),
        latent_strength=value.get("latent_strength", 1.0),
        latent_preference_fast_z=value.get("latent_preference_fast_z", ()),
        latent_fast_strength=value.get("latent_fast_strength", 0.0),
        latent_projection_seed=value.get("latent_projection_seed", DEFAULT_PROJECTION_SEED),
        **{f.name: value[f.name] for f in fields(SamplingConfig)
           if f.name.startswith("reference_prior_") and f.name in value},
    )
    prior_tokens = [t for route, _ in state.reference_prior_routes for t in route]
    for c in state.group_controls:
        prior_tokens.extend(t for route in c.triggers for t in route)
        if type(c.until) is int:
            prior_tokens.append(c.until)
    if any(t >= backend.vocabulary_size() for t in prior_tokens):
        raise EditorError("Steering preset token ID is outside the loaded vocabulary")
    return replace(state, group_controls=tuple(replace(c, history_start=None) for c in state.group_controls))


def project_biases_yaml(store, episode_id: str) -> str:
    """Export runtime group names as standalone compiler-input YAML.

    This intentionally exports human-readable group membership rather than
    active bias values or model-specific compiled routes.  Explicit catalog
    reference markers are stripped because a standalone YAML source cannot
    resolve an external catalog.
    """

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a package dependency.
        raise EditorError("PyYAML is required for editor-friendly bias export") from exc
    config = store.final_sampling(episode_id)
    groups = {
        group.name: [
            member[1:] if member.startswith("@") else member
            for member in group.members
        ]
        for group in config.bias_groups
    }
    return yaml.safe_dump(
        {"groups": groups},
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )


def project_biases(
    store,
    episode_id: str,
    *,
    rules_only: bool = False,
    editor_friendly: bool = False,
) -> str:
    if editor_friendly:
        if rules_only:
            raise EditorError("editor-friendly export cannot be combined with --rules-only")
        return project_biases_yaml(store, episode_id)
    episode = store.get_episode(episode_id)
    config = store.final_sampling(episode_id)
    if rules_only and any(c.enabled for c in config.group_controls):
        raise EditorError("adaptive group objectives cannot be flattened into manual rules; export with --biases-only")
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
        result.update({key: value for key, value in config.to_dict().items()
                       if key.startswith(("latent_", "reference_prior_"))})
        result["group_controls"] = [replace(c, history_start=None).to_dict() for c in config.group_controls]
    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )

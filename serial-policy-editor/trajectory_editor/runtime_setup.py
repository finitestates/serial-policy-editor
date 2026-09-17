"""Interactive pre-runtime setup for a new editor session.

This module deliberately edits the launch namespace instead of constructing an
engine.  The existing CLI remains the authority for backend loading, replay,
fork, and sampler validation; the setup surface is a small interactive front
end that produces the same inputs.
"""

from __future__ import annotations

import re
import shlex
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backend_factory import BACKEND_NAMES
from .controller_stack import build_controller_stack
from .domain import EditorError, SamplingConfig
from .learning_controls import DECAY_ON, REJECTION_TARGETS, WRITE_REDUCTIONS
from .token_preference_features import (
    DEFAULT_PROJECTION_CHUNK_SIZE,
    TOKEN_PREFERENCE_FEATURE_SCHEMES,
)


RUNTIME_PLAN_FORMAT = "spe-runtime-plan-v1"


_SOURCE_FIELDS = ("new_prompt", "new_prompt_file", "replay", "resume", "fork_from")
_PATH_FIELDS = {
    "model": "model",
    "biases": "biases",
    "groups": "groups",
    "reference": "reference",
    "activation": "activation_vector",
    "vector": "activation_vector",
}
_SAMPLER_FIELDS = {
    "temperature": float,
    "temp": float,
    "top_k": int,
    "top-p": float,
    "top_p": float,
    "min-p": float,
    "min_p": float,
    "repeat_penalty": float,
    "repeat-penalty": float,
    "repeat_last_n": int,
    "repeat-last-n": int,
    "presence_penalty": float,
    "presence-penalty": float,
    "frequency_penalty": float,
    "frequency-penalty": float,
}
_SAMPLER_ALIASES = {
    "temp": "temperature",
    "top-p": "top_p",
    "min-p": "min_p",
    "repeat-penalty": "repeat_penalty",
    "repeat-last-n": "repeat_last_n",
    "presence-penalty": "presence_penalty",
    "frequency-penalty": "frequency_penalty",
}


_LEARNING_FIELDS = (
    "online_learning", "learning_rate", "learning_epsilon", "learning_max_step",
    "learning_min_bias", "learning_max_bias", "learning_severity_cap",
    "learning_dead_zone_rank", "learning_no_severity_attenuation",
    "learning_rejection_strength", "learning_decay", "learning_decay_on",
    "learning_write_reduction", "learning_rejection_target", "learning_gate",
    "learnable_groups", "learn_from_write",
)
_PREFERENCE_FIELDS = (
    "token_preference", "token_preference_dimension", "token_preference_learning_rate",
    "token_preference_strength", "token_preference_feature_scheme",
    "token_preference_whitening_ridge", "token_preference_learning_scheme",
    "token_preference_influence_mode", "token_preference_influence_kl",
    "token_preference_min_gain", "token_preference_max_gain", "token_preference_max_step",
    "token_preference_max_norm", "token_preference_decay", "token_preference_decay_on",
    "token_preference_write_reduction", "token_preference_rejection_target",
    "token_preference_severity_cap", "token_preference_no_severity_attenuation",
    "token_preference_dead_zone_rank", "token_preference_learning_gate",
    "token_preference_rejection_strength", "token_preference_fast_slow",
    "token_preference_fast_learning_rate", "token_preference_fast_decay",
    "token_preference_fast_strength", "token_preference_fast_max_step",
    "token_preference_fast_max_norm", "token_preference_learning_metric",
    "token_preference_learning_kl", "token_preference_fast_learning_kl",
    "token_preference_fisher_ridge", "token_preference_fisher_mode",
    "token_preference_fisher_mass", "token_preference_fisher_max_support",
    "token_preference_projection_seed", "token_preference_random_projection_seed",
    "token_preference_projection_chunk_size",
)


_PLAN_FIELDS = (
    "new_prompt", "new_prompt_file", "replay", "resume", "fork_from", "at",
    "workspace", "model", "backend", "max_tokens", "biases", "groups", "reference",
    "activation_vector", "temperature", "top_k", "top_p", "min_p",
    "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty",
    "seed", "random_seed", *_LEARNING_FIELDS, *_PREFERENCE_FIELDS,
)


@dataclass
class RuntimePlan:
    """Typed state assembled before the episode engine is created.

    ``explicit_options`` records settings chosen by CLI or menu input.  The
    launcher uses that distinction when a replay source should supply its
    historical sampler state and the user has supplied only selected
    overrides.
    """

    new_prompt: str | None = None
    new_prompt_file: Path | None = None
    replay: str | None = None
    resume: str | None = None
    fork_from: str | None = None
    at: int | None = None
    workspace: Path = Path("episodes.sqlite3")
    model: Path | None = None
    backend: str | None = None
    max_tokens: int | None = None
    biases: Path | None = None
    groups: Path | None = None
    reference: Path | None = None
    activation_vector: Path | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    random_seed: bool = False
    online_learning: bool = False
    learning_rate: float = 0.05
    learning_epsilon: float = 0.05
    learning_max_step: float = 0.25
    learning_min_bias: float = -4.0
    learning_max_bias: float = 4.0
    learning_severity_cap: int = 1000
    learning_dead_zone_rank: int = 1
    learning_no_severity_attenuation: bool = False
    learning_rejection_strength: float = 0.0
    learning_decay: float = 0.0
    learning_decay_on: str = "update"
    learning_write_reduction: str = "sum"
    learning_rejection_target: str = "proposal"
    learning_gate: str = "rank"
    learnable_groups: list[str] | tuple[str, ...] | None = None
    learn_from_write: bool = False
    token_preference: bool = False
    token_preference_dimension: int = 64
    token_preference_learning_rate: float = 0.05
    token_preference_strength: float = 1.0
    token_preference_feature_scheme: str | None = None
    token_preference_whitening_ridge: float | None = None
    token_preference_learning_scheme: str | None = None
    token_preference_influence_mode: str | None = None
    token_preference_influence_kl: float | None = None
    token_preference_min_gain: float | None = None
    token_preference_max_gain: float | None = None
    token_preference_max_step: float = 0.25
    token_preference_max_norm: float = 4.0
    token_preference_decay: float = 0.0
    token_preference_decay_on: str = "update"
    token_preference_write_reduction: str = "sum"
    token_preference_rejection_target: str = "proposal"
    token_preference_severity_cap: int = 1000
    token_preference_no_severity_attenuation: bool = False
    token_preference_dead_zone_rank: int = 1
    token_preference_learning_gate: str = "rank"
    token_preference_rejection_strength: float = 0.0
    token_preference_fast_slow: bool = False
    token_preference_fast_learning_rate: float | None = None
    token_preference_fast_decay: float = 0.10
    token_preference_fast_strength: float | None = None
    token_preference_fast_max_step: float | None = None
    token_preference_fast_max_norm: float | None = None
    token_preference_learning_metric: str | None = None
    token_preference_learning_kl: float | None = None
    token_preference_fast_learning_kl: float | None = None
    token_preference_fisher_ridge: float | None = None
    token_preference_fisher_mode: str | None = None
    token_preference_fisher_mass: float | None = None
    token_preference_fisher_max_support: int | None = None
    token_preference_projection_seed: int | None = None
    token_preference_random_projection_seed: bool = False
    token_preference_projection_chunk_size: int = DEFAULT_PROJECTION_CHUNK_SIZE
    explicit_options: set[str] = field(default_factory=set)

    @classmethod
    def from_args(cls, args: Namespace) -> "RuntimePlan":
        values = {name: getattr(args, name, None) for name in _PLAN_FIELDS}
        values["explicit_options"] = set(getattr(args, "_explicit_options", ()))
        return cls(**values)

    def update(self, name: str, value: Any, *, explicit: bool = True) -> None:
        if name not in _PLAN_FIELDS:
            raise AttributeError(f"unknown runtime plan field {name!r}")
        setattr(self, name, value)
        if explicit:
            self.explicit_options.add(name)

    def has_source(self) -> bool:
        return any(getattr(self, field) is not None for field in _SOURCE_FIELDS)

    def apply_to_args(self, args: Namespace) -> None:
        """Project the finalized plan onto the legacy launcher boundary."""
        for name in _PLAN_FIELDS:
            setattr(args, name, getattr(self, name))
        args._explicit_options = set(self.explicit_options)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe profile representation of this plan."""
        path_fields = {
            "workspace", "model", "biases", "groups", "reference", "activation_vector",
        }
        values = {}
        for name in _PLAN_FIELDS:
            value = getattr(self, name)
            values[name] = str(value) if name in path_fields and value is not None else value
        return {
            "format": RUNTIME_PLAN_FORMAT,
            "values": values,
            "explicit_options": sorted(self.explicit_options),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RuntimePlan":
        """Restore a plan profile, rejecting unknown formats and fields."""
        if value.get("format") != RUNTIME_PLAN_FORMAT:
            raise EditorError(
                f"unsupported runtime plan format {value.get('format')!r}; expected {RUNTIME_PLAN_FORMAT}"
            )
        raw_values = value.get("values")
        if not isinstance(raw_values, dict):
            raise EditorError("runtime plan profile is missing its values")
        unknown = set(raw_values) - set(_PLAN_FIELDS)
        if unknown:
            raise EditorError(
                "runtime plan profile has unknown fields: " + ", ".join(sorted(unknown))
            )
        path_fields = {
            "workspace", "model", "biases", "groups", "reference", "activation_vector",
        }
        values = {name: raw_values.get(name, getattr(cls(), name)) for name in _PLAN_FIELDS}
        for name in path_fields:
            if values[name] is not None:
                values[name] = Path(values[name])
        explicit = value.get("explicit_options", ())
        if not isinstance(explicit, (list, tuple, set)):
            raise EditorError("runtime plan profile explicit_options must be a list")
        values["explicit_options"] = set(explicit)
        return cls(**values)


def _set(plan: RuntimePlan, name: str, value: Any, *, explicit: bool = True) -> None:
    plan.update(name, value, explicit=explicit)


def _clear_source(plan: RuntimePlan) -> None:
    plan.explicit_options.difference_update((*_SOURCE_FIELDS, "at"))
    for name in _SOURCE_FIELDS:
        _set(plan, name, None, explicit=False)
    _set(plan, "at", None, explicit=False)


def _path(value: str, *, label: str) -> Path:
    if not value:
        raise EditorError(f"{label} requires a path")
    return Path(value).expanduser()


def _episode_reference(value: str) -> str:
    value = value.strip()
    return f"#{value}" if value.isdigit() else value


def _source_summary(plan: RuntimePlan) -> str:
    if plan.new_prompt is not None:
        text = str(plan.new_prompt).replace("\n", " ")
        return f"new prompt: {text[:58]}{'…' if len(text) > 58 else ''}"
    if plan.new_prompt_file is not None:
        return f"new prompt file: {plan.new_prompt_file}"
    for field, label in (
        ("replay", "replay"),
        ("resume", "resume"),
        ("fork_from", "fork"),
    ):
        value = getattr(plan, field, None)
        if value is not None:
            suffix = f" at {plan.at}" if field == "fork_from" and plan.at is not None else ""
            return f"{label}: {value}{suffix}"
    return "not selected"


def _display_path(value: Any) -> str:
    return str(value) if value is not None else "none"


_LEARNING_ALIASES = {
    "enabled": "online_learning",
    "rate": "learning_rate",
    "epsilon": "learning_epsilon",
    "max_step": "learning_max_step",
    "min_bias": "learning_min_bias",
    "max_bias": "learning_max_bias",
    "severity_cap": "learning_severity_cap",
    "dead_zone_rank": "learning_dead_zone_rank",
    "no_severity_attenuation": "learning_no_severity_attenuation",
    "rejection_strength": "learning_rejection_strength",
    "decay": "learning_decay",
    "decay_on": "learning_decay_on",
    "write_reduction": "learning_write_reduction",
    "rejection_target": "learning_rejection_target",
    "gate": "learning_gate",
    "groups": "learnable_groups",
    "learnable": "learnable_groups",
    "learn_from_write": "learn_from_write",
    "from_write": "learn_from_write",
}
_PREFERENCE_ALIASES = {
    "enabled": "token_preference",
    "dimension": "token_preference_dimension",
    "rate": "token_preference_learning_rate",
    "strength": "token_preference_strength",
    "feature": "token_preference_feature_scheme",
    "feature_scheme": "token_preference_feature_scheme",
    "whitening_ridge": "token_preference_whitening_ridge",
    "ridge": "token_preference_whitening_ridge",
    "learning_scheme": "token_preference_learning_scheme",
    "scheme": "token_preference_learning_scheme",
    "influence_mode": "token_preference_influence_mode",
    "influence_kl": "token_preference_influence_kl",
    "min_gain": "token_preference_min_gain",
    "max_gain": "token_preference_max_gain",
    "max_step": "token_preference_max_step",
    "max_norm": "token_preference_max_norm",
    "decay": "token_preference_decay",
    "decay_on": "token_preference_decay_on",
    "write_reduction": "token_preference_write_reduction",
    "rejection_target": "token_preference_rejection_target",
    "severity_cap": "token_preference_severity_cap",
    "no_severity_attenuation": "token_preference_no_severity_attenuation",
    "dead_zone_rank": "token_preference_dead_zone_rank",
    "gate": "token_preference_learning_gate",
    "learning_gate": "token_preference_learning_gate",
    "rejection_strength": "token_preference_rejection_strength",
    "fast_slow": "token_preference_fast_slow",
    "fast_learning_rate": "token_preference_fast_learning_rate",
    "fast_decay": "token_preference_fast_decay",
    "fast_strength": "token_preference_fast_strength",
    "fast_max_step": "token_preference_fast_max_step",
    "fast_max_norm": "token_preference_fast_max_norm",
    "learning_metric": "token_preference_learning_metric",
    "metric": "token_preference_learning_metric",
    "learning_kl": "token_preference_learning_kl",
    "fast_learning_kl": "token_preference_fast_learning_kl",
    "fisher_ridge": "token_preference_fisher_ridge",
    "fisher_mode": "token_preference_fisher_mode",
    "fisher_mass": "token_preference_fisher_mass",
    "fisher_max_support": "token_preference_fisher_max_support",
    "projection_seed": "token_preference_projection_seed",
    "random_projection_seed": "token_preference_random_projection_seed",
    "projection_chunk_size": "token_preference_projection_chunk_size",
}
_CONTROL_BOOL_FIELDS = {
    "online_learning", "learning_no_severity_attenuation", "learn_from_write",
    "token_preference", "token_preference_no_severity_attenuation",
    "token_preference_fast_slow", "token_preference_random_projection_seed",
}
_CONTROL_INT_FIELDS = {
    "learning_severity_cap", "learning_dead_zone_rank", "token_preference_dimension",
    "token_preference_severity_cap", "token_preference_dead_zone_rank",
    "token_preference_fisher_max_support", "token_preference_projection_chunk_size",
}
_CONTROL_FLOAT_FIELDS = {
    "learning_rate", "learning_epsilon", "learning_max_step", "learning_min_bias",
    "learning_max_bias", "learning_rejection_strength", "learning_decay",
    "token_preference_learning_rate", "token_preference_strength",
    "token_preference_whitening_ridge", "token_preference_influence_kl",
    "token_preference_min_gain", "token_preference_max_gain", "token_preference_max_step",
    "token_preference_max_norm", "token_preference_decay",
    "token_preference_rejection_strength", "token_preference_fast_learning_rate",
    "token_preference_fast_decay", "token_preference_fast_strength",
    "token_preference_fast_max_step", "token_preference_fast_max_norm",
    "token_preference_learning_kl", "token_preference_fast_learning_kl",
    "token_preference_fisher_ridge", "token_preference_fisher_mass",
}
_CONTROL_OPTIONAL_FIELDS = {
    "token_preference_feature_scheme", "token_preference_whitening_ridge",
    "token_preference_learning_scheme", "token_preference_influence_mode",
    "token_preference_influence_kl", "token_preference_min_gain",
    "token_preference_max_gain", "token_preference_fast_learning_rate",
    "token_preference_fast_strength", "token_preference_fast_max_step",
    "token_preference_fast_max_norm", "token_preference_learning_metric",
    "token_preference_learning_kl", "token_preference_fast_learning_kl",
    "token_preference_fisher_ridge", "token_preference_fisher_mode",
    "token_preference_fisher_mass", "token_preference_fisher_max_support",
}
_CONTROL_CHOICES = {
    "learning_decay_on": DECAY_ON,
    "learning_write_reduction": WRITE_REDUCTIONS,
    "learning_rejection_target": REJECTION_TARGETS,
    "learning_gate": ("rank", "sampler"),
    "token_preference_feature_scheme": TOKEN_PREFERENCE_FEATURE_SCHEMES,
    "token_preference_learning_scheme": ("sgd-v1", "fisher-kl-v2"),
    "token_preference_influence_mode": ("manual", "kl"),
    "token_preference_decay_on": DECAY_ON,
    "token_preference_write_reduction": WRITE_REDUCTIONS,
    "token_preference_rejection_target": REJECTION_TARGETS,
    "token_preference_learning_gate": ("rank", "sampler"),
    "token_preference_learning_metric": ("euclidean", "fisher"),
    "token_preference_fisher_mode": ("diagonal", "full"),
}


def _control_destination(raw_key: str, *, preference: bool) -> str:
    key = raw_key.strip().lower().replace("-", "_")
    aliases = _PREFERENCE_ALIASES if preference else _LEARNING_ALIASES
    fields = _PREFERENCE_FIELDS if preference else _LEARNING_FIELDS
    if key in fields:
        return key
    prefix = "token_preference_" if preference else "learning_"
    if key.startswith(prefix):
        key = key[len(prefix):]
    destination = aliases.get(key)
    if destination is None:
        raise EditorError(f"unknown {'preference' if preference else 'group'} setting {raw_key!r}")
    return destination


def _parse_bool(value: str, *, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"on", "true", "yes", "1"}:
        return True
    if normalized in {"off", "false", "no", "0"}:
        return False
    raise EditorError(f"{label} must be on or off")


def _parse_control_value(destination: str, raw_value: str) -> Any:
    normalized = raw_value.strip().lower()
    if destination in _CONTROL_BOOL_FIELDS:
        return _parse_bool(raw_value, label=destination)
    if destination == "learnable_groups":
        if normalized in {"all", "none", "default", "off"}:
            return None
        values = tuple(part.strip() for part in raw_value.split(",") if part.strip())
        if not values:
            raise EditorError("groups must be all or a comma-separated list")
        return values
    if destination == "token_preference_projection_seed":
        if normalized in {"default", "none", "off"}:
            return None
        if normalized == "random":
            return None
        try:
            return int(raw_value)
        except ValueError as exc:
            raise EditorError("projection_seed must be an integer, random, or default") from exc
    if destination in _CONTROL_OPTIONAL_FIELDS and normalized in {"default", "none", "off"}:
        return None
    choices = _CONTROL_CHOICES.get(destination)
    if choices is not None:
        if normalized not in choices:
            joined = ", ".join(choices)
            raise EditorError(f"{destination} must be one of: {joined}")
        return normalized
    if destination in _CONTROL_INT_FIELDS:
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise EditorError(f"{destination} must be an integer") from exc
        if value < 1:
            raise EditorError(f"{destination} must be positive")
        return value
    if destination in _CONTROL_FLOAT_FIELDS:
        try:
            return float(raw_value)
        except ValueError as exc:
            raise EditorError(f"{destination} must be a number") from exc
    raise EditorError(f"unknown control setting {destination!r}")


def _parse_control_settings(plan: RuntimePlan, words: list[str], *, preference: bool) -> None:
    for piece in words[1:]:
        if "=" not in piece:
            raise EditorError("control changes use key=value")
        raw_key, raw_value = piece.split("=", 1)
        destination = _control_destination(raw_key, preference=preference)
        value = _parse_control_value(destination, raw_value)
        _set(plan, destination, value)
        if destination == "token_preference_projection_seed":
            _set(plan, "token_preference_random_projection_seed", raw_value.strip().lower() == "random")
        elif destination == "token_preference_random_projection_seed" and value:
            _set(plan, "token_preference_projection_seed", None)


def _setting_value(value: Any, *, none_label: str = "default") -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if value is None:
        return none_label
    if isinstance(value, (list, tuple)):
        return "all groups" if not value else ", ".join(str(item) for item in value)
    return str(value)


def learning_summary(plan: RuntimePlan) -> str:
    """Describe the manual-group learner and all of its menu controls."""

    rows = [
        "MANUAL GROUP LEARNING SETTINGS",
        "",
        f"enabled                 {_setting_value(plan.online_learning)}",
        f"learning_rate           {_setting_value(plan.learning_rate)}",
        f"epsilon                 {_setting_value(plan.learning_epsilon)}",
        f"max_step                {_setting_value(plan.learning_max_step)}",
        f"min_bias                {_setting_value(plan.learning_min_bias)}",
        f"max_bias                {_setting_value(plan.learning_max_bias)}",
        f"severity_cap            {_setting_value(plan.learning_severity_cap)}",
        f"dead_zone_rank          {_setting_value(plan.learning_dead_zone_rank)}",
        f"no_severity_attenuation {_setting_value(plan.learning_no_severity_attenuation)}",
        f"rejection_strength      {_setting_value(plan.learning_rejection_strength)}",
        f"decay                   {_setting_value(plan.learning_decay)}",
        f"decay_on                {_setting_value(plan.learning_decay_on)}",
        f"write_reduction         {_setting_value(plan.learning_write_reduction)}",
        f"rejection_target        {_setting_value(plan.learning_rejection_target)}",
        f"gate                    {_setting_value(plan.learning_gate)}",
        f"learnable_groups        {_setting_value(plan.learnable_groups, none_label='all groups')}",
        f"learn_from_write        {_setting_value(plan.learn_from_write)}",
        "",
        "Change with: group key=value [...]",
        "Aliases: learning key=value [...] and group on|off",
    ]
    return "\n".join(rows)


def preference_summary(plan: RuntimePlan) -> str:
    """Describe token-preference learning and all of its menu controls."""

    rows = [
        "TOKEN PREFERENCE SETTINGS",
        "",
        f"enabled                 {_setting_value(plan.token_preference)}",
        f"dimension               {_setting_value(plan.token_preference_dimension)}",
        f"learning_rate           {_setting_value(plan.token_preference_learning_rate)}",
        f"strength                {_setting_value(plan.token_preference_strength)}",
        f"feature_scheme          {_setting_value(plan.token_preference_feature_scheme)}",
        f"whitening_ridge         {_setting_value(plan.token_preference_whitening_ridge)}",
        f"learning_scheme         {_setting_value(plan.token_preference_learning_scheme)}",
        f"influence_mode          {_setting_value(plan.token_preference_influence_mode)}",
        f"influence_kl            {_setting_value(plan.token_preference_influence_kl)}",
        f"min_gain                {_setting_value(plan.token_preference_min_gain)}",
        f"max_gain                {_setting_value(plan.token_preference_max_gain)}",
        f"max_step                {_setting_value(plan.token_preference_max_step)}",
        f"max_norm                {_setting_value(plan.token_preference_max_norm)}",
        f"decay                   {_setting_value(plan.token_preference_decay)}",
        f"decay_on                {_setting_value(plan.token_preference_decay_on)}",
        f"write_reduction         {_setting_value(plan.token_preference_write_reduction)}",
        f"rejection_target        {_setting_value(plan.token_preference_rejection_target)}",
        f"severity_cap            {_setting_value(plan.token_preference_severity_cap)}",
        f"no_severity_attenuation {_setting_value(plan.token_preference_no_severity_attenuation)}",
        f"dead_zone_rank          {_setting_value(plan.token_preference_dead_zone_rank)}",
        f"learning_gate            {_setting_value(plan.token_preference_learning_gate)}",
        f"rejection_strength      {_setting_value(plan.token_preference_rejection_strength)}",
        f"fast_slow                {_setting_value(plan.token_preference_fast_slow)}",
        f"fast_learning_rate      {_setting_value(plan.token_preference_fast_learning_rate)}",
        f"fast_decay              {_setting_value(plan.token_preference_fast_decay)}",
        f"fast_strength            {_setting_value(plan.token_preference_fast_strength)}",
        f"fast_max_step            {_setting_value(plan.token_preference_fast_max_step)}",
        f"fast_max_norm            {_setting_value(plan.token_preference_fast_max_norm)}",
        f"learning_metric          {_setting_value(plan.token_preference_learning_metric)}",
        f"learning_kl              {_setting_value(plan.token_preference_learning_kl)}",
        f"fast_learning_kl         {_setting_value(plan.token_preference_fast_learning_kl)}",
        f"fisher_ridge             {_setting_value(plan.token_preference_fisher_ridge)}",
        f"fisher_mode              {_setting_value(plan.token_preference_fisher_mode)}",
        f"fisher_mass              {_setting_value(plan.token_preference_fisher_mass)}",
        f"fisher_max_support      {_setting_value(plan.token_preference_fisher_max_support)}",
        f"projection_seed          {_setting_value(plan.token_preference_projection_seed)}",
        f"random_projection_seed   {_setting_value(plan.token_preference_random_projection_seed)}",
        f"projection_chunk_size    {_setting_value(plan.token_preference_projection_chunk_size)}",
        "",
        "Change with: preference key=value [...]",
        "Examples: preference dimension=32 fast_slow=on projection_seed=random",
    ]
    return "\n".join(rows)


def setup_summary(plan: RuntimePlan) -> str:
    """Return the compact state shown by the pre-runtime menu."""

    sampler = " ".join(
        f"{name}={getattr(plan, name)}"
        for name in ("temperature", "top_k", "top_p", "min_p")
        if getattr(plan, name, None) is not None
    ) or "defaults"
    return "\n".join(
        (
            "PRE-RUNTIME SETUP",
            "────────────────────────────────────────",
            f"Workspace    {plan.workspace}",
            f"Source       {_source_summary(plan)}",
            f"Backend      {getattr(plan, 'backend', None) or 'auto'}",
            f"Model        {_display_path(getattr(plan, 'model', None))}",
            f"Budget       {getattr(plan, 'max_tokens', None) or 'unlimited'}",
            "",
            "Controls",
            f"  biases     {_display_path(getattr(plan, 'biases', None))}",
            f"  groups     {_display_path(getattr(plan, 'groups', None))}",
            f"  reference  {_display_path(getattr(plan, 'reference', None))}",
            f"  activation {_display_path(getattr(plan, 'activation_vector', None))}",
            "",
            f"Sampler      {sampler}",
            f"Seed         {getattr(plan, 'seed', None) if getattr(plan, 'seed', None) is not None else 'default'}",
            f"Group learn  {_setting_value(plan.online_learning)}",
            f"Preference   {_setting_value(plan.token_preference)}",
            f"Controllers  {build_controller_stack(plan=plan).compact()}",
            "",
            "Commands",
            "  workspace [PATH]            show or switch episode workspace",
            "  ls                          list episodes in this workspace",
            "  prompt [TEXT]               use a new prompt",
            "  prompt-file PATH            read a prompt from a file",
            "  source replay|resume|fork ID [at N]",
            "  model PATH                  choose a model",
            "  backend llama.cpp|transformers",
            "  biases|groups|reference|activation PATH",
            "  sampler key=value [...]     change sampler settings",
            "  budget N|off                set visible-token allowance",
            "  seed N|random|default",
            "  learning|group [key=value]  show/configure group learning",
            "  preference [key=value]      show/configure token preference",
            "  controllers|stack           show ordered control surfaces",
            "  #N or N                     inspect an episode",
            "  fm [#N]                     show its fork map",
            "  show                        redraw this summary",
            "  go                          validate and start",
            "  quit                        cancel setup",
        )
    )


def sampler_summary(plan: RuntimePlan) -> str:
    """Describe every ordinary sampler field and its planned value."""

    defaults = SamplingConfig()
    source_selected = any(
        getattr(plan, field) is not None for field in ("replay", "resume", "fork_from")
    )
    rows = ["SAMPLER SETTINGS", ""]
    for name in (
        "temperature", "top_k", "top_p", "min_p", "repeat_penalty",
        "repeat_last_n", "presence_penalty", "frequency_penalty",
    ):
        selected = getattr(plan, name)
        value = (
            selected
            if selected is not None
            else "inherited from source" if source_selected else getattr(defaults, name)
        )
        rows.append(f"{name:<17} {value}")
    seed = plan.seed if plan.seed is not None else (
        "random" if plan.random_seed else "inherited from source" if source_selected else defaults.seed
    )
    rows.extend(
        (
            f"{'seed':<17} {seed}",
            "",
            "Change with: sampler key=value [...]",
        )
    )
    return "\n".join(rows)


def effective_plan_summary(
    plan: RuntimePlan,
    sampling: SamplingConfig,
    *,
    source_sampling: SamplingConfig | None = None,
    provenance: dict[str, Any] | None = None,
    validated_artifacts: tuple[str, ...] = (),
) -> str:
    """Render the effective, backend-validated plan immediately before go."""

    provenance = provenance or {}
    source_selected = source_sampling is not None
    rows = [
        "RUNTIME PREFLIGHT",
        "────────────────────────────────────────",
        f"Workspace    {plan.workspace}",
        f"Source       {_source_summary(plan)}",
        f"Backend      {provenance.get('backend') or plan.backend or 'auto'}",
        f"Model        {provenance.get('model_path') or _display_path(plan.model)}",
        "",
        "SAMPLER",
    ]
    for name in (
        "temperature", "top_k", "top_p", "min_p", "repeat_penalty",
        "repeat_last_n", "presence_penalty", "frequency_penalty", "seed",
    ):
        value = getattr(sampling, name)
        if name in plan.explicit_options or getattr(plan, name, None) is not None:
            origin = "override"
        elif source_selected:
            origin = "inherited"
        else:
            origin = "default"
        rows.append(f"  {name:<16} {value}  [{origin}]")
    rows.extend(("", "STEERING"))
    if sampling.activation_vector_digest:
        layer = sampling.activation_vector_layer
        rows.append(
            f"  activation       {sampling.activation_vector_digest[:12]} "
            f"layer={layer} strength={sampling.activation_vector_strength:g}"
        )
    else:
        rows.append("  activation       none")
    if sampling.bias_rules or sampling.bias_groups:
        rows.append(
            f"  biases           {len(sampling.bias_rules)} rules, "
            f"{len(sampling.bias_groups)} groups"
        )
    else:
        rows.append("  biases           none")
    if sampling.token_preference_vector or sampling.token_preference_fast_vector:
        rows.append(
            f"  token preference slow={len(sampling.token_preference_vector)} "
            f"fast={len(sampling.token_preference_fast_vector)}"
        )
    else:
        rows.append("  token preference none")
    rows.extend(
        (
            "",
            "LEARNERS",
            f"  group learning   {_setting_value(plan.online_learning)}  "
            f"rate={plan.learning_rate:g} gate={plan.learning_gate}",
            f"  token preference {_setting_value(plan.token_preference)}  "
            f"dimension={plan.token_preference_dimension} "
            f"strength={plan.token_preference_strength:g} "
            f"scheme={plan.token_preference_learning_scheme or 'sgd-v1'}",
            "  Details: learning|group and preference",
            "",
        )
    )
    rows.extend(build_controller_stack(
        plan=plan,
        sampling=sampling,
        provenance=provenance,
    ).render().splitlines())
    rows.extend(
        (
            "",
            "VALIDATION",
            "  backend and sampler: ready",
        )
    )
    rows.extend(f"  {line}" for line in validated_artifacts)
    rows.extend(("", "Type go to create/start the runtime, or q to cancel."))
    return "\n".join(rows)


def _set_source(plan: RuntimePlan, kind: str, value: str, *, at: int | None = None) -> None:
    if kind not in {"new", "replay", "resume", "fork"}:
        raise EditorError("source must be new, replay, resume, or fork")
    _clear_source(plan)
    if kind == "new":
        if not value:
            raise EditorError("source new requires prompt text")
        _set(plan, "new_prompt", value)
        return
    field = {"replay": "replay", "resume": "resume", "fork": "fork_from"}[kind]
    _set(plan, field, _episode_reference(value))
    if kind == "fork":
        if at is not None and at < 0:
            raise EditorError("fork boundary must be nonnegative")
        _set(plan, "at", at)


def _parse_source(plan: RuntimePlan, words: list[str]) -> None:
    if len(words) < 2:
        raise EditorError("use source new|replay|resume|fork ID [at N]")
    kind = words[1].lower()
    if kind == "new":
        _set_source(plan, "new", " ".join(words[2:]))
        return
    if kind not in {"replay", "resume", "fork"} or len(words) < 3:
        raise EditorError("use source new|replay|resume|fork ID [at N]")
    if len(words) > 5 or (len(words) == 5 and words[3].lower() != "at"):
        raise EditorError("use source fork ID [at N]")
    boundary = None
    if len(words) == 5:
        try:
            boundary = int(words[4])
        except ValueError as exc:
            raise EditorError("fork boundary must be an integer") from exc
    _set_source(plan, kind, words[2], at=boundary)


def _parse_sampler(plan: RuntimePlan, words: list[str]) -> None:
    if len(words) < 2:
        raise EditorError("sampler changes use key=value (for example top_k=40)")
    for piece in words[1:]:
        if "=" not in piece:
            raise EditorError("sampler changes use key=value")
        raw_key, raw_value = piece.split("=", 1)
        key = raw_key.strip().lower()
        field = _SAMPLER_ALIASES.get(key, key.replace("-", "_"))
        kind = _SAMPLER_FIELDS.get(key) or _SAMPLER_FIELDS.get(field)
        if kind is None or field not in {
            "temperature", "top_k", "top_p", "min_p", "repeat_penalty",
            "repeat_last_n", "presence_penalty", "frequency_penalty",
        }:
            raise EditorError(f"unknown sampler field {raw_key!r}")
        try:
            value = kind(raw_value)
        except ValueError as exc:
            raise EditorError(f"invalid sampler value for {field}: {raw_value!r}") from exc
        _set(plan, field, value)


def episode_summary(store: Any, requested_id: str) -> str:
    """Return a compact, read-only workspace description for one episode."""

    reference = _episode_reference(requested_id)
    episode_id = store.resolve_id(reference)
    episode = store.get_episode(episode_id)
    visible = [
        row for row in store.tokens(episode_id)
        if bool(row["realized_visible"])
    ]
    parent = episode["parent_episode_id"]
    parent_text = (
        f"{store.label(parent)} at boundary {episode['fork_boundary']}"
        if parent else "none"
    )
    return "\n".join(
        (
            f"EPISODE {store.label(episode_id)}",
            f"  status: {episode['status']}",
            f"  tokens: {len(visible)}",
            f"  parent: {parent_text}",
            f"  actions: {len(store.actions(episode_id))}",
            f"  initial text: {str(episode['initial_text']).replace(chr(10), ' ')[:72]}",
            "",
            "Use one of:",
            f"  source replay {reference}",
            f"  source resume {reference}",
            f"  source fork {reference} at N",
            f"  fm {reference}",
        )
    )


def _selected_episode(plan: RuntimePlan) -> str | None:
    for field in ("replay", "resume", "fork_from"):
        value = getattr(plan, field)
        if value is not None:
            return str(value)
    return None


def apply_setup_command(
    raw: str,
    plan: RuntimePlan,
    *,
    store: Any | None = None,
) -> str:
    """Apply one setup command and return ``continue``, ``go``, or ``quit``."""

    try:
        words = shlex.split(raw)
    except ValueError as exc:
        raise EditorError(f"invalid setup command: {exc}") from exc
    if not words:
        return "continue"
    command = words[0].lower()
    if command in {"go", "start", "run"}:
        return "go"
    if command in {"q", "quit", "cancel"}:
        return "quit"
    if command in {"show", "status", "help", "?"}:
        return "show"
    if command in {"controllers", "controller", "stack"}:
        if len(words) != 1:
            raise EditorError(f"{command} does not take arguments")
        return "show-controllers"
    if command in {"ls", "episodes"}:
        return "list"
    if command == "workspace":
        if len(words) == 1:
            return "workspace"
        if len(words) != 2:
            raise EditorError("workspace requires a database path")
        selected = _path(words[1], label="workspace")
        if store is not None:
            switch = getattr(store, "switch_workspace", None)
            if not callable(switch):
                raise EditorError("the current launcher cannot switch workspaces")
            switch(selected)
        _clear_source(plan)
        _set(plan, "workspace", selected)
        return "workspace"
    if command in {"prompt", "new"}:
        _set_source(plan, "new", " ".join(words[1:]))
        return "continue"
    if command in {"prompt-file", "new-file"}:
        if len(words) != 2:
            raise EditorError("prompt-file requires a path")
        _clear_source(plan)
        _set(plan, "new_prompt_file", _path(words[1], label="prompt-file"))
        return "continue"
    if command == "source":
        _parse_source(plan, words)
        if store is not None and plan.replay is not None:
            store.resolve_id(plan.replay)
        if store is not None and plan.resume is not None:
            store.resolve_id(plan.resume)
        if store is not None and plan.fork_from is not None:
            store.resolve_id(plan.fork_from)
        return "continue"
    if command in _PATH_FIELDS:
        if len(words) != 2:
            raise EditorError(f"{command} requires a path")
        _set(plan, _PATH_FIELDS[command], _path(words[1], label=command))
        return "continue"
    if command == "backend":
        if len(words) != 2 or words[1].lower() not in BACKEND_NAMES:
            raise EditorError("backend must be llama.cpp or transformers")
        _set(plan, "backend", words[1].lower())
        return "continue"
    if command == "budget":
        if len(words) != 2:
            raise EditorError("budget requires a positive integer or off")
        if words[1].lower() in {"off", "none", "unlimited"}:
            _set(plan, "max_tokens", None)
        else:
            try:
                value = int(words[1])
            except ValueError as exc:
                raise EditorError("budget must be a positive integer or off") from exc
            if value < 1:
                raise EditorError("budget must be at least 1")
            _set(plan, "max_tokens", value)
        return "continue"
    if command == "seed":
        if len(words) != 2:
            raise EditorError("seed requires an integer, random, or default")
        value = words[1].lower()
        if value == "random":
            _set(plan, "seed", None)
            _set(plan, "random_seed", True)
        elif value in {"default", "none", "off"}:
            _set(plan, "seed", None)
            _set(plan, "random_seed", False)
        else:
            try:
                parsed = int(words[1])
            except ValueError as exc:
                raise EditorError("seed must be an integer, random, or default") from exc
            _set(plan, "seed", parsed)
            _set(plan, "random_seed", False)
        return "continue"
    if command in {"sampler", "s"}:
        if len(words) == 1:
            return "show-sampler"
        _parse_sampler(plan, words)
        return "continue"
    if command in {"learning", "group", "preference"}:
        if len(words) == 1:
            return "show-preference" if command == "preference" else "show-learning"
        preference = command == "preference"
        enabled_field = "token_preference" if preference else "online_learning"
        if words[1].lower() in {"on", "off"}:
            _set(plan, enabled_field, words[1].lower() == "on")
            if len(words) == 2:
                return "continue"
            words = [words[0], *words[2:]]
        _parse_control_settings(plan, words, preference=preference)
        return "continue"
    raise EditorError(f"unknown setup command {command!r}; use show for help")


def run_runtime_setup_menu(io: Any, args: Namespace, *, store: Any | None = None) -> bool:
    """Run the plain command setup surface; return false when cancelled."""

    plan = RuntimePlan.from_args(args)
    io.write(setup_summary(plan))
    while True:
        raw = io.read("Setup › ")
        if raw is None:
            return False
        try:
            words = shlex.split(raw)
        except ValueError:
            words = []
        if len(words) == 1 and re.fullmatch(r"#?\d+", words[0]):
            if store is None:
                io.write("No episode workspace is available.")
            else:
                try:
                    io.page(episode_summary(store, words[0]))
                except EditorError as exc:
                    io.write(str(exc))
            continue
        if words and words[0].lower() == "fm":
            if len(words) > 2:
                io.write("Use fm or fm #N.")
                continue
            target = words[1] if len(words) == 2 else _selected_episode(plan)
            if target is None:
                io.write("fm requires an episode number or a selected replay source.")
                continue
            if store is None:
                io.write("No episode workspace is available.")
                continue
            try:
                from .episode_projector import project_fork_map

                io.page(project_fork_map(store, store.resolve_id(_episode_reference(target))))
            except EditorError as exc:
                io.write(str(exc))
            continue
        if words and words[0].lower() in {"prompt", "new"} and len(words) == 1:
            prompt_text = io.read("Prompt › ")
            if prompt_text is None:
                return False
            raw = f"prompt {shlex.quote(prompt_text)}"
        try:
            result = apply_setup_command(raw, plan, store=store)
        except EditorError as exc:
            io.write(f"[invalid setup] {exc}")
            continue
        if result == "quit":
            io.write("Setup cancelled.")
            return False
        if result == "show":
            io.write(setup_summary(plan))
            continue
        if result == "show-sampler":
            io.page(sampler_summary(plan))
            continue
        if result == "show-learning":
            io.page(learning_summary(plan))
            continue
        if result == "show-preference":
            io.page(preference_summary(plan))
            continue
        if result == "show-controllers":
            io.page(build_controller_stack(plan=plan).render())
            continue
        if result == "list":
            if store is None:
                io.write("No episode workspace is available.")
            else:
                io.page(store.workspace_list(include_finished=True))
            continue
        if result == "workspace":
            current = plan.workspace
            io.write(f"WORKSPACE\n  path: {current}")
            if store is not None:
                io.page(store.workspace_list(include_finished=True))
            continue
        if result == "go":
            if not plan.has_source():
                io.write("[invalid setup] choose a prompt or source before go")
                continue
            plan.apply_to_args(args)
            return True
        if result == "continue":
            io.write(setup_summary(plan))

"""Interactive pre-runtime setup for the core editor.

The setup menu is deliberately a thin launcher front end. It owns launch
context, core sampler choices, and the path to an optional steering artifact;
it does not model learner, preference, or reference-prior state.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .backend_factory import BACKEND_NAMES
from .controller_stack import build_controller_stack
from .core.errors import EditorError
from .core.sampler_config import SamplerConfig


RUNTIME_PLAN_FORMAT = "spe-runtime-plan-v1"
CONTROLLER_PROFILE_FORMAT = "spe-controller-profile-v1"

_SOURCE_FIELDS = ("new_prompt", "new_prompt_file", "replay", "resume", "fork_from")
_SAMPLER_FIELDS = (
    "temperature", "top_k", "top_p", "min_p", "typical_p", "tail_free_z",
    "draw_kernel", "cfg_unconditional_prompt", "cfg_scale", "cfg_prefix_tokens",
    "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty",
    "seed", "random_seed",
)
_PLAN_FIELDS = (
    *_SOURCE_FIELDS, "at", "workspace", "model", "backend", "max_tokens",
    "activation_vector", *_SAMPLER_FIELDS, "explicit_options",
)
_PATH_FIELDS = {"workspace", "model", "activation_vector"}
_SAMPLER_ALIASES = {
    "temp": "temperature", "top-p": "top_p", "min-p": "min_p",
    "typical-p": "typical_p", "tail-free-z": "tail_free_z",
    "draw-kernel": "draw_kernel", "cfg-unconditional-prompt": "cfg_unconditional_prompt",
    "cfg-scale": "cfg_scale", "cfg-prefix-tokens": "cfg_prefix_tokens",
    "repeat-penalty": "repeat_penalty", "repeat-last-n": "repeat_last_n",
    "presence-penalty": "presence_penalty", "frequency-penalty": "frequency_penalty",
}
_SAMPLER_TYPES = {
    "temperature": float, "top_k": int, "top_p": float, "min_p": float,
    "typical_p": float, "tail_free_z": float, "draw_kernel": str,
    "cfg_unconditional_prompt": str, "cfg_scale": float, "cfg_prefix_tokens": int,
    "repeat_penalty": float, "repeat_last_n": int, "presence_penalty": float,
    "frequency_penalty": float,
}


@dataclass
class RuntimePlan:
    """Typed launch state assembled before the episode engine is created."""

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
    activation_vector: Path | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    typical_p: float | None = None
    tail_free_z: float | None = None
    draw_kernel: str | None = None
    cfg_unconditional_prompt: str | None = None
    cfg_scale: float | None = None
    cfg_prefix_tokens: int | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    random_seed: bool = False
    explicit_options: set[str] = field(default_factory=set)

    @classmethod
    def from_args(cls, args: Namespace) -> "RuntimePlan":
        values = {name: getattr(args, name, None) for name in _PLAN_FIELDS if name != "explicit_options"}
        values["explicit_options"] = set(getattr(args, "_explicit_options", ()))
        if values["workspace"] is None:
            values["workspace"] = Path("episodes.sqlite3")
        return cls(**values)

    def update(self, name: str, value: Any, *, explicit: bool = True) -> None:
        if name not in _PLAN_FIELDS or name == "explicit_options":
            raise AttributeError(f"unknown runtime plan field {name!r}")
        setattr(self, name, value)
        if explicit:
            self.explicit_options.add(name)

    def has_source(self) -> bool:
        return any(getattr(self, name) is not None for name in _SOURCE_FIELDS)

    def apply_to_args(self, args: Namespace) -> None:
        for name in _PLAN_FIELDS:
            if name != "explicit_options":
                setattr(args, name, getattr(self, name))
        args._explicit_options = set(self.explicit_options)

    def to_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name in _PLAN_FIELDS:
            if name == "explicit_options":
                continue
            value = getattr(self, name)
            values[name] = str(value) if name in _PATH_FIELDS and value is not None else value
        return {"format": RUNTIME_PLAN_FORMAT, "values": values,
                "explicit_options": sorted(self.explicit_options)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RuntimePlan":
        if value.get("format") != RUNTIME_PLAN_FORMAT:
            raise EditorError(
                f"unsupported runtime plan format {value.get('format')!r}; expected {RUNTIME_PLAN_FORMAT}"
            )
        raw_values = value.get("values")
        if not isinstance(raw_values, dict):
            raise EditorError("runtime plan profile is missing its values")
        allowed = set(_PLAN_FIELDS) - {"explicit_options"}
        unknown = set(raw_values) - allowed
        if unknown:
            raise EditorError("runtime plan profile has unknown fields: " + ", ".join(sorted(unknown)))
        defaults = cls()
        values = {name: raw_values.get(name, getattr(defaults, name)) for name in allowed}
        for name in _PATH_FIELDS:
            if values[name] is not None:
                values[name] = Path(values[name])
        explicit = value.get("explicit_options", ())
        if not isinstance(explicit, (list, tuple, set)):
            raise EditorError("runtime plan profile explicit_options must be a list")
        values["explicit_options"] = set(explicit)
        return cls(**values)


def _set(plan: RuntimePlan, name: str, value: Any, *, explicit: bool = True) -> None:
    plan.update(name, value, explicit=explicit)


def _path(value: str, *, label: str) -> Path:
    if not value:
        raise EditorError(f"{label} requires a path")
    return Path(value).expanduser()


def _episode_reference(value: str) -> str:
    value = value.strip()
    return f"#{value}" if value.isdigit() else value


def _clear_source(plan: RuntimePlan) -> None:
    plan.explicit_options.difference_update((*_SOURCE_FIELDS, "at"))
    for name in _SOURCE_FIELDS:
        _set(plan, name, None, explicit=False)
    _set(plan, "at", None, explicit=False)


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
        field_name = _SAMPLER_ALIASES.get(key, key.replace("-", "_"))
        kind = _SAMPLER_TYPES.get(field_name)
        if kind is None:
            raise EditorError(f"unknown sampler field {raw_key!r}")
        try:
            if field_name == "draw_kernel":
                if raw_value not in {"categorical", "gumbel-max"}:
                    raise ValueError
                parsed: Any = raw_value
            else:
                parsed = kind(raw_value)
        except ValueError as exc:
            raise EditorError(f"invalid sampler value for {field_name}: {raw_value!r}") from exc
        _set(plan, field_name, parsed)


def _source_summary(plan: RuntimePlan) -> str:
    if plan.new_prompt is not None:
        text = str(plan.new_prompt).replace("\n", " ")
        return f"new prompt: {text[:58]}{'…' if len(text) > 58 else ''}"
    if plan.new_prompt_file is not None:
        return f"new prompt file: {plan.new_prompt_file}"
    for field, label in (("replay", "replay"), ("resume", "resume"), ("fork_from", "fork")):
        value = getattr(plan, field)
        if value is not None:
            suffix = f" at {plan.at}" if field == "fork_from" and plan.at is not None else ""
            return f"{label}: {value}{suffix}"
    return "not selected"


def _display_path(value: Any) -> str:
    return str(value) if value is not None else "none"


def setup_summary(plan: RuntimePlan) -> str:
    sampler = " ".join(
        f"{name}={getattr(plan, name)}"
        for name in ("temperature", "top_k", "top_p", "min_p")
        if getattr(plan, name) is not None
    ) or "defaults"
    return "\n".join((
        "PRE-RUNTIME SETUP", "────────────────────────────────────────",
        "Choose a source, adjust the sampler, then type `go`.", "",
        "SESSION", f"  Workspace  {plan.workspace}", f"  Source     {_source_summary(plan)}",
        f"  Backend    {plan.backend or 'auto'}", f"  Model      {_display_path(plan.model)}",
        f"  Profile    {controller_profile_fingerprint(plan)[:12]}", "",
        "CONTROLS", f"  vector     {_display_path(plan.activation_vector)}", "",
        "SAMPLING", f"  sampler    {sampler}",
        f"  seed       {plan.seed if plan.seed is not None else 'default'}", "",
        "CORE COMMANDS",
        "  workspace [PATH]            show or switch episode workspace",
        "  ls                          list episodes in this workspace",
        "  prompt [TEXT]               use a new prompt",
        "  prompt-file PATH            read a prompt from a file",
        "  source replay|resume|fork ID [at N]",
        "  model PATH                  choose a model",
        "  backend llama.cpp|transformers",
        "  vector|steering PATH        load a steering vector",
        "  sampler key=value [...]     change sampler settings",
        "  seed N|random|default", "",
        "EPISODE VIEWS", "  #N or N                     inspect an episode",
        "  fm [#N]                     show its fork map", "  show                        redraw this summary", "",
        "PROFILES & FINISH", "  profile print               print reusable sampler/vector YAML",
        "  profile save PATH           save reusable sampler/vector YAML",
        "  profile load PATH           load reusable sampler/vector YAML",
        "  go                          validate and start", "  quit                        cancel setup",
    ))


def sampler_summary(plan: RuntimePlan) -> str:
    defaults = SamplerConfig()
    source_selected = any(getattr(plan, name) is not None for name in ("replay", "resume", "fork_from"))
    rows = ["SAMPLER SETTINGS", ""]
    for name in (
        "temperature", "top_k", "top_p", "min_p", "typical_p", "tail_free_z",
        "draw_kernel", "cfg_unconditional_prompt", "cfg_scale", "cfg_prefix_tokens",
        "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty",
    ):
        selected = getattr(plan, name)
        value = selected if selected is not None else (
            "inherited from source" if source_selected else getattr(defaults, name)
        )
        rows.append(f"{name:<21} {value}")
    seed = plan.seed if plan.seed is not None else (
        "random" if plan.random_seed else "inherited from source" if source_selected else defaults.seed
    )
    rows.extend((f"{'seed':<21} {seed}", "", "Change with: sampler key=value [...]"))
    return "\n".join(rows)


def effective_plan_summary(
    plan: RuntimePlan,
    sampling: SamplerConfig,
    *,
    source_sampling: SamplerConfig | None = None,
    provenance: dict[str, Any] | None = None,
    validated_artifacts: tuple[str, ...] = (),
) -> str:
    provenance = provenance or {}
    source_selected = source_sampling is not None
    rows = [
        "RUNTIME PREFLIGHT", "────────────────────────────────────────",
        f"Workspace    {plan.workspace}", f"Source       {_source_summary(plan)}",
        f"Backend      {provenance.get('backend') or plan.backend or 'auto'}",
        f"Model        {provenance.get('model_path') or _display_path(plan.model)}", "", "SAMPLER",
    ]
    for name in _SAMPLER_FIELDS:
        if name == "random_seed":
            continue
        value = getattr(sampling, name)
        origin = "override" if name in plan.explicit_options or getattr(plan, name, None) is not None else (
            "inherited" if source_selected else "default"
        )
        rows.append(f"  {name:<22} {value}  [{origin}]")
    rows.extend(("", "STEERING"))
    if sampling.activation_vector_digest:
        target = "output-head" if sampling.activation_vector_layer == "output" else "hidden-state"
        rows.append(f"  vector         {sampling.activation_vector_digest[:12]} target={target} strength={sampling.activation_vector_strength:g}")
    else:
        rows.append("  vector         none")
    rows.append(
        f"  biases         {len(sampling.bias_rules)} rules, {len(sampling.bias_groups)} groups"
        if sampling.bias_rules or sampling.bias_groups else "  biases         none"
    )
    rows.extend(("", "CONTROLLER STACK"))
    rows.extend(build_controller_stack(plan=plan, sampling=sampling, provenance=provenance).render().splitlines()[2:])
    rows.extend(("", "VALIDATION", "  backend and sampler: ready"))
    rows.extend(f"  {line}" for line in validated_artifacts)
    rows.extend(("", "Type go to create/start the runtime, or q to cancel."))
    return "\n".join(rows)


class _DuplicateKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _DuplicateKeyLoader, node: yaml.MappingNode, deep: bool = False):
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise EditorError(f"duplicate key {key!r} in controller profile")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_DuplicateKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)

_PROFILE_GROUPS = {"sampler": _SAMPLER_FIELDS, "steering": ("activation_vector",)}
_PROFILE_PUBLIC = {"activation_vector": "vector"}
_PROFILE_INTERNAL = {value: key for key, value in _PROFILE_PUBLIC.items()}


def _profile_controllers(plan: RuntimePlan) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for group, fields in _PROFILE_GROUPS.items():
        result[group] = {}
        for field_name in fields:
            key = _PROFILE_PUBLIC.get(field_name, field_name)
            value = getattr(plan, field_name)
            result[group][key] = str(value) if field_name in _PATH_FIELDS and value is not None else value
    return result


def _profile_payload(plan: RuntimePlan) -> dict[str, Any]:
    return {"format": CONTROLLER_PROFILE_FORMAT, "controllers": _profile_controllers(plan)}


def controller_profile_fingerprint(plan: RuntimePlan) -> str:
    encoded = json.dumps(_profile_payload(plan), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def controller_profile_json(plan: RuntimePlan) -> str:
    payload = _profile_payload(plan)
    payload["fingerprint"] = controller_profile_fingerprint(plan)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def controller_profile_yaml(plan: RuntimePlan) -> str:
    payload = _profile_payload(plan)
    payload["fingerprint"] = controller_profile_fingerprint(plan)
    return yaml.safe_dump(payload, sort_keys=False)


def _validate_profile_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise EditorError("controller profile must be a mapping")
    allowed_top = {"format", "controllers", "fingerprint"}
    unknown_top = set(payload) - allowed_top
    if unknown_top:
        raise EditorError("controller profile has unknown top-level fields: " + ", ".join(sorted(unknown_top)))
    if payload.get("format") != CONTROLLER_PROFILE_FORMAT:
        raise EditorError(
            f"unsupported controller profile format {payload.get('format')!r}; expected {CONTROLLER_PROFILE_FORMAT}"
        )
    controllers = payload.get("controllers")
    if not isinstance(controllers, dict):
        raise EditorError("controller profile controllers must be a mapping")
    unknown_groups = set(controllers) - set(_PROFILE_GROUPS)
    if unknown_groups:
        raise EditorError("controller profile has unknown groups: " + ", ".join(sorted(unknown_groups)))
    for group, values in controllers.items():
        if not isinstance(values, dict):
            raise EditorError(f"controller profile group {group!r} must be a mapping")
        allowed = {_PROFILE_PUBLIC.get(name, name) for name in _PROFILE_GROUPS[group]}
        unknown = set(values) - allowed
        if unknown:
            raise EditorError(f"controller profile group {group!r} has unknown fields: " + ", ".join(sorted(unknown)))
    sampler_values: dict[str, Any] = {}
    for public_name, value in controllers.get("sampler", {}).items():
        name = _PROFILE_INTERNAL.get(public_name, public_name)
        converted = _coerce_profile_value(name, value)
        if name not in {"random_seed", "seed"} and converted is not None:
            sampler_values[name] = converted
    try:
        SamplerConfig(**sampler_values)
    except (EditorError, TypeError) as exc:
        message = str(exc).replace("top_k must be at least 1", "top_k must be positive")
        raise EditorError(message) from exc
    fingerprint = payload.get("fingerprint")
    if not isinstance(fingerprint, str):
        raise EditorError("controller profile is missing its fingerprint")
    canonical = {"format": payload["format"], "controllers": controllers}
    expected = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if fingerprint != expected:
        raise EditorError("controller profile fingerprint mismatch")
    return controllers


def _coerce_profile_value(name: str, value: Any) -> Any:
    if name in _PATH_FIELDS:
        return Path(value).expanduser() if value is not None else None
    if name == "random_seed":
        if type(value) is not bool:
            raise EditorError(f"controller profile {name} must be a boolean")
        return value
    if name in _SAMPLER_TYPES and value is not None:
        kind = _SAMPLER_TYPES[name]
        if kind is int and type(value) is not int:
            raise EditorError(f"controller profile {name} must be an integer")
        if kind is float and type(value) not in {int, float}:
            raise EditorError(f"controller profile {name} must be a number")
        if kind is str and not isinstance(value, str):
            raise EditorError(f"controller profile {name} must be text")
    return value


def apply_controller_profile(plan: RuntimePlan, controllers: dict[str, Any]) -> str:
    """Apply a profile while preserving episode, workspace, and model context."""
    if not isinstance(controllers, dict):
        raise EditorError("controller profile controllers must be a mapping")
    updates: dict[str, Any] = {}
    for group, values in controllers.items():
        if group not in _PROFILE_GROUPS or not isinstance(values, dict):
            raise EditorError("invalid controller profile group")
        for public_name, value in values.items():
            name = _PROFILE_INTERNAL.get(public_name, public_name)
            if name not in _PROFILE_GROUPS[group]:
                raise EditorError(f"unknown controller profile field {public_name!r}")
            updates[name] = _coerce_profile_value(name, value)
    candidate = {
        name: getattr(plan, name)
        for name in _SAMPLER_FIELDS
        if name not in {"random_seed", "seed"}
    }
    candidate.update({name: value for name, value in updates.items() if name in candidate})
    candidate = {name: value for name, value in candidate.items() if value is not None}
    try:
        SamplerConfig(**candidate)
    except (EditorError, TypeError) as exc:
        message = str(exc).replace("top_k must be at least 1", "top_k must be positive")
        raise EditorError(message) from exc
    for name, value in updates.items():
        _set(plan, name, value)
    return controller_profile_fingerprint(plan)


def save_controller_profile(plan: RuntimePlan, path: Path | str) -> str:
    selected = Path(path)
    if selected.exists() and selected.is_dir():
        raise EditorError(f"controller profile path is a directory: {selected}")
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.write_text(controller_profile_yaml(plan), encoding="utf-8")
    return controller_profile_fingerprint(plan)


def load_controller_profile(path: Path | str) -> tuple[dict[str, Any], str]:
    selected = Path(path)
    try:
        text = selected.read_text(encoding="utf-8")
    except OSError as exc:
        raise EditorError(f"cannot read controller profile {selected}: {exc}") from exc
    try:
        payload = yaml.load(text, Loader=_DuplicateKeyLoader)
    except EditorError:
        raise
    except yaml.YAMLError as exc:
        raise EditorError(f"invalid controller profile YAML: {exc}") from exc
    controllers = _validate_profile_payload(payload)
    return controllers, str(payload["fingerprint"])


def _episode_summary(store: Any, requested_id: str) -> str:
    reference = _episode_reference(requested_id)
    episode_id = store.resolve_id(reference)
    episode = store.get_episode(episode_id)
    visible = [row for row in store.tokens(episode_id) if bool(row["realized_visible"])]
    parent = episode["parent_episode_id"]
    parent_text = f"{store.label(parent)} at boundary {episode['fork_boundary']}" if parent else "none"
    return "\n".join((
        f"EPISODE {store.label(episode_id)}", f"  status: {episode['status']}",
        f"  tokens: {len(visible)}", f"  parent: {parent_text}",
        f"  actions: {len(store.actions(episode_id))}",
        f"  initial text: {str(episode['initial_text']).replace(chr(10), ' ')[:72]}", "",
        "Use one of:", f"  source replay {reference}", f"  source resume {reference}",
        f"  source fork {reference} at N", f"  fm {reference}",
    ))


def _selected_episode(plan: RuntimePlan) -> str | None:
    for name in ("replay", "resume", "fork_from"):
        value = getattr(plan, name)
        if value is not None:
            return str(value)
    return None


def apply_setup_command(raw: str, plan: RuntimePlan, *, store: Any | None = None) -> str:
    """Apply one setup command and return a menu action."""
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
    if command == "profile":
        if len(words) == 1 or (len(words) == 2 and words[1].lower() == "print"):
            return "show-profile"
        if len(words) != 3 or words[1].lower() not in {"save", "load"}:
            raise EditorError("use profile print, profile save PATH, or profile load PATH")
        selected = _path(words[2], label="profile")
        if words[1].lower() == "save":
            save_controller_profile(plan, selected)
            return "profile-saved"
        payload, _ = load_controller_profile(selected)
        apply_controller_profile(plan, payload)
        return "profile-loaded"
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
        for name in ("replay", "resume", "fork_from"):
            if store is not None and getattr(plan, name) is not None:
                store.resolve_id(getattr(plan, name))
        return "continue"
    if command in {"vector", "steering"}:
        if len(words) != 2:
            raise EditorError(f"{command} requires a path")
        _set(plan, "activation_vector", _path(words[1], label=command))
        return "continue"
    if command == "model":
        if len(words) != 2:
            raise EditorError("model requires a path")
        _set(plan, "model", _path(words[1], label="model"))
        return "continue"
    if command == "backend":
        if len(words) != 2 or words[1].lower() not in BACKEND_NAMES:
            raise EditorError("backend must be llama.cpp or transformers")
        _set(plan, "backend", words[1].lower())
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
    raise EditorError(f"unknown setup command {command!r}; use show for help")


def run_runtime_setup_menu(io: Any, args: Namespace, *, store: Any | None = None) -> bool:
    """Run the command setup surface; return false when cancelled."""
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
                    io.page(_episode_summary(store, words[0]))
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
                from .projector import project_fork_map

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
        elif result == "show-sampler":
            io.page(sampler_summary(plan))
        elif result == "show-controllers":
            io.page(build_controller_stack(plan=plan).render())
        elif result == "show-profile":
            io.page(controller_profile_yaml(plan))
        elif result == "profile-saved":
            io.write("Controller profile saved.")
        elif result == "profile-loaded":
            io.write("Controller profile loaded; episode and workspace selection were preserved.")
            io.write(setup_summary(plan))
        elif result == "list":
            if store is None:
                io.write("No episode workspace is available.")
            else:
                io.page(store.workspace_list(include_finished=True))
        elif result == "workspace":
            io.write(f"WORKSPACE\n  path: {plan.workspace}")
            if store is not None:
                io.page(store.workspace_list(include_finished=True))
        elif result == "go":
            if not plan.has_source():
                io.write("[invalid setup] choose a prompt or source before go")
                continue
            plan.apply_to_args(args)
            return True
        elif result == "continue":
            io.write(setup_summary(plan))

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
from .domain import EditorError, SamplingConfig


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


_PLAN_FIELDS = (
    "new_prompt", "new_prompt_file", "replay", "resume", "fork_from", "at",
    "workspace", "model", "backend", "max_tokens", "biases", "groups", "reference",
    "activation_vector", "temperature", "top_k", "top_p", "min_p",
    "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty",
    "seed", "random_seed", "online_learning", "token_preference",
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
    token_preference: bool = False
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
            "  learning on|off             enable manual-group learning",
            "  preference on|off           enable token-preference learning",
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
    if command in {"learning", "preference"}:
        if len(words) != 2 or words[1].lower() not in {"on", "off"}:
            raise EditorError(f"{command} requires on or off")
        _set(plan, "online_learning" if command == "learning" else "token_preference", words[1].lower() == "on")
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

"""Interactive pre-runtime setup for a new editor session.

This module deliberately edits the launch namespace instead of constructing an
engine.  The existing CLI remains the authority for backend loading, replay,
fork, and sampler validation; the setup surface is a small interactive front
end that produces the same inputs.
"""

from __future__ import annotations

import shlex
from argparse import Namespace
from pathlib import Path
from typing import Any

from .backend_factory import BACKEND_NAMES
from .domain import EditorError


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


def _explicit(args: Namespace) -> set[str]:
    values = getattr(args, "_explicit_options", None)
    if values is None:
        values = set()
        args._explicit_options = values
    return values


def _set(args: Namespace, name: str, value: Any, *, explicit: bool = True) -> None:
    setattr(args, name, value)
    if explicit:
        _explicit(args).add(name)


def _clear_source(args: Namespace) -> None:
    for name in _SOURCE_FIELDS:
        _set(args, name, None)
    _set(args, "at", None)


def _path(value: str, *, label: str) -> Path:
    if not value:
        raise EditorError(f"{label} requires a path")
    return Path(value).expanduser()


def _source_summary(args: Namespace) -> str:
    if args.new_prompt is not None:
        text = str(args.new_prompt).replace("\n", " ")
        return f"new prompt: {text[:58]}{'…' if len(text) > 58 else ''}"
    if args.new_prompt_file is not None:
        return f"new prompt file: {args.new_prompt_file}"
    for field, label in (
        ("replay", "replay"),
        ("resume", "resume"),
        ("fork_from", "fork"),
    ):
        value = getattr(args, field, None)
        if value is not None:
            suffix = f" at {args.at}" if field == "fork_from" and args.at is not None else ""
            return f"{label}: {value}{suffix}"
    return "not selected"


def _display_path(value: Any) -> str:
    return str(value) if value is not None else "none"


def setup_summary(args: Namespace) -> str:
    """Return the compact state shown by the pre-runtime menu."""

    sampler = " ".join(
        f"{name}={getattr(args, name)}"
        for name in ("temperature", "top_k", "top_p", "min_p")
        if getattr(args, name, None) is not None
    ) or "defaults"
    return "\n".join(
        (
            "PRE-RUNTIME SETUP",
            "────────────────────────────────────────",
            f"Source       {_source_summary(args)}",
            f"Backend      {getattr(args, 'backend', None) or 'auto'}",
            f"Model        {_display_path(getattr(args, 'model', None))}",
            f"Budget       {getattr(args, 'max_tokens', None) or 'unlimited'}",
            "",
            "Controls",
            f"  biases     {_display_path(getattr(args, 'biases', None))}",
            f"  groups     {_display_path(getattr(args, 'groups', None))}",
            f"  reference  {_display_path(getattr(args, 'reference', None))}",
            f"  activation {_display_path(getattr(args, 'activation_vector', None))}",
            "",
            f"Sampler      {sampler}",
            f"Seed         {getattr(args, 'seed', None) if getattr(args, 'seed', None) is not None else 'default'}",
            "",
            "Commands",
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
            "  show                        redraw this summary",
            "  go                          validate and start",
            "  quit                        cancel setup",
        )
    )


def _set_source(args: Namespace, kind: str, value: str, *, at: int | None = None) -> None:
    if kind not in {"new", "replay", "resume", "fork"}:
        raise EditorError("source must be new, replay, resume, or fork")
    _clear_source(args)
    if kind == "new":
        if not value:
            raise EditorError("source new requires prompt text")
        _set(args, "new_prompt", value)
        return
    field = {"replay": "replay", "resume": "resume", "fork": "fork_from"}[kind]
    _set(args, field, value)
    if kind == "fork":
        if at is not None and at < 0:
            raise EditorError("fork boundary must be nonnegative")
        _set(args, "at", at)


def _parse_source(args: Namespace, words: list[str]) -> None:
    if len(words) < 2:
        raise EditorError("use source new|replay|resume|fork ID [at N]")
    kind = words[1].lower()
    if kind == "new":
        _set_source(args, "new", " ".join(words[2:]))
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
    _set_source(args, kind, words[2], at=boundary)


def _parse_sampler(args: Namespace, words: list[str]) -> None:
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
        _set(args, field, value)


def apply_setup_command(
    raw: str,
    args: Namespace,
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
    if command in {"prompt", "new"}:
        _set_source(args, "new", " ".join(words[1:]))
        return "continue"
    if command in {"prompt-file", "new-file"}:
        if len(words) != 2:
            raise EditorError("prompt-file requires a path")
        _clear_source(args)
        _set(args, "new_prompt_file", _path(words[1], label="prompt-file"))
        return "continue"
    if command == "source":
        _parse_source(args, words)
        if store is not None and getattr(args, "replay", None) is not None:
            store.resolve_id(args.replay)
        if store is not None and getattr(args, "resume", None) is not None:
            store.resolve_id(args.resume)
        if store is not None and getattr(args, "fork_from", None) is not None:
            store.resolve_id(args.fork_from)
        return "continue"
    if command in _PATH_FIELDS:
        if len(words) != 2:
            raise EditorError(f"{command} requires a path")
        _set(args, _PATH_FIELDS[command], _path(words[1], label=command))
        return "continue"
    if command == "backend":
        if len(words) != 2 or words[1].lower() not in BACKEND_NAMES:
            raise EditorError("backend must be llama.cpp or transformers")
        _set(args, "backend", words[1].lower())
        return "continue"
    if command == "budget":
        if len(words) != 2:
            raise EditorError("budget requires a positive integer or off")
        if words[1].lower() in {"off", "none", "unlimited"}:
            _set(args, "max_tokens", None)
        else:
            try:
                value = int(words[1])
            except ValueError as exc:
                raise EditorError("budget must be a positive integer or off") from exc
            if value < 1:
                raise EditorError("budget must be at least 1")
            _set(args, "max_tokens", value)
        return "continue"
    if command == "seed":
        if len(words) != 2:
            raise EditorError("seed requires an integer, random, or default")
        value = words[1].lower()
        if value == "random":
            _set(args, "seed", None)
            _set(args, "random_seed", True)
        elif value in {"default", "none", "off"}:
            _set(args, "seed", None)
            _set(args, "random_seed", False)
        else:
            try:
                parsed = int(words[1])
            except ValueError as exc:
                raise EditorError("seed must be an integer, random, or default") from exc
            _set(args, "seed", parsed)
            _set(args, "random_seed", False)
        return "continue"
    if command in {"sampler", "s"}:
        _parse_sampler(args, words)
        return "continue"
    if command in {"learning", "preference"}:
        if len(words) != 2 or words[1].lower() not in {"on", "off"}:
            raise EditorError(f"{command} requires on or off")
        _set(args, "online_learning" if command == "learning" else "token_preference", words[1].lower() == "on")
        return "continue"
    raise EditorError(f"unknown setup command {command!r}; use show for help")


def run_runtime_setup_menu(io: Any, args: Namespace, *, store: Any | None = None) -> bool:
    """Run the plain command setup surface; return false when cancelled."""

    io.write(setup_summary(args))
    while True:
        raw = io.read("Setup › ")
        if raw is None:
            return False
        try:
            words = shlex.split(raw)
        except ValueError:
            words = []
        if words and words[0].lower() in {"prompt", "new"} and len(words) == 1:
            prompt_text = io.read("Prompt › ")
            if prompt_text is None:
                return False
            raw = f"prompt {shlex.quote(prompt_text)}"
        try:
            result = apply_setup_command(raw, args, store=store)
        except EditorError as exc:
            io.write(f"[invalid setup] {exc}")
            continue
        if result == "quit":
            io.write("Setup cancelled.")
            return False
        if result == "show":
            io.write(setup_summary(args))
            continue
        if result == "list":
            if store is None:
                io.write("No episode workspace is available.")
            else:
                io.page(store.workspace_list(include_finished=True))
            continue
        if result == "go":
            if not any(getattr(args, field, None) is not None for field in _SOURCE_FIELDS):
                io.write("[invalid setup] choose a prompt or source before go")
                continue
            return True

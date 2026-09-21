"""Storage-neutral controller profiles for the command-line runtime.

Profiles are mappings from CLI option names to values.  YAML is the first
wire format, but the validated representation is just a mapping of visible
CLI option names, so another file format can be added without changing the
launcher or runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from .core.errors import EditorError
from .core.sampler_config import SamplerConfig


PROFILE_FORMAT = "spe-controller-profile-v2"
_PROFILE_METADATA = {"format", "fingerprint", "values"}
_UNPROFILEABLE_DESTS = {"help", "profile", "version"}


class _DuplicateKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _DuplicateKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise EditorError(f"duplicate key {key!r} in controller profile")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _actions(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    return [
        action
        for action in parser._actions
        if action.option_strings and action.dest != argparse.SUPPRESS
    ]


def _option_actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    result: dict[str, argparse.Action] = {}
    for action in _actions(parser):
        for option in action.option_strings:
            result[option] = action
    return result


def _dest_actions(parser: argparse.ArgumentParser) -> dict[str, list[argparse.Action]]:
    result: dict[str, list[argparse.Action]] = {}
    for action in _actions(parser):
        result.setdefault(action.dest, []).append(action)
    return result


def _canonical_key(
    key: Any,
    parser: argparse.ArgumentParser,
) -> tuple[str, argparse.Action | None]:
    if not isinstance(key, str) or not key.strip():
        raise EditorError("controller profile option names must be non-empty text")
    raw = key.strip()
    option = raw if raw.startswith("--") else "--" + raw.replace("_", "-")
    option_action = _option_actions(parser).get(option)
    if option_action is not None:
        if option_action.dest in _UNPROFILEABLE_DESTS:
            raise EditorError(f"controller profile cannot set {option_action.dest!r}")
        return option_action.dest, option_action
    raise EditorError(f"controller profile has unknown option {key!r}")


def _profile_values(payload: Any) -> tuple[dict[str, Any], str | None]:
    if not isinstance(payload, dict):
        raise EditorError("controller profile must be a mapping")
    if "format" in payload and payload["format"] != PROFILE_FORMAT:
        raise EditorError(
            f"unsupported controller profile format {payload.get('format')!r}; "
            f"expected {PROFILE_FORMAT}"
        )

    if "values" in payload:
        unknown = set(payload) - _PROFILE_METADATA
        if unknown:
            raise EditorError(
                "controller profile has unknown top-level fields: "
                + ", ".join(sorted(str(value) for value in unknown))
            )
        values = payload["values"]
        if not isinstance(values, dict):
            raise EditorError("controller profile values must be a mapping")
    else:
        values = {
            key: value
            for key, value in payload.items()
            if key not in {"format", "fingerprint"}
        }

    fingerprint = payload.get("fingerprint")
    if fingerprint is not None and not isinstance(fingerprint, str):
        raise EditorError("controller profile fingerprint must be text")
    return dict(values), fingerprint


def _action_for_value(
    parser: argparse.ArgumentParser,
    dest: str,
    value: Any,
    requested_action: argparse.Action | None,
) -> argparse.Action:
    if requested_action is not None:
        return requested_action
    candidates = _dest_actions(parser).get(dest, [])
    if not candidates:
        raise EditorError(f"controller profile option {dest!r} is not a CLI option")

    if isinstance(value, bool):
        for action in candidates:
            if isinstance(action, argparse._StoreTrueAction) and value:
                return action
            if isinstance(action, argparse._StoreFalseAction) and not value:
                return action
        for action in candidates:
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                return action
    for action in candidates:
        if not isinstance(action, (argparse._StoreConstAction, argparse._StoreTrueAction, argparse._StoreFalseAction)):
            return action
    return candidates[0]


def _profile_option_name(action: argparse.Action) -> str:
    """Return the long CLI spelling represented by an argparse action."""

    option = next(
        (item for item in action.option_strings if item.startswith("--")),
        action.option_strings[0],
    )
    return option.removeprefix("--")


def _value_tokens(
    action: argparse.Action,
    value: Any,
    *,
    option_name: str,
) -> list[str]:
    option = action.option_strings[0]
    if isinstance(
        action,
        (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._StoreConstAction),
    ):
        if type(value) is not bool:
            raise EditorError(f"controller profile {option_name} must be a boolean")
        # A profile flag is represented by its visible option spelling.  The
        # value says whether that spelling is present, including for
        # --no-* actions whose argparse destination stores the inverse.
        return [option] if value else []

    nargs = action.nargs
    if nargs in (None, "?"):
        if isinstance(value, (dict, list, tuple, set)):
            raise EditorError(f"controller profile {option_name} must be a scalar")
        return [option] if nargs == "?" and value is None else [option, str(value)]

    if not isinstance(value, (list, tuple)):
        raise EditorError(f"controller profile {option_name} must be a list")
    if isinstance(nargs, int) and len(value) != nargs:
        raise EditorError(
            f"controller profile {option_name} requires {nargs} values"
        )
    if nargs == "+" and not value:
        raise EditorError(f"controller profile {option_name} must not be empty")
    return [option, *(str(item) for item in value)]


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_canonical_json_value(item) for item in value)
    return value


def _canonical_values(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _canonical_json_value(value)
        for key, value in sorted(values.items(), key=lambda item: str(item[0]))
    }


def controller_profile_fingerprint(values: Mapping[str, Any]) -> str:
    """Return a stable identity for validated CLI profile values."""

    payload = {"format": PROFILE_FORMAT, "values": _canonical_values(values)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_values(
    raw_values: Mapping[str, Any],
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    tokens: list[str] = []
    requested: dict[str, tuple[Any, argparse.Action, str]] = {}
    for raw_key, value in raw_values.items():
        dest, requested_action = _canonical_key(raw_key, parser)
        if dest in requested:
            raise EditorError(
                f"controller profile specifies {dest!r} more than once"
            )
        action = _action_for_value(parser, dest, value, requested_action)
        tokens.extend(_value_tokens(action, value, option_name=str(raw_key)))
        requested[dest] = (value, action, _profile_option_name(action))

    try:
        parsed = parser.parse_args(tokens)
    except SystemExit as exc:
        raise EditorError("invalid controller profile CLI value") from exc

    values: dict[str, Any] = {}
    semantic_values: dict[str, Any] = {}
    for dest, (raw_value, action, option_name) in requested.items():
        if isinstance(
            action,
            (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._StoreConstAction),
        ):
            values[option_name] = raw_value
            semantic_values[dest] = (
                raw_value
                if isinstance(action, argparse._StoreTrueAction)
                else (not raw_value if isinstance(action, argparse._StoreFalseAction) else action.const)
            )
        else:
            parsed_value = getattr(parsed, dest)
            values[option_name] = parsed_value
            semantic_values[dest] = parsed_value
        if values[option_name] is None and raw_value is not None:
            raise EditorError(
                f"controller profile option {option_name!r} could not be parsed"
            )
    sampler_fields = {
        field.name
        for field in fields(SamplerConfig)
        if field.name not in {"activation_vector", "activation_vector_strength"}
    }
    sampler_values = {
        name: value
        for name, value in semantic_values.items()
        if name in sampler_fields
    }
    try:
        SamplerConfig(**sampler_values)
    except (EditorError, TypeError) as exc:
        raise EditorError(str(exc)) from exc
    return values


def parse_controller_profile(
    payload: Any,
    parser: argparse.ArgumentParser,
) -> tuple[dict[str, Any], str]:
    """Validate a decoded profile against a CLI parser.

    The returned mapping uses visible CLI option names and parsed Python values;
    it is independent of the source format or persistence layer.
    """

    raw_values, supplied_fingerprint = _profile_values(payload)
    values = _validate_values(raw_values, parser)
    fingerprint = controller_profile_fingerprint(values)
    if supplied_fingerprint is not None and supplied_fingerprint != fingerprint:
        raise EditorError("controller profile fingerprint mismatch")
    return values, fingerprint


def load_controller_profile(
    path: Path | str,
    parser: argparse.ArgumentParser,
) -> tuple[dict[str, Any], str]:
    selected = Path(path).expanduser()
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
    return parse_controller_profile(payload, parser)


def controller_profile_yaml(values: Mapping[str, Any]) -> str:
    """Render validated values as a portable, human-editable YAML profile."""

    serialized = _canonical_values(values)
    payload: dict[str, Any] = {"format": PROFILE_FORMAT, **serialized}
    payload["fingerprint"] = controller_profile_fingerprint(values)
    return yaml.safe_dump(payload, sort_keys=False)


def explicit_option_dests(
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> set[str]:
    option_actions = _option_actions(parser)
    return {
        option_actions[token.split("=", 1)[0]].dest
        for token in argv
        if token.split("=", 1)[0] in option_actions
    }


def _conflicting_dests(
    parser: argparse.ArgumentParser,
    explicit: set[str],
) -> set[str]:
    blocked = set(explicit)
    for group in parser._mutually_exclusive_groups:
        destinations = {action.dest for action in group._group_actions}
        if destinations & explicit:
            blocked.update(destinations)
    return blocked


def profile_arguments(
    parser: argparse.ArgumentParser,
    values: Mapping[str, Any],
    *,
    overridden: set[str] = frozenset(),
) -> tuple[list[str], set[str]]:
    """Turn profile values into parser tokens, omitting CLI overrides."""

    blocked = _conflicting_dests(parser, overridden)
    tokens: list[str] = []
    applied: set[str] = set()
    for option_name, value in values.items():
        dest, requested_action = _canonical_key(option_name, parser)
        if dest in blocked:
            continue
        action = _action_for_value(parser, dest, value, requested_action)
        option_tokens = _value_tokens(action, value, option_name=option_name)
        tokens.extend(option_tokens)
        if option_tokens:
            applied.add(dest)
    return tokens, applied


__all__ = [
    "PROFILE_FORMAT",
    "controller_profile_fingerprint",
    "controller_profile_yaml",
    "explicit_option_dests",
    "load_controller_profile",
    "parse_controller_profile",
    "profile_arguments",
]

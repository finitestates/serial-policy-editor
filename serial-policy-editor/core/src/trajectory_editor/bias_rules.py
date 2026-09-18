"""Logical bias rules and tail matching.

The compiler produces routes; this module applies a runtime bias amount to the
final token of each route once its preceding token sequence is present. A rule
may contain several alternate routes. They are treated as one logical target,
so shared prefixes and shared next-token edges are only biased once.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .core.errors import EditorError


BIAS_MODES = ("tail",)
GROUP_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def _sequence(value: Any, *, path: str, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise EditorError(f"{path} must be a token-id sequence")
    result = tuple(value)
    if not result and not allow_empty:
        raise EditorError(f"{path} must not be empty")
    if any(type(token) is not int or token < 0 for token in result):
        raise EditorError(f"{path} must contain nonnegative integer token IDs")
    return result


def _routes(value: Any, *, path: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise EditorError(f"{path} must be a list of token-id sequences")
    result = []
    for index, route in enumerate(value):
        result.append(_sequence(route, path=f"{path}[{index}]"))
    if not result:
        raise EditorError(f"{path} must not be empty")
    return tuple(dict.fromkeys(result))


def _triggers(value: Any) -> tuple[tuple[int, ...], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise EditorError("bias rule triggers must be a list of token-id sequences")
    return tuple(dict.fromkeys(
        _sequence(trigger, path=f"triggers[{index}]")
        for index, trigger in enumerate(value)
    ))


@dataclass(frozen=True)
class BiasRule:
    """One logical bias amount applied to one or more token routes."""

    routes: tuple[tuple[int, ...], ...]
    bias: float
    mode: str = "tail"
    triggers: tuple[tuple[int, ...], ...] = ()
    until: int | None = None
    logical_target: str | None = None

    def __post_init__(self) -> None:
        routes = _routes(self.routes, path="bias rule routes")
        if self.mode not in BIAS_MODES:
            raise EditorError(
                f"bias rule mode must be one of {', '.join(BIAS_MODES)}"
            )
        if type(self.bias) not in (int, float) or not math.isfinite(self.bias):
            raise EditorError("bias rule amount must be finite")
        triggers = _triggers(self.triggers)
        until = self.until
        if until is not None and (type(until) is not int or until < 0):
            raise EditorError(
                "bias rule stop token must be a nonnegative integer or omitted"
            )
        if until is None and triggers:
            raise EditorError("bias rule triggers require a stop token or lifetime")
        if until is not None and not triggers:
            raise EditorError("bias rule scopes require at least one trigger")
        if self.logical_target is not None and (
            not isinstance(self.logical_target, str) or not self.logical_target
        ):
            raise EditorError("bias rule logical_target must be a nonempty string")
        routes = tuple(sorted(routes))
        object.__setattr__(self, "routes", routes)
        object.__setattr__(self, "triggers", tuple(sorted(triggers)))
        object.__setattr__(self, "bias", float(self.bias))

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.routes,
            self.mode,
            self.triggers,
            self.until,
            self.logical_target,
        )

    @property
    def sort_key(self) -> tuple[Any, ...]:
        lifetime = (0, self.until) if type(self.until) is int else (1, self.until)
        return (
            self.routes,
            self.mode,
            self.triggers,
            lifetime,
            self.logical_target,
        )

    @classmethod
    def from_record(cls, value: Any) -> "BiasRule":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise EditorError("bias rule must be an object")
        routes = value.get("routes")
        allowed = {
            "routes", "bias", "mode", "triggers", "until",
            "logical_target",
        }
        unknown = set(value) - allowed
        if unknown:
            raise EditorError(f"unknown bias rule fields: {', '.join(sorted(unknown))}")
        if "bias" not in value:
            raise EditorError("bias rule requires bias")
        mode = value.get("mode", "tail")
        return cls(
            routes=routes,
            bias=value["bias"],
            mode=mode,
            triggers=value.get("triggers", ()),
            until=value.get("until"),
            logical_target=value.get("logical_target"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "routes": [list(route) for route in self.routes],
            "mode": self.mode,
            "bias": self.bias,
        }
        if self.triggers:
            result["triggers"] = [list(trigger) for trigger in self.triggers]
        if self.until is not None:
            result["until"] = self.until
        if self.logical_target is not None:
            result["logical_target"] = self.logical_target
        return result


@dataclass(frozen=True)
class BiasGroup:
    """A durable named collection of bias-rule templates and one adjustment."""

    name: str
    rules: tuple[BiasRule, ...]
    bias: float = 0.0
    members: tuple[str, ...] = ()
    surfaces: tuple[str, ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not GROUP_NAME_RE.fullmatch(self.name):
            raise EditorError(
                "bias group names must begin with a letter or underscore and "
                "contain only letters, numbers, underscores, periods, or hyphens"
            )
        try:
            rules = tuple(BiasRule.from_record(rule) for rule in self.rules)
        except TypeError as exc:
            raise EditorError("bias group rules must be a list of rules") from exc
        if not rules:
            raise EditorError("bias groups must contain at least one rule")
        if any(rule.bias != 0.0 for rule in rules):
            raise EditorError("bias group rules must be bias-free templates")
        if type(self.bias) not in (int, float) or not math.isfinite(self.bias):
            raise EditorError("bias group amount must be finite")
        if not isinstance(self.members, Sequence) or isinstance(self.members, (str, bytes, bytearray)):
            raise EditorError("bias group members must be a list of names")
        members = tuple(str(member) for member in self.members)
        if any(not member for member in members):
            raise EditorError("bias group members cannot be empty")
        if type(self.enabled) is not bool:
            raise EditorError("bias group enabled must be a boolean")
        if isinstance(self.surfaces, str) or any(not isinstance(s, str) or not s.strip() for s in self.surfaces):
            raise EditorError("bias group surfaces must contain nonempty text")
        object.__setattr__(self, "surfaces", tuple(dict.fromkeys(self.surfaces)))
        object.__setattr__(self, "rules", tuple(sorted(rules, key=lambda rule: rule.sort_key)))
        object.__setattr__(self, "bias", float(self.bias))
        object.__setattr__(self, "members", tuple(dict.fromkeys(members)))

    @property
    def key(self) -> str:
        return self.name

    def effective_rules(self) -> tuple[BiasRule, ...]:
        # All routes in a named group are one logical request.  In particular,
        # a group's alternate routes must not multiply the group's
        # amount when they converge on the same next token.
        return tuple(
            replace(
                rule,
                bias=self.bias,
                logical_target=f"group:{self.name}",
            )
            for rule in self.rules
            if self.enabled and self.bias != 0.0
        )

    @classmethod
    def from_record(cls, value: Any) -> "BiasGroup":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise EditorError("bias group must be an object")
        # ``learnable`` was a research-only field.  Accept it at the record
        # boundary so an old workspace can replay until its live edge, but do
        # not retain or expose it in the core group object.
        allowed = {"name", "rules", "bias", "members", "surfaces", "enabled", "learnable"}
        unknown = set(value) - allowed
        if unknown:
            raise EditorError(f"unknown bias group fields: {', '.join(sorted(unknown))}")
        if "name" not in value or "rules" not in value:
            raise EditorError("bias group requires name and rules")
        return cls(
            name=value["name"],
            rules=value["rules"],
            bias=value.get("bias", 0.0),
            members=value.get("members", ()),
            surfaces=value.get("surfaces", ()),
            enabled=value.get("enabled", True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bias": self.bias,
            "members": list(self.members),
            "rules": [rule.to_dict() for rule in self.rules],
            "surfaces": list(self.surfaces),
            "enabled": self.enabled,
        }


def merge_bias_rules(rules: Sequence[BiasRule]) -> tuple[BiasRule, ...]:
    """Combine exact rules while preserving logical-target boundaries.

    Unidentified rules retain the historical additive behavior.  Identified
    rules with the same exact route are duplicate representations of one
    logical target (for example, duplicate group members), so they are kept
    once and the matcher can deduplicate the remaining route overlap.
    """

    merged: dict[tuple[Any, ...], BiasRule] = {}
    for rule in rules:
        normalized = BiasRule.from_record(rule)
        existing = merged.get(normalized.key)
        if existing is None:
            merged[normalized.key] = normalized
        elif normalized.logical_target is not None:
            # The same identified rule can arrive through multiple group
            # members.  Its bias is already the group's single amount.
            continue
        else:
            merged[normalized.key] = replace(
                existing, bias=existing.bias + normalized.bias
            )
    return tuple(sorted(
        (rule for rule in merged.values() if rule.bias != 0.0),
        key=lambda rule: rule.sort_key,
    ))


def _endswith(history: Sequence[int], prefix: Sequence[int]) -> bool:
    if len(history) < len(prefix):
        return False
    if not prefix:
        return True
    return tuple(history[-len(prefix):]) == tuple(prefix)


def _scoped_span(rule: BiasRule, history: Sequence[int]) -> tuple[int, ...]:
    if rule.until is None:
        return tuple(history)
    start = len(history)
    while start and history[start - 1] != rule.until:
        start -= 1
    return tuple(history[start:])


def _trigger_matches(rule: BiasRule, span: Sequence[int]) -> bool:
    return any(
        any(tuple(span[index:index + len(trigger)]) == trigger
            for index in range(len(span) - len(trigger) + 1))
        for trigger in rule.triggers
    )


def _tail_tokens(routes: Sequence[Sequence[int]], history: Sequence[int]) -> dict[int, float]:
    result: dict[int, float] = {}
    for route in routes:
        prefix = route[:-1]
        if not prefix or _endswith(history, prefix):
            result[route[-1]] = 1.0
    return result


def _rule_tokens(rule: BiasRule, history: Sequence[int]) -> dict[int, float]:
    span = _scoped_span(rule, history)
    if rule.triggers and not _trigger_matches(rule, span):
        return {}
    return _tail_tokens(rule.routes, span)


class BiasMatcher:
    """Evaluate unified rules against an exact model-token history."""

    def __init__(self, rules: Sequence[BiasRule]) -> None:
        normalized = tuple(BiasRule.from_record(rule) for rule in rules)
        if len({rule.key for rule in normalized}) != len(normalized):
            raise EditorError("duplicate bias rule")
        self.rules = tuple(sorted(
            (rule for rule in normalized if rule.bias != 0),
            key=lambda rule: rule.sort_key,
        ))

    def _normalized_history(self, history):
        needs_history = any(
            rule.until is not None
            or any(len(route) > 1 for route in rule.routes)
            for rule in self.rules
        )
        if history is None:
            if needs_history:
                raise EditorError("route bias rules require exact context token IDs")
            history = ()
        return history

    def active_biases(self, history) -> dict[int, float]:
        history = self._normalized_history(history)
        # Rules with the same logical identity are one application of a target,
        # not one application per overlapping route. Unidentified rules remain
        # independent so hand-authored runtime rules retain their semantics.
        target_scales: dict[tuple[Any, ...], dict[int, float]] = {}
        for index, rule in enumerate(self.rules):
            target = rule.logical_target
            target_key = (
                target if target is not None else ("rule", index),
                rule.bias,
                rule.triggers,
                rule.until,
            )
            scales = target_scales.setdefault(target_key, {})
            for token, scale in _rule_tokens(rule, history).items():
                scales[token] = max(scales.get(token, 0.0), scale)

        result: dict[int, float] = {}
        for (_target, bias, _triggers, _until), scales in target_scales.items():
            for token, scale in scales.items():
                result[token] = result.get(token, 0.0) + bias * scale
        return result

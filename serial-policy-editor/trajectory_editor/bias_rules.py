"""Logical bias rules and route-aware matching.

The compiler produces routes; this module decides how a runtime bias amount is
applied to those routes.  A rule may contain several alternate routes.  They
are treated as one logical target, so shared prefixes and shared next-token
edges are only biased once.

``tail`` preserves the existing ``prefix -> final token`` behavior.  ``path``
is the telescoping form used for lexical terms that are split into multiple
tokens: it biases the first viable edge and then the next edge after each
matching prefix.  ``beheaded`` uses the same telescoping matcher but suppresses
the first edge entirely.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .domain import EditorError


BIAS_MODES = ("tail", "path", "beheaded")
LEGACY_LIFETIMES = ("sentence", "newline")
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


def _scale(value: Any, *, path: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise EditorError(f"{path} must be a finite nonnegative number")
    return float(value)


@dataclass(frozen=True)
class BiasRule:
    """One logical bias amount applied to one or more token routes."""

    routes: tuple[tuple[int, ...], ...]
    bias: float
    mode: str = "tail"
    triggers: tuple[tuple[int, ...], ...] = ()
    until: int | str | None = None
    head_scale: float = 1.0
    continuation_scale: float = 1.0
    logical_target: str | None = None
    route_weights: tuple[tuple[float, ...], ...] = ()

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
        if until is not None and not (
            (type(until) is int and until >= 0) or until in LEGACY_LIFETIMES
        ):
            raise EditorError(
                "bias rule lifetime must be an exact stop-token ID, sentence, newline, or omitted"
            )
        if until is None and triggers:
            raise EditorError("bias rule triggers require a stop token or lifetime")
        if until is not None and not triggers:
            raise EditorError("bias rule scopes require at least one trigger")
        head_scale = _scale(self.head_scale, path="bias rule head_scale")
        continuation_scale = _scale(
            self.continuation_scale,
            path="bias rule continuation_scale",
        )
        if self.logical_target is not None and (
            not isinstance(self.logical_target, str) or not self.logical_target
        ):
            raise EditorError("bias rule logical_target must be a nonempty string")
        route_weights = tuple(
            tuple(float(weight) for weight in weights)
            for weights in self.route_weights
        )
        if route_weights and len(route_weights) != len(routes):
            raise EditorError("bias rule route_weights must align with routes")
        if any(
            len(weights) != len(route)
            or any(not math.isfinite(weight) or weight < 0 for weight in weights)
            for route, weights in zip(routes, route_weights)
        ):
            raise EditorError("bias rule route_weights must be finite and route-aligned")
        if route_weights:
            ordered = sorted(zip(routes, route_weights), key=lambda item: item[0])
            routes = tuple(route for route, _weights in ordered)
            route_weights = tuple(weights for _route, weights in ordered)
        else:
            routes = tuple(sorted(routes))
        if self.mode == "beheaded":
            head_scale = 0.0
        object.__setattr__(self, "routes", routes)
        object.__setattr__(self, "triggers", tuple(sorted(triggers)))
        object.__setattr__(self, "bias", float(self.bias))
        object.__setattr__(self, "head_scale", head_scale)
        object.__setattr__(self, "continuation_scale", continuation_scale)
        object.__setattr__(self, "route_weights", route_weights)

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.routes,
            self.mode,
            self.triggers,
            self.until,
            self.head_scale,
            self.continuation_scale,
            self.logical_target,
            self.route_weights,
        )

    @property
    def sort_key(self) -> tuple[Any, ...]:
        lifetime = (0, self.until) if type(self.until) is int else (1, self.until)
        return (
            self.routes,
            self.mode,
            self.triggers,
            lifetime,
            self.head_scale,
            self.continuation_scale,
            self.logical_target,
            self.route_weights,
        )

    @classmethod
    def from_record(cls, value: Any) -> "BiasRule":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise EditorError("bias rule must be an object")
        routes = value.get("routes")
        if routes is None and "target" in value:
            routes = [value["target"]]
        allowed = {
            "routes", "target", "bias", "mode", "triggers", "until",
            "head_scale", "continuation_scale", "logical_target", "route_weights",
        }
        unknown = set(value) - allowed
        if unknown:
            raise EditorError(f"unknown bias rule fields: {', '.join(sorted(unknown))}")
        if "bias" not in value:
            raise EditorError("bias rule requires bias")
        return cls(
            routes=routes,
            bias=value["bias"],
            mode=value.get("mode", "tail"),
            triggers=value.get("triggers", ()),
            until=value.get("until"),
            head_scale=value.get("head_scale", 1.0),
            continuation_scale=value.get("continuation_scale", 1.0),
            logical_target=value.get("logical_target"),
            route_weights=value.get("route_weights", ()),
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
        if self.head_scale != 1.0:
            result["head_scale"] = self.head_scale
        if self.continuation_scale != 1.0:
            result["continuation_scale"] = self.continuation_scale
        if self.logical_target is not None:
            result["logical_target"] = self.logical_target
        if self.route_weights:
            result["route_weights"] = [list(weights) for weights in self.route_weights]
        return result


@dataclass(frozen=True)
class BiasGroup:
    """A durable named collection of bias-rule templates and one adjustment."""

    name: str
    rules: tuple[BiasRule, ...]
    bias: float = 0.0
    members: tuple[str, ...] = ()

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
        object.__setattr__(self, "rules", tuple(sorted(rules, key=lambda rule: rule.sort_key)))
        object.__setattr__(self, "bias", float(self.bias))
        object.__setattr__(self, "members", tuple(dict.fromkeys(members)))

    @property
    def key(self) -> str:
        return self.name

    def effective_rules(self) -> tuple[BiasRule, ...]:
        # All routes in a named group are one logical request.  In particular,
        # a group's path and beheaded routes must not multiply the group's
        # amount when they converge on the same next token.
        return tuple(
            replace(
                rule,
                bias=self.bias,
                logical_target=f"group:{self.name}",
            )
            for rule in self.rules
            if self.bias != 0.0
        )

    @classmethod
    def from_record(cls, value: Any) -> "BiasGroup":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise EditorError("bias group must be an object")
        allowed = {"name", "rules", "bias", "members"}
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
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bias": self.bias,
            "members": list(self.members),
            "rules": [rule.to_dict() for rule in self.rules],
        }


def merge_bias_rules(rules: Sequence[BiasRule]) -> tuple[BiasRule, ...]:
    """Combine exact rules while preserving logical-target boundaries.

    Unidentified rules retain the historical additive behavior.  Identified
    rules with the same exact route are duplicate representations of one
    logical target (for example, duplicate group members), so they are kept
    once and the matcher can deduplicate the remaining route-mode overlap.
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


def _scoped_span(
    rule: BiasRule,
    history: Sequence[int],
    boundaries: Any,
) -> tuple[int, ...]:
    if rule.until is None:
        return tuple(history)
    start = len(history)
    if type(rule.until) is int:
        while start and history[start - 1] != rule.until:
            start -= 1
    else:
        if boundaries is None:
            raise EditorError(
                "sentence/newline scoped bias rules require boundary classification"
            )
        while start and rule.until not in boundaries(history[start - 1]):
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


def _path_tokens(
    routes: Sequence[Sequence[int]],
    history: Sequence[int],
    *,
    head_scale: float,
    continuation_scale: float,
) -> dict[int, float]:
    """Return next edges for all routes whose longest prefix matches history."""

    result: dict[int, float] = {}
    for route in routes:
        # The empty prefix is the route's starting edge.  Search longest first
        # so a route never biases both its head and its continuation at once.
        for prefix_length in range(min(len(route) - 1, len(history)), -1, -1):
            prefix = route[:prefix_length]
            if _endswith(history, prefix):
                token = route[prefix_length]
                scale = head_scale if prefix_length == 0 else continuation_scale
                if scale == 0:
                    break
                result[token] = max(result.get(token, 0.0), scale)
                break
    return result


def _weighted_path_tokens(
    routes: Sequence[Sequence[int]],
    route_weights: Sequence[Sequence[float]],
    history: Sequence[int],
    *,
    mode: str,
    head_scale: float,
    continuation_scale: float,
) -> dict[int, float]:
    """Return weighted edges that are eligible under the rule's mode.

    Route allocation supplies the edge shape, but it does not replace the
    route-mode contract.  In particular, a weighted tail rule still waits for
    the complete route prefix, and a weighted beheaded rule never emits its
    first edge.
    """

    result: dict[int, float] = {}
    for route, weights in zip(routes, route_weights):
        if mode == "tail":
            prefix_length = len(route) - 1
            if not _endswith(history, route[:prefix_length]):
                continue
        else:
            first_prefix = 1 if mode == "beheaded" else 0
            max_prefix = min(len(route) - 1, len(history))
            prefix_length = None
            for candidate in range(max_prefix, first_prefix - 1, -1):
                if _endswith(history, route[:candidate]):
                    prefix_length = candidate
                    break
            if prefix_length is None:
                continue

        weight = weights[prefix_length]
        scale = (
            head_scale if prefix_length == 0 else continuation_scale
        )
        weight *= scale
        if weight > 0:
            token = route[prefix_length]
            result[token] = max(result.get(token, 0.0), weight)
    return result


def _rule_tokens(rule: BiasRule, history: Sequence[int], boundaries: Any) -> dict[int, float]:
    span = _scoped_span(rule, history, boundaries)
    if rule.triggers and not _trigger_matches(rule, span):
        return {}
    if rule.route_weights:
        return _weighted_path_tokens(
            rule.routes,
            rule.route_weights,
            span,
            mode=rule.mode,
            head_scale=rule.head_scale,
            continuation_scale=rule.continuation_scale,
        )
    if rule.mode in {"path", "beheaded"}:
        return _path_tokens(
            rule.routes,
            span,
            head_scale=0.0 if rule.mode == "beheaded" else rule.head_scale,
            continuation_scale=rule.continuation_scale,
        )
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

    def active_routes(self, history, boundaries=None) -> set[tuple[int, ...]]:
        """Return routes whose rule-level scope gate is currently enabled.

        This deliberately checks trigger/lifetime activation separately from
        whether a route mode has an outgoing edge at this exact position.  The
        result is used to scope a reference prior without changing the direct
        matcher semantics.
        """

        history = self._normalized_history(history)
        result: set[tuple[int, ...]] = set()
        for rule in self.rules:
            span = _scoped_span(rule, history, boundaries)
            if rule.triggers and not _trigger_matches(rule, span):
                continue
            result.update(rule.routes)
        return result

    def active_biases(self, history, boundaries=None) -> dict[int, float]:
        history = self._normalized_history(history)
        # A catalog entry may expose several route modes for one target.  The
        # modes are evaluated independently, but their overlapping next-token
        # contributions are one application of the target's bias, not one per
        # route mode.  Rules without an identity remain independent for
        # compatibility with hand-authored runtime rules.
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
            for token, scale in _rule_tokens(rule, history, boundaries).items():
                scales[token] = max(scales.get(token, 0.0), scale)

        result: dict[int, float] = {}
        for (_target, bias, _triggers, _until), scales in target_scales.items():
            for token, scale in scales.items():
                result[token] = result.get(token, 0.0) + bias * scale
        return result


def routes_for_catalog_entry(
    entry: Any,
    bias: float,
    *,
    mode: str | None = None,
    logical_target: str | None = None,
) -> tuple[BiasRule, ...]:
    """Turn a compiled catalog entry into one rule per route mode.

    This is deliberately separate from command parsing: catalog resolution is
    a runtime concern, while the catalog itself remains model-specific data.
    """

    if not hasattr(entry, "routes"):
        raise EditorError("catalog entry must expose compiled routes")
    if logical_target is None:
        logical_target = f"catalog:{entry.name}"
    grouped: dict[
        tuple[str, float, float, bool],
        list[tuple[tuple[int, ...], tuple[float, ...]]],
    ] = {}
    for route in entry.routes:
        selected = mode or route.mode
        if selected not in BIAS_MODES:
            raise EditorError(f"catalog route has unsupported bias mode {selected!r}")
        head_scale = 0.0 if selected == "beheaded" else route.head_scale
        grouped.setdefault(
            (
                selected,
                head_scale,
                route.continuation_scale,
                bool(route.edge_weights),
            ),
            [],
        ).append((tuple(route.token_ids), tuple(route.edge_weights)))
    return tuple(
        BiasRule(
            routes=tuple(route for route, _weights in routes),
            bias=bias,
            mode=selected,
            head_scale=head_scale,
            continuation_scale=continuation_scale,
            logical_target=logical_target,
            route_weights=(
                tuple(weights for _route, weights in routes)
                if weighted else ()
            ),
        )
        for (selected, head_scale, continuation_scale, weighted), routes
        in sorted(grouped.items())
    )

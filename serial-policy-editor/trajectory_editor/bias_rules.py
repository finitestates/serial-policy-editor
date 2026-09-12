"""Logical bias rules and route-aware matching.

The compiler produces routes; this module decides how a runtime bias amount is
applied to those routes.  A rule may contain several alternate routes.  They
are treated as one logical target, so shared prefixes and shared next-token
edges are only biased once.

``tail`` preserves the existing ``prefix -> final token`` behavior.  ``path``
is the telescoping form used for lexical terms that are split into multiple
tokens: it biases the first viable edge and then the next edge after each
matching prefix.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .domain import EditorError


BIAS_MODES = ("tail", "path")
LEGACY_LIFETIMES = ("sentence", "newline")


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
    until: int | str | None = None

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
        object.__setattr__(self, "routes", tuple(sorted(routes)))
        object.__setattr__(self, "triggers", tuple(sorted(triggers)))
        object.__setattr__(self, "bias", float(self.bias))

    @property
    def key(self) -> tuple[Any, ...]:
        return self.routes, self.mode, self.triggers, self.until

    @property
    def sort_key(self) -> tuple[Any, ...]:
        lifetime = (0, self.until) if type(self.until) is int else (1, self.until)
        return self.routes, self.mode, self.triggers, lifetime

    @classmethod
    def from_record(cls, value: Any) -> "BiasRule":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise EditorError("bias rule must be an object")
        routes = value.get("routes")
        if routes is None and "target" in value:
            routes = [value["target"]]
        allowed = {"routes", "target", "bias", "mode", "triggers", "until"}
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
        return result


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


def _tail_tokens(routes: Sequence[Sequence[int]], history: Sequence[int]) -> set[int]:
    result: set[int] = set()
    for route in routes:
        prefix = route[:-1]
        if not prefix or _endswith(history, prefix):
            result.add(route[-1])
    return result


def _path_tokens(routes: Sequence[Sequence[int]], history: Sequence[int]) -> set[int]:
    """Return next edges for all routes whose longest prefix matches history."""

    result: set[int] = set()
    for route in routes:
        # The empty prefix is the route's starting edge.  Search longest first
        # so a route never biases both its head and its continuation at once.
        for prefix_length in range(min(len(route) - 1, len(history)), -1, -1):
            prefix = route[:prefix_length]
            if _endswith(history, prefix):
                result.add(route[prefix_length])
                break
    return result


def _rule_tokens(rule: BiasRule, history: Sequence[int], boundaries: Any) -> set[int]:
    span = _scoped_span(rule, history, boundaries)
    if rule.triggers and not _trigger_matches(rule, span):
        return set()
    if rule.mode == "path":
        return _path_tokens(rule.routes, span)
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

    def active_biases(self, history, boundaries=None) -> dict[int, float]:
        needs_history = any(
            rule.until is not None
            or any(len(route) > 1 for route in rule.routes)
            for rule in self.rules
        )
        if history is None:
            if needs_history:
                raise EditorError("route bias rules require exact context token IDs")
            history = ()
        result: dict[int, float] = {}
        for rule in self.rules:
            for token in _rule_tokens(rule, history, boundaries):
                result[token] = result.get(token, 0.0) + rule.bias
        return result


def routes_for_catalog_entry(entry: Any, bias: float, *, mode: str | None = None) -> tuple[BiasRule, ...]:
    """Turn a compiled catalog entry into one rule per route mode.

    This is deliberately separate from command parsing: catalog resolution is
    a runtime concern, while the catalog itself remains model-specific data.
    """

    if not hasattr(entry, "routes"):
        raise EditorError("catalog entry must expose compiled routes")
    grouped: dict[str, list[tuple[int, ...]]] = {}
    for route in entry.routes:
        selected = mode or route.mode
        if selected not in BIAS_MODES:
            raise EditorError(f"catalog route has unsupported bias mode {selected!r}")
        grouped.setdefault(selected, []).append(tuple(route.token_ids))
    return tuple(
        BiasRule(routes=tuple(routes), bias=bias, mode=selected)
        for selected, routes in sorted(grouped.items())
    )

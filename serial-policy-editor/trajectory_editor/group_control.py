"""History-driven control of completed lexical group appearances.

Objectives are independent of teacher preference fitting. A control snapshot
and exact token history fully determine its response, including during Holds,
rewind, and replay. Observations never mutate controller memory.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .bias_rules import BiasRule, _scoped_span, _trigger_matches, _endswith
from .domain import EditorError


def _logodds(value):
    value = min(1 - 1e-9, max(1e-9, float(value)))
    return math.log(value) - math.log1p(-value)


@dataclass(frozen=True)
class GroupControl:
    group: str
    direction: str
    baseline_rate: float
    level: float = 1.0
    enabled: bool = True
    window: int = 256
    max_bias: float = 4.0
    tolerance: float = 0.2
    triggers: tuple = ()
    until: int | str | None = None
    history_start: int | None = None
    scheme: str = "appearance-feedback-v1"

    def __post_init__(self):
        if self.scheme != "appearance-feedback-v1":
            raise EditorError(f"unsupported group control scheme: {self.scheme}")
        if not isinstance(self.group, str) or not self.group:
            raise EditorError("group control requires a group name")
        if self.direction not in {"promote", "suppress", "maintain"}:
            raise EditorError("group control direction must be promote, suppress, or maintain")
        if type(self.enabled) is not bool:
            raise EditorError("group control enabled must be a boolean")
        for name in ("baseline_rate", "level", "max_bias", "tolerance"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise EditorError(f"group control {name} must be finite and nonnegative")
        if self.baseline_rate > 1 or self.tolerance > 1:
            raise EditorError("group control rate and tolerance must be at most 1")
        if type(self.window) is not int or self.window < 1:
            raise EditorError("group control window must be a positive token count")
        if self.history_start is not None and (type(self.history_start) is not int or self.history_start < 0):
            raise EditorError("group control history_start must be a nonnegative token offset")
        # Share the existing exact trigger/lifetime contract and validation.
        gate = BiasRule(routes=((0,),), bias=1, triggers=self.triggers, until=self.until)
        object.__setattr__(self, "triggers", gate.triggers)

    @property
    def key(self):
        return self.group, self.triggers, self.until

    @property
    def target_rate(self):
        if self.baseline_rate == 0 or self.direction == "maintain":
            return self.baseline_rate
        sign = 1 if self.direction == "promote" else -1
        odds = _logodds(self.baseline_rate) + sign * min(20., self.level) * math.log(2)
        return 1 / (1 + math.exp(-odds))

    @classmethod
    def from_record(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise EditorError("group control must be an object")
        try:
            return cls(**value)
        except TypeError as exc:
            raise EditorError(f"invalid group control: {exc}") from exc

    def to_dict(self):
        return dict(group=self.group, direction=self.direction, baseline_rate=self.baseline_rate,
                    level=self.level, enabled=self.enabled, window=self.window,
                    max_bias=self.max_bias, tolerance=self.tolerance,
                    triggers=[list(route) for route in self.triggers], until=self.until, history_start=self.history_start,
                    scheme=self.scheme)


def control_span(control, history, boundaries=None):
    gate = BiasRule(routes=((0,),), bias=1, triggers=control.triggers, until=control.until)
    span = _scoped_span(gate, history, boundaries)
    offset = len(history) - len(span)
    if gate.triggers:
        if not _trigger_matches(gate, span):
            return None
        # Count only the text following the first complete trigger in this
        # lifetime. Repeated triggers do not multiply or restart the control.
        first_end = min(i + len(trigger) for trigger in gate.triggers
                        for i in range(len(span) - len(trigger) + 1)
                        if tuple(span[i:i + len(trigger)]) == trigger)
        span = span[first_end:]
        offset += first_end
    start = control.history_start or 0
    span = span[max(0, start - offset):]
    return tuple(span[-control.window:])


@lru_cache(maxsize=256)
def _surface_pattern(surfaces):
    forms = sorted({s.strip() for s in surfaces if s.strip()}, key=lambda s: (-len(s), s))
    if not forms:
        return None
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(s) for s in forms) + r")(?!\w)")


@lru_cache(maxsize=256)
def group_routes(group):
    return tuple(sorted({route for rule in group.rules for route in rule.routes}))


def appearances(group, history, render_tokens=None):
    pattern = _surface_pattern(group.surfaces)
    if pattern is not None and render_tokens is not None:
        return sum(1 for _ in pattern.finditer(render_tokens(list(history), special=False)))
    # Exact-token groups and low-level callers can use route completion. One
    # ending position is one occurrence even if several members match there.
    routes = group_routes(group)
    return sum(any(_endswith(history[:end], route) for route in routes)
               for end in range(1, len(history) + 1))


def outgoing(group, history):
    routes = group_routes(group)
    continuations = {}
    for route in routes:
        for n in range(min(len(history), len(route) - 1), 0, -1):
            if _endswith(history, route[:n]):
                continuations[route[n]] = 1.0
                break
    if continuations:
        return continuations, True
    # All approved canonical paths participate. Long phrases get gentler
    # entry pressure; their continuation pressure remains full strength.
    roots = {}
    for route in routes:
        roots[route[0]] = max(roots.get(route[0], 0.), 1 / math.sqrt(len(route)))
    return roots, False


def estimate_baseline(group, history, probabilities, render_tokens=None, window=256):
    span = tuple(history[-window:])
    edges, _ = outgoing(group, ())
    route_length = sum(len(r) for r in group_routes(group)) / len(group_routes(group))
    opportunity = sum(float(probabilities[t]) for t in edges) / route_length
    # A small prior allows useful activation before enough text has appeared.
    # With evidence, observed completed appearances dominate the estimate.
    return min(1., (appearances(group, span, render_tokens) + 16 * opportunity) / (len(span) + 16))


def control_adjustments(controls, groups, history, baseline_logits, boundaries=None, render_tokens=None):
    if not controls:
        return {}, ()
    groups = {g.name: g for g in groups}
    shifted = np.asarray(baseline_logits) - np.max(baseline_logits)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    totals = {}
    diagnostics = []
    for control in controls:
        group = groups[control.group]
        span = control_span(control, history, boundaries) if control.enabled and group.enabled else None
        if span is None:
            diagnostics.append(dict(group=group.name, active=False, direction=control.direction))
            continue
        # Measurement excludes the prompt and old window. Prefix recognition
        # still needs that context to complete a term begun at the boundary.
        gate = BiasRule(routes=((0,),), bias=1, triggers=control.triggers, until=control.until)
        edges, continuing = outgoing(group, _scoped_span(gate, history, boundaries))
        count = appearances(group, span, render_tokens)
        observed = (count + 16 * control.baseline_rate) / (len(span) + 16)
        target = control.target_rate
        feedback = _logodds(target) - _logodds(observed)
        if abs(target - observed) <= control.tolerance * max(target, 1 / (control.window + 16)):
            feedback = 0.
        # Combine completed-appearance feedback with the present opportunity.
        # This responds during autonomous generation without treating samples
        # as teacher preferences. It is bounded even for unavailable phrases.
        mass = sum(float(probabilities[t]) for t in edges)
        sign = {"promote": 1, "suppress": -1, "maintain": 0}[control.direction]
        base = sign * min(control.level, 20.) * math.log(2)
        if control.direction == "maintain" and not continuing:
            base = _logodds(control.baseline_rate) - _logodds(mass)
        pressure = base + feedback
        if sign > 0:
            pressure = max(0., pressure)
        elif sign < 0:
            pressure = min(0., pressure)
        pressure = max(-control.max_bias, min(control.max_bias, pressure))
        for token, scale in edges.items():
            totals[token] = totals.get(token, 0.) + scale * pressure
        diagnostics.append(dict(group=group.name, active=True, direction=control.direction,
                                baseline_rate=control.baseline_rate, target_rate=target,
                                observed_rate=observed, appearances=count, tokens=len(span),
                                continuing=continuing, opportunity_mass=mass, pressure=pressure))
    # Overlapping groups share a total intervention budget.
    limit = max((c.max_bias for c in controls if c.enabled), default=0.)
    return {t: max(-limit, min(limit, b)) for t, b in totals.items()}, tuple(diagnostics)

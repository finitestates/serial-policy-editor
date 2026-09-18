"""History-driven control of completed lexical group appearances.

Objectives are independent of teacher preference fitting. A control snapshot
and exact token history fully determine its response, including during Holds,
rewind, and replay. Observations never mutate controller memory.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from collections import deque
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
    prior_exposure: float = 16.0
    deadband_z: float = 1.0
    feedforward_gain: float = 1.0
    proportional_gain: float = 1.0
    integral_gain: float = 0.0
    integral_decay: float = 0.9
    integral_limit: float | None = None
    phrase_entry_scheme: str = "sqrt_length_v1"

    def __post_init__(self):
        if self.scheme not in {"appearance-feedback-v1", "appearance-rate-v2"}:
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
        for name in (
            "prior_exposure", "deadband_z", "feedforward_gain",
            "proportional_gain", "integral_gain",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
                raise EditorError(f"group control {name} must be finite and nonnegative")
        if (
            type(self.integral_decay) not in (int, float)
            or not math.isfinite(float(self.integral_decay))
            or not 0.0 <= float(self.integral_decay) <= 1.0
        ):
            raise EditorError("group control integral_decay must be between 0 and 1")
        if self.integral_limit is not None and (
            type(self.integral_limit) not in (int, float)
            or not math.isfinite(float(self.integral_limit))
            or self.integral_limit < 0
        ):
            raise EditorError("group control integral_limit must be finite and nonnegative")
        if self.phrase_entry_scheme not in {"sqrt_length_v1", "uniform_v2"}:
            raise EditorError("unsupported phrase_entry_scheme")
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
        if self.scheme == "appearance-rate-v2":
            sign = 1 if self.direction == "promote" else -1
            return self.baseline_rate * math.exp2(
                sign * min(20.0, float(self.level))
            )
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
                    scheme=self.scheme, prior_exposure=self.prior_exposure,
                    deadband_z=self.deadband_z, feedforward_gain=self.feedforward_gain,
                    proportional_gain=self.proportional_gain, integral_gain=self.integral_gain,
                    integral_decay=self.integral_decay, integral_limit=self.integral_limit,
                    phrase_entry_scheme=self.phrase_entry_scheme)


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


@dataclass
class _GroupRouteNode:
    children: dict[int, int]
    failure: int = 0
    terminal: bool = False
    prefix: tuple[int, ...] = ()


class GroupRouteTrie:
    """Finite-state route controller with deterministic suffix fallback."""

    def __init__(self, routes):
        self.nodes = [_GroupRouteNode(children={})]
        for route in routes:
            state = 0
            prefix = []
            for token in route:
                token = int(token)
                prefix.append(token)
                child = self.nodes[state].children.get(token)
                if child is None:
                    child = len(self.nodes)
                    self.nodes[state].children[token] = child
                    self.nodes.append(_GroupRouteNode(
                        children={}, prefix=tuple(prefix)
                    ))
                state = child
            self.nodes[state].terminal = True
        queue = deque(self.nodes[0].children.values())
        while queue:
            state = queue.popleft()
            node = self.nodes[state]
            for token, child in node.children.items():
                failure = node.failure
                while failure and token not in self.nodes[failure].children:
                    failure = self.nodes[failure].failure
                self.nodes[child].failure = self.nodes[failure].children.get(token, 0)
                queue.append(child)

    def transition(self, state, token):
        token = int(token)
        while state and token not in self.nodes[state].children:
            state = self.nodes[state].failure
        return self.nodes[state].children.get(token, 0)

    def state_for_history(self, history):
        state = 0
        for token in history:
            state = self.transition(state, token)
        while state and self.nodes[state].terminal and not self.nodes[state].children:
            state = self.nodes[state].failure
        return state

    def outgoing(self, state):
        return tuple(sorted(self.nodes[state].children))


@lru_cache(maxsize=256)
def group_route_trie(routes):
    return GroupRouteTrie(routes)


def _v2_outgoing(group, history, phrase_entry_scheme):
    routes = group_routes(group)
    trie = group_route_trie(routes)
    state = trie.state_for_history(history)
    if state:
        return (
            {token: 1.0 for token in trie.outgoing(state)},
            True,
            trie,
            state,
        )
    roots = {}
    for route in routes:
        scale = (
            1.0
            if phrase_entry_scheme == "uniform_v2"
            else 1.0 / math.sqrt(len(route))
        )
        roots[route[0]] = max(roots.get(route[0], 0.0), scale)
    return roots, False, trie, state


def estimate_baseline(group, history, probabilities, render_tokens=None, window=256):
    span = tuple(history[-window:])
    edges, _ = outgoing(group, ())
    route_length = sum(len(r) for r in group_routes(group)) / len(group_routes(group))
    opportunity = sum(float(probabilities[t]) for t in edges) / route_length
    # A small prior allows useful activation before enough text has appeared.
    # With evidence, observed completed appearances dominate the estimate.
    return min(1., (appearances(group, span, render_tokens) + 16 * opportunity) / (len(span) + 16))


def gamma_poisson_rate(count, exposure, baseline_rate, prior_exposure=16.0):
    """Return posterior mean and variance for an appearance rate."""
    count = max(0.0, float(count))
    exposure = max(0.0, float(exposure))
    kappa = max(0.0, float(prior_exposure))
    denominator = exposure + kappa
    if denominator <= 0.0:
        return float(baseline_rate), 0.0
    shape = count + kappa * max(0.0, float(baseline_rate))
    return shape / denominator, shape / (denominator * denominator)


def estimate_rate(group, history, render_tokens=None, window=256, prior_exposure=16.0,
                  baseline_rate=0.0):
    """Estimate a group's appearance rate and posterior variance."""
    span = tuple(history[-window:])
    count = appearances(group, span, render_tokens)
    return gamma_poisson_rate(count, len(span), baseline_rate, prior_exposure)


def _log_rate_error(target, observed):
    epsilon = 1.0e-12
    return math.log(max(epsilon, float(target) + epsilon)) - math.log(
        max(epsilon, float(observed) + epsilon)
    )


def _reconstructed_integral(group, control, span, render_tokens, target):
    if control.integral_gain == 0.0:
        return 0.0
    state = 0.0
    limit = (
        control.max_bias
        if control.integral_limit is None
        else min(control.max_bias, control.integral_limit)
    )
    for end in range(1, len(span) + 1):
        prefix = span[:end]
        count = appearances(group, prefix, render_tokens)
        observed, variance = gamma_poisson_rate(
            count, len(prefix), control.baseline_rate, control.prior_exposure
        )
        error = _log_rate_error(target, observed)
        deadband = control.deadband_z * math.sqrt(max(variance, 1.0e-24))
        if abs(float(target) - observed) <= deadband:
            error = 0.0
        candidate = control.integral_decay * state + error
        feedforward = (
            0.0 if control.direction == "maintain" or control.baseline_rate == 0.0 else
            (1.0 if control.direction == "promote" else -1.0)
            * min(20.0, control.level) * math.log(2.0)
        )
        command = (
            control.feedforward_gain * feedforward
            + control.proportional_gain * error
            + control.integral_gain * candidate
        )
        if (command > control.max_bias and error > 0.0) or (
            command < -control.max_bias and error < 0.0
        ):
            candidate = control.integral_decay * state
        state = max(-limit, min(limit, candidate))
    return state


def _scaled_actuator_biases(edges, pressure, logits, probabilities):
    """Solve for edge biases when route-entry scales are nonuniform."""
    if not edges:
        return {}
    tokens = np.asarray(sorted(edges), dtype=np.int64)
    scales = np.asarray([edges[int(token)] for token in tokens], dtype=np.float64)
    if np.allclose(scales, scales[0]):
        return {int(token): float(pressure) for token in tokens}
    mass = float(np.sum(probabilities[tokens]))
    if mass <= 0.0 or mass >= 1.0:
        return {int(token): float(scale * pressure) for token, scale in edges.items()}
    desired = 1.0 / (1.0 + math.exp(-(_logodds(mass) + float(pressure))))
    maximum = float(np.max(logits))
    weights = np.exp(np.asarray(logits, dtype=np.float64) - maximum)
    edge_weights = weights[tokens]
    outside = float(np.sum(weights) - np.sum(edge_weights))

    def mass_at(command):
        numerator = float(np.sum(edge_weights * np.exp(scales * command)))
        return numerator / (outside + numerator)

    low, high = -64.0, 64.0
    low_mass, high_mass = mass_at(low), mass_at(high)
    if desired <= low_mass:
        command = low
    elif desired >= high_mass:
        command = high
    else:
        for _ in range(64):
            command = (low + high) * 0.5
            if mass_at(command) < desired:
                low = command
            else:
                high = command
        command = (low + high) * 0.5
    return {
        int(token): float(scale * command)
        for token, scale in zip(tokens, scales)
    }


def control_adjustments(
    controls, groups, history, baseline_logits, boundaries=None, render_tokens=None,
    *, scheme=None,
):
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
        scoped_history = _scoped_span(gate, history, boundaries)
        effective_scheme = control.scheme
        if effective_scheme == "appearance-feedback-v1" and scheme in {
            "appearance-feedback-v1", "appearance-rate-v2"
        }:
            effective_scheme = scheme
        if effective_scheme == "appearance-rate-v2":
            edges, continuing, trie, state = _v2_outgoing(
                group, scoped_history, control.phrase_entry_scheme
            )
        else:
            edges, continuing = outgoing(group, scoped_history)
        count = appearances(group, span, render_tokens)
        if effective_scheme == "appearance-rate-v2":
            observed, posterior_variance = gamma_poisson_rate(
                count, len(span), control.baseline_rate, control.prior_exposure
            )
            if control.baseline_rate == 0.0 or control.direction == "maintain":
                target = control.baseline_rate
            else:
                sign = 1 if control.direction == "promote" else -1
                target = control.baseline_rate * math.exp2(
                    sign * min(20.0, float(control.level))
                )
            feedback = _log_rate_error(target, observed)
            deadband = control.deadband_z * math.sqrt(
                max(posterior_variance, 1.0e-24)
            )
            if abs(target - observed) <= deadband:
                feedback = 0.0
        else:
            observed = (count + 16 * control.baseline_rate) / (len(span) + 16)
            posterior_variance = 0.0
            target = control.target_rate
            feedback = _logodds(target) - _logodds(observed)
            if abs(target - observed) <= control.tolerance * max(target, 1 / (control.window + 16)):
                feedback = 0.
        # Combine completed-appearance feedback with the present opportunity.
        # This responds during autonomous generation without treating samples
        # as teacher preferences. It is bounded even for unavailable phrases.
        mass = sum(float(probabilities[t]) for t in edges)
        sign = {"promote": 1, "suppress": -1, "maintain": 0}[control.direction]
        if effective_scheme == "appearance-rate-v2":
            feedforward = (
                sign * min(control.level, 20.0) * math.log(2.0)
                if sign and control.baseline_rate > 0.0 else 0.0
            )
            integral_state = _reconstructed_integral(
                group, control, span, render_tokens, target
            )
            feedforward_term = control.feedforward_gain * feedforward
            proportional_term = control.proportional_gain * feedback
            integral_term = control.integral_gain * integral_state
            raw_pressure = feedforward_term + proportional_term + integral_term
            pressure = max(-control.max_bias, min(control.max_bias, raw_pressure))
            actuator_biases = _scaled_actuator_biases(
                edges, pressure, np.asarray(baseline_logits), probabilities
            )
            for token, amount in actuator_biases.items():
                totals[token] = totals.get(token, 0.0) + amount
            diagnostics.append(dict(
                group=group.name, active=True, direction=control.direction,
                scheme=effective_scheme, baseline_rate=control.baseline_rate,
                target_rate=target, observed_rate=observed,
                posterior_variance=posterior_variance, deadband=deadband,
                appearances=count, tokens=len(span), continuing=continuing,
                opportunity_mass=mass, feedforward=feedforward_term,
                proportional=proportional_term, integral=integral_term,
                integral_state=integral_state, raw_pressure=raw_pressure,
                pressure=pressure, saturated=pressure != raw_pressure,
                prefix_state=list(trie.nodes[state].prefix),
                outgoing_tokens=sorted(edges),
            ))
        else:
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
                                    continuing=continuing, opportunity_mass=mass, pressure=pressure,
                                    scheme=effective_scheme))
    # Overlapping groups share a total intervention budget.
    limit = max((c.max_bias for c in controls if c.enabled), default=0.)
    return {t: max(-limit, min(limit, b)) for t, b in totals.items()}, tuple(diagnostics)

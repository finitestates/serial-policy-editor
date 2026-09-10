"""Immutable trigger rules; activation is derived from the current context."""
from dataclasses import dataclass
import math
from collections.abc import Mapping

from .domain import EditorError


@dataclass(frozen=True)
class ScopedBias:
    triggers: tuple[tuple[int, ...], ...]
    target: tuple[int, ...]
    until: str
    bias: float

    def __post_init__(self):
        try:
            triggers = tuple(tuple(tokens) for tokens in self.triggers)
            target = tuple(self.target)
        except TypeError as exc:
            raise EditorError("Scoped bias requires trigger sequences and a target sequence") from exc
        if not triggers or any(not tokens or any(type(t) is not int or t < 0 for t in tokens)
                               for tokens in (*triggers, target)):
            raise EditorError("Scoped bias sequences require nonnegative integer token IDs")
        if self.until not in ("sentence", "newline"):
            raise EditorError("Scoped bias lifetime must be sentence or newline")
        if type(self.bias) not in (int, float) or not math.isfinite(self.bias):
            raise EditorError("Scoped bias must be finite")
        object.__setattr__(self, "triggers", tuple(sorted(set(triggers))))
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "bias", float(self.bias))

    @property
    def key(self):
        return self.triggers, self.target, self.until

    @classmethod
    def from_record(cls, value):
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or set(value) != {"triggers", "target", "until", "bias"}:
            raise EditorError("Scoped bias requires triggers, target, until, and bias")
        return cls(**value)

    def to_dict(self):
        return {"triggers": [list(tokens) for tokens in self.triggers],
                "target": list(self.target), "until": self.until, "bias": self.bias}


def active_scoped_biases(rules, history, boundaries):
    if history is None or boundaries is None:
        raise EditorError("Scoped biases require exact context tokens and boundary classification")
    spans = {}
    result = {}
    for rule in rules:
        if rule.until not in spans:
            start = len(history)
            while start and rule.until not in boundaries(history[start - 1]):
                start -= 1
            spans[rule.until] = tuple(history[start:])
        span = spans[rule.until]
        prefix = rule.target[:-1]
        if prefix and (len(span) < len(prefix) or span[-len(prefix):] != prefix):
            continue
        matched = any(
            span[index:index + len(trigger)] == trigger
            for trigger in rule.triggers
            for index in range(len(span) - len(trigger) + 1)
        )
        if matched:
            token = rule.target[-1]
            result[token] = result.get(token, 0.0) + rule.bias
    return result

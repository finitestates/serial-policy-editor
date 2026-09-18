"""Shared experiment semantics for the group and preference teacher learners."""

import math

from .domain import EditorError


DECAY_ON = ("update", "rejection", "evidence")
WRITE_REDUCTIONS = ("sum", "mean", "sqrt")
REJECTION_TARGETS = ("proposal", "sampler")


def validate_controls(decay_on, write_reduction, rejection_target):
    for name, value, choices in (
        ("decay_on", decay_on, DECAY_ON),
        ("write_reduction", write_reduction, WRITE_REDUCTIONS),
        ("rejection_target", rejection_target, REJECTION_TARGETS),
    ):
        if value not in choices:
            raise EditorError(f"learning {name} must be one of {', '.join(choices)}")


def decay_applies(mode, *, rejected, severity):
    """Evidence means admitted by the gate, even if a later limit clips it away."""
    return mode == "update" or (mode == "rejection" and rejected) or (
        mode == "evidence" and severity > 0)


def write_scale(mode, results):
    """Gate-skipped tokens do not dilute a write's admitted evidence."""
    count = sum(result.severity > 0 for result in results)
    denominator = max(1, count)
    scale = (1. / denominator if mode == "mean" else
             1. / math.sqrt(denominator) if mode == "sqrt" else 1.)
    return scale, count

"""Shared policy diagnostics and user-selected candidate-table columns.

Presentation is an identity core (rank | token-id | text — rank/text stay
outside the column tuple) plus named overlays. Soft-max % overlays are
opt-in so the default menu does not wake dense logsumexp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .core.candidates import Candidate


@dataclass(frozen=True)
class OverlaySpec:
    """Named column overlay and the observation metrics it requests."""

    name: str
    labels: tuple[str, ...]
    label_metrics: tuple[frozenset[str], ...] = ()

    @property
    def metrics(self) -> frozenset[str]:
        return frozenset().union(*self.label_metrics)


# Registry of named overlays. Width / policy gating stays in CandidateColumns.
OVERLAYS: Mapping[str, OverlaySpec] = {
    "noise": OverlaySpec(
        name="noise", labels=("noise",), label_metrics=(frozenset({"noise"}),),
    ),
    "probability": OverlaySpec(
        name="probability",
        labels=("model-softmax", "policy-softmax", "eligible-softmax"),
        label_metrics=(frozenset({"raw_probability"}),
                       frozenset({"policy_probability"}),
                       frozenset({"eligible_softmax"})),
    ),
    "logit": OverlaySpec(
        name="logit",
        labels=("model-logit",),
        label_metrics=(frozenset({"raw_logit"}),),
    ),
    "diff": OverlaySpec(
        name="diff",
        labels=("model-gap",),
        label_metrics=(frozenset({"raw_logit", "top_raw_logit"}),),
    ),
}

# Shared names for shortcuts, explicit toggles, and exact column selection.
OVERLAY_ALIASES: Mapping[str, str] = {
    "raw": "logit", "logits": "logit", "gap": "diff", "gap_k1": "diff",
    "pct": "probability", "probs": "probability", "%": "probability",
    "~": "noise",
}


_COLUMN_WIDTHS: Mapping[str, int] = {
    "noise": 11,
    "model-logit": 11,
    "model-gap": 10,
    "Δrank": 6,
    "pol-rank": 8,
    "model-softmax": 14,
    "policy-softmax": 14,
    "eligible-softmax": 16,
    "token-id": 8,
}

@dataclass(frozen=True)
class CandidateViewPlan:
    """Resolved visible columns and the only candidate metrics to calculate."""

    columns: tuple[tuple[str, int], ...]
    metrics: frozenset[str]

    def needs(self, metric: str) -> bool:
        return metric in self.metrics

    def with_policy_rank(self) -> CandidateViewPlan:
        return CandidateViewPlan(self.columns, self.metrics | {"policy_rank"})


@dataclass(frozen=True)
class CandidateColumns:
    policy: bool = False
    raw_k1_logit: float | None = None
    overlays: frozenset[str] | None = None

    @property
    def enabled_overlays(self) -> frozenset[str]:
        return frozenset(
            OVERLAY_ALIASES.get(name, name) for name in (self.overlays or frozenset())
            if OVERLAY_ALIASES.get(name, name) in OVERLAYS
        )

    @property
    def plan(self) -> CandidateViewPlan:
        labels = {label for label, _ in self.columns}
        metrics: set[str] = set()
        for overlay in self.enabled_overlays:
            spec = OVERLAYS.get(overlay)
            if spec is None:
                continue
            for label, needs in zip(spec.labels, spec.label_metrics):
                if label in labels:
                    metrics.update(needs)
        if "Δrank" in labels or "pol-rank" in labels:
            metrics.add("policy_rank")
        return CandidateViewPlan(self.columns, frozenset(metrics))

    @property
    def columns(self) -> tuple[tuple[str, int], ...]:
        enabled = self.enabled_overlays
        requested: list[str] = []

        if "logit" in enabled:
            requested.append("model-logit")
        if "diff" in enabled:
            requested.append("model-gap")

        if "noise" in enabled:
            requested.append("noise")

        if self.policy:
            requested.extend(("Δrank", "pol-rank"))

        if "probability" in enabled:
            requested.append("model-softmax")
            if self.policy:
                requested.append("policy-softmax")
            requested.append("eligible-softmax")

        # Column visibility follows user preferences, never terminal geometry.
        columns = [(label, _COLUMN_WIDTHS[label]) for label in requested]
        columns.append(("token-id", _COLUMN_WIDTHS["token-id"]))
        return tuple(columns)

    @property
    def heading(self) -> str:
        return "".join(
            f"  {label:>{width}}"
            for label, width in self.columns
        )

    def values(self, candidate: Candidate) -> str:
        def value(label: str) -> str:
            if label == "noise":
                return f"{candidate.noise:+.3f}" if candidate.noise is not None else "--"
            if label == "model-logit":
                return (
                    f"{candidate.raw_logit:+.3f}"
                    if candidate.raw_logit is not None
                    else "--"
                )
            if label == "model-gap":
                return (
                    f"{candidate.raw_logit - self.raw_k1_logit:+.3f}"
                    if candidate.raw_logit is not None and self.raw_k1_logit is not None
                    else "--"
                )
            if label == "Δrank":
                return (
                    f"{candidate.rank - candidate.policy_rank:+d}"
                    if candidate.policy_rank is not None
                    else "--"
                )
            if label == "pol-rank":
                return (
                    str(candidate.policy_rank)
                    if candidate.policy_rank is not None
                    else "--"
                )
            if label == "token-id":
                return str(candidate.token_id)
            probability = {
                "model-softmax": candidate.model_probability,
                "policy-softmax": candidate.policy_probability,
                "eligible-softmax": candidate.eligible_softmax,
            }[label]
            return (
                f"{probability:.2%}"
                if probability is not None
                else "--"
            )

        return "".join(f"  {value(label):>{width}}" for label, width in self.columns)

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
    # Declared stubs may set wired=False until rendering exists.
    wired: bool = True

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
        labels=("raw-p", "pol-p", "decode-p"),
        label_metrics=(frozenset({"raw_probability"}),
                       frozenset({"policy_probability"}),
                       frozenset({"decoder_probability"})),
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
    "raw-p": 8,
    "pol-p": 8,
    "decode-p": 8,
    "token-id": 8,
}

# Keep the internal column key stable while naming its user-facing meaning.
_COLUMN_DISPLAY_LABELS: Mapping[str, str] = {"raw-p": "model-p"}


def overlays_from_preferences(
    *,
    logit_view: str = "none",
    show_model_probabilities: bool = False,
    overlays: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Derive the enabled overlay set from session presentation prefs.

    Shortcut and explicit overlays are additive.
    """
    enabled = {OVERLAY_ALIASES.get(name, name) for name in overlays}
    enabled.intersection_update(OVERLAYS)
    if logit_view in {"raw", "both"}:
        enabled.add("logit")
    if logit_view in {"gap", "both"}:
        enabled.add("diff")
    if show_model_probabilities:
        enabled.add("probability")
    return frozenset(enabled)


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
    logit_view: str = "none"
    raw_k1_logit: float | None = None
    show_model_probabilities: bool = False
    overlays: frozenset[str] | None = None

    @property
    def enabled_overlays(self) -> frozenset[str]:
        return overlays_from_preferences(
            logit_view=self.logit_view,
            show_model_probabilities=self.show_model_probabilities,
            overlays=self.overlays or frozenset(),
        )

    @property
    def plan(self) -> CandidateViewPlan:
        labels = {label for label, _ in self.columns}
        metrics: set[str] = set()
        for overlay in self.enabled_overlays:
            spec = OVERLAYS.get(overlay)
            if spec is None or not spec.wired:
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

        if "logit" in enabled and OVERLAYS["logit"].wired:
            requested.append("model-logit")
        if "diff" in enabled:
            requested.append("model-gap")

        if "noise" in enabled:
            requested.append("noise")

        if self.policy:
            requested.extend(("Δrank", "pol-rank"))

        if "probability" in enabled:
            requested.append("raw-p")
            if self.policy:
                requested.append("pol-p")
            requested.append("decode-p")

        # Column visibility follows user preferences, never terminal geometry.
        columns = [(label, _COLUMN_WIDTHS[label]) for label in requested]
        columns.append(("token-id", _COLUMN_WIDTHS["token-id"]))
        return tuple(columns)

    @property
    def heading(self) -> str:
        return "".join(
            f"  {_COLUMN_DISPLAY_LABELS.get(label, label):>{width}}"
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
                "raw-p": candidate.model_probability,
                "pol-p": candidate.policy_probability,
                "decode-p": candidate.decoder_probability,
            }[label]
            return (
                f"{probability:.2%}"
                if probability is not None and probability > 0
                else "--"
            )

        return "".join(f"  {value(label):>{width}}" for label, width in self.columns)

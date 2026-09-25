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
    "pct": OverlaySpec(
        name="pct",
        labels=("raw-p", "pol-p"),
        label_metrics=(frozenset({"raw_probability"}), frozenset({"policy_probability"})),
    ),
    "decode_pct": OverlaySpec(
        name="decode_pct",
        labels=("decode-p",),
        label_metrics=(frozenset({"decoder_probability"}),),
    ),
    "logit": OverlaySpec(
        name="logit",
        labels=("model-logit",),
        label_metrics=(frozenset({"raw_logit"}),),
    ),
    "gap_k1": OverlaySpec(
        name="gap_k1",
        labels=("model-gap",),
        label_metrics=(frozenset({"raw_logit", "top_raw_logit"}),),
    ),
    "margin_neighbor": OverlaySpec(
        name="margin_neighbor",
        labels=("margin",),
        label_metrics=(frozenset({"neighbor_margin"}),),
        wired=True,
    ),
    "z": OverlaySpec(
        name="z",
        labels=("z",),
        label_metrics=(frozenset({"logit_z"}),),
        wired=True,
    ),
}

# Single middle-column focus cycle (wired overlays only; stubs excluded).
COLUMN_FOCUS_CYCLE: tuple[str, ...] = ("logit", "gap_k1", "margin_neighbor", "z", "pct", "decode_pct")


def next_column_focus(current: str | None) -> str:
    """Advance one step through COLUMN_FOCUS_CYCLE (wraps; None → first)."""
    if current is None or current not in COLUMN_FOCUS_CYCLE:
        return COLUMN_FOCUS_CYCLE[0]
    return COLUMN_FOCUS_CYCLE[
        (COLUMN_FOCUS_CYCLE.index(current) + 1) % len(COLUMN_FOCUS_CYCLE)
    ]


_COLUMN_WIDTHS: Mapping[str, int] = {
    "model-logit": 11,
    "model-gap": 10,
    "margin": 10,
    "z": 10,
    "Δrank": 6,
    "pol-rank": 8,
    "raw-p": 8,
    "pol-p": 8,
    "decode-p": 8,
    "token-id": 8,
}


def overlays_from_preferences(
    *,
    logit_view: str = "none",
    show_model_probabilities: bool = False,
    column_focus: str | None = None,
    overlays: frozenset[str] = frozenset(),
) -> frozenset[str]:
    """Derive the enabled overlay set from session presentation prefs.

    Shortcut and explicit overlays are additive.
    """
    enabled: set[str] = set(overlays)
    if column_focus in OVERLAYS:
        enabled.add(column_focus)
    if logit_view in {"raw", "both"}:
        enabled.add("logit")
    if logit_view in {"gap", "both"}:
        enabled.add("gap_k1")
    if show_model_probabilities:
        enabled.add("pct")
        enabled.add("decode_pct")
    return frozenset(enabled)


@dataclass(frozen=True)
class CandidateViewPlan:
    """Resolved visible columns and the only candidate metrics to calculate."""

    columns: tuple[tuple[str, int], ...]
    metrics: frozenset[str]
    order: str = "raw"

    def needs(self, metric: str) -> bool:
        return metric in self.metrics

    def policy_ordered(self) -> CandidateViewPlan:
        return CandidateViewPlan(self.columns, self.metrics | {"policy_rank"}, "policy")


@dataclass(frozen=True)
class CandidateColumns:
    policy: bool = False
    logit_view: str = "none"
    raw_k1_logit: float | None = None
    show_model_probabilities: bool = False
    column_focus: str | None = None
    overlays: frozenset[str] | None = None

    @property
    def enabled_overlays(self) -> frozenset[str]:
        return overlays_from_preferences(
            logit_view=self.logit_view,
            show_model_probabilities=self.show_model_probabilities,
            column_focus=self.column_focus,
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
        if "gap_k1" in enabled and OVERLAYS["gap_k1"].wired:
            requested.append("model-gap")
        if "margin_neighbor" in enabled and OVERLAYS["margin_neighbor"].wired:
            requested.append("margin")
        if "z" in enabled and OVERLAYS["z"].wired:
            requested.append("z")

        if self.policy:
            requested.extend(("Δrank", "pol-rank"))

        if "pct" in enabled and OVERLAYS["pct"].wired:
            requested.append("raw-p")
            if self.policy:
                requested.append("pol-p")

        if "decode_pct" in enabled and OVERLAYS["decode_pct"].wired:
            requested.append("decode-p")

        # Column visibility follows user preferences, never terminal geometry.
        columns = [(label, _COLUMN_WIDTHS[label]) for label in requested]
        columns.append(("token-id", _COLUMN_WIDTHS["token-id"]))
        return tuple(columns)

    @property
    def heading(self) -> str:
        return "".join(f"  {label:>{width}}" for label, width in self.columns)

    def values(self, candidate: Candidate) -> str:
        def value(label: str) -> str:
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
            if label == "margin":
                return (
                    f"{candidate.neighbor_margin:+.3f}"
                    if candidate.neighbor_margin is not None
                    else "--"
                )
            if label == "z":
                return (
                    f"{candidate.logit_z:+.2f}"
                    if candidate.logit_z is not None
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

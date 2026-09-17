"""Ordered description of the runtime's model-control surfaces.

The stack is intentionally descriptive in this first slice.  Existing
sampling and learning code remains authoritative; this module gives the setup
menu one vocabulary for showing which surfaces are present, in what order
they affect a decision, and which command opens their controls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .domain import SamplingConfig


@dataclass(frozen=True)
class ControllerEntry:
    """One ordered runtime influence or feedback surface."""

    phase: str
    order: int
    name: str
    state: str
    detail: str
    command: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControllerStack:
    """The ordered control surfaces visible to the setup/preflight UI."""

    entries: tuple[ControllerEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [entry.to_dict() for entry in self.entries]}

    def compact(self) -> str:
        """Return a one-line overview suitable for the setup summary."""
        return " → ".join(
            f"{entry.name} [{entry.state}]"
            for entry in self.entries
            if entry.phase in {"model", "policy"}
        )

    def render(self) -> str:
        """Render the stack with policy order and feedback separated."""
        rows = ["CONTROLLER STACK", "────────────────────────────────────────"]
        for phase, label in (
            ("model", "MODEL PREPARATION"),
            ("policy", "POLICY SURFACE"),
            ("feedback", "FEEDBACK"),
        ):
            rows.extend(("", label))
            for entry in self.entries:
                if entry.phase != phase:
                    continue
                rows.append(
                    f"  {entry.order:>2}. {entry.name:<28} {entry.state:<11} "
                    f"{entry.detail}  ({entry.command})"
                )
        return "\n".join(rows)


def _state(enabled: bool, *, configured: bool = False, unavailable: bool = False) -> str:
    if unavailable:
        return "unavailable"
    if enabled:
        return "on"
    if configured:
        return "configured"
    return "off"


def _path_detail(plan: Any, *names: str) -> str:
    values = [str(getattr(plan, name)) for name in names if getattr(plan, name, None) is not None]
    return ", ".join(values)


def build_controller_stack(
    plan: Any | None = None,
    sampling: SamplingConfig | None = None,
    *,
    backend: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> ControllerStack:
    """Build a descriptive stack from planned and/or resolved runtime state.

    ``sampling`` is preferred once backend validation has completed.  A plan
    alone still produces useful setup output by marking selected artifacts as
    ``configured`` until the launcher resolves them.
    """

    plan = plan or object()
    provenance = provenance or {}
    backend_name = backend or provenance.get("backend") or getattr(plan, "backend", None)

    if sampling is not None:
        history_active = sampling.history_penalties_active
        control_active = (
            sampling.activation_vector_layer == "control-vector"
            and bool(sampling.activation_vector)
            and sampling.activation_vector_strength != 0.0
        )
        output_active = (
            sampling.activation_vector_layer == "output"
            and bool(sampling.activation_vector)
            and sampling.activation_vector_strength != 0.0
        )
        activation_configured = bool(sampling.activation_vector)
        manual_active = bool(sampling.bias_rules or sampling.bias_groups)
        group_active = bool(sampling.group_controls)
        reference_active = sampling.reference_prior_active
        reference_configured = bool(sampling.reference_prior_routes)
        preference_active = bool(
            sampling.token_preference_vector or sampling.token_preference_fast_vector
        )
        activation_detail = (
            f"{sampling.activation_vector_layer}, strength={sampling.activation_vector_strength:g}"
            if activation_configured else "no vector"
        )
        manual_detail = (
            f"{len(sampling.bias_rules)} rules, {len(sampling.bias_groups)} groups"
            if manual_active else "no manual routes"
        )
        group_detail = (
            f"{len(sampling.group_controls)} controls"
            if group_active else "no active controls"
        )
        reference_detail = (
            f"{len(sampling.reference_prior_routes)} routes"
            if reference_configured else "no reference routes"
        )
        preference_detail = (
            f"slow={len(sampling.token_preference_vector)}, "
            f"fast={len(sampling.token_preference_fast_vector)}"
            if preference_active else "no vector memory"
        )
    else:
        history_active = any(
            getattr(plan, name, None) not in (None, 0, 1.0)
            for name in ("repeat_penalty", "presence_penalty", "frequency_penalty")
        )
        activation_configured = getattr(plan, "activation_vector", None) is not None
        control_active = False
        output_active = False
        manual_active = False
        group_active = False
        reference_active = False
        reference_configured = getattr(plan, "reference", None) is not None
        preference_active = False
        activation_detail = _path_detail(plan, "activation_vector") or "no vector selected"
        manual_detail = _path_detail(plan, "biases", "groups") or "no bias artifact selected"
        group_detail = _path_detail(plan, "groups") or "no group controls selected"
        reference_detail = _path_detail(plan, "reference") or "no reference selected"
        preference_detail = "learner state is created after launch"

    control_unavailable = control_active and backend_name not in (None, "llama.cpp")
    entries = (
        ControllerEntry(
            "model", 1, "layerwise activation", _state(
                control_active,
                configured=activation_configured and not output_active,
                unavailable=control_unavailable,
            ),
            activation_detail + ("; llama.cpp required" if control_unavailable else ""),
            "activation",
        ),
        ControllerEntry(
            "policy", 1, "base model", "on",
            "backend logits after model preparation", "model/backend",
        ),
        ControllerEntry(
            "policy", 2, "history penalties", _state(history_active),
            "repeat/presence/frequency", "sampler",
        ),
        ControllerEntry(
            "policy", 3, "output activation", _state(
                output_active,
                configured=activation_configured and not control_active,
            ),
            activation_detail if activation_configured else "no vector",
            "activation",
        ),
        ControllerEntry(
            "policy", 4, "manual biases/groups", _state(
                manual_active,
                configured=bool(_path_detail(plan, "biases", "groups")),
            ),
            manual_detail,
            "b / groups",
        ),
        ControllerEntry(
            "policy", 5, "reference prior", _state(
                reference_active,
                configured=reference_configured and not reference_active,
            ),
            reference_detail + "; scope can depend on active manual routes",
            "reference",
        ),
        ControllerEntry(
            "policy", 6, "token preference actuator", _state(preference_active),
            preference_detail,
            "preference",
        ),
        ControllerEntry(
            "policy", 7, "group control", _state(
                group_active,
                configured=bool(_path_detail(plan, "groups")),
            ),
            group_detail,
            "learning / group",
        ),
        ControllerEntry(
            "policy", 8, "sampler / token draw", "on",
            "temperature and filtering", "sampler",
        ),
        ControllerEntry(
            "feedback", 1, "manual-group learner", _state(
                bool(getattr(plan, "online_learning", False))
            ),
            "teacher updates learnable groups",
            "learning / group",
        ),
        ControllerEntry(
            "feedback", 2, "token-preference learner", _state(
                bool(getattr(plan, "token_preference", False))
            ),
            "teacher updates preference memory",
            "preference",
        ),
    )
    return ControllerStack(entries)

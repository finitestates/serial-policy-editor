"""Describe the control surfaces that belong to the core runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .core.sampler_config import SamplerConfig


@dataclass(frozen=True)
class ControllerEntry:
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
    entries: tuple[ControllerEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [entry.to_dict() for entry in self.entries]}

    def compact(self) -> str:
        return " → ".join(
            f"{entry.name} [{entry.state}]"
            for entry in self.entries
            if entry.phase in {"model", "policy"}
        )

    def render(self) -> str:
        rows = ["CONTROLLER STACK", "────────────────────────────────────────"]
        for phase, label in (("model", "MODEL PREPARATION"), ("policy", "POLICY SURFACE")):
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


def _path_detail(plan: Any, name: str) -> str:
    value = getattr(plan, name, None)
    return str(value) if value is not None else ""


def build_controller_stack(
    plan: Any | None = None,
    sampling: SamplerConfig | None = None,
    *,
    backend: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> ControllerStack:
    """Build the core model, policy, and draw ordering for the UI."""
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
        vector_configured = bool(sampling.activation_vector)
        manual_active = bool(sampling.bias_rules or sampling.bias_groups)
        vector_detail = (
            f"{('output-head' if sampling.activation_vector_layer == 'output' else 'hidden-state')}, "
            f"strength={sampling.activation_vector_strength:g}"
            if vector_configured else "no vector"
        )
        manual_detail = (
            f"{len(sampling.bias_rules)} rules, {len(sampling.bias_groups)} groups"
            if manual_active else "no manual routes"
        )
    else:
        history_active = any(
            getattr(plan, name, None) not in (None, 0, 1.0)
            for name in ("repeat_penalty", "presence_penalty", "frequency_penalty")
        )
        vector_configured = getattr(plan, "activation_vector", None) is not None
        control_active = output_active = False
        vector_detail = _path_detail(plan, "activation_vector") or "no vector selected"
        manual_active = False
        manual_detail = "no manual routes"

    control_unavailable = control_active and backend_name not in (None, "llama.cpp")
    entries = (
        ControllerEntry(
            "model", 1, "layerwise hidden-state control", _state(
                control_active,
                configured=vector_configured and not output_active,
                unavailable=control_unavailable,
            ),
            vector_detail + ("; llama.cpp required" if control_unavailable else ""),
            "vector",
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
            "policy", 3, "output-head steering", _state(
                output_active, configured=vector_configured and not control_active
            ),
            vector_detail if vector_configured else "no vector", "vector",
        ),
        ControllerEntry(
            "policy", 4, "manual biases/groups", _state(
                manual_active, configured=manual_active
            ),
            manual_detail, "b / groups",
        ),
        ControllerEntry(
            "policy", 5, "sampler / token draw", "on",
            "temperature, filtering, and draw kernel", "sampler",
        ),
    )
    return ControllerStack(entries)

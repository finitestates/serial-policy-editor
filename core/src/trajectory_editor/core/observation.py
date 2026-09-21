"""Core model-observation and policy-surface calculations.

The core observer owns the surfaces required by the interactive projector:
raw model logits, history penalties, manual/conditional biases, optional
output-head steering, candidate filtering, and replay-stable draw metadata.
Research actuators are intentionally absent. The core episode engine uses
this observer for all policy calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .sampler_config import SamplerConfig
from .sampling import (
    SparseDistribution,
    _rank,
    _softmax,
    _top_ids,
    _validated_logits,
    apply_candidate_filter,
    top_raw_ids,
)


@dataclass(frozen=True)
class ControllerTraceStage:
    """One immutable intermediate logit surface from a policy decision."""

    name: str
    phase: str
    surface: np.ndarray = field(repr=False, compare=False)
    delta: np.ndarray = field(repr=False, compare=False)
    diagnostics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        surface = np.asarray(self.surface, dtype=np.float64).copy()
        delta = np.asarray(self.delta, dtype=np.float64).copy()
        if surface.ndim != 1 or delta.shape != surface.shape:
            raise ValueError("controller trace surfaces must be equal one-dimensional arrays")
        if not np.all(np.isfinite(surface)) or not np.all(np.isfinite(delta)):
            raise ValueError("controller trace surfaces must be finite")
        surface.setflags(write=False)
        delta.setflags(write=False)
        object.__setattr__(self, "surface", surface)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def to_dict(self, *, include_surface: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "phase": self.phase,
            "shape": list(self.surface.shape),
            "diagnostics": dict(self.diagnostics),
        }
        if include_surface:
            result["surface"] = self.surface.tolist()
            result["delta"] = self.delta.tolist()
        return result


@dataclass(frozen=True)
class ControllerTrace:
    """Captured core policy surfaces; never persisted as episode state."""

    stages: tuple[ControllerTraceStage, ...]
    filtered_token_ids: tuple[int, ...] = ()

    def stage(self, name: str) -> ControllerTraceStage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)

    def to_dict(self, *, include_surfaces: bool = False) -> dict[str, object]:
        return {
            "stages": [
                stage.to_dict(include_surface=include_surfaces)
                for stage in self.stages
            ],
            "filtered_token_ids": list(self.filtered_token_ids),
        }


def _validated_history(history_token_ids, vocabulary_size: int) -> np.ndarray:
    history = np.asarray(history_token_ids, dtype=np.int64)
    if history.ndim != 1 or np.any(history < 0) or np.any(history >= vocabulary_size):
        raise ValueError("history token ids must address the decoder vocabulary")
    return history


def _history_penalty_surface(
    values: np.ndarray,
    config: SamplerConfig,
    history_token_ids: list[int] | tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply the reconstructable core history policy to raw logits."""

    if history_token_ids is None:
        if config.history_penalties_active:
            raise ValueError("active history penalties require exact prefix token ids")
        return values.copy(), np.zeros(len(values), dtype=np.int64), 0
    history = _validated_history(history_token_ids, len(values))
    if config.repeat_last_n == 0 or not len(history):
        considered = history[:0]
    elif config.repeat_last_n == -1:
        considered = history
    else:
        considered = history[-config.repeat_last_n :]
    counts = np.bincount(considered, minlength=len(values)).astype(np.int64, copy=False)
    adjusted = values.copy()
    present = counts > 0
    if float(config.repeat_penalty) != 1.0 and np.any(present):
        selected = adjusted[present]
        adjusted[present] = np.where(
            selected < 0.0,
            selected * float(config.repeat_penalty),
            selected / float(config.repeat_penalty),
        )
    if float(config.presence_penalty) != 0.0:
        adjusted[present] -= float(config.presence_penalty)
    if float(config.frequency_penalty) != 0.0:
        adjusted -= counts * float(config.frequency_penalty)
    if not np.all(np.isfinite(adjusted)):
        raise ValueError("history penalties produced non-finite policy logits")
    return adjusted, counts, int(len(considered))


class ObservationStatistics:
    """Core numeric snapshot used by the interactive projector."""

    def __init__(
        self,
        logits: Any,
        config: SamplerConfig,
        history_token_ids,
        *,
        render_tokens=None,
        activation_logit_adjustments=None,
        model_phase_diagnostics=None,
        ephemeral_logit_biases=None,
        capture_trace: bool = False,
    ) -> None:
        self.logits = _validated_logits(logits).copy()
        self.render_tokens = render_tokens
        trace_stages: list[ControllerTraceStage] = []

        def record_stage(name: str, phase: str, surface, previous=None, **details) -> None:
            if not capture_trace:
                return
            values = np.asarray(surface, dtype=np.float64)
            prior = np.zeros_like(values) if previous is None else np.asarray(previous, dtype=np.float64)
            delta = values - prior
            details = {
                "delta_rms": float(np.sqrt(np.mean(delta ** 2))),
                "delta_min": float(np.min(delta)),
                "delta_max": float(np.max(delta)),
                "affected_tokens": int(np.count_nonzero(delta)),
                **details,
            }
            trace_stages.append(ControllerTraceStage(name, phase, values, delta, details))

        if model_phase_diagnostics:
            record_stage(
                str(model_phase_diagnostics.get("name", "model-phase guidance")),
                "model",
                self.logits,
                **{
                    key: value
                    for key, value in model_phase_diagnostics.items()
                    if key != "name"
                },
            )
        record_stage("backend logits", "policy", self.logits)
        trace_previous = self.logits
        if config.history_penalties_active:
            self.adjusted, _, _ = _history_penalty_surface(
                self.logits, config, history_token_ids
            )
        else:
            if history_token_ids is not None:
                _validated_history(history_token_ids, len(self.logits))
            self.adjusted = self.logits
        record_stage("history penalties", "policy", self.adjusted, trace_previous)
        trace_previous = self.adjusted

        self.activation_logit_adjustments = np.zeros_like(self.logits)
        activation_active = (
            config.activation_vector_layer == "output"
            and config.activation_vector
            and config.activation_vector_strength != 0.0
        )
        if activation_active:
            if activation_logit_adjustments is None:
                raise ValueError(
                    "output-head steering adjustments are required when steering state is active"
                )
            raw_activation = np.asarray(activation_logit_adjustments, dtype=np.float64)
            if raw_activation.shape != self.logits.shape:
                raise ValueError(
                    "output-head steering adjustments do not match the policy vocabulary"
                )
            if not np.all(np.isfinite(raw_activation)):
                raise ValueError("output-head steering adjustments must be finite")
            self.activation_logit_adjustments = (
                float(config.activation_vector_strength) * raw_activation
            )
            if not np.all(np.isfinite(self.activation_logit_adjustments)):
                raise ValueError("output-head steering produced non-finite policy logits")
            self.adjusted = self.adjusted.copy()
            self.adjusted += self.activation_logit_adjustments
        record_stage(
            "output-head steering",
            "policy",
            self.adjusted,
            trace_previous,
            active=bool(activation_active),
        )
        trace_previous = self.adjusted
        self.activation_diagnostics = {
            "vector_norm": float(np.linalg.norm(np.asarray(config.activation_vector, dtype=np.float64)))
            if config.activation_vector
            else 0.0,
            "strength": float(config.activation_vector_strength),
            "logit_rms": float(np.sqrt(np.mean(self.activation_logit_adjustments ** 2))),
            "logit_min": float(np.min(self.activation_logit_adjustments)),
            "logit_max": float(np.max(self.activation_logit_adjustments)),
            "digest": config.activation_vector_digest,
        }

        self.active_biases = config.active_biases(history_token_ids)
        if self.active_biases:
            self.adjusted = self.adjusted.copy()
            for token, bias in self.active_biases.items():
                if token < 0 or token >= len(self.logits):
                    raise ValueError("bias token id is outside the decoder vocabulary")
                self.adjusted[token] += bias
            if not np.all(np.isfinite(self.adjusted)):
                raise ValueError("biases produced non-finite policy logits")
        record_stage(
            "manual biases",
            "policy",
            self.adjusted,
            trace_previous,
            manual_tokens=len(self.active_biases),
            reference_tokens=0,
        )
        trace_previous = self.adjusted

        self.ephemeral_logit_biases = {
            int(token): float(amount)
            for token, amount in (ephemeral_logit_biases or {}).items()
        }
        if self.ephemeral_logit_biases:
            if any(
                token < 0
                or token >= len(self.logits)
                or not np.isfinite(amount)
                for token, amount in self.ephemeral_logit_biases.items()
            ):
                raise ValueError("ephemeral logit biases are invalid")
            self.adjusted = self.adjusted.copy()
            for token, amount in self.ephemeral_logit_biases.items():
                self.adjusted[token] += amount
            if not np.all(np.isfinite(self.adjusted)):
                raise ValueError("ephemeral logit biases produced non-finite policy logits")
            record_stage(
                "temporary phrase bias",
                "policy",
                self.adjusted,
                trace_previous,
                active=True,
                controlled_tokens=len(self.ephemeral_logit_biases),
            )
            trace_previous = self.adjusted

        penalties_active = config.policy_active or bool(self.ephemeral_logit_biases)
        self.maximum = float(np.max(self.logits))
        exponentials = np.exp(self.logits - self.maximum)
        self.denominator = float(np.sum(exponentials))
        self.log_z = self.maximum + float(np.log(self.denominator))
        self.policy_probabilities = (
            _softmax(self.adjusted)
            if penalties_active
            else exponentials / self.denominator
        )
        result = apply_candidate_filter(self.adjusted, config)
        scaled = result.scaled_logits
        stages = result.stages
        ids = stages["after_min_p"]
        assert ids is not None
        self.candidate_filter_diagnostics = {
            "boundary": "candidate-filter -> draw-kernel",
            "filter": "standard",
            "typical_p": float(config.typical_p),
            "tail_free_z": float(config.tail_free_z),
            "draw_kernel": config.draw_kernel,
            "stage_counts": {
                name: None if value is None else int(len(value))
                for name, value in stages.items()
            },
        }
        self.distribution = SparseDistribution(
            ids,
            _softmax(scaled[ids]),
            np.asarray(scaled[ids], dtype=np.float64),
        )
        record_stage(
            "sampler / token draw",
            "policy",
            self.adjusted,
            trace_previous,
            filtered_tokens=len(ids),
            temperature=float(config.temperature),
            candidate_filter="standard",
            draw_kernel=config.draw_kernel,
            typical_p=float(config.typical_p),
            tail_free_z=float(config.tail_free_z),
        )
        self.controller_trace = (
            ControllerTrace(tuple(trace_stages), tuple(int(value) for value in ids))
            if capture_trace
            else None
        )
        for array in (
            self.logits,
            self.adjusted,
            self.policy_probabilities,
            self.activation_logit_adjustments,
            self.distribution.ids,
            self.distribution.probabilities,
        ):
            array.setflags(write=False)
        if self.distribution.scores is not None:
            self.distribution.scores.setflags(write=False)
        self._raw_ranks: dict[int, int] = {}
        self._policy_ranks: dict[int, int] = {} if penalties_active else self._raw_ranks
        self._ordered: list[int] = []
        self._policy_ordered: list[int] = []

    def raw_probabilities(self, token_ids):
        if self.adjusted is self.logits:
            return self.policy_probabilities[list(token_ids)]
        return np.exp(self.logits[list(token_ids)] - self.maximum) / self.denominator

    @property
    def backend_logits(self):
        return self.logits

    @property
    def policy_logits(self):
        return self.adjusted

    def model_probabilities(self, token_ids):
        return self.raw_probabilities(token_ids)

    def model_nll(self, token_id: int) -> float:
        return self.raw_nll(token_id)

    def model_rank(self, token_id: int) -> int:
        return self.raw_rank(token_id)

    def top_model_ids(self, count: int) -> list[int]:
        return self.top_raw_ids(count)

    def raw_nll(self, token_id: int) -> float:
        return self.log_z - float(self.logits[token_id])

    def raw_rank(self, token_id: int) -> int:
        if token_id not in self._raw_ranks:
            self._raw_ranks[token_id] = _rank(self.logits, token_id)
        return self._raw_ranks[token_id]

    def policy_rank(self, token_id: int) -> int:
        if token_id not in self._policy_ranks:
            self._policy_ranks[token_id] = _rank(self.adjusted, token_id)
        return self._policy_ranks[token_id]

    def top_raw_ids(self, count: int) -> list[int]:
        if count > len(self._ordered):
            self._ordered = top_raw_ids(self.logits, count)
            self._raw_ranks.update(
                (token_id, rank) for rank, token_id in enumerate(self._ordered, 1)
            )
        return self._ordered[:count]

    def top_policy_ids(self, count: int) -> list[int]:
        if self.adjusted is self.logits:
            return self.top_raw_ids(count)
        if count > len(self._policy_ordered):
            self._policy_ordered = [int(value) for value in _top_ids(self.adjusted, count)]
            self._policy_ranks.update(
                (token_id, rank) for rank, token_id in enumerate(self._policy_ordered, 1)
            )
        return self._policy_ordered[:count]


__all__ = [
    "ControllerTrace",
    "ControllerTraceStage",
    "ObservationStatistics",
]

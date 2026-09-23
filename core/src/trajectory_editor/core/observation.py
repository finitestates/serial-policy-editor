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
        # Dense V soft-max (exp+sum) is deferred until NLL / percentages are
        # requested. Ranking, top-ids, adjusted logits, and the sparse draw
        # filter do not need it.
        self._raw_maximum: float | None = None
        self._denominator: float | None = None
        self._log_z: float | None = None
        self._policy_maximum: float | None = None
        self._policy_denominator: float | None = None
        self._policy_log_z: float | None = None
        self._raw_logsumexp_ready = False
        self._policy_logsumexp_ready = False
        # Lazy full-vocab logit mean/std for z-score overlays (not soft-max).
        self._logit_mean: float | None = None
        self._logit_std: float | None = None
        self._logit_mean_std_ready = False
        self._policy_shares_raw = not (penalties_active and self.adjusted is not self.logits)
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

    def _ensure_raw_maximum(self) -> float:
        if self._raw_maximum is None:
            self._raw_maximum = float(np.max(self.logits))
        return self._raw_maximum

    # Population std floor: below this (or non-finite) z-scores are undefined.
    _LOGIT_STD_EPS = 1e-12

    def _ensure_logit_mean_std(self) -> tuple[float, float] | None:
        """Lazy mean/std of raw/backend logits (population ddof=0).

        Does not wake soft-max / logsumexp. Returns None when std is unusable.
        """
        if self._logit_mean_std_ready:
            if self._logit_mean is None or self._logit_std is None:
                return None
            return self._logit_mean, self._logit_std
        mean = float(np.mean(self.logits))
        # Population standard deviation over the full vocabulary.
        std = float(np.std(self.logits, ddof=0))
        self._logit_mean_std_ready = True
        if (
            not np.isfinite(mean)
            or not np.isfinite(std)
            or std < self._LOGIT_STD_EPS
        ):
            self._logit_mean = mean if np.isfinite(mean) else None
            self._logit_std = None
            return None
        self._logit_mean = mean
        self._logit_std = std
        return mean, std

    def logit_z(self, token_id: int) -> float | None:
        """z-score of one raw logit vs full-vocab mean/std, or None if undefined."""
        token_id = int(token_id)
        if token_id < 0 or token_id >= len(self.logits):
            raise ValueError("token id is outside the decoder vocabulary")
        stats = self._ensure_logit_mean_std()
        if stats is None:
            return None
        mean, std = stats
        return (float(self.logits[token_id]) - mean) / std

    def logit_z_scores(self, token_ids) -> list[float | None]:
        """z-scores for selected ids; None entries when mean/std is unusable."""
        ids = [int(token_id) for token_id in token_ids]
        if not ids:
            return []
        if any(token_id < 0 or token_id >= len(self.logits) for token_id in ids):
            raise ValueError("token id is outside the decoder vocabulary")
        stats = self._ensure_logit_mean_std()
        if stats is None:
            return [None] * len(ids)
        mean, std = stats
        return [(float(self.logits[token_id]) - mean) / std for token_id in ids]

    def _ensure_raw_logsumexp(self) -> None:
        if self._raw_logsumexp_ready:
            return
        maximum = self._ensure_raw_maximum()
        exponentials = np.exp(self.logits - maximum)
        denominator = float(np.sum(exponentials))
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError("model soft-max normalization failed")
        self._denominator = denominator
        self._log_z = maximum + float(np.log(denominator))
        self._raw_logsumexp_ready = True
        if self._policy_shares_raw:
            self._policy_maximum = maximum
            self._policy_denominator = denominator
            self._policy_log_z = self._log_z
            self._policy_logsumexp_ready = True

    def _ensure_policy_logsumexp(self) -> None:
        if self._policy_logsumexp_ready:
            return
        if self._policy_shares_raw:
            self._ensure_raw_logsumexp()
            return
        maximum = float(np.max(self.adjusted))
        exponentials = np.exp(self.adjusted - maximum)
        denominator = float(np.sum(exponentials))
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError("policy soft-max normalization failed")
        self._policy_maximum = maximum
        self._policy_denominator = denominator
        self._policy_log_z = maximum + float(np.log(denominator))
        self._policy_logsumexp_ready = True

    @property
    def maximum(self) -> float:
        """Top raw logit (cheap max). Does not force soft-max exp+sum."""
        return self._ensure_raw_maximum()

    @property
    def denominator(self) -> float:
        self._ensure_raw_logsumexp()
        assert self._denominator is not None
        return self._denominator

    @property
    def log_z(self) -> float:
        self._ensure_raw_logsumexp()
        assert self._log_z is not None
        return self._log_z

    def raw_probabilities(self, token_ids):
        """Return model soft-max probabilities for ``token_ids`` only."""
        ids = np.asarray(list(token_ids), dtype=np.int64)
        if ids.size == 0:
            return np.asarray([], dtype=np.float64)
        if np.any(ids < 0) or np.any(ids >= len(self.logits)):
            raise ValueError("token id is outside the decoder vocabulary")
        self._ensure_raw_logsumexp()
        assert self._denominator is not None
        return np.exp(self.logits[ids] - self.maximum) / self._denominator

    def policy_probabilities_at(self, token_ids):
        """Return policy soft-max probabilities for ``token_ids`` only."""
        ids = np.asarray(list(token_ids), dtype=np.int64)
        if ids.size == 0:
            return np.asarray([], dtype=np.float64)
        if np.any(ids < 0) or np.any(ids >= len(self.adjusted)):
            raise ValueError("token id is outside the decoder vocabulary")
        if self._policy_shares_raw:
            return self.raw_probabilities(ids)
        self._ensure_policy_logsumexp()
        assert self._policy_maximum is not None and self._policy_denominator is not None
        return np.exp(self.adjusted[ids] - self._policy_maximum) / self._policy_denominator

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

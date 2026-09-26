"""Core model-observation and policy-surface calculations.

The core observer owns the surfaces required by the interactive projector:
raw model logits, history penalties, manual/conditional biases, optional
output-head steering, candidate filtering, and replay-stable draw metadata.
Research actuators are intentionally absent. The core episode engine uses
this observer for all policy calculations.
"""

from __future__ import annotations

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


def _validated_history(history_token_ids, vocabulary_size: int) -> np.ndarray:
    history = np.asarray(history_token_ids, dtype=np.int64)
    if history.ndim != 1 or np.any(history < 0) or np.any(history >= vocabulary_size):
        raise ValueError("history token ids must address the decoder vocabulary")
    return history


def _history_penalty_surface(
    values: np.ndarray,
    config: SamplerConfig,
    history_token_ids: list[int] | tuple[int, ...] | None,
) -> np.ndarray:
    """Apply the reconstructable core history policy to raw logits."""

    if history_token_ids is None:
        if config.history_penalties_active:
            raise ValueError("active history penalties require exact prefix token ids")
        return values.copy()
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
    return adjusted


class ObservationStatistics:
    """Core numeric snapshot used by the interactive projector."""

    def __init__(
        self,
        logits: Any,
        config: SamplerConfig,
        history_token_ids,
        *,
        activation_logit_adjustments=None,
        ephemeral_logit_biases=None,
        take_logits_ownership: bool = False,
    ) -> None:
        # Direct callers retain copy semantics; EpisodeEngine transfers its
        # owned snapshot so validation and freezing need no second copy.
        validated_logits = _validated_logits(logits)
        self.logits = (
            validated_logits
            if take_logits_ownership and validated_logits.flags.owndata
            else validated_logits.copy()
        )
        if config.history_penalties_active:
            self.adjusted = _history_penalty_surface(
                self.logits, config, history_token_ids
            )
        else:
            if history_token_ids is not None:
                _validated_history(history_token_ids, len(self.logits))
            self.adjusted = self.logits
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
            activation_adjustments = (
                float(config.activation_vector_strength) * raw_activation
            )
            if not np.all(np.isfinite(activation_adjustments)):
                raise ValueError("output-head steering produced non-finite policy logits")
            self.adjusted = self.adjusted.copy()
            self.adjusted += activation_adjustments
        self.active_biases = config.active_biases(history_token_ids)
        if self.active_biases:
            self.adjusted = self.adjusted.copy()
            for token, bias in self.active_biases.items():
                if token < 0 or token >= len(self.logits):
                    raise ValueError("bias token id is outside the decoder vocabulary")
                self.adjusted[token] += bias
            if not np.all(np.isfinite(self.adjusted)):
                raise ValueError("biases produced non-finite policy logits")
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
        self.distribution = SparseDistribution(
            ids,
            _softmax(scaled[ids]),
            np.asarray(scaled[ids], dtype=np.float64),
        )
        for array in (
            self.logits,
            self.adjusted,
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


__all__ = ["ObservationStatistics"]

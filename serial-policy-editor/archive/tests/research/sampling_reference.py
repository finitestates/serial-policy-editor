"""Reference calculations for checking the production observation statistics."""

from typing import Any

import numpy as np

from trajectory_editor.domain import SamplingConfig
from trajectory_editor.sampling import (
    SparseDistribution, _validated_logits, _history_penalty_surface,
    _softmax, _rank, _stages_from_adjusted,
)


def policy_token_evidence(
    logits: np.ndarray,
    config: SamplingConfig,
    history_token_ids: list[int] | tuple[int, ...] | None,
    token_ids: list[int] | tuple[int, ...],
) -> list[dict[str, Any]]:
    """Return compact before/after policy evidence for selected token ids."""

    values = _validated_logits(logits)
    adjusted, counts, considered_count = _history_penalty_surface(
        values, config, history_token_ids
    )
    probabilities = _softmax(adjusted)
    rows: list[dict[str, Any]] = []
    for raw_token_id in token_ids:
        token_id = int(raw_token_id)
        if not 0 <= token_id < len(values):
            raise ValueError("token id is outside the decoder vocabulary")
        rows.append(
            {
                "token_id": token_id,
                "policy_rank": _rank(adjusted, token_id),
                "policy_probability": float(probabilities[token_id]),
                "policy_logit": float(adjusted[token_id]),
                "policy_logit_adjustment": float(
                    adjusted[token_id] - values[token_id]
                ),
                "history_occurrences": int(counts[token_id]),
                "history_tokens_considered": considered_count,
            }
        )
    return rows


def _sampling_stage_ids(
    values: np.ndarray,
    config: SamplingConfig,
    history_token_ids: list[int] | tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | None]]:
    """Return the exact surviving token ids after each decoder stage."""
    adjusted, _, _ = _history_penalty_surface(values, config, history_token_ids)
    return _stages_from_adjusted(adjusted, config)


def sampling_distribution(
    logits: np.ndarray,
    config: SamplingConfig,
    history_token_ids: list[int] | tuple[int, ...] | None = None,
) -> SparseDistribution:
    values = _validated_logits(logits)
    _, scaled, stages = _sampling_stage_ids(values, config, history_token_ids)
    ids = stages["after_min_p"]
    assert ids is not None
    return SparseDistribution(
        ids=ids.astype(np.int64, copy=False),
        probabilities=_softmax(scaled[ids]),
    )


def raw_probability(logits: np.ndarray, token_id: int) -> float:
    values = _validated_logits(logits)
    maximum = float(np.max(values))
    denominator = float(np.sum(np.exp(values - maximum)))
    return float(np.exp(float(values[int(token_id)]) - maximum) / denominator)


def raw_nll(logits: np.ndarray, token_id: int) -> float:
    values = _validated_logits(logits)
    maximum = float(np.max(values))
    log_z = maximum + float(np.log(np.sum(np.exp(values - maximum))))
    return log_z - float(values[int(token_id)])

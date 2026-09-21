"""Matched, context-conditioned evaluation of standalone vectors.

An episode is treated as a teacher-forced context rather than as ground truth
about the vector that happened to produce it.  Each saved context is replayed
with the target vector disabled and enabled at the same token boundaries.  The
resulting effective-logit deltas are therefore separated from the later
autoregressive trajectory, which is reported only when ``rollout=True``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .activation_vectors import (
    FORMAT as STEERING_FORMAT,
    SteeringVectorArtifact,
    model_identity_json,
)
from .domain import EditorError, SamplingConfig
from .episode_engine import EpisodeEngine
from .episode_store import EpisodeStore
from .token_preference_features import DEFAULT_PROJECTION_CHUNK_SIZE
from .vector_artifacts import (
    FORMAT as TOKEN_PREFERENCE_FORMAT,
    TokenPreferenceVectorArtifact,
)


def policy_kl(probabilities: np.ndarray, reference: np.ndarray) -> float:
    """Return KL(probabilities || reference), tolerating underflowed tails."""
    left = np.asarray(probabilities, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("KL inputs must be one-dimensional arrays of equal shape")
    mask = left > 0.0
    return float(np.sum(left[mask] * (
        np.log(np.maximum(left[mask], np.finfo(np.float64).tiny))
        - np.log(np.maximum(right[mask], np.finfo(np.float64).tiny))
    )))


FORMAT = "spe-vector-impact-v1"
DEFAULT_IMPACT_STRENGTHS = (-1.0, -0.5, 0.0, 0.5, 1.0)

VectorArtifact = SteeringVectorArtifact | TokenPreferenceVectorArtifact


def load_vector_artifact(path: Path) -> VectorArtifact:
    """Load either supported portable vector artifact kind."""
    if path.suffix.lower() == ".gguf":
        return SteeringVectorArtifact.from_path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EditorError(f"could not read vector artifact: {exc}") from exc
    if not isinstance(value, Mapping):
        raise EditorError("vector artifact must contain a JSON object")
    if value.get("format") == STEERING_FORMAT:
        return SteeringVectorArtifact.from_mapping(value)
    if value.get("format") == TOKEN_PREFERENCE_FORMAT:
        return TokenPreferenceVectorArtifact.from_mapping(value)
    raise EditorError(
        "vector artifact must use "
        f"{STEERING_FORMAT} or {TOKEN_PREFERENCE_FORMAT}"
    )


def _norm(values: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64))) if values else 0.0


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=np.float64)
    shifted = shifted - float(np.max(shifted))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def _artifact_descriptor(artifact: VectorArtifact) -> dict[str, Any]:
    if isinstance(artifact, SteeringVectorArtifact):
        return {
            "format": STEERING_FORMAT,
            "kind": artifact.kind,
            "model": dict(artifact.model),
            "dimension": artifact.dimension,
            "norm": artifact.norm,
            "stored_strength": float(artifact.strength),
            "layer": artifact.layer,
            "position": artifact.position,
            "layer_start": artifact.layer_start,
            "layer_end": artifact.layer_end,
            "method": artifact.method,
            "digest": artifact.digest,
            "source": dict(artifact.source) if artifact.source is not None else None,
        }
    identity = artifact.coordinate_identity
    return {
        "format": TOKEN_PREFERENCE_FORMAT,
        "kind": "token-preference",
        "model": dict(artifact.model),
        "dimension": artifact.dimension,
        "slow_dimension": len(artifact.token_preference_vector),
        "fast_dimension": len(artifact.token_preference_fast_vector),
        "slow_norm": _norm(
            tuple(
                float(artifact.token_preference_strength) * value
                for value in artifact.token_preference_vector
            )
        ),
        "fast_norm": _norm(
            tuple(
                float(artifact.token_preference_fast_strength) * value
                for value in artifact.token_preference_fast_vector
            )
        ),
        "stored_strength": float(artifact.token_preference_strength),
        "stored_fast_strength": float(artifact.token_preference_fast_strength),
        "coordinate_identity": (
            identity.to_dict() if identity is not None else None
        ),
        "source": dict(artifact.source) if artifact.source is not None else None,
    }


def _clear_target(sampling: SamplingConfig, kind: str) -> SamplingConfig:
    if kind == "steering":
        return replace(
            sampling,
            activation_vector=(),
            activation_vector_strength=0.0,
            activation_vector_layer="output",
            activation_vector_position="current",
            activation_vector_layer_start=None,
            activation_vector_layer_end=None,
            activation_vector_model="",
            activation_vector_digest="",
        )
    return replace(
        sampling,
        token_preference_vector=(),
        token_preference_fast_vector=(),
        token_preference_strength=0.0,
        token_preference_fast_strength=0.0,
    )


def _apply_target(
    sampling: SamplingConfig,
    artifact: VectorArtifact,
    multiplier: float,
) -> SamplingConfig:
    base = _clear_target(sampling, "steering" if isinstance(artifact, SteeringVectorArtifact) else "token-preference")
    multiplier = float(multiplier)
    sign = -1.0 if multiplier < 0.0 else 1.0
    magnitude = abs(multiplier)
    if isinstance(artifact, SteeringVectorArtifact):
        return replace(
            base,
            activation_vector=tuple(sign * value for value in artifact.vector),
            activation_vector_strength=float(artifact.strength) * magnitude,
            activation_vector_layer=artifact.layer,
            activation_vector_position=artifact.position,
            activation_vector_layer_start=artifact.layer_start,
            activation_vector_layer_end=artifact.layer_end,
            activation_vector_model=model_identity_json(artifact.model),
            activation_vector_digest=artifact.digest,
        )
    identity = artifact.coordinate_identity
    if identity is None:
        raise EditorError("token preference impact requires a coordinate identity")
    return replace(
        base,
        token_preference_vector=tuple(sign * value for value in artifact.token_preference_vector),
        token_preference_fast_vector=tuple(
            sign * value for value in artifact.token_preference_fast_vector
        ),
        token_preference_strength=float(artifact.token_preference_strength) * magnitude,
        token_preference_fast_strength=float(artifact.token_preference_fast_strength) * magnitude,
        token_preference_projection_seed=int(identity.projection_seed),
        token_preference_feature_scheme=str(identity.feature_scheme),
        token_preference_whitening_ridge=float(identity.whitening_ridge),
        token_preference_coordinate_identity=identity,
    )


@dataclass(frozen=True)
class _Snapshot:
    boundary: int
    token_id: int
    raw: np.ndarray
    effective: np.ndarray
    probabilities: np.ndarray
    temperature: float


def _context_rows(store: EpisodeStore, episode_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode = store.get_episode(episode_id)
    rows = [
        row for row in store.tokens(episode_id) if bool(row["realized_visible"])
    ]
    return episode, rows


def _clear_backend_control_vector(backend: Any) -> None:
    """Avoid carrying a prior cvector between independent analysis passes."""
    clear = getattr(backend, "clear_activation_control_vector", None)
    if callable(clear):
        try:
            clear()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise EditorError(f"could not clear the backend control vector: {exc}") from exc


def _condition_snapshots(
    store: EpisodeStore,
    episode_id: str,
    episode: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    backend: Any,
    transform: Callable[[SamplingConfig], SamplingConfig],
    *,
    max_positions: int | None,
) -> list[_Snapshot]:
    selected = list(rows if max_positions is None else rows[:max_positions])
    if not selected:
        return []
    _clear_backend_control_vector(backend)
    segments: list[tuple[SamplingConfig, str, int]] = []
    for row in selected:
        segment = store.sampling_segment(episode_id, int(row["boundary"]))
        segments.append(
            (
                transform(SamplingConfig.from_mapping(segment["sampling"])),
                str(segment["stream_fingerprint"]),
                int(segment["coordinate_offset"]),
            )
        )
    engine = EpisodeEngine(
        backend,
        sampling=segments[0][0],
        max_tokens=None,
        initial_text=str(episode["initial_text"]),
        initial_token_ids=[int(value) for value in episode["initial_token_ids"]],
        stream_fingerprint=segments[0][1],
        coordinate_offset=segments[0][2],
    )
    snapshots: list[_Snapshot] = []
    for row, (sampling, _fingerprint, _offset) in zip(selected, segments):
        if engine.sampling != sampling:
            engine.sampling = sampling
        observation = engine.observe()
        snapshots.append(
            _Snapshot(
                boundary=int(row["boundary"]),
                token_id=int(row["token_id"]),
                raw=np.asarray(observation.statistics.logits, dtype=np.float64).copy(),
                effective=np.asarray(observation.statistics.adjusted, dtype=np.float64).copy(),
                probabilities=np.asarray(
                    observation.statistics.policy_probabilities, dtype=np.float64
                ).copy(),
                temperature=float(sampling.temperature),
            )
        )
        token_id = int(row["token_id"])
        backend.eval([token_id])
        engine.visible_token_ids.append(token_id)
        engine._invalidate_observation()
    return snapshots


class _Accumulator:
    def __init__(self, vocabulary_size: int) -> None:
        self.count = 0
        self.sampling_count = 0
        self.sum_raw = np.zeros(vocabulary_size, dtype=np.float64)
        self.sum_effective = np.zeros(vocabulary_size, dtype=np.float64)
        self.sum_centered = np.zeros(vocabulary_size, dtype=np.float64)
        self.sum_sampling = np.zeros(vocabulary_size, dtype=np.float64)
        self.sum_abs_centered = np.zeros(vocabulary_size, dtype=np.float64)
        self.positive = np.zeros(vocabulary_size, dtype=np.int64)
        self.negative = np.zeros(vocabulary_size, dtype=np.int64)
        self.rms_raw = 0.0
        self.rms_effective = 0.0
        self.rms_centered = 0.0
        self.mean_abs_centered = 0.0
        self.max_abs_centered = 0.0
        self.kl = 0.0
        self.total_variation = 0.0

    def add(self, baseline: _Snapshot, active: _Snapshot) -> None:
        if baseline.raw.shape != active.raw.shape:
            raise EditorError("baseline and vector logits have different vocabulary shapes")
        raw = active.raw - baseline.raw
        effective = active.effective - baseline.effective
        centered = effective - float(np.mean(effective))
        self.sum_raw += raw
        self.sum_effective += effective
        self.sum_centered += centered
        self.sum_abs_centered += np.abs(centered)
        self.positive += centered > 0.0
        self.negative += centered < 0.0
        self.rms_raw += float(np.sqrt(np.mean(raw * raw)))
        self.rms_effective += float(np.sqrt(np.mean(effective * effective)))
        self.rms_centered += float(np.sqrt(np.mean(centered * centered)))
        self.mean_abs_centered += float(np.mean(np.abs(centered)))
        self.max_abs_centered = max(self.max_abs_centered, float(np.max(np.abs(centered))))
        self.kl += policy_kl(active.probabilities, baseline.probabilities)
        self.total_variation += 0.5 * float(
            np.sum(np.abs(active.probabilities - baseline.probabilities))
        )
        self.count += 1
        if active.temperature > 0.0:
            self.sum_sampling += centered / active.temperature
            self.sampling_count += 1

    def merge(self, other: "_Accumulator") -> None:
        if self.sum_raw.shape != other.sum_raw.shape:
            raise EditorError("cannot aggregate impact reports with different vocabularies")
        for name in (
            "sum_raw", "sum_effective", "sum_centered", "sum_sampling",
            "sum_abs_centered", "positive", "negative",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in (
            "rms_raw", "rms_effective", "rms_centered", "mean_abs_centered",
            "kl", "total_variation",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.max_abs_centered = max(self.max_abs_centered, other.max_abs_centered)
        self.count += other.count
        self.sampling_count += other.sampling_count


def _token_rows(
    order: np.ndarray,
    values: np.ndarray,
    accumulator: _Accumulator,
    backend: Any,
    *,
    top: int,
    positive: bool | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for token_id in order:
        token_id = int(token_id)
        value = float(values[token_id])
        if positive is True and value <= 0.0:
            continue
        if positive is False and value >= 0.0:
            continue
        try:
            text = backend.token_text(token_id)
        except (RuntimeError, TypeError, ValueError):
            text = ""
        signed = int(accumulator.positive[token_id] + accumulator.negative[token_id])
        result.append(
            {
                "token_id": token_id,
                "text": text,
                "mean_centered_effective_delta": value,
                "mean_abs_centered_effective_delta": float(
                    accumulator.sum_abs_centered[token_id] / accumulator.count
                ),
                "positive_fraction": float(accumulator.positive[token_id] / accumulator.count),
                "negative_fraction": float(accumulator.negative[token_id] / accumulator.count),
                "sign_consistency": float(
                    max(accumulator.positive[token_id], accumulator.negative[token_id]) / signed
                ) if signed else 0.0,
            }
        )
        if len(result) >= top:
            break
    return result


def _metrics(
    accumulator: _Accumulator,
    backend: Any,
    *,
    top: int,
    include_eog: bool,
    include_vectors: bool,
) -> dict[str, Any]:
    if accumulator.count == 0:
        return {
            "available": False,
            "positions": 0,
            "reason": "context has no recorded visible token positions",
        }
    count = accumulator.count
    mean_raw = accumulator.sum_raw / count
    mean_effective = accumulator.sum_effective / count
    mean_centered = accumulator.sum_centered / count
    eligible = np.arange(len(mean_centered), dtype=np.int64)
    if not include_eog:
        try:
            eog = {int(value) for value in backend.eog_token_ids()}
        except (RuntimeError, TypeError, ValueError):
            eog = set()
        if eog:
            eligible = np.asarray(
                [token_id for token_id in eligible if int(token_id) not in eog],
                dtype=np.int64,
            )
    positive_order = eligible[np.lexsort((eligible, -mean_centered[eligible]))]
    negative_order = eligible[np.lexsort((eligible, mean_centered[eligible]))]
    absolute_order = eligible[
        np.lexsort((eligible, -accumulator.sum_abs_centered[eligible]))
    ]
    report: dict[str, Any] = {
        "available": True,
        "positions": count,
        "eligible_token_count": int(len(eligible)),
        "raw_delta": {
            "rms": accumulator.rms_raw / count,
            "max_abs": float(np.max(np.abs(mean_raw))),
            "mean_over_vocab": float(np.mean(mean_raw)),
        },
        "effective_delta": {
            "rms": accumulator.rms_effective / count,
            "centered_rms": accumulator.rms_centered / count,
            "mean_abs_centered": accumulator.mean_abs_centered / count,
            "max_abs_centered": accumulator.max_abs_centered,
            "mean_over_vocab": float(np.mean(mean_effective)),
            "centered_mean_over_vocab": float(np.mean(mean_centered)),
        },
        "sampling_delta": (
            {
                "positions": accumulator.sampling_count,
                "rms": float(
                    np.sqrt(
                        np.mean(
                            (accumulator.sum_sampling / accumulator.sampling_count)
                            ** 2
                        )
                    )
                ),
                "mean_over_vocab": float(
                    np.mean(accumulator.sum_sampling / accumulator.sampling_count)
                ),
            }
            if accumulator.sampling_count
            else None
        ),
        "mean_policy_kl_on_vs_off": accumulator.kl / count,
        "mean_policy_total_variation": accumulator.total_variation / count,
        "top_positive": _token_rows(
            positive_order, mean_centered, accumulator, backend, top=top, positive=True
        ),
        "top_negative": _token_rows(
            negative_order, mean_centered, accumulator, backend, top=top, positive=False
        ),
        "top_absolute": _token_rows(
            absolute_order,
            mean_centered,
            accumulator,
            backend,
            top=top,
            positive=None,
        ),
    }
    if include_vectors:
        report["mean_raw_delta_vector"] = mean_raw.tolist()
        report["mean_effective_delta_vector"] = mean_effective.tolist()
        report["mean_centered_effective_delta_vector"] = mean_centered.tolist()
        if accumulator.sampling_count:
            report["mean_sampling_delta_vector"] = (
                accumulator.sum_sampling / accumulator.sampling_count
            ).tolist()
    return report


def _rollout(
    store: EpisodeStore,
    episode_id: str,
    episode: Mapping[str, Any],
    backend: Any,
    transform: Callable[[SamplingConfig], SamplingConfig],
    *,
    limit: int,
) -> dict[str, Any]:
    if limit <= 0:
        return {"tokens": [], "terminal_token_id": None}
    _clear_backend_control_vector(backend)
    first = store.sampling_segment(episode_id, 0)
    first_sampling = transform(SamplingConfig.from_mapping(first["sampling"]))
    engine = EpisodeEngine(
        backend,
        sampling=first_sampling,
        max_tokens=None,
        initial_text=str(episode["initial_text"]),
        initial_token_ids=[int(value) for value in episode["initial_token_ids"]],
        stream_fingerprint=str(first["stream_fingerprint"]),
        coordinate_offset=int(first["coordinate_offset"]),
    )
    generated: list[int] = []
    terminal: int | None = None
    for boundary in range(limit):
        segment = store.sampling_segment(episode_id, boundary)
        sampling = transform(SamplingConfig.from_mapping(segment["sampling"]))
        if engine.sampling != sampling:
            engine.sampling = sampling
        observation = engine.observe()
        token_id = int(observation.proposal_token_id)
        if backend.is_eog(token_id):
            terminal = token_id
            break
        generated.append(token_id)
        backend.eval([token_id])
        engine.visible_token_ids.append(token_id)
        engine._invalidate_observation()
    return {"tokens": generated, "terminal_token_id": terminal}


def _rollout_comparison(
    baseline: Mapping[str, Any],
    active: Mapping[str, Any],
    backend: Any,
) -> dict[str, Any]:
    left = [int(value) for value in baseline["tokens"]]
    right = [int(value) for value in active["tokens"]]
    common = 0
    while common < len(left) and common < len(right) and left[common] == right[common]:
        common += 1
    return {
        "baseline_token_count": len(left),
        "vector_token_count": len(right),
        "common_visible_token_prefix": common,
        "same_rollout": (
            left == right
            and baseline.get("terminal_token_id") == active.get("terminal_token_id")
        ),
        "first_divergence_offset": None if left == right else common,
        "baseline_terminal_token_id": baseline.get("terminal_token_id"),
        "vector_terminal_token_id": active.get("terminal_token_id"),
        "baseline_text": backend.render(left, special=False),
        "vector_text": backend.render(right, special=False),
    }


def impact_vector(
    store: EpisodeStore,
    episode_ids: Sequence[str],
    backend: Any,
    artifact: VectorArtifact,
    *,
    strengths: Sequence[float] = DEFAULT_IMPACT_STRENGTHS,
    top: int = 10,
    include_eog: bool = False,
    include_vectors: bool = False,
    projection_chunk_size: int = DEFAULT_PROJECTION_CHUNK_SIZE,
    max_positions: int | None = None,
    rollout: bool = False,
) -> dict[str, Any]:
    """Measure one portable vector against multiple saved episode contexts."""
    if not episode_ids:
        raise EditorError("impact requires at least one episode ID")
    if type(top) is not int or top < 1:
        raise EditorError("impact top must be a positive integer")
    if max_positions is not None and (type(max_positions) is not int or max_positions < 1):
        raise EditorError("max positions must be a positive integer")
    if not strengths:
        raise EditorError("impact requires at least one strength")
    strengths = tuple(float(value) for value in strengths)
    if any(not math.isfinite(value) for value in strengths):
        raise EditorError("impact strengths must be finite numbers")
    if len(set(strengths)) != len(strengths):
        raise EditorError("impact strengths must not contain duplicates")

    provenance = backend.provenance(include_model_sha256=True)
    if isinstance(artifact, SteeringVectorArtifact):
        artifact.validate_against_backend(backend, provenance)
        kind = "steering"
    else:
        artifact.validate_against_backend(
            backend,
            provenance,
            projection_chunk_size=projection_chunk_size,
        )
        kind = "token-preference"
    clear = lambda sampling: _clear_target(sampling, kind)
    resolved: list[str] = []
    for value in episode_ids:
        identifier = store.resolve_id(str(value))
        if identifier in resolved:
            raise EditorError(f"impact received duplicate episode {value!r}")
        resolved.append(identifier)

    aggregate = [_Accumulator(backend.vocabulary_size()) for _ in strengths]
    contexts: list[dict[str, Any]] = []
    for episode_id in resolved:
        episode, rows = _context_rows(store, episode_id)
        selected_count = len(rows) if max_positions is None else min(len(rows), max_positions)
        baseline_snapshots = _condition_snapshots(
            store,
            episode_id,
            episode,
            rows,
            backend,
            clear,
            max_positions=max_positions,
        )
        baseline_rollout = (
            _rollout(store, episode_id, episode, backend, clear, limit=selected_count)
            if rollout
            else None
        )
        sweep: list[dict[str, Any]] = []
        for index, multiplier in enumerate(strengths):
            transform = lambda sampling, value=multiplier: _apply_target(
                sampling, artifact, value
            )
            active_snapshots = _condition_snapshots(
                store,
                episode_id,
                episode,
                rows,
                backend,
                transform,
                max_positions=max_positions,
            )
            if len(active_snapshots) != len(baseline_snapshots):
                raise EditorError(
                    f"impact replay produced different position counts for {episode_id!r}"
                )
            local = _Accumulator(backend.vocabulary_size())
            for baseline, active in zip(baseline_snapshots, active_snapshots):
                if baseline.boundary != active.boundary:
                    raise EditorError(
                        f"impact replay lost boundary alignment for {episode_id!r}"
                    )
                local.add(baseline, active)
            aggregate[index].merge(local)
            item: dict[str, Any] = {
                "multiplier": multiplier,
                "effective_multiplier": (
                    multiplier * float(artifact.strength)
                    if isinstance(artifact, SteeringVectorArtifact)
                    else multiplier * float(artifact.token_preference_strength)
                ),
                "metrics": _metrics(
                    local,
                    backend,
                    top=top,
                    include_eog=include_eog,
                    include_vectors=include_vectors,
                ),
            }
            if rollout:
                active_rollout = _rollout(
                    store,
                    episode_id,
                    episode,
                    backend,
                    transform,
                    limit=selected_count,
                )
                item["trajectory"] = _rollout_comparison(
                    baseline_rollout or {"tokens": [], "terminal_token_id": None},
                    active_rollout,
                    backend,
                )
            sweep.append(item)
        contexts.append(
            {
                "episode_id": episode_id,
                "label": store.label(episode_id),
                "status": str(episode["status"]),
                "initial_text": str(episode["initial_text"]),
                "recorded_continuation_text": str(episode["visible_text"]),
                "metadata": dict(episode.get("metadata") or {}),
                "recorded_visible_token_count": len(rows),
                "evaluated_position_count": len(baseline_snapshots),
                "truncated": len(baseline_snapshots) < len(rows),
                "strength_sweep": sweep,
            }
        )

    return {
        "format": FORMAT,
        "vector": _artifact_descriptor(artifact),
        "workspace": str(store.path),
        "episode_ids": resolved,
        "interpretation": {
            "intervention_delta": (
                "vector-on minus vector-off effective logits under the same "
                "teacher-forced context, centered per position for ranking"
            ),
            "trajectory_delta": (
                "optional rollout difference after sampling with the vector; "
                "it includes autoregressive context divergence"
            ),
            "logit_surface": "effective deltas are measured before temperature and vocabulary filtering",
        },
        "contexts": contexts,
        "aggregate": {
            "context_count": len(contexts),
            "strength_sweep": [
                {
                    "multiplier": multiplier,
                    "metrics": _metrics(
                        accumulator,
                        backend,
                        top=top,
                        include_eog=include_eog,
                        include_vectors=include_vectors,
                    ),
                }
                for multiplier, accumulator in zip(strengths, aggregate)
            ],
        },
    }


def _text_metric(metric: Mapping[str, Any]) -> list[str]:
    if not metric.get("available"):
        return [f"  unavailable: {metric.get('reason', 'no data')}"]
    effective = metric["effective_delta"]
    sampling = metric.get("sampling_delta")
    lines = [
        f"  positions={metric['positions']} centered-rms={effective['centered_rms']:.6g} "
        f"mean-abs={effective['mean_abs_centered']:.6g} "
        f"KL={metric['mean_policy_kl_on_vs_off']:.6g}",
        f"  raw-rms={metric['raw_delta']['rms']:.6g} "
        f"effective-rms={effective['rms']:.6g} "
        f"max-centered={effective['max_abs_centered']:.6g}",
    ]
    if sampling is not None:
        lines.append(
            f"  sampling-space-rms={sampling['rms']:.6g} "
            f"({sampling['positions']} non-greedy positions)"
        )
    for title, key in (
        ("top +", "top_positive"),
        ("top -", "top_negative"),
    ):
        rows = metric[key]
        if rows:
            lines.append(
                f"  {title}: "
                + "; ".join(
                    f"{row['token_id']} {row['text']!r} "
                    f"{row['mean_centered_effective_delta']:+.4g}"
                    for row in rows
                )
            )
        else:
            lines.append(f"  {title}: none")
    return lines


def render_impact_report(report: Mapping[str, Any]) -> str:
    """Render compact top/bottom and statistics without full vocabulary arrays."""
    vector = report["vector"]
    lines = [
        "vector impact",
        f"kind: {vector['kind']} dimension={vector['dimension']}",
        f"digest: {vector.get('digest', 'coordinate-bound')}",
        f"contexts: {len(report['contexts'])}",
        "measurement: matched teacher-forced intervention deltas",
    ]
    for context in report["contexts"]:
        lines.extend(
            [
                "",
                f"context: {context['episode_id']} ({context['label']})",
                f"  positions: {context['evaluated_position_count']} / "
                f"{context['recorded_visible_token_count']}"
                + (" (truncated)" if context["truncated"] else ""),
            ]
        )
        for item in context["strength_sweep"]:
            lines.append(f"  strength {item['multiplier']:+g}:")
            lines.extend(_text_metric(item["metrics"]))
            trajectory = item.get("trajectory")
            if trajectory is not None:
                lines.append(
                    f"  trajectory: common-prefix={trajectory['common_visible_token_prefix']} "
                    f"baseline={trajectory['baseline_text']!r} "
                    f"vector={trajectory['vector_text']!r}"
                )
    lines.extend(["", "aggregate across contexts:"])
    for item in report["aggregate"]["strength_sweep"]:
        lines.append(f"strength {item['multiplier']:+g}:")
        lines.extend(_text_metric(item["metrics"]))
    return "\n".join(lines)

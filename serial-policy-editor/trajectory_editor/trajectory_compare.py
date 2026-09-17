"""Compare recorded episodes as paired trajectories and counterfactuals.

The comparison layer is deliberately analysis-only.  Episode names and labels
are treated as provenance, not as supervised truth.  A report describes what
differs between a reference episode and one or more candidate episodes under
their recorded context.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Sequence

import numpy as np

from .activation_vectors import SteeringVectorArtifact
from .domain import EditorError, SamplingConfig
from .episode_store import EpisodeStore
from .token_preference_features import (
    DEFAULT_PROJECTION_CHUNK_SIZE,
    sampling_coordinate_identity,
)
from .vector_artifacts import materialize_features


FORMAT = "spe-trajectory-compare-v1"
_TEACHER_ACTION_KINDS = {"accept", "select-raw-rank", "write"}
DEFAULT_ACTIVATION_STRENGTHS = (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5)


def _norm(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left or not right:
        return None
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    left_norm = float(np.linalg.norm(left_values))
    right_norm = float(np.linalg.norm(right_values))
    if left_norm <= 1.0e-15 or right_norm <= 1.0e-15:
        return None
    return float(np.dot(left_values, right_values) / (left_norm * right_norm))


def _identity_dict(identity: Any) -> dict[str, Any] | None:
    if identity is None:
        return None
    to_dict = getattr(identity, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    if isinstance(identity, dict):
        return dict(identity)
    return None


def _vector_state(sampling: SamplingConfig, *, include_vectors: bool) -> dict[str, Any]:
    slow = tuple(float(value) for value in sampling.token_preference_vector)
    fast = tuple(float(value) for value in sampling.token_preference_fast_vector)
    slow_effective = tuple(
        float(sampling.token_preference_strength) * value for value in slow
    )
    fast_effective = tuple(
        float(sampling.token_preference_fast_strength) * value for value in fast
    )

    result: dict[str, Any] = {
        "available": bool(slow or fast),
        "coordinate_identity": _identity_dict(
            sampling.token_preference_coordinate_identity
        ),
        "slow": {
            "dimension": len(slow),
            "strength": float(sampling.token_preference_strength),
            "norm": _norm(slow_effective),
        },
        "fast": {
            "dimension": len(fast),
            "strength": float(sampling.token_preference_fast_strength),
            "norm": _norm(fast_effective),
        },
    }
    if include_vectors:
        result["slow"]["vector"] = list(slow)
        result["slow"]["effective_vector"] = list(slow_effective)
        result["fast"]["vector"] = list(fast)
        result["fast"]["effective_vector"] = list(fast_effective)
    return result


def _sampling_summary(sampling: SamplingConfig) -> dict[str, Any]:
    return {
        "seed": int(sampling.seed),
        "temperature": float(sampling.temperature),
        "top_k": int(sampling.top_k),
        "top_p": float(sampling.top_p),
        "min_p": float(sampling.min_p),
        "repeat_penalty": float(sampling.repeat_penalty),
        "presence_penalty": float(sampling.presence_penalty),
        "frequency_penalty": float(sampling.frequency_penalty),
        "bias_rule_count": len(sampling.bias_rules),
        "bias_group_count": len(sampling.bias_groups),
        "group_control_count": len(sampling.group_controls),
        "reference_prior_route_count": len(sampling.reference_prior_routes),
        "token_preference": {
            "slow_dimension": len(sampling.token_preference_vector),
            "slow_strength": float(sampling.token_preference_strength),
            "fast_dimension": len(sampling.token_preference_fast_vector),
            "fast_strength": float(sampling.token_preference_fast_strength),
            "projection_seed": int(sampling.token_preference_projection_seed),
            "feature_scheme": str(sampling.token_preference_feature_scheme),
        },
        "steering": {
            "dimension": len(sampling.activation_vector),
            "strength": float(sampling.activation_vector_strength),
            "kind": (
                "hidden-state-vector"
                if sampling.activation_vector_layer == "control-vector"
                else "output-head-steering-vector"
            ),
            "position": str(sampling.activation_vector_position),
            "digest": str(sampling.activation_vector_digest),
        },
    }


def _action_summary(actions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    kinds = Counter(str(action["kind"]) for action in actions)
    visible_by_kind: Counter[str] = Counter()
    for action in actions:
        visible_by_kind[str(action["kind"])] += max(
            0,
            int(action["boundary_after"]) - int(action["boundary_before"]),
        )
    return {
        "count": len(actions),
        "kinds": {key: kinds[key] for key in sorted(kinds)},
        "teacher_action_count": sum(
            count for kind, count in kinds.items() if kind in _TEACHER_ACTION_KINDS
        ),
        "visible_tokens_by_kind": {
            key: visible_by_kind[key] for key in sorted(visible_by_kind)
        },
    }


def _episode_record(store: EpisodeStore, episode_id: str) -> dict[str, Any]:
    episode = store.get_episode(episode_id)
    tokens = [
        row for row in store.tokens(episode_id) if bool(row["realized_visible"])
    ]
    actions = store.actions(episode_id)
    sampling = store.final_sampling(episode_id)
    continuation_text = str(episode["visible_text"])
    initial_text = str(episode["initial_text"])
    return {
        "episode_id": episode_id,
        "label": store.label(episode_id),
        "status": str(episode["status"]),
        "parent_episode_id": episode.get("parent_episode_id"),
        "fork_boundary": episode.get("fork_boundary"),
        "created_at": episode.get("created_at"),
        "finished_at": episode.get("finished_at"),
        "terminal_reason": episode.get("terminal_reason"),
        "initial_text": initial_text,
        "continuation_text": continuation_text,
        "text": initial_text + continuation_text,
        "initial_token_ids": tuple(int(value) for value in episode["initial_token_ids"]),
        "tokens": tokens,
        "actions": actions,
        "sampling": sampling,
        "metadata": dict(episode.get("metadata") or {}),
    }


def _public_episode_record(
    record: dict[str, Any], *, include_vectors: bool
) -> dict[str, Any]:
    sampling = record["sampling"]
    result = {
        "episode_id": record["episode_id"],
        "label": record["label"],
        "status": record["status"],
        "parent_episode_id": record["parent_episode_id"],
        "fork_boundary": record["fork_boundary"],
        "created_at": record["created_at"],
        "finished_at": record["finished_at"],
        "terminal_reason": record["terminal_reason"],
        "initial_text": record["initial_text"],
        "continuation_text": record["continuation_text"],
        "text": record["text"],
        "initial_token_count": len(record["initial_token_ids"]),
        "visible_token_count": len(record["tokens"]),
        "action_summary": _action_summary(record["actions"]),
        "sampling": _sampling_summary(sampling),
        "learner_vector": _vector_state(
            sampling,
            include_vectors=include_vectors,
        ),
        "metadata": record["metadata"],
    }
    return result


def _token_descriptor(rows: Sequence[dict[str, Any]], index: int) -> dict[str, Any] | None:
    if index >= len(rows):
        return None
    row = rows[index]
    return {
        "offset": int(index),
        "boundary": int(row["boundary"]),
        "token_id": int(row["token_id"]),
        "text": str(row["text"]),
    }


def _continuation_comparison(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    reference_rows = reference["tokens"]
    candidate_rows = candidate["tokens"]
    common = 0
    while (
        common < len(reference_rows)
        and common < len(candidate_rows)
        and int(reference_rows[common]["token_id"])
        == int(candidate_rows[common]["token_id"])
    ):
        common += 1

    if common == len(reference_rows) == len(candidate_rows):
        first_divergence = None
    else:
        first_divergence = {
            "offset": common,
            "reason": (
                "different-token"
                if common < len(reference_rows) and common < len(candidate_rows)
                else "reference-ended"
                if common == len(reference_rows) and common < len(candidate_rows)
                else "candidate-ended"
            ),
            "reference": _token_descriptor(reference_rows, common),
            "candidate": _token_descriptor(candidate_rows, common),
        }

    shorter = min(len(reference_rows), len(candidate_rows))
    return {
        "reference_token_count": len(reference_rows),
        "candidate_token_count": len(candidate_rows),
        "common_visible_token_prefix": common,
        "same_continuation": first_divergence is None,
        "first_divergence": first_divergence,
        "reference_tail_token_count": len(reference_rows) - shorter,
        "candidate_tail_token_count": len(candidate_rows) - shorter,
    }


def _coordinate_compatible(
    reference: SamplingConfig,
    candidate: SamplingConfig,
    dimension: int,
) -> bool:
    left = sampling_coordinate_identity(reference, dimension=dimension)
    right = sampling_coordinate_identity(candidate, dimension=dimension)
    return all(
        left_value == right_value
        or left_value is None
        or right_value is None
        for left_value, right_value in zip(left.basis_key, right.basis_key)
    )


def _channel_delta(
    reference: SamplingConfig,
    candidate: SamplingConfig,
    *,
    fast: bool,
    include_vectors: bool,
) -> tuple[dict[str, Any] | None, tuple[float, ...] | None]:
    name = "fast" if fast else "slow"
    reference_raw = tuple(
        float(value)
        for value in (
            reference.token_preference_fast_vector
            if fast
            else reference.token_preference_vector
        )
    )
    candidate_raw = tuple(
        float(value)
        for value in (
            candidate.token_preference_fast_vector
            if fast
            else candidate.token_preference_vector
        )
    )
    if not reference_raw and not candidate_raw:
        return None, None
    if reference_raw and candidate_raw and len(reference_raw) != len(candidate_raw):
        return {
            "available": False,
            "reason": f"{name} vector dimensions differ",
            "reference_dimension": len(reference_raw),
            "candidate_dimension": len(candidate_raw),
        }, None
    dimension = len(reference_raw or candidate_raw)
    if not _coordinate_compatible(reference, candidate, dimension):
        return {
            "available": False,
            "reason": f"{name} vectors use incompatible coordinate bases",
        }, None
    reference_strength = float(
        reference.token_preference_fast_strength
        if fast
        else reference.token_preference_strength
    )
    candidate_strength = float(
        candidate.token_preference_fast_strength
        if fast
        else candidate.token_preference_strength
    )
    reference_effective = np.zeros(dimension, dtype=np.float64)
    candidate_effective = np.zeros(dimension, dtype=np.float64)
    if reference_raw:
        reference_effective[:] = reference_strength * np.asarray(
            reference_raw, dtype=np.float64
        )
    if candidate_raw:
        candidate_effective[:] = candidate_strength * np.asarray(
            candidate_raw, dtype=np.float64
        )
    delta = candidate_effective - reference_effective
    result: dict[str, Any] = {
        "available": True,
        "dimension": dimension,
        "reference_present": bool(reference_raw),
        "candidate_present": bool(candidate_raw),
        "reference_norm": _norm(reference_effective.tolist()),
        "candidate_norm": _norm(candidate_effective.tolist()),
        "delta_norm": _norm(delta.tolist()),
        "cosine_reference_candidate": _cosine(
            reference_effective.tolist(), candidate_effective.tolist()
        ),
        "cosine_reference_delta": _cosine(
            reference_effective.tolist(), delta.tolist()
        ),
    }
    if include_vectors:
        result["reference_effective_vector"] = reference_effective.tolist()
        result["candidate_effective_vector"] = candidate_effective.tolist()
        result["delta_vector"] = delta.tolist()
    return result, tuple(float(value) for value in delta)


def _learner_delta(
    reference: SamplingConfig,
    candidate: SamplingConfig,
    *,
    include_vectors: bool,
) -> tuple[dict[str, Any], dict[str, tuple[float, ...]]]:
    slow, slow_delta = _channel_delta(
        reference,
        candidate,
        fast=False,
        include_vectors=include_vectors,
    )
    fast, fast_delta = _channel_delta(
        reference,
        candidate,
        fast=True,
        include_vectors=include_vectors,
    )
    usable = any(
        value is not None and bool(value.get("available"))
        for value in (slow, fast)
    )
    report: dict[str, Any] = {
        "available": usable,
        "interpretation": (
            "candidate effective learner state minus reference effective learner state"
            if usable
            else "learner vector comparison is unavailable for these episodes"
        ),
        "slow": slow,
        "fast": fast,
    }
    return report, {
        "slow": slow_delta or (),
        "fast": fast_delta or (),
    }


def _content_vectors(
    records: Sequence[dict[str, Any]],
    backend: Any,
    *,
    feature_dimension: int,
    projection_chunk_size: int,
) -> tuple[dict[str, Any], list[tuple[float, ...]]]:
    reference_sampling = records[0]["sampling"]
    identity = sampling_coordinate_identity(
        reference_sampling,
        dimension=feature_dimension,
    )
    identity_method = getattr(backend, "token_preference_coordinate_identity", None)
    if callable(identity_method):
        identity = identity_method(
            feature_dimension=identity.dimension,
            projection_seed=identity.projection_seed,
            feature_scheme=identity.feature_scheme,
            whitening_ridge=identity.whitening_ridge,
        )
    features = materialize_features(
        backend,
        identity,
        projection_chunk_size=projection_chunk_size,
    )
    vectors: list[tuple[float, ...]] = []
    for record in records:
        token_ids = [int(row["token_id"]) for row in record["tokens"]]
        if token_ids:
            values = np.asarray(features[token_ids], dtype=np.float64)
            mean = np.mean(values, axis=0)
        else:
            mean = np.zeros(int(identity.dimension), dtype=np.float64)
        vectors.append(tuple(float(value) for value in mean))
    return {
        "basis": _identity_dict(identity),
        "reference_token_count": len(records[0]["tokens"]),
        "dimension": int(identity.dimension),
    }, vectors


def _simple_direction(
    reference_vector: Sequence[float],
    candidate_vector: Sequence[float],
    *,
    include_vectors: bool,
) -> dict[str, Any]:
    if len(reference_vector) != len(candidate_vector):
        return {
            "available": False,
            "reason": "vectors have different dimensions",
            "reference_dimension": len(reference_vector),
            "candidate_dimension": len(candidate_vector),
        }
    reference_values = np.asarray(reference_vector, dtype=np.float64)
    candidate_values = np.asarray(candidate_vector, dtype=np.float64)
    delta = candidate_values - reference_values
    result: dict[str, Any] = {
        "available": True,
        "dimension": len(delta),
        "reference_norm": _norm(reference_values.tolist()),
        "candidate_norm": _norm(candidate_values.tolist()),
        "delta_norm": _norm(delta.tolist()),
        "cosine_reference_candidate": _cosine(
            reference_values.tolist(), candidate_values.tolist()
        ),
        "cosine_reference_delta": _cosine(
            reference_values.tolist(), delta.tolist()
        ),
    }
    if include_vectors:
        result["reference_vector"] = reference_values.tolist()
        result["candidate_vector"] = candidate_values.tolist()
        result["delta_vector"] = delta.tolist()
    return result


def _activation_contrast(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    backend: Any,
    *,
    capture_position: str,
    normalize: bool,
    activation_strengths: Sequence[float],
    include_vectors: bool,
) -> dict[str, Any]:
    provenance = backend.provenance(include_model_sha256=False)
    artifact = SteeringVectorArtifact.from_prompt_pair(
        backend,
        provenance,
        candidate["text"],
        reference["text"],
        capture_position=capture_position,
        normalize=normalize,
    )
    result: dict[str, Any] = {
        "available": True,
        "direction": "candidate minus reference",
        "dimension": artifact.dimension,
        "norm": artifact.norm,
        "target": artifact.target_description,
        "method": artifact.method,
        "digest": artifact.digest,
        "model": dict(artifact.model),
        "source": dict(artifact.source or {}),
        "strength_sweep": [
            {
                "multiplier": float(multiplier),
                "effective_norm": abs(float(multiplier)) * artifact.norm,
                "orientation": (
                    "opposite"
                    if float(multiplier) < 0.0
                    else "zero"
                    if float(multiplier) == 0.0
                    else "same"
                ),
            }
            for multiplier in activation_strengths
        ],
    }
    if include_vectors:
        result["vector"] = list(artifact.vector)
    return result


def _context_report(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    initial_prefixes = [record["initial_token_ids"] for record in records]
    reference_prefix = initial_prefixes[0]
    same_initial_prefix = all(prefix == reference_prefix for prefix in initial_prefixes[1:])
    parent_ids = [record["parent_episode_id"] for record in records]
    fork_boundaries = [record["fork_boundary"] for record in records]
    warnings: list[str] = []
    if not same_initial_prefix:
        warnings.append("episodes do not share the same initial token prefix")
    if len(set(parent_ids)) > 1:
        warnings.append("episodes do not share the same recorded parent episode")
    if len(set(fork_boundaries)) > 1:
        warnings.append("episodes do not share the same recorded fork boundary")
    counts = [len(record["tokens"]) for record in records]
    if len(set(counts)) > 1:
        warnings.append("episodes have different visible-token spans")
    return {
        "same_initial_token_prefix": same_initial_prefix,
        "initial_token_count": len(reference_prefix),
        "parent_episode_ids": parent_ids,
        "fork_boundaries": fork_boundaries,
        "same_parent_episode": len(set(parent_ids)) == 1,
        "same_fork_boundary": len(set(fork_boundaries)) == 1,
        "visible_token_counts": counts,
        "equal_visible_token_span": len(set(counts)) == 1,
        "warnings": warnings,
    }


def compare_episodes(
    store: EpisodeStore,
    episode_ids: Sequence[str],
    *,
    backend: Any | None = None,
    include_content: bool = False,
    include_hidden_state: bool = False,
    feature_dimension: int = 64,
    projection_chunk_size: int = DEFAULT_PROJECTION_CHUNK_SIZE,
    capture_position: str = "last",
    normalize_activation: bool = True,
    activation_strengths: Sequence[float] = DEFAULT_ACTIVATION_STRENGTHS,
    include_vectors: bool = False,
) -> dict[str, Any]:
    """Build a reference-vs-candidate report without mutating the workspace."""
    if len(episode_ids) < 2:
        raise EditorError("compare requires at least two episode IDs")
    if type(feature_dimension) is not int or feature_dimension < 1:
        raise EditorError("feature dimension must be a positive integer")
    if capture_position not in {"first", "last"}:
        raise EditorError("hidden-state capture position must be first or last")
    if any(not math.isfinite(float(value)) for value in activation_strengths):
        raise EditorError("hidden-state strengths must be finite numbers")
    if (include_content or include_hidden_state) and backend is None:
        raise EditorError(
            "--model is required when content or hidden-state vectors are requested"
        )

    resolved_ids: list[str] = []
    for value in episode_ids:
        identifier = store.resolve_id(str(value))
        if identifier in resolved_ids:
            raise EditorError(f"compare received duplicate episode {value!r}")
        resolved_ids.append(identifier)
    records = [_episode_record(store, identifier) for identifier in resolved_ids]

    content_basis: dict[str, Any] | None = None
    content_vectors: list[tuple[float, ...]] = []
    if include_content:
        try:
            content_basis, content_vectors = _content_vectors(
                records,
                backend,
                feature_dimension=feature_dimension,
                projection_chunk_size=projection_chunk_size,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise EditorError(f"could not compute content-feature vectors: {exc}") from exc

    comparisons: list[dict[str, Any]] = []
    learner_deltas: dict[str, list[tuple[float, ...]]] = {"slow": [], "fast": []}
    content_deltas: list[tuple[float, ...]] = []
    reference = records[0]
    for index, candidate in enumerate(records[1:], start=1):
        learner_report, learner_delta = _learner_delta(
            reference["sampling"],
            candidate["sampling"],
            include_vectors=include_vectors,
        )
        for channel in ("slow", "fast"):
            if learner_delta[channel]:
                learner_deltas[channel].append(learner_delta[channel])
        comparison: dict[str, Any] = {
            "candidate_episode_id": candidate["episode_id"],
            "candidate_label": candidate["label"],
            "direction": "candidate minus reference",
            "continuation": _continuation_comparison(reference, candidate),
            "learner_vector": learner_report,
        }
        if include_content:
            content_report = _simple_direction(
                content_vectors[0],
                content_vectors[index],
                include_vectors=include_vectors,
            )
            content_report["interpretation"] = (
                "mean fixed token-feature vector of candidate continuation "
                "minus reference continuation"
            )
            comparison["content_feature_vector"] = content_report
            if len(content_vectors[0]) == len(content_vectors[index]):
                content_deltas.append(
                    tuple(
                        float(candidate_value - reference_value)
                        for reference_value, candidate_value in zip(
                            content_vectors[0], content_vectors[index]
                        )
                    )
                )
        if include_hidden_state:
            try:
                comparison["hidden_state_vector"] = _activation_contrast(
                    reference,
                    candidate,
                    backend,
                    capture_position=capture_position,
                    normalize=normalize_activation,
                    activation_strengths=activation_strengths,
                    include_vectors=include_vectors,
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                raise EditorError(
                    "could not compute hidden-state contrast for "
                    f"{candidate['episode_id']!r}: {exc}"
                ) from exc
        comparisons.append(comparison)

    aggregate: dict[str, Any] = {}
    for channel in ("slow", "fast"):
        deltas = learner_deltas[channel]
        if deltas and all(len(value) == len(deltas[0]) for value in deltas):
            mean = np.mean(np.asarray(deltas, dtype=np.float64), axis=0)
            aggregate[f"learner_{channel}_delta"] = {
                "count": len(deltas),
                "dimension": len(mean),
                "norm": _norm(mean.tolist()),
                **({"mean_delta_vector": mean.tolist()} if include_vectors else {}),
            }
    if content_deltas and all(len(value) == len(content_deltas[0]) for value in content_deltas):
        mean = np.mean(np.asarray(content_deltas, dtype=np.float64), axis=0)
        aggregate["content_feature_delta"] = {
            "count": len(content_deltas),
            "dimension": len(mean),
            "norm": _norm(mean.tolist()),
            **({"mean_delta_vector": mean.tolist()} if include_vectors else {}),
        }

    return {
        "format": FORMAT,
        "reference_episode_id": reference["episode_id"],
        "episode_ids": resolved_ids,
        "context": _context_report(records),
        "episodes": [
            _public_episode_record(record, include_vectors=include_vectors)
            for record in records
        ],
        "comparisons": comparisons,
        "aggregate": aggregate,
        **(
            {"content_feature_basis": content_basis}
            if content_basis is not None
            else {}
        ),
    }


def _format_token(value: dict[str, Any] | None) -> str:
    if value is None:
        return "<end>"
    return f"{value['token_id']} {value['text']!r}"


def render_compare_report(report: dict[str, Any]) -> str:
    """Render the compact human-facing form of a comparison report."""
    context = report["context"]
    lines = [
        "trajectory comparison",
        f"reference: {report['reference_episode_id']}",
        "initial prefix: "
        + ("shared" if context["same_initial_token_prefix"] else "different"),
        "visible-token span: "
        + ("equal" if context["equal_visible_token_span"] else "different")
        + " ("
        + ", ".join(str(value) for value in context["visible_token_counts"])
        + ")",
    ]
    for warning in context["warnings"]:
        lines.append(f"warning: {warning}")
    for comparison in report["comparisons"]:
        continuation = comparison["continuation"]
        lines.extend(
            [
                "",
                f"candidate: {comparison['candidate_episode_id']}",
                f"  tokens: {continuation['candidate_token_count']} · "
                f"common prefix: {continuation['common_visible_token_prefix']}",
            ]
        )
        divergence = continuation["first_divergence"]
        if divergence is None:
            lines.append("  divergence: none")
        else:
            lines.append(
                f"  divergence: offset {divergence['offset']} "
                f"({divergence['reason']}) · "
                f"reference={_format_token(divergence['reference'])} · "
                f"candidate={_format_token(divergence['candidate'])}"
            )
        learner = comparison["learner_vector"]
        if not learner["available"]:
            lines.append("  learner vector: unavailable")
        else:
            for channel in ("slow", "fast"):
                value = learner.get(channel)
                if value and value.get("available"):
                    lines.append(
                        f"  learner {channel}: delta-norm={value['delta_norm']:.6g}"
                    )
        content = comparison.get("content_feature_vector")
        if content is not None:
            if content.get("available"):
                lines.append(
                    f"  content features: delta-norm={content['delta_norm']:.6g}"
                )
            else:
                lines.append("  content features: unavailable")
        hidden_state = comparison.get("hidden_state_vector")
        if hidden_state is not None:
            if hidden_state.get("available"):
                lines.append(
                    f"  hidden state: dimension={hidden_state['dimension']} "
                    f"norm={hidden_state['norm']:.6g} digest={hidden_state['digest'][:12]}"
                )
            else:
                lines.append("  hidden state: unavailable")
    if report["aggregate"]:
        lines.extend(["", "aggregate candidate-minus-reference directions:"])
        for name, value in sorted(report["aggregate"].items()):
            lines.append(
                f"  {name}: count={value['count']} "
                f"dimension={value['dimension']} norm={value['norm']:.6g}"
            )
    return "\n".join(lines)

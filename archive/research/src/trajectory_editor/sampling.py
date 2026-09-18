"""Replayable sampling and ranked raw-logit menus."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .domain import SamplingConfig
from .core.sampling import (
    CandidateFilterResult,
    SparseDistribution,
    StandardCandidateFilter,
    _rank,
    _softmax,
    _top_ids,
    _validated_logits,
    apply_candidate_filter,
    draw_token,
    position_uniform,
    position_uniform_token,
    raw_rank,
    top_raw_ids,
)
from .core.observation import ControllerTrace, ControllerTraceStage


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


def calibrated_token_preference_gain(
    probabilities: np.ndarray,
    scores: np.ndarray,
    target_kl: float,
    *,
    min_gain: float = 0.0,
    max_gain: float = 8.0,
) -> float:
    """Find the nonnegative gain that gives a requested policy KL change.

    The exact exponential-tilt KL equation is one-dimensional.  Bisection
    keeps this deterministic and costs little compared with model inference.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    a = np.asarray(scores, dtype=np.float64)
    if p.shape != a.shape or p.ndim != 1:
        raise ValueError("preference calibration inputs must have equal 1-D shapes")
    if (
        not np.all(np.isfinite(p)) or np.any(p < 0.0)
        or not np.all(np.isfinite(a)) or float(np.sum(p)) <= 0.0
    ):
        raise ValueError("preference calibration inputs must be finite probabilities and scores")
    p = p / float(np.sum(p))
    centered = a - float(np.dot(p, a))
    variance = float(np.dot(p, centered * centered))
    target = float(target_kl)
    if variance <= 1.0e-24 or target <= 0.0 or max_gain <= 0.0:
        return 0.0

    log_probability = np.log(np.maximum(p, np.finfo(np.float64).tiny))

    def exact_kl(gain: float) -> float:
        tilted = log_probability + float(gain) * centered
        maximum = float(np.max(tilted))
        weights = np.exp(tilted - maximum)
        normalizer = maximum + math.log(float(np.sum(weights)))
        tilted_probabilities = weights / float(np.sum(weights))
        return float(gain * np.dot(tilted_probabilities, centered) - normalizer)

    lower = max(0.0, float(min_gain))
    upper = float(max_gain)
    if lower >= upper:
        return upper
    if exact_kl(lower) >= target:
        return lower
    if exact_kl(upper) <= target:
        return upper
    for _ in range(64):
        midpoint = (lower + upper) * 0.5
        if exact_kl(midpoint) < target:
            lower = midpoint
        else:
            upper = midpoint
    return float((lower + upper) * 0.5)


def _validated_history(history_token_ids, vocabulary_size: int) -> np.ndarray:
    history = np.asarray(history_token_ids, dtype=np.int64)
    if history.ndim != 1 or np.any(history < 0) or np.any(history >= vocabulary_size):
        raise ValueError("history token ids must address the decoder vocabulary")
    return history


@dataclass
class _ReferenceTrieNode:
    children: dict[int, int]
    failure: int = 0
    descendant_mass: float = 0.0
    terminal_mass: float = 0.0
    prefix: tuple[int, ...] = ()


class ReferencePriorTrie:
    """Weighted token trie with deterministic suffix/failure transitions."""

    def __init__(self, routes) -> None:
        self.nodes = [_ReferenceTrieNode(children={})]
        for route, weight in routes:
            self._insert(tuple(route), float(weight))
        self._build_failure_links()

    def _insert(self, route: tuple[int, ...], weight: float) -> None:
        node_index = 0
        self.nodes[node_index].descendant_mass += weight
        prefix: list[int] = []
        for token in route:
            prefix.append(int(token))
            child = self.nodes[node_index].children.get(int(token))
            if child is None:
                child = len(self.nodes)
                self.nodes[node_index].children[int(token)] = child
                self.nodes.append(_ReferenceTrieNode(
                    children={}, prefix=tuple(prefix)
                ))
            node_index = child
            self.nodes[node_index].descendant_mass += weight
        self.nodes[node_index].terminal_mass += weight

    def _build_failure_links(self) -> None:
        queue: deque[int] = deque()
        for child in self.nodes[0].children.values():
            self.nodes[child].failure = 0
            queue.append(child)
        while queue:
            node_index = queue.popleft()
            node = self.nodes[node_index]
            for token, child in node.children.items():
                failure = node.failure
                while failure and token not in self.nodes[failure].children:
                    failure = self.nodes[failure].failure
                self.nodes[child].failure = self.nodes[failure].children.get(token, 0)
                queue.append(child)

    def transition(self, state: int, token: int) -> int:
        token = int(token)
        while state and token not in self.nodes[state].children:
            state = self.nodes[state].failure
        return self.nodes[state].children.get(token, 0)

    def state_for_history(self, history) -> int:
        state = 0
        for token in history:
            state = self.transition(state, int(token))
        # A completed terminal route releases the state unless it is also a
        # prefix of a longer route.  Failure preserves overlapping suffixes.
        while state and not self.nodes[state].children:
            state = self.nodes[state].failure
        return state

    def outgoing(self, state: int) -> tuple[tuple[int, float], ...]:
        node = self.nodes[state]
        return tuple(
            (token, self.nodes[child].descendant_mass)
            for token, child in sorted(node.children.items())
        )

    def diagnostics(self, state: int) -> dict[str, object]:
        node = self.nodes[state]
        return {
            "state_prefix": list(node.prefix),
            "root_mass": self.nodes[0].descendant_mass,
            "state_mass": node.descendant_mass,
            "terminal_mass": node.terminal_mass,
            "outgoing": {
                token: mass for token, mass in self.outgoing(state)
            },
        }


@lru_cache(maxsize=32)
def reference_trie(routes):
    """Reuse immutable route structure when a strength or learner state changes."""
    return ReferencePriorTrie(routes)


@dataclass(frozen=True)
class ReferencePriorSnapshot:
    scope: str
    mode: str
    state_prefix: tuple[int, ...]
    root_mass: float
    state_mass: float
    terminal_mass: float
    outgoing: tuple[tuple[int, float, float, float, float, float], ...]
    biases: dict[int, float]


def reference_prior_snapshot(
    routes,
    history_token_ids,
    *,
    active_routes=None,
    strength: float,
    attraction: float,
    exit_strength: float = 0.25,
    scope: str = "global",
    mode: str = "contrastive",
    trie: ReferencePriorTrie | None = None,
) -> ReferencePriorSnapshot:
    """Evaluate one stateful lexical prior from model-visible token history."""

    # Accept the pre-mode scope names at this low-level boundary so catalogs
    # or callers created by the previous experimental interface remain easy
    # to inspect while the saved runtime representation stays normalized.
    if scope == "ballistic-global":
        scope, mode = "global", "ballistic"
    elif scope == "ballistic-global-exit":
        scope, mode = "global", "ballistic-exit"
    if scope not in {"active", "global"}:
        raise ValueError("reference prior scope must be active or global")
    if mode not in {"lexical", "contrastive", "contrastive-exit", "ballistic", "ballistic-exit"}:
        raise ValueError("unknown reference prior mode")

    selected_routes = routes
    if active_routes is not None:
        selected_routes = tuple(
            (route, weight) for route, weight in routes if route in active_routes
        )
    if not selected_routes:
        return ReferencePriorSnapshot(scope, mode, (), 0.0, 0.0, 0.0, (), {})
    if history_token_ids is None:
        if any(len(route) > 1 for route, _weight in selected_routes):
            raise ValueError("reference priors require exact context token IDs")
        history = ()
    else:
        history = tuple(int(token) for token in history_token_ids)
    if trie is None or active_routes is not None:
        trie = reference_trie(tuple(selected_routes))
    state = trie.state_for_history(history)
    node = trie.nodes[state]
    outgoing = trie.outgoing(state)
    if not outgoing:
        return ReferencePriorSnapshot(
            scope, mode, node.prefix, trie.nodes[0].descendant_mass,
            node.descendant_mass, node.terminal_mass, (), {},
        )

    log_masses = {token: math.log(mass) for token, mass in outgoing}
    center = sum(log_masses.values()) / len(log_masses)
    continuation_mass = max(0.0, node.descendant_mass - node.terminal_mass)
    exit_continue = 0.0
    if mode.endswith("-exit") and node.terminal_mass > 0.0 and continuation_mass > 0.0:
        # Terminal mass is an implicit EXIT option. Since no decoder token
        # represents EXIT, apply its log-odds against CONTINUE uniformly to
        # all continuation children. This is separate from relative child
        # scoring and from lexical commitment attraction.
        exit_continue = float(exit_strength) * math.log(
            continuation_mass / node.terminal_mass
        )

    state_attraction = 0.0
    root_attraction = mode.startswith("ballistic")
    if attraction > 0.0 and (state or root_attraction):
        # The constant term makes a singleton continuation attractive even
        # when it is the only route and therefore has no branch contrast.
        state_attraction = float(attraction) * (
            1.0 + math.log(trie.nodes[0].descendant_mass / node.descendant_mass)
        )
    rows = []
    biases = {}
    for token, mass in outgoing:
        branch = float(strength) * (log_masses[token] - center)
        total = branch + state_attraction + exit_continue
        if mode == "lexical":
            # Relative preference at every lexical branch, with modest support
            # for completing a prefix the model has already entered. Root
            # words receive no unconditional attraction. Extreme weights have
            # bounded influence and unlisted decoder tokens stay available.
            commitment = float(strength) if state and not node.terminal_mass else 0.0
            if state and node.terminal_mass and continuation_mass:
                commitment = float(strength) * math.log(continuation_mass / node.terminal_mass)
            total = max(-2.0, min(2.0, branch + commitment))
            state_attraction = commitment
        rows.append((token, mass, branch, state_attraction, exit_continue, total))
        biases[token] = total
    return ReferencePriorSnapshot(
        scope, mode, node.prefix, trie.nodes[0].descendant_mass,
        node.descendant_mass, node.terminal_mass, tuple(rows), biases,
    )


def reference_prior_biases(
    routes,
    history_token_ids,
    *,
    active_routes=None,
    strength: float,
    attraction: float = 0.0,
    exit_strength: float = 0.25,
    scope: str = "global",
    mode: str = "contrastive",
) -> dict[int, float]:
    """Compatibility wrapper returning only online prior logit adjustments."""

    return reference_prior_snapshot(
        routes,
        history_token_ids,
        active_routes=active_routes,
        strength=strength,
        attraction=attraction,
        exit_strength=exit_strength,
        scope=scope,
        mode=mode,
    ).biases


def _history_penalty_surface(
    values: np.ndarray,
    config: SamplingConfig,
    history_token_ids: list[int] | tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply the canonical reconstructable history policy to raw logits."""

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
    counts = np.bincount(considered, minlength=len(values)).astype(
        np.int64, copy=False
    )
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


def _stages_from_adjusted(
    adjusted: np.ndarray, config: SamplingConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | None]]:
    """Run decoder stages on an already prepared history-policy surface."""
    result = apply_candidate_filter(adjusted, config)
    return adjusted, result.scaled_logits, result.stages


class ObservationStatistics:
    """Owned numeric snapshot, with selected-token ranks computed lazily."""

    def __init__(
        self,
        logits,
        config,
        history_token_ids,
        boundaries=None,
        token_preference_features=None,
        render_tokens=None,
        token_preference_coordinate_identity=None,
        activation_logit_adjustments=None,
        model_phase_diagnostics=None,
        ephemeral_logit_biases=None,
        capture_trace=False,
    ):
        self.logits = _validated_logits(logits).copy()
        self.boundaries = boundaries
        self.token_preference_features = token_preference_features
        self.token_preference_coordinate_identity = token_preference_coordinate_identity
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
            trace_stages.append(
                ControllerTraceStage(name, phase, values, delta, details)
            )

        if model_phase_diagnostics:
            record_stage(
                str(model_phase_diagnostics.get("name", "model-phase guidance")),
                "model",
                self.logits,
                **{
                    key: value for key, value in model_phase_diagnostics.items()
                    if key != "name"
                },
            )
        record_stage("backend logits", "policy", self.logits)
        trace_previous = self.logits
        penalties_active = config.history_penalties_active
        if penalties_active:
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
        if (
            config.activation_vector_layer == "output"
            and config.activation_vector
            and config.activation_vector_strength != 0.0
        ):
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
            "output-head steering", "policy", self.adjusted, trace_previous,
            active=bool(
                config.activation_vector_layer == "output"
                and config.activation_vector
                and config.activation_vector_strength != 0.0
            ),
        )
        trace_previous = self.adjusted
        self.activation_diagnostics = {
            "vector_norm": float(
                np.linalg.norm(np.asarray(getattr(config, "activation_vector", ()), dtype=np.float64))
            ) if getattr(config, "activation_vector", ()) else 0.0,
            "strength": float(getattr(config, "activation_vector_strength", 0.0)),
            "logit_rms": float(
                np.sqrt(np.mean(self.activation_logit_adjustments ** 2))
            ),
            "logit_min": float(np.min(self.activation_logit_adjustments)),
            "logit_max": float(np.max(self.activation_logit_adjustments)),
            "digest": getattr(config, "activation_vector_digest", ""),
        }
        active_biases = config.active_biases(history_token_ids, boundaries)
        self.active_biases = active_biases
        reference_snapshot = getattr(config, "active_reference_prior_snapshot", None)
        self.reference_prior_snapshot = (
            reference_snapshot(history_token_ids, boundaries)
            if callable(reference_snapshot)
            else ReferencePriorSnapshot("active", "contrastive", (), 0.0, 0.0, 0.0, (), {})
        )
        self.reference_prior_biases = self.reference_prior_snapshot.biases
        adjustments = dict(active_biases)
        for token, bias in self.reference_prior_biases.items():
            adjustments[token] = adjustments.get(token, 0.0) + bias
        if adjustments:
            self.adjusted = self.adjusted.copy()
            for token, bias in adjustments.items():
                if token >= len(self.logits):
                    raise ValueError("bias token id is outside the decoder vocabulary")
                self.adjusted[token] += bias
            if not np.all(np.isfinite(self.adjusted)):
                raise ValueError("biases produced non-finite policy logits")
        record_stage(
            "manual/reference", "policy", self.adjusted, trace_previous,
            manual_tokens=len(self.active_biases),
            reference_tokens=len(self.reference_prior_biases),
        )
        trace_previous = self.adjusted
        # This is the canonical surface immediately before preference actuation.
        # Keep it separate from both the deployed policy and the raw model.
        self.preference_base_logits = np.asarray(self.adjusted, dtype=np.float64).copy()
        self.preference_base_probabilities = _softmax(self.preference_base_logits)
        self.learning_logits = self.preference_base_logits.copy()
        self.learning_probabilities = self.preference_base_probabilities.copy()
        token_preference_vector = tuple(getattr(config, "token_preference_vector", ()))
        fast_vector = tuple(getattr(config, "token_preference_fast_vector", ()))
        token_preference_strength = float(getattr(config, "token_preference_strength", 1.0))
        token_preference_fast_strength = float(
            getattr(config, "token_preference_fast_strength", 0.0)
        )
        token_preference_influence_mode = getattr(
            config, "token_preference_influence_mode", "manual"
        )
        token_preference_influence_kl = float(
            getattr(config, "token_preference_influence_kl", 0.05)
        )
        token_preference_min_gain = float(getattr(config, "token_preference_min_gain", 0.0))
        token_preference_max_gain = float(getattr(config, "token_preference_max_gain", 8.0))
        token_preference_feature_scheme = getattr(
            config, "token_preference_feature_scheme", "random-projection-unit-v1"
        )
        self.token_preference_effective_strength = 0.0
        self.token_preference_effective_fast_strength = 0.0
        self.token_preference_raw_scores = np.zeros(len(self.logits), dtype=np.float64)
        self.token_preference_diagnostics = {
            "z_norm": float(np.linalg.norm(np.asarray(token_preference_vector, dtype=np.float64)))
            if token_preference_vector else 0.0,
            "raw_fz_rms": 0.0,
            "effective_logit_rms": 0.0,
            "effective_strength": 0.0,
            "effective_fast_strength": 0.0,
            "pre_post_token_preference_kl": 0.0,
            "top_token_preference_logit_min": 0.0,
            "top_token_preference_logit_max": 0.0,
            "slow_raw_logit_rms": 0.0,
            "fast_raw_logit_rms": 0.0,
            "combined_raw_logit_rms": 0.0,
            "effective_gain": 0.0,
            "user_multiplier": token_preference_strength,
            "gain_capped": False,
            "deployment_kl": 0.0,
            "relative_fast_weight": 0.0,
        }
        if token_preference_vector or fast_vector:
            if token_preference_features is None:
                raise ValueError(
                    "preference token features are required when token preference state is active"
                )
            features = np.asarray(token_preference_features, dtype=np.float32)
            if features.shape != (len(self.logits), len(token_preference_vector or fast_vector)):
                raise ValueError(
                    "preference token features do not match the policy vocabulary and state"
                )
            if not np.all(np.isfinite(features)):
                raise ValueError("preference token features must be finite")
            slow_scores = (
                features @ np.asarray(token_preference_vector, dtype=np.float32)
                if token_preference_vector else np.zeros(len(self.logits), dtype=np.float32)
            )
            fast_scores = (
                features @ np.asarray(fast_vector, dtype=np.float32)
                if fast_vector else np.zeros(len(self.logits), dtype=np.float32)
            )
            relative_fast_weight = token_preference_fast_strength if fast_vector else 0.0
            combined_scores = np.asarray(slow_scores, dtype=np.float64) + (
                relative_fast_weight * np.asarray(fast_scores, dtype=np.float64)
            )
            if token_preference_influence_mode == "kl":
                auto_gain = calibrated_token_preference_gain(
                    self.preference_base_probabilities,
                    combined_scores,
                    token_preference_influence_kl,
                    min_gain=token_preference_min_gain,
                    max_gain=token_preference_max_gain,
                ) if np.any(combined_scores) else 0.0
                slow_strength = token_preference_strength * auto_gain
                fast_strength = token_preference_strength * auto_gain * relative_fast_weight
            else:
                slow_strength = token_preference_strength
                fast_strength = token_preference_fast_strength
            if (
                token_preference_feature_scheme == "random-projection-unit-v1"
                and token_preference_influence_mode == "manual"
            ):
                # Preserve the original float32 actuator arithmetic for all
                # v1 records.  The new diagnostics are observational only.
                token_preference_adjustments = (
                    token_preference_strength * slow_scores
                    if token_preference_vector else np.zeros(len(self.logits), dtype=np.float32)
                )
                if fast_vector:
                    token_preference_adjustments += token_preference_fast_strength * fast_scores
            else:
                token_preference_adjustments = (
                    slow_strength * np.asarray(slow_scores, dtype=np.float64)
                    + fast_strength * np.asarray(fast_scores, dtype=np.float64)
                )
            if fast_vector:
                self.token_preference_effective_fast_strength = fast_strength
            if not np.all(np.isfinite(token_preference_adjustments)):
                raise ValueError("token preference produced non-finite policy logits")
            self.token_preference_features = features
            self.token_preference_logit_adjustments = token_preference_adjustments
            self.token_preference_raw_scores = np.asarray(slow_scores, dtype=np.float64)
            if token_preference_vector or fast_vector:
                # The internal learner sees the complete memory state, while
                # deployment may apply a different user gain.
                self.learning_logits += combined_scores
                self.learning_probabilities = _softmax(self.learning_logits)
            self.token_preference_effective_strength = slow_strength
            slow_raw_rms = float(np.sqrt(np.mean(np.asarray(slow_scores, dtype=np.float64) ** 2)))
            fast_raw_rms = float(np.sqrt(np.mean(np.asarray(fast_scores, dtype=np.float64) ** 2)))
            raw_rms = float(np.sqrt(np.mean(combined_scores ** 2)))
            effective_rms = float(np.sqrt(np.mean(token_preference_adjustments ** 2)))
            self.token_preference_diagnostics = {
                "z_norm": float(np.linalg.norm(np.asarray(token_preference_vector, dtype=np.float64)))
                if token_preference_vector else 0.0,
                "raw_fz_rms": raw_rms,
                "effective_logit_rms": effective_rms,
                "effective_strength": slow_strength,
                "effective_fast_strength": fast_strength,
                "slow_raw_logit_rms": slow_raw_rms,
                "fast_raw_logit_rms": fast_raw_rms,
                "combined_raw_logit_rms": raw_rms,
                "effective_gain": float(auto_gain) if token_preference_influence_mode == "kl" else 1.0,
                "user_multiplier": token_preference_strength,
                "gain_capped": bool(
                    token_preference_influence_mode == "kl" and auto_gain in {
                        token_preference_min_gain, token_preference_max_gain
                    }
                ),
                "relative_fast_weight": relative_fast_weight,
                "pre_post_token_preference_kl": 0.0,
                "top_token_preference_logit_min": float(np.min(token_preference_adjustments)),
                "top_token_preference_logit_max": float(np.max(token_preference_adjustments)),
            }
            self.adjusted = self.adjusted.copy()
            self.adjusted += token_preference_adjustments
        else:
            self.token_preference_logit_adjustments = np.zeros_like(self.adjusted)
        record_stage(
            "token preference", "policy", self.adjusted, trace_previous,
            active=bool(token_preference_vector or fast_vector),
        )
        trace_previous = self.adjusted
        self.baseline_probabilities = _softmax(self.adjusted)
        if token_preference_vector or fast_vector:
            self.token_preference_diagnostics["pre_post_token_preference_kl"] = policy_kl(
                self.baseline_probabilities, self.preference_base_probabilities
            )
            self.token_preference_diagnostics["deployment_kl"] = self.token_preference_diagnostics[
                "pre_post_token_preference_kl"
            ]
        from .group_control import control_adjustments
        group_controls = tuple(getattr(config, "group_controls", ()))
        bias_groups = tuple(getattr(config, "bias_groups", ()))
        group_control_scheme = getattr(config, "group_control_scheme", "appearance-feedback-v1")
        self.group_control_biases, self.group_control_diagnostics = control_adjustments(
            group_controls, bias_groups, () if history_token_ids is None else history_token_ids,
            self.adjusted, boundaries, render_tokens, scheme=group_control_scheme,
        )
        if self.group_control_biases:
            self.adjusted = self.adjusted.copy()
            for token, amount in self.group_control_biases.items():
                self.adjusted[token] += amount
        record_stage(
            "group control", "policy", self.adjusted, trace_previous,
            active=bool(group_controls),
            controlled_tokens=len(self.group_control_biases),
        )
        trace_previous = self.adjusted
        self.ephemeral_logit_biases = {
            int(token): float(amount)
            for token, amount in (ephemeral_logit_biases or {}).items()
        }
        if self.ephemeral_logit_biases:
            if any(
                token < 0 or token >= len(self.logits)
                or not np.isfinite(amount)
                for token, amount in self.ephemeral_logit_biases.items()
            ):
                raise ValueError("ephemeral logit biases are invalid")
            self.adjusted = self.adjusted.copy()
            for token, amount in self.ephemeral_logit_biases.items():
                self.adjusted[token] += amount
            if not np.all(np.isfinite(self.adjusted)):
                raise ValueError("ephemeral logit biases produced non-finite policy logits")
        if self.ephemeral_logit_biases:
            record_stage(
                "temporary phrase bias", "policy", self.adjusted, trace_previous,
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
            _softmax(self.adjusted) if penalties_active
            else exponentials / self.denominator
        )
        _, scaled, stages = _stages_from_adjusted(self.adjusted, config)
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
            ids, _softmax(scaled[ids]), np.asarray(scaled[ids], dtype=np.float64)
        )
        record_stage(
            "sampler / token draw", "policy", self.adjusted, trace_previous,
            filtered_tokens=len(ids),
            temperature=float(config.temperature),
            candidate_filter="standard",
            draw_kernel=config.draw_kernel,
            typical_p=float(config.typical_p),
            tail_free_z=float(config.tail_free_z),
        )
        self.controller_trace = (
            ControllerTrace(tuple(trace_stages), tuple(int(value) for value in ids))
            if capture_trace else None
        )
        for array in (
            self.logits, self.adjusted, self.policy_probabilities, self.baseline_probabilities,
            self.preference_base_logits, self.preference_base_probabilities,
            self.learning_logits, self.learning_probabilities, self.token_preference_raw_scores,
            self.activation_logit_adjustments,
            self.distribution.ids, self.distribution.probabilities,
        ):
            array.setflags(write=False)
        if self.distribution.scores is not None:
            self.distribution.scores.setflags(write=False)
        self._raw_ranks: dict[int, int] = {}
        self._policy_ranks: dict[int, int] = (
            {} if penalties_active else self._raw_ranks
        )
        self._ordered: list[int] = []
        self._policy_ordered: list[int] = []

    def raw_probabilities(self, token_ids):
        if self.adjusted is self.logits:
            return self.policy_probabilities[list(token_ids)]
        return np.exp(self.logits[list(token_ids)] - self.maximum) / self.denominator

    # Canonical names for new controller and diagnostic code. The historical
    # aliases remain because saved evidence and third-party callers use them.
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
        """Full-vocabulary policy ordering, with the same tie-break as raw rank."""
        if self.adjusted is self.logits:
            return self.top_raw_ids(count)
        if count > len(self._policy_ordered):
            self._policy_ordered = [int(value) for value in _top_ids(self.adjusted, count)]
            self._policy_ranks.update(
                (token_id, rank) for rank, token_id in enumerate(self._policy_ordered, 1)
            )
        return self._policy_ordered[:count]

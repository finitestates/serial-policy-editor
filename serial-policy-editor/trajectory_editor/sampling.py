"""Replayable sampling and ranked raw-logit menus."""

from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .domain import RNG_SCHEME, SamplingConfig, EditorError, MIN_SEED, MAX_SEED
from .episode_hash import validate_coordinate, validate_fingerprint


@dataclass(frozen=True)
class SparseDistribution:
    ids: np.ndarray
    probabilities: np.ndarray

    def probability(self, token_id: int) -> float:
        matches = np.flatnonzero(self.ids == int(token_id))
        return float(self.probabilities[matches[0]]) if len(matches) else 0.0


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.array(values, dtype=np.float64, copy=True)
    if shifted.ndim != 1 or not len(shifted) or not np.all(np.isfinite(shifted)):
        raise ValueError("softmax values must be a finite nonempty vector")
    shifted -= np.max(shifted)
    exponentials = np.exp(shifted)
    denominator = float(np.sum(exponentials))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("softmax normalization failed")
    return exponentials / denominator


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


def calibrated_latent_gain(
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
        raise ValueError("latent calibration inputs must have equal 1-D shapes")
    if (
        not np.all(np.isfinite(p)) or np.any(p < 0.0)
        or not np.all(np.isfinite(a)) or float(np.sum(p)) <= 0.0
    ):
        raise ValueError("latent calibration inputs must be finite probabilities and scores")
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


def _top_ids(values: np.ndarray, count: int) -> np.ndarray:
    """Return descending-logit ids with token-id ascending as the tie-break."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("decoder logits must be a finite nonempty vector")
    count = min(max(1, int(count)), len(values))
    if count == len(values):
        selected = np.arange(len(values), dtype=np.int64)
    else:
        threshold = float(np.partition(values, len(values) - count)[len(values) - count])
        above = np.flatnonzero(values > threshold)
        tied = np.flatnonzero(values == threshold)
        needed = count - len(above)
        selected = np.concatenate((above, tied[:needed])).astype(np.int64, copy=False)
    order = np.lexsort((selected, -values[selected]))
    return selected[order]


def _validated_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(
            "decoder logits must be a finite nonempty one-dimensional array"
        )
    return values


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


def _rank(values: np.ndarray, token_id: int) -> int:
    token_id = int(token_id)
    if not 0 <= token_id < len(values):
        raise ValueError("token id is outside the decoder vocabulary")
    target = float(values[token_id])
    higher = int(np.count_nonzero(values > target))
    tied_before = int(np.count_nonzero(values[:token_id] == target))
    return 1 + higher + tied_before


def _stages_from_adjusted(
    adjusted: np.ndarray, config: SamplingConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | None]]:
    """Run decoder stages on an already prepared history-policy surface."""
    if config.temperature == 0.0:
        greedy = _top_ids(adjusted, 1)
        return adjusted, adjusted, {
            "after_temperature": greedy,
            "after_top_k": greedy,
            "after_top_p": greedy,
            "after_min_p": greedy,
        }

    scaled = adjusted / float(config.temperature)
    if not np.all(np.isfinite(scaled)):
        raise ValueError("temperature produced non-finite scaled logits")

    after_top_k = _top_ids(scaled, min(config.top_k, len(scaled)))
    after_top_p = after_top_k
    if config.top_p < 1.0:
        probabilities = _softmax(scaled[after_top_k])
        keep_count = int(
            np.searchsorted(np.cumsum(probabilities), config.top_p, side="left")
        ) + 1
        after_top_p = after_top_k[: max(1, keep_count)]

    after_min_p = after_top_p
    if config.min_p > 0.0:
        probabilities = _softmax(scaled[after_top_p])
        mask = probabilities >= config.min_p * float(np.max(probabilities))
        if np.any(mask):
            after_min_p = after_top_p[mask]

    return adjusted, scaled, {
        # None means the stage retains the full vocabulary. Avoid allocating a
        # vocabulary-sized id array when no filtering happens at this stage.
        "after_temperature": None,
        "after_top_k": after_top_k,
        "after_top_p": after_top_p,
        "after_min_p": after_min_p,
    }


def position_uniform(seed: int, stream_fingerprint: str, aligned_step: int) -> float:
    if type(seed) is not int or not MIN_SEED <= seed <= MAX_SEED:
        raise EditorError("seed must be a signed-64-bit integer")
    validate_fingerprint(stream_fingerprint)
    validate_coordinate(aligned_step, "sampling position")
    payload = f"{RNG_SCHEME}:{seed}:{stream_fingerprint}:{aligned_step}".encode()
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return (value + 0.5) / float(1 << 64)


def draw_token(
    distribution: SparseDistribution,
    *,
    seed: int,
    stream_fingerprint: str,
    aligned_step: int,
) -> int:
    draw = position_uniform(seed, stream_fingerprint, aligned_step)
    index = int(
        np.searchsorted(np.cumsum(distribution.probabilities), draw, side="right")
    )
    return int(distribution.ids[min(index, len(distribution.ids) - 1)])


def raw_rank(logits: np.ndarray, token_id: int) -> int:
    """Return the one-based full-vocabulary rank with the menu tie-break."""
    values = _validated_logits(logits)
    return _rank(values, token_id)


def top_raw_ids(logits: np.ndarray, count: int) -> list[int]:
    return [int(value) for value in _top_ids(np.asarray(logits), count)]


class ObservationStatistics:
    """Owned numeric snapshot, with selected-token ranks computed lazily."""

    def __init__(
        self,
        logits,
        config,
        history_token_ids,
        boundaries=None,
        latent_features=None,
        render_tokens=None,
        latent_coordinate_identity=None,
    ):
        self.logits = _validated_logits(logits).copy()
        self.boundaries = boundaries
        self.latent_features = latent_features
        self.latent_coordinate_identity = latent_coordinate_identity
        self.render_tokens = render_tokens
        penalties_active = config.history_penalties_active
        if penalties_active:
            self.adjusted, _, _ = _history_penalty_surface(
                self.logits, config, history_token_ids
            )
        else:
            if history_token_ids is not None:
                _validated_history(history_token_ids, len(self.logits))
            self.adjusted = self.logits
        active_biases = config.active_biases(history_token_ids, boundaries)
        self.active_biases = active_biases
        self.reference_prior_snapshot = config.active_reference_prior_snapshot(
            history_token_ids, boundaries
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
        # This is the canonical surface immediately before latent actuation.
        # Keep it separate from both the deployed policy and the raw model.
        self.pre_latent_logits = np.asarray(self.adjusted, dtype=np.float64).copy()
        self.pre_latent_probabilities = _softmax(self.pre_latent_logits)
        self.learning_logits = self.pre_latent_logits.copy()
        self.learning_probabilities = self.pre_latent_probabilities.copy()
        latent_z = tuple(config.latent_preference_z)
        fast_z = tuple(config.latent_preference_fast_z)
        self.latent_effective_strength = 0.0
        self.latent_effective_fast_strength = 0.0
        self.latent_raw_scores = np.zeros(len(self.logits), dtype=np.float64)
        self.latent_diagnostics = {
            "z_norm": float(np.linalg.norm(np.asarray(latent_z, dtype=np.float64)))
            if latent_z else 0.0,
            "raw_fz_rms": 0.0,
            "effective_logit_rms": 0.0,
            "effective_strength": 0.0,
            "effective_fast_strength": 0.0,
            "pre_post_latent_kl": 0.0,
            "top_latent_logit_min": 0.0,
            "top_latent_logit_max": 0.0,
            "slow_raw_logit_rms": 0.0,
            "fast_raw_logit_rms": 0.0,
            "combined_raw_logit_rms": 0.0,
            "effective_gain": 0.0,
            "user_multiplier": float(config.latent_strength),
            "gain_capped": False,
            "deployment_kl": 0.0,
            "relative_fast_weight": 0.0,
        }
        if latent_z or fast_z:
            if latent_features is None:
                raise ValueError(
                    "latent token features are required when latent preference state is active"
                )
            features = np.asarray(latent_features, dtype=np.float32)
            if features.shape != (len(self.logits), len(latent_z or fast_z)):
                raise ValueError(
                    "latent token features do not match the policy vocabulary and state"
                )
            if not np.all(np.isfinite(features)):
                raise ValueError("latent token features must be finite")
            slow_scores = (
                features @ np.asarray(latent_z, dtype=np.float32)
                if latent_z else np.zeros(len(self.logits), dtype=np.float32)
            )
            fast_scores = (
                features @ np.asarray(fast_z, dtype=np.float32)
                if fast_z else np.zeros(len(self.logits), dtype=np.float32)
            )
            relative_fast_weight = float(config.latent_fast_strength) if fast_z else 0.0
            combined_scores = np.asarray(slow_scores, dtype=np.float64) + (
                relative_fast_weight * np.asarray(fast_scores, dtype=np.float64)
            )
            if config.latent_influence_mode == "kl":
                auto_gain = calibrated_latent_gain(
                    self.pre_latent_probabilities,
                    combined_scores,
                    config.latent_influence_kl,
                    min_gain=config.latent_min_gain,
                    max_gain=config.latent_max_gain,
                ) if np.any(combined_scores) else 0.0
                slow_strength = float(config.latent_strength) * auto_gain
                fast_strength = float(config.latent_strength) * auto_gain * relative_fast_weight
            else:
                slow_strength = float(config.latent_strength)
                fast_strength = float(config.latent_fast_strength)
            if (
                config.latent_feature_scheme == "random-projection-unit-v1"
                and config.latent_influence_mode == "manual"
            ):
                # Preserve the original float32 actuator arithmetic for all
                # v1 records.  The new diagnostics are observational only.
                latent_adjustments = (
                    float(config.latent_strength) * slow_scores
                    if latent_z else np.zeros(len(self.logits), dtype=np.float32)
                )
                if fast_z:
                    latent_adjustments += float(config.latent_fast_strength) * fast_scores
            else:
                latent_adjustments = (
                    slow_strength * np.asarray(slow_scores, dtype=np.float64)
                    + fast_strength * np.asarray(fast_scores, dtype=np.float64)
                )
            if fast_z:
                self.latent_effective_fast_strength = fast_strength
            if not np.all(np.isfinite(latent_adjustments)):
                raise ValueError("latent preference produced non-finite policy logits")
            self.latent_features = features
            self.latent_logit_adjustments = latent_adjustments
            self.latent_raw_scores = np.asarray(slow_scores, dtype=np.float64)
            if latent_z or fast_z:
                # The internal learner sees the complete memory state, while
                # deployment may apply a different user gain.
                self.learning_logits += combined_scores
                self.learning_probabilities = _softmax(self.learning_logits)
            self.latent_effective_strength = slow_strength
            slow_raw_rms = float(np.sqrt(np.mean(np.asarray(slow_scores, dtype=np.float64) ** 2)))
            fast_raw_rms = float(np.sqrt(np.mean(np.asarray(fast_scores, dtype=np.float64) ** 2)))
            raw_rms = float(np.sqrt(np.mean(combined_scores ** 2)))
            effective_rms = float(np.sqrt(np.mean(latent_adjustments ** 2)))
            self.latent_diagnostics = {
                "z_norm": float(np.linalg.norm(np.asarray(latent_z, dtype=np.float64)))
                if latent_z else 0.0,
                "raw_fz_rms": raw_rms,
                "effective_logit_rms": effective_rms,
                "effective_strength": slow_strength,
                "effective_fast_strength": fast_strength,
                "slow_raw_logit_rms": slow_raw_rms,
                "fast_raw_logit_rms": fast_raw_rms,
                "combined_raw_logit_rms": raw_rms,
                "effective_gain": float(auto_gain) if config.latent_influence_mode == "kl" else 1.0,
                "user_multiplier": float(config.latent_strength),
                "gain_capped": bool(
                    config.latent_influence_mode == "kl" and auto_gain in {
                        float(config.latent_min_gain), float(config.latent_max_gain)
                    }
                ),
                "relative_fast_weight": relative_fast_weight,
                "pre_post_latent_kl": 0.0,
                "top_latent_logit_min": float(np.min(latent_adjustments)),
                "top_latent_logit_max": float(np.max(latent_adjustments)),
            }
            self.adjusted = self.adjusted.copy()
            self.adjusted += latent_adjustments
        else:
            self.latent_logit_adjustments = np.zeros_like(self.adjusted)
        self.baseline_probabilities = _softmax(self.adjusted)
        if latent_z or fast_z:
            self.latent_diagnostics["pre_post_latent_kl"] = policy_kl(
                self.baseline_probabilities, self.pre_latent_probabilities
            )
            self.latent_diagnostics["deployment_kl"] = self.latent_diagnostics[
                "pre_post_latent_kl"
            ]
        from .group_control import control_adjustments
        self.group_control_biases, self.group_control_diagnostics = control_adjustments(
            config.group_controls, config.bias_groups, () if history_token_ids is None else history_token_ids,
            self.adjusted, boundaries, render_tokens, scheme=config.group_control_scheme,
        )
        if self.group_control_biases:
            self.adjusted = self.adjusted.copy()
            for token, amount in self.group_control_biases.items():
                self.adjusted[token] += amount
        penalties_active = config.policy_active
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
        self.distribution = SparseDistribution(ids, _softmax(scaled[ids]))
        for array in (
            self.logits, self.adjusted, self.policy_probabilities, self.baseline_probabilities,
            self.pre_latent_logits, self.pre_latent_probabilities,
            self.learning_logits, self.learning_probabilities, self.latent_raw_scores,
            self.distribution.ids, self.distribution.probabilities,
        ):
            array.setflags(write=False)
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

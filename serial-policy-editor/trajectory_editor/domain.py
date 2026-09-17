"""Pure domain types for the reduced policy editor."""

from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .token_preference_features import (
    DEFAULT_PROJECTION_SEED,
    DEFAULT_WHITENING_RIDGE,
    TOKEN_PREFERENCE_FEATURE_SCHEMES,
    TokenPreferenceCoordinateIdentity,
    coordinate_identity,
)


class EditorError(ValueError):
    """Raised when an editor command or configuration is invalid."""


RNG_SCHEME = "blake2b64-token-prefix-quantile-v2"
SAMPLING_POLICY_SCHEME = "spe-history-aware-decoder-policy-v1"
MIN_SEED = -(1 << 63)
MAX_SEED = (1 << 63) - 1


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    repeat_penalty: float = 1.0
    repeat_last_n: int = 64
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = 12345
    bias_step: float = 0.5
    bias_rules: tuple = ()
    bias_groups: tuple = ()
    group_controls: tuple = ()
    token_preference_vector: tuple = ()
    token_preference_strength: float = 1.0
    token_preference_fast_vector: tuple = ()
    token_preference_fast_strength: float = 0.0
    token_preference_projection_seed: int = DEFAULT_PROJECTION_SEED
    token_preference_feature_scheme: str = "random-projection-unit-v1"
    token_preference_whitening_ridge: float = DEFAULT_WHITENING_RIDGE
    token_preference_learning_scheme: str = "sgd-v1"
    token_preference_influence_mode: str = "manual"
    token_preference_influence_kl: float = 0.05
    token_preference_min_gain: float = 0.0
    token_preference_max_gain: float = 8.0
    token_preference_coordinate_identity: TokenPreferenceCoordinateIdentity | Mapping[str, Any] | None = None
    activation_vector: tuple = ()
    activation_vector_strength: float = 0.0
    activation_vector_layer: str = "output"
    activation_vector_position: str = "current"
    activation_vector_layer_start: int | None = None
    activation_vector_layer_end: int | None = None
    # Stored as canonical JSON so the frozen sampler remains hashable while
    # retaining the model identity needed for replay diagnostics.
    activation_vector_model: str = ""
    activation_vector_digest: str = ""
    group_control_scheme: str = "appearance-feedback-v1"
    reference_prior_routes: tuple = ()
    reference_prior_scope: str = "active"
    reference_prior_mode: str = "contrastive"
    reference_prior_strength: float = 0.25
    reference_prior_attraction: float = 0.0
    reference_prior_exit_strength: float = 0.25
    _reference_prior_trie: Any = field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self) -> None:
        from .bias_rules import BiasGroup, BiasRule
        if type(self.bias_step) not in (int, float) or not math.isfinite(self.bias_step) or self.bias_step <= 0:
            raise EditorError("bias_step must be a finite positive number")
        try:
            rules = tuple(BiasRule.from_record(rule) for rule in self.bias_rules)
        except TypeError as exc:
            raise EditorError("bias_rules must be a list of rules") from exc
        if len({rule.key for rule in rules}) != len(rules):
            raise EditorError("duplicate bias rule")
        object.__setattr__(self, "bias_rules", tuple(sorted(
            (rule for rule in rules if rule.bias != 0),
            key=lambda rule: rule.sort_key,
        )))
        try:
            groups = tuple(BiasGroup.from_record(group) for group in self.bias_groups)
        except TypeError as exc:
            raise EditorError("bias_groups must be a list of groups") from exc
        if len({group.name for group in groups}) != len(groups):
            raise EditorError("duplicate bias group")
        object.__setattr__(self, "bias_groups", tuple(sorted(groups, key=lambda group: group.name)))
        from .group_control import GroupControl
        try:
            controls = tuple(GroupControl.from_record(c) for c in self.group_controls)
        except TypeError as exc:
            raise EditorError("group_controls must be a list of controls") from exc
        if len({c.key for c in controls}) != len(controls):
            raise EditorError("duplicate group control scope")
        if any(c.group not in {g.name for g in groups} for c in controls):
            raise EditorError("group control refers to a missing group")
        object.__setattr__(self, "group_controls", controls)
        for name in ("token_preference_vector", "token_preference_fast_vector"):
            raw = getattr(self, name)
            if isinstance(raw, (str, bytes, bytearray)):
                raise EditorError(f"{name} must be a numeric vector")
            try:
                vector = tuple(float(value) for value in raw)
            except (TypeError, ValueError) as exc:
                raise EditorError(f"{name} must be a numeric vector") from exc
            if any(not math.isfinite(value) for value in vector):
                raise EditorError(f"{name} must contain finite numbers")
            object.__setattr__(self, name, vector)
        if (self.token_preference_vector and self.token_preference_fast_vector
                and len(self.token_preference_vector) != len(self.token_preference_fast_vector)):
            raise EditorError("slow and fast token preference vectors must have the same dimension")
        identity = self.token_preference_coordinate_identity
        if identity is not None and not isinstance(identity, TokenPreferenceCoordinateIdentity):
            try:
                identity = TokenPreferenceCoordinateIdentity.from_mapping(identity)
            except (KeyError, TypeError, ValueError) as exc:
                raise EditorError("token preference coordinate identity is malformed") from exc
            object.__setattr__(self, "token_preference_coordinate_identity", identity)
        if (
            self.token_preference_feature_scheme == "random-projection-unit-v1"
            and (self.token_preference_vector or self.token_preference_fast_vector)
        ):
            dimension = len(self.token_preference_vector or self.token_preference_fast_vector)
            object.__setattr__(
                self,
                "token_preference_coordinate_identity",
                coordinate_identity(
                    dimension=dimension,
                    projection_seed=self.token_preference_projection_seed,
                    feature_scheme=self.token_preference_feature_scheme,
                    whitening_ridge=self.token_preference_whitening_ridge,
                ),
            )
        elif identity is None and (self.token_preference_vector or self.token_preference_fast_vector):
            dimension = len(self.token_preference_vector or self.token_preference_fast_vector)
            object.__setattr__(
                self,
                "token_preference_coordinate_identity",
                coordinate_identity(
                    dimension=dimension,
                    projection_seed=self.token_preference_projection_seed,
                    feature_scheme=self.token_preference_feature_scheme,
                    whitening_ridge=self.token_preference_whitening_ridge,
                ),
            )
        for name in ("token_preference_strength", "token_preference_fast_strength"):
            value = getattr(self, name)
            if (type(value) not in (int, float)
                    or not math.isfinite(float(value)) or value < 0.0):
                raise EditorError(f"{name} must be a finite nonnegative number")
            object.__setattr__(self, name, float(value))
        if (type(self.token_preference_projection_seed) is not int
                or not MIN_SEED <= self.token_preference_projection_seed <= MAX_SEED):
            raise EditorError("preference projection seed must be a signed 64-bit integer")
        if self.token_preference_feature_scheme not in TOKEN_PREFERENCE_FEATURE_SCHEMES:
            raise EditorError(
                "token_preference_feature_scheme must be random-projection-unit-v1 or "
                "whitened-projection-v2"
            )
        if (
            type(self.token_preference_whitening_ridge) not in (int, float)
            or not math.isfinite(float(self.token_preference_whitening_ridge))
            or self.token_preference_whitening_ridge < 0.0
        ):
            raise EditorError("token_preference_whitening_ridge must be finite and nonnegative")
        object.__setattr__(self, "token_preference_whitening_ridge", float(self.token_preference_whitening_ridge))
        raw_activation = self.activation_vector
        if isinstance(raw_activation, (str, bytes, bytearray)):
            raise EditorError("activation_vector must be a numeric vector")
        try:
            activation = tuple(float(value) for value in raw_activation)
        except (TypeError, ValueError) as exc:
            raise EditorError("activation_vector must be a numeric vector") from exc
        if any(not math.isfinite(value) for value in activation):
            raise EditorError("activation_vector must contain finite numbers")
        object.__setattr__(self, "activation_vector", activation)
        if self.activation_vector_layer == "output":
            if self.activation_vector_position != "current":
                raise EditorError("output-head steering position must be current")
            if (
                self.activation_vector_layer_start is not None
                or self.activation_vector_layer_end is not None
            ):
                raise EditorError("output-head steering vectors cannot specify a layer range")
        elif self.activation_vector_layer == "control-vector":
            if self.activation_vector_position != "layers":
                raise EditorError("hidden-state vector position must be layers")
            if (
                type(self.activation_vector_layer_start) is not int
                or self.activation_vector_layer_start < 1
                or type(self.activation_vector_layer_end) is not int
                or self.activation_vector_layer_end < self.activation_vector_layer_start
            ):
                raise EditorError("hidden-state vector layer range must be a positive interval")
        else:
            raise EditorError("steering vector target must be output or control-vector")
        value = self.activation_vector_strength
        if (
            type(value) not in (int, float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise EditorError("activation_vector_strength must be finite and nonnegative")
        object.__setattr__(self, "activation_vector_strength", float(value))
        model = self.activation_vector_model
        if isinstance(model, Mapping):
            model = json.dumps(
                dict(model), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        if not isinstance(model, str):
            raise EditorError("activation_vector_model must be a JSON object")
        if model:
            try:
                parsed_model = json.loads(model)
            except (TypeError, ValueError) as exc:
                raise EditorError("activation_vector_model must be valid JSON") from exc
            if not isinstance(parsed_model, dict):
                raise EditorError("activation_vector_model must be a JSON object")
            model = json.dumps(
                parsed_model, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        object.__setattr__(self, "activation_vector_model", model)
        digest = self.activation_vector_digest
        if not isinstance(digest, str) or (
            digest and re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise EditorError("activation_vector_digest must be a lowercase SHA-256 digest")
        object.__setattr__(self, "activation_vector_digest", digest)
        if (
            self.activation_vector_layer == "control-vector"
            and self.activation_vector
            and self.activation_vector_strength != 0.0
            and not digest
        ):
            raise EditorError(
                "active control-vector activation requires a verified digest"
            )
        if activation and model:
            model_metadata = json.loads(model)
            width = model_metadata.get("hidden_state_width", model_metadata.get("activation_width"))
            if (
                width is not None
                and self.activation_vector_layer == "output"
                and width != len(activation)
            ):
                raise EditorError("steering vector dimension does not match model width")
            if (
                width is not None
                and self.activation_vector_layer == "control-vector"
                and (type(width) is not int or width < 1 or len(activation) % width)
            ):
                raise EditorError("control-vector dimension is not layer-aligned")
            if (
                self.activation_vector_layer == "control-vector"
                and width is None
            ):
                raise EditorError("control-vector activation requires model width")
            layer_count = model_metadata.get(
                "hidden_state_layer_count", model_metadata.get("activation_layer_count")
            )
            if (
                layer_count is not None
                and self.activation_vector_layer == "control-vector"
                and (type(layer_count) is not int or layer_count < 1
                     or len(activation) // int(width) != layer_count
                     or self.activation_vector_layer_end > layer_count)
            ):
                raise EditorError("control-vector dimension does not match its layer metadata")
        if self.token_preference_learning_scheme not in {"sgd-v1", "fisher-kl-v2"}:
            raise EditorError("unsupported token_preference_learning_scheme")
        if self.token_preference_influence_mode not in {"manual", "kl"}:
            raise EditorError("token_preference_influence_mode must be manual or kl")
        for name in ("token_preference_influence_kl", "token_preference_min_gain", "token_preference_max_gain"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise EditorError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))
        if self.token_preference_max_gain < self.token_preference_min_gain:
            raise EditorError("token_preference_max_gain must be at least token_preference_min_gain")
        if self.group_control_scheme not in {
            "appearance-feedback-v1", "appearance-rate-v2"
        }:
            raise EditorError("unsupported group_control_scheme")
        if self.reference_prior_scope not in {"active", "global"}:
            raise EditorError("reference_prior_scope must be active or global")
        if self.reference_prior_mode not in {
            "lexical", "contrastive", "contrastive-exit", "ballistic", "ballistic-exit"
        }:
            raise EditorError(
                "reference_prior_mode must be contrastive, contrastive-exit, "
                "ballistic, or ballistic-exit"
            )
        prior_routes = []
        try:
            for index, raw_route in enumerate(self.reference_prior_routes):
                if isinstance(raw_route, Mapping):
                    route = raw_route.get("route", raw_route.get("token_ids"))
                    weight = raw_route.get("weight")
                else:
                    route, weight = raw_route
                if not isinstance(route, (tuple, list)) or not route:
                    raise EditorError(
                        f"reference_prior_routes[{index}] must contain a nonempty route"
                    )
                route = tuple(route)
                if any(type(token) is not int or token < 0 for token in route):
                    raise EditorError(
                        f"reference_prior_routes[{index}] has invalid token IDs"
                    )
                if (
                    type(weight) not in (int, float)
                    or not math.isfinite(float(weight))
                    or float(weight) <= 0
                ):
                    raise EditorError(
                        f"reference_prior_routes[{index}] has an invalid weight"
                    )
                prior_routes.append((route, float(weight)))
        except TypeError as exc:
            raise EditorError("reference_prior_routes must be a list of routes") from exc
        object.__setattr__(self, "reference_prior_routes", tuple(prior_routes))
        if (
            type(self.reference_prior_strength) not in (int, float)
            or not math.isfinite(float(self.reference_prior_strength))
            or self.reference_prior_strength < 0.0
        ):
            raise EditorError(
                "reference_prior_strength must be a finite nonnegative number"
            )
        object.__setattr__(
            self, "reference_prior_strength", float(self.reference_prior_strength)
        )
        if (
            type(self.reference_prior_attraction) not in (int, float)
            or not math.isfinite(float(self.reference_prior_attraction))
            or self.reference_prior_attraction < 0.0
        ):
            raise EditorError(
                "reference_prior_attraction must be a finite nonnegative number"
            )
        object.__setattr__(
            self, "reference_prior_attraction", float(self.reference_prior_attraction)
        )
        if (
            type(self.reference_prior_exit_strength) not in (int, float)
            or not math.isfinite(float(self.reference_prior_exit_strength))
            or self.reference_prior_exit_strength < 0.0
        ):
            raise EditorError(
                "reference_prior_exit_strength must be a finite nonnegative number"
            )
        object.__setattr__(
            self, "reference_prior_exit_strength", float(self.reference_prior_exit_strength)
        )
        if prior_routes:
            from .sampling import reference_trie
            object.__setattr__(self, "_reference_prior_trie", reference_trie(tuple(prior_routes)))
        if (
            type(self.temperature) not in {int, float}
            or not math.isfinite(float(self.temperature))
        ):
            raise EditorError("temperature must be a finite number")
        if self.temperature < 0.0:
            raise EditorError("temperature cannot be negative")
        if type(self.top_k) is not int:
            raise EditorError("top_k must be an integer")
        if self.top_k < 1:
            raise EditorError("top_k must be at least 1")
        if (
            type(self.top_p) not in {int, float}
            or not math.isfinite(float(self.top_p))
        ):
            raise EditorError("top_p must be a finite number")
        if not 0.0 < self.top_p <= 1.0:
            raise EditorError("top_p must be in (0, 1]")
        if (
            type(self.min_p) not in {int, float}
            or not math.isfinite(float(self.min_p))
        ):
            raise EditorError("min_p must be a finite number")
        if not 0.0 <= self.min_p <= 1.0:
            raise EditorError("min_p must be in [0, 1]")
        if (
            type(self.repeat_penalty) not in {int, float}
            or not math.isfinite(float(self.repeat_penalty))
            or self.repeat_penalty <= 0.0
        ):
            raise EditorError("repeat_penalty must be a finite number greater than 0")
        if type(self.repeat_last_n) is not int or self.repeat_last_n < -1:
            raise EditorError("repeat_last_n must be -1 or a nonnegative integer")
        for name in ("presence_penalty", "frequency_penalty"):
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise EditorError(f"{name} must be a finite number")
        if type(self.seed) is not int:
            raise EditorError("seed must be an integer")
        if not MIN_SEED <= self.seed <= MAX_SEED:
            raise EditorError(
                f"seed must be between {MIN_SEED} and {MAX_SEED} inclusive"
            )

    @property
    def policy_active(self) -> bool:
        return (
            self.history_penalties_active
            or bool(self.bias_rules)
            or any(group.enabled and group.bias != 0.0 for group in self.bias_groups)
            or any(c.enabled for c in self.group_controls)
            or bool(self.token_preference_vector)
            or bool(self.token_preference_fast_vector)
            or (bool(self.activation_vector) and self.activation_vector_strength != 0.0)
            or self.reference_prior_active
        )

    @property
    def reference_prior_active(self) -> bool:
        return bool(self.reference_prior_routes) and (
            self.reference_prior_strength > 0.0
            or self.reference_prior_attraction > 0.0
            or (
                self.reference_prior_mode.endswith("-exit")
                and self.reference_prior_exit_strength > 0.0
            )
        )

    @property
    def effective_bias_rules(self) -> tuple:
        from .bias_rules import merge_bias_rules
        return merge_bias_rules(
            (*self.bias_rules,
             *(rule for group in self.bias_groups for rule in group.effective_rules()))
        )

    def active_biases(self, history, boundaries=None) -> dict[int, float]:
        """Return the logical rules active for the current model-token tail."""
        from .bias_rules import BiasMatcher
        return BiasMatcher(self.effective_bias_rules).active_biases(history, boundaries)

    def active_reference_prior(self, history, boundaries=None) -> dict[int, float]:
        return self.active_reference_prior_snapshot(history, boundaries).biases

    def active_reference_prior_snapshot(self, history, boundaries=None):
        if not self.reference_prior_active:
            from .sampling import ReferencePriorSnapshot
            return ReferencePriorSnapshot(
                self.reference_prior_scope, self.reference_prior_mode,
                (), 0.0, 0.0, 0.0, (), {}
            )
        active_routes = None
        if self.reference_prior_scope == "active":
            from .bias_rules import BiasMatcher
            active_routes = BiasMatcher(self.effective_bias_rules).active_routes(
                history, boundaries
            )
        from .sampling import reference_prior_snapshot
        return reference_prior_snapshot(
            self.reference_prior_routes,
            history,
            active_routes=active_routes,
            strength=self.reference_prior_strength,
            attraction=self.reference_prior_attraction,
            exit_strength=self.reference_prior_exit_strength,
            scope=self.reference_prior_scope,
            mode=self.reference_prior_mode,
            trie=self._reference_prior_trie,
        )

    @property
    def history_penalties_active(self) -> bool:
        return bool(
            self.repeat_last_n != 0
            and (
                float(self.repeat_penalty) != 1.0
                or float(self.presence_penalty) != 0.0
                or float(self.frequency_penalty) != 0.0
            )
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SamplingConfig":
        """Build configuration from partial values; use from_record for saved state."""

        removed = {"logit_bias", "sequence_bias", "scoped_bias"} & set(value)
        if removed:
            raise EditorError(
                "unsupported bias fields: " + ", ".join(sorted(removed))
            )
        defaults = cls()
        raw_scope = value.get("reference_prior_scope", defaults.reference_prior_scope)
        raw_mode = value.get("reference_prior_mode")
        raw_exit_strength = value.get(
            "reference_prior_exit_strength", defaults.reference_prior_exit_strength
        )
        legacy_exit = value.get("reference_prior_exit_strength", 0.0)
        if raw_mode is None:
            # Normalize records written before scope and behavior mode were
            # separated. A completely partial mapping should retain the
            # current defaults rather than infer a legacy mode from them.
            if not ({
                "reference_prior_scope", "reference_prior_exit_strength"
            } & set(value)):
                raw_mode = defaults.reference_prior_mode
            if raw_mode is None and raw_scope == "ballistic-global":
                raw_scope = "global"
                raw_mode = (
                    "ballistic-exit"
                    if float(legacy_exit) > 0.0
                    else "ballistic"
                )
            elif raw_mode is None and raw_scope == "ballistic-global-exit":
                raw_scope = "global"
                raw_mode = "ballistic-exit"
            elif raw_mode is None:
                raw_mode = (
                    "contrastive-exit"
                    if float(legacy_exit) > 0.0
                    else "contrastive"
                )
        steering_vector = value.get(
            "steering_vector", value.get("activation_vector", defaults.activation_vector)
        )
        steering_strength = value.get(
            "steering_strength",
            value.get("activation_vector_strength", defaults.activation_vector_strength),
        )
        steering_layer = value.get(
            "activation_vector_layer", defaults.activation_vector_layer
        )
        steering_kind = value.get("steering_kind")
        if steering_kind is not None:
            if steering_kind == "output-head-steering-vector":
                steering_layer = "output"
            elif steering_kind == "hidden-state-vector":
                steering_layer = "control-vector"
            else:
                raise EditorError(
                    "steering_kind must identify an output-head or hidden-state vector"
                )
        steering_position = value.get(
            "steering_position",
            value.get("activation_vector_position", defaults.activation_vector_position),
        )
        if steering_kind == "hidden-state-vector" and "steering_position" not in value:
            steering_position = "layers"
        return cls(
            temperature=value.get("temperature", defaults.temperature),
            top_k=value.get("top_k", defaults.top_k),
            top_p=value.get("top_p", defaults.top_p),
            min_p=value.get("min_p", defaults.min_p),
            repeat_penalty=value.get("repeat_penalty", defaults.repeat_penalty),
            repeat_last_n=value.get("repeat_last_n", defaults.repeat_last_n),
            presence_penalty=value.get(
                "presence_penalty", defaults.presence_penalty
            ),
            frequency_penalty=value.get(
                "frequency_penalty", defaults.frequency_penalty
            ),
            seed=value.get("seed", defaults.seed),
            bias_step=value.get("bias_step", defaults.bias_step),
            bias_rules=value.get("bias_rules", ()),
            bias_groups=value.get("bias_groups", ()),
            group_controls=value.get("group_controls", ()),
            token_preference_vector=value.get(
                "token_preference_vector", defaults.token_preference_vector
            ),
            token_preference_fast_vector=value.get("token_preference_fast_vector", ()),
            token_preference_fast_strength=value.get("token_preference_fast_strength", 0.0),
            token_preference_projection_seed=value.get("token_preference_projection_seed", DEFAULT_PROJECTION_SEED),
            token_preference_feature_scheme=value.get(
                "token_preference_feature_scheme", defaults.token_preference_feature_scheme
            ),
            token_preference_whitening_ridge=value.get(
                "token_preference_whitening_ridge", defaults.token_preference_whitening_ridge
            ),
            token_preference_learning_scheme=value.get(
                "token_preference_learning_scheme", defaults.token_preference_learning_scheme
            ),
            token_preference_influence_mode=value.get(
                "token_preference_influence_mode", defaults.token_preference_influence_mode
            ),
            token_preference_influence_kl=value.get(
                "token_preference_influence_kl", defaults.token_preference_influence_kl
            ),
            token_preference_min_gain=value.get("token_preference_min_gain", defaults.token_preference_min_gain),
            token_preference_max_gain=value.get("token_preference_max_gain", defaults.token_preference_max_gain),
            token_preference_coordinate_identity=value.get(
                "token_preference_coordinate_identity", defaults.token_preference_coordinate_identity
            ),
            activation_vector=steering_vector,
            activation_vector_strength=steering_strength,
            activation_vector_layer=steering_layer,
            activation_vector_position=steering_position,
            activation_vector_layer_start=value.get(
                "steering_layer_start",
                value.get("activation_vector_layer_start", defaults.activation_vector_layer_start),
            ),
            activation_vector_layer_end=value.get(
                "steering_layer_end",
                value.get("activation_vector_layer_end", defaults.activation_vector_layer_end),
            ),
            activation_vector_model=value.get(
                "steering_model",
                value.get("activation_vector_model", defaults.activation_vector_model),
            ),
            activation_vector_digest=value.get(
                "steering_digest",
                value.get("activation_vector_digest", defaults.activation_vector_digest),
            ),
            group_control_scheme=value.get(
                "group_control_scheme", defaults.group_control_scheme
            ),
            token_preference_strength=value.get(
                "token_preference_strength", defaults.token_preference_strength
            ),
            reference_prior_routes=value.get(
                "reference_prior_routes", defaults.reference_prior_routes
            ),
            reference_prior_scope=value.get(
                "reference_prior_scope", raw_scope
            ),
            reference_prior_mode=raw_mode,
            reference_prior_strength=value.get(
                "reference_prior_strength", defaults.reference_prior_strength
            ),
            reference_prior_attraction=value.get(
                "reference_prior_attraction", defaults.reference_prior_attraction
            ),
            reference_prior_exit_strength=value.get(
                "reference_prior_exit_strength", defaults.reference_prior_exit_strength
            ),
        )

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "SamplingConfig":
        """Restore a complete saved configuration without filling defaults."""
        if not isinstance(value, Mapping):
            raise EditorError("saved sampler settings must be an object")
        value = dict(value)
        legacy_layer = value.get("activation_vector_layer", "output")
        legacy_position = value.get("activation_vector_position", "current")
        for old_name, new_name in (
            ("activation_vector", "steering_vector"),
            ("activation_vector_strength", "steering_strength"),
            ("activation_vector_position", "steering_position"),
            ("activation_vector_layer_start", "steering_layer_start"),
            ("activation_vector_layer_end", "steering_layer_end"),
            ("activation_vector_model", "steering_model"),
            ("activation_vector_digest", "steering_digest"),
        ):
            if old_name in value:
                if new_name in value:
                    raise EditorError(
                        f"saved sampler settings contain both {old_name} and {new_name}"
                    )
                value[new_name] = value.pop(old_name)
        if "steering_kind" not in value:
            value["steering_kind"] = (
                "hidden-state-vector"
                if legacy_layer == "control-vector"
                else "output-head-steering-vector"
            )
        value.setdefault(
            "steering_position",
            "layers" if legacy_layer == "control-vector" else legacy_position,
        )
        # Scheme fields were introduced after the original v1 records. Missing
        # fields mean the original mathematics, never an implicit upgrade.
        for name, default in (
            ("token_preference_feature_scheme", "random-projection-unit-v1"),
            ("token_preference_whitening_ridge", DEFAULT_WHITENING_RIDGE),
            ("token_preference_learning_scheme", "sgd-v1"),
            ("token_preference_influence_mode", "manual"),
            ("token_preference_influence_kl", 0.05),
            ("token_preference_min_gain", 0.0),
            ("token_preference_max_gain", 8.0),
            ("token_preference_coordinate_identity", None),
            ("group_control_scheme", "appearance-feedback-v1"),
            ("steering_vector", ()),
            ("steering_strength", 0.0),
            ("steering_kind", "output-head-steering-vector"),
            ("steering_position", "current"),
            ("steering_layer_start", None),
            ("steering_layer_end", None),
            ("steering_model", ""),
            ("steering_digest", ""),
        ):
            value.setdefault(name, default)
        if "reference_prior_mode" not in value:
            # Older saved segments used ballistic-global as a scope and had
            # no separate mode field. Fill the new fields before checking
            # completeness so replay remains reconstructable.
            legacy_scope = value.get("reference_prior_scope", "active")
            legacy_exit = value.get("reference_prior_exit_strength", 0.0)
            if legacy_scope == "ballistic-global":
                value["reference_prior_scope"] = "global"
                value["reference_prior_mode"] = (
                    "ballistic-exit" if float(legacy_exit) > 0.0 else "ballistic"
                )
            elif legacy_scope == "ballistic-global-exit":
                value["reference_prior_scope"] = "global"
                value["reference_prior_mode"] = "ballistic-exit"
            else:
                value["reference_prior_mode"] = (
                    "contrastive-exit" if float(legacy_exit) > 0.0 else "contrastive"
                )
        if "reference_prior_exit_strength" not in value:
            value["reference_prior_exit_strength"] = 0.0
        removed = {"logit_bias", "sequence_bias", "scoped_bias"} & set(value)
        if removed:
            raise EditorError(
                "saved sampler settings use removed bias fields: "
                + ", ".join(sorted(removed))
            )
        expected = cls().to_dict()
        missing = sorted(expected.keys() - value.keys())
        if missing:
            raise EditorError("saved sampler settings missing: " + ", ".join(missing))
        for name in ("rng_scheme", "policy_scheme", "history_scope"):
            if value[name] != expected[name]:
                raise EditorError(f"unsupported {name}: {value[name]!r}")
        return cls.from_mapping(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            **({"bias_rules": [rule.to_dict() for rule in self.bias_rules]} if self.bias_rules else {}),
            **({"bias_groups": [group.to_dict() for group in self.bias_groups]} if self.bias_groups else {}),
            **({"group_controls": [c.to_dict() for c in self.group_controls]} if self.group_controls else {}),
            **(
                {
                    "token_preference_vector": list(self.token_preference_vector),
                    "token_preference_strength": self.token_preference_strength,
                    "token_preference_fast_vector": list(self.token_preference_fast_vector),
                    "token_preference_fast_strength": self.token_preference_fast_strength,
                    "token_preference_projection_seed": self.token_preference_projection_seed,
                }
                if (self.token_preference_vector or self.token_preference_strength != 1.0
                    or self.token_preference_fast_vector or self.token_preference_fast_strength != 0.0
                    or self.token_preference_projection_seed != DEFAULT_PROJECTION_SEED)
                else {}
            ),
            **({"bias_step": self.bias_step} if self.bias_step != 0.5 else {}),
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": self.repeat_last_n,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "history_scope": "model-visible-prefix-tail-v1",
            "policy_scheme": SAMPLING_POLICY_SCHEME,
            "seed": self.seed,
            "rng_scheme": RNG_SCHEME,
            "token_preference_feature_scheme": self.token_preference_feature_scheme,
            "token_preference_whitening_ridge": self.token_preference_whitening_ridge,
            "token_preference_learning_scheme": self.token_preference_learning_scheme,
            "token_preference_influence_mode": self.token_preference_influence_mode,
            "token_preference_influence_kl": self.token_preference_influence_kl,
            "token_preference_min_gain": self.token_preference_min_gain,
            "token_preference_max_gain": self.token_preference_max_gain,
            "token_preference_coordinate_identity": (
                self.token_preference_coordinate_identity.to_dict()
                if self.token_preference_coordinate_identity is not None else None
            ),
            **(
                {
                    "steering_vector": list(self.activation_vector),
                    "steering_strength": self.activation_vector_strength,
                    "steering_kind": (
                        "hidden-state-vector"
                        if self.activation_vector_layer == "control-vector"
                        else "output-head-steering-vector"
                    ),
                    "steering_position": self.activation_vector_position,
                    "steering_layer_start": self.activation_vector_layer_start,
                    "steering_layer_end": self.activation_vector_layer_end,
                    "steering_model": (
                        json.loads(self.activation_vector_model)
                        if self.activation_vector_model else {}
                    ),
                    "steering_digest": self.activation_vector_digest,
                }
                if (
                    self.activation_vector
                    or self.activation_vector_strength != 0.0
                    or self.activation_vector_model
                    or self.activation_vector_digest
                    or self.activation_vector_layer_start is not None
                    or self.activation_vector_layer_end is not None
                )
                else {}
            ),
            "group_control_scheme": self.group_control_scheme,
            "reference_prior_routes": [
                {"route": list(route), "weight": weight}
                for route, weight in self.reference_prior_routes
            ],
            "reference_prior_scope": self.reference_prior_scope,
            "reference_prior_mode": self.reference_prior_mode,
            "reference_prior_strength": self.reference_prior_strength,
            "reference_prior_attraction": self.reference_prior_attraction,
            "reference_prior_exit_strength": self.reference_prior_exit_strength,
        }


@dataclass(frozen=True)
class Candidate:
    rank: int
    token_id: int
    text: str
    raw_probability: float
    decoder_probability: float
    is_eog: bool
    bias: float = 0.0
    policy_rank: int | None = None
    policy_probability: float | None = None
    policy_logit_adjustment: float | None = None
    raw_logit: float | None = None
    effective_logit: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bias": self.bias,
            "rank": self.rank,
            "token_id": self.token_id,
            "text": self.text,
            "raw_probability": self.raw_probability,
            "decoder_probability": self.decoder_probability,
            "decoder_supported": self.decoder_probability > 0.0,
            "is_eog": self.is_eog,
            "policy_rank": self.policy_rank,
            "policy_probability": self.policy_probability,
            "policy_logit_adjustment": self.policy_logit_adjustment,
            "raw_logit": self.raw_logit,
            "effective_logit": self.effective_logit,
        }


@dataclass(frozen=True)
class ChoiceSet:
    choice_set_id: str
    prompt_id: str
    aligned_step: int
    sampling_coordinate: int
    context_token_sha256: str
    context_text_tail: str
    proposal_token_id: int
    proposal_text: str
    proposal_raw_probability: float
    proposal_decoder_probability: float
    proposal_is_eog: bool
    candidates: tuple[Candidate, ...]
    vocabulary_size: int | None = None
    proposal_raw_rank: int | None = None
    proposal_policy_rank: int | None = None
    proposal_policy_probability: float | None = None
    proposal_policy_logit_adjustment: float | None = None


class ActionKind(str, Enum):
    ACCEPT = "accept"
    SELECT = "select"
    INSERT = "insert"


class InsertMode(str, Enum):
    CONTINUATION = "continuation"
    EXACT = "exact"


@dataclass(frozen=True)
class EditAction:
    kind: ActionKind
    selected_rank: int | None = None
    supplied_text: str | None = None
    insert_mode: InsertMode | None = None

    @classmethod
    def accept(cls) -> "EditAction":
        return cls(ActionKind.ACCEPT)

    @classmethod
    def select(cls, rank: int) -> "EditAction":
        if rank < 1:
            raise EditorError("selected rank must be positive")
        return cls(ActionKind.SELECT, selected_rank=rank)

    @classmethod
    def insert(cls, text: str, mode: InsertMode) -> "EditAction":
        if not text:
            raise EditorError("inserted text cannot be empty")
        return cls(ActionKind.INSERT, supplied_text=text, insert_mode=mode)

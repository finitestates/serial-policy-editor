"""Pure domain types for the reduced policy editor."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .latent_features import DEFAULT_PROJECTION_SEED


class EditorError(ValueError):
    """Raised when an editor command or configuration is invalid."""


RNG_SCHEME = "blake2b64-token-prefix-quantile-v2"
SAMPLING_POLICY_SCHEME = "spe-history-aware-decoder-policy-v1"
MIN_SEED = -(1 << 63)
MAX_SEED = (1 << 63) - 1


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.8
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
    latent_preference_z: tuple = ()
    latent_strength: float = 1.0
    latent_preference_fast_z: tuple = ()
    latent_fast_strength: float = 0.0
    latent_projection_seed: int = DEFAULT_PROJECTION_SEED
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
        for name in ("latent_preference_z", "latent_preference_fast_z"):
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
        if (self.latent_preference_z and self.latent_preference_fast_z
                and len(self.latent_preference_z) != len(self.latent_preference_fast_z)):
            raise EditorError("slow and fast latent vectors must have the same dimension")
        for name in ("latent_strength", "latent_fast_strength"):
            value = getattr(self, name)
            if (type(value) not in (int, float)
                    or not math.isfinite(float(value)) or value < 0.0):
                raise EditorError(f"{name} must be a finite nonnegative number")
            object.__setattr__(self, name, float(value))
        if (type(self.latent_projection_seed) is not int
                or not MIN_SEED <= self.latent_projection_seed <= MAX_SEED):
            raise EditorError("latent projection seed must be a signed 64-bit integer")
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
            or bool(self.latent_preference_z)
            or bool(self.latent_preference_fast_z)
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
            latent_preference_z=value.get(
                "latent_preference_z", defaults.latent_preference_z
            ),
            latent_preference_fast_z=value.get("latent_preference_fast_z", ()),
            latent_fast_strength=value.get("latent_fast_strength", 0.0),
            latent_projection_seed=value.get("latent_projection_seed", DEFAULT_PROJECTION_SEED),
            latent_strength=value.get(
                "latent_strength", defaults.latent_strength
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
                    "latent_preference_z": list(self.latent_preference_z),
                    "latent_strength": self.latent_strength,
                    "latent_preference_fast_z": list(self.latent_preference_fast_z),
                    "latent_fast_strength": self.latent_fast_strength,
                    "latent_projection_seed": self.latent_projection_seed,
                }
                if (self.latent_preference_z or self.latent_strength != 1.0
                    or self.latent_preference_fast_z or self.latent_fast_strength != 0.0
                    or self.latent_projection_seed != DEFAULT_PROJECTION_SEED)
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

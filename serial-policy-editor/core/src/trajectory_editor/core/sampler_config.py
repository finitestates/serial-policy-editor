"""The core sampler configuration contract.

``SamplerConfig`` owns settings needed to produce and replay a token draw:
candidate filters, CFG, history penalties, manual/conditional bias rules,
and optional steering-vector application.  Research controls are deliberately
not fields here. Historical records may contain wider fields; persistence
projects them into this type before execution.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .errors import EditorError
from .sampling import MAX_SEED, MIN_SEED, RNG_SCHEME


SAMPLING_POLICY_SCHEME = "spe-history-aware-decoder-policy-v1"


@dataclass(frozen=True)
class SamplerConfig:
    """Core, replayable sampler settings.

    The tuple fields intentionally accept the serialized rule records used by
    the existing project format.  They are normalized to canonical rule and
    group objects during validation.
    """

    temperature: float = 1.0
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    typical_p: float = 1.0
    tail_free_z: float = 1.0
    draw_kernel: str = "categorical"
    cfg_unconditional_prompt: str | None = None
    cfg_scale: float = 1.0
    cfg_prefix_tokens: int = 0
    repeat_penalty: float = 1.0
    repeat_last_n: int = 64
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = 12345
    bias_step: float = 0.5
    bias_rules: tuple = ()
    bias_groups: tuple = ()
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

    def __post_init__(self) -> None:
        from ..bias_rules import BiasGroup, BiasRule

        if (
            type(self.bias_step) not in (int, float)
            or not math.isfinite(self.bias_step)
            or self.bias_step <= 0
        ):
            raise EditorError("bias_step must be a finite positive number")
        try:
            rules = tuple(BiasRule.from_record(rule) for rule in self.bias_rules)
        except TypeError as exc:
            raise EditorError("bias_rules must be a list of rules") from exc
        if len({rule.key for rule in rules}) != len(rules):
            raise EditorError("duplicate bias rule")
        object.__setattr__(
            self,
            "bias_rules",
            tuple(sorted((rule for rule in rules if rule.bias != 0), key=lambda rule: rule.sort_key)),
        )
        try:
            groups = tuple(BiasGroup.from_record(group) for group in self.bias_groups)
        except TypeError as exc:
            raise EditorError("bias_groups must be a list of groups") from exc
        if len({group.name for group in groups}) != len(groups):
            raise EditorError("duplicate bias group")
        object.__setattr__(self, "bias_groups", tuple(sorted(groups, key=lambda group: group.name)))

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
            model = json.dumps(dict(model), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if not isinstance(model, str):
            raise EditorError("activation_vector_model must be a JSON object")
        if model:
            try:
                parsed_model = json.loads(model)
            except (TypeError, ValueError) as exc:
                raise EditorError("activation_vector_model must be valid JSON") from exc
            if not isinstance(parsed_model, dict):
                raise EditorError("activation_vector_model must be a JSON object")
            model = json.dumps(parsed_model, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "activation_vector_model", model)
        digest = self.activation_vector_digest
        if not isinstance(digest, str) or (digest and re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise EditorError("activation_vector_digest must be a lowercase SHA-256 digest")
        object.__setattr__(self, "activation_vector_digest", digest)
        if (
            self.activation_vector_layer == "control-vector"
            and self.activation_vector
            and self.activation_vector_strength != 0.0
            and not digest
        ):
            raise EditorError("active control-vector activation requires a verified digest")
        if type(self.temperature) not in {int, float} or not math.isfinite(float(self.temperature)):
            raise EditorError("temperature must be a finite number")
        if self.temperature < 0.0:
            raise EditorError("temperature cannot be negative")
        if type(self.top_k) is not int:
            raise EditorError("top_k must be an integer")
        if self.top_k < 1:
            raise EditorError("top_k must be at least 1")
        if type(self.top_p) not in {int, float} or not math.isfinite(float(self.top_p)):
            raise EditorError("top_p must be a finite number")
        if not 0.0 < self.top_p <= 1.0:
            raise EditorError("top_p must be in (0, 1]")
        if type(self.min_p) not in {int, float} or not math.isfinite(float(self.min_p)):
            raise EditorError("min_p must be a finite number")
        if not 0.0 <= self.min_p <= 1.0:
            raise EditorError("min_p must be in [0, 1]")
        for name in ("typical_p", "tail_free_z"):
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(float(value)):
                raise EditorError(f"{name} must be a finite number")
            if not 0.0 < float(value) <= 1.0:
                raise EditorError(f"{name} must be in (0, 1]")
        if self.draw_kernel not in {"categorical", "gumbel-max"}:
            raise EditorError("draw_kernel must be categorical or gumbel-max")
        if self.cfg_unconditional_prompt is not None and not isinstance(self.cfg_unconditional_prompt, str):
            raise EditorError("cfg_unconditional_prompt must be text or null")
        if (
            type(self.cfg_scale) not in {int, float}
            or not math.isfinite(float(self.cfg_scale))
            or float(self.cfg_scale) < 0.0
        ):
            raise EditorError("cfg_scale must be a finite nonnegative number")
        object.__setattr__(self, "cfg_scale", float(self.cfg_scale))
        if type(self.cfg_prefix_tokens) is not int or self.cfg_prefix_tokens < 0:
            raise EditorError("cfg_prefix_tokens must be a nonnegative integer")
        if self.cfg_unconditional_prompt is not None and self.cfg_prefix_tokens == 0:
            raise EditorError("cfg_prefix_tokens must be positive when a CFG unconditional prompt is set")
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
            raise EditorError(f"seed must be between {MIN_SEED} and {MAX_SEED} inclusive")

    @property
    def policy_active(self) -> bool:
        return (
            self.history_penalties_active
            or bool(self.bias_rules)
            or any(group.enabled and group.bias != 0.0 for group in self.bias_groups)
            or (bool(self.activation_vector) and self.activation_vector_strength != 0.0)
        )

    @property
    def effective_bias_rules(self) -> tuple:
        from ..bias_rules import merge_bias_rules

        return merge_bias_rules(
            (*self.bias_rules, *(rule for group in self.bias_groups for rule in group.effective_rules()))
        )

    def active_biases(self, history) -> dict[int, float]:
        from ..bias_rules import BiasMatcher

        return BiasMatcher(self.effective_bias_rules).active_biases(history)

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
    def from_mapping(cls, value: Mapping[str, Any]) -> "SamplerConfig":
        """Build a core sampler from a partial mapping."""

        if not isinstance(value, Mapping):
            raise EditorError("sampler settings must be an object")
        defaults = cls()
        steering_kind = value.get("steering_kind")
        steering_layer = value.get("activation_vector_layer", defaults.activation_vector_layer)
        if steering_kind == "output-head-steering-vector":
            steering_layer = "output"
        elif steering_kind == "hidden-state-vector":
            steering_layer = "control-vector"
        elif steering_kind is not None:
            raise EditorError(
                "steering_kind must identify an output-head or hidden-state vector"
            )
        steering_vector = value.get(
            "steering_vector", value.get("activation_vector", defaults.activation_vector)
        )
        steering_position = value.get(
            "steering_position", value.get("activation_vector_position", defaults.activation_vector_position)
        )
        if steering_kind == "hidden-state-vector" and "steering_position" not in value:
            steering_position = "layers"
        return cls(
            temperature=value.get("temperature", defaults.temperature),
            top_k=value.get("top_k", defaults.top_k),
            top_p=value.get("top_p", defaults.top_p),
            min_p=value.get("min_p", defaults.min_p),
            typical_p=value.get("typical_p", defaults.typical_p),
            tail_free_z=value.get("tail_free_z", defaults.tail_free_z),
            draw_kernel=value.get("draw_kernel", defaults.draw_kernel),
            cfg_unconditional_prompt=value.get(
                "cfg_unconditional_prompt", defaults.cfg_unconditional_prompt
            ),
            cfg_scale=value.get("cfg_scale", defaults.cfg_scale),
            cfg_prefix_tokens=value.get("cfg_prefix_tokens", defaults.cfg_prefix_tokens),
            repeat_penalty=value.get("repeat_penalty", defaults.repeat_penalty),
            repeat_last_n=value.get("repeat_last_n", defaults.repeat_last_n),
            presence_penalty=value.get("presence_penalty", defaults.presence_penalty),
            frequency_penalty=value.get("frequency_penalty", defaults.frequency_penalty),
            seed=value.get("seed", defaults.seed),
            bias_step=value.get("bias_step", defaults.bias_step),
            bias_rules=value.get("bias_rules", defaults.bias_rules),
            bias_groups=value.get("bias_groups", defaults.bias_groups),
            activation_vector=steering_vector,
            activation_vector_strength=value.get(
                "steering_strength",
                value.get("activation_vector_strength", defaults.activation_vector_strength),
            ),
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
        )

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "SamplerConfig":
        """Restore only the core portion of a saved sampler record."""

        if not isinstance(value, Mapping):
            raise EditorError("saved sampler settings must be an object")
        for name, expected in (
            ("rng_scheme", RNG_SCHEME),
            ("policy_scheme", SAMPLING_POLICY_SCHEME),
            ("history_scope", "model-visible-prefix-tail-v1"),
        ):
            if name in value and value[name] != expected:
                raise EditorError(f"unsupported {name}: {value[name]!r}")
        return cls.from_mapping(value)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "typical_p": self.typical_p,
            "tail_free_z": self.tail_free_z,
            "draw_kernel": self.draw_kernel,
            "cfg_unconditional_prompt": self.cfg_unconditional_prompt,
            "cfg_scale": self.cfg_scale,
            "cfg_prefix_tokens": self.cfg_prefix_tokens,
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": self.repeat_last_n,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "history_scope": "model-visible-prefix-tail-v1",
            "policy_scheme": SAMPLING_POLICY_SCHEME,
            "seed": self.seed,
            "rng_scheme": RNG_SCHEME,
        }
        if self.bias_rules:
            result["bias_rules"] = [rule.to_dict() for rule in self.bias_rules]
        if self.bias_groups:
            result["bias_groups"] = [group.to_dict() for group in self.bias_groups]
        if self.bias_step != 0.5:
            result["bias_step"] = self.bias_step
        if (
            self.activation_vector
            or self.activation_vector_strength != 0.0
            or self.activation_vector_model
            or self.activation_vector_digest
            or self.activation_vector_layer_start is not None
            or self.activation_vector_layer_end is not None
        ):
            result.update(
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
                    "steering_model": json.loads(self.activation_vector_model)
                    if self.activation_vector_model
                    else {},
                    "steering_digest": self.activation_vector_digest,
                }
            )
        return result


__all__ = ["SAMPLING_POLICY_SCHEME", "SamplerConfig"]

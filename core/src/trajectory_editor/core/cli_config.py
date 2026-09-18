"""Core command-line sampler configuration.

This module is intentionally limited to the replayable ``SamplerConfig``
contract. Wider experimental configuration is outside the core package and
must not be added here.
"""

from __future__ import annotations

import argparse
import secrets
from dataclasses import replace
from pathlib import Path
from typing import Any

from .errors import EditorError
from .sampler_config import SamplerConfig
from .sampling import MAX_SEED, MIN_SEED


CORE_SAMPLER_FIELDS = (
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "typical_p",
    "tail_free_z",
    "draw_kernel",
    "cfg_unconditional_prompt",
    "cfg_scale",
    "cfg_prefix_tokens",
    "repeat_penalty",
    "repeat_last_n",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "bias_rules",
    "bias_groups",
    "bias_step",
)

SAMPLER_ALIASES = {
    "temp": "temperature",
    "rep": "repeat_penalty",
    "rep_pen": "repeat_penalty",
    "repeat": "repeat_penalty",
    "presence": "presence_penalty",
    "frequency": "frequency_penalty",
}


def add_core_sampler_arguments(
    parser: argparse.ArgumentParser, *, include_vector: bool = True
) -> None:
    """Add only the sampler options understood by the core runtime."""

    sampling = parser.add_argument_group("sampling")
    for name, kind in (
        ("temperature", float),
        ("top_k", int),
        ("top_p", float),
        ("min_p", float),
        ("typical_p", float),
        ("tail_free_z", float),
        ("repeat_penalty", float),
        ("repeat_last_n", int),
        ("presence_penalty", float),
        ("frequency_penalty", float),
    ):
        sampling.add_argument("--" + name.replace("_", "-"), type=kind)
    sampling.add_argument(
        "--draw-kernel",
        choices=("categorical", "gumbel-max"),
        help="candidate draw kernel (default: categorical)",
    )
    sampling.add_argument(
        "--cfg-unconditional-prompt",
        dest="cfg_unconditional_prompt",
        help="unconditional prompt for prefix classifier-free guidance",
    )
    sampling.add_argument(
        "--cfg-scale",
        type=float,
        help="classifier-free guidance scale (requires an unconditional prompt)",
    )
    sampling.add_argument(
        "--cfg-prefix-tokens",
        type=int,
        help="number of generated prefix tokens to guide with CFG",
    )
    sampling.add_argument(
        "--bias-step",
        type=float,
        help="default positive bias adjustment (default: 0.5)",
    )
    if include_vector:
        sampling.add_argument(
            "--steering-vector",
            dest="activation_vector",
            type=Path,
            metavar="PATH",
            help="load an output-head steering or hidden-state vector artifact",
        )
        sampling.add_argument(
            "--steering-strength",
            dest="activation_strength",
            type=float,
            metavar="VALUE",
            help="override the steering vector artifact strength",
        )
    parser.set_defaults(bias_rules=None, bias_groups=None)
    seeds = sampling.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int)
    seeds.add_argument(
        "--random-seed",
        action="store_true",
        help="choose a random sampler seed from the supported signed 64-bit range",
    )


def random_seed() -> int:
    return secrets.randbelow(MAX_SEED - MIN_SEED + 1) + MIN_SEED


def sampler_from_args(
    args: argparse.Namespace, source: SamplerConfig | None = None
) -> SamplerConfig:
    """Construct a core sampler from parsed core options and an optional source."""

    base = source if source is not None else SamplerConfig()
    if getattr(args, "_model_changed", False):
        updates: dict[str, Any] = {
            "bias_rules": (),
            "bias_groups": (),
        }
        if hasattr(base, "activation_vector"):
            updates.update(
                activation_vector=(),
                activation_vector_model="",
                activation_vector_digest="",
                activation_vector_strength=0.0,
                activation_vector_layer_start=None,
                activation_vector_layer_end=None,
            )
        base = replace(base, **updates)
    values = {
        name: getattr(args, name)
        if getattr(args, name, None) is not None
        else getattr(base, name)
        for name in CORE_SAMPLER_FIELDS
    }
    return SamplerConfig(**values)


def sampler_overrides_present(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name, None) is not None for name in CORE_SAMPLER_FIELDS
    ) or bool(
        getattr(args, "activation_vector", None) is not None
        or getattr(args, "activation_strength", None) is not None
    )


def sampler_override(
    current: SamplerConfig, raw: str, *, seed_factory=random_seed
) -> SamplerConfig:
    """Apply live core sampler edits without reconstructing discarded fields."""

    values = {name: getattr(current, name) for name in CORE_SAMPLER_FIELDS}
    pieces = raw.replace(",", " ").split()
    if not pieces:
        return current
    if len(pieces) == 1 and pieces[0].lower() in {"random", "random-seed"}:
        values["seed"] = seed_factory()
        return replace(current, **values)
    for piece in pieces:
        if "=" not in piece:
            raise EditorError("sampler changes use key=value (for example top_k=20)")
        key, value = piece.split("=", 1)
        key = SAMPLER_ALIASES.get(key.strip().lower(), key.strip().lower())
        if key not in values or key in {"bias_rules", "bias_groups"}:
            raise EditorError(f"unknown sampler field {key!r}")
        try:
            if key in {"top_k", "repeat_last_n", "seed", "cfg_prefix_tokens"}:
                values[key] = int(value)
            elif key == "draw_kernel":
                if value not in {"categorical", "gumbel-max"}:
                    raise ValueError
                values[key] = value
            else:
                values[key] = float(value)
        except ValueError as exc:
            raise EditorError(f"invalid value for {key}: {value!r}") from exc
    return replace(current, **values)


def apply_activation_artifact(sampling: SamplerConfig, artifact, args) -> SamplerConfig:
    """Apply a validated steering artifact to a core sampler."""

    if artifact is None:
        return sampling
    strength = (
        args.activation_strength
        if "activation_strength" in getattr(args, "_explicit_options", set())
        and args.activation_strength is not None
        else artifact.strength
    )
    return artifact.apply_to_sampling(sampling, strength=strength)


__all__ = [
    "CORE_SAMPLER_FIELDS",
    "SAMPLER_ALIASES",
    "add_core_sampler_arguments",
    "apply_activation_artifact",
    "random_seed",
    "sampler_from_args",
    "sampler_override",
    "sampler_overrides_present",
]

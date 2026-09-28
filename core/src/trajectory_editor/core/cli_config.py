"""Core command-line sampler configuration.

This module is intentionally limited to the replayable ``SamplerConfig``
contract. Wider experimental configuration is outside the core package and
must not be added here.
"""

from __future__ import annotations

import argparse
import json
import secrets
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from .errors import EditorError
from .sampler_config import SamplerConfig
from .sampling import GUMBEL_NOISE_ADDRESSES, MAX_SEED, MIN_SEED


CORE_SAMPLER_FIELDS = (
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "typical_p",
    "tail_free_z",
    "draw_kernel",
    "gaussian_noise_std",
    "gumbel_top_k",
    "gumbel_noise_address",
    "gumbel_noise_scale",
    "cfg_unconditional_prompt",
    "cfg_scale",
    "cfg_prefix_tokens",
    "repeat_penalty",
    "repeat_last_n",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "token_biases",
    "bias_groups",
)

_UNFILTERED_VALUES = {
    "temperature": 1.0,
    "top_k": None,
    "top_p": 1.0,
    "min_p": 0.0,
    "typical_p": 1.0,
    "tail_free_z": 1.0,
}

SAMPLER_ALIASES = {
    "temp": "temperature",
    "rep": "repeat_penalty",
    "rep_pen": "repeat_penalty",
    "repeat": "repeat_penalty",
    "presence": "presence_penalty",
    "frequency": "frequency_penalty",
    "gumbel-noise-address": "gumbel_noise_address",
    "gumbel-noise-scale": "gumbel_noise_scale",
}


class _StoreTopK(argparse.Action):
    """Store an optional top-k while retaining whether None was explicit."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, "_top_k_specified", True)


class _StoreGumbelTopK(argparse.Action):
    """Store an optional Gumbel menu size and retain explicit ``none``."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, "_gumbel_top_k_specified", True)


def add_core_sampler_arguments(
    parser: argparse.ArgumentParser, *, include_vector: bool = True
) -> None:
    """Add only the sampler options understood by the core runtime."""

    sampling = parser.add_argument_group("sampling")
    for name, kind in (
        ("temperature", float),
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
        "--top-k",
        type=_parse_top_k,
        action=_StoreTopK,
        default=None,
        metavar="N|none",
        help="keep the top N candidates; none disables top-k (default: 40)",
    )
    parser.set_defaults(_top_k_specified=False)
    sampling.add_argument(
        "--unfiltered",
        action="store_true",
        help="disable temperature and all candidate filters",
    )
    sampling.add_argument(
        "--draw-kernel",
        choices=("categorical", "gumbel-max", "gaussian-max"),
        help=(
            "candidate draw kernel (categorical, gumbel-max, or gaussian-max; "
            "default: categorical)"
        ),
    )
    sampling.add_argument(
        "--gumbel-top-k",
        type=_parse_top_k,
        action=_StoreGumbelTopK,
        default=None,
        metavar="N|none",
        help=(
            "use Gumbel-Max and limit the menu to the top N Gumbel-ranked "
            "candidates; none disables the menu limit"
        ),
    )
    parser.set_defaults(_gumbel_top_k_specified=False)
    sampling.add_argument(
        "--gumbel-noise-address",
        choices=GUMBEL_NOISE_ADDRESSES,
        help=(
            "address Gumbel noise by token ID or one-based full-vocabulary "
            "model rank (default: token-id)"
        ),
    )
    sampling.add_argument(
        "--gumbel-noise-scale",
        type=float,
        help=(
            "select Gumbel-Max and scale perturbations after filtering "
            "(default: 1; 0 disables perturbations)"
        ),
    )
    sampling.add_argument(
        "--gaussian-noise-std",
        type=float,
        help="standard deviation for gaussian-max noise in scaled-logit units",
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
    parser.set_defaults(token_biases=None, bias_groups=None)
    seeds = sampling.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int)
    seeds.add_argument(
        "--random-seed",
        action="store_true",
        help="choose a random sampler seed from the supported signed 64-bit range",
    )


def random_seed() -> int:
    return secrets.randbelow(MAX_SEED - MIN_SEED + 1) + MIN_SEED


def _parse_top_k(value: str) -> int | None:
    if value.strip().lower() == "none":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer or 'none'") from exc


def _top_k_was_specified(args: argparse.Namespace) -> bool:
    specified = getattr(args, "_top_k_specified", None)
    return (
        specified
        if specified is not None
        else getattr(args, "top_k", None) is not None
    )


def _gumbel_top_k_was_specified(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "_gumbel_top_k_specified", False))


def _gumbel_noise_address_was_specified(args: argparse.Namespace) -> bool:
    return getattr(args, "gumbel_noise_address", None) is not None


def _gumbel_noise_scale_was_specified(args: argparse.Namespace) -> bool:
    return getattr(args, "gumbel_noise_scale", None) is not None


def _gumbel_requested(args: argparse.Namespace) -> bool:
    return (
        _gumbel_noise_address_was_specified(args)
        or _gumbel_noise_scale_was_specified(args)
        or (_gumbel_top_k_was_specified(args) and args.gumbel_top_k is not None)
    )


def _unfiltered_values(args: argparse.Namespace) -> dict[str, Any]:
    values = dict(_UNFILTERED_VALUES)
    cli_explicit = getattr(args, "_cli_explicit_options", set())
    if "unfiltered" not in cli_explicit:
        for name in _UNFILTERED_VALUES:
            if name not in cli_explicit:
                continue
            values[name] = getattr(args, name, None)
    return values


def sampler_from_args(
    args: argparse.Namespace, source: SamplerConfig | None = None
) -> SamplerConfig:
    """Construct a core sampler from parsed core options and an optional source."""

    base = source if source is not None else SamplerConfig()
    if getattr(args, "_model_changed", False):
        updates: dict[str, Any] = {
            "token_biases": (),
            "bias_groups": (),
        }
        if hasattr(base, "activation_vector"):
            updates.update(
                activation_vector=(),
                activation_vector_digest="",
                activation_vector_strength=0.0,
                activation_vector_layer_start=None,
                activation_vector_layer_end=None,
            )
        base = replace(base, **updates)
    values = {}
    for name in CORE_SAMPLER_FIELDS:
        if name == "top_k":
            values[name] = (
                getattr(args, name, None)
                if _top_k_was_specified(args)
                else base.top_k
            )
        elif name == "gumbel_top_k":
            values[name] = (
                getattr(args, name, None)
                if _gumbel_top_k_was_specified(args)
                else base.gumbel_top_k
            )
        else:
            value = getattr(args, name, None)
            values[name] = getattr(base, name) if value is None else value
    if _gumbel_requested(args) and getattr(args, "draw_kernel", None) is None:
        values["draw_kernel"] = "gumbel-max"
    elif (
        getattr(args, "draw_kernel", None) is not None
        and args.draw_kernel != "gumbel-max"
        and not _gumbel_top_k_was_specified(args)
    ):
        values["gumbel_top_k"] = None
    if getattr(args, "unfiltered", False):
        values.update(_unfiltered_values(args))
    return SamplerConfig(**values)


def sampler_overrides_present(args: argparse.Namespace) -> bool:
    for name in CORE_SAMPLER_FIELDS:
        if name == "top_k":
            if _top_k_was_specified(args):
                return True
            continue
        if name == "gumbel_top_k":
            if _gumbel_top_k_was_specified(args):
                return True
            continue
        if getattr(args, name, None) is not None:
            return True
    return bool(getattr(args, "unfiltered", False)) or bool(
        getattr(args, "activation_vector", None) is not None
        or getattr(args, "activation_strength", None) is not None
    )


def sampler_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Return only sampler values explicitly supplied on the CLI/profile."""

    values: dict[str, Any] = {}
    for name in CORE_SAMPLER_FIELDS:
        if name == "top_k":
            if _top_k_was_specified(args):
                values[name] = getattr(args, name, None)
            continue
        if name == "gumbel_top_k":
            if _gumbel_top_k_was_specified(args):
                values[name] = getattr(args, name, None)
            continue
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    if _gumbel_requested(args) and getattr(args, "draw_kernel", None) is None:
        values["draw_kernel"] = "gumbel-max"
    elif (
        getattr(args, "draw_kernel", None) is not None
        and args.draw_kernel != "gumbel-max"
        and not _gumbel_top_k_was_specified(args)
    ):
        values["gumbel_top_k"] = None
    if getattr(args, "unfiltered", False):
        values.update(_unfiltered_values(args))
    return values


def sampler_override(current: SamplerConfig, raw: str) -> SamplerConfig:
    """Apply live core sampler edits without reconstructing discarded fields.

    The returned configuration is recorded as one ``SetSampler`` action by
    the live-session owner, including seed changes. A complete serialized
    ``SamplerConfig`` may also be supplied as JSON after ``s``.
    """

    payload = raw.strip()
    if payload.startswith("{"):
        try:
            record = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise EditorError(f"invalid sampler JSON: {exc.msg}") from exc
        if not isinstance(record, Mapping):
            raise EditorError("sampler JSON must be an object")
        return SamplerConfig.from_record(record)

    values = {name: getattr(current, name) for name in CORE_SAMPLER_FIELDS}
    pieces = payload.replace(",", " ").split()
    if not pieces:
        return current
    explicit_keys: set[str] = set()
    for piece in pieces:
        if "=" not in piece:
            raise EditorError("sampler changes use key=value (for example top_k=20 or top_k=none)")
        key, value = piece.split("=", 1)
        key = SAMPLER_ALIASES.get(key.strip().lower(), key.strip().lower())
        explicit_keys.add(key)
        if key not in values or key in {"token_biases", "bias_groups"}:
            raise EditorError(f"unknown sampler field {key!r}")
        try:
            if key in {"top_k", "gumbel_top_k"} and value.strip().lower() == "none":
                values[key] = None
            elif key in {
                "top_k", "gumbel_top_k", "repeat_last_n", "cfg_prefix_tokens", "seed"
            }:
                values[key] = int(value)
            elif key == "draw_kernel":
                if value not in {"categorical", "gumbel-max", "gaussian-max"}:
                    raise ValueError
                values[key] = value
            elif key == "gumbel_noise_address":
                if value not in GUMBEL_NOISE_ADDRESSES:
                    raise ValueError
                values[key] = value
            else:
                values[key] = float(value)
        except ValueError as exc:
            raise EditorError(f"invalid value for {key}: {value!r}") from exc
    if "gumbel_top_k" in explicit_keys and values["gumbel_top_k"] is not None:
        if "draw_kernel" not in explicit_keys:
            values["draw_kernel"] = "gumbel-max"
    elif (
        {"gumbel_noise_address", "gumbel_noise_scale"} & explicit_keys
        and "draw_kernel" not in explicit_keys
    ):
        values["draw_kernel"] = "gumbel-max"
    elif (
        "draw_kernel" in explicit_keys
        and values["draw_kernel"] != "gumbel-max"
        and "gumbel_top_k" not in explicit_keys
    ):
        values["gumbel_top_k"] = None
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
    "sampler_overrides_from_args",
    "sampler_overrides_present",
]

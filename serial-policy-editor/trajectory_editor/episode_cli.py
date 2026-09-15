"""Command line for the reduced Serial Policy Editor.

A live episode alternates between the ordinary teacher loop and live edges.
Token budgets, replay exhaustion, and replay divergence yield at a live edge;
only EOG or explicit ``end`` seals the episode.
"""

from __future__ import annotations

import argparse
import copy
import math
from contextlib import ExitStack
from dataclasses import replace
import secrets
import sys
from pathlib import Path
from typing import Any

from prompt_toolkit import prompt
from prompt_toolkit.validation import Validator

from .bias_presets import load_bias_preset, project_biases
from .bias_catalog import load_catalog, validate_catalog, compile_catalog, load_yaml_source, BiasCatalog
from .lexical_reference import load_reference
from .backend_factory import BACKEND_NAMES, create_backend
from .decoder import KV_CACHE_TYPES, LlamaCppSettings
from .domain import MAX_SEED, MIN_SEED, EditorError, SamplingConfig
from .episode_lifecycle import (
    POLICY_FIELDS, _inherit_budget, _model_continuation, _visible_tokens, _restore_engine,
    _create_episode, _rewind_episode, _fork_engine, _spr_engine_from_source,
)
from .episode_actions import Write
from .episode_engine import EpisodeEngine
from .episode_policy import (
    EdgeRequested,
    EpisodeRunner,
    ForkRequested,
    SeamlessEdgeRequested,
    SeamlessRewindRequested,
    TapeStep,
    ReplayPlan,
    WriteLearningResult,
)
from .episode_projector import project_episode, project_fork_map, project_lineage, project_procedure
from .episode_store import EpisodeStore
from .episode_recovery import recover_sampler_record
from .episode_ui import InteractivePolicy, PolicyViewPreferences
from .latent_features import DEFAULT_PROJECTION_CHUNK_SIZE, DEFAULT_PROJECTION_SEED
from .latent_preference import LatentPreferenceConfig, LatentPreferenceLearner, LatentPreferenceResult
from .online_learning import LearningResult, OnlineLearner
from .transformers_backend import TransformersSettings
from .tui import TerminalIO
from .ui_themes import LIVE_THEME_NAMES
from .version import VERSION


SAMPLER_FIELDS = (
    "temperature",
    "top_k",
    "top_p",
    "min_p",
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
REFERENCE_PRIOR_PRESETS = {
    "active": ("active", "contrastive"),
    "active-exit": ("active", "contrastive-exit"),
    "ballistic-active": ("active", "ballistic"),
    "ballistic-active-exit": ("active", "ballistic-exit"),
    "global": ("global", "contrastive"),
    "global-exit": ("global", "contrastive-exit"),
    "ballistic-global": ("global", "ballistic"),
    "ballistic-global-exit": ("global", "ballistic-exit"),
}


def _random_seed() -> int:
    """Return a uniformly chosen seed from the supported signed 64-bit range."""

    return secrets.randbelow(MAX_SEED - MIN_SEED + 1) + MIN_SEED


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _read_initial_prompt() -> str:
    return prompt(
        "Write at least one character. Press Escape then Enter to continue.\n\n",
        multiline=True,
        validator=Validator.from_callable(
            lambda text: len(text) >= 1,
            error_message="Write at least one character.",
        ),
        validate_while_typing=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="policy-editor",
        description=(
            "Interactive token-policy editor with Serial Policy Replay. "
            "Token budgets are checkpoints, not run termination."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("episodes.sqlite3"),
        help="compact episode workspace used for SPR and token evidence",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--new-prompt", metavar="TEXT")
    source.add_argument("--new-prompt-file", type=Path, metavar="FILE")
    source.add_argument(
        "--replay",
        metavar="EPISODE_ID",
        help="Serial Policy Replay of an existing episode, then yield live",
    )
    parser.add_argument("--until", type=int, metavar="Y",
                        help="replay through source token boundary Y, then return to EDGE")
    source.add_argument(
        "--resume",
        "--continue-from",
        dest="resume",
        metavar="EPISODE_ID",
        help="resume an unsealed/checkpointed episode in the same episode id",
    )
    parser.add_argument(
        "--fixed-config", action="store_true",
        help="freeze source-initial sampler settings plus explicit overrides during --replay",
    )
    source.add_argument("--fork-from", metavar="EPISODE_ID")
    source.add_argument("--project", metavar="EPISODE_ID")
    parser.add_argument("--procedure", action="store_true", help="show a manual replay procedure with --project")
    source.add_argument("--list", action="store_true", dest="list_episodes")
    parser.add_argument("--at", type=int, metavar="BOUNDARY")
    parser.add_argument(
        "--lineage",
        metavar="EPISODE_ID",
        help="show the ordinary fork family and related replays for an episode (with --list)",
    )
    parser.add_argument("--episode-id")
    parser.add_argument(
        "--output",
        type=Path,
        help="write full final episode text here when the episode is sealed",
    )

    parser.add_argument("--backend", choices=BACKEND_NAMES, default=None)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="optional visible-token allowance before a live edge (default: unlimited)",
    )
    parser.add_argument("--table-depth", type=int, default=12)
    parser.add_argument("--search-radius", type=int, default=3)
    parser.add_argument("--hold-default", type=int, default=100)
    parser.add_argument("--context-chars", type=int, default=0, help="Context character limit (0 keeps all context)")
    parser.add_argument("--plain-ui", action="store_true")
    parser.add_argument(
        "--cache",
        choices=("auto", "off"),
        default="auto",
        help="use backend cache when available (default: auto; use off for full-prefix evaluation)",
    )
    parser.add_argument(
        "--no-cache",
        dest="cache",
        action="store_const",
        const="off",
        help="disable backend caching and use full-prefix evaluation",
    )
    parser.add_argument(
        "--seamless",
        action="store_true",
        help="accepted for compatibility; live history rewind is always enabled",
    )
    parser.add_argument(
        "--manual-acceptance",
        action="store_true",
        help="leave each teacher command blank instead of prefilling the sampled proposal",
    )
    policy_view = parser.add_mutually_exclusive_group()
    policy_view.add_argument(
        "--policy-view", "--show-policy-rank", dest="show_policy_rank", action="store_true",
        default=None, help="show policy diagnostics without changing raw-rank ordering (default: automatic)",
    )
    policy_view.add_argument(
        "--no-policy-view", dest="show_policy_rank", action="store_false",
        help="hide automatic policy diagnostics; V can toggle them during the session",
    )
    parser.add_argument(
        "--bias-catalog",
        type=Path,
        help="load a model-matched human-readable bias catalog for b name and b @name",
    )
    parser.add_argument("--groups", type=Path, help="load semantic term/group YAML directly")
    parser.add_argument("--reference", type=Path, help="load standalone relative lexical weights from YAML")
    parser.add_argument("--reference-strength", type=float, help="overall lexical influence (default: 0.25)")
    parser.add_argument("--group-level", type=float, default=1.0, help="appearance objective level; 1 requests twice/half the baseline odds")
    parser.add_argument(
        "--reference-prior",
        choices=(
            "off",
            "active", "active-exit",
            "ballistic-active", "ballistic-active-exit",
            "global", "global-exit",
            "ballistic-global", "ballistic-global-exit",
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reference-prior-strength",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reference-prior-attraction",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reference-prior-exit-strength",
        type=float,
        help=argparse.SUPPRESS,
    )
    learning = parser.add_argument_group("online learning")
    learning.add_argument(
        "--online-learning",
        "--online-learning-enabled",
        dest="online_learning",
        action="store_true",
        help="fit explicitly learnable manual group weights to teacher selections; appearance objectives run independently",
    )
    learning.add_argument("--learning-rate", type=float, default=0.05)
    learning.add_argument(
        "--epsilon", "--learning-epsilon", dest="learning_epsilon",
        type=float, default=0.05,
    )
    learning.add_argument(
        "--max-step", "--learning-max-step", dest="learning_max_step",
        type=float, default=0.25,
    )
    learning.add_argument(
        "--min-bias", "--learning-min-bias", dest="learning_min_bias",
        type=float, default=-4.0,
    )
    learning.add_argument(
        "--max-bias", "--learning-max-bias", dest="learning_max_bias",
        type=float, default=4.0,
    )
    learning.add_argument("--learning-severity-cap", type=_positive_int, default=1000)
    learning.add_argument("--learning-dead-zone-rank", type=_positive_int, default=1)
    learning.add_argument("--learning-no-severity-attenuation", action="store_true")
    learning.add_argument("--learning-rejection-strength", type=float, default=0.)
    learning.add_argument("--learning-decay", type=float, default=0.)
    learning.add_argument(
        "--learnable-groups",
        nargs="+",
        metavar="GROUP",
        help="restrict online learning to these named bias groups",
    )
    learning.add_argument(
        "--learn-from-write",
        action="store_true",
        help="allow enabled learners to learn from live typed writes (off by default)",
    )
    latent = parser.add_argument_group("latent preference learning")
    latent.add_argument(
        "--latent-preference",
        "--latent-preference-enabled",
        dest="latent_preference",
        action="store_true",
        help="learn an anonymous latent preference vector from live raw-rank selections (off by default)",
    )
    latent.add_argument("--latent-dimension", type=int, default=64)
    latent.add_argument("--latent-learning-rate", type=float, default=0.05)
    latent.add_argument("--latent-strength", type=float, default=1.0)
    latent.add_argument("--latent-max-step", type=float, default=0.25)
    latent.add_argument("--latent-max-norm", type=float, default=4.0)
    latent.add_argument("--latent-decay", type=float, default=0.0)
    latent.add_argument("--latent-severity-cap", type=_positive_int, default=1000)
    latent.add_argument(
        "--latent-no-severity-attenuation", action="store_true",
        help="use severity 1 outside the dead zone; retain learning step and memory norm limits",
    )
    latent.add_argument("--latent-dead-zone-rank", type=_positive_int, default=1)
    latent.add_argument("--latent-rejection-strength", type=float, default=0.0)
    latent.add_argument("--latent-fast-slow", action="store_true")
    latent.add_argument("--latent-fast-learning-rate", type=float)
    latent.add_argument("--latent-fast-decay", type=float, default=0.10)
    latent.add_argument("--latent-fast-strength", type=float)
    latent.add_argument("--latent-fast-max-step", type=float)
    latent.add_argument("--latent-fast-max-norm", type=float)
    latent_seeds = latent.add_mutually_exclusive_group()
    latent_seeds.add_argument("--latent-seed", type=int)
    latent_seeds.add_argument("--latent-random-seed", action="store_true")
    latent.add_argument(
        "--latent-projection-chunk-size",
        type=_positive_int,
        default=DEFAULT_PROJECTION_CHUNK_SIZE,
        help=(
            "rows projected at once when building latent features; lower this "
            "to reduce peak memory at the cost of slower initialization"
        ),
    )
    parser.add_argument("--theme", choices=LIVE_THEME_NAMES)
    parser.add_argument(
        "--divergence-policy", choices=("handoff", "ballistic"), default="handoff"
    )

    sampling = parser.add_argument_group("sampling")
    for name, kind in (
        ("temperature", float),
        ("top_k", int),
        ("top_p", float),
        ("min_p", float),
        ("repeat_penalty", float),
        ("repeat_last_n", int),
        ("presence_penalty", float),
        ("frequency_penalty", float),
    ):
        sampling.add_argument("--" + name.replace("_", "-"), type=kind)
    sampling.add_argument("--biases", type=Path, help="load a JSON bias preset (replaces the saved bias set)")
    sampling.add_argument("--bias-step", type=float, help="default positive bias adjustment (default: 0.5)")
    parser.set_defaults(bias_rules=None, bias_groups=None)
    seed_options = sampling.add_mutually_exclusive_group()
    seed_options.add_argument("--seed", type=int)
    seed_options.add_argument(
        "--random-seed",
        action="store_true",
        help="choose a random sampler seed from the supported signed 64-bit range",
    )

    llama = parser.add_argument_group("llama.cpp")
    llama.add_argument("--n-ctx", type=int, default=2048)
    llama.add_argument("--n-batch", type=int, default=256)
    llama.add_argument("--n-ubatch", type=int)
    llama.add_argument("--n-threads", type=int)
    llama.add_argument("--n-threads-batch", type=int)
    llama.add_argument("--n-gpu-layers", type=int)
    llama.add_argument("--main-gpu", type=int)
    llama.add_argument("--no-flash-attn", action="store_true")
    for component in ("k", "v"):
        llama.add_argument(f"--cache-type-{component}", dest=f"type_{component}",
            choices=KV_CACHE_TYPES, help="KV cache precision (default: library default; quantized V requires Flash Attention)")
    llama.add_argument("--no-mmap", action="store_true")
    llama.add_argument("--use-mlock", action="store_true")

    transformers = parser.add_argument_group("Transformers")
    transformers.add_argument("--transformers-device", default="auto")
    transformers.add_argument(
        "--transformers-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    transformers.add_argument(
        "--transformers-device-map",
        choices=("auto", "balanced", "balanced_low_0", "sequential"),
    )
    transformers.add_argument(
        "--transformers-attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2", "flex_attention"),
    )
    transformers.add_argument(
        "--transformers-quantization",
        choices=("none", "bitsandbytes-8bit", "bitsandbytes-4bit"),
        default="none",
    )
    transformers.add_argument("--transformers-trust-remote-code", action="store_true")
    transformers.add_argument("--transformers-slow-tokenizer", action="store_true")
    transformers.add_argument("--transformers-torch-threads", type=int)
    transformers.add_argument("--transformers-torch-interop-threads", type=int)

    projection = parser.add_argument_group("projection")
    projection.add_argument("--biases-only", action="store_true", help="with --project, emit a loadable JSON bias preset")
    projection.add_argument(
        "--rules-only",
        action="store_true",
        help="with --biases-only, flatten named groups into ordinary logical rules",
    )
    projection.add_argument(
        "--editor-friendly",
        action="store_true",
        help="with --biases-only, emit standalone YAML group definitions for policy-editor-bias",
    )
    projection.add_argument(
        "--annotations", choices=("none", "inline", "footnotes"), default="none"
    )
    projection.add_argument("--with-loss", action="store_true")
    projection.add_argument("--with-rank", action="store_true")
    projection.add_argument("--with-policy-rank", action="store_true")
    projection.add_argument(
        "--full-evidence",
        action="store_true",
        help="footnote teacher-selected tokens with proposal agreement, NLL, and ranks",
    )
    projection.add_argument(
        "--with-model-probs",
        "--with-model",
        dest="with_model_probs",
        action="store_true",
        help="add raw-model and decoder probabilities to teacher-token footnotes",
    )
    projection.add_argument(
        "--with-lineage",
        action="store_true",
        help="append fork-family and replay metadata to a projection",
    )
    return parser


def _sampling_from_args(
    args: argparse.Namespace, source: SamplingConfig | None = None
) -> SamplingConfig:
    base = source if source is not None else SamplingConfig()
    if getattr(args, "_model_changed", False):
        base = replace(base, bias_rules=(), bias_groups=(), group_controls=(),
                       reference_prior_routes=(), latent_preference_z=(), latent_preference_fast_z=())
    values = {
        name: getattr(args, name)
        if getattr(args, name) is not None
        else getattr(base, name)
        for name in SAMPLER_FIELDS
    }
    values.update({
        "group_controls": base.group_controls,
        "latent_preference_z": base.latent_preference_z,
        "latent_strength": base.latent_strength,
        "latent_preference_fast_z": base.latent_preference_fast_z,
        "latent_fast_strength": base.latent_fast_strength,
        "latent_projection_seed": base.latent_projection_seed,
        "reference_prior_routes": base.reference_prior_routes,
        "reference_prior_scope": base.reference_prior_scope,
        "reference_prior_mode": base.reference_prior_mode,
        "reference_prior_strength": base.reference_prior_strength,
        "reference_prior_attraction": base.reference_prior_attraction,
        "reference_prior_exit_strength": base.reference_prior_exit_strength,
    })
    if getattr(args, "bias_groups", None) is not None:
        preset = getattr(args, "_bias_preset", None)
        values["group_controls"] = preset.group_controls if preset is not None else ()
    return SamplingConfig(**values)


def _apply_latent_preset(
    sampling: SamplingConfig,
    preset_latent: SamplingConfig | None,
) -> SamplingConfig:
    if preset_latent is None:
        return sampling
    return replace(
        sampling,
        group_controls=preset_latent.group_controls,
        latent_preference_z=preset_latent.latent_preference_z,
        latent_strength=preset_latent.latent_strength,
        latent_preference_fast_z=preset_latent.latent_preference_fast_z,
        latent_fast_strength=preset_latent.latent_fast_strength,
        latent_projection_seed=preset_latent.latent_projection_seed,
    )


def _latent_config_from_args(args: argparse.Namespace) -> LatentPreferenceConfig:
    return LatentPreferenceConfig(
        enabled=args.latent_preference, dimension=args.latent_dimension,
        learning_rate=args.latent_learning_rate, latent_strength=args.latent_strength,
        max_step=args.latent_max_step, max_norm=args.latent_max_norm,
        decay=args.latent_decay, severity_cap=args.latent_severity_cap,
        no_severity_attenuation=args.latent_no_severity_attenuation,
        dead_zone_rank=args.latent_dead_zone_rank,
        rejection_strength=args.latent_rejection_strength, fast_slow=args.latent_fast_slow,
        fast_learning_rate=args.latent_fast_learning_rate, fast_decay=args.latent_fast_decay,
        fast_strength=args.latent_fast_strength, fast_max_step=args.latent_fast_max_step,
        fast_max_norm=args.latent_fast_max_norm,
    )


def _apply_latent_seed(sampling: SamplingConfig, seed: int | None, io: TerminalIO,
                       *, replay: bool = False) -> SamplingConfig:
    if seed is None or seed == sampling.latent_projection_seed:
        return sampling
    if replay:
        raise EditorError("explicit latent seed conflicts with saved replay seed")
    if sampling.latent_preference_z or sampling.latent_preference_fast_z:
        io.write("Latent projection coordinate system changed: latent preference memory reset (slow and fast).")
    return replace(sampling, latent_projection_seed=seed,
                   latent_preference_z=(), latent_preference_fast_z=())


def _apply_catalog_reference_prior(
    sampling: SamplingConfig,
    catalog,
    args: argparse.Namespace,
) -> SamplingConfig:
    """Attach compiled reference routes to the episode's saved sampler state."""

    direct = getattr(args, "_reference_routes", None)
    strength = getattr(args, "reference_strength", None)
    preset = getattr(args, "_bias_preset", None)
    requested = args.reference_prior
    if direct is not None:
        if requested not in (None, "off"):
            raise EditorError("--reference uses one standalone lexical policy; omit legacy --reference-prior modes")
        return replace(sampling, reference_prior_routes=() if requested == "off" else direct,
                       reference_prior_scope="global", reference_prior_mode="lexical",
                       reference_prior_strength=0.25 if strength is None else strength,
                       reference_prior_attraction=0., reference_prior_exit_strength=0.)
    if requested is None:
        if catalog is not None and catalog.reference_prior_routes:
            return replace(sampling, reference_prior_routes=tuple((r.token_ids, r.weight) for r in catalog.reference_prior_routes),
                           reference_prior_scope="global", reference_prior_mode="lexical",
                           reference_prior_strength=0.25 if strength is None else strength,
                           reference_prior_attraction=0., reference_prior_exit_strength=0.)
        if preset is not None:
            sampling = replace(sampling, **{k: v for k, v in preset.__dict__.items() if k.startswith("reference_prior_")})
        return sampling if strength is None else replace(sampling, reference_prior_strength=strength)
    if requested == "off":
        return replace(
            sampling,
            reference_prior_routes=(),
            reference_prior_scope="active",
            reference_prior_mode="contrastive",
        )
    if catalog is None or not catalog.reference_prior_routes:
        raise EditorError(
            "--reference-prior requires --bias-catalog compiled with --reference"
        )
    strength = (
        sampling.reference_prior_strength
        if args.reference_prior_strength is None
        else args.reference_prior_strength
    )
    attraction = (
        sampling.reference_prior_attraction
        if args.reference_prior_attraction is None
        else args.reference_prior_attraction
    )
    exit_strength = (
        sampling.reference_prior_exit_strength
        if args.reference_prior_exit_strength is None
        else args.reference_prior_exit_strength
    )
    return replace(
        sampling,
        reference_prior_routes=tuple(
            (route.token_ids, route.weight)
            for route in catalog.reference_prior_routes
        ),
        reference_prior_scope=REFERENCE_PRIOR_PRESETS[requested][0],
        reference_prior_mode=REFERENCE_PRIOR_PRESETS[requested][1],
        reference_prior_strength=strength,
        reference_prior_attraction=attraction,
        reference_prior_exit_strength=exit_strength,
    )


def _sampler_override(current: SamplingConfig, raw: str) -> SamplingConfig:
    values = {name: getattr(current, name) for name in SAMPLER_FIELDS}
    values.update({
        "latent_preference_z": current.latent_preference_z,
        "latent_strength": current.latent_strength,
        "latent_preference_fast_z": current.latent_preference_fast_z,
        "latent_fast_strength": current.latent_fast_strength,
        "latent_projection_seed": current.latent_projection_seed,
    })
    pieces = raw.replace(",", " ").split()
    if not pieces:
        return current
    if len(pieces) == 1 and pieces[0].lower() in {"random", "random-seed"}:
        values["seed"] = _random_seed()
        return replace(current, **values)
    for piece in pieces:
        if "=" not in piece:
            raise EditorError("sampler changes use key=value (for example top_k=20)")
        key, value = piece.split("=", 1)
        key = SAMPLER_ALIASES.get(key.strip().lower(), key.strip().lower())
        if key not in values or key in {"bias_rules", "bias_groups"}:
            raise EditorError(f"unknown sampler field {key!r}")
        try:
            values[key] = int(value) if key in {"top_k", "repeat_last_n", "seed"} else float(value)
        except ValueError as exc:
            raise EditorError(f"invalid value for {key}: {value!r}") from exc
    return replace(current, **values)


def _backend(args: argparse.Namespace):
    args.backend = args.backend or "llama.cpp"
    if args.model is None:
        raise EditorError("--model is required to start, resume, fork, or replay")
    llama = LlamaCppSettings(
        type_k=args.type_k,
        type_v=args.type_v,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_ubatch=args.n_ubatch,
        n_threads=args.n_threads,
        n_threads_batch=args.n_threads_batch,
        n_gpu_layers=args.n_gpu_layers,
        main_gpu=args.main_gpu,
        flash_attn=not args.no_flash_attn,
        use_mmap=not args.no_mmap,
        use_mlock=args.use_mlock,
    )
    transformers = TransformersSettings(
        device=args.transformers_device,
        dtype=args.transformers_dtype,
        device_map=args.transformers_device_map,
        attention_implementation=args.transformers_attention_implementation,
        quantization_method=args.transformers_quantization,
        trust_remote_code=args.transformers_trust_remote_code,
        use_fast_tokenizer=not args.transformers_slow_tokenizer,
        torch_num_threads=args.transformers_torch_threads,
        torch_num_interop_threads=args.transformers_torch_interop_threads,
    )
    print(f"Loading model with {args.backend}: {args.model} ...", flush=True)
    result = create_backend(
        args.backend,
        args.model,
        llama_settings=llama,
        transformers_settings=transformers,
        cache_mode=args.cache,
    )
    print("Model loaded.", flush=True)
    return result


def _load_episode_backend(args, source, io, *, use_saved=False, current_backend=None, current_provenance=None):
    """Load a saved execution context, confirming deliberate model changes."""
    selected = copy.copy(args)
    saved = source["backend"] if source else {}
    old_path = saved.get("model_path")
    if use_saved:
        selected.model = None
        selected.backend = None
    if selected.model is None and old_path:
        selected.model = Path(old_path)
    selected.backend = selected.backend or saved.get("backend") or "llama.cpp"
    if selected.backend not in BACKEND_NAMES:
        selected.backend = args.backend or "llama.cpp"
    # Persisted launch options avoid reconstructing device/quantization settings
    # from diagnostic effective values. Explicit launch flags take precedence.
    explicit = getattr(args, "_explicit_options", set()) if not use_saved else set()
    saved_options = dict(saved.get("load_options", {}))
    if not saved_options:
        for key, value in saved.get("runtime_configuration", {}).items():
            if saved.get("backend") == "transformers":
                key = "transformers_" + key
            elif key in {"flash_attn", "use_mmap"}:
                key, value = {"flash_attn": "no_flash_attn", "use_mmap": "no_mmap"}[key], not value
            if hasattr(selected, key) and key != "seed":
                saved_options[key] = value
    for key, value in saved_options.items():
        if key not in explicit:
            setattr(selected, key, value)
    while True:
        if selected.model is None:
            path = io.read("Saved model location unavailable. Model path (Enter cancels)> ")
            if not path:
                raise EditorError("model loading cancelled")
            selected.model = Path(path).expanduser()
        changed = bool(source and old_path and (
            Path(old_path).resolve() != selected.model.resolve()
            or (saved.get("backend") in BACKEND_NAMES and saved.get("backend") != selected.backend)
        ))
        if changed:
            answer = io.read(f"Previously used {old_path} ({saved.get('backend')}). Continue with {selected.model} ({selected.backend}) in a new linked episode? [y/N]> ")
            if not answer or answer.strip().lower() not in {"y", "yes"}:
                raise EditorError("model change cancelled")
        try:
            if (current_backend is not None and current_provenance
                and current_provenance.get("model_path") == str(selected.model.resolve())
                and current_provenance.get("backend") == selected.backend
                and all(getattr(selected, key, None) == value
                        for key, value in current_provenance.get("load_options", {}).items())):
                return current_backend, current_provenance, changed
            backend = _backend(selected)
            provenance = dict(backend.provenance(include_model_sha256=False))
            provenance["model_path"] = str(selected.model.resolve())
            provenance["load_options"] = {
                key: value for key, value in vars(selected).items()
                if key.startswith("transformers_") or key in {
                    "n_ctx", "n_batch", "n_ubatch", "n_threads", "n_threads_batch",
                    "n_gpu_layers", "main_gpu", "no_flash_attn", "no_mmap", "use_mlock", "cache", "type_k", "type_v"
                }
            }
            return backend, provenance, changed
        except (EditorError, OSError, RuntimeError) as exc:
            if not source:
                raise
            io.write(f"Could not load {selected.model}: {exc}")
            path = io.read("Replacement model path (Enter cancels)> ")
            if not path:
                raise EditorError("model loading cancelled") from exc
            kind = io.read("Backend: llama.cpp or transformers (Enter keeps current)> ")
            if kind:
                if kind.strip() not in BACKEND_NAMES:
                    io.write("Unknown backend.")
                    continue
                selected.backend = kind.strip()
            selected.model = Path(path).expanduser()


def _interactive_policy(
    args: argparse.Namespace, store: EpisodeStore, episode_id: str, io: TerminalIO,
    *, catalog=None,
) -> InteractivePolicy:
    preferences = getattr(args, "_policy_view_preferences", None)
    if preferences is None:
        preferences = PolicyViewPreferences(show=args.show_policy_rank)
        args._policy_view_preferences = preferences
    return InteractivePolicy(
        io=io,
        menu_size=args.table_depth,
        search_radius=args.search_radius,
        default_hold_tokens=args.hold_default,
        context_characters=args.context_chars,
        manual_acceptance=args.manual_acceptance,
        view_preferences=preferences,
        learning_enabled=args.online_learning or args.latent_preference,
        store=store,
        episode_id=episode_id,
        seamless=io.supports_live_choices,
        catalog=catalog,
        group_level=getattr(args, "group_level", 1.0),
    )


def _sampler_summary(config: SamplingConfig) -> str:
    summary = (
        f"temp={config.temperature:g} top_k={config.top_k} top_p={config.top_p:g} "
        f"min_p={config.min_p:g} rep={config.repeat_penalty:g}/{config.repeat_last_n} "
        f"presence={config.presence_penalty:g} frequency={config.frequency_penalty:g} "
        f"seed={config.seed}"
    )
    if config.bias_groups:
        summary += " groups=" + ",".join(
            f"{group.name}:{group.bias:g}" for group in config.bias_groups
        )
    if (config.latent_preference_z or config.latent_preference_fast_z
            or config.latent_projection_seed != DEFAULT_PROJECTION_SEED):
        norm = sum(value * value for value in config.latent_preference_z) ** 0.5
        fast_norm = sum(value * value for value in config.latent_preference_fast_z) ** 0.5
        summary += (f" latent_norm={norm:g} latent_strength={config.latent_strength:g}"
                    f" latent_fast_norm={fast_norm:g} latent_fast_strength={config.latent_fast_strength:g}"
                    f" latent_seed={config.latent_projection_seed}")
    return summary


def _online_learning_notice(io: TerminalIO, result: LearningResult) -> None:
    weights = ", ".join(
        f"{name}={value:g}" for name, value in result.new_group_weights.items()
    )
    io.write(
        f"Online learning @ boundary {result.observation_boundary + 1}: "
        f"selected token {result.chosen_token_id}, "
        f"rank {result.old_policy_rank}, "
        f"update norm {result.update_norm:.4g} · groups {weights}"
    )


def _latent_preference_notice(
    io: TerminalIO, result: LatentPreferenceResult
) -> None:
    io.write(
        f"Latent preference @ boundary {result.observation_boundary + 1}: "
        f"selected token {result.chosen_token_id}, "
        f"rank {result.old_policy_rank}, "
        f"update norm {result.update_norm:.4g}, z norm {result.z_norm:.4g}"
        + (f", fast update {result.fast_update_norm:.4g}, fast norm {result.fast_z_norm:.4g}"
           if result.old_fast_z else "")
    )


def _write_learning_notice(io: TerminalIO, result: WriteLearningResult) -> None:
    parts = [
        f"Write learning @ boundary {result.boundary_after}: "
        f"{result.token_count} typed tokens"
    ]
    if result.group_result is not None:
        weights = ", ".join(
            f"{name}={value:g}"
            for name, value in result.group_result.new_group_weights.items()
        )
        parts.append(
            f"group update norm {result.group_result.update_norm:.4g} · groups {weights}"
        )
    if result.latent_result is not None:
        parts.append(
            f"latent update norm {result.latent_result.update_norm:.4g}, "
            f"z norm {result.latent_result.z_norm:.4g}"
        )
    io.write(" · ".join(parts))


def _live_edge_menu(
    io: TerminalIO,
    store: EpisodeStore,
    episode_id: str,
    engine: EpisodeEngine,
) -> tuple[str, Any]:
    live_surface = bool(
        getattr(io, "supports_live_choices", False)
        and callable(getattr(io, "read_live_edge_command", None))
    )
    while True:
        if live_surface:
            raw = io.read_live_edge_command(  # type: ignore[attr-defined]
                episode_id=store.label(episode_id),
                boundary=engine.boundary,
                current_budget=engine.max_tokens,
                remaining_tokens=engine.remaining,
                sampler_summary=_sampler_summary(engine.sampling),
            )
        else:
            io.write(store.label(episode_id))
            io.write("[ls / ls all] episodes  [#N] switch  [name TITLE] rename  [rewind N] delete back to N")
            io.write(
                f"\nLive edge @ boundary {engine.boundary} · {_sampler_summary(engine.sampling)}"
            )
            raw = io.read(
                "[c]ontinue  [n N/off] budget  [s key=value] sampler  "
                "([s random-seed] randomize)  [f N] fork  [fm] fork map  "
                "[spr ID [--until Y | m]] replay  [p]roject  [e]nd  [q]uit > "
            )
        if raw is None:
            return "quit", None
        text = raw.strip()
        lower = text.lower()
        if lower in {"ls", "ls all"}:
            io.page(store.workspace_list(include_finished=lower == "ls all", current=episode_id))
            selected = io.read("Episode #number (Enter returns)> ")
            if selected and selected.strip():
                try:
                    return "switch", store.resolve_id(selected.strip())
                except EditorError as exc:
                    io.write(str(exc))
            continue
        if lower.startswith("name "):
            store.rename(episode_id, text[5:])
            continue
        if lower.startswith("#"):
            try:
                return "switch", store.resolve_id(text)
            except EditorError as exc:
                io.write(str(exc))
            continue
        if lower.startswith("rewind "):
            try:
                target = int(text.split()[1])
                _rewind_episode(store, episode_id, engine, target)
                store.record_interaction(episode_id, target, "seamless-rewind", {"to_boundary": target})
            except (ValueError, EditorError) as exc:
                io.write(str(exc))
            continue
        if lower in {"q", "quit"}:
            return "quit", None
        if lower in {"e", "end"}:
            return "end", None
        if lower in {"c", "continue", ""}:
            return "continue", "keep"
        if lower in {"p", "project", "r", "review"}:
            io.page(
                project_episode(
                    store,
                    episode_id,
                    annotations="footnotes",
                    full_evidence=True,
                ).text
            )
            continue
        if lower in {"fm", "fork-map", "forkmap"}:
            io.page(project_fork_map(store, episode_id))
            while True:
                entered = io.read(
                    f"Fork boundary (0..{engine.boundary}; blank cancels) > "
                )
                if entered is None:
                    return "quit", None
                value = entered.strip()
                if not value:
                    break
                try:
                    target = int(value)
                except ValueError:
                    io.write("Fork boundary must be an integer.")
                    continue
                if not 0 <= target <= engine.boundary:
                    io.write(f"Fork boundary must be 0..{engine.boundary}.")
                    continue
                return "fork", target
            continue
        parts = text.split(maxsplit=1)
        if len(parts) == 2 and parts[0].lower() in {"n", "next"}:
            if parts[1].lower() in {"off", "none", "unlimited"}:
                return "continue", None
            try:
                budget = int(parts[1])
            except ValueError:
                io.write("Budget must be a positive integer.")
                continue
            if budget < 1:
                io.write("Budget must be at least 1.")
                continue
            return "continue", budget
        if parts and parts[0].lower() in {"s", "sampler"}:
            payload = parts[1] if len(parts) == 2 else ""
            if not payload:
                entered = io.read(
                    "sampler key=value changes (blank cancels; e.g. top_k=20 temperature=.8)> "
                )
                payload = entered or ""
            if not payload.strip():
                continue
            try:
                updated = _sampler_override(engine.sampling, payload)
            except EditorError as exc:
                io.write(f"[invalid sampler change] {exc}")
                continue
            engine.sampling = updated
            store.record_sampling_segment(
                episode_id,
                start_boundary=engine.boundary,
                sampling=updated,
                stream_fingerprint=engine.stream_fingerprint,
                coordinate_offset=engine.coordinate_offset,
            )
            store.record_interaction(
                episode_id,
                engine.boundary,
                "sampling-transition",
                {"sampling": updated.to_dict()},
            )
            continue
        if len(parts) == 2 and parts[0].lower() in {"f", "fork"}:
            try:
                target = int(parts[1])
            except ValueError:
                io.write("Fork boundary must be an integer.")
                continue
            if not 0 <= target <= engine.boundary:
                io.write(f"Fork boundary must be 0..{engine.boundary}.")
                continue
            return "fork", target
        if len(parts) == 2 and parts[0].lower() in {"spr", "replay"}:
            try:
                replay_args = parts[1].split()
                if not replay_args:
                    raise EditorError("Use spr EPISODE [--until Y | m].")
                source_id = store.resolve_id(replay_args[0])
                until = None
                if replay_args[1:] == ["m"]:
                    io.page("Source replay map (recorded output; replay may differ).\n"
                            "Source 0 inserts only the prompt; destination tokens remain individually indexed.\n"
                            + project_fork_map(store, source_id))
                    while True:
                        entered = io.read("Replay through source boundary (blank cancels) > ")
                        if entered is None or not entered.strip():
                            break
                        try:
                            until = int(entered)
                            store.replay_until(source_id, until)
                        except (ValueError, EditorError):
                            until = None
                            io.write("Choose a valid source token boundary.")
                            continue
                        break
                    if until is None:
                        continue
                elif len(replay_args) == 3 and replay_args[1] == "--until":
                    try:
                        until = int(replay_args[2])
                    except ValueError:
                        raise EditorError("Replay boundary must be an integer.") from None
                elif len(replay_args) != 1:
                    raise EditorError("Use spr EPISODE [--until Y | m].")
                source_id = recover_sampler_record(store, source_id, io)
                store.replay_until(source_id, until)
                return "spr", (source_id, until)
            except EditorError as exc:
                io.write(str(exc))
            continue
        io.write("Unknown live-edge command.")


def _seal(
    store: EpisodeStore,
    episode_id: str,
    engine: EpisodeEngine,
    *,
    reason: str | None = None,
    output: Path | None = None,
) -> None:
    if reason is not None and not engine.ended:
        engine.terminate(reason)
    store.finish_episode(
        episode_id,
        visible_text=engine.backend.render(engine.visible_token_ids),
        terminal_token_id=engine.terminal_token_id,
        terminal_reason=engine.terminal_reason,
        status="completed",
    )
    text = engine.text
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"Text: {output}", flush=True)
    else:
        print("\n--- final text ---")
        print(text)


def _print_list(store: EpisodeStore) -> None:
    print(store.workspace_list(include_finished=True))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    args._explicit_options = {
        action.dest for action in parser._actions
        if any(token.split("=", 1)[0] in action.option_strings for token in arguments)
    }
    try:
        if args.until is not None and args.replay is None:
            raise EditorError("--until requires --replay")
        if args.fixed_config and args.replay is None:
            raise EditorError("--fixed-config requires --replay")
        with EpisodeStore(args.workspace) as store, ExitStack() as ui_stack:
            for field in ("resume", "fork_from", "replay", "project", "lineage"):
                value = getattr(args, field)
                if value:
                    setattr(args, field, store.resolve_id(value))
            if args.at is not None and args.fork_from is None:
                raise EditorError("--at is only valid with --fork-from")
            if args.rules_only and not args.biases_only:
                raise EditorError("--rules-only requires --biases-only")
            if args.editor_friendly and not args.biases_only:
                raise EditorError("--editor-friendly requires --biases-only")
            if args.editor_friendly and args.rules_only:
                raise EditorError("--editor-friendly cannot be combined with --rules-only")
            if args.biases_only and (not args.project or args.procedure):
                raise EditorError("--biases-only requires --project and cannot be combined with --procedure")
            if args.procedure and not args.project:
                raise EditorError("--procedure requires --project EPISODE_ID")
            if args.lineage is not None:
                if not args.list_episodes:
                    raise EditorError("--lineage requires --list")
                print(project_lineage(store, args.lineage))
                return 0
            if args.list_episodes:
                _print_list(store)
                return 0
            if args.project:
                if args.biases_only:
                    print(project_biases(
                        store,
                        args.project,
                        rules_only=args.rules_only,
                        editor_friendly=args.editor_friendly,
                    ))
                    return 0
                if args.procedure:
                    print(project_procedure(store, args.project))
                    return 0
                print(
                    project_episode(
                        store,
                        args.project,
                        annotations=args.annotations,
                        with_loss=args.with_loss,
                        with_rank=args.with_rank,
                        with_policy_rank=args.with_policy_rank,
                        full_evidence=args.full_evidence,
                        with_model_probs=args.with_model_probs,
                        with_lineage=args.with_lineage,
                    ).text
                )
                return 0
            if not any(
                (
                    args.new_prompt is not None,
                    args.new_prompt_file is not None,
                    args.replay is not None,
                    args.resume is not None,
                    args.fork_from is not None,
                )
            ):
                if not sys.stdin.isatty() or not sys.stdout.isatty():
                    raise EditorError(
                        "no episode source supplied; use --new-prompt, "
                        "--new-prompt-file, --replay, --resume, or --fork-from"
                    )

                args.new_prompt = _read_initial_prompt()

            latent_config = _latent_config_from_args(args)
            if args.latent_random_seed:
                if args.replay is not None and not args.fixed_config:
                    raise EditorError("source-following replay restores saved latent seeds; --latent-random-seed requires --fixed-config")
                args.latent_seed = _random_seed()
                print(f"Random latent seed: {args.latent_seed}", flush=True)
            if args.latent_seed is not None:
                # Validate even if learning is disabled and before creating records.
                SamplingConfig(latent_projection_seed=args.latent_seed)

            if args.random_seed:
                args.seed = _random_seed()
                print(f"Random seed: {args.seed}", flush=True)

            io = TerminalIO(live_choices=not args.plain_ui, live_theme=args.theme)
            for field in ("resume", "fork_from", "replay"):
                if getattr(args, field):
                    setattr(args, field, recover_sampler_record(store, getattr(args, field), io))
            source_id = args.resume or args.fork_from or args.replay
            source = store.get_episode(source_id) if source_id else None
            backend, provenance, model_changed = _load_episode_backend(args, source, io)
            catalog = None
            if args.bias_catalog is not None:
                catalog = validate_catalog(
                    load_catalog(args.bias_catalog), backend, provenance
                )
            if args.groups is not None:
                supplied = compile_catalog(load_yaml_source(args.groups), backend)
                catalog = BiasCatalog.merge((catalog, supplied)) if catalog is not None else supplied
            if args.reference is not None:
                args._reference_routes = load_reference(args.reference, backend)
            if not math.isfinite(args.group_level) or args.group_level < 0:
                raise EditorError("group level must be finite and nonnegative")
            args._model_changed = model_changed
            if model_changed:
                args.bias_rules = ()
                args.bias_groups = ()
                io.write("Model changed: token-ID biases reset; load a matching preset to apply biases.")
            preset_latent = None
            if args.biases is not None:
                preset = load_bias_preset(args.biases, backend, provenance)
                args.bias_rules = preset.bias_rules
                args.bias_groups = preset.bias_groups
                preset_latent = preset
                args._bias_preset = preset
            requested_id = args.episode_id
            parent_id: str | None = None
            fork_boundary: int | None = None
            pending_tape: ReplayPlan | None = None
            learner = OnlineLearner(
                enabled=args.online_learning,
                learning_rate=args.learning_rate,
                epsilon=args.learning_epsilon,
                max_step=args.learning_max_step,
                min_bias=args.learning_min_bias,
                max_bias=args.learning_max_bias,
                learnable_groups=args.learnable_groups,
                severity_cap=args.learning_severity_cap, dead_zone_rank=args.learning_dead_zone_rank,
                no_severity_attenuation=args.learning_no_severity_attenuation,
                rejection_strength=args.learning_rejection_strength, decay=args.learning_decay,
            )

            if args.resume is not None:
                # Only explicit CLI sampler flags override the stored segment.
                explicit = any(getattr(args, name) is not None for name in SAMPLER_FIELDS)
                segment = store.sampling_segment(
                    args.resume, len(_visible_tokens(store, args.resume))
                )
                source_sampling = SamplingConfig.from_record(segment["sampling"])
                sampling = (
                    _sampling_from_args(args, source_sampling)
                    if explicit
                    else source_sampling
                )
                sampling = _apply_catalog_reference_prior(sampling, catalog, args)
                sampling = _apply_latent_preset(sampling, preset_latent)
                sampling = _apply_latent_seed(
                    sampling, args.latent_seed, io,
                    replay=args.replay is not None and not args.fixed_config,
                )
                explicit = sampling != source_sampling
                io.write("Restoring saved context...")
                if model_changed:
                    engine, episode_id = _model_continuation(store, args.resume, backend, provenance)
                    if args.max_tokens is not None:
                        engine.resume(max_tokens=args.max_tokens)
                    if explicit or args.reference_prior is not None or (
                        catalog is not None and catalog.reference_prior_routes
                    ):
                        engine.sampling = sampling
                        store.record_sampling_segment(episode_id, start_boundary=0, sampling=sampling,
                            stream_fingerprint=engine.stream_fingerprint, coordinate_offset=0)
                else:
                    engine = _restore_engine(
                        store, args.resume, backend, max_tokens=args.max_tokens,
                        sampling_override=sampling if explicit else None,
                    )
                    episode_id = args.resume
            elif args.new_prompt is not None or args.new_prompt_file is not None:
                initial_text = (
                    args.new_prompt
                    if args.new_prompt is not None
                    else args.new_prompt_file.read_text(encoding="utf-8")
                )
                sampling = _apply_catalog_reference_prior(
                    _sampling_from_args(args), catalog, args
                )
                sampling = _apply_latent_preset(sampling, preset_latent)
                sampling = _apply_latent_seed(
                    sampling, args.latent_seed, io,
                    replay=args.replay is not None and not args.fixed_config,
                )
                engine = EpisodeEngine(
                    backend,
                    sampling=sampling,
                    max_tokens=args.max_tokens,
                    initial_text=initial_text,
                )
                episode_id = _create_episode(
                    store,
                    engine,
                    backend_provenance=provenance,
                    requested_id=requested_id,
                )
            elif args.replay is not None:
                source_segment = store.sampling_segment(args.replay, 0)
                source_sampling = SamplingConfig.from_record(source_segment["sampling"])
                overrides = {
                    name: getattr(args, name) for name in SAMPLER_FIELDS
                    if getattr(args, name) is not None
                }
                sampling = _apply_catalog_reference_prior(
                    _sampling_from_args(args, source_sampling), catalog, args
                )
                sampling = _apply_latent_preset(sampling, preset_latent)
                # Explicit steering imports apply to every replay segment, just
                # like explicit sampler flags. Unspecified fields follow source.
                if args.bias_groups is not None:
                    overrides["group_controls"] = sampling.group_controls
                for name in POLICY_FIELDS:
                    reference_override = name.startswith("reference_prior_") and (
                        args.reference is not None or args.reference_prior is not None
                        or args.reference_prior_strength is not None or args.reference_strength is not None
                        or (catalog is not None and catalog.reference_prior_routes)
                    )
                    if preset_latent is not None or model_changed or reference_override:
                        overrides[name] = getattr(sampling, name)
                sampling = _apply_latent_seed(
                    sampling, args.latent_seed, io,
                    replay=args.replay is not None and not args.fixed_config,
                )
                replay_prefix = None
                if model_changed:
                    replay_prefix = backend.tokenize(store.get_episode(args.replay)["initial_text"], add_bos=True, special=True)
                engine, pending_tape = _spr_engine_from_source(
                    store,
                    args.replay,
                    backend,
                    sampling=sampling,
                    max_tokens=args.max_tokens,
                    until=args.until,
                    follow_source_sampling=not args.fixed_config,
                    sampling_overrides=overrides,
                    initial_token_ids=replay_prefix,
                    stream_fingerprint=source_segment["stream_fingerprint"] if model_changed else None,
                    coordinate_offset=source_segment["coordinate_offset"] if model_changed else None,
                )
                if args.latent_seed is not None and pending_tape.follow_source_sampling:
                    states = [step.sampling for step in pending_tape]
                    states.append(pending_tape.final_sampling)
                    if any(state is not None and state.latent_projection_seed != args.latent_seed
                           for state in states):
                        raise EditorError("explicit latent seed conflicts with a saved replay segment")
                episode_id = _create_episode(
                    store,
                    engine,
                    backend_provenance=provenance,
                    requested_id=requested_id,
                    parent_episode_id=args.replay,
                    fork_boundary=0,
                    mode="serial-policy-replay",
                    metadata={"spr_source": args.replay},
                )
            else:
                assert args.fork_from is not None
                source = store.get_episode(args.fork_from)
                visible = _visible_tokens(store, args.fork_from)
                target = len(visible) if args.at is None else args.at
                if not 0 <= target <= len(visible):
                    raise EditorError(f"fork boundary must be 0..{len(visible)}")
                prefix = [*source["initial_token_ids"], *visible[:target]]
                if model_changed:
                    retained = "".join(row["text"] for row in store.tokens(args.fork_from)
                                       if row["realized_visible"] and row["boundary"] < target)
                    prefix = backend.tokenize(source["initial_text"] + retained, add_bos=True, special=True)
                segment = store.sampling_segment(args.fork_from, target)
                source_sampling = SamplingConfig.from_record(segment["sampling"])
                explicit = any(getattr(args, name) is not None for name in SAMPLER_FIELDS)
                sampling = _apply_catalog_reference_prior(
                    _sampling_from_args(args, source_sampling), catalog, args
                )
                sampling = _apply_latent_preset(sampling, preset_latent)
                sampling = _apply_latent_seed(
                    sampling, args.latent_seed, io,
                    replay=args.replay is not None and not args.fixed_config,
                )
                engine = EpisodeEngine(
                    backend,
                    sampling=sampling,
                    max_tokens=source["max_tokens"] if args.max_tokens is None else args.max_tokens,
                    initial_text=backend.render(prefix, special=True),
                    initial_token_ids=prefix,
                    stream_fingerprint=segment["stream_fingerprint"],
                    coordinate_offset=segment["coordinate_offset"] + target,
                )
                if args.max_tokens is None:
                    _inherit_budget(store, args.fork_from, engine, target, rebase=True)
                episode_id = _create_episode(
                    store,
                    engine,
                    backend_provenance=provenance,
                    requested_id=requested_id,
                    parent_episode_id=args.fork_from,
                    fork_boundary=target,
                    mode="fork",
                )

            latent_learner = None
            if args.latent_preference:
                latent_learner = LatentPreferenceLearner(
                    feature_provider=lambda *, feature_dimension, projection_seed: backend.latent_token_features(
                        feature_dimension=feature_dimension,
                        projection_seed=projection_seed,
                        projection_chunk_size=args.latent_projection_chunk_size,
                    ),
                    config=replace(latent_config,
                                   projection_seed=engine.sampling.latent_projection_seed),
                )

            open_live_session = getattr(io, "live_session", None)
            if callable(open_live_session):
                ui_stack.enter_context(open_live_session())
            store.visit(episode_id)
            enter_edge = False
            while True:
                runner = EpisodeRunner(
                    engine,
                    store,
                    episode_id,
                    divergence_policy=args.divergence_policy,
                    learner=learner,
                    on_learning_update=lambda result: _online_learning_notice(
                        io, result
                    ),
                    latent_learner=latent_learner,
                    on_latent_learning_update=lambda result: _latent_preference_notice(
                        io, result
                    ),
                    learn_from_write=args.learn_from_write,
                    on_write_learning_update=lambda result: _write_learning_notice(
                        io, result
                    ),
                )
                try:
                    if enter_edge:
                        enter_edge = False
                        raise EdgeRequested()
                    result = runner.run(
                        tape=pending_tape,
                        live_policy=_interactive_policy(
                            args, store, episode_id, io, catalog=catalog
                        ),
                        stop_after_tape=True,
                    )
                except EdgeRequested:
                    pending_tape = None
                    action, value = _live_edge_menu(io, store, episode_id, engine)
                except SeamlessRewindRequested as request:
                    from_boundary = engine.boundary
                    io.write(f"Restoring context at boundary {request.boundary}...")
                    details = _rewind_episode(
                        store, episode_id, engine, request.boundary
                    )
                    store.record_interaction(
                        episode_id,
                        request.boundary,
                        "seamless-rewind",
                        {
                            "from_boundary": from_boundary,
                            "to_boundary": request.boundary,
                            "trimmed_action": details["trimmed_action"],
                        },
                    )
                    pending_tape = None
                    continue
                except SeamlessEdgeRequested as request:
                    from_boundary = engine.boundary
                    io.write(f"Restoring context at boundary {request.boundary}...")
                    details = _rewind_episode(
                        store, episode_id, engine, request.boundary
                    )
                    store.record_interaction(
                        episode_id,
                        request.boundary,
                        "seamless-edge-open",
                        {
                            "from_boundary": from_boundary,
                            "to_boundary": request.boundary,
                            "trimmed_action": details["trimmed_action"],
                        },
                    )
                    pending_tape = None
                    action, value = _live_edge_menu(io, store, episode_id, engine)
                except ForkRequested as request:
                    action, value = "fork", request.boundary
                else:
                    if engine.ended:
                        _seal(store, episode_id, engine, output=args.output)
                        return 0
                    if result.outcomes and result.outcomes[-1].stop_reason == "replay-eog":
                        io.write("Replay encountered EOG; tape stopped, live edge reached.")
                    elif result.handed_off:
                        if result.handoff_reason:
                            io.write(result.handoff_reason)
                        io.write(
                            f"Execution handed off at boundary {engine.boundary}; live edge reached."
                        )
                    elif result.replay_exhausted:
                        io.write(
                            f"SPR route exhausted at boundary {engine.boundary}; live edge reached."
                        )
                    elif engine.checkpointed:
                        io.write(f"Checkpoint reached at boundary {engine.boundary}.")
                    pending_tape = None
                    action, value = _live_edge_menu(io, store, episode_id, engine)

                if action == "switch":
                    destination = str(value)
                    target_episode = store.get_episode(destination)
                    sealed = target_episode["status"] in {"completed", "failed"}
                    if sealed:
                        io.page(project_episode(store, destination).text)
                        reply = io.read("Finished episode. Fork from end? [y/N]> ")
                        if not reply or reply.strip().lower() not in {"y", "yes"}:
                            enter_edge = True
                            continue
                    store.update_episode(episode_id, visible_text=engine.backend.render(engine.visible_token_ids),
                                         max_tokens=engine.max_tokens, status="open")
                    store.record_budget(episode_id, engine.boundary, engine.max_tokens, engine.checkpoint_boundary)
                    try:
                        destination = recover_sampler_record(store, destination, io)
                        target_episode = store.get_episode(destination)
                        new_backend, new_provenance, changed = _load_episode_backend(args, target_episode, io, use_saved=True,
                            current_backend=backend, current_provenance=provenance)
                        if changed:
                            new_engine, destination = _model_continuation(store, destination, new_backend, new_provenance)
                        elif sealed:
                            visible = _visible_tokens(store, destination)
                            segment = store.sampling_segment(destination, len(visible))
                            new_engine = EpisodeEngine(new_backend,
                                initial_token_ids=[*target_episode["initial_token_ids"], *visible],
                                sampling=SamplingConfig.from_record(segment["sampling"]),
                                stream_fingerprint=segment["stream_fingerprint"],
                                coordinate_offset=segment["coordinate_offset"] + len(visible),
                                max_tokens=target_episode["max_tokens"])
                            _inherit_budget(store, destination, new_engine, len(visible), rebase=True)
                            destination = _create_episode(store, new_engine, backend_provenance=new_provenance,
                                parent_episode_id=destination, fork_boundary=len(visible), mode="fork")
                        else:
                            new_engine = _restore_engine(store, destination, new_backend,
                                max_tokens=None, sampling_override=None)
                    except (EditorError, OSError, RuntimeError) as exc:
                        # A reused backend may already have been repositioned.
                        engine.backend.reset(engine.token_ids)
                        io.write(str(exc))
                        enter_edge = True
                        continue
                    engine, backend, provenance, episode_id = new_engine, new_backend, new_provenance, destination
                    store.visit(episode_id)
                    pending_tape = None
                    enter_edge = True
                    continue
                if action == "quit":
                    store.update_episode(
                        episode_id,
                        visible_text=engine.backend.render(engine.visible_token_ids),
                        max_tokens=engine.max_tokens,
                        status="open",
                    )
                    print(
                        f"Episode {store.label(episode_id)} remains unsealed. Resume with --resume '{store.label(episode_id).split()[0]}'",
                        flush=True,
                    )
                    return 0
                if action == "end":
                    _seal(store, episode_id, engine, reason="menu-end", output=args.output)
                    return 0
                if action == "continue":
                    checkpoint_transition = engine.checkpointed or value != "keep"
                    engine.resume(max_tokens=value)
                    store.record_budget(episode_id, engine.boundary, engine.max_tokens, engine.checkpoint_boundary)
                    store.record_interaction(
                        episode_id,
                        engine.boundary,
                        "checkpoint-resume" if checkpoint_transition else "edge-continue",
                        {"next_max_tokens": engine.max_tokens},
                    )
                    pending_tape = None
                    continue
                if action == "fork":
                    target = int(value)
                    parent_id = episode_id
                    parent_engine = engine
                    engine = _fork_engine(
                        store,
                        parent_id,
                        parent_engine,
                        target,
                        backend=backend,
                        max_tokens=None,
                    )
                    episode_id = _create_episode(
                        store,
                        engine,
                        backend_provenance=provenance,
                        parent_episode_id=parent_id,
                        fork_boundary=target,
                        mode="fork",
                    )
                    pending_tape = None
                    continue
                if action == "spr":
                    source_id, until = value
                    # Snapshot before appending, including self-replay. EDGE
                    # composition inserts the source prompt as literal text;
                    # CLI replay continues to use it as initial context.
                    source_prompt = store.get_episode(source_id)["initial_text"]
                    source_steps = store.replay_until(source_id, until)
                    steps = []
                    if source_prompt:
                        steps.append(TapeStep(
                            Write(source_prompt, "exact"), None,
                            source_episode_id=source_id, source_boundary=0,
                            source_part="prompt",
                        ))
                    steps.extend(
                        TapeStep(
                            step["action"], step["expectation"],
                            source_episode_id=source_id,
                            source_boundary=step["boundary"],
                        )
                        for step in source_steps
                    )
                    pending_tape = ReplayPlan(tuple(steps), follow_source_sampling=False)
                    store.record_interaction(
                        episode_id, engine.boundary, "replay-start",
                        {"source_episode_id": source_id, "action_count": len(steps), "until": until},
                    )
                    # Keep the existing engine and ledger: boundary zero,
                    # prefix, sampler stream, and remaining budget do not move.
                    continue
                raise AssertionError(f"unhandled live-edge action {action!r}")
    except (EditorError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

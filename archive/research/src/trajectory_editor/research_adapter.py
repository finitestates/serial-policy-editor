"""Research-side compatibility adapter for the core episode store.

The core store owns the replayable sampler projection.  Research extensions
may use the wider historical ``SamplingConfig`` while they are installed, but
they must enter and leave the core boundary through this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from .core.cli_config import (
    CORE_SAMPLER_FIELDS,
    apply_activation_artifact,
    sampler_override as core_sampler_override,
    sampler_overrides_present as core_sampler_overrides_present,
)
from .core.errors import EditorError
from .core.sampler_config import SamplerConfig
from .domain import SamplingConfig as ResearchSamplerConfig


TOKEN_PREFERENCE_OVERRIDE_FIELDS = (
    "token_preference_feature_scheme",
    "token_preference_whitening_ridge",
    "token_preference_learning_scheme",
    "token_preference_influence_mode",
    "token_preference_influence_kl",
    "token_preference_min_gain",
    "token_preference_max_gain",
    "token_preference_dimension",
    "token_preference_strength",
    "token_preference_fast_strength",
)

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


def research_sampler_from_args(
    args,
    source: ResearchSamplerConfig | None = None,
) -> ResearchSamplerConfig:
    """Construct the wider research sampler from core and research options."""

    base = source if source is not None else ResearchSamplerConfig()
    if getattr(args, "_model_changed", False):
        base = replace(
            base,
            bias_rules=(),
            bias_groups=(),
            group_controls=(),
            reference_prior_routes=(),
            token_preference_vector=(),
            token_preference_fast_vector=(),
            activation_vector=(),
            activation_vector_model="",
            activation_vector_digest="",
            activation_vector_strength=0.0,
            activation_vector_layer_start=None,
            activation_vector_layer_end=None,
        )
    values = {
        name: getattr(args, name)
        if getattr(args, name, None) is not None
        else getattr(base, name)
        for name in CORE_SAMPLER_FIELDS
    }
    values.update(
        group_controls=base.group_controls,
        token_preference_vector=base.token_preference_vector,
        token_preference_strength=(
            args.token_preference_strength
            if "token_preference_strength" in getattr(args, "_explicit_options", set())
            else base.token_preference_strength
        ),
        token_preference_fast_vector=base.token_preference_fast_vector,
        token_preference_fast_strength=(
            args.token_preference_fast_strength
            if getattr(args, "token_preference_fast_strength", None) is not None
            else base.token_preference_fast_strength
        ),
        token_preference_projection_seed=base.token_preference_projection_seed,
        token_preference_coordinate_identity=base.token_preference_coordinate_identity,
        activation_vector=base.activation_vector,
        activation_vector_strength=base.activation_vector_strength,
        activation_vector_layer=base.activation_vector_layer,
        activation_vector_position=base.activation_vector_position,
        activation_vector_layer_start=base.activation_vector_layer_start,
        activation_vector_layer_end=base.activation_vector_layer_end,
        activation_vector_model=base.activation_vector_model,
        activation_vector_digest=base.activation_vector_digest,
        token_preference_feature_scheme=(
            base.token_preference_feature_scheme
            if getattr(args, "token_preference_feature_scheme", None) is None
            else args.token_preference_feature_scheme
        ),
        token_preference_whitening_ridge=(
            base.token_preference_whitening_ridge
            if getattr(args, "token_preference_whitening_ridge", None) is None
            else args.token_preference_whitening_ridge
        ),
        token_preference_learning_scheme=(
            base.token_preference_learning_scheme
            if getattr(args, "token_preference_learning_scheme", None) is None
            else args.token_preference_learning_scheme
        ),
        token_preference_influence_mode=(
            base.token_preference_influence_mode
            if getattr(args, "token_preference_influence_mode", None) is None
            else args.token_preference_influence_mode
        ),
        token_preference_influence_kl=(
            base.token_preference_influence_kl
            if getattr(args, "token_preference_influence_kl", None) is None
            else args.token_preference_influence_kl
        ),
        token_preference_min_gain=(
            base.token_preference_min_gain
            if getattr(args, "token_preference_min_gain", None) is None
            else args.token_preference_min_gain
        ),
        token_preference_max_gain=(
            base.token_preference_max_gain
            if getattr(args, "token_preference_max_gain", None) is None
            else args.token_preference_max_gain
        ),
        group_control_scheme=(
            base.group_control_scheme
            if getattr(args, "group_control_scheme", None) is None
            else args.group_control_scheme
        ),
        reference_prior_routes=base.reference_prior_routes,
        reference_prior_scope=base.reference_prior_scope,
        reference_prior_mode=base.reference_prior_mode,
        reference_prior_strength=base.reference_prior_strength,
        reference_prior_attraction=base.reference_prior_attraction,
        reference_prior_exit_strength=base.reference_prior_exit_strength,
    )
    preset = getattr(args, "_bias_preset", None)
    if getattr(args, "bias_groups", None) is not None:
        values["group_controls"] = preset.group_controls if preset is not None else ()
    return ResearchSamplerConfig(**values)


def research_sampler_overrides_present(args) -> bool:
    explicit = getattr(args, "_explicit_options", set())
    preference = any(
        getattr(args, name, None) is not None
        and (name not in {"token_preference_dimension", "token_preference_strength"} or name in explicit)
        for name in TOKEN_PREFERENCE_OVERRIDE_FIELDS
    )
    return core_sampler_overrides_present(args) or preference


def validate_research_sampler_seed(seed: int) -> None:
    """Validate a research-only projection seed without exposing its record type."""

    ResearchSamplerConfig(token_preference_projection_seed=seed)


def apply_token_preference_preset(sampling, preset_preference):
    if preset_preference is None:
        return sampling
    return replace(
        sampling,
        group_controls=preset_preference.group_controls,
        token_preference_vector=preset_preference.token_preference_vector,
        token_preference_strength=preset_preference.token_preference_strength,
        token_preference_fast_vector=preset_preference.token_preference_fast_vector,
        token_preference_fast_strength=preset_preference.token_preference_fast_strength,
        token_preference_projection_seed=preset_preference.token_preference_projection_seed,
        token_preference_feature_scheme=preset_preference.token_preference_feature_scheme,
        token_preference_whitening_ridge=preset_preference.token_preference_whitening_ridge,
        token_preference_learning_scheme=preset_preference.token_preference_learning_scheme,
        token_preference_influence_mode=preset_preference.token_preference_influence_mode,
        token_preference_influence_kl=preset_preference.token_preference_influence_kl,
        token_preference_min_gain=preset_preference.token_preference_min_gain,
        token_preference_max_gain=preset_preference.token_preference_max_gain,
        token_preference_coordinate_identity=preset_preference.token_preference_coordinate_identity,
        activation_vector=preset_preference.activation_vector,
        activation_vector_strength=preset_preference.activation_vector_strength,
        activation_vector_layer=preset_preference.activation_vector_layer,
        activation_vector_position=preset_preference.activation_vector_position,
        activation_vector_layer_start=preset_preference.activation_vector_layer_start,
        activation_vector_layer_end=preset_preference.activation_vector_layer_end,
        activation_vector_model=preset_preference.activation_vector_model,
        activation_vector_digest=preset_preference.activation_vector_digest,
        group_control_scheme=preset_preference.group_control_scheme,
    )


def token_preference_config_from_args(args):
    from .token_preference import TokenPreferenceConfig

    return TokenPreferenceConfig(
        enabled=args.token_preference,
        dimension=args.token_preference_dimension,
        learning_rate=args.token_preference_learning_rate,
        token_preference_strength=args.token_preference_strength,
        max_step=args.token_preference_max_step,
        max_norm=args.token_preference_max_norm,
        decay=args.token_preference_decay,
        severity_cap=args.token_preference_severity_cap,
        no_severity_attenuation=args.token_preference_no_severity_attenuation,
        dead_zone_rank=args.token_preference_dead_zone_rank,
        rejection_strength=args.token_preference_rejection_strength,
        fast_slow=args.token_preference_fast_slow,
        fast_learning_rate=args.token_preference_fast_learning_rate,
        fast_decay=args.token_preference_fast_decay,
        fast_strength=args.token_preference_fast_strength,
        fast_max_step=args.token_preference_fast_max_step,
        fast_max_norm=args.token_preference_fast_max_norm,
        learning_gate=args.token_preference_learning_gate,
        decay_on=args.token_preference_decay_on,
        write_reduction=args.token_preference_write_reduction,
        rejection_target=args.token_preference_rejection_target,
        learning_scheme=getattr(args, "token_preference_learning_scheme", None) or "sgd-v1",
        learning_metric=getattr(args, "token_preference_learning_metric", None) or "euclidean",
        learning_kl=getattr(args, "token_preference_learning_kl", None) or 0.05,
        fast_learning_kl=getattr(args, "token_preference_fast_learning_kl", None),
        fisher_ridge=getattr(args, "token_preference_fisher_ridge", None) or 1.0e-3,
        fisher_mode=getattr(args, "token_preference_fisher_mode", None) or "diagonal",
        fisher_mass=getattr(args, "token_preference_fisher_mass", None) or 0.999,
        fisher_max_support=getattr(args, "token_preference_fisher_max_support", None) or 2048,
    )


def apply_token_preference_seed(sampling, seed, io, *, replay=False):
    if seed is None or seed == sampling.token_preference_projection_seed:
        return sampling
    if replay:
        raise EditorError("explicit token preference projection seed conflicts with saved replay seed")
    if sampling.token_preference_vector or sampling.token_preference_fast_vector:
        io.write("token preference projection coordinate system changed: token preference memory reset (slow and fast).")
    return replace(
        sampling,
        token_preference_projection_seed=seed,
        token_preference_vector=(),
        token_preference_fast_vector=(),
        token_preference_coordinate_identity=None,
    )


def apply_token_preference_coordinate_overrides(sampling, args, io, *, replay=False):
    explicit = getattr(args, "_explicit_options", set())
    requested_dimension = (
        args.token_preference_dimension
        if "token_preference_dimension" in explicit
        else None
    )
    if not (sampling.token_preference_vector or sampling.token_preference_fast_vector):
        return sampling
    from .token_preference_features import coordinate_identity_matches

    if coordinate_identity_matches(sampling, dimension=requested_dimension):
        return sampling
    if replay:
        raise EditorError("explicit token preference coordinate override conflicts with saved replay")
    io.write("token preference coordinate system changed: token preference memory reset (slow and fast).")
    return replace(
        sampling,
        token_preference_vector=(),
        token_preference_fast_vector=(),
        token_preference_coordinate_identity=None,
    )


def apply_catalog_reference_prior(sampling, catalog, args):
    direct = getattr(args, "_reference_routes", None)
    strength = getattr(args, "reference_strength", None)
    preset = getattr(args, "_bias_preset", None)
    requested = args.reference_prior
    if direct is not None:
        if requested not in (None, "off"):
            raise EditorError("--reference uses one standalone lexical policy; omit legacy --reference-prior modes")
        return replace(
            sampling,
            reference_prior_routes=() if requested == "off" else direct,
            reference_prior_scope="global",
            reference_prior_mode="lexical",
            reference_prior_strength=0.25 if strength is None else strength,
            reference_prior_attraction=0.0,
            reference_prior_exit_strength=0.0,
        )
    if requested is None:
        if catalog is not None and catalog.reference_prior_routes:
            return replace(
                sampling,
                reference_prior_routes=tuple(
                    (route.token_ids, route.weight)
                    for route in catalog.reference_prior_routes
                ),
                reference_prior_scope="global",
                reference_prior_mode="lexical",
                reference_prior_strength=0.25 if strength is None else strength,
                reference_prior_attraction=0.0,
                reference_prior_exit_strength=0.0,
            )
        if preset is not None:
            sampling = replace(
                sampling,
                **{
                    key: value
                    for key, value in preset.__dict__.items()
                    if key.startswith("reference_prior_")
                },
            )
        return sampling if strength is None else replace(sampling, reference_prior_strength=strength)
    if requested == "off":
        return replace(
            sampling,
            reference_prior_routes=(),
            reference_prior_scope="active",
            reference_prior_mode="contrastive",
        )
    if catalog is None or not catalog.reference_prior_routes:
        raise EditorError("--reference-prior requires --bias-catalog compiled with --reference")
    return replace(
        sampling,
        reference_prior_routes=tuple(
            (route.token_ids, route.weight) for route in catalog.reference_prior_routes
        ),
        reference_prior_scope=REFERENCE_PRIOR_PRESETS[requested][0],
        reference_prior_mode=REFERENCE_PRIOR_PRESETS[requested][1],
        reference_prior_strength=(
            sampling.reference_prior_strength
            if args.reference_prior_strength is None
            else args.reference_prior_strength
        ),
        reference_prior_attraction=(
            sampling.reference_prior_attraction
            if args.reference_prior_attraction is None
            else args.reference_prior_attraction
        ),
        reference_prior_exit_strength=(
            sampling.reference_prior_exit_strength
            if args.reference_prior_exit_strength is None
            else args.reference_prior_exit_strength
        ),
    )


def research_sampler_override(current, raw, *, seed_factory=None):
    return core_sampler_override(
        current,
        raw,
        **({} if seed_factory is None else {"seed_factory": seed_factory}),
    )


@dataclass(frozen=True)
class ResearchSurface:
    """Research implementations supplied to the otherwise core CLI.

    Keeping these imports behind one loader prevents the core command from
    knowing which learning, catalog, or controller modules happen to exist.
    The surface is deliberately a bundle of implementations rather than a
    second runner: replay and persistence remain owned by core.
    """

    research_sampler_config: type[ResearchSamplerConfig]
    load_bias_preset: Any
    project_biases: Any
    BiasCatalog: type
    compile_catalog: Any
    load_catalog: Any
    load_yaml_source: Any
    validate_catalog: Any
    load_reference: Any
    TokenPreferenceLearner: type
    OnlineLearner: type
    episode_runner: type
    ControllerPipeline: type
    research_sampler_from_args: Any
    research_sampler_overrides_present: Any
    validate_research_sampler_seed: Any
    apply_token_preference_preset: Any
    apply_token_preference_seed: Any
    apply_token_preference_coordinate_overrides: Any
    apply_catalog_reference_prior: Any
    apply_activation_artifact: Any
    token_preference_config_from_args: Any
    sampler_override: Any


def load_surface() -> ResearchSurface:
    """Load the optional research implementations on explicit request."""

    from .bias_presets import load_bias_preset, project_biases
    from .bias_catalog import (
        BiasCatalog,
        compile_catalog,
        load_catalog,
        load_yaml_source,
        validate_catalog,
    )
    from .lexical_reference import load_reference
    from .token_preference import TokenPreferenceLearner
    from .online_learning import OnlineLearner
    from .episode_policy import EpisodeRunner
    from .controller_pipeline import ControllerPipeline

    return ResearchSurface(
        research_sampler_config=ResearchSamplerConfig,
        load_bias_preset=load_bias_preset,
        project_biases=project_biases,
        BiasCatalog=BiasCatalog,
        compile_catalog=compile_catalog,
        load_catalog=load_catalog,
        load_yaml_source=load_yaml_source,
        validate_catalog=validate_catalog,
        load_reference=load_reference,
        TokenPreferenceLearner=TokenPreferenceLearner,
        OnlineLearner=OnlineLearner,
        episode_runner=EpisodeRunner,
        ControllerPipeline=ControllerPipeline,
        research_sampler_from_args=research_sampler_from_args,
        research_sampler_overrides_present=research_sampler_overrides_present,
        validate_research_sampler_seed=validate_research_sampler_seed,
        apply_token_preference_preset=apply_token_preference_preset,
        apply_token_preference_seed=apply_token_preference_seed,
        apply_token_preference_coordinate_overrides=apply_token_preference_coordinate_overrides,
        apply_catalog_reference_prior=apply_catalog_reference_prior,
        apply_activation_artifact=apply_activation_artifact,
        token_preference_config_from_args=token_preference_config_from_args,
        sampler_override=research_sampler_override,
    )


def sampling_from_record(record: Mapping[str, Any]) -> ResearchSamplerConfig:
    """Reconstruct the wider research sampler from a raw saved mapping."""

    return ResearchSamplerConfig.from_record(record)


def sampling_at(store, episode_id: str, boundary: int = 0) -> ResearchSamplerConfig:
    """Load the research sampler at a token boundary."""

    return sampling_from_record(store.sampling_segment(episode_id, boundary)["sampling"])


def final_sampling(store, episode_id: str) -> ResearchSamplerConfig:
    """Load the latest research sampler, including opaque adapter fields."""

    return sampling_from_record(store.final_sampling_record(episode_id))


def core_sampling(sampling: ResearchSamplerConfig) -> SamplerConfig:
    """Project a research sampler before handing it to core-only code."""

    return SamplerConfig.from_record(sampling.to_dict())


__all__ = [
    "ResearchSurface",
    "ResearchSamplerConfig",
    "core_sampling",
    "final_sampling",
    "apply_activation_artifact",
    "apply_catalog_reference_prior",
    "apply_token_preference_coordinate_overrides",
    "apply_token_preference_preset",
    "apply_token_preference_seed",
    "load_surface",
    "research_sampler_from_args",
    "sampling_at",
    "sampling_from_record",
    "research_sampler_overrides_present",
    "validate_research_sampler_seed",
    "token_preference_config_from_args",
]

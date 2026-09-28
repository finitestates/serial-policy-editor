"""Render compact sampler state for EDGE status surfaces."""

from __future__ import annotations

from .core.sampler_config import SamplerConfig


def sampler_summary(config: SamplerConfig) -> str:
    """Describe the active sampler without exposing backend implementation."""

    summary = (
        f"temp={config.temperature:g} top_k={'none' if config.top_k is None else config.top_k} "
        f"top_p={config.top_p:g} "
        f"min_p={config.min_p:g} typical_p={config.typical_p:g} "
        f"tfs_z={config.tail_free_z:g} draw={config.draw_kernel} "
        f"rep={config.repeat_penalty:g}/{config.repeat_last_n} "
        f"presence={config.presence_penalty:g} "
        f"frequency={config.frequency_penalty:g} seed={config.seed}"
    )
    if config.gumbel_top_k is not None:
        summary += f" gumbel_top_k={config.gumbel_top_k}"
    if config.draw_kernel == "gaussian-max":
        summary += f" gaussian_noise_std={config.gaussian_noise_std:g}"
    if config.bias_groups:
        summary += " groups=" + ",".join(
            f"{group.name}:{group.bias:g}" for group in config.bias_groups
        )
    if config.token_biases:
        summary += " token_biases=" + ",".join(
            f"#{item.token_id}:{item.bias:g}" for item in config.token_biases
        )
    if config.activation_vector or config.activation_vector_digest:
        norm = sum(value * value for value in config.activation_vector) ** 0.5
        summary += (
            f" steering_vector_norm={norm:g}"
            f" steering_strength={config.activation_vector_strength:g}"
            f" steering_digest={config.activation_vector_digest[:12]}"
        )
    return summary


__all__ = ["sampler_summary"]

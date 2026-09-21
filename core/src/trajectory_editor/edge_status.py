"""Render compact sampler state for EDGE status surfaces."""

from __future__ import annotations

from .core.sampler_config import SamplerConfig


def sampler_summary(config: SamplerConfig) -> str:
    """Describe the active sampler without exposing backend implementation."""

    summary = (
        f"temp={config.temperature:g} top_k={config.top_k} top_p={config.top_p:g} "
        f"min_p={config.min_p:g} typical_p={config.typical_p:g} "
        f"tfs_z={config.tail_free_z:g} draw={config.draw_kernel} "
        f"rep={config.repeat_penalty:g}/{config.repeat_last_n} "
        f"presence={config.presence_penalty:g} "
        f"frequency={config.frequency_penalty:g} seed={config.seed}"
    )
    if config.bias_groups:
        summary += " groups=" + ",".join(
            f"{group.name}:{group.bias:g}" for group in config.bias_groups
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

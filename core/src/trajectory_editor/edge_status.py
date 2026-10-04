"""Render compact sampler state for EDGE status surfaces."""

from __future__ import annotations

import math

from .core.sampler_config import SamplerConfig


def sampler_summary(config: SamplerConfig) -> str:
    """Describe the active sampler without exposing backend implementation."""

    summary = (
        f"temp={config.temperature:g} top_k={'none' if config.top_k is None else config.top_k} "
        f"min_gap={-math.log(config.min_p) if config.min_p > 0 else 'off'} draw={config.draw_kernel} "
        f"selective_noise_k={'none' if config.selective_noise_k is None else config.selective_noise_k} "
        f"rep={config.repeat_penalty:g}/{config.repeat_last_n} "
        f"presence={config.presence_penalty:g} "
        f"frequency={config.frequency_penalty:g} seed={config.seed}"
    )
    if config.gumbel_top_k is not None:
        summary += f" gumbel_top_k={config.gumbel_top_k}"
    if (
        config.draw_kernel == "gumbel-max"
        and config.gumbel_noise_address != "token-id"
    ):
        summary += f" gumbel_noise_address={config.gumbel_noise_address}"
    if config.draw_kernel == "gumbel-max" and config.gumbel_noise_scale != 1.0:
        summary += f" gumbel_noise_scale={config.gumbel_noise_scale:g}"
    if config.draw_kernel == "gaussian-max":
        summary += f" gaussian_noise_std={config.gaussian_noise_std:g}"
    if config.draw_kernel in {
        "logistic-max", "student-t-max", "laplace-max", "uniform-max"
    }:
        summary += f" perturb_noise_std={config.perturb_noise_std:g}"
    if config.draw_kernel == "student-t-max":
        summary += f" student_t_df={config.student_t_df:g}"
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

#!/usr/bin/env python3
"""Compare the Rust policy surface using live logits from local real models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "core/src"), str(ROOT)]

from benchmarks.real_model import load_profile  # noqa: E402
from trajectory_editor.bias_groups import BiasGroup, BiasMember, BiasRoute, BiasToken  # noqa: E402
from trajectory_editor.core.policy_calculations import PolicyCalculations  # noqa: E402
from trajectory_editor.core.sampler_config import SamplerConfig  # noqa: E402
from trajectory_editor.core.sampling import apply_candidate_filter  # noqa: E402
from trajectory_editor.episode_backend_loader import load_backend  # noqa: E402
import rust_sampler  # noqa: E402


PROMPT = "A short list of everyday objects:"
FLOAT_TOLERANCE = 1e-14


def _close(actual, expected, label: str) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=0.0,
        atol=FLOAT_TOLERANCE,
        err_msg=label,
    )


def _check_boundary(backend, guidance_backend, prefix, visible, profile) -> dict:
    conditional = np.asarray(backend.last_logits(), dtype=np.float64)
    unconditional = np.asarray(guidance_backend.last_logits(), dtype=np.float64)
    if conditional.shape != unconditional.shape:
        raise AssertionError("real CFG backend vocabularies differ")
    scale = float(profile["options"].cfg_scale)
    cfg_reference = unconditional + scale * (conditional - unconditional)
    cfg_rust = rust_sampler.cfg_combine_logits(conditional, unconditional, scale)
    _close(cfg_rust, cfg_reference, "CFG combination")

    width = int(backend.activation_width())
    direction = np.linspace(-0.001, 0.001, width, dtype=np.float32)
    activation = np.asarray(
        backend.activation_logit_adjustments(
            direction, layer="output", position="current"
        ),
        dtype=np.float64,
    )
    if activation.shape != cfg_reference.shape:
        raise AssertionError("real output-head projection has the wrong vocabulary")

    vocab = len(cfg_reference)
    target = int(np.argmax(cfg_reference))
    direct_target = (target + 1) % vocab
    ephemeral_target = (target + 2) % vocab
    route_tokens = (int(prefix[-1]), target) if prefix else (target,)
    config = SamplerConfig(
        temperature=0.85,
        top_k=12,
        top_p=1.0,
        min_p=0.0,
        typical_p=1.0,
        tail_free_z=1.0,
        cfg_unconditional_prompt=profile["options"].cfg_unconditional_prompt,
        cfg_scale=scale,
        cfg_prefix_tokens=0,
        repeat_penalty=1.1,
        repeat_last_n=8,
        presence_penalty=0.05,
        frequency_penalty=0.005,
        activation_vector=(0.1,),
        activation_vector_strength=0.2,
        bias_groups=(
            BiasGroup(
                "real_model_phrase",
                (
                    BiasMember(
                        "chosen route",
                        (BiasRoute(route_tokens, ("chosen route",)),),
                    ),
                ),
                0.125,
            ),
        ),
        token_biases=(BiasToken(direct_target, -0.05),),
    )
    ephemeral = {ephemeral_target: 0.075}
    python_policy = PolicyCalculations(
        cfg_reference,
        config,
        prefix,
        activation_logit_adjustments=activation,
        ephemeral_logit_biases=ephemeral,
    )
    rust_adjusted = rust_sampler.apply_history_penalties(cfg_reference, prefix, config)
    rust_adjusted = rust_sampler.apply_activation_adjustments(
        rust_adjusted, activation, config.activation_vector_strength
    )
    rust_adjusted = rust_sampler.apply_biases(rust_adjusted, prefix, config)
    rust_adjusted = rust_sampler.apply_ephemeral_biases(rust_adjusted, ephemeral)
    _close(rust_adjusted, python_policy.adjusted, "ordered real-model adjustments")

    python_filter = apply_candidate_filter(python_policy.adjusted, config)
    rust_filter = rust_sampler.apply_candidate_filter(rust_adjusted, config)
    _close(rust_filter.scaled_logits, python_filter.scaled_logits, "real-model filter logits")
    for name, expected in python_filter.stages.items():
        actual = rust_filter.stages[name]
        if (actual is None) != (expected is None):
            raise AssertionError(f"real-model filter stage {name} availability differs")
        if actual is not None and not np.array_equal(actual, expected):
            raise AssertionError(f"real-model filter stage {name} IDs differ")

    raw_ids = np.argsort(-cfg_reference, kind="stable")[:12].astype(int).tolist()
    selected = sorted({target, direct_target, ephemeral_target})
    metrics = rust_sampler.PolicyMetrics(cfg_reference, rust_adjusted)
    if metrics.raw_logsumexp_ready or metrics.logit_mean_std_ready:
        raise AssertionError("real-model metrics were calculated eagerly")
    if metrics.top_raw_ids(12) != raw_ids:
        raise AssertionError("real-model raw top IDs differ")
    if [metrics.raw_rank(token) for token in raw_ids] != [
        python_policy.raw_rank(token) for token in raw_ids
    ]:
        raise AssertionError("real-model raw ranks differ")
    if [metrics.policy_rank(token) for token in raw_ids] != [
        python_policy.policy_rank(token) for token in raw_ids
    ]:
        raise AssertionError("real-model policy ranks differ")
    actual_z = metrics.logit_z_scores(selected)
    expected_z = python_policy.logit_z_scores(selected)
    _close(actual_z, expected_z, "real-model raw logit z scores")
    if metrics.raw_logsumexp_ready:
        raise AssertionError("real-model z-score calculation forced soft-max normalization")
    _close(
        metrics.raw_probabilities(selected),
        python_policy.raw_probabilities(selected),
        "real-model raw probabilities",
    )
    _close(
        metrics.policy_probabilities_at(selected),
        python_policy.policy_probabilities_at(selected),
        "real-model policy probabilities",
    )
    _close(
        [metrics.raw_nll(token) for token in selected],
        [python_policy.raw_nll(token) for token in selected],
        "real-model raw NLL",
    )

    # The next exact continuation token is shared by both model branches, as in
    # EpisodeEngine's CFG path. The next call observes boundary 1 incrementally.
    continuation_token = int(python_policy.distribution.ids[0])
    backend.eval([continuation_token])
    guidance_backend.eval([continuation_token])
    return {
        "boundary": len(visible),
        "vocabulary_size": vocab,
        "cfg_scale": scale,
        "activation_width": width,
        "projected_adjustments": len(activation),
        "max_abs_cfg_delta": float(np.max(np.abs(cfg_rust - cfg_reference))),
        "max_abs_policy_delta": float(np.max(np.abs(rust_adjusted - python_policy.adjusted))),
        "candidate_ids": [int(value) for value in rust_filter.stages["after_min_p"]],
        "raw_top_ids": raw_ids,
        "policy_top_ids": metrics.top_policy_ids(12),
        "continuation_token_id": continuation_token,
        "z_score_ids": selected,
    }


def run_profile(profile_path: Path, model_root: Path) -> dict:
    profile = load_profile(profile_path, model_root, None, None, None, None)
    backend = guidance_backend = None
    try:
        backend = load_backend(profile["options"])
        guidance_backend = load_backend(profile["options"])
        root = [
            int(token)
            for token in backend.tokenize(PROMPT, add_bos=True, special=True)
        ]
        if not root:
            raise AssertionError("real-model prompt produced no token IDs")
        guidance_prompt = [
            int(token)
            for token in guidance_backend.tokenize(
                profile["options"].cfg_unconditional_prompt,
                add_bos=True,
                special=True,
            )
        ]
        if not guidance_prompt:
            raise AssertionError("real-model CFG prompt produced no token IDs")
        backend.reset(root)
        guidance_backend.reset(guidance_prompt + root)
        visible = []
        boundaries = []
        for _ in range(2):
            prefix = [*root, *visible]
            boundaries.append(
                _check_boundary(backend, guidance_backend, prefix, visible, profile)
            )
            visible.append(boundaries[-1]["continuation_token_id"])
        provenance = dict(backend.provenance(include_model_sha256=True))
        return {
            "profile": profile["name"],
            "backend": profile["backend"],
            "model_path": str(profile["model_path"]),
            "tokenizer_id": str(backend.tokenizer_id()),
            "root_token_count": len(root),
            "cfg_prompt_token_count": len(guidance_prompt),
            "model_identity": provenance,
            "boundaries": boundaries,
            "status": "passed",
        }
    finally:
        for value in (guidance_backend, backend):
            close = getattr(value, "close", None)
            if callable(close):
                close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = {
        "format": "spe-rust-policy-real-model-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "float_tolerance": FLOAT_TOLERANCE,
        "results": [],
    }
    try:
        for profile_path in args.profile:
            report["results"].append(run_profile(profile_path, args.model_root))
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

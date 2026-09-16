"""Offline workbench for portable token-preference vector artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .backend_factory import BACKEND_NAMES, create_backend
from .bias_presets import FORMAT as BIAS_FORMAT
from .decoder import LlamaCppSettings
from .domain import EditorError
from .episode_store import EpisodeStore
from .token_preference_features import DEFAULT_PROJECTION_CHUNK_SIZE
from .transformers_backend import TransformersSettings
from .vector_artifacts import (
    FORMAT,
    TokenPreferenceVectorArtifact,
    assert_compatible,
    blend_artifacts,
    materialize_features,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _add_backend_args(parser: argparse.ArgumentParser, *, require_model: bool = False) -> None:
    parser.add_argument("--model", type=Path, required=require_model)
    parser.add_argument("--backend", choices=BACKEND_NAMES, default="llama.cpp")
    parser.add_argument("--cache", choices=("auto", "off"), default="auto")
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-threads", type=int)
    parser.add_argument("--n-gpu-layers", type=int)
    parser.add_argument("--transformers-device", default="auto")
    parser.add_argument(
        "--projection-chunk-size",
        type=_positive_int,
        default=DEFAULT_PROJECTION_CHUNK_SIZE,
        help="rows projected at once when materializing model token features",
    )


def _add_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="policy-editor-vector",
        description="Inspect and manage offline token-preference vector artifacts.",
    )
    commands = parser.add_subparsers(dest="kind", required=True)
    token_preference = commands.add_parser(
        "token-preference",
        help="work with vectors over fixed model token features",
    )
    actions = token_preference.add_subparsers(dest="action", required=True)

    extract = actions.add_parser("extract", help="extract a vector from an episode or v4 bias preset")
    source = extract.add_mutually_exclusive_group(required=True)
    source.add_argument("--workspace", type=Path)
    source.add_argument("--preset", type=Path)
    extract.add_argument("--episode", metavar="EPISODE_ID")
    extract.add_argument("--output", type=Path)

    inspect = actions.add_parser("inspect", help="show artifact metadata and vector norms")
    inspect.add_argument("artifact", type=Path)
    _add_report_args(inspect)

    validate = actions.add_parser("validate", help="validate an artifact, optionally against a model")
    validate.add_argument("artifact", type=Path)
    _add_backend_args(validate)
    _add_report_args(validate)

    explain = actions.add_parser(
        "explain",
        help="rank vocabulary tokens by their vector-induced logit adjustment",
    )
    explain.add_argument("artifact", type=Path)
    _add_backend_args(explain, require_model=True)
    explain.add_argument("--top", type=_positive_int, default=20)
    explain.add_argument("--include-eog", action="store_true")
    _add_report_args(explain)

    blend = actions.add_parser("blend", help="combine compatible artifacts by effective actuation")
    blend.add_argument("artifacts", type=Path, nargs="+")
    blend.add_argument("--weights", type=float, nargs="+")
    blend.add_argument("--output", type=Path)

    apply = actions.add_parser("apply", help="apply a standalone vector to a v4 bias preset")
    apply.add_argument("artifact", type=Path)
    apply.add_argument("--biases", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    return parser


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EditorError(f"could not read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise EditorError(f"{label} must contain a JSON object")
    return value


def _write_text(text: str, output: Path | None) -> None:
    if output is None:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return
    try:
        output.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    except OSError as exc:
        raise EditorError(f"could not write output: {exc}") from exc


def _load_backend(args: argparse.Namespace):
    if args.model is None:
        raise EditorError("--model is required for this operation")
    try:
        return create_backend(
            args.backend,
            args.model,
            llama_settings=LlamaCppSettings(
                n_ctx=args.n_ctx,
                n_threads=args.n_threads,
                n_gpu_layers=args.n_gpu_layers,
            ),
            transformers_settings=TransformersSettings(device=args.transformers_device),
            cache_mode=args.cache,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise EditorError(str(exc)) from exc


def _close_backend(backend: Any) -> None:
    for owner in (backend, getattr(backend, "_model", None)):
        close = getattr(owner, "close", None)
        if callable(close):
            close()
            return


def _extract(args: argparse.Namespace) -> TokenPreferenceVectorArtifact:
    if args.workspace is not None:
        if args.episode is None:
            raise EditorError("--episode is required with --workspace")
        with EpisodeStore(args.workspace) as store:
            episode_id = store.resolve_id(args.episode)
            episode = store.get_episode(episode_id)
            sampling = store.final_sampling(episode_id)
        return TokenPreferenceVectorArtifact.from_sampling(
            sampling,
            episode["backend"],
            source={
                "type": "episode",
                "workspace": str(args.workspace),
                "episode_id": episode_id,
            },
        )
    if args.episode is not None:
        raise EditorError("--episode requires --workspace")
    value = _load_json(args.preset, "bias preset")
    return TokenPreferenceVectorArtifact.from_preset_mapping(
        value,
        source={"type": "bias-preset", "path": str(args.preset)},
    )


def _norm(vector: tuple[float, ...]) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float64))) if vector else 0.0


def _inspect_report(artifact: TokenPreferenceVectorArtifact) -> dict[str, Any]:
    identity = artifact.coordinate_identity
    return {
        "format": FORMAT,
        "kind": "token-preference",
        "model": dict(artifact.model),
        "coordinate_identity": identity.to_dict() if identity is not None else None,
        "dimension": artifact.dimension,
        "slow_dimension": len(artifact.token_preference_vector),
        "fast_dimension": len(artifact.token_preference_fast_vector),
        "slow_norm": _norm(artifact.token_preference_vector),
        "fast_norm": _norm(artifact.token_preference_fast_vector),
        "token_preference_strength": artifact.token_preference_strength,
        "token_preference_fast_strength": artifact.token_preference_fast_strength,
        "source": dict(artifact.source) if artifact.source is not None else None,
    }


def _text_inspect(report: dict[str, Any]) -> str:
    lines = ([f"valid: {report['valid']}"] if "valid" in report else []) + [
        f"format: {report['format']}",
        f"kind: {report['kind']}",
        f"dimension: {report['dimension']}",
        f"slow: dimension={report['slow_dimension']} norm={report['slow_norm']:.6g} strength={report['token_preference_strength']:.6g}",
        f"fast: dimension={report['fast_dimension']} norm={report['fast_norm']:.6g} strength={report['token_preference_fast_strength']:.6g}",
    ]
    identity = report["coordinate_identity"]
    if identity is None:
        lines.append("coordinates: none")
    else:
        lines.append(
            "coordinates: "
            f"seed={identity['projection_seed']} scheme={identity['feature_scheme']} "
            f"whitening_ridge={identity['whitening_ridge']}"
        )
    model = report["model"]
    if model:
        lines.append("model: " + " ".join(f"{key}={value}" for key, value in sorted(model.items())))
    if report["source"] is not None:
        lines.append("source: " + json.dumps(report["source"], ensure_ascii=False, sort_keys=True))
    return "\n".join(lines)


def _explain_report(
    artifact: TokenPreferenceVectorArtifact,
    backend: Any,
    provenance: dict[str, Any],
    *,
    top: int,
    include_eog: bool,
    projection_chunk_size: int,
) -> dict[str, Any]:
    features = artifact.validate_against_backend(
        backend,
        provenance,
        projection_chunk_size=projection_chunk_size,
    )
    if features is None or artifact.dimension is None:
        raise EditorError("cannot explain an empty token preference artifact")
    slow_scores = (
        features @ np.asarray(artifact.token_preference_vector, dtype=np.float32)
        if artifact.token_preference_vector
        else np.zeros(features.shape[0], dtype=np.float32)
    )
    fast_scores = (
        features @ np.asarray(artifact.token_preference_fast_vector, dtype=np.float32)
        if artifact.token_preference_fast_vector
        else np.zeros(features.shape[0], dtype=np.float32)
    )
    combined_scores = (
        float(artifact.token_preference_strength) * np.asarray(slow_scores, dtype=np.float64)
        + float(artifact.token_preference_fast_strength) * np.asarray(fast_scores, dtype=np.float64)
    )
    eog_ids = set(backend.eog_token_ids())
    token_ids = [
        token_id
        for token_id in range(backend.vocabulary_size())
        if include_eog or token_id not in eog_ids
    ]
    descending = sorted(token_ids, key=lambda token_id: (-float(combined_scores[token_id]), token_id))
    ascending = sorted(token_ids, key=lambda token_id: (float(combined_scores[token_id]), token_id))

    def rows(order: list[int], *, positive: bool) -> list[dict[str, Any]]:
        result = []
        for token_id in order:
            score = float(combined_scores[token_id])
            if (positive and score <= 0.0) or (not positive and score >= 0.0):
                continue
            try:
                text = backend.token_text(token_id)
            except (RuntimeError, TypeError, ValueError):
                text = ""
            result.append({
                "token_id": token_id,
                "text": text,
                "score": score,
                "slow_score": float(slow_scores[token_id]),
                "fast_score": float(fast_scores[token_id]),
            })
            if len(result) >= top:
                break
        return result

    return {
        **_inspect_report(artifact),
        "eligible_token_count": len(token_ids),
        "score_rms": float(np.sqrt(np.mean(combined_scores[token_ids] ** 2))) if token_ids else 0.0,
        "top_positive": rows(descending, positive=True),
        "top_negative": rows(ascending, positive=False),
    }


def _text_explain(report: dict[str, Any]) -> str:
    lines = [
        f"token preference explanation: dimension={report['dimension']} eligible_tokens={report['eligible_token_count']}",
        f"score_rms={report['score_rms']:.6g}",
    ]
    identity = report["coordinate_identity"]
    if identity is not None:
        lines.append(
            f"coordinates: seed={identity['projection_seed']} scheme={identity['feature_scheme']}"
        )
    for title, key in (("top positive", "top_positive"), ("top negative", "top_negative")):
        lines.append(f"{title}:")
        if not report[key]:
            lines.append("  (none)")
        for row in report[key]:
            lines.append(
                f"  {row['token_id']:>8} {row['score']:>+12.6g} "
                f"slow={row['slow_score']:>+10.6g} fast={row['fast_score']:>+10.6g} "
                f"{row['text']!r}"
            )
    return "\n".join(lines)


def _render(report: dict[str, Any], output: Path | None, format_name: str, text_renderer) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) if format_name == "json" else text_renderer(report)
    _write_text(text, output)


def _apply(artifact: TokenPreferenceVectorArtifact, biases: Path, output: Path) -> None:
    value = _load_json(biases, "bias preset")
    if value.get("format") != BIAS_FORMAT:
        raise EditorError(f"bias preset must use format {BIAS_FORMAT}")
    current = TokenPreferenceVectorArtifact.from_preset_mapping(value)
    assert_compatible(artifact, current)
    result = dict(value)
    result.update({
        "token_preference_vector": list(artifact.token_preference_vector),
        "token_preference_fast_vector": list(artifact.token_preference_fast_vector),
        "token_preference_strength": artifact.token_preference_strength,
        "token_preference_fast_strength": artifact.token_preference_fast_strength,
    })
    identity = artifact.coordinate_identity
    if identity is None:
        result["token_preference_coordinate_identity"] = None
    else:
        result.update({
            "token_preference_projection_seed": identity.projection_seed,
            "token_preference_feature_scheme": identity.feature_scheme,
            "token_preference_whitening_ridge": identity.whitening_ridge,
            "token_preference_coordinate_identity": identity.to_dict(),
        })
    _write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), output)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backend = None
    try:
        if args.kind != "token-preference":
            raise EditorError("unsupported vector kind")
        if args.action == "extract":
            if args.workspace is not None and args.preset is not None:
                raise EditorError("choose either --workspace or --preset")
            artifact = _extract(args)
            _write_text(artifact.to_json(), args.output)
            return 0
        if args.action == "inspect":
            artifact = TokenPreferenceVectorArtifact.from_path(args.artifact)
            report = _inspect_report(artifact)
            _render(report, args.output, args.format, _text_inspect)
            return 0
        if args.action == "validate":
            artifact = TokenPreferenceVectorArtifact.from_path(args.artifact)
            if args.model is not None:
                backend = _load_backend(args)
                artifact.validate_against_backend(
                    backend,
                    backend.provenance(include_model_sha256=False),
                    projection_chunk_size=args.projection_chunk_size,
                )
            report = _inspect_report(artifact)
            report["valid"] = True
            report["validated_against_model"] = args.model is not None
            _render(report, args.output, args.format, _text_inspect)
            return 0
        if args.action == "explain":
            artifact = TokenPreferenceVectorArtifact.from_path(args.artifact)
            backend = _load_backend(args)
            report = _explain_report(
                artifact,
                backend,
                backend.provenance(include_model_sha256=False),
                top=args.top,
                include_eog=args.include_eog,
                projection_chunk_size=args.projection_chunk_size,
            )
            _render(report, args.output, args.format, _text_explain)
            return 0
        if args.action == "blend":
            if args.weights is None:
                weights = [1.0] * len(args.artifacts)
            else:
                weights = args.weights
            artifacts = [TokenPreferenceVectorArtifact.from_path(path) for path in args.artifacts]
            result = blend_artifacts(
                artifacts,
                weights,
                source={
                    "type": "blend",
                    "inputs": [str(path) for path in args.artifacts],
                    "weights": weights,
                },
            )
            _write_text(result.to_json(), args.output)
            return 0
        if args.action == "apply":
            artifact = TokenPreferenceVectorArtifact.from_path(args.artifact)
            _apply(artifact, args.biases, args.output)
            return 0
        raise EditorError(f"unsupported token preference action {args.action!r}")
    except (EditorError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"policy-editor-vector: {exc}", file=sys.stderr)
        return 2
    finally:
        if backend is not None:
            _close_backend(backend)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

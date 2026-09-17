"""Offline workbench for portable steering and token-preference vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .backend_factory import BACKEND_NAMES, create_backend
from .bias_presets import FORMAT as BIAS_FORMAT
from .activation_vectors import (
    FORMAT as STEERING_FORMAT,
    HIDDEN_STATE_KIND,
    OUTPUT_HEAD_KIND,
    SteeringVectorArtifact,
    assert_compatible as assert_steering_compatible,
    blend_artifacts as blend_steering_artifacts,
    model_identity,
)
from .decoder import LlamaCppSettings
from .domain import EditorError
from .episode_store import EpisodeStore
from .token_preference_features import DEFAULT_PROJECTION_CHUNK_SIZE
from .trajectory_compare import (
    DEFAULT_ACTIVATION_STRENGTHS,
    compare_episodes,
    render_compare_report,
)
from .vector_impact import (
    DEFAULT_IMPACT_STRENGTHS,
    impact_vector,
    load_vector_artifact,
    render_impact_report,
)
from .transformers_backend import TransformersSettings
from .vector_artifacts import (
    FORMAT,
    TokenPreferenceVectorArtifact,
    assert_compatible,
    blend_artifacts,
    materialize_features,
)


ACTIVATION_PAIRS_FORMAT = "spe-activation-pairs-v1"


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
        description="Inspect and manage offline steering and token-preference vector artifacts.",
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

    compare = commands.add_parser(
        "compare",
        help="compare recorded episodes as reference and counterfactual trajectories",
    )
    compare.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="episode workspace SQLite database",
    )
    compare.add_argument(
        "--episodes",
        nargs="+",
        required=True,
        metavar="EPISODE_ID",
        help="reference episode first, followed by one or more candidates",
    )
    compare.add_argument(
        "--content",
        action="store_true",
        help="compute mean fixed token-feature contrasts (requires --model)",
    )
    compare.add_argument(
        "--hidden-state",
        dest="hidden_state",
        action="store_true",
        help="compute final hidden-state contrasts (requires --model)",
    )
    compare.add_argument(
        "--feature-dimension",
        type=_positive_int,
        default=64,
        help="dimension for model-backed content contrasts",
    )
    compare.add_argument(
        "--capture-position",
        choices=("first", "last"),
        default="last",
        help="hidden-state position used for model-backed contrasts",
    )
    compare.add_argument(
        "--no-normalize-hidden-state",
        dest="no_normalize_hidden_state",
        action="store_true",
        help="retain the raw hidden-state delta instead of unit-normalizing it",
    )
    compare.add_argument(
        "--hidden-state-strengths",
        dest="hidden_state_strengths",
        type=float,
        nargs="+",
        default=list(DEFAULT_ACTIVATION_STRENGTHS),
        metavar="MULTIPLIER",
        help="signed hidden-state multipliers to report in the strength sweep",
    )
    compare.add_argument(
        "--include-vectors",
        action="store_true",
        help="include full numeric vectors in the JSON report",
    )
    _add_backend_args(compare)
    _add_report_args(compare)

    impact = commands.add_parser(
        "impact",
        help="measure one vector's matched logit effect across episode contexts",
    )
    impact.add_argument(
        "--vector",
        type=Path,
        required=True,
        help="steering or token-preference vector artifact",
    )
    impact.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="episode workspace SQLite database",
    )
    impact.add_argument(
        "--episodes",
        nargs="+",
        required=True,
        metavar="EPISODE_ID",
        help="saved episodes to use as teacher-forced evaluation contexts",
    )
    impact.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=list(DEFAULT_IMPACT_STRENGTHS),
        metavar="MULTIPLIER",
        help="signed vector multipliers to sweep (default: -1 -.5 0 .5 1)",
    )
    impact.add_argument(
        "--top",
        type=_positive_int,
        default=10,
        help="top/bottom affected tokens per context and strength",
    )
    impact.add_argument(
        "--include-eog",
        action="store_true",
        help="allow EOG tokens in top/bottom token summaries",
    )
    impact.add_argument(
        "--include-vectors",
        action="store_true",
        help="include full-vocabulary mean delta arrays in the JSON report",
    )
    impact.add_argument(
        "--max-positions",
        type=_positive_int,
        help="cap recorded context positions per episode",
    )
    impact.add_argument(
        "--rollout",
        action="store_true",
        help="also compare sampled baseline/vector rollouts (includes autoregressive effects)",
    )
    _add_backend_args(impact, require_model=True)
    _add_report_args(impact)

    apply = actions.add_parser("apply", help="apply a standalone vector to a v4 bias preset")
    apply.add_argument("artifact", type=Path)
    apply.add_argument("--biases", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)

    output_head = commands.add_parser(
        "output-head",
        help="create and manage vectors projected through the model output head",
    )
    output_head_actions = output_head.add_subparsers(dest="action", required=True)

    create = output_head_actions.add_parser(
        "create", help="create an output-head steering vector from two prompts"
    )
    _add_backend_args(create, require_model=True)
    prompt_a = create.add_mutually_exclusive_group(required=True)
    prompt_a.add_argument("--prompt-a")
    prompt_a.add_argument("--prompt-a-file", type=Path)
    prompt_b = create.add_mutually_exclusive_group(required=True)
    prompt_b.add_argument("--prompt-b")
    prompt_b.add_argument("--prompt-b-file", type=Path)
    create.add_argument("--capture-position", choices=("first", "last"), default="last")
    create.add_argument("--no-normalize", action="store_true")
    create.add_argument("--strength", type=float, default=1.0)
    create.add_argument("--output", type=Path)

    derive = output_head_actions.add_parser(
        "derive",
        help="derive an output-head steering vector from paired episodes",
    )
    derive.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="episode workspace SQLite database",
    )
    derive.add_argument(
        "--positive",
        nargs="+",
        required=True,
        metavar="EPISODE_ID",
        help="episodes demonstrating the desired behavior",
    )
    derive.add_argument(
        "--negative",
        nargs="+",
        required=True,
        metavar="EPISODE_ID",
        help="baseline or contrasting episodes, paired by position",
    )
    _add_backend_args(derive, require_model=True)
    derive.add_argument(
        "--capture-position",
        choices=("first", "last"),
        default="last",
        help="final hidden-state position captured from each episode text",
    )
    derive.add_argument(
        "--no-normalize",
        action="store_true",
        help="retain the mean hidden-state difference magnitude",
    )
    derive.add_argument("--strength", type=float, default=1.0)
    derive.add_argument("--output", type=Path)

    hidden_state = commands.add_parser(
        "hidden-state",
        help="create, import, and manage layerwise hidden-state control vectors",
    )
    hidden_state_actions = hidden_state.add_subparsers(dest="action", required=True)

    hidden_create = hidden_state_actions.add_parser(
        "create",
        help="create a hidden-state vector from two prompts at selected layers",
    )
    _add_backend_args(hidden_create, require_model=True)
    hidden_prompt_a = hidden_create.add_mutually_exclusive_group(required=True)
    hidden_prompt_a.add_argument("--prompt-a")
    hidden_prompt_a.add_argument("--prompt-a-file", type=Path)
    hidden_prompt_b = hidden_create.add_mutually_exclusive_group(required=True)
    hidden_prompt_b.add_argument("--prompt-b")
    hidden_prompt_b.add_argument("--prompt-b-file", type=Path)
    hidden_layers = hidden_create.add_mutually_exclusive_group(required=True)
    hidden_layers.add_argument("--layer", type=_positive_int)
    hidden_layers.add_argument("--layer-range", nargs=2, type=_positive_int, metavar=("START", "END"))
    hidden_create.add_argument(
        "--capture-position",
        choices=("first", "last"),
        default="last",
        help="token position captured from each prompt",
    )
    hidden_create.add_argument("--no-normalize", action="store_true")
    hidden_create.add_argument("--strength", type=float, default=1.0)
    hidden_create.add_argument("--output", type=Path)

    export_pairs = hidden_state_actions.add_parser(
        "export-pairs",
        help="export paired episodes as cvector-generator prompt files",
    )
    export_pairs.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="episode workspace SQLite database",
    )
    export_pairs.add_argument(
        "--positive",
        nargs="+",
        required=True,
        metavar="EPISODE_ID",
        help="episodes demonstrating the desired behavior",
    )
    export_pairs.add_argument(
        "--negative",
        nargs="+",
        required=True,
        metavar="EPISODE_ID",
        help="baseline or contrasting episodes, paired by position",
    )
    export_pairs.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for positive.txt, negative.txt, and manifest.json",
    )

    import_cvector = hidden_state_actions.add_parser(
        "import-cvector",
        help="import a llama.cpp cvector-generator GGUF as a portable artifact",
    )
    import_cvector.add_argument("cvector", type=Path)
    import_cvector.add_argument("--strength", type=float, default=1.0)
    import_cvector.add_argument("--output", type=Path)
    _add_backend_args(import_cvector)

    def add_vector_management(parent, label: str):
        inspect = parent.add_parser(
            "inspect", help=f"show {label} metadata and norm"
        )
        inspect.add_argument("artifact", type=Path)
        _add_report_args(inspect)

        validate = parent.add_parser(
            "validate", help=f"validate a {label}, optionally against a model"
        )
        validate.add_argument("artifact", type=Path)
        _add_backend_args(validate)
        _add_report_args(validate)

        blend = parent.add_parser(
            "blend", help=f"combine compatible {label}s by effective actuation"
        )
        blend.add_argument("artifacts", type=Path, nargs="+")
        blend.add_argument("--weights", type=float, nargs="+")
        blend.add_argument("--output", type=Path)

    add_vector_management(output_head_actions, "output-head steering vector")
    add_vector_management(hidden_state_actions, "hidden-state vector")

    output_head_explain = output_head_actions.add_parser(
        "explain", help="rank vocabulary tokens by output-head steering effect"
    )
    output_head_explain.add_argument("artifact", type=Path)
    _add_backend_args(output_head_explain, require_model=True)
    output_head_explain.add_argument("--top", type=_positive_int, default=20)
    output_head_explain.add_argument("--include-eog", action="store_true")
    _add_report_args(output_head_explain)
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


def _prompt_value(args: argparse.Namespace, text_name: str, file_name: str, label: str) -> str:
    text = getattr(args, text_name)
    path = getattr(args, file_name)
    if text is not None:
        return text
    try:
        value = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EditorError(f"could not read {label}: {exc}") from exc
    if not value:
        raise EditorError(f"{label} must be nonempty")
    return value


def _episode_prompt_record(store: EpisodeStore, requested_id: str) -> dict[str, Any]:
    episode_id = store.resolve_id(requested_id)
    episode = store.get_episode(episode_id)
    visible_tokens = [
        row for row in store.tokens(episode_id) if bool(row["realized_visible"])
    ]
    actions = store.actions(episode_id)
    text = str(episode["initial_text"]) + str(episode["visible_text"])
    return {
        "requested_id": requested_id,
        "episode_id": episode_id,
        "label": store.label(episode_id),
        "parent_episode_id": episode.get("parent_episode_id"),
        "fork_boundary": episode.get("fork_boundary"),
        "status": str(episode["status"]),
        "visible_token_count": len(visible_tokens),
        "action_kinds": [str(row["kind"]) for row in actions],
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _activation_episode_pairs(
    store: EpisodeStore,
    positive_ids: list[str],
    negative_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(positive_ids) != len(negative_ids):
        raise EditorError(
            "activation positive and negative episode lists must have the same length"
        )
    if not positive_ids:
        raise EditorError("at least one positive/negative episode pair is required")
    positive = [_episode_prompt_record(store, value) for value in positive_ids]
    negative = [_episode_prompt_record(store, value) for value in negative_ids]
    return positive, negative


def _episode_pair_source(
    workspace: Path,
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
) -> dict[str, Any]:
    def metadata(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: record[key]
            for key in (
                "requested_id",
                "episode_id",
                "label",
                "parent_episode_id",
                "fork_boundary",
                "status",
                "visible_token_count",
                "action_kinds",
                "text_sha256",
            )
        }

    return {
        "type": "episode-pairs",
        "workspace": str(workspace),
        "positive": [metadata(record) for record in positive],
        "negative": [metadata(record) for record in negative],
    }


def _cvector_escape(text: str) -> str:
    """Encode one episode as a single cvector-generator prompt-file line."""
    result: list[str] = []
    for character in text:
        codepoint = ord(character)
        if character == "\\":
            result.append("\\\\")
        elif character == "\n":
            result.append("\\n")
        elif character == "\r":
            result.append("\\r")
        elif character == "\t":
            result.append("\\t")
        elif codepoint < 0x20:
            result.append(f"\\x{codepoint:02x}")
        else:
            result.append(character)
    return "".join(result)


def _export_activation_pairs(
    workspace: Path,
    output_dir: Path,
    positive: list[dict[str, Any]],
    negative: list[dict[str, Any]],
) -> dict[str, Any]:
    if output_dir.exists() and not output_dir.is_dir():
        raise EditorError(f"activation pair output is not a directory: {output_dir}")
    manifest = {
        "format": ACTIVATION_PAIRS_FORMAT,
        "workspace": str(workspace),
        "positive_file": "positive.txt",
        "negative_file": "negative.txt",
        "pair_count": len(positive),
        "pairs": [
            {
                "positive": {
                    key: value
                    for key, value in record.items()
                    if key != "text"
                },
                "negative": {
                    key: value
                    for key, value in counterpart.items()
                    if key != "text"
                },
            }
            for record, counterpart in zip(positive, negative)
        ],
    }
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "positive.txt").write_text(
            "\n".join(_cvector_escape(record["text"]) for record in positive) + "\n",
            encoding="utf-8",
        )
        (output_dir / "negative.txt").write_text(
            "\n".join(_cvector_escape(record["text"]) for record in negative) + "\n",
            encoding="utf-8",
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise EditorError(f"could not write activation pair files: {exc}") from exc
    return manifest


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


def _steering_inspect_report(artifact: SteeringVectorArtifact) -> dict[str, Any]:
    return {
        "format": STEERING_FORMAT,
        "kind": artifact.kind,
        "model": dict(artifact.model),
        "dimension": artifact.dimension,
        "norm": artifact.norm,
        "target": artifact.target_description,
        "layer_start": artifact.layer_start,
        "layer_end": artifact.layer_end,
        "strength": artifact.strength,
        "method": artifact.method,
        "digest": artifact.digest,
        "source": dict(artifact.source) if artifact.source is not None else None,
    }


def _text_steering_inspect(report: dict[str, Any]) -> str:
    lines = [
        f"valid: {report['valid']}" if "valid" in report else None,
        f"format: {report['format']}",
        f"kind: {report['kind']}",
        f"dimension: {report['dimension']}",
        f"norm: {report['norm']:.6g} strength={report['strength']:.6g}",
        f"target: {report['target']}"
        + (
            f" range={report['layer_start']}..{report['layer_end']}"
            if report["layer_start"] is not None
            else ""
        ),
        f"method: {report['method']}",
        f"digest: {report['digest']}",
    ]
    lines = [line for line in lines if line is not None]
    if report["model"]:
        lines.append("model: " + " ".join(
            f"{key}={value}" for key, value in sorted(report["model"].items())
        ))
    if report["source"] is not None:
        lines.append("source: " + json.dumps(
            report["source"], ensure_ascii=False, sort_keys=True
        ))
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


def _output_head_explain_report(
    artifact: SteeringVectorArtifact,
    backend: Any,
    provenance: dict[str, Any],
    *,
    top: int,
    include_eog: bool,
) -> dict[str, Any]:
    if artifact.layer == "control-vector":
        raise EditorError(
            "hidden-state vectors do not have a static token-logit explanation; "
            "use hidden-state validate and inspect them with the model runtime"
        )
    adjustments = artifact.validate_against_backend(backend, provenance)
    adjustments = float(artifact.strength) * np.asarray(adjustments, dtype=np.float64)
    eog_ids = set(backend.eog_token_ids())
    token_ids = [
        token_id
        for token_id in range(backend.vocabulary_size())
        if include_eog or token_id not in eog_ids
    ]
    descending = sorted(token_ids, key=lambda token_id: (-float(adjustments[token_id]), token_id))
    ascending = sorted(token_ids, key=lambda token_id: (float(adjustments[token_id]), token_id))

    def rows(order: list[int], *, positive: bool) -> list[dict[str, Any]]:
        result = []
        for token_id in order:
            score = float(adjustments[token_id])
            if (positive and score <= 0.0) or (not positive and score >= 0.0):
                continue
            try:
                text = backend.token_text(token_id)
            except (RuntimeError, TypeError, ValueError):
                text = ""
            result.append({"token_id": token_id, "text": text, "score": score})
            if len(result) >= top:
                break
        return result

    return {
        **_steering_inspect_report(artifact),
        "eligible_token_count": len(token_ids),
        "logit_rms": float(np.sqrt(np.mean(adjustments[token_ids] ** 2))) if token_ids else 0.0,
        "top_positive": rows(descending, positive=True),
        "top_negative": rows(ascending, positive=False),
    }


def _text_output_head_explain(report: dict[str, Any]) -> str:
    lines = [
        f"output-head steering explanation: dimension={report['dimension']} eligible_tokens={report['eligible_token_count']}",
        f"logit_rms={report['logit_rms']:.6g} strength={report['strength']:.6g}",
    ]
    for title, key in (("top positive", "top_positive"), ("top negative", "top_negative")):
        lines.append(f"{title}:")
        if not report[key]:
            lines.append("  (none)")
        for row in report[key]:
            lines.append(
                f"  {row['token_id']:>8} {row['score']:>+12.6g} {row['text']!r}"
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
        if args.kind == "impact":
            artifact = load_vector_artifact(args.vector)
            backend = _load_backend(args)
            with EpisodeStore(args.workspace) as store:
                report = impact_vector(
                    store,
                    args.episodes,
                    backend,
                    artifact,
                    strengths=args.strengths,
                    top=args.top,
                    include_eog=args.include_eog,
                    include_vectors=args.include_vectors,
                    projection_chunk_size=args.projection_chunk_size,
                    max_positions=args.max_positions,
                    rollout=args.rollout,
                )
            if args.format == "json":
                _render(report, args.output, "json", render_impact_report)
            else:
                _write_text(render_impact_report(report), args.output)
            return 0
        if args.kind == "compare":
            if (args.content or args.hidden_state) and args.model is None:
                raise EditorError(
                    "--model is required when --content or --hidden-state is requested"
                )
            if args.model is not None and not (args.content or args.hidden_state):
                raise EditorError(
                    "--model is only used with --content or --hidden-state"
                )
            if args.model is not None:
                backend = _load_backend(args)
            with EpisodeStore(args.workspace) as store:
                report = compare_episodes(
                    store,
                    args.episodes,
                    backend=backend,
                    include_content=args.content,
                    include_hidden_state=args.hidden_state,
                    feature_dimension=args.feature_dimension,
                    projection_chunk_size=args.projection_chunk_size,
                    capture_position=args.capture_position,
                    normalize_activation=not args.no_normalize_hidden_state,
                    activation_strengths=args.hidden_state_strengths,
                    include_vectors=args.include_vectors,
                )
            # Keep the format identifier visible to programmatic consumers even
            # when the human renderer is selected.
            if args.format == "json":
                _render(report, args.output, "json", render_compare_report)
            else:
                _write_text(render_compare_report(report), args.output)
            return 0
        if args.kind in {"output-head", "hidden-state"}:
            if args.kind == "hidden-state" and args.action == "export-pairs":
                with EpisodeStore(args.workspace) as store:
                    positive, negative = _activation_episode_pairs(
                        store, args.positive, args.negative
                    )
                manifest = _export_activation_pairs(
                    args.workspace, args.output_dir, positive, negative
                )
                _write_text(json.dumps(manifest, ensure_ascii=False, indent=2), None)
                return 0
            if args.kind == "output-head" and args.action == "derive":
                with EpisodeStore(args.workspace) as store:
                    positive, negative = _activation_episode_pairs(
                        store, args.positive, args.negative
                    )
                backend = _load_backend(args)
                artifact = SteeringVectorArtifact.from_prompt_pairs(
                    backend,
                    backend.provenance(include_model_sha256=False),
                    [(left["text"], right["text"]) for left, right in zip(positive, negative)],
                    capture_position=args.capture_position,
                    normalize=not args.no_normalize,
                    strength=args.strength,
                    source=_episode_pair_source(args.workspace, positive, negative),
                )
                _write_text(artifact.to_json(), args.output)
                return 0
            if args.kind == "output-head" and args.action == "create":
                backend = _load_backend(args)
                artifact = SteeringVectorArtifact.from_prompt_pair(
                    backend,
                    backend.provenance(include_model_sha256=False),
                    _prompt_value(args, "prompt_a", "prompt_a_file", "prompt A"),
                    _prompt_value(args, "prompt_b", "prompt_b_file", "prompt B"),
                    capture_position=args.capture_position,
                    normalize=not args.no_normalize,
                )
                if args.strength != artifact.strength:
                    artifact = replace(artifact, strength=args.strength)
                _write_text(artifact.to_json(), args.output)
                return 0
            if args.kind == "hidden-state" and args.action == "create":
                backend = _load_backend(args)
                if args.layer is not None:
                    layer_start = layer_end = args.layer
                else:
                    layer_start, layer_end = args.layer_range
                if layer_end < layer_start:
                    raise EditorError("hidden-state layer range end must be at least its start")
                artifact = SteeringVectorArtifact.from_hidden_state_prompt_pair(
                    backend,
                    backend.provenance(include_model_sha256=False),
                    _prompt_value(args, "prompt_a", "prompt_a_file", "prompt A"),
                    _prompt_value(args, "prompt_b", "prompt_b_file", "prompt B"),
                    layer_start=layer_start,
                    layer_end=layer_end,
                    capture_position=args.capture_position,
                    normalize=not args.no_normalize,
                    strength=args.strength,
                )
                _write_text(artifact.to_json(), args.output)
                return 0
            if args.kind == "hidden-state" and args.action == "import-cvector":
                artifact = SteeringVectorArtifact.from_cvector_path(
                    args.cvector, strength=args.strength
                )
                if args.model is not None:
                    backend = _load_backend(args)
                    artifact.validate_against_backend(
                        backend, backend.provenance(include_model_sha256=False)
                    )
                    artifact = replace(
                        artifact,
                        model=model_identity(
                            backend.provenance(include_model_sha256=False),
                            hidden_state_width=backend.activation_width(),
                        ),
                    )
                _write_text(artifact.to_json(), args.output)
                return 0
            if args.action == "inspect":
                artifact = SteeringVectorArtifact.from_path(args.artifact)
                expected_kind = (
                    OUTPUT_HEAD_KIND if args.kind == "output-head" else HIDDEN_STATE_KIND
                )
                if artifact.kind != expected_kind:
                    raise EditorError(
                        f"{args.kind} command cannot inspect {artifact.kind}"
                    )
                report = _steering_inspect_report(artifact)
                _render(report, args.output, args.format, _text_steering_inspect)
                return 0
            if args.action == "validate":
                artifact = SteeringVectorArtifact.from_path(args.artifact)
                expected_kind = (
                    OUTPUT_HEAD_KIND if args.kind == "output-head" else HIDDEN_STATE_KIND
                )
                if artifact.kind != expected_kind:
                    raise EditorError(
                        f"{args.kind} command cannot validate {artifact.kind}"
                    )
                if args.model is not None:
                    backend = _load_backend(args)
                    artifact.validate_against_backend(
                        backend, backend.provenance(include_model_sha256=False)
                    )
                report = _steering_inspect_report(artifact)
                report["valid"] = True
                report["validated_against_model"] = args.model is not None
                _render(report, args.output, args.format, _text_steering_inspect)
                return 0
            if args.kind == "output-head" and args.action == "explain":
                artifact = SteeringVectorArtifact.from_path(args.artifact)
                backend = _load_backend(args)
                report = _output_head_explain_report(
                    artifact,
                    backend,
                    backend.provenance(include_model_sha256=False),
                    top=args.top,
                    include_eog=args.include_eog,
                )
                _render(report, args.output, args.format, _text_output_head_explain)
                return 0
            if args.action == "blend":
                weights = args.weights or [1.0] * len(args.artifacts)
                artifacts = [
                    SteeringVectorArtifact.from_path(path)
                    for path in args.artifacts
                ]
                expected_kind = (
                    OUTPUT_HEAD_KIND if args.kind == "output-head" else HIDDEN_STATE_KIND
                )
                if any(artifact.kind != expected_kind for artifact in artifacts):
                    raise EditorError(
                        f"{args.kind} blend requires {expected_kind} artifacts"
                    )
                result = blend_steering_artifacts(
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
            raise EditorError(f"unsupported {args.kind} action {args.action!r}")

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

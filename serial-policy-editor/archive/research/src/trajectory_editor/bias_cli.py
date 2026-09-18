"""Standalone compiler/assembler for model-specific bias catalogs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .backend_factory import BACKEND_NAMES, create_backend
from .bias_catalog import (
    ALLOCATIONS,
    BiasCatalog,
    ROUTE_POLICIES,
    compile_catalog,
    load_catalog,
    load_yaml_source,
)
from .decoder import LlamaCppSettings
from .domain import EditorError
from .transformers_backend import TransformersSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="policy-editor-bias",
        description="Compile and assemble model-specific human-readable bias catalogs.",
    )
    parser.add_argument("--model", type=Path, help="local model used for tokenization")
    parser.add_argument("--backend", choices=BACKEND_NAMES, default="llama.cpp")
    parser.add_argument("--input", type=Path, help="YAML term/group definition file")
    parser.add_argument(
        "--reference",
        type=Path,
        help="optional YAML list or surface-to-weight mapping for reference allocation and online priors",
    )
    parser.add_argument(
        "--term",
        action="append",
        default=[],
        help="compile one semantic term; may be repeated",
    )
    parser.add_argument("--level", choices=("minimal", "standard", "exhaustive"), default=None)
    parser.add_argument("--explore", action="store_true", help="include alternate decompositions for inspection; runtime routes remain canonical")
    parser.add_argument(
        "--allocation",
        choices=ALLOCATIONS,
        default=None,
        help="edge-bias allocation strategy; legacy preserves current behavior",
    )
    parser.add_argument(
        "--allocation-floor",
        type=float,
        default=None,
        help="minimum per-edge weight for information allocation (default: 0.05)",
    )
    parser.add_argument(
        "--max-routes",
        type=int,
        default=None,
        help="maximum unique routes retained per term; preferred routes are selected first",
    )
    parser.add_argument(
        "--max-route-tokens",
        type=int,
        default=None,
        help="maximum tokens in one route",
    )
    parser.add_argument(
        "--route-policy",
        choices=ROUTE_POLICIES,
        default=None,
        help="alternate route selection policy",
    )
    parser.add_argument(
        "--min-route-piece-chars",
        type=int,
        default=None,
        help="minimum content characters for a non-whole-word cohesive piece",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        action="append",
        default=[],
        help="merge an existing catalog; may be repeated",
    )
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="print experimental allocation diagnostics to stderr",
    )
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-threads", type=int)
    parser.add_argument("--n-gpu-layers", type=int)
    parser.add_argument("--transformers-device", default="auto")
    return parser


def _load_backend(args: argparse.Namespace):
    if args.model is None:
        raise EditorError("--model is required when compiling YAML or inline terms")
    backend = create_backend(
        args.backend,
        args.model,
        llama_settings=LlamaCppSettings(
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_gpu_layers=args.n_gpu_layers,
        ),
        transformers_settings=TransformersSettings(device=args.transformers_device),
    )
    return backend


def _source_from_args(args: argparse.Namespace) -> Any:
    source = load_yaml_source(args.input) if args.input is not None else {"terms": []}
    if not args.term:
        return source
    if isinstance(source, list):
        return [*source, *args.term]
    if not isinstance(source, dict):
        raise EditorError("inline --term values require a YAML mapping or list source")
    result = dict(source)
    existing = result.get("terms", [])
    if isinstance(existing, dict):
        raise EditorError("cannot append --term values to mapping-form terms")
    result["terms"] = [*(existing or []), *args.term]
    return result


def _write_output(catalog: BiasCatalog, output: Path | None) -> None:
    text = catalog.to_json() + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.write_text(text, encoding="utf-8")


def _write_diagnostics(catalog: BiasCatalog) -> None:
    for name, entry in sorted(catalog.entries.items()):
        for route in entry.routes:
            if not route.allocation_diagnostics:
                continue
            print(
                f"{name}: route {list(route.token_ids)} "
                f"(allocation={route.allocation})",
                file=sys.stderr,
            )
            print(
                "token | remaining_mass | information | Phi | edge_weight",
                file=sys.stderr,
            )
            for index, row in enumerate(route.allocation_diagnostics):
                token_text = route.token_texts[index] if index < len(route.token_texts) else ""
                print(
                    f"{route.token_ids[index]} {token_text!r} | "
                    f"{row[0]:.6g} | {row[1]:.6g} | {row[2]:.6g} | {row[3]:.6g}",
                    file=sys.stderr,
                )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.merge:
            if args.input is not None or args.term or args.reference is not None:
                raise EditorError(
                    "--merge cannot be combined with --input, --term, or --reference"
                )
            catalogs = [load_catalog(path) for path in args.merge]
            catalog = BiasCatalog.merge(catalogs)
            if args.diagnostics:
                _write_diagnostics(catalog)
            _write_output(catalog, args.output)
            return 0
        if args.input is None and not args.term:
            raise EditorError("provide --input, --term, or --merge")
        backend = _load_backend(args)
        source = _source_from_args(args)
        reference = load_yaml_source(args.reference) if args.reference is not None else None
        overrides: dict[str, Any] = {}
        if args.explore:
            overrides["route_policy"] = "all"
        if args.max_routes is not None:
            overrides["max_routes"] = args.max_routes
        if args.max_route_tokens is not None:
            overrides["max_route_tokens"] = args.max_route_tokens
        if args.route_policy is not None:
            overrides["route_policy"] = args.route_policy
        if args.min_route_piece_chars is not None:
            overrides["min_route_piece_chars"] = args.min_route_piece_chars
        if args.level is not None:
            overrides["level"] = args.level
        if args.allocation is not None:
            overrides["allocation"] = args.allocation
        if args.allocation_floor is not None:
            overrides["allocation_floor"] = args.allocation_floor
        catalog = compile_catalog(
            source,
            backend,
            options_override=overrides or None,
            reference=reference,
        )
        for warning in catalog.compiler.get("warnings", ()):
            print(f"policy-editor-bias: warning: {warning}", file=sys.stderr)
        if args.diagnostics:
            _write_diagnostics(catalog)
        _write_output(catalog, args.output)
        close = getattr(getattr(backend, "_model", None), "close", None)
        if callable(close):
            close()
        return 0
    except (EditorError, OSError, RuntimeError) as exc:
        print(f"policy-editor-bias: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""CLI for conventional hidden-state activation/steering vectors.

Post-output and other research-derived vector families deliberately do not
appear in this command.  The core runtime can still load externally produced
artifacts; this package only supplies the optional production and inspection
surface for ordinary residual-stream steering vectors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trajectory_editor.activation_vectors import SteeringVectorArtifact
from trajectory_editor.backend_factory import create_backend
from trajectory_editor.core.errors import EditorError
from trajectory_editor.steering_vector_production import create_hidden_state_prompt_pair


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="policy-editor-vector",
        description="Create Transformers-based prompt-pair vectors and inspect artifacts.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser(
        "inspect", help="inspect an externally produced steering artifact"
    )
    inspect.add_argument("artifact", type=Path)

    import_cvector = commands.add_parser(
        "import-cvector",
        help="convert a llama.cpp cvector GGUF into a portable steering artifact",
    )
    import_cvector.add_argument("artifact", type=Path)
    import_cvector.add_argument("--output", type=Path, required=True)
    import_cvector.add_argument("--strength", type=float, default=1.0)

    create = commands.add_parser(
        "create",
        help="create a hidden-state steering vector from a Transformers prompt pair",
    )
    create.add_argument("--model", type=Path, required=True)
    create.add_argument("--prompt-a", required=True)
    create.add_argument("--prompt-b", required=True)
    create.add_argument("--layer-start", type=int, required=True)
    create.add_argument("--layer-end", type=int, required=True)
    create.add_argument(
        "--capture-position", choices=("first", "last"), default="last"
    )
    create.add_argument("--no-normalize", action="store_true")
    create.add_argument("--strength", type=float, default=1.0)
    create.add_argument("--output", type=Path, required=True)
    return parser


def _inspect(path: Path) -> int:
    artifact = SteeringVectorArtifact.from_path(path)
    print(f"format: spe-steering-vector-v1")
    print(f"kind: {artifact.kind}")
    print(f"dimension: {artifact.dimension}")
    print(f"norm: {artifact.norm:.6g}")
    print(f"strength: {artifact.strength:g}")
    if artifact.layer_start is not None:
        print(f"layers: {artifact.layer_start}..{artifact.layer_end}")
    if artifact.model:
        print(f"model: {artifact.model}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
        if args.command == "inspect":
            return _inspect(args.artifact)
        if args.command == "import-cvector":
            artifact = SteeringVectorArtifact.from_cvector_path(
                args.artifact, strength=args.strength
            )
            artifact.write(args.output)
            return 0
        if args.command == "create":
            backend = create_backend("transformers", args.model)
            try:
                artifact = create_hidden_state_prompt_pair(
                    backend,
                    backend.provenance(),
                    args.prompt_a,
                    args.prompt_b,
                    layer_start=args.layer_start,
                    layer_end=args.layer_end,
                    capture_position=args.capture_position,
                    normalize=not args.no_normalize,
                    strength=args.strength,
                )
                artifact.write(args.output)
            finally:
                close = getattr(backend, "close", None)
                if callable(close):
                    close()
            return 0
        parser.error(f"unknown command: {args.command}")
    except (EditorError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"policy-editor-vector: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]

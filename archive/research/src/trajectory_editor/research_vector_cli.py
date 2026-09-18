"""Public entry point for experimental vector workbench tooling."""

from __future__ import annotations

import argparse
import sys


RESEARCH_KINDS = ("token-preference", "compare", "impact")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="policy-editor-research-vector",
        description="Inspect and evaluate experimental vector artifacts.",
    )
    parser.add_argument(
        "kind",
        choices=RESEARCH_KINDS,
        help="research artifact or evaluation family",
    )
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        help="the research subcommand and its options",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values in (["--help"], ["-h"]):
        build_parser().print_help()
        return 0
    if values[0] not in RESEARCH_KINDS:
        print(
            "policy-editor-research-vector: output-head and hidden-state "
            "steering vectors belong to policy-editor-vector",
            file=sys.stderr,
        )
        return 2

    from .vector_cli import main as legacy_main

    return legacy_main(values)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["RESEARCH_KINDS", "build_parser", "main"]

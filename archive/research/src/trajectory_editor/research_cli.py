"""Optional episode CLI surface for research extensions.

The default ``policy-editor`` command intentionally exposes only the core
runtime parser.  This entry point keeps the existing research controls
available without making them part of the core install's help or command
contract.
"""

from __future__ import annotations

from .episode_cli import main as _episode_main


def main(argv: list[str] | None = None) -> int:
    return _episode_main(
        argv,
        include_research=True,
        include_vector=True,
        prog="policy-editor-research",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]

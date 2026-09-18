"""Presentation-only theme selection for the live editor surface."""

from __future__ import annotations

import os
from collections.abc import Mapping


LIVE_THEME_NAMES = ("amber-cyan", "monochrome", "high-contrast")
DEFAULT_LIVE_THEME = "amber-cyan"


def resolve_live_theme(
    requested: str | None,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve a live theme without adding it to editor authority or records."""
    if requested is not None:
        if requested not in LIVE_THEME_NAMES:
            choices = ", ".join(LIVE_THEME_NAMES)
            raise ValueError(f"unknown live UI theme {requested!r}; choose {choices}")
        return requested
    active_environment = os.environ if environment is None else environment
    if "NO_COLOR" in active_environment:
        return "monochrome"
    return DEFAULT_LIVE_THEME

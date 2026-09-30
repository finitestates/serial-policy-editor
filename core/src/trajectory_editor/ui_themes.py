"""Small terminal palettes shared by the curses and plain presentations."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

LIVE_THEME_NAMES = ("amber-cyan", "monochrome", "high-contrast")
DEFAULT_LIVE_THEME = "amber-cyan"


@dataclass(frozen=True)
class ThemePalette:
    name: str
    dark: bool
    background: str
    foreground: str
    primary: str
    secondary: str
    accent: str
    error: str
    muted: str
    rich_primary: str
    rich_secondary: str
    rich_accent: str
    rich_error: str
    rich_muted: str


_HEX: dict[str, dict[str, dict[str, str]]] = {
    "amber-cyan": {
        "dark": {
            "background": "#111217", "foreground": "#F4F1DE",
            "primary": "#F2B544", "secondary": "#39D4E5",
            "accent": "#E483E2", "error": "#FF7777", "muted": "#A6A5AC",
        },
        "light": {
            "background": "#FFFDF7", "foreground": "#292635",
            "primary": "#805200", "secondary": "#006A75",
            "accent": "#783A7A", "error": "#A51C24", "muted": "#595664",
        },
    },
    "monochrome": {
        "dark": {
            "background": "#000000", "foreground": "#FFFFFF",
            "primary": "#FFFFFF", "secondary": "#FFFFFF",
            "accent": "#FFFFFF", "error": "#FFFFFF", "muted": "#FFFFFF",
        },
        "light": {
            "background": "#FFFFFF", "foreground": "#000000",
            "primary": "#000000", "secondary": "#000000",
            "accent": "#000000", "error": "#000000", "muted": "#000000",
        },
    },
    "high-contrast": {
        "dark": {
            "background": "#000000", "foreground": "#FFFFFF",
            "primary": "#FFFF00", "secondary": "#00FFFF",
            "accent": "#FF00FF", "error": "#FF8080", "muted": "#FFFFFF",
        },
        "light": {
            "background": "#FFFFFF", "foreground": "#000000",
            "primary": "#593A00", "secondary": "#00505E",
            "accent": "#720064", "error": "#A00000", "muted": "#292929",
        },
    },
}
_ANSI = {
    "primary": "ansi_yellow", "secondary": "ansi_cyan",
    "accent": "ansi_magenta", "error": "ansi_bright_red",
    "muted": "ansi_bright_black", "background": "ansi_default",
    "foreground": "ansi_default",
}
_RICH_ANSI = {
    "primary": "yellow", "secondary": "cyan", "accent": "magenta",
    "error": "bright_red", "muted": "bright_black", "background": "default",
    "foreground": "default",
}


def _active_environment(environment: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environment is None else environment


def is_dark_terminal(environment: Mapping[str, str] | None = None) -> bool:
    """Read COLORFGBG's background index; unknown values default to dark."""
    colorfgbg = _active_environment(environment).get("COLORFGBG", "")
    try:
        return int(colorfgbg.rsplit(";", 1)[-1]) < 8
    except (TypeError, ValueError):
        return True


def resolve_live_theme(
    requested: str | None,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve a named terminal palette without changing editor authority."""
    if requested is not None:
        if requested not in LIVE_THEME_NAMES:
            choices = ", ".join(LIVE_THEME_NAMES)
            raise ValueError(f"unknown live UI theme {requested!r}; choose {choices}")
        return requested
    if "NO_COLOR" in _active_environment(environment):
        return "monochrome"
    return DEFAULT_LIVE_THEME


def theme_palette(
    name: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> ThemePalette:
    """Resolve semantic Rich colors for text returned by shared render helpers."""
    if name not in LIVE_THEME_NAMES:
        choices = ", ".join(LIVE_THEME_NAMES)
        raise ValueError(f"unknown live UI theme {name!r}; choose {choices}")
    active_environment = _active_environment(environment)
    dark = is_dark_terminal(active_environment)
    values = _HEX[name]["dark" if dark else "light"]
    use_hex = active_environment.get("COLORTERM", "").lower() == "truecolor"
    colors = dict(values) if use_hex or name == "monochrome" else {
        key: _ANSI.get(key, value) for key, value in values.items()
    }
    rich_colors = dict(values) if use_hex or name == "monochrome" else {
        key: _RICH_ANSI.get(key, value) for key, value in values.items()
    }
    return ThemePalette(
        name=name,
        dark=dark,
        background=colors["background"],
        foreground=colors["foreground"],
        primary=colors["primary"],
        secondary=colors["secondary"],
        accent=colors["accent"],
        error=colors["error"],
        muted=colors["muted"],
        rich_primary=rich_colors["primary"],
        rich_secondary=rich_colors["secondary"],
        rich_accent=rich_colors["accent"],
        rich_error=rich_colors["error"],
        rich_muted=rich_colors["muted"],
    )


def semantic_style(
    semantic: str,
    name: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Map a semantic fragment name to a Rich style string."""
    palette = theme_palette(name, environment=environment)
    roles = {
        "status-strong": ("rich_primary", "bold"),
        "muted": ("rich_muted", ""),
        "rule": ("rich_muted", ""),
        "section": ("rich_secondary", "bold"),
        "proposal": ("rich_primary", "bold reverse"),
        "proposal-label": ("rich_primary", "bold"),
        "effect": ("rich_secondary", "bold"),
        "invalid": ("rich_error", "bold underline"),
        "pending": ("rich_secondary", "underline"),
        "table-header": ("rich_muted", "underline"),
        "table-row": ("rich_foreground", ""),
        "selected-row": ("rich_secondary", "bold reverse"),
        "match-row": ("rich_accent", "bold"),
        "help-key": ("rich_primary", "bold"),
        "prompt-label": ("rich_secondary", "bold"),
        "prompt": ("rich_secondary", "bold"),
        "input": ("rich_foreground", ""),
        "hint": ("rich_muted", ""),
        "feedback-error": ("rich_error", "bold"),
        "feedback-info": ("rich_secondary", "bold"),
        "feedback-search": ("rich_accent", "bold"),
        "feedback-detail": ("rich_foreground", ""),
        "beam-selected": ("rich_secondary", "bold reverse"),
        "beam-score": ("rich_muted", ""),
        "beam-continuation": ("rich_foreground", ""),
        "beam-family": ("rich_primary", ""),
        "beam-protected": ("rich_primary", "bold underline"),
        "beam-pane": ("rich_foreground", ""),
    }
    if name == "monochrome":
        return {
            "status-strong": "bold",
            "section": "bold underline",
            "proposal": "bold reverse",
            "proposal-label": "bold underline",
            "effect": "bold",
            "invalid": "bold underline",
            "pending": "underline",
            "table-header": "underline",
            "selected-row": "bold reverse",
            "match-row": "bold underline",
            "help-key": "bold",
            "prompt-label": "bold",
            "prompt": "bold reverse",
            "feedback-error": "bold underline",
            "feedback-info": "bold",
            "feedback-search": "bold underline",
            "beam-selected": "bold reverse",
            "beam-family": "bold",
            "beam-protected": "bold underline",
        }.get(semantic, "")
    role, attributes = roles.get(semantic, ("rich_foreground", ""))
    if role == "rich_foreground":
        return attributes
    color = getattr(palette, role, palette.rich_muted)
    if color in {"default", "ansi_default"}:
        return attributes
    return f"{color} {attributes}".strip()

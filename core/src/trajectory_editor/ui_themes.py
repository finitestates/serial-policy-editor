"""Textual theme names, colors, CSS, and semantic Rich styles."""

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

_ANSI: dict[str, str] = {
    "primary": "ansi_yellow",
    "secondary": "ansi_cyan",
    "accent": "ansi_magenta",
    "error": "ansi_bright_red",
    "muted": "ansi_bright_black",
    "background": "ansi_default",
    "foreground": "ansi_default",
}
_RICH_ANSI: dict[str, str] = {
    "primary": "yellow",
    "secondary": "cyan",
    "accent": "magenta",
    "error": "bright_red",
    "muted": "bright_black",
    "background": "default",
    "foreground": "default",
}


def _active_environment(environment: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environment is None else environment


def is_dark_terminal(environment: Mapping[str, str] | None = None) -> bool:
    """Read COLORFGBG's background index; unknown values default to dark."""
    colorfgbg = _active_environment(environment).get("COLORFGBG", "")
    try:
        background = int(colorfgbg.rsplit(";", 1)[-1])
    except (TypeError, ValueError):
        return True
    return background < 8


def resolve_live_theme(
    requested: str | None,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve a registered Textual theme without changing editor authority."""
    if requested is not None:
        if requested not in LIVE_THEME_NAMES:
            choices = ", ".join(LIVE_THEME_NAMES)
            raise ValueError(f"unknown live UI theme {requested!r}; choose {choices}")
        return requested
    active_environment = _active_environment(environment)
    if "NO_COLOR" in active_environment:
        return "monochrome"
    return DEFAULT_LIVE_THEME


def theme_palette(
    name: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> ThemePalette:
    """Resolve light/dark colors and terminal color support for a theme."""
    if name not in LIVE_THEME_NAMES:
        choices = ", ".join(LIVE_THEME_NAMES)
        raise ValueError(f"unknown live UI theme {name!r}; choose {choices}")
    active_environment = _active_environment(environment)
    dark = is_dark_terminal(active_environment)
    values = _HEX[name]["dark" if dark else "light"]
    truecolor = active_environment.get("COLORTERM", "").lower() == "truecolor"
    if truecolor or name == "monochrome":
        colors = dict(values)
        rich_colors = dict(values)
    else:
        colors = {
            key: _ANSI.get(key, value) for key, value in values.items()
        }
        rich_colors = {
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


def textual_theme_values(
    palette: ThemePalette,
) -> dict[str, str | bool | dict[str, str]]:
    """Return constructor values for Textual's ``Theme`` registry."""
    return {
        "name": palette.name,
        "dark": palette.dark,
        "primary": palette.primary,
        "secondary": palette.secondary,
        "accent": palette.accent,
        "error": palette.error,
        "foreground": palette.foreground,
        "background": palette.background,
        "surface": palette.background,
        "panel": palette.background,
        "boost": palette.secondary,
        "variables": {
            "scrollbar": palette.muted,
            "scrollbar-hover": palette.secondary,
            "scrollbar-active": palette.secondary,
            "scrollbar-background": palette.background,
            "scrollbar-background-hover": palette.background,
            "scrollbar-background-active": palette.background,
            "scrollbar-corner-color": palette.background,
        },
    }


def semantic_style(
    semantic: str,
    name: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Map a former semantic fragment class to an inline Rich style."""
    palette = theme_palette(name, environment=environment)
    color_roles = {
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
        style = {
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
        return style
    role, attributes = color_roles.get(semantic, ("rich_foreground", ""))
    if role == "rich_foreground":
        return attributes
    color = getattr(palette, role, palette.rich_muted)
    if color in {"default", "ansi_default"}:
        return attributes
    return f"{color} {attributes}".strip()


def theme_stylesheet(
    name: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Return the Textual CSS palette and semantic widget classes."""
    palette = theme_palette(name, environment=environment)
    del palette
    color_rules = "" if name == "monochrome" else """
    .status-strong { color: $primary; text-style: bold; }
    .muted { color: $foreground; }
    .section { color: $secondary; text-style: bold; }
    .proposal { color: $primary; text-style: bold reverse; }
    .selected-row { color: $secondary; text-style: bold reverse; }
    .beam-selected { color: $secondary; text-style: bold reverse; }
    .feedback-error { color: $error; text-style: bold underline; }
    .feedback-info { color: $secondary; text-style: bold; }
    .feedback-search { color: $accent; text-style: bold; }
    """
    monochrome_rules = "" if name != "monochrome" else """
    .status-strong { text-style: bold; }
    .section { text-style: bold underline; }
    .proposal { text-style: bold reverse; }
    .selected-row, .beam-selected { text-style: bold reverse; }
    .feedback-error { text-style: bold underline; }
    .feedback-info { text-style: bold; }
    .feedback-search { text-style: bold underline; }
    """
    return f"""
    Screen {{ background: $background; color: $foreground; }}
    #root {{ height: 1fr; width: 1fr; align-horizontal: center; }}
    #context-scroll {{ height: auto; min-height: 2; max-height: 5; width: 1fr; border-bottom: solid $secondary; }}
    #context, #review-context {{ height: auto; width: 1fr; padding: 0 1; }}
    #choice-heading, #review-header, #edge-header {{ width: 1fr; }}
    #choice-preview, #choice-feedback, #beam-detail, #beam-notice {{ height: auto; width: 1fr; padding: 0 1; }}
    #choice-table {{ width: 1fr; height: 1fr; min-height: 4; }}
    #edge-commands {{ width: 1fr; max-width: 160; height: auto; max-height: 30%; min-height: 3; }}
    #beam-body {{ height: 1fr; width: 1fr; max-width: 200; min-height: 3; layout: horizontal; }}
    BeamScreen.-stacked #beam-body {{ layout: vertical; max-width: 160; }}
    #beam-table {{ width: 2fr; height: 1fr; min-width: 48; min-height: 3; overflow-y: scroll; }}
    #beam-detail-pane {{ width: 1fr; height: 1fr; min-width: 36; border-left: solid $secondary; padding: 0 1; }}
    BeamScreen.-stacked #beam-body #beam-table {{ width: 1fr; min-width: 0; max-height: 50%; }}
    BeamScreen.-stacked #beam-body #beam-detail-pane {{ width: 1fr; min-width: 0; max-height: 25%; border-left: none; border-top: solid $secondary; }}
    BeamScreen #beam-notice {{ height: 1; min-height: 1; max-height: 1; }}
    BeamScreen #hint {{ height: 4; min-height: 4; max-height: 4; }}
    BeamScreen.-side-by-side #hint {{ height: 2; min-height: 2; max-height: 2; }}
    #choice-command-row {{ width: 1fr; height: 1; }}
    #command-row {{ width: 1fr; max-width: 120; height: 1; }}
    .prompt-label {{ width: auto; }}
    #choice-input, #beam-input, #edge-input {{ width: 1fr; min-width: 8; height: 1; min-height: 1; border: none; padding: 0; background: $surface; }}
    #choice-input:focus, #beam-input:focus, #edge-input:focus {{ color: $background; background: $secondary; text-style: bold; }}
    #choice-input.expanded {{ height: 8; min-height: 3; border: solid $secondary; }}
    #prompt-group {{ height: auto; width: 1fr; max-width: 96; margin-top: 2; padding: 0 1; }}
    #prompt-group.page {{ height: 1fr; max-width: 160; margin-top: 0; }}
    PromptScreen.-short #prompt-group {{ margin-top: 0; }}
    PromptScreen.-short #prompt-body {{ max-height: 12%; }}
    PromptScreen.-short #multiline-input {{ height: 5; max-height: 25%; }}
    #prompt-input {{ width: 1fr; min-width: 8; height: 3; min-height: 3; border: solid $secondary; padding: 0 1; }}
    #prompt-input:focus {{ border: heavy $secondary; }}
    #multiline-input {{ height: 8; max-height: 30%; min-height: 3; width: 1fr; }}
    #prompt-body {{ height: auto; max-height: 30%; width: 1fr; max-width: 84; border: solid $secondary; padding: 0 1; }}
    #prompt-instructions, #prompt-label {{ height: auto; width: 1fr; padding: 0 1; }}
    #page-return, #single-key-hint {{ height: 1; width: 1fr; min-height: 1; padding: 0 1; border: none; background: $surface; }}
    #prompt-status {{ height: auto; width: 1fr; padding: 0 1; }}
    #page-scroll {{ height: 1fr; width: 1fr; }}
    #page-body {{ height: auto; width: 1fr; }}
    #page-return:focus, #single-key-hint:focus {{ color: $background; background: $secondary; text-style: bold; }}
    #choice-hint {{ height: 1; min-height: 1; max-height: 1; width: 1fr; padding: 0 1; text-wrap: nowrap; text-overflow: ellipsis; }}
    #hint {{ height: auto; max-height: 30%; width: 1fr; max-width: 160; padding: 0 1; dock: bottom; text-wrap: wrap; text-align: center; }}
    #edge-header {{ height: auto; padding: 0 1; }}
    #edge-commands {{ margin: 0 1; }}
    #beam-context {{ height: auto; min-height: 1; max-height: 20%; width: 1fr; max-width: 160; padding: 0 1; }}
    #output-dialog {{ width: 92vw; height: 85vh; max-width: 100vw; max-height: 100vh; min-width: 0; min-height: 0; border: tall $secondary; background: $background; padding: 1 2; }}
    #output-body {{ width: 1fr; height: 1fr; min-height: 0; border-top: solid $secondary; }}
    CommandPalette CommandList {{ width: 75vw; offset-x: 12.5vw; max-height: 40vh; }}
    ModalScreen {{ align: center middle; }}
    #help-dialog {{ width: 90vw; height: 85vh; max-width: 100vw; max-height: 100vh; min-width: 0; min-height: 0; border: tall $secondary; background: $background; padding: 1 2; }}
    #help-scroll {{ height: 1fr; min-height: 0; }}
    #help-body {{ height: auto; width: 1fr; }}
    #help-dialog #hint {{ width: 1fr; max-width: 100%; min-height: 1; max-height: 30%; dock: bottom; text-wrap: wrap; text-align: center; }}
    #output-dialog #hint {{ width: 1fr; max-width: 100%; min-height: 1; max-height: 30%; dock: bottom; text-wrap: wrap; text-align: center; }}
    {color_rules}
    {monochrome_rules}
    DataTable > .datatable--cursor {{ text-style: bold reverse; }}
    """

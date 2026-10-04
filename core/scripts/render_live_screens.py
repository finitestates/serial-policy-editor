"""Render live terminal views to HTML (and PNG via headless Chrome) for review.

Uses the same headless harness as the tests, so each image is exactly the frame
the UI would present at that size and theme.

    .venv/bin/python core/scripts/render_live_screens.py --output /tmp/spe-screens \
        --theme chill --size 100x30 --view choice beam
"""

from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "core" / "src"), str(ROOT)]

from rich.console import Console
from rich.style import Style
from rich.terminal_theme import TerminalTheme
from rich.text import Text
from trajectory_editor.terminal_contracts import (
    BoundaryReview,
    ChoiceFeedback,
)
from trajectory_editor.ui_themes import LIVE_THEME_NAMES

from tests.core.term_support import (
    Harness,
    beam_state,
    choice_state,
    edge_state,
    prompt_state,
)

CONTEXT = (
    "The lighthouse keeper had not spoken to anyone in eleven days. Each evening he "
    "climbed the spiral stair, trimmed the wick, and watched the fog roll in over the "
    "shoals, counting the seconds between the horn's long calls."
)


def _views():
    feedback = ChoiceFeedback("info", "SEARCH ' fog'", ("3 matches in the top 40", "Tab cycles matches"))
    return {
        "choice": lambda: choice_state(feedback=feedback, choice=_choice_with_context()),
        "review": lambda: choice_state(review=BoundaryReview(
            42, 37, CONTEXT, {"kind": "action-boundary", "action_kind": "select", "side": "before"},
            {"text": " fog", "token_id": 9},
        )),
        "edge": edge_state,
        "beam": lambda: beam_state(row_count=8),
        "prompt": lambda: prompt_state("Episode name › "),
        "multiline": lambda: prompt_state("New prompt", multiline=True),
        "page": lambda: prompt_state("", body=CONTEXT + "\n\n" + "\n".join(f"line {i}" for i in range(60)), page=True),
    }


def _choice_with_context():
    from dataclasses import replace

    state = choice_state()
    return replace(state.choice, context_text_tail=CONTEXT, aligned_step=37)


def _overlay(ui: Harness, name: str) -> None:
    if name == "help":
        ui.press("f1")
    elif name == "palette":
        ui.press("ctrl+k")
        ui.type("bia")
    elif name == "output":
        ui.app.write_output("loaded model in 2.3s\nselector: argmax temperature=1 eligible-k=none min-p=0\n")
        ui.press("ctrl+l")


def canvas_html(ui: Harness, title: str) -> str:
    canvas = ui.canvas
    console = Console(record=True, width=canvas.width, height=canvas.height, color_system="truecolor",
                      file=io.StringIO(), force_terminal=True, legacy_windows=False)
    for y in range(canvas.height):
        row = Text(no_wrap=True, end="\n")
        for x, (character, style) in enumerate(canvas.cells[y]):
            if character == "":
                continue
            if canvas.cursor == (x, y):
                style = style + Style(reverse=True)
            row.append(character, style)
        console.print(row)
    palette = ui.app.styles.palette
    background = palette.background if palette.background.startswith("#") else "#111217"
    foreground = palette.foreground if palette.foreground.startswith("#") else "#f0f0f0"
    hex_to_rgb = lambda value: tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))
    theme = TerminalTheme(hex_to_rgb(background), hex_to_rgb(foreground), [(0, 0, 0)] * 8, [(128, 128, 128)] * 8)
    html = console.export_html(theme=theme, inline_styles=True)
    return html.replace(
        "<body>", f"<body style='margin:0;padding:12px;background:{background}'><title>{title}</title>"
    ).replace(
        "font-family:Menlo", "font-family:'JetBrains Mono','DejaVu Sans Mono',Menlo"
    ).replace("<pre ", "<pre style='margin:0;line-height:1.2;font-size:14px' ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--theme", nargs="*", default=["chill"], choices=LIVE_THEME_NAMES)
    parser.add_argument("--size", nargs="*", default=["100x30"])
    parser.add_argument("--view", nargs="*", default=list(_views()))
    parser.add_argument("--overlay", nargs="*", default=[], choices=["help", "palette", "output"])
    parser.add_argument("--png", action="store_true", help="screenshot each page with headless Chrome")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    views = _views()
    for theme in args.theme:
        for size_text in args.size:
            width, height = (int(part) for part in size_text.split("x"))
            for view in args.view:
                for overlay in [None, *args.overlay]:
                    with Harness(views[view](), size=(width, height), theme=theme,
                                 environment={"COLORTERM": "truecolor"}) as ui:
                        if view in {"prompt", "multiline"}:
                            ui.type("a lighthouse at dusk")
                        if overlay:
                            _overlay(ui, overlay)
                        ui.frame()
                        name = f"{theme}-{view}{'-' + overlay if overlay else ''}-{width}x{height}"
                        path = args.output / f"{name}.html"
                        path.write_text(canvas_html(ui, name), encoding="utf-8")
                        print(path)
                        if args.png and chrome:
                            subprocess.run(
                                [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                                 f"--screenshot={path.with_suffix('.png')}",
                                 f"--window-size={width * 9 + 40},{int(height * 17.2) + 40}",
                                 path.as_uri()],
                                check=False, capture_output=True, timeout=60,
                            )


if __name__ == "__main__":
    main()

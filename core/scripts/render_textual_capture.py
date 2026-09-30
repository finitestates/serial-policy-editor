#!/usr/bin/env python3
"""Render selected display frames from a recorded Textual PTY capture.

Example::

    core/.venv/bin/python core/scripts/render_textual_capture.py \
        /tmp/serial-policy-editor-recordings/run/capture.json \
        --sequence 137 141 --png

The capture must be a ``*.capture.json`` file produced by
``record_textual_journey.py`` or a Textual PTY test. Each image uses the
terminal cell grid reconstructed from the captured PTY bytes at that display
boundary, rather than a widget or DOM snapshot.
"""

from __future__ import annotations

import argparse
import codecs
import html
import json
from pathlib import Path
import shutil
import subprocess

import pyte


_DEFAULT_FG = "#e5e7eb"
_DEFAULT_BG = "#101216"
_ANSI_COLORS = {
    "black": "#000000",
    "red": "#cd0000",
    "green": "#00cd00",
    "brown": "#cdcd00",
    "blue": "#0000ee",
    "magenta": "#cd00cd",
    "cyan": "#00cdcd",
    "white": "#e5e5e5",
    "brightblack": "#7f7f7f",
    "brightred": "#ff0000",
    "brightgreen": "#00ff00",
    "brightyellow": "#ffff00",
    "brightblue": "#5c5cff",
    "brightmagenta": "#ff00ff",
    "brightcyan": "#00ffff",
    "brightwhite": "#ffffff",
}

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="recorded *.capture.json file")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--sequence",
        type=int,
        nargs="+",
        help="display sequence number or numbers to render",
    )
    selection.add_argument(
        "--all",
        dest="all_frames",
        action="store_true",
        help="render every display event in the capture",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: a sibling <capture>.images directory)",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="also rasterize each SVG to PNG with ffmpeg",
    )
    return parser


def _terminal_cells(capture: dict[str, object], wanted: set[int]):
    raw_path = Path(str(capture["raw_pty_path"]))
    if not raw_path.is_absolute():
        raw_path = Path(str(capture["_capture_path"])).parent / raw_path
    output = raw_path.read_bytes()
    resizes = capture.get("resize_sequence", ())
    if not resizes:
        raise ValueError("capture has no initial terminal dimensions")
    columns, rows = resizes[0]["size"]
    screen = pyte.Screen(int(columns), int(rows))
    stream = pyte.Stream(screen)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    cursor = 0
    captured = {}
    timeline = [
        (int(event["offset"]), 1, "event", event)
        for event in capture.get("ordered_events", ())
    ]
    timeline.extend(
        (int(resize["offset"]), 2, "resize", resize)
        for resize in resizes[1:]
    )
    timeline.sort(key=lambda item: (item[0], item[1]))

    for offset, _order, kind, item in timeline:
        if offset < cursor:
            raise ValueError("capture events are not ordered by PTY byte offset")
        decoded = decoder.decode(output[cursor:offset], final=False)
        if decoded:
            stream.feed(decoded)
        cursor = offset
        if kind == "resize":
            columns, rows = item["size"]
            screen.resize(lines=int(rows), columns=int(columns))
            continue
        if item.get("kind") != "display":
            continue
        sequence = int(item["sequence"])
        if sequence not in wanted:
            continue
        actual_grid = list(screen.display)
        expected_grid = [str(row) for row in item.get("grid", ())]
        if actual_grid != expected_grid:
            raise ValueError(
                f"ANSI replay disagrees with captured grid at display {sequence}"
            )
        captured[sequence] = tuple(
            tuple(screen.buffer[y][x] for x in range(screen.columns))
            for y in range(screen.lines)
        )
    return captured


def _color(value: str, *, foreground: bool) -> str:
    if value == "default":
        return _DEFAULT_FG if foreground else _DEFAULT_BG
    if value.startswith("#"):
        return value
    if value in _ANSI_COLORS:
        return _ANSI_COLORS[value]
    if value.isdecimal():
        index = int(value)
        palette = (
            "#000000", "#cd0000", "#00cd00", "#cdcd00",
            "#0000ee", "#cd00cd", "#00cdcd", "#e5e5e5",
            "#7f7f7f", "#ff0000", "#00ff00", "#ffff00",
            "#5c5cff", "#ff00ff", "#00ffff", "#ffffff",
        )
        if index < len(palette):
            return palette[index]
        if index < 232:
            levels = (0, 95, 135, 175, 215, 255)
            index -= 16
            red, green, blue = (
                levels[index // 36],
                levels[(index // 6) % 6],
                levels[index % 6],
            )
            return f"#{red:02x}{green:02x}{blue:02x}"
        gray = 8 + (index - 232) * 10
        return f"#{gray:02x}{gray:02x}{gray:02x}"
    return _DEFAULT_FG if foreground else _DEFAULT_BG


def _svg(event: dict[str, object], cells) -> str:
    grid = event.get("grid")
    if not isinstance(grid, list) or not grid:
        raise ValueError("display event has no reconstructed terminal grid")
    columns, rows = event.get("size", (0, 0))
    columns, rows = int(columns), int(rows)
    generation = event.get("generation", "?")
    sequence = event.get("sequence", "?")
    geometry = event.get("geometry") or {}
    selected = geometry.get("selected_label")
    details = (
        f"display {sequence} · generation {generation} · {columns}×{rows}"
        + (f" · selected {selected}" if selected else "")
    )
    cell_width = 10
    line_height = 18
    left = 12
    top = 44
    width = columns * cell_width + left * 2
    height = rows * line_height + top + 12
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#101216"/>',
        (
            f'<text x="{left}" y="24" fill="#ffcc66" '
            'font-family="sans-serif" font-size="16">'
            f"{html.escape(details)}</text>"
        ),
    ]
    for row_index, row in enumerate(cells):
        baseline = top + row_index * line_height
        for column, cell in enumerate(row):
            foreground = _color(cell.fg, foreground=True)
            background = _color(cell.bg, foreground=False)
            if cell.reverse:
                foreground, background = background, foreground
            x = left + column * cell_width
            if background != _DEFAULT_BG:
                parts.append(
                    f'<rect x="{x}" y="{baseline - 15}" width="{cell_width}" '
                    f'height="{line_height}" fill="{background}"/>'
                )
            if not cell.data or cell.data == " ":
                continue
            weight = ' font-weight="bold"' if cell.bold else ""
            style = ' font-style="italic"' if cell.italics else ""
            decoration = (
                ' text-decoration="underline"'
                if cell.underscore or cell.blink else ""
            )
            parts.append(
                f'<text x="{x}" y="{baseline}" fill="{foreground}"'
                f'{weight}{style}{decoration} '
                'font-family="DejaVu Sans Mono, monospace" font-size="14">'
                f"{html.escape(cell.data)}</text>"
            )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _run(args: argparse.Namespace) -> int:
    capture_path = args.capture.expanduser().resolve()
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    capture["_capture_path"] = str(capture_path)
    events = {
        int(event["sequence"]): event
        for event in capture.get("ordered_events", ())
        if event.get("kind") == "display" and event.get("sequence") is not None
    }
    sequences = sorted(events) if args.all_frames else args.sequence
    missing = [sequence for sequence in sequences if sequence not in events]
    if missing:
        raise SystemExit(f"display sequence not found in capture: {missing}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else capture_path.parent / f"{capture_path.name.removesuffix('.capture.json')}.images"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg") if args.png else None
    if args.png and ffmpeg is None:
        raise SystemExit("--png requires ffmpeg; the SVG files can be opened directly")

    styled_cells = _terminal_cells(capture, set(sequences))
    for sequence in sequences:
        event = events[sequence]
        columns, rows = event["size"]
        generation = event.get("generation", "unknown")
        stem = f"display-{sequence}-g{generation}-{columns}x{rows}"
        svg_path = output_dir / f"{stem}.svg"
        svg_path.write_text(_svg(event, styled_cells[sequence]), encoding="utf-8")
        print(svg_path)
        if ffmpeg is not None:
            png_path = output_dir / f"{stem}.png"
            subprocess.run(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(svg_path), "-frames:v", "1", "-update", "1",
                    str(png_path),
                ],
                check=True,
            )
            print(png_path)
    return 0


def main() -> int:
    return _run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

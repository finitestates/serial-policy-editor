"""Terminal modes and frame output for the immediate-mode interface."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from rich.color import ColorSystem

from .canvas import Canvas, Line

# Alternate screen, hidden cursor, no autowrap (so the bottom-right cell never
# scrolls), SGR mouse clicks/wheel, bracketed paste, cleared once on entry.
ENTER = (
    "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[?1000h\x1b[?1006h\x1b[?2004h"
    "\x1b[0m\x1b[H\x1b[2J"
)
EXIT = "\x1b[0m\x1b[?2004l\x1b[?1006l\x1b[?1000l\x1b[?7h\x1b[?25h\x1b[?1049l"
SYNC_START = "\x1b[?2026h"
SYNC_END = "\x1b[?2026l"

_COLOR_SYSTEMS = {
    "truecolor": ColorSystem.TRUECOLOR,
    "256": ColorSystem.EIGHT_BIT,
    "standard": ColorSystem.STANDARD,
}


def detect_color_system(environment: Mapping[str, str]) -> str:
    if environment.get("COLORTERM", "").lower() in {"truecolor", "24bit"}:
        return "truecolor"
    if "256color" in environment.get("TERM", ""):
        return "256"
    return "standard"


def terminal_size(*fds: int) -> tuple[int, int]:
    for fd in fds:
        try:
            size = os.get_terminal_size(fd)
        except (OSError, ValueError):
            continue
        if size.columns > 0 and size.lines > 0:
            return size.columns, size.lines
    return 80, 24


class FrameWriter:
    """Diff each frame against the last one and write it in one transaction.

    Lines are only ever overwritten in place; after the initial clear the
    writer never erases the screen, so even a terminal without synchronized
    output never shows a blank intermediate state.
    """

    def __init__(self, fd: int, color_system: str, *, frame_log: str | None = None) -> None:
        self.fd = fd
        self.color_system = _COLOR_SYSTEMS.get(color_system, ColorSystem.STANDARD)
        self._previous: list[Line] | None = None
        self._previous_size: tuple[int, int] | None = None
        self._previous_cursor: tuple[int, int] | None = None
        self._bytes_written = 0
        self._frame_log_path = frame_log
        self._frame_log: Any = None
        self._sequence = 0
        self._render_cache: dict[Line, str] = {}

    def enter(self) -> None:
        if self._frame_log_path:
            self._frame_log = open(self._frame_log_path, "a", encoding="utf-8")  # noqa: SIM115
        self._write(ENTER)
        self.invalidate()

    def exit(self) -> None:
        self._write(EXIT)
        if self._frame_log is not None:
            self._frame_log.close()
            self._frame_log = None

    def invalidate(self) -> None:
        """Force the next frame to rewrite every line (resize, resume)."""
        self._previous = None
        self._previous_cursor = None

    def present(self, canvas: Canvas) -> bool:
        """Write ``canvas``; return False when nothing on screen changed."""
        size = (canvas.width, canvas.height)
        if size != self._previous_size:
            self._previous = None
        lines = [canvas.line(y) for y in range(canvas.height)]
        previous = self._previous
        changed = [
            y for y, line in enumerate(lines)
            if previous is None or y >= len(previous) or previous[y] != line
        ]
        cursor = canvas.cursor
        if not changed and cursor == self._previous_cursor:
            return False
        parts = [SYNC_START, "\x1b[?25l"]
        for y in changed:
            parts.append(f"\x1b[{y + 1};1H")
            parts.append(self._render_line(lines[y]))
        if cursor is not None:
            x, y = cursor
            parts.append(f"\x1b[{y + 1};{x + 1}H\x1b[?25h")
        parts.append(SYNC_END)
        self._write("".join(parts))
        self._previous = lines
        self._previous_size = size
        self._previous_cursor = cursor
        self._log_frame(canvas)
        return True

    def _render_line(self, line: Line) -> str:
        cached = self._render_cache.get(line)
        if cached is not None:
            return cached
        out = ["\x1b[0m"]
        for text, style in line:
            if style:
                out.append(style.render(text, color_system=self.color_system))
            else:
                out.append(text)
        rendered = "".join(out)
        if len(self._render_cache) > 4096:
            self._render_cache.clear()
        self._render_cache[line] = rendered
        return rendered

    def _write(self, text: str) -> None:
        data = text.encode("utf-8", "replace")
        view = memoryview(data)
        while view:
            written = os.write(self.fd, view)
            view = view[written:]
        self._bytes_written += len(data)

    def _log_frame(self, canvas: Canvas) -> None:
        if self._frame_log is None:
            return
        self._sequence += 1
        record = {
            "sequence": self._sequence,
            "end_offset": self._bytes_written,
            "size": [canvas.width, canvas.height],
            "lines": canvas.text_lines(),
            "cursor": list(canvas.cursor) if canvas.cursor is not None else None,
        }
        self._frame_log.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._frame_log.flush()

"""A styled cell grid and Rich-backed text layout."""

from __future__ import annotations

import io
import unicodedata
from collections import OrderedDict
from collections.abc import Iterable, Sequence

from rich.cells import get_character_cell_size
from rich.console import Console
from rich.segment import Segment
from rich.style import Style
from rich.text import Text

# A rendered line: (text, style) runs whose cell widths sum to the line width.
Line = tuple[tuple[str, Style | None], ...]

# The second cell of a double-width character.
WIDE_TAIL = ""

_NULL_STYLE = Style()


def _safe_char(character: str) -> str:
    codepoint = ord(character)
    if codepoint < 32 or codepoint == 127 or 0x80 <= codepoint < 0xA0:
        return "\N{REPLACEMENT CHARACTER}"
    return character


class Canvas:
    """A width x height grid of (character, style) cells plus a cursor."""

    def __init__(self, width: int, height: int, base_style: Style | None = None) -> None:
        self.width = max(0, width)
        self.height = max(0, height)
        self.base_style = base_style or _NULL_STYLE
        blank = (" ", self.base_style)
        self.cells: list[list[tuple[str, Style]]] = [
            [blank] * self.width for _ in range(self.height)
        ]
        self.cursor: tuple[int, int] | None = None

    # -- primitive writes -------------------------------------------------

    def _style(self, style: Style | None) -> Style:
        if style is None or not style:
            return self.base_style
        return self.base_style + style

    def _clear_cell(self, x: int, y: int) -> None:
        """Blank a cell, repairing any double-width character it splits."""
        row = self.cells[y]
        character, style = row[x]
        if character == WIDE_TAIL and x > 0:
            row[x - 1] = (" ", row[x - 1][1])
        elif x + 1 < self.width and row[x + 1][0] == WIDE_TAIL:
            row[x + 1] = (" ", row[x + 1][1])
        row[x] = (" ", style)

    def put(self, x: int, y: int, text: str, style: Style | None = None,
            *, limit: int | None = None) -> int:
        """Write text at (x, y), clipped to the canvas and an optional end column.

        Returns the column after the last written cell.
        """
        if not 0 <= y < self.height:
            return x
        end = self.width if limit is None else min(self.width, limit)
        resolved = self._style(style)
        row = self.cells[y]
        for character in text:
            character = _safe_char(character)
            size = get_character_cell_size(character)
            if size == 0:
                # Combining marks join the previous visible cell.
                if 0 < x <= end and x - 1 < self.width:
                    target = x - 1
                    if row[target][0] == WIDE_TAIL and target > 0:
                        target -= 1
                    if 0 <= target < self.width:
                        previous, previous_style = row[target]
                        row[target] = (
                            unicodedata.normalize("NFC", previous + character),
                            previous_style,
                        )
                continue
            if x >= end:
                break
            if x < 0:
                x += size
                continue
            if size == 2 and x + 1 >= end:
                # A wide character that would straddle the clip edge.
                self._clear_cell(x, y)
                row[x] = (" ", resolved)
                x += 1
                break
            self._clear_cell(x, y)
            row[x] = (character, resolved)
            if size == 2:
                self._clear_cell(x + 1, y)
                row[x + 1] = (WIDE_TAIL, resolved)
            x += size
        return x

    def fill(self, x: int, y: int, width: int, height: int,
             style: Style | None = None, character: str = " ") -> None:
        for row in range(max(0, y), min(self.height, y + height)):
            self.put(x, row, character * max(0, width), style, limit=x + width)

    def put_line(self, x: int, y: int, line: Line, *, width: int | None = None) -> None:
        """Blit one pre-rendered line, clipped to ``width`` cells."""
        end = self.width if width is None else min(self.width, x + width)
        column = x
        for text, style in line:
            if column >= end:
                break
            column = self.put(column, y, text, style, limit=end)
        if column < end:
            self.put(column, y, " " * (end - column), None, limit=end)

    def stylize(self, x: int, y: int, width: int, height: int, style: Style) -> None:
        """Layer a style over existing cells, e.g. a selected-row highlight."""
        for row in range(max(0, y), min(self.height, y + height)):
            cells = self.cells[row]
            for column in range(max(0, x), min(self.width, x + width)):
                character, current = cells[column]
                cells[column] = (character, current + style)

    # -- inspection ------------------------------------------------------

    def line(self, y: int) -> Line:
        """Return row ``y`` as style runs, the unit the writer diffs."""
        runs: list[tuple[str, Style | None]] = []
        text: list[str] = []
        current: Style | None = None
        for character, style in self.cells[y]:
            if character == WIDE_TAIL:
                continue
            if style != current and text:
                runs.append(("".join(text), current))
                text = []
            current = style
            text.append(character)
        if text:
            runs.append(("".join(text), current))
        return tuple(runs)

    def text_lines(self) -> list[str]:
        """Plain text rows, one character per cell (wide tails omitted)."""
        return [
            "".join(character for character, _style in row if character != WIDE_TAIL)
            for row in self.cells
        ]

    def text(self) -> str:
        return "\n".join(line.rstrip() for line in self.text_lines())


class TextLayout:
    """Wrap Rich text into fixed-width lines, with a bounded cache."""

    def __init__(self, color_system: str | None = "truecolor", cache_size: int = 512) -> None:
        self._console = Console(
            file=io.StringIO(),
            width=80,
            color_system=color_system,
            force_terminal=True,
            legacy_windows=False,
            emoji=False,
            markup=False,
            highlight=False,
            no_color=False,
        )
        self._cache: OrderedDict[tuple, list[Line]] = OrderedDict()
        self._cache_size = cache_size

    def lines(self, text: Text | str, width: int, *, wrap: bool = True,
              cache_key: object = None) -> list[Line]:
        """Return ``text`` laid out at ``width`` cells.

        Wrapped text soft-wraps on words and folds long words. Unwrapped text
        keeps one line per hard newline and ends overflowing lines in an
        ellipsis.
        """
        width = max(1, width)
        if isinstance(text, str):
            text = Text(text)
        key = None
        if cache_key is not None:
            key = (cache_key, width, wrap)
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        rendered = self._render(text, width, wrap)
        if key is not None:
            self._cache[key] = rendered
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return rendered

    def _render(self, text: Text, width: int, wrap: bool) -> list[Line]:
        if not text.plain:
            return []
        text = text.copy()
        text.end = ""
        if wrap:
            text.no_wrap = False
            text.overflow = "fold"
        else:
            text.no_wrap = True
            text.overflow = "ellipsis"
        options = self._console.options.update(width=width, height=None)
        segment_lines = self._console.render_lines(text, options, pad=False, new_lines=False)
        result = [_to_line(segments) for segments in segment_lines]
        # Rich renders a trailing newline as an empty final line; keep the
        # visual content only.
        if result and not result[-1] and text.plain.endswith("\n"):
            result.pop()
        return result


def _to_line(segments: Iterable[Segment]) -> Line:
    runs: list[tuple[str, Style | None]] = []
    for segment in segments:
        if segment.control or not segment.text:
            continue
        style = segment.style if segment.style else None
        if runs and runs[-1][1] == style:
            runs[-1] = (runs[-1][0] + segment.text, style)
        else:
            runs.append((segment.text, style))
    return tuple(runs)


def line_width(line: Line) -> int:
    from rich.cells import cell_len

    return sum(cell_len(text) for text, _style in line)


def plain(line: Line) -> str:
    return "".join(text for text, _style in line)


def styled_line(text: str, style: Style | str | None = None) -> Line:
    if isinstance(style, str):
        style = Style.parse(style) if style else None
    return ((text, style),) if text else ()


def concat(*parts: Sequence[tuple[str, Style | None]]) -> Line:
    return tuple(run for part in parts for run in part)

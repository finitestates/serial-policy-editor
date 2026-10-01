"""Layout and drawing helpers shared by the request views."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rich.cells import cell_len, get_character_cell_size
from rich.style import Style
from rich.text import Text

from ..ui_themes import semantic_style, supports_palette_colors, theme_palette
from .canvas import Canvas, Line, TextLayout


def _blend(background: str, foreground: str, amount: float) -> str:
    """Mix two #RRGGBB colors; ``amount`` of ``foreground`` over ``background``."""
    channels = []
    for index in (1, 3, 5):
        low, high = int(background[index:index + 2], 16), int(foreground[index:index + 2], 16)
        channels.append(round(low + (high - low) * amount))
    return "#" + "".join(f"{channel:02X}" for channel in channels)


class Styles:
    """Semantic style lookup for one theme, plus the frame's base style."""

    def __init__(self, theme: str, environment: Mapping[str, str] | None,
                 color_system: str | None) -> None:
        self.theme = theme
        self.environment = dict(environment or {})
        self.palette = theme_palette(theme, environment=self.environment)
        self._cache: dict[str, Style] = {}
        self.painted = (
            supports_palette_colors(self.environment) and color_system in {"truecolor", "256"}
        )
        if self.painted:
            self.base = Style(color=self.palette.foreground, bgcolor=self.palette.background)
        else:
            self.base = Style()

    def __call__(self, semantic: str) -> Style:
        style = self._cache.get(semantic)
        if style is None:
            if semantic in {"prompt-input-focus", "prompt-input-idle"}:
                # A quiet field: a slightly raised background where the theme
                # paints its own colors, an underline elsewhere. The colored
                # prompt label and the caret mark focus, not a loud bar.
                if self.painted:
                    style = Style(bgcolor=_blend(self.palette.background, self.palette.foreground, 0.09))
                else:
                    style = Style(underline=True)
                if semantic == "prompt-input-idle":
                    muted = semantic_style("muted", self.theme, environment=self.environment)
                    if not muted or self.palette.muted == self.palette.foreground:
                        muted = "dim"
                    style += Style.parse(muted)
            elif semantic == "prompt-input-selected":
                # Proposed text that typing replaces.
                style = self("prompt-input-focus") + Style.parse(
                    semantic_style("prompt-label", self.theme, environment=self.environment) or "bold"
                ) + Style(underline=True)
            else:
                spec = semantic_style(semantic, self.theme, environment=self.environment)
                style = Style.parse(spec) if spec else Style()
            self._cache[semantic] = style
        return style

    @property
    def cursor_row(self) -> Style:
        return self("selected-row")


@dataclass
class Hit:
    """A clickable rectangle recorded during rendering."""

    x: int
    y: int
    width: int
    height: int
    action: Callable[[], None]

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


@dataclass
class RenderContext:
    canvas: Canvas
    layout: TextLayout
    styles: Styles
    hits: list[Hit] = field(default_factory=list)
    scroll_regions: list[tuple[Hit, Callable[[int], None]]] = field(default_factory=list)

    def click(self, x: int, y: int, width: int, height: int, action: Callable[[], None]) -> None:
        self.hits.append(Hit(x, y, width, height, action))

    def wheel(self, x: int, y: int, width: int, height: int, scroll: Callable[[int], None]) -> None:
        self.scroll_regions.append((Hit(x, y, width, height, lambda: None), scroll))


# -- text helpers ------------------------------------------------------------


def runs(layout: TextLayout, text: Text | str, style: Style | None = None) -> Line:
    """Single-line style runs for ``text`` (newlines become spaces)."""
    if isinstance(text, str):
        text = Text(text, style=style or "")
    elif style is not None:
        text = text.copy()
        text.stylize(style, 0, len(text))
    text = text.copy()
    text.plain = text.plain.replace("\n", " ")
    console = layout._console
    # Text.render() applies spans only; the base style is applied by callers.
    base = console.get_style(text.style) if text.style else None
    result: list[tuple[str, Style | None]] = []
    for segment in text.render(console, end=""):
        if not segment.text:
            continue
        seg_style = segment.style or None
        if base is not None:
            seg_style = base + seg_style if seg_style else base
        if result and result[-1][1] == seg_style:
            result[-1] = (result[-1][0] + segment.text, seg_style)
        else:
            result.append((segment.text, seg_style))
    return tuple(result)


def clip(line: Line, width: int, *, ellipsis: bool = True) -> Line:
    """Clip a line to ``width`` cells, ending in an ellipsis when it overflows."""
    if width <= 0:
        return ()
    total = sum(cell_len(text) for text, _style in line)
    if total <= width:
        return line
    budget = width - 1 if ellipsis else width
    result: list[tuple[str, Style | None]] = []
    used = 0
    last_style: Style | None = None
    for text, style in line:
        last_style = style
        kept: list[str] = []
        for character in text:
            size = get_character_cell_size(character)
            if used + size > budget:
                break
            kept.append(character)
            used += size
        if kept:
            result.append(("".join(kept), style))
        if used >= budget or len(kept) < len(text):
            break
    if ellipsis:
        result.append(("…", last_style))
        used += 1
    if used < width:
        result.append((" " * (width - used), None))
    return tuple(result)


def pad(line: Line, width: int, style: Style | None = None) -> Line:
    total = sum(cell_len(text) for text, _style in line)
    if total >= width:
        return clip(line, width)
    return (*line, (" " * (width - total), style))


def restyle(line: Line, style: Style) -> Line:
    return tuple((text, (current + style) if current else style) for text, current in line)


# -- vertical allocation -----------------------------------------------------


@dataclass
class Slot:
    """A region competing for rows.

    ``minimum`` rows are reserved first, in ``priority`` order (0 is most
    important); slots that cannot get their minimum are dropped. Remaining rows
    go to non-flex slots up to ``desired``, then to flex slots by ``share`` of
    what is left, capped at ``desired`` when given.
    """

    name: str
    minimum: int
    desired: int | None = None
    priority: int = 0
    flex: bool = False
    share: float = 1.0


def allocate(total: int, slots: Sequence[Slot]) -> dict[str, int]:
    heights = {slot.name: 0 for slot in slots}
    remaining = max(0, total)
    kept: list[Slot] = []
    for slot in sorted(slots, key=lambda item: item.priority):
        if slot.minimum <= remaining:
            heights[slot.name] = slot.minimum
            remaining -= slot.minimum
            kept.append(slot)
    for slot in kept:
        if slot.flex:
            continue
        desired = slot.minimum if slot.desired is None else slot.desired
        extra = min(remaining, max(0, desired - heights[slot.name]))
        heights[slot.name] += extra
        remaining -= extra
    flex = [slot for slot in kept if slot.flex]
    for index, slot in enumerate(flex):
        if remaining <= 0:
            break
        last = index == len(flex) - 1
        amount = remaining if last else int(remaining * slot.share)
        if slot.desired is not None:
            amount = min(amount, max(0, slot.desired - heights[slot.name]))
        heights[slot.name] += amount
        remaining -= amount
    if remaining > 0:
        # Capped flex slots leave space; give it to the first uncapped flex slot.
        for slot in flex:
            if slot.desired is None:
                heights[slot.name] += remaining
                break
    return heights


# -- scrolling text ----------------------------------------------------------


@dataclass
class Scroll:
    """A scroll offset in lines; ``follow`` pins the view to the end."""

    top: int = 0
    follow: bool = False
    # Scrolling back to the end resumes following (context, captured output).
    sticky: bool = False
    # Size of the last drawn window, so key handlers can page by it.
    total: int = 0
    height: int = 1

    def window(self, total: int, height: int) -> int:
        self.total, self.height = total, max(1, height)
        maximum = max(0, total - height)
        if self.follow:
            self.top = maximum
        self.top = max(0, min(self.top, maximum))
        return self.top

    def by(self, amount: int) -> None:
        maximum = max(0, self.total - self.height)
        if self.follow:
            self.top = maximum
        self.top = max(0, min(self.top + amount, maximum))
        if amount < 0 and self.top < maximum:
            self.follow = False
        elif self.top >= maximum and self.sticky:
            self.follow = True

    def page(self, direction: int) -> None:
        self.by(direction * max(1, self.height - 1))

    def reset(self, *, follow: bool | None = None) -> None:
        self.top = 0
        if follow is not None:
            self.follow = follow


def draw_lines(ctx: RenderContext, lines: Sequence[Line], x: int, y: int,
               width: int, height: int, scroll: Scroll | None = None,
               *, scrollbar: bool = True) -> None:
    """Draw a window of pre-wrapped lines, with a thin scroll indicator."""
    if width <= 0 or height <= 0:
        return
    top = scroll.window(len(lines), height) if scroll is not None else 0
    overflow = len(lines) > height
    text_width = width - 1 if overflow and scrollbar and width > 2 else width
    for offset in range(height):
        index = top + offset
        line = lines[index] if index < len(lines) else ()
        ctx.canvas.put_line(x, y + offset, line, width=text_width)
    if overflow and scrollbar and width > 2:
        _scrollbar(ctx, x + width - 1, y, height, top, len(lines))
    if scroll is not None:
        ctx.wheel(x, y, width, height, scroll.by)


def _scrollbar(ctx: RenderContext, x: int, y: int, height: int, top: int, total: int) -> None:
    muted = ctx.styles("muted")
    thumb = max(1, round(height * height / max(1, total)))
    span = max(1, total - height)
    start = round((height - thumb) * top / span)
    for offset in range(height):
        inside = start <= offset < start + thumb
        ctx.canvas.put(x, y + offset, "┃" if inside else "│", muted)


# -- tables ------------------------------------------------------------------


@dataclass
class Column:
    key: str
    header: str
    width: int | None = None  # None: natural width
    flex: bool = False
    wrap: bool = False
    align_right: bool = False


@dataclass
class TableRow:
    key: Any
    cells: Sequence[Text | str]
    style: Style | None = None  # layered over the whole row, e.g. cursor
    cache_key: Any = None


def table_widths(columns: Sequence[Column], rows: Sequence[TableRow], width: int,
                 gap: int = 1) -> list[int]:
    widths: list[int] = []
    for index, column in enumerate(columns):
        if column.width is not None:
            widths.append(column.width)
            continue
        natural = cell_len(column.header)
        if not column.flex:
            for row in rows:
                cell = row.cells[index]
                plain = cell.plain if isinstance(cell, Text) else cell
                natural = max(natural, max((cell_len(part) for part in plain.split("\n")), default=0))
        widths.append(natural)
    flex = [index for index, column in enumerate(columns) if column.flex]
    used = sum(width_ for index, width_ in enumerate(widths) if index not in flex)
    used += gap * max(0, len(columns) - 1)
    if flex:
        each = max(4, (width - used) // len(flex))
        for index in flex:
            widths[index] = each
        widths[flex[-1]] += max(0, width - used - each * len(flex))
    return widths


def render_table_row(ctx: RenderContext, columns: Sequence[Column], widths: Sequence[int],
                     row: TableRow, gap: int = 1) -> list[Line]:
    """Lay out one row as one or more lines of exactly the table width."""
    cell_lines: list[list[Line]] = []
    for index, (column, width) in enumerate(zip(columns, widths)):
        cell = row.cells[index]
        if column.wrap:
            key = None if row.cache_key is None else ("cell", row.cache_key, column.key)
            lines = ctx.layout.lines(cell, width, cache_key=key) or [()]
        else:
            line = runs(ctx.layout, cell)
            if column.align_right:
                size = sum(cell_len(text) for text, _ in line)
                if size < width:
                    line = ((" " * (width - size), None), *line)
            lines = [clip(line, width)]
        cell_lines.append(lines)
    height = max(len(lines) for lines in cell_lines) if cell_lines else 1
    result: list[Line] = []
    for line_index in range(height):
        parts: list[tuple[str, Style | None]] = []
        for column_index, (lines, width) in enumerate(zip(cell_lines, widths)):
            line = lines[line_index] if line_index < len(lines) else ()
            parts.extend(pad(line, width))
            if column_index < len(widths) - 1:
                parts.append((" " * gap, None))
        line = tuple(parts)
        if row.style is not None:
            line = restyle(line, row.style)
        result.append(line)
    return result


@dataclass
class TableView:
    """Row-addressed scrolling that keeps a focus row visible."""

    top: int = 0
    manual: bool = False


def draw_table(ctx: RenderContext, columns: Sequence[Column], rows: Sequence[TableRow],
               x: int, y: int, width: int, height: int, *,
               focus: int | None, view: TableView,
               on_click: Callable[[Any], None] | None = None,
               header: bool = True, max_row_lines: int | None = None) -> None:
    """Draw a header and the rows that fit, scrolled to keep ``focus`` visible."""
    if width <= 0 or height <= 0:
        return
    canvas = ctx.canvas
    body_y, body_height = y, height
    if header and height >= 2:
        widths = table_widths(columns, rows, width)
        header_line: list[tuple[str, Style | None]] = []
        for index, (column, column_width) in enumerate(zip(columns, widths)):
            text = column.header.rjust(column_width) if column.align_right else column.header
            header_line.extend(clip(((text, ctx.styles("table-header")),), column_width, ellipsis=False)
                               if cell_len(text) > column_width else
                               pad(((text, ctx.styles("table-header")),), column_width))
            if index < len(columns) - 1:
                header_line.append((" ", None))
        canvas.put_line(x, y, tuple(header_line), width=width)
        body_y, body_height = y + 1, height - 1
    else:
        widths = table_widths(columns, rows, width)
    rendered = [render_table_row(ctx, columns, widths, row) for row in rows]
    if max_row_lines is not None:
        rendered = [lines[:max_row_lines] for lines in rendered]
    # Keep the focus row fully visible unless the user scrolled by hand.
    if rows:
        view.top = max(0, min(view.top, len(rows) - 1))
        if focus is not None and not view.manual:
            if focus < view.top:
                view.top = focus
            else:
                while view.top < focus and sum(len(lines) for lines in rendered[view.top:focus + 1]) > body_height:
                    view.top += 1
        # Do not leave blank space below the last row when scrolled down.
        while view.top > 0 and sum(len(lines) for lines in rendered[view.top - 1:]) <= body_height:
            view.top -= 1
    else:
        view.top = 0
    row_y = body_y
    index = view.top
    while row_y < body_y + body_height and index < len(rendered):
        lines = rendered[index]
        start = row_y
        for line in lines:
            if row_y >= body_y + body_height:
                break
            canvas.put_line(x, row_y, line, width=width)
            row_y += 1
        if on_click is not None:
            key = rows[index].key
            ctx.click(x, start, width, row_y - start, lambda key=key: on_click(key))
        index += 1
    if row_y < body_y + body_height:
        canvas.fill(x, row_y, width, body_y + body_height - row_y)
    hidden_below = index < len(rendered)
    if (view.top > 0 or hidden_below) and width > 2:
        muted = ctx.styles("muted")
        if view.top > 0:
            canvas.put(x + width - 1, body_y, "▲", muted)
        if hidden_below:
            canvas.put(x + width - 1, body_y + body_height - 1, "▼", muted)

    def scroll(amount: int) -> None:
        view.manual = True
        view.top = max(0, min(len(rows) - 1, view.top + amount))

    ctx.wheel(x, body_y, width, body_height, scroll)


# -- boxes -------------------------------------------------------------------


def draw_box(ctx: RenderContext, x: int, y: int, width: int, height: int,
             title: str = "", style: Style | None = None) -> tuple[int, int, int, int]:
    """Draw a bordered box and return its interior rectangle."""
    canvas = ctx.canvas
    border = style or ctx.styles("section")
    if width < 4 or height < 3:
        canvas.fill(x, y, width, height)
        return x, y, width, height
    canvas.put(x, y, "╭" + "─" * (width - 2) + "╮", border)
    for row in range(y + 1, y + height - 1):
        canvas.put(x, row, "│", border)
        canvas.fill(x + 1, row, width - 2, 1)
        canvas.put(x + width - 1, row, "│", border)
    canvas.put(x, y + height - 1, "╰" + "─" * (width - 2) + "╯", border)
    if title and width > 6:
        label = f" {title} "
        canvas.put(x + 2, y, label, border + Style(bold=True), limit=x + width - 2)
    return x + 1, y + 1, width - 2, height - 2

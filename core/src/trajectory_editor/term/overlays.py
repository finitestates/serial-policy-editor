"""Modal overlays drawn over the active request: help, output, and palette."""

from __future__ import annotations

from collections.abc import Callable

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text

from .editor import TextEditor, draw_editor
from .keys import Key
from .palette import PaletteEntry, search
from .widgets import RenderContext, Scroll, clip, draw_box, draw_lines


def _dialog_rect(width: int, height: int, *, fraction_w: float = 0.9,
                 fraction_h: float = 0.85) -> tuple[int, int, int, int]:
    if width < 50 or height < 14:
        return 0, 0, width, height
    dialog_width = max(40, int(width * fraction_w))
    dialog_height = max(10, int(height * fraction_h))
    return (width - dialog_width) // 2, (height - dialog_height) // 2, dialog_width, dialog_height


class Overlay:
    def on_key(self, key: Key) -> bool:
        """Handle a key; return True to close the overlay."""
        raise NotImplementedError

    def on_paste(self, text: str) -> None:
        del text

    def render(self, ctx: RenderContext) -> None:
        raise NotImplementedError


class DocumentOverlay(Overlay):
    """A scrollable document, e.g. help or the current choice details."""

    def __init__(self, title: str, document: Text, *, close_keys: set[str] | None = None) -> None:
        self.title = title
        self.document = document
        self.scroll = Scroll()
        self.close_keys = close_keys or {"escape", "q", "?", "f1"}

    def on_key(self, key: Key) -> bool:
        if key.name in self.close_keys:
            return True
        if key.name in {"pageup", "pagedown"}:
            self.scroll.page(-1 if key.name == "pageup" else 1)
        elif key.name in {"up", "down"}:
            self.scroll.by(-1 if key.name == "up" else 1)
        elif key.name in {"home", "end"}:
            self.scroll.by(-10**9 if key.name == "home" else 10**9)
        return False

    def hint(self) -> str:
        names = {"escape": "Esc", "ctrl+l": "Ctrl+L", "f1": "F1", "f2": "F2"}
        order = ["escape", "f1", "f2", "ctrl+l", "q", "?"]
        keys = "/".join(names.get(name, name) for name in order if name in self.close_keys)
        return f"PgUp/PgDn ↑↓ scroll · {keys} closes"

    def lines(self, ctx: RenderContext, width: int):
        return ctx.layout.lines(self.document, width, cache_key=("document", id(self), self.document.plain))

    def render(self, ctx: RenderContext) -> None:
        x, y, width, height = _dialog_rect(ctx.canvas.width, ctx.canvas.height)
        ix, iy, iw, ih = draw_box(ctx, x, y, width, height, self.title)
        if (iw, ih) == (width, height):
            # No room for a border: use the whole screen.
            ctx.canvas.fill(x, y, width, height)
        pad = 1 if iw > 20 else 0
        body_height = max(0, ih - 1)
        draw_lines(ctx, self.lines(ctx, max(1, iw - 2 * pad - 1)), ix + pad, iy, iw - 2 * pad, body_height, self.scroll)
        hint = self.hint()
        line = clip(((hint, ctx.styles("hint")),), iw)
        size = min(iw, cell_len(hint))
        ctx.canvas.put_line(ix + max(0, (iw - size) // 2), iy + ih - 1, line, width=min(iw, size))
        ctx.canvas.cursor = None


class OutputOverlay(DocumentOverlay):
    """Captured output; follows the tail while scrolled to the end."""

    def __init__(self, source: Callable[[], str]) -> None:
        super().__init__("Captured output", Text(), close_keys={"escape", "q", "ctrl+l"})
        self.source = source
        self.scroll = Scroll(follow=True, sticky=True)

    def lines(self, ctx: RenderContext, width: int):
        history = self.source()
        if not history:
            return ctx.layout.lines(Text("(no output yet)", style=ctx.styles("muted")), width)
        return ctx.layout.lines(Text(history), width, cache_key=("output", history))


class PaletteOverlay(Overlay):
    """Fuzzy command search; Enter hands the chosen template to ``on_choose``."""

    def __init__(self, entries: tuple[PaletteEntry, ...],
                 on_choose: Callable[[PaletteEntry], None]) -> None:
        self.entries = entries
        self.on_choose = on_choose
        self.query = TextEditor()
        self.index = 0
        self.top = 0

    def _matches(self):
        return search(self.entries, self.query.text)

    def on_key(self, key: Key) -> bool:
        name = key.name
        if name in {"escape", "ctrl+k"}:
            return True
        matches = self._matches()
        if name == "enter":
            if matches:
                entry = matches[min(self.index, len(matches) - 1)][0]
                self.on_choose(entry)
            return True
        if name in {"up", "shift+tab"}:
            self.index = max(0, self.index - 1)
        elif name in {"down", "tab"}:
            self.index = min(max(0, len(matches) - 1), self.index + 1)
        elif name == "pageup":
            self.index = max(0, self.index - 8)
        elif name == "pagedown":
            self.index = min(max(0, len(matches) - 1), self.index + 8)
        else:
            before = self.query.text
            self.query.handle_key(name, key.char)
            if self.query.text != before:
                self.index = 0
                self.top = 0
        return False

    def on_paste(self, text: str) -> None:
        self.query.insert(text.replace("\n", " "))
        self.index = 0

    def render(self, ctx: RenderContext) -> None:
        canvas = ctx.canvas
        styles = ctx.styles
        width = min(canvas.width, max(40, int(canvas.width * 0.75)))
        x = (canvas.width - width) // 2
        matches = self._matches()
        height = min(canvas.height, max(5, min(len(matches) + 4, int(canvas.height * 0.6))))
        y = 0 if canvas.height < 14 else 1
        ix, iy, iw, ih = draw_box(ctx, x, y, width, height, "Commands")
        label = "› "
        label_width = cell_len(label)
        canvas.put(ix, iy, label, styles("prompt-label"))
        input_style = styles("prompt-input-focus")
        draw_editor(canvas, self.query, ix + label_width, iy, max(1, iw - label_width), 1,
                    style=input_style, placeholder="Search commands",
                    placeholder_style=input_style + Style(dim=True))
        rows = max(0, ih - 1)
        self.index = min(self.index, max(0, len(matches) - 1))
        if self.index < self.top:
            self.top = self.index
        elif self.index >= self.top + rows:
            self.top = self.index - rows + 1
        title_width = min(32, max(12, iw // 3))
        for offset in range(rows):
            row_y = iy + 1 + offset
            index = self.top + offset
            canvas.fill(ix, row_y, iw, 1)
            if index >= len(matches):
                if not matches and offset == 0:
                    canvas.put(ix + 1, row_y, "No matching command", styles("muted"))
                continue
            entry, positions = matches[index]
            selected = index == self.index
            title = Text(entry.title)
            for position in positions:
                title.stylize(styles("help-key"), position, position + 1)
            line = []
            line.extend(clip(tuple((text, style) for text, style in _plain_runs(title)), title_width))
            line.append(("  ", None))
            line.extend(clip(((entry.help, styles("muted")),), max(1, iw - title_width - 3)))
            canvas.put_line(ix + 1, row_y, tuple(line), width=iw - 1)
            if selected:
                canvas.stylize(ix, row_y, iw, 1, styles("selected-row"))
            ctx.click(ix, row_y, iw, 1, lambda index=index: self._click(index))

    def _click(self, index: int) -> None:
        self.index = index


def _plain_runs(text: Text):
    plain = text.plain
    styles: list[Style | None] = [None] * len(plain)
    for span in text.spans:
        style = span.style if isinstance(span.style, Style) else Style.parse(str(span.style))
        for position in range(span.start, min(span.end, len(plain))):
            styles[position] = style if styles[position] is None else styles[position] + style
    run_start = 0
    for position in range(1, len(plain) + 1):
        if position == len(plain) or styles[position] != styles[run_start]:
            yield plain[run_start:position], styles[run_start]
            run_start = position

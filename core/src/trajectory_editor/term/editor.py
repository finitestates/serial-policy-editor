"""A small multiline text editor model and its wrapped rendering."""

from __future__ import annotations

from rich.cells import get_character_cell_size
from rich.style import Style

from .canvas import Canvas

_WORD_BREAKS = " \t\n"


class TextEditor:
    """Text plus a cursor offset.

    ``replace_on_type`` marks text the program proposed (e.g. a completion):
    it is shown selected, typing replaces it, and Backspace clears it.
    """

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.cursor = len(text)
        self.replace_on_type = False

    def set(self, text: str, *, replace_on_type: bool = False) -> None:
        self.text = text
        self.cursor = len(text)
        self.replace_on_type = replace_on_type

    def clear(self) -> None:
        self.set("")

    # -- edits -------------------------------------------------------------

    def insert(self, value: str) -> None:
        if self.replace_on_type:
            self.text, self.cursor, self.replace_on_type = "", 0, False
        self.text = self.text[:self.cursor] + value + self.text[self.cursor:]
        self.cursor += len(value)

    def backspace(self) -> None:
        if self.replace_on_type:
            self.set("")
            return
        if self.cursor > 0:
            self.text = self.text[:self.cursor - 1] + self.text[self.cursor:]
            self.cursor -= 1

    def delete(self) -> None:
        if self.replace_on_type:
            self.set("")
            return
        self.text = self.text[:self.cursor] + self.text[self.cursor + 1:]

    def delete_word_left(self) -> None:
        if self.replace_on_type:
            self.set("")
            return
        start = self._word_left(self.cursor)
        self.text = self.text[:start] + self.text[self.cursor:]
        self.cursor = start

    def delete_to_line_start(self) -> None:
        self.replace_on_type = False
        start = self.text.rfind("\n", 0, self.cursor) + 1
        self.text = self.text[:start] + self.text[self.cursor:]
        self.cursor = start

    def delete_to_line_end(self) -> None:
        self.replace_on_type = False
        end = self.text.find("\n", self.cursor)
        end = len(self.text) if end < 0 else end
        self.text = self.text[:self.cursor] + self.text[end:]

    # -- movement ----------------------------------------------------------

    def left(self) -> None:
        self.replace_on_type = False
        self.cursor = max(0, self.cursor - 1)

    def right(self) -> None:
        self.replace_on_type = False
        self.cursor = min(len(self.text), self.cursor + 1)

    def word_left(self) -> None:
        self.replace_on_type = False
        self.cursor = self._word_left(self.cursor)

    def word_right(self) -> None:
        self.replace_on_type = False
        position = self.cursor
        while position < len(self.text) and self.text[position] in _WORD_BREAKS:
            position += 1
        while position < len(self.text) and self.text[position] not in _WORD_BREAKS:
            position += 1
        self.cursor = position

    def home(self) -> None:
        self.replace_on_type = False
        self.cursor = self.text.rfind("\n", 0, self.cursor) + 1

    def end(self) -> None:
        self.replace_on_type = False
        end = self.text.find("\n", self.cursor)
        self.cursor = len(self.text) if end < 0 else end

    def up(self) -> bool:
        """Move to the previous logical line; False when already on the first."""
        self.replace_on_type = False
        start = self.text.rfind("\n", 0, self.cursor) + 1
        if start == 0:
            return False
        column = self.cursor - start
        previous_start = self.text.rfind("\n", 0, start - 1) + 1
        self.cursor = min(previous_start + column, start - 1)
        return True

    def down(self) -> bool:
        self.replace_on_type = False
        end = self.text.find("\n", self.cursor)
        if end < 0:
            return False
        start = self.text.rfind("\n", 0, self.cursor) + 1
        column = self.cursor - start
        next_end = self.text.find("\n", end + 1)
        next_end = len(self.text) if next_end < 0 else next_end
        self.cursor = min(end + 1 + column, next_end)
        return True

    def _word_left(self, position: int) -> int:
        while position > 0 and self.text[position - 1] in _WORD_BREAKS:
            position -= 1
        while position > 0 and self.text[position - 1] not in _WORD_BREAKS:
            position -= 1
        return position

    def handle_key(self, name: str, char: str | None, *, multiline: bool = False) -> bool:
        """Apply a standard editing key. Returns False when the key is not an edit."""
        if char is not None and name == char:
            self.insert(char)
        elif name == "backspace":
            self.backspace()
        elif name == "delete":
            self.delete()
        elif name in {"ctrl+w", "alt+backspace", "ctrl+backspace"}:
            self.delete_word_left()
        elif name == "ctrl+u":
            self.delete_to_line_start()
        elif name == "left":
            self.left()
        elif name == "right":
            self.right()
        elif name in {"ctrl+left", "alt+left", "alt+b"}:
            self.word_left()
        elif name in {"ctrl+right", "alt+right", "alt+f"}:
            self.word_right()
        elif name in {"home", "ctrl+a"}:
            self.home()
        elif name in {"end", "ctrl+e"}:
            self.end()
        elif name == "up" and multiline:
            return self.up()
        elif name == "down" and multiline:
            return self.down()
        else:
            return False
        return True


def _display_char(character: str) -> str:
    if character == "\t":
        return "→"
    if ord(character) < 32 or ord(character) == 127:
        return "\N{REPLACEMENT CHARACTER}"
    return character


def wrap_editor(text: str, cursor: int, width: int) -> tuple[list[list[str]], tuple[int, int]]:
    """Character-wrap ``text`` to ``width`` cells; return rows and cursor (row, col)."""
    width = max(1, width)
    rows: list[list[str]] = [[]]
    column = 0
    cursor_position = (0, 0)
    for index, character in enumerate(text):
        shown = _display_char(character)
        size = get_character_cell_size(shown)
        if index == cursor:
            if column >= width and size > 0:
                rows.append([])
                column = 0
            cursor_position = (len(rows) - 1, min(column, width - 1))
        if character == "\n":
            rows.append([])
            column = 0
            continue
        if column + size > width:
            rows.append([])
            column = 0
        rows[-1].append(shown)
        column += size
    if cursor >= len(text):
        if column >= width:
            rows.append([])
            column = 0
        cursor_position = (len(rows) - 1, column)
    return rows, cursor_position


def editor_height(editor: TextEditor, width: int) -> int:
    rows, _cursor = wrap_editor(editor.text, editor.cursor, width)
    return len(rows)


def draw_editor(
    canvas: Canvas,
    editor: TextEditor,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    style: Style | None = None,
    selected_style: Style | None = None,
    placeholder: str = "",
    placeholder_style: Style | None = None,
    show_cursor: bool = True,
) -> None:
    """Draw the editor's visible rows, scrolled so the cursor stays in view."""
    if width <= 0 or height <= 0:
        return
    rows, (cursor_row, cursor_column) = wrap_editor(editor.text, editor.cursor, width)
    top = max(0, min(cursor_row - height + 1, len(rows) - height))
    text_style = selected_style if editor.replace_on_type and selected_style else style
    canvas.fill(x, y, width, height, style)
    if not editor.text and placeholder:
        canvas.put(x, y, placeholder, placeholder_style, limit=x + width)
    for offset in range(height):
        index = top + offset
        if index >= len(rows):
            break
        canvas.put(x, y + offset, "".join(rows[index]), text_style, limit=x + width)
    if show_cursor:
        canvas.cursor = (x + min(cursor_column, width - 1), y + cursor_row - top)

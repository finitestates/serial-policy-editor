from rich.cells import cell_len

from trajectory_editor.term.canvas import Canvas, WIDE_TAIL
from trajectory_editor.term.editor import TextEditor, draw_editor, wrap_editor


def test_combining_mark_keeps_its_base_cell_and_cursor_column() -> None:
    text = "e\u0301x"

    rows, cursor = wrap_editor(text, cursor=2, width=1)

    assert rows == [["e", "\u0301"], ["x"]]
    assert [cell_len("".join(row)) for row in rows] == [1, 1]
    assert cursor == (1, 0)

    canvas = Canvas(1, 2)
    editor = TextEditor(text)
    editor.cursor = 2
    draw_editor(canvas, editor, 0, 0, 1, 2)
    assert canvas.text_lines() == ["é", "x"]
    assert canvas.cursor == (0, 1)


def test_cursor_before_combining_mark_does_not_split_it_from_its_base() -> None:
    text = "e\u0301"

    rows, cursor = wrap_editor(text, cursor=1, width=1)

    assert rows == [["e", "\u0301"]]
    assert cursor == (0, 0)

    canvas = Canvas(1, 1)
    editor = TextEditor(text)
    editor.cursor = 1
    draw_editor(canvas, editor, 0, 0, 1, 1)
    assert canvas.text_lines() == ["é"]
    assert canvas.cursor == (0, 0)


def test_wide_character_wraps_as_two_cells_and_cursor_uses_next_row() -> None:
    rows, cursor = wrap_editor("A界B", cursor=2, width=3)

    assert rows == [["A", "界"], ["B"]]
    assert [cell_len("".join(row)) for row in rows] == [3, 1]
    assert cursor == (1, 0)

    canvas = Canvas(3, 2)
    editor = TextEditor("A界B")
    editor.cursor = 2
    draw_editor(canvas, editor, 0, 0, 3, 2)
    assert canvas.text_lines() == ["A界", "B  "]
    assert canvas.cells[0][2][0] == WIDE_TAIL
    assert canvas.cursor == (0, 1)


def test_overwriting_or_clipping_wide_character_never_leaves_half_a_glyph() -> None:
    for column in (1, 2):
        canvas = Canvas(5, 1)
        canvas.put(0, 0, "A界BC")

        canvas.put(column, 0, "x")

        assert canvas.cells[0][1][0] != "界"
        assert canvas.cells[0][2][0] != WIDE_TAIL

    canvas = Canvas(4, 1)
    canvas.put(0, 0, "ABCD")
    canvas.put(1, 0, "界", limit=2)

    assert canvas.cells[0][1][0] == " "
    assert canvas.cells[0][2][0] == "C"

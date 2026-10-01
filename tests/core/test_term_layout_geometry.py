from rich.cells import cell_len
from rich.text import Text

from trajectory_editor.term.canvas import Canvas, TextLayout, WIDE_TAIL
from trajectory_editor.term.overlays import DocumentOverlay
from trajectory_editor.term.views import RequestView, pack_hint
from trajectory_editor.term.widgets import RenderContext, Styles


def _context(width: int, height: int) -> RenderContext:
    return RenderContext(
        Canvas(width, height),
        TextLayout(color_system=None),
        Styles("chill", environment=None, color_system=None),
    )


def test_pack_hint_uses_cell_width_for_wide_characters() -> None:
    assert cell_len("界 · x") == 6

    assert pack_hint("界 · x", width=5, height=2) == ["界", "x"]


def test_pack_hint_uses_cell_width_for_combining_sequences() -> None:
    hint = "ab · e\u0301"
    assert cell_len(hint) == 6

    assert pack_hint(hint, width=6, height=1) == [hint]


def test_draw_hint_centers_wide_glyphs_by_terminal_columns() -> None:
    ctx = _context(10, 1)

    RequestView.draw_hint(None, ctx, "界語", y=0, height=1)

    assert ctx.canvas.cells[0][3][0] == "界"
    assert ctx.canvas.cells[0][4][0] == WIDE_TAIL
    assert ctx.canvas.cells[0][5][0] == "語"
    assert ctx.canvas.cells[0][6][0] == WIDE_TAIL


class _WideHintOverlay(DocumentOverlay):
    def hint(self) -> str:
        return "界語"


def test_document_overlay_centers_hint_by_terminal_columns() -> None:
    ctx = _context(60, 20)
    overlay = _WideHintOverlay("Help", Text())

    overlay.render(ctx)

    # The 60x20 viewport gives the dialog a 52-column interior. Its 4-column
    # hint is centered at interior column 24 (screen column 28).
    hint_row = ctx.canvas.cells[16]
    assert hint_row[28][0] == "界"
    assert hint_row[29][0] == WIDE_TAIL
    assert hint_row[30][0] == "語"
    assert hint_row[31][0] == WIDE_TAIL
